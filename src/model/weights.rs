use std::{
    collections::{HashMap, HashSet},
    fs::File,
    path::{Path, PathBuf},
};

use memmap2::{Mmap, MmapOptions};
use safetensors::{Dtype, SafeTensors};
use serde::Deserialize;

use super::{ModelError, read_file};

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum WeightDtype {
    Bf16,
    Other(String),
}

impl std::fmt::Display for WeightDtype {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Bf16 => formatter.write_str("BF16"),
            Self::Other(name) => formatter.write_str(name),
        }
    }
}

/// Format-neutral, borrowed view of one checkpoint tensor.
///
/// Weight sources own storage; consumers decide how and where to materialize
/// the bytes. Keeping this type independent of SafeTensors and CUDA lets tests,
/// remote stores, and future quantized formats use the same load boundary.
pub(crate) struct WeightTensor<'a> {
    dtype: WeightDtype,
    shape: Vec<usize>,
    bytes: &'a [u8],
}

impl<'a> WeightTensor<'a> {
    pub(crate) fn new(dtype: WeightDtype, shape: Vec<usize>, bytes: &'a [u8]) -> Self {
        Self {
            dtype,
            shape,
            bytes,
        }
    }

    pub(crate) fn dtype(&self) -> &WeightDtype {
        &self.dtype
    }

    pub(crate) fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub(crate) fn byte_len(&self) -> usize {
        self.bytes.len()
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn bytes(&self) -> &'a [u8] {
        self.bytes
    }
}

/// Read-only checkpoint transport. This is a load-time boundary, not a kernel
/// interface: implementations may mmap shards, synthesize test tensors, or
/// fetch tensors remotely without changing the model program or executor.
pub(crate) trait WeightSource: Send + Sync {
    fn tensor(&self, name: &str) -> Result<WeightTensor<'_>, ModelError>;
    fn names(&self) -> Vec<String>;

    fn tensor_count(&self) -> usize {
        self.names().len()
    }

    fn validate_bf16(&self, name: &str, expected: &[usize]) -> Result<(), ModelError> {
        let tensor = self.tensor(name)?;
        if tensor.dtype() != &WeightDtype::Bf16 {
            return Err(ModelError::WrongDtype {
                name: name.into(),
                actual: tensor.dtype().to_string(),
            });
        }
        if tensor.shape() != expected {
            return Err(ModelError::WrongShape {
                name: name.into(),
                expected: expected.to_vec(),
                actual: tensor.shape().to_vec(),
            });
        }
        let expected_bytes = expected
            .iter()
            .try_fold(2usize, |bytes, dimension| bytes.checked_mul(*dimension))
            .ok_or_else(|| ModelError::InvalidTensor {
                name: name.into(),
                message: "BF16 shape byte count overflowed".into(),
            })?;
        if tensor.byte_len() != expected_bytes {
            return Err(ModelError::InvalidTensor {
                name: name.into(),
                message: format!(
                    "has {} data bytes; expected {expected_bytes}",
                    tensor.byte_len()
                ),
            });
        }
        Ok(())
    }
}

#[derive(Debug, Deserialize)]
struct SafetensorsIndex {
    weight_map: HashMap<String, String>,
}

pub(crate) struct SafeTensorSource {
    weight_map: HashMap<String, String>,
    shards: HashMap<String, Mmap>,
}

impl SafeTensorSource {
    pub(crate) fn open(model_dir: &Path) -> Result<Self, ModelError> {
        let index_path = model_dir.join("model.safetensors.index.json");
        let weight_map = if index_path.exists() {
            let text = read_file(&index_path)?;
            serde_json::from_str::<SafetensorsIndex>(&text)
                .map_err(|source| ModelError::Json {
                    path: index_path,
                    source,
                })?
                .weight_map
        } else {
            let filename = "model.safetensors";
            let path = model_dir.join(filename);
            let mmap = mmap(&path)?;
            let tensors =
                SafeTensors::deserialize(&mmap).map_err(|error| ModelError::SafeTensors {
                    path: path.clone(),
                    message: error.to_string(),
                })?;
            let map = tensors
                .names()
                .into_iter()
                .map(|name| (name.to_owned(), filename.to_owned()))
                .collect();
            drop(tensors);
            drop(mmap);
            map
        };

        let shard_names: HashSet<_> = weight_map.values().cloned().collect();
        let mut shards = HashMap::with_capacity(shard_names.len());
        for filename in shard_names {
            let path = model_dir.join(&filename);
            shards.insert(filename, mmap(&path)?);
        }
        Ok(Self { weight_map, shards })
    }
}

impl WeightSource for SafeTensorSource {
    fn tensor(&self, name: &str) -> Result<WeightTensor<'_>, ModelError> {
        let filename = self
            .weight_map
            .get(name)
            .ok_or_else(|| ModelError::MissingTensor(name.into()))?;
        let mmap = self
            .shards
            .get(filename)
            .ok_or_else(|| ModelError::MissingTensor(name.into()))?;
        let tensors = SafeTensors::deserialize(mmap).map_err(|error| ModelError::SafeTensors {
            path: PathBuf::from(filename),
            message: error.to_string(),
        })?;
        let view = tensors
            .tensor(name)
            .map_err(|_| ModelError::MissingTensor(name.into()))?;
        Ok(WeightTensor::new(
            dtype(view.dtype()),
            view.shape().to_vec(),
            view.data(),
        ))
    }

    fn names(&self) -> Vec<String> {
        let mut names: Vec<_> = self.weight_map.keys().cloned().collect();
        names.sort_unstable();
        names
    }

    fn tensor_count(&self) -> usize {
        self.weight_map.len()
    }
}

fn dtype(dtype: Dtype) -> WeightDtype {
    if dtype == Dtype::BF16 {
        WeightDtype::Bf16
    } else {
        WeightDtype::Other(format!("{dtype:?}"))
    }
}

fn mmap(path: &Path) -> Result<Mmap, ModelError> {
    let file = File::open(path).map_err(|source| ModelError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    // SAFETY: the returned map owns its file-backed mapping, and
    // SafeTensorSource never mutates or truncates model files while a borrowed
    // WeightTensor can exist.
    unsafe { MmapOptions::new().map(&file) }.map_err(|source| ModelError::Io {
        path: path.to_path_buf(),
        source,
    })
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;

    use super::*;

    struct MemorySource {
        shape: Vec<usize>,
        bytes: Vec<u8>,
    }

    impl WeightSource for MemorySource {
        fn tensor(&self, name: &str) -> Result<WeightTensor<'_>, ModelError> {
            if name != "weight" {
                return Err(ModelError::MissingTensor(name.into()));
            }
            Ok(WeightTensor::new(
                WeightDtype::Bf16,
                self.shape.clone(),
                &self.bytes,
            ))
        }

        fn names(&self) -> Vec<String> {
            vec!["weight".into()]
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(128))]

        #[test]
        fn bf16_contract_accepts_exact_shape_and_storage(rows in 1usize..64, columns in 1usize..64) {
            let shape = vec![rows, columns];
            let source = MemorySource {
                shape: shape.clone(),
                bytes: vec![0; rows * columns * 2],
            };
            prop_assert!(source.validate_bf16("weight", &shape).is_ok());
            prop_assert_eq!(source.tensor_count(), 1);
        }

        #[test]
        fn bf16_contract_rejects_truncated_storage(rows in 1usize..64, columns in 1usize..64) {
            let shape = vec![rows, columns];
            let source = MemorySource {
                shape: shape.clone(),
                bytes: vec![0; rows * columns * 2 - 1],
            };
            let rejected = matches!(
                source.validate_bf16("weight", &shape),
                Err(ModelError::InvalidTensor { .. })
            );
            prop_assert!(rejected);
        }
    }
}
