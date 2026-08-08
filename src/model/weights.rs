use std::{
    collections::{HashMap, HashSet},
    fs::File,
    path::Path,
};

use memmap2::{Mmap, MmapOptions};
use safetensors::{Dtype, SafeTensors, tensor::Metadata};
use serde::Deserialize;

use super::{ModelError, read_file};

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum WeightDtype {
    Bf16,
    F32,
    F8E4M3,
    U8,
    Other(String),
}

impl std::fmt::Display for WeightDtype {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Bf16 => formatter.write_str("BF16"),
            Self::F32 => formatter.write_str("F32"),
            Self::F8E4M3 => formatter.write_str("F8_E4M3"),
            Self::U8 => formatter.write_str("U8"),
            Self::Other(name) => formatter.write_str(name),
        }
    }
}

impl WeightDtype {
    fn bits(&self) -> Option<usize> {
        match self {
            Self::Bf16 => Some(16),
            Self::F32 => Some(32),
            Self::F8E4M3 | Self::U8 => Some(8),
            Self::Other(_) => None,
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
        self.validate_tensor(name, &WeightDtype::Bf16, expected)
    }

    fn validate_tensor(
        &self,
        name: &str,
        expected_dtype: &WeightDtype,
        expected_shape: &[usize],
    ) -> Result<(), ModelError> {
        let tensor = self.tensor(name)?;
        if tensor.dtype() != expected_dtype {
            return Err(ModelError::WrongDtype {
                name: name.into(),
                expected: expected_dtype.to_string(),
                actual: tensor.dtype().to_string(),
            });
        }
        if tensor.shape() != expected_shape {
            return Err(ModelError::WrongShape {
                name: name.into(),
                expected: expected_shape.to_vec(),
                actual: tensor.shape().to_vec(),
            });
        }
        let element_count = expected_shape
            .iter()
            .try_fold(1usize, |count, dimension| count.checked_mul(*dimension))
            .ok_or_else(|| ModelError::InvalidTensor {
                name: name.into(),
                message: "shape element count overflowed".into(),
            })?;
        let bits = expected_dtype
            .bits()
            .ok_or_else(|| ModelError::InvalidTensor {
                name: name.into(),
                message: format!("cannot validate storage size for dtype {expected_dtype}"),
            })?;
        let bit_count =
            element_count
                .checked_mul(bits)
                .ok_or_else(|| ModelError::InvalidTensor {
                    name: name.into(),
                    message: "shape bit count overflowed".into(),
                })?;
        if !bit_count.is_multiple_of(8) {
            return Err(ModelError::InvalidTensor {
                name: name.into(),
                message: format!("{bit_count} bits do not form a whole-byte tensor"),
            });
        }
        let expected_bytes = bit_count / 8;
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
    shards: HashMap<String, Shard>,
}

/// One mapped shard with its SafeTensors header parsed exactly once.
///
/// Large MoE checkpoints can contain hundreds of thousands of individually
/// named expert tensors. Re-deserializing a multi-megabyte JSON header for
/// every tensor lookup turns otherwise linear model loading into pathological
/// repeated work.
struct Shard {
    mmap: Mmap,
    metadata: Metadata,
    data_offset: usize,
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
            let shard = Shard::open(&path)?;
            shard
                .metadata
                .offset_keys()
                .into_iter()
                .map(|name| (name, filename.to_owned()))
                .collect()
        };

        let shard_names: HashSet<_> = weight_map.values().cloned().collect();
        let mut shards = HashMap::with_capacity(shard_names.len());
        for filename in shard_names {
            let path = model_dir.join(&filename);
            shards.insert(filename, Shard::open(&path)?);
        }

        // Validate the index once at construction. Tensor lookup can then use
        // direct metadata and byte-range access without reparsing a shard.
        for (name, filename) in &weight_map {
            let shard = shards
                .get(filename)
                .ok_or_else(|| ModelError::InvalidTensor {
                    name: name.clone(),
                    message: format!("index references unopened shard `{filename}`"),
                })?;
            if shard.metadata.info(name).is_none() {
                return Err(ModelError::InvalidTensor {
                    name: name.clone(),
                    message: format!("index points to `{filename}`, which does not contain it"),
                });
            }
        }
        Ok(Self { weight_map, shards })
    }
}

impl Shard {
    fn open(path: &Path) -> Result<Self, ModelError> {
        let mmap = mmap(path)?;
        let (header_len, metadata) =
            SafeTensors::read_metadata(&mmap).map_err(|error| ModelError::SafeTensors {
                path: path.to_path_buf(),
                message: error.to_string(),
            })?;
        let data_offset =
            8usize
                .checked_add(header_len)
                .ok_or_else(|| ModelError::SafeTensors {
                    path: path.to_path_buf(),
                    message: "header offset overflowed".into(),
                })?;
        Ok(Self {
            mmap,
            metadata,
            data_offset,
        })
    }

    fn tensor(&self, name: &str) -> Result<WeightTensor<'_>, ModelError> {
        let info = self
            .metadata
            .info(name)
            .ok_or_else(|| ModelError::MissingTensor(name.into()))?;
        let start = self
            .data_offset
            .checked_add(info.data_offsets.0)
            .ok_or_else(|| invalid_range(name, "start offset overflowed"))?;
        let end = self
            .data_offset
            .checked_add(info.data_offsets.1)
            .ok_or_else(|| invalid_range(name, "end offset overflowed"))?;
        let bytes = self
            .mmap
            .get(start..end)
            .ok_or_else(|| invalid_range(name, "data range is outside the shard"))?;
        Ok(WeightTensor::new(
            dtype(info.dtype),
            info.shape.clone(),
            bytes,
        ))
    }
}

fn invalid_range(name: &str, message: &str) -> ModelError {
    ModelError::InvalidTensor {
        name: name.into(),
        message: message.into(),
    }
}

impl WeightSource for SafeTensorSource {
    fn tensor(&self, name: &str) -> Result<WeightTensor<'_>, ModelError> {
        let filename = self
            .weight_map
            .get(name)
            .ok_or_else(|| ModelError::MissingTensor(name.into()))?;
        let shard = self
            .shards
            .get(filename)
            .ok_or_else(|| ModelError::MissingTensor(name.into()))?;
        shard.tensor(name)
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
    match dtype {
        Dtype::BF16 => WeightDtype::Bf16,
        Dtype::F32 => WeightDtype::F32,
        Dtype::F8_E4M3 => WeightDtype::F8E4M3,
        Dtype::U8 => WeightDtype::U8,
        other => WeightDtype::Other(format!("{other:?}")),
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
    use safetensors::{Dtype, tensor::TensorView};
    use tempfile::tempdir;

    use super::*;

    struct MemorySource {
        dtype: WeightDtype,
        shape: Vec<usize>,
        bytes: Vec<u8>,
    }

    impl WeightSource for MemorySource {
        fn tensor(&self, name: &str) -> Result<WeightTensor<'_>, ModelError> {
            if name != "weight" {
                return Err(ModelError::MissingTensor(name.into()));
            }
            Ok(WeightTensor::new(
                self.dtype.clone(),
                self.shape.clone(),
                &self.bytes,
            ))
        }

        fn names(&self) -> Vec<String> {
            vec!["weight".into()]
        }
    }

    fn write_shard(path: &Path, tensors: &[(&str, Dtype, Vec<usize>, &[u8])]) {
        let views: Vec<_> = tensors
            .iter()
            .map(|(name, dtype, shape, bytes)| {
                (
                    *name,
                    TensorView::new(*dtype, shape.clone(), bytes).unwrap(),
                )
            })
            .collect();
        let serialized = safetensors::tensor::serialize(views, None).unwrap();
        std::fs::write(path, serialized).unwrap();
    }

    #[test]
    fn indexed_source_reads_cached_shard_metadata_and_exact_ranges() {
        let directory = tempdir().unwrap();
        let first = [1_u8, 2, 3, 4];
        let second = [5_u8, 6];
        write_shard(
            &directory.path().join("weights.safetensors"),
            &[
                ("first", Dtype::U8, vec![2, 2], &first),
                ("second", Dtype::BF16, vec![1], &second),
            ],
        );
        std::fs::write(
            directory.path().join("model.safetensors.index.json"),
            r#"{"weight_map":{"first":"weights.safetensors","second":"weights.safetensors"}}"#,
        )
        .unwrap();

        let source = SafeTensorSource::open(directory.path()).unwrap();
        let first = source.tensor("first").unwrap();
        assert_eq!(first.dtype(), &WeightDtype::U8);
        assert_eq!(first.shape(), &[2, 2]);
        assert_eq!(first.bytes, &[1, 2, 3, 4]);
        let second = source.tensor("second").unwrap();
        assert_eq!(second.dtype(), &WeightDtype::Bf16);
        assert_eq!(second.bytes, &[5, 6]);
    }

    #[test]
    fn indexed_source_rejects_tensor_mapped_to_wrong_shard() {
        let directory = tempdir().unwrap();
        let bytes = [0_u8, 0];
        write_shard(
            &directory.path().join("weights.safetensors"),
            &[("present", Dtype::BF16, vec![1], &bytes)],
        );
        std::fs::write(
            directory.path().join("model.safetensors.index.json"),
            r#"{"weight_map":{"missing":"weights.safetensors"}}"#,
        )
        .unwrap();

        assert!(matches!(
            SafeTensorSource::open(directory.path()),
            Err(ModelError::InvalidTensor { name, .. }) if name == "missing"
        ));
    }

    #[test]
    fn typed_contract_validates_quantized_weights_scales_and_scalars() {
        let cases = [
            (WeightDtype::U8, vec![512, 1024], 512 * 1024),
            (WeightDtype::F8E4M3, vec![512, 128], 512 * 128),
            (WeightDtype::F32, vec![], 4),
        ];
        for (dtype, shape, byte_len) in cases {
            let source = MemorySource {
                dtype: dtype.clone(),
                shape: shape.clone(),
                bytes: vec![0; byte_len],
            };
            source.validate_tensor("weight", &dtype, &shape).unwrap();
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(128))]

        #[test]
        fn bf16_contract_accepts_exact_shape_and_storage(rows in 1usize..64, columns in 1usize..64) {
            let shape = vec![rows, columns];
            let source = MemorySource {
                dtype: WeightDtype::Bf16,
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
                dtype: WeightDtype::Bf16,
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
