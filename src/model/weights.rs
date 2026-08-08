use std::{
    collections::{HashMap, HashSet},
    fs::File,
    path::{Path, PathBuf},
};

use memmap2::{Mmap, MmapOptions};
use safetensors::{Dtype, SafeTensors, tensor::TensorView};
use serde::Deserialize;

use super::{ModelError, read_file};

#[derive(Debug, Deserialize)]
struct SafetensorsIndex {
    weight_map: HashMap<String, String>,
}

pub(super) struct WeightStore {
    weight_map: HashMap<String, String>,
    shards: HashMap<String, Mmap>,
}

impl WeightStore {
    pub(super) fn open(model_dir: &Path) -> Result<Self, ModelError> {
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

    pub(super) fn tensor<'a>(&'a self, name: &str) -> Result<TensorView<'a>, ModelError> {
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
        tensors
            .tensor(name)
            .map_err(|_| ModelError::MissingTensor(name.into()))
    }

    pub(super) fn tensor_count(&self) -> usize {
        self.weight_map.len()
    }

    #[cfg(feature = "cuda")]
    pub(super) fn names(&self) -> Vec<String> {
        let mut names: Vec<_> = self.weight_map.keys().cloned().collect();
        names.sort_unstable();
        names
    }

    #[cfg(feature = "cuda")]
    pub(super) fn load_device_bf16(
        &self,
        name: &str,
        stream: &std::sync::Arc<cuda_core::Stream>,
    ) -> Result<cutile::tensor::Tensor<cutile::core::bf16>, ModelError> {
        use cuda_async::device_operation::DeviceOp;
        use cutile::tensor::Reshape;

        let view = self.tensor(name)?;
        if view.dtype() != Dtype::BF16 {
            return Err(ModelError::WrongDtype {
                name: name.into(),
                actual: view.dtype(),
            });
        }
        let shape = view.shape().to_vec();
        let host = std::sync::Arc::new(
            view.data()
                .chunks_exact(2)
                .map(|bytes| {
                    cutile::core::bf16::from_bits(u16::from_le_bytes([bytes[0], bytes[1]]))
                })
                .collect::<Vec<_>>(),
        );
        let tensor = cutile::api::copy_host_vec_to_device(&host)
            .sync_on(stream)
            .map_err(|error| ModelError::Cuda(format!("upload `{name}`: {error:?}")))?;
        tensor
            .reshape(&shape)
            .map_err(|error| ModelError::Cuda(format!("reshape `{name}`: {error:?}")))
    }

    pub(super) fn validate_bf16(&self, name: &str, expected: &[usize]) -> Result<(), ModelError> {
        let tensor = self.tensor(name)?;
        if tensor.dtype() != Dtype::BF16 {
            return Err(ModelError::WrongDtype {
                name: name.into(),
                actual: tensor.dtype(),
            });
        }
        if tensor.shape() != expected {
            return Err(ModelError::WrongShape {
                name: name.into(),
                expected: expected.to_vec(),
                actual: tensor.shape().to_vec(),
            });
        }
        Ok(())
    }
}

fn mmap(path: &Path) -> Result<Mmap, ModelError> {
    let file = File::open(path).map_err(|source| ModelError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    // SAFETY: the returned map owns its file-backed mapping, and WeightStore
    // never mutates or truncates model files while any TensorView can exist.
    unsafe { MmapOptions::new().map(&file) }.map_err(|source| ModelError::Io {
        path: path.to_path_buf(),
        source,
    })
}
