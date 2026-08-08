use std::{
    collections::{HashMap, HashSet},
    fs::File,
    path::{Path, PathBuf},
};

use memmap2::{Mmap, MmapOptions};
use safetensors::{Dtype, SafeTensors, tensor::TensorView};
use serde::Deserialize;

use super::{LlamaConfig, ModelError, read_file};

#[derive(Debug, Deserialize)]
struct SafetensorsIndex {
    weight_map: HashMap<String, String>,
}

pub struct WeightStore {
    weight_map: HashMap<String, String>,
    shards: HashMap<String, Mmap>,
}

impl WeightStore {
    pub fn open(model_dir: &Path) -> Result<Self, ModelError> {
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

    pub fn tensor<'a>(&'a self, name: &str) -> Result<TensorView<'a>, ModelError> {
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

    pub fn tensor_count(&self) -> usize {
        self.weight_map.len()
    }

    pub fn validate_llama(&self, config: &LlamaConfig) -> Result<(), ModelError> {
        let mut expected = Vec::with_capacity(config.num_hidden_layers * 9 + 3);
        expected.push((
            "model.embed_tokens.weight".to_string(),
            vec![config.vocab_size, config.hidden_size],
        ));
        expected.push(("model.norm.weight".to_string(), vec![config.hidden_size]));
        if !config.tie_word_embeddings {
            expected.push((
                "lm_head.weight".to_string(),
                vec![config.vocab_size, config.hidden_size],
            ));
        }

        for layer in 0..config.num_hidden_layers {
            let prefix = format!("model.layers.{layer}");
            expected.extend([
                (
                    format!("{prefix}.input_layernorm.weight"),
                    vec![config.hidden_size],
                ),
                (
                    format!("{prefix}.post_attention_layernorm.weight"),
                    vec![config.hidden_size],
                ),
                (
                    format!("{prefix}.self_attn.q_proj.weight"),
                    vec![config.q_width(), config.hidden_size],
                ),
                (
                    format!("{prefix}.self_attn.k_proj.weight"),
                    vec![config.kv_width(), config.hidden_size],
                ),
                (
                    format!("{prefix}.self_attn.v_proj.weight"),
                    vec![config.kv_width(), config.hidden_size],
                ),
                (
                    format!("{prefix}.self_attn.o_proj.weight"),
                    vec![config.hidden_size, config.q_width()],
                ),
                (
                    format!("{prefix}.mlp.gate_proj.weight"),
                    vec![config.intermediate_size, config.hidden_size],
                ),
                (
                    format!("{prefix}.mlp.up_proj.weight"),
                    vec![config.intermediate_size, config.hidden_size],
                ),
                (
                    format!("{prefix}.mlp.down_proj.weight"),
                    vec![config.hidden_size, config.intermediate_size],
                ),
            ]);
        }

        for (name, shape) in expected {
            let tensor = self.tensor(&name)?;
            if tensor.dtype() != Dtype::BF16 {
                return Err(ModelError::WrongDtype {
                    name,
                    actual: tensor.dtype(),
                });
            }
            if tensor.shape() != shape {
                return Err(ModelError::WrongShape {
                    name,
                    expected: shape,
                    actual: tensor.shape().to_vec(),
                });
            }
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
