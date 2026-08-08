mod config;
mod tokenizer;
mod weights;

use std::{io, path::PathBuf};

use thiserror::Error;

pub use config::{LlamaConfig, RopeScaling};
pub use tokenizer::{IncrementalDecoder, LlamaTokenizer};
pub use weights::WeightStore;

#[derive(Debug, Error)]
pub enum ModelError {
    #[error("failed to access `{path}`: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: io::Error,
    },
    #[error("failed to parse JSON in `{path}`: {source}")]
    Json {
        path: PathBuf,
        #[source]
        source: serde_json::Error,
    },
    #[error("invalid model configuration: {0}")]
    InvalidConfig(String),
    #[error("failed to load tokenizer: {0}")]
    Tokenizer(String),
    #[error("invalid SafeTensors file `{path}`: {message}")]
    SafeTensors { path: PathBuf, message: String },
    #[error("required tensor `{0}` is missing")]
    MissingTensor(String),
    #[error("tensor `{name}` has dtype {actual:?}; expected BF16")]
    WrongDtype {
        name: String,
        actual: safetensors::Dtype,
    },
    #[error("tensor `{name}` has shape {actual:?}; expected {expected:?}")]
    WrongShape {
        name: String,
        expected: Vec<usize>,
        actual: Vec<usize>,
    },
}

pub(crate) fn read_file(path: &std::path::Path) -> Result<String, ModelError> {
    std::fs::read_to_string(path).map_err(|source| ModelError::Io {
        path: path.to_path_buf(),
        source,
    })
}
