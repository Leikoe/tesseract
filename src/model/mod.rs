mod llama_3_2;
pub(crate) mod weights;

use std::{io, path::Path, path::PathBuf, sync::Arc};

use thiserror::Error;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChatRole {
    System,
    User,
    Assistant,
}

#[derive(Debug, Clone, Copy)]
pub struct ChatMessage<'a> {
    pub role: ChatRole,
    pub content: &'a str,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelSummary {
    pub id: String,
    pub architecture: String,
    pub dtype: String,
    pub layers: usize,
    pub hidden_size: usize,
    pub attention_heads: usize,
    pub kv_heads: usize,
    pub vocab_size: usize,
    pub tensors: usize,
}

#[cfg(feature = "cuda")]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CudaModelReport {
    pub model_id: String,
    pub device_id: usize,
    pub tensors: usize,
    pub bytes: usize,
}

#[cfg(feature = "cuda")]
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct CudaForwardReport {
    pub model_id: String,
    pub prompt_tokens: usize,
    pub next_token_id: u32,
    pub next_token_text: String,
    pub top_logits: Vec<CudaTokenLogit>,
}

#[cfg(feature = "cuda")]
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize)]
pub struct CudaTokenLogit {
    pub token_id: u32,
    pub logit: f32,
}

pub trait IncrementalDecoder: Send {
    fn push(&mut self, token_id: u32) -> Result<String, ModelError>;
}

/// Model-neutral contract used by serving and scheduling code. Architecture
/// details, prompt syntax, tensor names, and model-specific validation stay in
/// the implementing model's source file.
pub trait Model: Send + Sync {
    fn id(&self) -> &str;
    fn render_chat(&self, messages: &[ChatMessage<'_>]) -> Result<String, ModelError>;
    fn encode(&self, text: &str) -> Result<Vec<u32>, ModelError>;
    fn decoder(&self) -> Box<dyn IncrementalDecoder>;
    fn eos_token_ids(&self) -> &[u32];
    fn summary(&self) -> ModelSummary;
}

pub fn load(model_id: &str, model_dir: &Path) -> Result<Arc<dyn Model>, ModelError> {
    if llama_3_2::supports(model_id) {
        return llama_3_2::load(model_id, model_dir).map(|model| Arc::new(model) as Arc<dyn Model>);
    }
    Err(ModelError::UnsupportedModel(model_id.into()))
}

#[cfg(feature = "cuda")]
pub fn load_cuda_executor(
    model_id: &str,
    model_dir: &Path,
    device_id: usize,
    kv_capacity_tokens: usize,
    max_batch_tokens: usize,
    max_running: usize,
) -> Result<Box<dyn crate::engine::ModelExecutor>, ModelError> {
    crate::cuda::enable_persistent_cubin_cache()?;
    if llama_3_2::supports(model_id) {
        return llama_3_2::load_cuda_executor(
            model_id,
            model_dir,
            device_id,
            kv_capacity_tokens,
            max_batch_tokens,
            max_running,
        );
    }
    Err(ModelError::UnsupportedModel(model_id.into()))
}

#[cfg(feature = "cuda")]
pub fn validate_cuda_model(
    model_id: &str,
    model_dir: &Path,
    device_id: usize,
) -> Result<CudaModelReport, ModelError> {
    crate::cuda::enable_persistent_cubin_cache()?;
    if llama_3_2::supports(model_id) {
        return llama_3_2::validate_cuda(model_id, model_dir, device_id);
    }
    Err(ModelError::UnsupportedModel(model_id.into()))
}

#[cfg(feature = "cuda")]
pub fn validate_cuda_next_token(
    model_id: &str,
    model_dir: &Path,
    device_id: usize,
    prompt: &str,
) -> Result<CudaForwardReport, ModelError> {
    crate::cuda::enable_persistent_cubin_cache()?;
    if llama_3_2::supports(model_id) {
        return llama_3_2::validate_cuda_next_token(model_id, model_dir, device_id, prompt);
    }
    Err(ModelError::UnsupportedModel(model_id.into()))
}

#[derive(Debug, Error)]
pub enum ModelError {
    #[error("unsupported model `{0}`")]
    UnsupportedModel(String),
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
    #[error("invalid model input: {0}")]
    InvalidInput(String),
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
    #[error("CUDA model operation failed: {0}")]
    Cuda(String),
    #[cfg(feature = "cuda")]
    #[error(transparent)]
    CudaInfrastructure(#[from] crate::cuda::CudaError),
}

pub(crate) fn read_file(path: &Path) -> Result<String, ModelError> {
    std::fs::read_to_string(path).map_err(|source| ModelError::Io {
        path: path.to_path_buf(),
        source,
    })
}
