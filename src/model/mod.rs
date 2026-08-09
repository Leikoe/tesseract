mod llama_3_2;
mod qwen3_5_moe;
mod tokenizer;
pub(crate) mod weights;

use std::{io, path::Path, path::PathBuf, sync::Arc};

use serde::Deserialize;
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

#[derive(Debug, Deserialize)]
struct ModelManifest {
    #[serde(default)]
    architectures: Vec<String>,
    #[serde(default)]
    model_type: String,
}

impl ModelManifest {
    fn unsupported(&self, model_id: &str) -> ModelError {
        ModelError::UnsupportedArchitecture {
            model_id: model_id.into(),
            architectures: self.architectures.clone(),
            model_type: self.model_type.clone(),
        }
    }
}

fn model_manifest(model_dir: &Path) -> Result<ModelManifest, ModelError> {
    let path = model_dir.join("config.json");
    let text = read_file(&path)?;
    serde_json::from_str(&text).map_err(|source| ModelError::Json { path, source })
}

fn declared_architecture<'a>(
    manifest: &'a ModelManifest,
    model_id: &str,
) -> Result<&'a str, ModelError> {
    manifest
        .architectures
        .first()
        .map(String::as_str)
        .ok_or_else(|| ModelError::NoArchitecture(model_id.into()))
}

pub fn load(model_id: &str, model_dir: &Path) -> Result<Arc<dyn Model>, ModelError> {
    let manifest = model_manifest(model_dir)?;
    match declared_architecture(&manifest, model_id)? {
        llama_3_2::Llama32::ARCH_NAME => {
            Ok(Arc::new(llama_3_2::Llama32::load(model_id, model_dir)?))
        }
        qwen3_5_moe::Qwen35MoeText::ARCH_NAME => Ok(Arc::new(qwen3_5_moe::Qwen35MoeText::load(
            model_id, model_dir,
        )?)),
        _ => Err(manifest.unsupported(model_id)),
    }
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
    let manifest = model_manifest(model_dir)?;
    match declared_architecture(&manifest, model_id)? {
        llama_3_2::Llama32::ARCH_NAME => llama_3_2::load_cuda_executor(
            model_id,
            model_dir,
            device_id,
            kv_capacity_tokens,
            max_batch_tokens,
            max_running,
        ),
        qwen3_5_moe::Qwen35MoeText::ARCH_NAME => qwen3_5_moe::load_cuda_executor(
            model_id,
            model_dir,
            device_id,
            kv_capacity_tokens,
            max_batch_tokens,
            max_running,
        ),
        _ => Err(manifest.unsupported(model_id)),
    }
}

#[cfg(feature = "cuda")]
pub fn validate_cuda_model(
    model_id: &str,
    model_dir: &Path,
    device_id: usize,
) -> Result<CudaModelReport, ModelError> {
    crate::cuda::enable_persistent_cubin_cache()?;
    let manifest = model_manifest(model_dir)?;
    match declared_architecture(&manifest, model_id)? {
        llama_3_2::Llama32::ARCH_NAME => {
            llama_3_2::validate_cuda_model(model_id, model_dir, device_id)
        }
        qwen3_5_moe::Qwen35MoeText::ARCH_NAME => {
            qwen3_5_moe::validate_cuda_model(model_id, model_dir, device_id)
        }
        _ => Err(manifest.unsupported(model_id)),
    }
}

#[cfg(feature = "cuda")]
pub fn validate_cuda_next_token(
    model_id: &str,
    model_dir: &Path,
    device_id: usize,
    prompt: &str,
) -> Result<CudaForwardReport, ModelError> {
    crate::cuda::enable_persistent_cubin_cache()?;
    let manifest = model_manifest(model_dir)?;
    match declared_architecture(&manifest, model_id)? {
        llama_3_2::Llama32::ARCH_NAME => {
            llama_3_2::validate_cuda_next_token(model_id, model_dir, device_id, prompt)
        }
        qwen3_5_moe::Qwen35MoeText::ARCH_NAME => {
            qwen3_5_moe::validate_cuda_next_token(model_id, model_dir, device_id, prompt)
        }
        _ => Err(manifest.unsupported(model_id)),
    }
}

#[derive(Debug, Error)]
pub enum ModelError {
    #[error("unsupported model `{0}`")]
    UnsupportedModel(String),
    #[error("model `{0}` does not declare an architecture")]
    NoArchitecture(String),
    #[error("unsupported execution path: {0}")]
    UnsupportedExecution(String),
    #[error(
        "model `{model_id}` declares unsupported architecture(s) {architectures:?} and model type `{model_type}`"
    )]
    UnsupportedArchitecture {
        model_id: String,
        architectures: Vec<String>,
        model_type: String,
    },
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
    #[error("tensor `{name}` has dtype {actual}; expected {expected}")]
    WrongDtype {
        name: String,
        expected: String,
        actual: String,
    },
    #[error("tensor `{name}` has shape {actual:?}; expected {expected:?}")]
    WrongShape {
        name: String,
        expected: Vec<usize>,
        actual: Vec<usize>,
    },
    #[error("tensor `{name}` is invalid: {message}")]
    InvalidTensor { name: String, message: String },
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dispatch_resolves_architecture_without_model_id_checks() {
        let manifest = ModelManifest {
            architectures: vec!["LlamaForCausalLM".into()],
            model_type: "llama".into(),
        };
        assert_eq!(
            declared_architecture(&manifest, "test-model").unwrap(),
            llama_3_2::Llama32::ARCH_NAME
        );
    }

    #[test]
    fn dispatch_resolves_qwen_text_from_the_outer_conditional_architecture() {
        let manifest = ModelManifest {
            architectures: vec!["Qwen3_5MoeForConditionalGeneration".into()],
            model_type: "qwen3_5_moe".into(),
        };
        assert_eq!(
            declared_architecture(&manifest, "test-model").unwrap(),
            qwen3_5_moe::Qwen35MoeText::ARCH_NAME
        );
    }

    #[test]
    fn dispatch_rejects_unknown_architectures() {
        let manifest = ModelManifest {
            architectures: vec!["UnknownForCausalLM".into()],
            model_type: "unknown".into(),
        };
        assert_eq!(
            declared_architecture(&manifest, "test-model").unwrap(),
            "UnknownForCausalLM"
        );
    }

    #[test]
    fn dispatch_distinguishes_a_missing_architecture() {
        let manifest = ModelManifest {
            architectures: Vec::new(),
            model_type: "unknown".into(),
        };
        let error = declared_architecture(&manifest, "test-model").unwrap_err();
        assert!(matches!(error, ModelError::NoArchitecture(model) if model == "test-model"));
    }
}
