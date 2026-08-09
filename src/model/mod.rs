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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Architecture {
    Llama,
    Qwen35MoeText,
}

impl Architecture {
    const fn name(self) -> &'static str {
        match self {
            Self::Llama => "llama",
            Self::Qwen35MoeText => "qwen3_5_moe_text",
        }
    }
}

fn architecture(model_id: &str, model_dir: &Path) -> Result<Architecture, ModelError> {
    let path = model_dir.join("config.json");
    let text = read_file(&path)?;
    let manifest: ModelManifest =
        serde_json::from_str(&text).map_err(|source| ModelError::Json {
            path: path.clone(),
            source,
        })?;
    let architecture =
        resolve_architecture(&manifest).ok_or_else(|| ModelError::UnsupportedArchitecture {
            model_id: model_id.into(),
            architectures: manifest.architectures,
            model_type: manifest.model_type,
        })?;
    tracing::debug!(
        architecture = architecture.name(),
        model_id,
        "resolved model architecture"
    );
    Ok(architecture)
}

fn resolve_architecture(manifest: &ModelManifest) -> Option<Architecture> {
    let declares = |name: &str| {
        manifest
            .architectures
            .iter()
            .any(|architecture| architecture == name)
    };
    match manifest.model_type.as_str() {
        "llama" if declares("LlamaForCausalLM") => Some(Architecture::Llama),
        "qwen3_5_moe" if declares("Qwen3_5MoeForConditionalGeneration") => {
            Some(Architecture::Qwen35MoeText)
        }
        _ => None,
    }
}

pub fn load(model_id: &str, model_dir: &Path) -> Result<Arc<dyn Model>, ModelError> {
    match architecture(model_id, model_dir)? {
        Architecture::Llama => llama_3_2::load(model_id, model_dir),
        Architecture::Qwen35MoeText => qwen3_5_moe::load(model_id, model_dir),
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
    match architecture(model_id, model_dir)? {
        Architecture::Llama => llama_3_2::load_cuda_executor(
            model_id,
            model_dir,
            device_id,
            kv_capacity_tokens,
            max_batch_tokens,
            max_running,
        ),
        Architecture::Qwen35MoeText => qwen3_5_moe::load_cuda_executor(
            model_id,
            model_dir,
            device_id,
            kv_capacity_tokens,
            max_batch_tokens,
            max_running,
        ),
    }
}

#[cfg(feature = "cuda")]
pub fn validate_cuda_model(
    model_id: &str,
    model_dir: &Path,
    device_id: usize,
) -> Result<CudaModelReport, ModelError> {
    crate::cuda::enable_persistent_cubin_cache()?;
    match architecture(model_id, model_dir)? {
        Architecture::Llama => llama_3_2::validate_cuda_model(model_id, model_dir, device_id),
        Architecture::Qwen35MoeText => {
            qwen3_5_moe::validate_cuda_model(model_id, model_dir, device_id)
        }
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
    match architecture(model_id, model_dir)? {
        Architecture::Llama => {
            llama_3_2::validate_cuda_next_token(model_id, model_dir, device_id, prompt)
        }
        Architecture::Qwen35MoeText => {
            qwen3_5_moe::validate_cuda_next_token(model_id, model_dir, device_id, prompt)
        }
    }
}

#[derive(Debug, Error)]
pub enum ModelError {
    #[error("unsupported model `{0}`")]
    UnsupportedModel(String),
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
        assert_eq!(resolve_architecture(&manifest), Some(Architecture::Llama));
    }

    #[test]
    fn dispatch_resolves_qwen_text_from_the_outer_conditional_architecture() {
        let manifest = ModelManifest {
            architectures: vec!["Qwen3_5MoeForConditionalGeneration".into()],
            model_type: "qwen3_5_moe".into(),
        };
        assert_eq!(
            resolve_architecture(&manifest),
            Some(Architecture::Qwen35MoeText)
        );
    }

    #[test]
    fn dispatch_rejects_unknown_architectures() {
        let manifest = ModelManifest {
            architectures: vec!["UnknownForCausalLM".into()],
            model_type: "unknown".into(),
        };
        assert!(resolve_architecture(&manifest).is_none());
    }
}
