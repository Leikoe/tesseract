use std::{path::Path, sync::Arc};

use super::ModelError;

#[derive(Clone)]
pub struct LlamaTokenizer {
    inner: Arc<tokenizers::Tokenizer>,
}

impl LlamaTokenizer {
    pub fn from_model_dir(model_dir: &Path) -> Result<Self, ModelError> {
        let path = model_dir.join("tokenizer.json");
        let inner = tokenizers::Tokenizer::from_file(&path)
            .map_err(|error| ModelError::Tokenizer(error.to_string()))?;
        Ok(Self {
            inner: Arc::new(inner),
        })
    }

    pub fn encode(&self, text: &str) -> Result<Vec<u32>, ModelError> {
        self.inner
            .encode(text, true)
            .map(|encoding| encoding.get_ids().to_vec())
            .map_err(|error| ModelError::Tokenizer(error.to_string()))
    }

    pub fn decode(&self, ids: &[u32], skip_special_tokens: bool) -> Result<String, ModelError> {
        self.inner
            .decode(ids, skip_special_tokens)
            .map_err(|error| ModelError::Tokenizer(error.to_string()))
    }

    pub fn incremental_decoder(&self) -> IncrementalDecoder {
        IncrementalDecoder {
            tokenizer: self.clone(),
            token_ids: Vec::new(),
            decoded: String::new(),
        }
    }
}

/// Correct reference incremental decoding. The initial implementation decodes
/// the complete generated suffix so byte fallback and whitespace behavior are
/// identical to the tokenizer. It will be replaced by a bounded-window stream
/// decoder after GPU correctness is established.
pub struct IncrementalDecoder {
    tokenizer: LlamaTokenizer,
    token_ids: Vec<u32>,
    decoded: String,
}

impl IncrementalDecoder {
    pub fn push(&mut self, token_id: u32) -> Result<String, ModelError> {
        self.token_ids.push(token_id);
        let decoded = self.tokenizer.decode(&self.token_ids, true)?;
        let delta = decoded
            .strip_prefix(&self.decoded)
            .unwrap_or(&decoded)
            .to_owned();
        self.decoded = decoded;
        Ok(delta)
    }
}
