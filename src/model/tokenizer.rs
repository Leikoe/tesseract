use std::{path::Path, sync::Arc};

use super::{IncrementalDecoder, ModelError};

/// Shared tokenizer transport and incremental text reconstruction.
///
/// Architecture adapters own prompt syntax and warmup text. This type owns
/// only tokenizer.json mechanics, so adding a model never duplicates the
/// subtle incremental-decoding behavior.
#[derive(Clone)]
pub(super) struct Tokenizer {
    inner: Arc<tokenizers::Tokenizer>,
}

impl Tokenizer {
    pub(super) fn load(model_dir: &Path) -> Result<Self, ModelError> {
        let path = model_dir.join("tokenizer.json");
        let inner = tokenizers::Tokenizer::from_file(&path)
            .map_err(|error| ModelError::Tokenizer(error.to_string()))?;
        Ok(Self {
            inner: Arc::new(inner),
        })
    }

    pub(super) fn encode(&self, text: &str) -> Result<Vec<u32>, ModelError> {
        self.inner
            .encode(text, true)
            .map(|encoding| encoding.get_ids().to_vec())
            .map_err(|error| ModelError::Tokenizer(error.to_string()))
    }

    pub(super) fn warm(&self, representative_text: &str) -> Result<(), ModelError> {
        let token_ids = self.encode(representative_text)?;
        if token_ids.is_empty() {
            return Err(ModelError::Tokenizer(
                "tokenizer warmup produced no tokens".into(),
            ));
        }
        Ok(())
    }

    pub(super) fn incremental_decoder(&self) -> Box<dyn IncrementalDecoder> {
        Box::new(Decoder {
            tokenizer: self.clone(),
            token_ids: Vec::new(),
            decoded: String::new(),
        })
    }

    fn decode(&self, ids: &[u32]) -> Result<String, ModelError> {
        self.inner
            .decode(ids, true)
            .map_err(|error| ModelError::Tokenizer(error.to_string()))
    }
}

struct Decoder {
    tokenizer: Tokenizer,
    token_ids: Vec<u32>,
    decoded: String,
}

impl IncrementalDecoder for Decoder {
    fn push(&mut self, token_id: u32) -> Result<String, ModelError> {
        self.token_ids.push(token_id);
        let decoded = self.tokenizer.decode(&self.token_ids)?;
        let delta = decoded
            .strip_prefix(&self.decoded)
            .unwrap_or(&decoded)
            .to_owned();
        self.decoded = decoded;
        Ok(delta)
    }
}
