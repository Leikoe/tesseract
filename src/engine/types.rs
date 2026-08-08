use serde::{Deserialize, Serialize};
use uuid::Uuid;

pub type RequestId = Uuid;

#[derive(Debug, Clone)]
pub struct GenerateRequest {
    pub id: RequestId,
    pub prompt: String,
    pub params: GenerationParams,
}

#[derive(Debug, Clone)]
pub struct GenerationParams {
    pub max_tokens: usize,
    pub temperature: f32,
    pub top_p: f32,
    pub seed: u64,
    pub stop: Vec<String>,
}

impl GenerationParams {
    pub fn validate(&self) -> Result<(), String> {
        if self.max_tokens == 0 {
            return Err("max_tokens must be positive".into());
        }
        if !self.temperature.is_finite() || self.temperature < 0.0 {
            return Err("temperature must be finite and non-negative".into());
        }
        if !self.top_p.is_finite() || !(0.0..=1.0).contains(&self.top_p) || self.top_p == 0.0 {
            return Err("top_p must be in (0, 1]".into());
        }
        if self.stop.iter().any(|stop| stop.is_empty()) {
            return Err("stop strings must not be empty".into());
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FinishReason {
    Stop,
    Length,
    Cancelled,
    Error,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Usage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
}

impl Usage {
    pub fn new(prompt_tokens: usize, completion_tokens: usize) -> Self {
        Self {
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens + completion_tokens,
        }
    }
}

#[derive(Debug, Clone)]
pub enum GenerationEvent {
    Delta { text: String, token_id: Option<u32> },
    Finished { reason: FinishReason, usage: Usage },
    Failed { message: String },
}
