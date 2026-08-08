use std::{net::SocketAddr, path::PathBuf, time::Duration};

use clap::Parser;

pub const DEFAULT_MODEL_ID: &str = "meta-llama/Llama-3.2-1B-Instruct";

#[derive(Debug, Clone, Parser)]
#[command(
    name = "tesseract",
    version,
    about = "A production BF16 inference server"
)]
pub struct ServerConfig {
    #[arg(long, env = "TESSERACT_LISTEN", default_value = "0.0.0.0:8000")]
    pub listen: SocketAddr,

    #[arg(long, env = "TESSERACT_MODEL", default_value = DEFAULT_MODEL_ID)]
    pub model: String,

    #[arg(
        long,
        env = "TESSERACT_MODEL_PATH",
        default_value = "/home/ubuntu/models/Llama-3.2-1B-Instruct"
    )]
    pub model_path: PathBuf,

    #[arg(long, env = "TESSERACT_MAX_QUEUE", default_value_t = 256)]
    pub max_queue: usize,

    #[arg(long, env = "TESSERACT_MAX_RUNNING", default_value_t = 32)]
    pub max_running: usize,

    #[arg(long, env = "TESSERACT_MAX_BATCH_TOKENS", default_value_t = 4096)]
    pub max_batch_tokens: usize,

    #[arg(long, env = "TESSERACT_PREFILL_CHUNK_TOKENS", default_value_t = 512)]
    pub prefill_chunk_tokens: usize,

    #[arg(long, env = "TESSERACT_MAX_SEQUENCE_LENGTH", default_value_t = 131_072)]
    pub max_sequence_length: usize,

    #[arg(long, env = "TESSERACT_KV_CAPACITY_TOKENS", default_value_t = 32_768)]
    pub kv_capacity_tokens: usize,

    #[arg(long, env = "TESSERACT_OUTPUT_BUFFER", default_value_t = 32)]
    pub output_buffer: usize,

    #[arg(long, env = "TESSERACT_SHUTDOWN_GRACE_MS", default_value_t = 30_000)]
    pub shutdown_grace_ms: u64,

    #[arg(long, env = "TESSERACT_LOG", default_value = "info")]
    pub log: String,

    #[arg(long, env = "TESSERACT_JSON_LOGS", default_value_t = false)]
    pub json_logs: bool,
}

impl ServerConfig {
    pub fn shutdown_grace(&self) -> Duration {
        Duration::from_millis(self.shutdown_grace_ms)
    }

    pub fn validate(&self) -> anyhow::Result<()> {
        anyhow::ensure!(self.max_queue > 0, "max_queue must be positive");
        anyhow::ensure!(self.max_running > 0, "max_running must be positive");
        anyhow::ensure!(
            self.max_batch_tokens >= self.max_running,
            "max_batch_tokens must be at least max_running"
        );
        anyhow::ensure!(
            self.prefill_chunk_tokens > 0,
            "prefill_chunk_tokens must be positive"
        );
        anyhow::ensure!(
            self.max_sequence_length > 0,
            "max_sequence_length must be positive"
        );
        anyhow::ensure!(
            self.kv_capacity_tokens >= self.max_running,
            "kv_capacity_tokens must be at least max_running"
        );
        anyhow::ensure!(self.output_buffer > 0, "output_buffer must be positive");
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct EngineConfig {
    pub max_pending: usize,
    pub max_running: usize,
    pub max_batch_tokens: usize,
    pub prefill_chunk_tokens: usize,
    pub max_sequence_length: usize,
    pub kv_capacity_tokens: usize,
    pub output_buffer: usize,
}

impl From<&ServerConfig> for EngineConfig {
    fn from(value: &ServerConfig) -> Self {
        Self {
            max_pending: value.max_queue,
            max_running: value.max_running,
            max_batch_tokens: value.max_batch_tokens,
            prefill_chunk_tokens: value.prefill_chunk_tokens,
            max_sequence_length: value.max_sequence_length,
            kv_capacity_tokens: value.kv_capacity_tokens,
            output_buffer: value.output_buffer,
        }
    }
}
