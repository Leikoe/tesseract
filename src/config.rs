use std::{net::SocketAddr, path::PathBuf, time::Duration};

use clap::{Args, Parser, Subcommand, ValueEnum};

pub const DEFAULT_MODEL_ID: &str = "meta-llama/Llama-3.2-1B-Instruct";

#[derive(Debug, Clone, Parser)]
#[command(
    name = "tesseract",
    version,
    about = "A production BF16 inference server",
    args_conflicts_with_subcommands = true
)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Option<CliCommand>,

    #[command(flatten)]
    pub server: ServerConfig,
}

#[derive(Debug, Clone, Subcommand)]
pub enum CliCommand {
    /// Benchmark an OpenAI-compatible serving endpoint.
    Bench(BenchmarkConfig),
}

#[derive(Debug, Clone, Args)]
pub struct BenchmarkConfig {
    #[arg(long, default_value = "http://127.0.0.1:8000")]
    pub base_url: String,

    #[arg(long, value_enum, default_value_t = BenchmarkApi::ChatCompletions)]
    pub api: BenchmarkApi,

    #[arg(long, help = "Override the API's default endpoint path")]
    pub endpoint: Option<String>,

    #[arg(long, env = "TESSERACT_MODEL", default_value = DEFAULT_MODEL_ID)]
    pub model: String,

    #[arg(long, value_enum, default_value_t = BenchmarkDataset::Random)]
    pub dataset: BenchmarkDataset,

    #[arg(long, help = "Tokenizer JSON used to construct token-length workloads")]
    pub tokenizer: Option<PathBuf>,

    #[arg(long, default_value_t = 1000)]
    pub num_prompts: usize,

    #[arg(long)]
    pub max_concurrency: Option<usize>,

    #[arg(long, default_value = "inf")]
    pub request_rate: f64,

    #[arg(long, default_value_t = 128)]
    pub output_len: usize,

    #[arg(long, default_value_t = 1024, help = "Mean random context token count")]
    pub input_len: usize,

    #[arg(
        long,
        default_value_t = 0.0,
        help = "Symmetric input/output length variation in [0, 1)"
    )]
    pub length_variation: f64,

    #[arg(
        long,
        default_value_t = 0,
        help = "Shared prefix tokens prepended in addition to input-len"
    )]
    pub shared_prefix_len: usize,

    #[arg(long, default_value_t = 1)]
    pub warmup_requests: usize,

    #[arg(long, help = "Use this prompt for every request")]
    pub prompt: Option<String>,

    #[arg(long, default_value_t = 42)]
    pub seed: u64,

    #[arg(long, default_value_t = 300.0)]
    pub timeout_seconds: f64,

    #[arg(long, value_name = "NAME:VALUE")]
    pub header: Vec<String>,

    #[arg(long)]
    pub output: Option<PathBuf>,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
pub enum BenchmarkApi {
    ChatCompletions,
    Completions,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
pub enum BenchmarkDataset {
    Random,
    Builtin,
}

impl BenchmarkDataset {
    pub const fn name(self) -> &'static str {
        match self {
            Self::Random => "random",
            Self::Builtin => "builtin",
        }
    }
}

impl BenchmarkApi {
    pub const fn endpoint(self) -> &'static str {
        match self {
            Self::ChatCompletions => "/v1/chat/completions",
            Self::Completions => "/v1/completions",
        }
    }

    pub const fn name(self) -> &'static str {
        match self {
            Self::ChatCompletions => "chat-completions",
            Self::Completions => "completions",
        }
    }
}

#[derive(Debug, Clone, Args)]
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

    #[arg(long, env = "TESSERACT_DEVICE", default_value_t = 0)]
    pub device: usize,

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
            self.max_sequence_length <= u32::MAX as usize,
            "max_sequence_length must fit in a u32 position"
        );
        anyhow::ensure!(
            self.kv_capacity_tokens >= self.max_running,
            "kv_capacity_tokens must be at least max_running"
        );
        anyhow::ensure!(
            self.kv_capacity_tokens <= u32::MAX as usize,
            "kv_capacity_tokens must fit in a u32 slot ID"
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bare_server_flags_remain_compatible() {
        let cli = Cli::try_parse_from([
            "tesseract",
            "--listen",
            "127.0.0.1:9000",
            "--max-running",
            "4",
        ])
        .unwrap();
        assert!(cli.command.is_none());
        assert_eq!(cli.server.listen.to_string(), "127.0.0.1:9000");
        assert_eq!(cli.server.max_running, 4);
    }

    #[test]
    fn parses_serving_benchmark_subcommand() {
        let cli = Cli::try_parse_from([
            "tesseract",
            "bench",
            "--max-concurrency",
            "4",
            "--output",
            "/tmp/tesseract-bench.json",
        ])
        .unwrap();
        let Some(CliCommand::Bench(bench)) = cli.command else {
            panic!("expected bench subcommand");
        };
        assert_eq!(bench.max_concurrency, Some(4));
        assert_eq!(
            bench.output,
            Some(PathBuf::from("/tmp/tesseract-bench.json"))
        );
        assert_eq!(cli.server.listen.to_string(), "0.0.0.0:8000");
    }
}
