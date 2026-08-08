#[cfg(feature = "cuda")]
use std::{env, path::PathBuf, process::Command, sync::Arc};

use anyhow::Context;
use clap::Parser;
#[cfg(feature = "cuda")]
use tesseract::config::{BenchmarkConfig, ServerConfig};
use tesseract::config::{Cli, CliCommand};

#[cfg(feature = "cuda")]
#[tokio::main]
async fn main() -> anyhow::Result<()> {
    use axum::http::HeaderName;
    use tesseract::{
        api::{self, AppState},
        config::EngineConfig,
        engine::EngineHandle,
        metrics::Metrics,
        model,
    };
    use tower_http::{
        request_id::{MakeRequestUuid, PropagateRequestIdLayer, SetRequestIdLayer},
        trace::TraceLayer,
    };

    let cli = Cli::parse();
    if let Some(CliCommand::Bench(config)) = cli.command {
        return run_benchmark(config);
    }
    let config = cli.server;
    config.validate().context("invalid server configuration")?;
    init_tracing(&config)?;

    let metrics = Arc::new(Metrics::default());
    let engine_config = EngineConfig::from(&config);
    let model_id = config.model.clone();
    let model_path = config.model_path.clone();
    let device = config.device;
    let kv_capacity_tokens = config.kv_capacity_tokens;
    let max_batch_tokens = config.max_batch_tokens;
    let max_running = config.max_running;
    let engine = EngineHandle::spawn_with_factory(
        move || {
            model::load_cuda_backend(
                &model_id,
                &model_path,
                device,
                kv_capacity_tokens,
                max_batch_tokens,
                max_running,
            )
        },
        engine_config,
        config.max_queue,
        Arc::clone(&metrics),
    )
    .context("start inference engine")?;

    let request_id_header = HeaderName::from_static("x-request-id");
    let app = api::router(AppState {
        engine: engine.clone(),
        metrics: Arc::clone(&metrics),
    })
    .layer(PropagateRequestIdLayer::new(request_id_header.clone()))
    .layer(SetRequestIdLayer::new(request_id_header, MakeRequestUuid))
    .layer(TraceLayer::new_for_http());
    let listener = tokio::net::TcpListener::bind(config.listen)
        .await
        .with_context(|| format!("bind {}", config.listen))?;
    tracing::info!(
        listen = %config.listen,
        model = %engine.model_id(),
        device,
        "inference server ready"
    );

    let shutdown_engine = engine.clone();
    let shutdown_metrics = Arc::clone(&metrics);
    let shutdown_grace = config.shutdown_grace();
    axum::serve(listener, app)
        .with_graceful_shutdown(async move {
            if let Err(error) = tokio::signal::ctrl_c().await {
                tracing::error!(%error, "failed to install shutdown signal handler");
            }
            tracing::info!("shutdown requested");
            shutdown_metrics.set_ready(false);
            if let Err(error) = shutdown_engine.shutdown(shutdown_grace).await {
                tracing::error!(%error, "engine did not shut down cleanly");
            }
        })
        .await
        .context("serve HTTP")?;
    Ok(())
}

#[cfg(feature = "cuda")]
fn run_benchmark(config: BenchmarkConfig) -> anyhow::Result<()> {
    anyhow::ensure!(
        !cfg!(debug_assertions),
        "serving benchmarks require a release binary; use `cargo bench-a100`"
    );
    let repo_path = env::var_os("TESSERACT_REPO_PATH")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")));
    let script = repo_path.join("scripts/benchmark/run_a100.py");
    anyhow::ensure!(
        script.is_file(),
        "benchmark runner does not exist at `{}`; run from a Tesseract checkout or set TESSERACT_REPO_PATH",
        script.display()
    );
    let executable = env::current_exe().context("resolve the Tesseract executable")?;
    let invocation = env::args().collect::<Vec<_>>().join(" ");
    let mut benchmark = Command::new("python3");
    benchmark
        .arg(script)
        .arg("--server-binary")
        .arg(executable)
        .arg("--invocation")
        .arg(invocation)
        .arg("--listen")
        .arg(config.listen)
        .arg("--batch1-requests")
        .arg(config.batch1_requests.to_string())
        .arg("--concurrency")
        .arg(config.concurrency.to_string())
        .arg("--concurrent-requests")
        .arg(config.concurrent_requests.to_string())
        .arg("--max-tokens")
        .arg(config.max_tokens.to_string())
        .arg("--warmup-requests")
        .arg(config.warmup_requests.to_string())
        .arg("--ready-timeout-seconds")
        .arg(config.ready_timeout_seconds.to_string());
    if let Some(path) = config.model_path {
        benchmark.arg("--model-path").arg(path);
    }
    if let Some(revision) = config.model_revision {
        benchmark.arg("--model-revision").arg(revision);
    }
    if let Some(model) = config.model {
        benchmark.arg("--model").arg(model);
    }
    if let Some(output) = config.output {
        benchmark.arg("--output").arg(output);
    }
    for argument in config.server_arg {
        benchmark.arg("--server-arg").arg(argument);
    }
    if config.allow_dirty {
        benchmark.arg("--allow-dirty");
    }
    let status = benchmark
        .status()
        .context("launch the serving benchmark runner")?;
    anyhow::ensure!(status.success(), "serving benchmark failed with {status}");
    Ok(())
}

#[cfg(feature = "cuda")]
fn init_tracing(config: &ServerConfig) -> anyhow::Result<()> {
    use tracing_subscriber::EnvFilter;

    let filter = EnvFilter::try_new(&config.log).context("invalid log filter")?;
    if config.json_logs {
        tracing_subscriber::fmt()
            .json()
            .with_env_filter(filter)
            .try_init()
            .map_err(|error| anyhow::anyhow!("initialize JSON logging: {error}"))?;
    } else {
        tracing_subscriber::fmt()
            .with_env_filter(filter)
            .try_init()
            .map_err(|error| anyhow::anyhow!("initialize logging: {error}"))?;
    }
    Ok(())
}

#[cfg(not(feature = "cuda"))]
fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Some(CliCommand::Bench(_)) => {
            anyhow::bail!("serving benchmarks require --features cuda")
        }
        None => {
            cli.server
                .validate()
                .context("invalid server configuration")?;
            anyhow::bail!("tesseract must be built with --features cuda")
        }
    }
}
