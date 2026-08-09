#[cfg(feature = "cuda")]
use std::sync::Arc;

use anyhow::Context;
use clap::Parser;
#[cfg(feature = "cuda")]
use tesseract::config::ServerConfig;
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
        return tesseract::benchmark::run(config).await;
    }
    let config = cli.server;
    config.validate().context("invalid server configuration")?;
    init_tracing(&config)?;

    let metrics = Arc::new(Metrics::default());
    let engine_config = EngineConfig::from(&config);
    let model_path = config.model_path.clone();
    let device = config.device;
    let kv_capacity_tokens = config.kv_capacity_tokens;
    let max_batch_tokens = config.max_batch_tokens;
    let max_running = config.max_running;
    let engine = EngineHandle::spawn_with_factory(
        config.model.clone(),
        move || {
            model::load_cuda_executor(
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
#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Some(CliCommand::Bench(config)) => tesseract::benchmark::run(config).await,
        None => {
            cli.server
                .validate()
                .context("invalid server configuration")?;
            anyhow::bail!("tesseract must be built with --features cuda")
        }
    }
}
