use anyhow::Context;
use clap::Parser;
use tesseract::config::ServerConfig;

fn main() -> anyhow::Result<()> {
    let config = ServerConfig::parse();
    config.validate().context("invalid server configuration")?;
    anyhow::bail!(
        "the A100 CUDA backend is not linked yet; run the library test suite while v1 GPU integration is in progress"
    )
}
