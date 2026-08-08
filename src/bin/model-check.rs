use std::path::PathBuf;

use anyhow::Context;
use clap::Parser;
use tesseract::{
    config::DEFAULT_MODEL_ID,
    model::{self, ChatMessage, ChatRole},
};

#[derive(Debug, Parser)]
#[command(
    name = "tesseract-model-check",
    about = "Validate a Tesseract model checkpoint without allocating GPU memory"
)]
struct Args {
    #[arg(long)]
    model_path: PathBuf,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let model = model::load(DEFAULT_MODEL_ID, &args.model_path)
        .context("model validation and loading failed")?;
    let prompt = model
        .render_chat(&[ChatMessage {
            role: ChatRole::User,
            content: "Hello",
        }])
        .context("model prompt rendering probe failed")?;
    let probe = model.encode(&prompt).context("tokenizer probe failed")?;

    let summary = model.summary();
    println!("model_id={}", summary.id);
    println!("architecture={}", summary.architecture);
    println!("dtype={}", summary.dtype);
    println!("layers={}", summary.layers);
    println!("hidden_size={}", summary.hidden_size);
    println!("attention_heads={}", summary.attention_heads);
    println!("kv_heads={}", summary.kv_heads);
    println!("vocab_size={}", summary.vocab_size);
    println!("tensors={}", summary.tensors);
    println!("tokenizer_probe_tokens={}", probe.len());
    println!("validation=ok");
    Ok(())
}
