use std::path::PathBuf;

use clap::Parser;

#[derive(Debug, Parser)]
#[command(about = "Run one real BF16 Llama forward pass on CUDA")]
struct Args {
    #[arg(long, default_value = tesseract::config::DEFAULT_MODEL_ID)]
    model: String,
    #[arg(long)]
    model_path: PathBuf,
    #[arg(long, default_value_t = 0)]
    device: usize,
    #[arg(long, default_value = "The capital of France is")]
    prompt: String,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let report = tesseract::model::validate_cuda_next_token(
        &args.model,
        &args.model_path,
        args.device,
        &args.prompt,
    )?;
    println!("model_id={}", report.model_id);
    println!("prompt_tokens={}", report.prompt_tokens);
    println!("next_token_id={}", report.next_token_id);
    println!("next_token_text={:?}", report.next_token_text);
    println!("cuda_forward_validation=ok");
    Ok(())
}
