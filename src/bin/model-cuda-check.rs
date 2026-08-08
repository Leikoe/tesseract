use std::path::PathBuf;

use clap::Parser;

#[derive(Debug, Parser)]
#[command(about = "Load and validate a complete BF16 model on CUDA")]
struct Args {
    #[arg(long, default_value = tesseract::config::DEFAULT_MODEL_ID)]
    model: String,
    #[arg(long)]
    model_path: PathBuf,
    #[arg(long, default_value_t = 0)]
    device: usize,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let report = tesseract::model::validate_cuda_model(&args.model, &args.model_path, args.device)?;
    println!("model_id={}", report.model_id);
    println!("device_id={}", report.device_id);
    println!("dtype=bfloat16");
    println!("tensors={}", report.tensors);
    println!("bytes={}", report.bytes);
    println!("cuda_model_validation=ok");
    Ok(())
}
