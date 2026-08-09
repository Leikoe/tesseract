use std::{io::Write, path::PathBuf};

use clap::Parser;

#[derive(Debug, Parser)]
#[command(about = "Compare Tesseract quantized CUDA kernels with native Marlin")]
struct Args {
    #[arg(long, default_value_t = 0)]
    device: usize,

    #[arg(long, value_delimiter = ',', default_value = "16,512,8192")]
    rows: Vec<usize>,

    #[arg(long, default_value_t = 2)]
    warmup_iterations: usize,

    #[arg(long, default_value_t = 7)]
    iterations: usize,

    #[arg(long)]
    output: Option<PathBuf>,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let report = tesseract::cuda::marlin::benchmark(
        args.device,
        &args.rows,
        args.warmup_iterations,
        args.iterations,
    )?;
    let encoded = serde_json::to_vec_pretty(&report)?;
    if let Some(path) = args.output {
        let mut file = std::fs::File::create(path)?;
        file.write_all(&encoded)?;
        file.write_all(b"\n")?;
    } else {
        std::io::stdout().write_all(&encoded)?;
        std::io::stdout().write_all(b"\n")?;
    }
    Ok(())
}
