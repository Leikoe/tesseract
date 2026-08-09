use clap::Parser;

#[derive(Debug, Parser)]
#[command(about = "Compile and execute Tesseract's BF16 cuTile validation kernel")]
struct Args {
    #[arg(long, default_value_t = 0)]
    device: usize,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let report = tesseract::cuda::validate_bf16_cutile(args.device)?;
    let nvfp4 = tesseract::cuda::probe_nvfp4_scaled_mma(args.device)?;
    println!("device_id={}", report.device_id);
    println!("dtype=bfloat16");
    println!("elements={}", report.elements);
    println!("gemm_rows={}", report.gemm_rows);
    match nvfp4.status {
        tesseract::cuda::Nvfp4ScaledMmaStatus::Available { max_abs_error } => {
            println!("nvfp4_scaled_mma=available");
            println!("nvfp4_scaled_mma_max_abs_error={max_abs_error}");
        }
        tesseract::cuda::Nvfp4ScaledMmaStatus::Unavailable { detail } => {
            println!("nvfp4_scaled_mma=unavailable");
            println!("nvfp4_scaled_mma_detail={detail}");
        }
    }
    println!("cutile_validation=ok");
    Ok(())
}
