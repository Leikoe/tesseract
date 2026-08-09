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
    let nvfp4 = tesseract::cuda::probe_nvfp4(args.device)?;
    println!("device_id={}", report.device_id);
    println!("dtype=bfloat16");
    println!("elements={}", report.elements);
    println!("gemm_rows={}", report.gemm_rows);
    print_capability("nvfp4_scaled_mma", nvfp4.scaled_mma);
    print_capability("nvfp4_byte_decode_mma", nvfp4.byte_decode_mma);
    print_capability("nvfp4_w4a16_linear", nvfp4.w4a16_linear);
    print_capability("nvfp4_grouped_w4a16", nvfp4.grouped_w4a16);
    println!("cutile_validation=ok");
    Ok(())
}

fn print_capability(name: &str, status: tesseract::cuda::CudaKernelCapability) {
    match status {
        tesseract::cuda::CudaKernelCapability::Available { max_abs_error } => {
            println!("{name}=available");
            println!("{name}_max_abs_error={max_abs_error}");
        }
        tesseract::cuda::CudaKernelCapability::Unavailable { detail } => {
            println!("{name}=unavailable");
            println!("{name}_detail={detail}");
        }
    }
}
