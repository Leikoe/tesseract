use clap::Parser;

#[derive(Debug, Parser)]
#[command(about = "Probe Tesseract's CUDA kernel capabilities")]
struct Args {
    #[arg(long, default_value_t = 0)]
    device: usize,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let report = tesseract::cuda::probe_bf16_cutile(args.device)?;
    let quantized = tesseract::cuda::probe_quantized_linears(args.device)?;
    let marlin = tesseract::cuda::probe_marlin(args.device)?;
    println!("device_id={}", report.device_id);
    println!("dtype=bfloat16");
    println!("elements={}", report.elements);
    println!("gemm_rows={}", report.gemm_rows);
    print_capability("fp8_w8a16_linear", quantized.fp8_w8a16);
    print_capability("nvfp4_scaled_mma", quantized.scaled_mma);
    print_capability("nvfp4_byte_decode_mma", quantized.byte_decode_mma);
    print_capability("nvfp4_w4a16_linear", quantized.w4a16_linear);
    print_capability("nvfp4_grouped_w4a16", quantized.grouped_w4a16);
    print_capability("moe_device_routing", quantized.moe_routing);
    print_capability("gdn_recurrent_decode", quantized.gdn_decode);
    print_capability("qwen_full_attention", quantized.qwen_full_attention);
    println!("marlin_fp8_max_abs_error={}", marlin.fp8_max_abs_error);
    println!("marlin_nvfp4_max_abs_error={}", marlin.nvfp4_max_abs_error);
    println!("cutile_probe=ok");
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
