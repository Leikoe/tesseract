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
    println!("device_id={}", report.device_id);
    println!("dtype=bfloat16");
    println!("elements={}", report.elements);
    println!("cutile_validation=ok");
    Ok(())
}
