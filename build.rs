fn main() {
    println!("cargo:rerun-if-changed=src/cuda/marlin/native");
    if std::env::var_os("CARGO_FEATURE_CUDA").is_none() {
        return;
    }

    let nvcc = std::env::var_os("NVCC")
        .map(std::path::PathBuf::from)
        .or_else(|| {
            std::env::var_os("CUDA_HOME")
                .map(std::path::PathBuf::from)
                .map(|root| root.join("bin/nvcc"))
        })
        .or_else(|| {
            let candidate = std::path::PathBuf::from("/usr/local/cuda/bin/nvcc");
            candidate.is_file().then_some(candidate)
        })
        .unwrap_or_else(|| std::path::PathBuf::from("nvcc"));

    // cc-rs reads NVCC when CUDA mode constructs its compiler wrapper. Passing
    // nvcc through `compiler()` would misclassify it as a host compiler and
    // omit the required `-Xcompiler` forwarding for ELF/PIC flags.
    unsafe {
        std::env::set_var("NVCC", nvcc);
    }
    cc::Build::new()
        .cuda(true)
        .cpp(true)
        .include("src/cuda/marlin/native")
        .include("src/cuda/marlin/native/moe/include")
        .file("src/cuda/marlin/native/tesseract_marlin.cu")
        .file("src/cuda/marlin/native/tesseract_marlin_moe.cu")
        .flag("-std=c++17")
        .flag("--expt-relaxed-constexpr")
        .flag("-gencode=arch=compute_80,code=sm_80")
        .compile("tesseract_marlin");
}
