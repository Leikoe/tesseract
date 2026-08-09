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
            [
                "/usr/local/cuda/bin/nvcc",
                "/usr/local/cuda-13.3/bin/nvcc",
                "/usr/local/cuda-13.2/bin/nvcc",
                "/usr/local/cuda-13.1/bin/nvcc",
                "/usr/local/cuda-13.0/bin/nvcc",
                "/usr/local/cuda-12.9/bin/nvcc",
            ]
            .into_iter()
            .map(std::path::PathBuf::from)
            .find(|candidate| candidate.is_file())
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
        .file("src/cuda/marlin/native/tesseract_marlin.cu")
        .flag("-std=c++17")
        .flag("--expt-relaxed-constexpr")
        .flag("-gencode=arch=compute_80,code=sm_80")
        .compile("tesseract_marlin");
}
