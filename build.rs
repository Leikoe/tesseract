fn main() {
    println!("cargo:rerun-if-changed=src/cuda/marlin/native");
    if std::env::var_os("CARGO_FEATURE_CUDA").is_none() {
        return;
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
