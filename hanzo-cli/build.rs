// Build script for hanzo-cli.
//
// When built with `--features rocm`, supply the ROCm/HIP import libraries for the
// final `hanzo` binary link. The `rocm-rs` crate is built with SKIP_BINDGEN=1 to
// avoid a libclang dependency (the Windows ROCm SDK ships only the clang driver,
// not libclang.dll); in that mode rocm-rs uses its committed bindings but does NOT
// emit these link directives itself. Emitting them here scopes the libraries to
// the binary link — putting them in global RUSTFLAGS would wrongly add an
// `amdhip64` import to every proc-macro dylib.
//
// Only the modules the active feature set compiles are linked. MIOpen and
// rocSOLVER are excluded: MIOpen is not in the Windows ROCm SDK (see the
// `rocm-miopen` feature in hanzo-ml) and rocSOLVER is feature-gated off in rocm-rs.
fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=CARGO_FEATURE_ROCM");
    println!("cargo:rerun-if-env-changed=ROCM_PATH");

    if std::env::var_os("CARGO_FEATURE_ROCM").is_some() {
        let rocm_path = std::env::var("ROCM_PATH")
            .unwrap_or_else(|_| "C:/Program Files/AMD/ROCm/7.1".to_string());
        println!("cargo:rustc-link-search=native={rocm_path}/lib");
        for lib in ["amdhip64", "rocblas", "rocfft", "rocrand", "rocsparse"] {
            println!("cargo:rustc-link-lib=dylib={lib}");
        }
    }
}
