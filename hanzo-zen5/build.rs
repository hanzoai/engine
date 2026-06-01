//! Build script for hanzo-zen5.
//!
//! With `--features=ffi` (default) this script compiles the vendored zen5-engine
//! C sources and emits Rust bindings for `ds4.h`. With only `--features=native`
//! the script is a no-op so the crate `cargo check`s standalone without a C
//! toolchain.
//!
//! Vendoring layout:
//!   hanzo-zen5/zen5-engine-src/        (git submodule of zenlm/zen5-engine)
//!     ds4.h ds4.c ds4_cli.c ds4_server.c
//!     ds4_metal.m  ds4_cuda.cu  ds4_gpu.h
//!     rax.c linenoise.c ...
//!
//! When zen5-engine-src/ds4.h is missing, we print a single cargo:warning and
//! exit successfully so downstream crates that don't use the FFI backend can
//! still build. The FFI symbols become link errors only if you actually call
//! into `ffi::sys::*`.

use std::env;
use std::path::PathBuf;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=zen5-engine-src/ds4.h");
    // Register custom cfg flags so rustc 1.80+ doesn't warn.
    println!("cargo:rustc-check-cfg=cfg(zen5_bindings)");
    println!("cargo:rustc-check-cfg=cfg(ds4_backend, values(\"metal\", \"cuda\", \"cpu\"))");

    let ffi_enabled = env::var("CARGO_FEATURE_FFI").is_ok();
    if !ffi_enabled {
        return;
    }

    let vendor = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap()).join("zen5-engine-src");
    let header = vendor.join("ds4.h");
    if !header.exists() {
        println!(
            "cargo:warning=hanzo-zen5: zen5-engine-src/ds4.h not found at {}. \
             Run: (cd $(git rev-parse --show-toplevel) && \
             git submodule add https://github.com/zenlm/zen5-engine \
             hanzo-zen5/zen5-engine-src). FFI symbols will not link.",
            header.display()
        );
        return;
    }

    // -- 1. Decide backend.
    let metal = cfg!(target_os = "macos") || env::var("CARGO_FEATURE_METAL").is_ok();
    let cuda = env::var("CARGO_FEATURE_CUDA").is_ok();

    // -- 2. Compile C/ObjC/CUDA sources.
    let mut build = cc::Build::new();
    build
        .include(&vendor)
        .file(vendor.join("ds4.c"))
        .file(vendor.join("rax.c"))
        .file(vendor.join("linenoise.c"))
        .flag_if_supported("-std=c11")
        .flag_if_supported("-O3")
        .flag_if_supported("-Wno-unused-function")
        .flag_if_supported("-Wno-unused-parameter");

    if metal {
        build.file(vendor.join("ds4_metal.m"));
        println!("cargo:rustc-link-lib=framework=Metal");
        println!("cargo:rustc-link-lib=framework=MetalPerformanceShaders");
        println!("cargo:rustc-link-lib=framework=Foundation");
        println!("cargo:rustc-link-lib=framework=Accelerate");
        println!("cargo:rustc-cfg=ds4_backend=\"metal\"");
    } else if cuda {
        // CUDA build is a separate `cc::Build` because cc-rs treats .cu as C.
        let mut cuda_build = cc::Build::new();
        cuda_build
            .cuda(true)
            .include(&vendor)
            .file(vendor.join("ds4_cuda.cu"))
            .flag("-O3");
        cuda_build.compile("ds4_cuda");
        println!("cargo:rustc-link-lib=cudart");
        println!("cargo:rustc-link-lib=cublas");
        println!("cargo:rustc-cfg=ds4_backend=\"cuda\"");
    } else {
        println!("cargo:rustc-cfg=ds4_backend=\"cpu\"");
    }

    build.compile("ds4");

    // -- 3. Generate Rust bindings for ds4.h.
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let bindings = bindgen::Builder::default()
        .header(header.to_string_lossy())
        .allowlist_function("ds4_.*")
        .allowlist_type("ds4_.*")
        .allowlist_var("DS4_.*")
        .blocklist_type("FILE")
        // NOTE: bindgen used to emit `#![allow(...)]` (inner attribute) but the
        // generated file is `include!`d inside `mod gen { ... }` in sys.rs and
        // inner attrs in include!d files trip "an inner attribute is not
        // permitted in this context". We apply the same allows as outer attrs
        // on the wrapping `mod gen` instead.
        .raw_line("use libc::FILE;")
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .generate()
        .expect("generate ds4.h bindings");
    bindings
        .write_to_file(out_dir.join("bindings.rs"))
        .expect("write bindings");

    // Signal to sys.rs that real bindings are available.
    println!("cargo:rustc-cfg=zen5_bindings");
}
