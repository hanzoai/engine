#[cfg(feature = "cuda")]
const CUDA_NVCC_FLAGS: Option<&'static str> = option_env!("CUDA_NVCC_FLAGS");

fn main() {
    set_git_revision();

    #[cfg(feature = "cudnn")]
    add_cudnn_link_search();

    #[cfg(feature = "cuda")]
    {
        use std::path::PathBuf;
        println!("cargo:rerun-if-changed=build.rs");
        println!("cargo:rerun-if-env-changed=CUDA_NVCC_FLAGS");
        let build_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap());

        let mut builder = cudaforge::KernelBuilder::new()
            .source_glob("src/cuda/*.cu")
            .out_dir(&build_dir)
            .arg("-std=c++17")
            .arg("-O3")
            .arg("-U__CUDA_NO_HALF_OPERATORS__")
            .arg("-U__CUDA_NO_HALF_CONVERSIONS__")
            .arg("-U__CUDA_NO_HALF2_OPERATORS__")
            .arg("-U__CUDA_NO_BFLOAT16_CONVERSIONS__")
            .arg("--expt-relaxed-constexpr")
            .arg("--expt-extended-lambda")
            .arg("--use_fast_math")
            .arg("--verbose")
            .arg("--compiler-options")
            .arg("-fPIC");

        // Check if CUDA_COMPUTE_CAP < 80 and disable bf16 kernels if so.
        // bf16 WMMA operations and certain bf16 intrinsics are only available on sm_80+.
        if let Some(compute_cap) = builder.get_compute_cap() {
            if compute_cap < 80 {
                builder = builder.arg("-DNO_BF16_KERNEL");
            }
        }

        // https://github.com/hanzoai/engine/issues/286
        if let Some(cuda_nvcc_flags_env) = CUDA_NVCC_FLAGS {
            builder = builder.arg("--compiler-options");
            builder = builder.arg(cuda_nvcc_flags_env);
        }

        let target = std::env::var("TARGET").unwrap();

        // https://github.com/hanzoai/engine/issues/588
        let out_file = if target.contains("msvc") {
            // Windows case
            build_dir.join("hanzocuda.lib")
        } else {
            build_dir.join("libhanzocuda.a")
        };

        builder
            .build_lib(out_file)
            .expect("Build mistral-core failed!");
        println!("cargo:rustc-link-search={}", build_dir.display());
        println!("cargo:rustc-link-lib=hanzocuda");
        println!("cargo:rustc-link-lib=dylib=cudart");

        if target.contains("msvc") {
            // nothing to link to
        } else if target.contains("apple")
            || target.contains("freebsd")
            || target.contains("openbsd")
        {
            println!("cargo:rustc-link-lib=dylib=c++");
        } else if target.contains("android") {
            println!("cargo:rustc-link-lib=dylib=c++_shared");
        } else {
            println!("cargo:rustc-link-lib=dylib=stdc++");
        }
    }

    #[cfg(feature = "rocm")]
    {
        use std::path::PathBuf;
        use std::process::Command;
        println!("cargo:rerun-if-changed=build.rs");
        println!("cargo:rerun-if-changed=src/rocm/sort.hip.cpp");
        let build_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap());

        let rocm_path = std::env::var("ROCM_PATH").unwrap_or_else(|_| "/opt/rocm".to_string());
        // The Windows ROCm SDK ships hipcc.exe / hipcc.bat (not an extensionless
        // `hipcc`), so probe the real filenames before falling back to PATH.
        let hipcc = {
            let bin = PathBuf::from(&rocm_path).join("bin");
            ["hipcc.exe", "hipcc.bat", "hipcc"]
                .iter()
                .map(|n| bin.join(n))
                .find(|p| p.exists())
                .map(|p| p.to_string_lossy().into_owned())
                .unwrap_or_else(|| "hipcc".to_string())
        };
        let gfx = std::env::var("ROCM_GFX_ARCH").unwrap_or_else(|_| "gfx1151".to_string());

        // Compile the HIP sort/topk kernels into a relocatable object.
        // `-fPIC` is rejected by the windows-msvc clang target ("unsupported
        // option '-fPIC'"); code there is position-independent by default.
        let is_windows = std::env::var("CARGO_CFG_TARGET_OS")
            .map(|s| s == "windows")
            .unwrap_or(cfg!(windows));
        let obj = build_dir.join("sort.hip.o");
        let mut cmd = Command::new(&hipcc);
        cmd.args(["-c", "-std=c++17", "-O3"]);
        if !is_windows {
            cmd.arg("-fPIC");
        }
        let status = cmd
            .arg(format!("--offload-arch={gfx}"))
            .arg("src/rocm/sort.hip.cpp")
            .arg("-o")
            .arg(&obj)
            .status()
            .expect("failed to invoke hipcc for src/rocm/sort.hip.cpp");
        assert!(
            status.success(),
            "hipcc failed to compile src/rocm/sort.hip.cpp"
        );

        // Archive into a static library so the Rust linker pulls in the fatbin.
        // windows-msvc rustc resolves `static=hanzorocm` to `hanzorocm.lib`, while
        // GNU targets want `libhanzorocm.a`. The Windows ROCm SDK ships `llvm-ar`
        // (there is no Unix `ar`); llvm-ar writes a COFF archive link.exe accepts.
        let lib = build_dir.join(if is_windows { "hanzorocm.lib" } else { "libhanzorocm.a" });
        let _ = std::fs::remove_file(&lib);
        let ar = {
            let bin = PathBuf::from(&rocm_path).join("bin");
            let cands: &[&str] = if is_windows {
                &["llvm-ar.exe", "llvm-ar"]
            } else {
                &["ar"]
            };
            cands
                .iter()
                .map(|n| bin.join(n))
                .find(|p| p.exists())
                .map(|p| p.to_string_lossy().into_owned())
                .unwrap_or_else(|| if is_windows { "llvm-ar".into() } else { "ar".into() })
        };
        let status = Command::new(&ar)
            .arg("rcs")
            .arg(&lib)
            .arg(&obj)
            .status()
            .expect("failed to invoke ar/llvm-ar to archive sort.hip.o");
        assert!(status.success(), "ar/llvm-ar failed to archive sort.hip.o");

        println!("cargo:rustc-link-search=native={}", build_dir.display());
        println!("cargo:rustc-link-lib=static=hanzorocm");
        println!("cargo:rustc-link-search=native={rocm_path}/lib");
        println!("cargo:rustc-link-lib=dylib=amdhip64");
        // libstdc++ is the HIP/clang C++ runtime on GNU targets; windows-msvc
        // links the MSVC C++ runtime automatically and ships no stdc++.lib.
        if !is_windows {
            println!("cargo:rustc-link-lib=dylib=stdc++");
        }
    }
}

#[cfg(feature = "cudnn")]
fn add_cudnn_link_search() {
    use std::path::PathBuf;

    println!("cargo:rerun-if-env-changed=CUDNN_LIB_DIR");
    println!("cargo:rerun-if-env-changed=CUDA_PATH");

    let target = std::env::var("TARGET").unwrap_or_default();
    if !target.contains("msvc") {
        return;
    }

    if let Ok(dir) = std::env::var("CUDNN_LIB_DIR") {
        println!("cargo:rustc-link-search=native={dir}");
        return;
    }

    let mut candidates: Vec<PathBuf> = Vec::new();
    if let Ok(cuda_path) = std::env::var("CUDA_PATH") {
        candidates.push(PathBuf::from(&cuda_path).join("lib").join("x64"));
    }
    let cudnn_root = PathBuf::from(r"C:\Program Files\NVIDIA\CUDNN");
    if let Ok(versions) = std::fs::read_dir(&cudnn_root) {
        for version in versions.flatten() {
            let lib = version.path().join("lib");
            candidates.push(lib.join("x64"));
            if let Ok(cuda_vers) = std::fs::read_dir(&lib) {
                for cuda_ver in cuda_vers.flatten() {
                    candidates.push(cuda_ver.path().join("x64"));
                }
            }
        }
    }

    for dir in candidates {
        if dir.join("cudnn.lib").is_file() {
            println!("cargo:rustc-link-search=native={}", dir.display());
            return;
        }
    }

    println!(
        "cargo:warning=cudnn feature enabled but cudnn.lib not found; set CUDNN_LIB_DIR to its directory"
    );
}

fn set_git_revision() {
    let commit = std::process::Command::new("git")
        .args(["rev-parse", "HEAD"])
        .output()
        .ok()
        .and_then(|output| {
            if output.status.success() {
                String::from_utf8(output.stdout).ok()
            } else {
                None
            }
        })
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "unknown".to_string());

    println!("cargo:rustc-env=GIT_REVISION={commit}");
    println!("cargo:rerun-if-changed=.git/HEAD");
    if let Ok(head) = std::fs::read_to_string(".git/HEAD") {
        if let Some(ref_path) = head.strip_prefix("ref:") {
            let ref_path = ref_path.trim();
            if !ref_path.is_empty() {
                println!("cargo:rerun-if-changed=.git/{}", ref_path);
            }
        }
    }
}
