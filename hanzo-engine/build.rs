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
        let target = std::env::var("TARGET").unwrap_or_default();
        let is_msvc = target.contains("msvc");

        let rocm_path = std::env::var("ROCM_PATH").unwrap_or_else(|_| "/opt/rocm".to_string());
        let bin = PathBuf::from(&rocm_path).join("bin");
        let tool = |name: &str| {
            let exe = bin.join(format!("{name}.exe"));
            if is_msvc && exe.exists() {
                exe.to_string_lossy().into_owned()
            } else {
                let plain = bin.join(name);
                if plain.exists() {
                    plain.to_string_lossy().into_owned()
                } else {
                    name.to_string()
                }
            }
        };
        let hipcc = tool("hipcc");
        let gfx = std::env::var("ROCM_GFX_ARCH").unwrap_or_else(|_| "gfx1151".to_string());

        // hipcc on Windows is a cmd.exe wrapper that can't cd into a UNC working dir (\\wsl.localhost\..),
        // so copy the source into OUT_DIR (a real C: path) and compile with absolute in/out paths.
        let src = build_dir.join("sort.hip.cpp");
        std::fs::copy("src/rocm/sort.hip.cpp", &src)
            .expect("failed to stage src/rocm/sort.hip.cpp");

        // Compile the HIP sort/topk kernels into a relocatable object. MSVC wants COFF (no -fPIC).
        let obj = build_dir.join(if is_msvc {
            "sort.hip.obj"
        } else {
            "sort.hip.o"
        });
        let mut cmd = Command::new(&hipcc);
        cmd.args(["-c", "-std=c++17", "-O3"]);
        if !is_msvc {
            cmd.arg("-fPIC");
        }
        let status = cmd
            .arg(format!("--offload-arch={gfx}"))
            .arg(&src)
            .arg("-o")
            .arg(&obj)
            .status()
            .expect("failed to invoke hipcc for src/rocm/sort.hip.cpp");
        assert!(
            status.success(),
            "hipcc failed to compile src/rocm/sort.hip.cpp"
        );

        // Archive into a static library so the Rust linker pulls in the fatbin.
        // ROCm ships llvm-ar on both platforms; MSVC linker reads the COFF archive as a .lib.
        let lib = build_dir.join(if is_msvc {
            "hanzorocm.lib"
        } else {
            "libhanzorocm.a"
        });
        let _ = std::fs::remove_file(&lib);
        let archiver = tool("llvm-ar");
        let status = Command::new(&archiver)
            .arg("rcs")
            .arg(&lib)
            .arg(&obj)
            .status()
            .unwrap_or_else(|_| panic!("failed to invoke {archiver} to archive {obj:?}"));
        assert!(
            status.success(),
            "llvm-ar failed to archive sort.hip object"
        );

        println!("cargo:rustc-link-search=native={}", build_dir.display());
        println!("cargo:rustc-link-lib=static=hanzorocm");
        println!("cargo:rustc-link-search=native={rocm_path}/lib");
        println!("cargo:rustc-link-lib=dylib=amdhip64");
        if !is_msvc {
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
