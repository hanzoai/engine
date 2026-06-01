#[cfg(feature = "cuda")]
const CUDA_NVCC_FLAGS: Option<&'static str> = option_env!("CUDA_NVCC_FLAGS");

fn main() {
    set_git_revision();

    #[cfg(feature = "cuda")]
    {
        use std::path::PathBuf;
        println!("cargo:rerun-if-changed=build.rs");
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
        let hipcc = {
            let p = PathBuf::from(&rocm_path).join("bin/hipcc");
            if p.exists() {
                p.to_string_lossy().into_owned()
            } else {
                "hipcc".to_string()
            }
        };
        let gfx = std::env::var("ROCM_GFX_ARCH").unwrap_or_else(|_| "gfx1151".to_string());

        // Compile the HIP sort/topk kernels into a relocatable object.
        let obj = build_dir.join("sort.hip.o");
        let status = Command::new(&hipcc)
            .args(["-c", "-std=c++17", "-O3", "-fPIC"])
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
        let lib = build_dir.join("libhanzorocm.a");
        let _ = std::fs::remove_file(&lib);
        let status = Command::new("ar")
            .arg("rcs")
            .arg(&lib)
            .arg(&obj)
            .status()
            .expect("failed to invoke ar to archive sort.hip.o");
        assert!(status.success(), "ar failed to archive sort.hip.o");

        println!("cargo:rustc-link-search=native={}", build_dir.display());
        println!("cargo:rustc-link-lib=static=hanzorocm");
        println!("cargo:rustc-link-search=native={rocm_path}/lib");
        println!("cargo:rustc-link-lib=dylib=amdhip64");
        println!("cargo:rustc-link-lib=dylib=stdc++");
    }
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

    println!("cargo:rustc-env=HANZO_GIT_REVISION={commit}");
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
