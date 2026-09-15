use half::{bf16, f16};

pub(crate) const HAVE_NVFP4_GEMM_KERNELS: bool = cfg!(has_nvfp4_kernels);

extern "C" {
    pub(crate) fn launch_nvfp4_matmul_f16(
        input: *const f16,
        weight: *const u8,
        weight_scale: *const u8,
        global_scale: f32,
        bias: *const f16,
        output: *mut f16,
        m: i32,
        n: i32,
        k: i32,
        has_bias: bool,
        stream: hanzo_ml::cuda::cudarc::driver::sys::CUstream,
    );

    pub(crate) fn launch_nvfp4_matmul_bf16(
        input: *const bf16,
        weight: *const u8,
        weight_scale: *const u8,
        global_scale: f32,
        bias: *const bf16,
        output: *mut bf16,
        m: i32,
        n: i32,
        k: i32,
        has_bias: bool,
        stream: hanzo_ml::cuda::cudarc::driver::sys::CUstream,
    );
}
