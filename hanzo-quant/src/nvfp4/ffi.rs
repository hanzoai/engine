use half::{bf16, f16};

pub(crate) const HAVE_NVFP4_GEMM_KERNELS: bool = cfg!(has_nvfp4_kernels);
pub(crate) const HAVE_NVFP4_WMMA_KERNELS: bool = cfg!(has_nvfp4_wmma_kernels);

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

    pub(crate) fn launch_nvfp4_matmul_wmma_f16(
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

    pub(crate) fn launch_nvfp4_matmul_wmma_bf16(
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

    // W4A4 in the scaled domain (kernels/nvfp4/nvfp4_moe.cu).
    pub(crate) fn launch_nvfp4_moe_vecmat_f16(
        a_codes: *const u8,
        a_scales: *const u8,
        w_codes: *const u8,
        w_scales: *const u8,
        alpha: *const f32,
        indices: *const u32,
        out: *mut f16,
        work_items: i32,
        topk: i32,
        experts: i32,
        n: i32,
        k: i32,
        input_has_topk_dim: bool,
        stream: hanzo_ml::cuda::cudarc::driver::sys::CUstream,
    );
    pub(crate) fn launch_nvfp4_moe_vecmat_bf16(
        a_codes: *const u8,
        a_scales: *const u8,
        w_codes: *const u8,
        w_scales: *const u8,
        alpha: *const f32,
        indices: *const u32,
        out: *mut bf16,
        work_items: i32,
        topk: i32,
        experts: i32,
        n: i32,
        k: i32,
        input_has_topk_dim: bool,
        stream: hanzo_ml::cuda::cudarc::driver::sys::CUstream,
    );
    pub(crate) fn launch_nvfp4_moe_vecmat_f32(
        a_codes: *const u8,
        a_scales: *const u8,
        w_codes: *const u8,
        w_scales: *const u8,
        alpha: *const f32,
        indices: *const u32,
        out: *mut f32,
        work_items: i32,
        topk: i32,
        experts: i32,
        n: i32,
        k: i32,
        input_has_topk_dim: bool,
        stream: hanzo_ml::cuda::cudarc::driver::sys::CUstream,
    );
    pub(crate) fn launch_nvfp4_moe_grouped_f16(
        a_codes: *const u8,
        a_scales: *const u8,
        w_codes: *const u8,
        w_scales: *const u8,
        alpha: *const f32,
        bounds: *const u32,
        sorted_work: *const u32,
        out: *mut f16,
        experts: i32,
        topk: i32,
        n: i32,
        k: i32,
        input_has_topk_dim: bool,
        stream: hanzo_ml::cuda::cudarc::driver::sys::CUstream,
    );
    pub(crate) fn launch_nvfp4_moe_grouped_bf16(
        a_codes: *const u8,
        a_scales: *const u8,
        w_codes: *const u8,
        w_scales: *const u8,
        alpha: *const f32,
        bounds: *const u32,
        sorted_work: *const u32,
        out: *mut bf16,
        experts: i32,
        topk: i32,
        n: i32,
        k: i32,
        input_has_topk_dim: bool,
        stream: hanzo_ml::cuda::cudarc::driver::sys::CUstream,
    );
    pub(crate) fn launch_nvfp4_moe_grouped_f32(
        a_codes: *const u8,
        a_scales: *const u8,
        w_codes: *const u8,
        w_scales: *const u8,
        alpha: *const f32,
        bounds: *const u32,
        sorted_work: *const u32,
        out: *mut f32,
        experts: i32,
        topk: i32,
        n: i32,
        k: i32,
        input_has_topk_dim: bool,
        stream: hanzo_ml::cuda::cudarc::driver::sys::CUstream,
    );
}
