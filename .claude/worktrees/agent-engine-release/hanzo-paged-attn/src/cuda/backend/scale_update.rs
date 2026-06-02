use crate::cuda::backend::slice_ptr;
#[cfg(feature = "cuda")]
use crate::cuda::ffi;
use hanzo_ml::{DType, Result, Tensor};

#[derive(Debug, Clone)]
struct KvScaleUpdate {
    k_scales: Tensor,
    v_scales: Tensor,
}

impl hanzo_ml::InplaceOp2 for KvScaleUpdate {
    fn name(&self) -> &'static str {
        "kvscale-update"
    }

    fn cpu_fwd(
        &self,
        _: &mut hanzo_ml::CpuStorage,
        _: &hanzo_ml::Layout,
        _: &hanzo_ml::CpuStorage,
        _: &hanzo_ml::Layout,
    ) -> Result<()> {
        panic!("kvscale-update is not implemented on CPU!")
    }

    #[cfg(feature = "cuda")]
    fn cuda_fwd(
        &self,
        k: &mut hanzo_ml::CudaStorage,
        k_layout: &hanzo_ml::Layout,
        v: &hanzo_ml::CudaStorage,
        _: &hanzo_ml::Layout,
    ) -> Result<()> {
        use hanzo_ml::backend::BackendStorage;
        use hanzo_ml::cuda_backend::cudarc::driver::DevicePtr;
        use hanzo_ml::cuda_backend::CudaStorageSlice;
        let dev = k.device();
        let elem_count = k_layout.shape().elem_count();

        use std::ffi::c_void;

        let (src_ptr, dst_ptr) = match (&k.slice, &v.slice) {
            (CudaStorageSlice::BF16(inp_k), CudaStorageSlice::BF16(inp_v)) => (
                inp_k.device_ptr(&dev.cuda_stream()).0 as *const c_void,
                inp_v.device_ptr(&dev.cuda_stream()).0 as *const c_void,
            ),
            (CudaStorageSlice::F16(inp_k), CudaStorageSlice::F16(inp_v)) => (
                inp_k.device_ptr(&dev.cuda_stream()).0 as *const c_void,
                inp_v.device_ptr(&dev.cuda_stream()).0 as *const c_void,
            ),
            (CudaStorageSlice::F32(inp_k), CudaStorageSlice::F32(inp_v)) => (
                inp_k.device_ptr(&dev.cuda_stream()).0 as *const c_void,
                inp_v.device_ptr(&dev.cuda_stream()).0 as *const c_void,
            ),
            _ => {
                panic!("Invalid dtype for kv scale update!")
            }
        };
        let stream = dev.cuda_stream().cu_stream() as i64;

        let (k_scales, k_scales_layout) = self.k_scales.storage_and_layout();
        let k_scales = match &*k_scales {
            hanzo_ml::Storage::Cuda(c) => c.as_cuda_slice::<f32>()?,
            _ => hanzo_ml::bail!("k_scales must be a cuda tensor"),
        };
        let (k_scales, _) = slice_ptr(k_scales, k_scales_layout.start_offset());

        let (v_scales, v_scales_layout) = self.v_scales.storage_and_layout();
        let v_scales = match &*v_scales {
            hanzo_ml::Storage::Cuda(c) => c.as_cuda_slice::<f32>()?,
            _ => hanzo_ml::bail!("v_scales must be a cuda tensor"),
        };
        let (v_scales, _) = slice_ptr(v_scales, v_scales_layout.start_offset());

        let k_scales_ptr = k_scales as *mut f32;
        let v_scales_ptr = v_scales as *mut f32;
        unsafe {
            match k.dtype() {
                DType::F32 => ffi::update_kv_scales_f32(
                    src_ptr,
                    dst_ptr,
                    elem_count as i64,
                    k_scales_ptr,
                    v_scales_ptr,
                    stream,
                ),
                DType::F16 => ffi::update_kv_scales_f16(
                    src_ptr,
                    dst_ptr,
                    elem_count as i64,
                    k_scales_ptr,
                    v_scales_ptr,
                    stream,
                ),
                DType::BF16 => ffi::update_kv_scales_bf16(
                    src_ptr,
                    dst_ptr,
                    elem_count as i64,
                    k_scales_ptr,
                    v_scales_ptr,
                    stream,
                ),
                _ => {
                    panic!("Invalid dtype for kv scale update!")
                }
            }
        }
        Ok(())
    }
}

pub fn kv_scale_update(
    key: &Tensor,
    value: &Tensor,
    k_scales: &Tensor,
    v_scales: &Tensor,
) -> Result<()> {
    let op = KvScaleUpdate {
        k_scales: k_scales.to_owned(),
        v_scales: v_scales.to_owned(),
    };
    key.inplace_op2(value, &op)
}
