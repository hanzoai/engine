use hanzo_ml::{backend::BackendStorage, DType, Result, Storage, Tensor};

use crate::metal::kernels::{self, PagedAttentionDType};

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
        hanzo_ml::bail!("kvscale-update is not implemented on CPU!")
    }

    fn metal_fwd(
        &self,
        k: &mut hanzo_ml::MetalStorage,
        k_layout: &hanzo_ml::Layout,
        v: &hanzo_ml::MetalStorage,
        _: &hanzo_ml::Layout,
    ) -> Result<()> {
        let ty = match k.dtype() {
            DType::F16 => PagedAttentionDType::F16,
            DType::BF16 => PagedAttentionDType::BF16,
            DType::F32 => PagedAttentionDType::F32,
            dtype => hanzo_ml::bail!("dtype {dtype:?} is not supported for kv_scale_update"),
        };

        let dev = k.device();
        let elem_count = k_layout.shape().elem_count();

        let (k_scales_storage, _) = self.k_scales.storage_and_layout();
        let k_scales = match &*k_scales_storage {
            Storage::Metal(m) => m,
            _ => hanzo_ml::bail!("k_scales must be a metal tensor"),
        };

        let (v_scales_storage, _) = self.v_scales.storage_and_layout();
        let v_scales = match &*v_scales_storage {
            Storage::Metal(m) => m,
            _ => hanzo_ml::bail!("v_scales must be a metal tensor"),
        };

        let encoder = dev.command_encoder()?;
        encoder.set_label("kv-scale-update");

        kernels::call_kv_scale_update(
            dev.device(),
            &encoder,
            &kernels::Kernels::new(),
            ty,
            k.buffer(),
            k_layout.start_offset() * k.dtype().size_in_bytes(),
            v.buffer(),
            0, // v_layout already incorporated by caller
            k_scales.buffer(),
            v_scales.buffer(),
            elem_count as i64,
        )
        .map_err(hanzo_ml::Error::wrap)?;

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
