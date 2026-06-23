// ROCm/HIP backend for PagedAttention v1 decode + reshape_and_cache.
//
// This mirrors src/cuda/backend/paged_attention.rs, retargeted onto hanzo-ml's
// RocmStorage / RocmDevice. The kernels live in ../../rocm/*.hip.cpp and are
// reached through the extern "C" bindings in ../ffi.rs.
//
// KEYSTONE: `context_lens` is passed as a DEVICE pointer; the kernel reads the
// per-sequence length from device memory. The launch parameters depend only on
// the static shapes (num_seqs, num_heads, head_size), never on a host-side
// seqlen, which is what enables decode hipGraph capture later.

use crate::rocm::ffi;
use half::{bf16, f16};
use hanzo_ml::backend::BackendStorage;
use hanzo_ml::rocm_backend::{RocmStorage, RocmStorageSlice};
use hanzo_ml::{CpuStorage, DType, Layout, Result, Shape, Storage, Tensor};
use std::ffi::c_int;

/// Element pointer (byte address) into a ROCm storage slice at `offset`
/// elements. `RocmStorageSlice::offset_ptr` is private, so match the variant and
/// use the public `SendSyncDeviceMemory::offset_ptr` on the inner buffer (the
/// same idiom the in-tree rocm ops use).
fn slice_ptr_at(slice: &RocmStorageSlice, offset: usize) -> *const std::ffi::c_void {
    use RocmStorageSlice::*;
    unsafe {
        match slice {
            U8(m) | F8E4M3(m) => m.offset_ptr(offset),
            U32(m) => m.offset_ptr(offset),
            I16(m) => m.offset_ptr(offset),
            I32(m) => m.offset_ptr(offset),
            I64(m) => m.offset_ptr(offset),
            BF16(m) => m.offset_ptr(offset),
            F16(m) => m.offset_ptr(offset),
            F32(m) => m.offset_ptr(offset),
            F64(m) => m.offset_ptr(offset),
        }
    }
    .cast_const()
}

struct PagedAttention {
    softmax_scale: f32,
    softcapping: f32,

    key_cache: Tensor,
    value_cache: Tensor,
    block_tables: Tensor,
    context_lens: Tensor,
    alibi_slopes: Option<Tensor>,
    max_context_len: usize,
    sinks: Option<Tensor>,
}

impl PagedAttention {
    fn rocm_fwd_inner(&self, q: &RocmStorage, q_l: &Layout) -> Result<(RocmStorage, Shape)> {
        let dtype = q.dtype();
        let cache_dtype = match self.key_cache.dtype() {
            DType::F16 => 0u32,
            DType::BF16 => 1,
            DType::F32 => 2,
            dtype => hanzo_ml::bail!("cache dtype {dtype:?} is not supported on rocm"),
        };
        if cache_dtype == 3 {
            hanzo_ml::bail!("FP8 KV cache is not supported on the rocm backend (stage 1)");
        }

        let dev = q.device().clone();
        let out_shape = q_l.shape().clone();

        let (kc, kc_l) = self.key_cache.storage_and_layout();
        let kc = match &*kc {
            Storage::Rocm(kc) => kc,
            _ => hanzo_ml::bail!("key_cache must be a rocm tensor"),
        };

        let (vc, vc_l) = self.value_cache.storage_and_layout();
        let vc = match &*vc {
            Storage::Rocm(vc) => vc,
            _ => hanzo_ml::bail!("value_cache must be a rocm tensor"),
        };

        let (bt, bt_l) = self.block_tables.storage_and_layout();
        let bt = match &*bt {
            Storage::Rocm(bt) => bt,
            _ => hanzo_ml::bail!("block_tables must be a rocm tensor"),
        };

        let (cl, cl_l) = self.context_lens.storage_and_layout();
        let cl = match &*cl {
            Storage::Rocm(cl) => cl,
            _ => hanzo_ml::bail!("context_lens must be a rocm tensor"),
        };

        let q_rank = q_l.stride().len();
        let kc_rank = kc_l.stride().len();
        let vc_rank = vc_l.stride().len();
        if q_rank != 3 {
            hanzo_ml::bail!("paged-attention expects `q` of rank 3 (q: {q_l:?})")
        }
        if kc_rank != 5 {
            hanzo_ml::bail!("paged-attention expects `key_cache` of rank 5 (key_cache: {kc_l:?})")
        }
        if vc_rank != 4 {
            hanzo_ml::bail!(
                "paged-attention expects `value_cache` of rank 4 (value_cache: {vc_l:?})"
            )
        }

        // Pointers into the paged cache + tables (DEVICE memory).
        let kc_ptr = slice_ptr_at(&kc.slice, kc_l.start_offset());
        let vc_ptr = slice_ptr_at(&vc.slice, vc_l.start_offset());
        // block_tables / context_lens may be stored as i32 or u32 — the kernel
        // only reads the 32-bit pattern. Pass the raw pointer either way.
        let bt_ptr = slice_ptr_at(&bt.slice, bt_l.start_offset()) as *const c_int;
        let cl_ptr = slice_ptr_at(&cl.slice, cl_l.start_offset()) as *const c_int;

        let alibi_s_ptr = if let Some(alibi_slopes) = self.alibi_slopes.as_ref() {
            let (alibi_s, alibi_s_l) = alibi_slopes.storage_and_layout();
            let alibi_s = match &*alibi_s {
                Storage::Rocm(a) => a,
                _ => hanzo_ml::bail!("alibi_slopes must be a rocm tensor"),
            };
            slice_ptr_at(&alibi_s.slice, alibi_s_l.start_offset())
        } else {
            std::ptr::null()
        };

        let sinks_ptr = if let Some(sinks) = self.sinks.as_ref() {
            let (s, s_l) = sinks.storage_and_layout();
            let s = match &*s {
                Storage::Rocm(s) => s,
                _ => hanzo_ml::bail!("sinks must be a rocm tensor"),
            };
            slice_ptr_at(&s.slice, s_l.start_offset()) as *const f32
        } else {
            std::ptr::null()
        };

        let (num_seqs, num_heads, head_size) = q_l.shape().dims3()?;
        if !(head_size == 64
            || head_size == 80
            || head_size == 96
            || head_size == 112
            || head_size == 128
            || head_size == 192
            || head_size == 256
            || head_size == 512)
        {
            hanzo_ml::bail!("`head_size` must be one of 64, 80, 96, 112, 128, 192, 256 or 512");
        }

        let (num_seqs_bt, max_num_blocks_per_seq) = bt_l.shape().dims2()?;
        if num_seqs_bt != num_seqs {
            hanzo_ml::bail!(
                "shape mismatch block_tables {:?}, expected ({num_seqs}, {max_num_blocks_per_seq})",
                bt_l.shape()
            )
        }

        let (num_blocks, num_kv_heads, head_size_kc, block_size, x) = kc_l.shape().dims5()?;
        if head_size_kc != head_size / x {
            hanzo_ml::bail!("shape mismatch key_cache {:?}", kc_l.shape())
        }
        if (num_blocks, num_kv_heads, head_size, block_size) != vc_l.shape().dims4()? {
            hanzo_ml::bail!(
                "shape mismatch key_cache {:?} and value_cache {:?}",
                kc_l.shape(),
                vc_l.shape()
            )
        }
        if num_seqs != cl_l.shape().dims1()? {
            hanzo_ml::bail!(
                "shape mismatch context_lens {:?}, expected ({num_seqs})",
                cl_l.shape()
            )
        }

        let q_stride = q_l.stride()[0];
        let kv_block_stride = kc_l.stride()[0];
        let kv_head_stride = kc_l.stride()[1];
        let effective_max_context_len =
            (max_num_blocks_per_seq * block_size).min(self.max_context_len);

        // Allocate output (device), then launch.
        let elem_count = out_shape.elem_count();
        let stream_raw = dev.stream().as_raw() as i64;

        macro_rules! launch {
            ($variant:ident, $ty:ty, $func:ident) => {{
                let q_mem = match &q.slice {
                    RocmStorageSlice::$variant(m) => m,
                    _ => unreachable!(),
                };
                let q_ptr =
                    unsafe { q_mem.offset_ptr(q_l.start_offset()) as *const std::ffi::c_void };
                let out = dev.alloc::<$ty>(elem_count)?;
                let out_ptr = out.as_ptr() as *const std::ffi::c_void;
                unsafe {
                    ffi::$func(
                        out_ptr,
                        q_ptr,
                        kc_ptr,
                        vc_ptr,
                        alibi_s_ptr,
                        num_kv_heads as c_int,
                        self.softmax_scale,
                        self.softcapping,
                        bt_ptr,
                        cl_ptr,
                        block_size as c_int,
                        effective_max_context_len as c_int,
                        num_seqs as c_int,
                        num_heads as c_int,
                        head_size as c_int,
                        max_num_blocks_per_seq as c_int,
                        q_stride as c_int,
                        kv_block_stride as c_int,
                        kv_head_stride as c_int,
                        stream_raw,
                        cache_dtype,
                        std::ptr::null(),
                        std::ptr::null(),
                        sinks_ptr,
                    );
                }
                // No per-call dev.synchronize() here: the kernel is enqueued on
                // `stream_raw` and all downstream consumers run on the same
                // stream, so stream ordering already guarantees correctness.
                // Synchronizing per call would break decode hipGraph capture
                // (stage 3); the CUDA backend likewise does not synchronize.
                RocmStorage {
                    slice: RocmStorageSlice::$variant(out),
                    device: dev.clone(),
                }
            }};
        }

        let out = match dtype {
            DType::F16 => launch!(F16, f16, paged_attention_v1_f16),
            DType::BF16 => launch!(BF16, bf16, paged_attention_v1_bf16),
            dt => hanzo_ml::bail!("paged-attention on rocm supports f16/bf16 only ({dt:?})"),
        };

        Ok((out, out_shape))
    }
}

impl hanzo_ml::CustomOp1 for PagedAttention {
    fn name(&self) -> &'static str {
        "paged-attention"
    }

    fn cpu_fwd(&self, _: &CpuStorage, _: &Layout) -> Result<(CpuStorage, Shape)> {
        hanzo_ml::bail!("no cpu support for paged-attention")
    }

    fn rocm_fwd(&self, q: &RocmStorage, q_l: &Layout) -> Result<(RocmStorage, Shape)> {
        self.rocm_fwd_inner(q, q_l)
    }
}

/// PagedAttention v1 decode: `softmax(Q @ K^T . softmax_scale) @ V` over the
/// paged KV cache. See src/cuda/backend/paged_attention.rs for the full doc on
/// tensor shapes; this is the ROCm mirror.
#[allow(clippy::too_many_arguments)]
pub fn paged_attention(
    q: &Tensor,
    k_scale: Option<&Tensor>,
    v_scale: Option<&Tensor>,
    key_cache: &Tensor,
    value_cache: &Tensor,
    block_tables: &Tensor,
    context_lens: &Tensor,
    alibi_slopes: Option<&Tensor>,
    max_context_len: usize,
    softmax_scale: f32,
    softcapping: f32,
    sinks: Option<&Tensor>,
) -> Result<Tensor> {
    // `k_scale`/`v_scale` are intentionally ignored on ROCm. The PagedAttention
    // layer always supplies a default identity (1.0) scale, even for a non-FP8
    // (f16/bf16/f32) cache. Real (non-identity) KV scales only ever accompany an
    // FP8 cache, and an FP8 cache is rejected up front by the `cache_dtype`
    // check in `rocm_fwd_inner` — so any cache that reaches the kernel is
    // non-FP8 and its scale is identity. The v1 decode kernel does not read
    // scales in the non-FP8 path, making this a safe no-op (and avoiding a
    // per-call host read of the scale, which would block decode graph capture).
    let _ = (k_scale, v_scale);
    let op = PagedAttention {
        softmax_scale,
        key_cache: key_cache.clone(),
        value_cache: value_cache.clone(),
        block_tables: block_tables.clone(),
        context_lens: context_lens.clone(),
        max_context_len,
        softcapping,
        alibi_slopes: alibi_slopes.cloned(),
        sinks: sinks.map(|s| s.to_dtype(DType::F32)).transpose()?,
    };
    q.apply_op1(op)
}

fn update_cache(
    key: &Tensor,
    value: &Tensor,
    key_cache: &Tensor,
    value_cache: &Tensor,
    slot_mapping: &Tensor,
) -> Result<()> {
    let dtype = key.dtype();
    let internal_type = match dtype {
        DType::F16 => 0u32,
        DType::BF16 => 1,
        DType::F32 => 2,
        dtype => hanzo_ml::bail!("dtype {dtype:?} is not supported on rocm"),
    };
    let cache_dtype = match key_cache.dtype() {
        DType::F16 => 0u32,
        DType::BF16 => 1,
        DType::F32 => 2,
        dtype => hanzo_ml::bail!("cache dtype {dtype:?} is not supported on rocm"),
    };

    let (k, k_l) = key.storage_and_layout();
    let k = match &*k {
        Storage::Rocm(k) => k,
        _ => hanzo_ml::bail!("key must be a rocm tensor"),
    };
    let (v, v_l) = value.storage_and_layout();
    let v = match &*v {
        Storage::Rocm(v) => v,
        _ => hanzo_ml::bail!("value must be a rocm tensor"),
    };
    let (kc, kc_l) = key_cache.storage_and_layout();
    let kc = match &*kc {
        Storage::Rocm(kc) => kc,
        _ => hanzo_ml::bail!("key_cache must be a rocm tensor"),
    };
    let (vc, vc_l) = value_cache.storage_and_layout();
    let vc = match &*vc {
        Storage::Rocm(vc) => vc,
        _ => hanzo_ml::bail!("value_cache must be a rocm tensor"),
    };
    let (s, s_l) = slot_mapping.storage_and_layout();
    let s = match &*s {
        Storage::Rocm(s) => s,
        _ => hanzo_ml::bail!("slot_mapping must be a rocm tensor"),
    };

    if k_l.stride().len() != 3 || v_l.stride().len() != 3 {
        hanzo_ml::bail!("reshape_and_cache expects k/v of rank 3 (k: {k_l:?}, v: {v_l:?})")
    }
    if kc_l.stride().len() != 5 {
        hanzo_ml::bail!("reshape_and_cache expects key_cache of rank 5 (key_cache: {kc_l:?})")
    }
    if vc_l.stride().len() != 4 {
        hanzo_ml::bail!("reshape_and_cache expects value_cache of rank 4 (value_cache: {vc_l:?})")
    }

    let dev = k.device().clone();

    let kc_ptr = slice_ptr_at(&kc.slice, kc_l.start_offset());
    let vc_ptr = slice_ptr_at(&vc.slice, vc_l.start_offset());
    let k_ptr = slice_ptr_at(&k.slice, k_l.start_offset());
    let v_ptr = slice_ptr_at(&v.slice, v_l.start_offset());
    // slot_mapping is i64.
    let s_ptr = match &s.slice {
        RocmStorageSlice::I64(m) => unsafe {
            m.offset_ptr(s_l.start_offset()) as *const core::ffi::c_long
        },
        _ => hanzo_ml::bail!("slot_mapping must be an i64 rocm tensor"),
    };

    let (num_tokens, num_heads, head_size) = k_l.shape().dims3()?;
    if (num_tokens, num_heads, head_size) != v_l.shape().dims3()? {
        hanzo_ml::bail!("shape mismatch k {:?} and v {:?}", k_l.shape(), v_l.shape())
    }
    let (num_blocks, num_heads_kc, head_size_kc, block_size, x) = kc_l.shape().dims5()?;
    if num_heads_kc != num_heads || head_size_kc != head_size / x {
        hanzo_ml::bail!("shape mismatch key_cache {:?}", kc_l.shape())
    }
    if (num_blocks, num_heads, head_size, block_size) != vc_l.shape().dims4()? {
        hanzo_ml::bail!(
            "shape mismatch key_cache {:?} and value_cache {:?}",
            kc_l.shape(),
            vc_l.shape()
        )
    }
    if num_tokens != s_l.shape().dims1()? {
        hanzo_ml::bail!(
            "shape mismatch slot_mapping {:?}, expected ({num_tokens})",
            s_l.shape()
        )
    }

    let key_stride = k_l.stride()[0] as c_int;
    let value_stride = v_l.stride()[0] as c_int;
    let stream_raw = dev.stream().as_raw() as i64;

    unsafe {
        ffi::reshape_and_cache(
            k_ptr,
            v_ptr,
            kc_ptr,
            vc_ptr,
            s_ptr,
            num_tokens as c_int,
            num_heads as c_int,
            head_size as c_int,
            block_size as c_int,
            x as c_int,
            key_stride,
            value_stride,
            stream_raw,
            internal_type,
            cache_dtype,
            std::ptr::null(),
            std::ptr::null(),
        )
    }
    // No per-call dev.synchronize(): reshape_and_cache is enqueued on the
    // device stream and the subsequent paged_attention read of the same cache
    // runs on the same stream, so stream ordering is sufficient. A per-call
    // sync would prevent decode hipGraph capture (stage 3) and the CUDA mirror
    // does not synchronize either.
    Ok(())
}

/// FP8 KV-scale update. Not supported on the ROCm backend (stage 1/2 scope is
/// the f16/bf16 v1 decode path), but provided so the shared PagedAttention
/// layer compiles under `feature = "rocm"`. The ROCm cache path never produces
/// FP8 scales, so this is unreachable at runtime; it bails loudly if hit.
pub fn kv_scale_update(
    _key: &Tensor,
    _value: &Tensor,
    _k_scales: &Tensor,
    _v_scales: &Tensor,
) -> Result<()> {
    hanzo_ml::bail!("FP8 kv_scale_update is not supported on the rocm backend")
}

/// Gather the paged KV cache into contiguous K/V tensors (used by the prefix-
/// cache / sliding-window prefill paths). Not ported to ROCm in stage 1/2: the
/// fresh-prompt + steady-decode flow that stage 2 targets does not take the
/// gather path (Qwen3 head_dim 128 <= STANDARD_PAGED_ATTENTION_MAX_HEAD_SIZE,
/// and a fresh prompt has no cached prefix). Provided so the shared layer
/// compiles under `feature = "rocm"`; bails loudly if a gather path is hit.
#[allow(clippy::too_many_arguments)]
pub fn gather_kv_cache(
    _key_cache: &Tensor,
    _value_cache: &Tensor,
    _k_scale: Option<&Tensor>,
    _v_scale: Option<&Tensor>,
    _block_table: &Tensor,
    _cu_seq_lens: &Tensor,
    _out_dtype: DType,
) -> Result<(Tensor, Tensor)> {
    hanzo_ml::bail!(
        "gather_kv_cache (paged prefix-cache / SWA gather) is not supported on the rocm backend"
    )
}

/// Insert key/values at `slot_mapping` into the paged KV cache. ROCm mirror of
/// src/cuda/backend/paged_attention.rs::reshape_and_cache.
pub fn reshape_and_cache(
    key: &Tensor,
    value: &Tensor,
    k_scale: Option<&Tensor>,
    v_scale: Option<&Tensor>,
    key_cache: &Tensor,
    value_cache: &Tensor,
    slot_mapping: &Tensor,
) -> Result<()> {
    // Identity (1.0) scales for a non-FP8 cache are a no-op for the write path;
    // see `paged_attention` above. The cache dtype is validated in
    // `update_cache`, which rejects FP8, so any real (non-identity) scale cannot
    // reach the kernel here.
    let _ = (k_scale, v_scale);
    match key.dtype() {
        DType::F16 | DType::BF16 | DType::F32 => {
            update_cache(key, value, key_cache, value_cache, slot_mapping)
        }
        dt => hanzo_ml::bail!("reshape_and_cache on rocm supports f32/f16/bf16 ({dt:?})"),
    }
}
