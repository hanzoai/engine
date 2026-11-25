#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

//! Quantized (GGUF) loader for Qwen3.5 / Qwen3.6 hybrid models.
//!
//! Architecture string: `qwen35moe` (35B-A3B MoE, primary target) or `qwen35` (27B dense).
//! Hybrid per-layer schedule: every `full_attention_interval`-th layer (default 4) is a gated
//! full-attention layer (partial-rotary interleaved mRoPE + sigmoid output gate); the rest are
//! Gated-DeltaNet (GDN) linear-attention layers (causal conv1d kernel=4, gated RMSNorm, recurrent
//! state). MoE variant uses 256 experts (8 routed + 1 shared); dense variant uses a plain SwiGLU MLP.
//!
//! The GDN math mirrors `models::gdn` / `models::qwen3_next` / `vision_models::qwen3_5_moe::text`.
//! The recurrent state is driven through the pipeline `HybridCache` exactly like `qwen3_next`.
//!
//! NOT YET CORRECTNESS-VERIFIED on a GPU with a real GGUF. The GGUF tensor-name mapping and the
//! GDN V-head ordering are documented inline; every assumption is called out. MTP and vision are
//! ignored (text-only).
//!
//! ===================== GGUF tensor-name mapping =====================
//! The `qwen35` / `qwen35moe` GGUF archs, as converted GGUF files name the HF tensors.
//!
//! Per layer `blk.{i}`:
//!   GDN (linear-attention) layers:
//!     attn_qkv.weight   <- linear_attn.in_proj_qkv  (merged q,k,v; key_dim*2 + value_dim, hidden)
//!     attn_gate.weight  <- linear_attn.in_proj_z    (the z gate; value_dim, hidden)
//!     ssm_conv1d.weight <- linear_attn.conv1d squeezed to 2D (conv_dim, kernel)
//!     ssm_dt.bias       <- linear_attn.dt_bias      (1D, num_v_heads, f32) [note: GGUF suffix is `bias`]
//!     ssm_a             <- -exp(A_log) precomputed at conversion (1D, num_v_heads, f32) [no `.weight`]
//!     ssm_beta.weight   <- linear_attn.in_proj_b    (num_v_heads, hidden)
//!     ssm_alpha.weight  <- linear_attn.in_proj_a    (num_v_heads, hidden)
//!     ssm_norm.weight   <- linear_attn.norm         (gated RMSNorm; head_v_dim)
//!     ssm_out.weight    <- linear_attn.out_proj     (value_dim, hidden)
//!   Full-attention layers:
//!     attn_q.weight     <- self_attn.q_proj  (DOUBLED: num_heads*head_dim*2 = query + gate)
//!     attn_k/attn_v/attn_output, attn_q_norm, attn_k_norm  (standard GQA + qk-norm)
//!   Shared (both): attn_norm (input_layernorm), post_attention_norm (post-attn layernorm).
//!   MoE FFN: ffn_gate_inp, ffn_gate_exps, ffn_up_exps, ffn_down_exps,
//!            ffn_gate_inp_shexp, ffn_gate_shexp, ffn_up_shexp, ffn_down_shexp.
//!   Dense FFN: ffn_gate, ffn_up, ffn_down.
//!   Global: token_embd, output_norm, output (tied to token_embd when absent).
//!
//! Metadata keys (prefixed with the arch string by ContentMetadata):
//!   attention.head_count / head_count_kv / key_length / value_length / layer_norm_rms_epsilon
//!   block_count, context_length, rope.freq_base, rope.dimension_count (= rot_dim),
//!   rope.dimension_sections (mrope [t,h,w,0]), full_attention_interval,
//!   ssm.conv_kernel (=linear_conv_kernel_dim), ssm.state_size (=linear_key_head_dim, also head_v_dim),
//!   ssm.group_count (=linear_num_key_heads), ssm.time_step_rank (=linear_num_value_heads),
//!   ssm.inner_size (=value_dim), expert_count, expert_used_count, expert_feed_forward_length.
//!
//! V-head ordering: the GGUF conversion REORDERS the V
//! heads of in_proj_qkv(v part) / in_proj_z / in_proj_a / in_proj_b / out_proj / conv1d(v part) /
//! A_log / dt_bias from HF grouped order [K0_v0..v{r-1}, K1_v0..v{r-1}, ...] into TILED order
//! [v0_K0..v0_K{K-1}, v1_K0..v1_K{K-1}, ...]. We do NOT undo this at load; instead the recurrence
//! consumes tiled order natively. Every per-V-head tensor (v, z, beta, g, conv V-channels) comes
//! straight from a GGUF projection in tiled order, and out_proj's input columns are tiled too, so
//! the layer is self-consistent end-to-end. The only place that mixes K-indexed and V-indexed
//! tensors is the q/k repeat in `QGatedDeltaNet::forward`, which TILES q/k (V head j -> K head
//! j % num_k_heads) to match. When num_k_heads == num_v_heads the reorder is a no-op and the tile
//! collapses to identity, leaving the original (verified) path unchanged. The shared safetensors
//! `gdn.rs` recurrence instead consumes HF grouped order (it repeats grouped: j -> j / v_per_group);
//! that path is untouched and still correct for its grouped inputs.

use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use crate::attention::{AttentionMask, SdpaParams};
use crate::device_map::{DeviceMappedMask, DeviceMapper};
use crate::gguf::Content;
use crate::layers::{CausalMaskConfig, CausalMasker, QRmsNorm, Qwen3VLRotaryEmbedding, Sdpa};
use crate::layers_masker::PastKvLenCache;
use crate::models::gdn::{
    forward_pooled, l2_norm, sigmoid, softplus, GdnLayerCache, PoolSlots, RmsNormGated,
};
use crate::ops::{TopKLastDimOp, TopKOutput};
use crate::paged_attention::{AttentionImplementation, PagedAttention};
use crate::pipeline::text_models_inputs_processor::PagedAttentionInputMetadata;
use crate::pipeline::{extract_logits, EitherCache, KvCache};
use crate::utils::gguf_metadata::{ContentMetadata, DEFAULT_FULL_ATTENTION_INTERVAL};
use crate::utils::model_config as ModelConfig;
use crate::utils::progress::{new_multi_progress, NiceProgressBar};
use hanzo_ml::quantized::{gguf_file, QMatMul, QTensor};
use hanzo_ml::{DType, Device, Result, Tensor, D};
use hanzo_nn::{Embedding, Module};
use hanzo_quant::{GgufMatMul, QuantMethod, QuantMethodConfig};

use crate::kv_cache::{
    HybridCache, HybridCacheConfig, HybridLayerCache, HybridLayerType, RecurrentLayerConfig,
};

const DEFAULT_MAX_SEQ_LEN: u32 = 4096;
const DEFAULT_PARTIAL_ROTARY_FACTOR: f64 = 0.25;
const L2_NORM_EPS: f64 = 1e-6;

mod prof {
    use hanzo_ml::{Device, Result};
    use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering::Relaxed};

    pub(super) static QPROJ_MLP_NS: AtomicU64 = AtomicU64::new(0);
    pub(super) static QPROJ_GDN_NS: AtomicU64 = AtomicU64::new(0);
    pub(super) static QPROJ_ATTN_NS: AtomicU64 = AtomicU64::new(0);
    pub(super) static GDN_RECURR_NS: AtomicU64 = AtomicU64::new(0);
    pub(super) static FULL_ATTN_NS: AtomicU64 = AtomicU64::new(0);
    pub(super) static HADAMARD_NS: AtomicU64 = AtomicU64::new(0);
    pub(super) static LM_HEAD_NS: AtomicU64 = AtomicU64::new(0);
    pub(super) static TOTAL_NS: AtomicU64 = AtomicU64::new(0);
    pub(super) static DECODE_STEPS: AtomicUsize = AtomicUsize::new(0);

    #[inline]
    pub(super) fn enabled() -> bool {
        #[cfg(feature = "cuda")]
        {
            hanzo_ml::quantized::cuda::cuda_profile_enabled()
        }
        #[cfg(not(feature = "cuda"))]
        {
            false
        }
    }

    #[inline]
    pub(super) fn timed<T, F: FnOnce() -> Result<T>>(
        dev: &Device,
        acc: &AtomicU64,
        f: F,
    ) -> Result<T> {
        if !enabled() {
            return f();
        }
        dev.synchronize()?;
        let t0 = std::time::Instant::now();
        let out = f()?;
        dev.synchronize()?;
        acc.fetch_add(t0.elapsed().as_nanos() as u64, Relaxed);
        Ok(out)
    }

    pub(super) fn reset() {
        QPROJ_MLP_NS.store(0, Relaxed);
        QPROJ_GDN_NS.store(0, Relaxed);
        QPROJ_ATTN_NS.store(0, Relaxed);
        GDN_RECURR_NS.store(0, Relaxed);
        FULL_ATTN_NS.store(0, Relaxed);
        HADAMARD_NS.store(0, Relaxed);
        LM_HEAD_NS.store(0, Relaxed);
        TOTAL_NS.store(0, Relaxed);
        #[cfg(feature = "cuda")]
        hanzo_ml::quantized::cuda::cuda_profile_reset();
    }

    pub(super) fn dump(kind: &str, seq_len: usize, steps: usize) {
        let qproj_mlp = QPROJ_MLP_NS.load(Relaxed) as f64 / 1e6;
        let qproj_gdn = QPROJ_GDN_NS.load(Relaxed) as f64 / 1e6;
        let qproj_attn = QPROJ_ATTN_NS.load(Relaxed) as f64 / 1e6;
        let gdn_recur = GDN_RECURR_NS.load(Relaxed) as f64 / 1e6;
        let full_attn = FULL_ATTN_NS.load(Relaxed) as f64 / 1e6;
        let had = HADAMARD_NS.load(Relaxed) as f64 / 1e6;
        let lm_head = LM_HEAD_NS.load(Relaxed) as f64 / 1e6;
        let total = TOTAL_NS.load(Relaxed) as f64 / 1e6;
        let qproj_all = qproj_mlp + qproj_gdn + qproj_attn;
        let accounted = qproj_all + gdn_recur + full_attn + had + lm_head;
        let other = (total - accounted).max(0.0);
        #[cfg(feature = "cuda")]
        let c = hanzo_ml::quantized::cuda::cuda_profile_snapshot();
        #[cfg(feature = "cuda")]
        eprintln!(
            "[27B_PROF {kind} seq={seq_len} steps={steps}] total={total:.2}ms \
             qproj={qproj_all:.2}ms(mlp={qproj_mlp:.2},gdn={qproj_gdn:.2},attn={qproj_attn:.2}) \
             gdn_recur={gdn_recur:.2}ms full_attn={full_attn:.2}ms had={had:.2}ms(fwht={:.2}ms/{}) \
             lm_head={lm_head:.2}ms other={other:.2}ms | \
             bf16_deq={:.2}ms({}) bf16_gemm={:.2}ms \
             dp4a_dec={:.2}ms({}) dp4a_ver={:.2}ms({}) dp4a_ge64M={:.2}ms({}) \
             mmq={:.2}ms(q={:.2},k={:.2},c={:.2}/{})",
            c.fwht_ms,
            c.fwht_calls,
            c.bf16_dequant_ms,
            c.bf16_dequant_calls,
            c.bf16_gemm_ms,
            c.dp4a_decode_ms,
            c.dp4a_decode_calls,
            c.dp4a_verify_ms,
            c.dp4a_verify_calls,
            c.dp4a_ge64m_ms,
            c.dp4a_ge64m_calls,
            c.mmq_quant_ms + c.mmq_kernel_ms + c.mmq_cast_ms,
            c.mmq_quant_ms,
            c.mmq_kernel_ms,
            c.mmq_cast_ms,
            c.mmq_calls,
        );
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LayerType {
    FullAttention,
    LinearAttention,
}

// ===================== Blockwise Hadamard activation rotation =====================

/// Parsed GGUF Hadamard folding metadata (`prism.hadamard.*` or `zen.hadamard.*`).
#[derive(Debug, Clone)]
pub(crate) struct HadamardMeta {
    pub(crate) block_size: usize,
    pub(crate) explicit_signs: bool,
    pub(crate) signs: HashMap<usize, Vec<f32>>,
    pub(crate) weight_names: HashSet<String>,
    pub(crate) inverse_weight_names: HashSet<String>,
    pub(crate) tied_output: bool,
    pub(crate) gdn_v_grouped: bool,
}

impl HadamardMeta {
    pub(crate) fn from_gguf_metadata(
        meta: &HashMap<String, gguf_file::Value>,
    ) -> Result<Option<Self>> {
        let prefix = if meta.contains_key("prism.hadamard.version") {
            "prism.hadamard"
        } else if meta.contains_key("zen.hadamard.version") {
            "zen.hadamard"
        } else {
            return Ok(None);
        };

        let get_u32 = |suffix: &str| -> Result<u32> {
            let key = format!("{prefix}.{suffix}");
            match meta.get(&key) {
                Some(gguf_file::Value::U32(v)) => Ok(*v),
                Some(gguf_file::Value::I32(v)) if *v >= 0 => Ok(*v as u32),
                Some(gguf_file::Value::U64(v)) => Ok(*v as u32),
                Some(gguf_file::Value::I64(v)) if *v >= 0 => Ok(*v as u32),
                Some(other) => hanzo_ml::bail!("Invalid type for `{key}`: {other:?}"),
                None => hanzo_ml::bail!("Missing required Hadamard metadata key `{key}`"),
            }
        };
        let get_opt_str = |suffix: &str| -> Result<Option<&str>> {
            let key = format!("{prefix}.{suffix}");
            match meta.get(&key) {
                Some(gguf_file::Value::String(s)) => Ok(Some(s.as_str())),
                Some(other) => hanzo_ml::bail!("Expected string for `{key}`, got {other:?}"),
                None => Ok(None),
            }
        };
        let get_opt_bool = |suffix: &str| -> Result<bool> {
            let key = format!("{prefix}.{suffix}");
            match meta.get(&key) {
                Some(gguf_file::Value::Bool(b)) => Ok(*b),
                Some(gguf_file::Value::U8(b)) => Ok(*b != 0),
                Some(gguf_file::Value::I8(b)) => Ok(*b != 0),
                Some(gguf_file::Value::U32(b)) => Ok(*b != 0),
                Some(gguf_file::Value::I32(b)) => Ok(*b != 0),
                Some(other) => hanzo_ml::bail!("Expected bool for `{key}`, got {other:?}"),
                None => Ok(false),
            }
        };
        let get_i32_arr = |suffix: &str| -> Result<Vec<i32>> {
            let key = format!("{prefix}.{suffix}");
            let Some(gguf_file::Value::Array(arr)) = meta.get(&key) else {
                hanzo_ml::bail!("Missing or non-array Hadamard metadata key `{key}`");
            };
            arr.iter()
                .map(|v| match v {
                    gguf_file::Value::I32(x) => Ok(*x),
                    gguf_file::Value::U32(x) => Ok(*x as i32),
                    gguf_file::Value::I16(x) => Ok(i32::from(*x)),
                    gguf_file::Value::U16(x) => Ok(i32::from(*x)),
                    gguf_file::Value::I8(x) => Ok(i32::from(*x)),
                    gguf_file::Value::U8(x) => Ok(i32::from(*x)),
                    gguf_file::Value::I64(x) => Ok(*x as i32),
                    gguf_file::Value::U64(x) => Ok(*x as i32),
                    other => hanzo_ml::bail!("Non-integer element in `{key}`: {other:?}"),
                })
                .collect()
        };
        let get_str_set = |suffix: &str, required: bool| -> Result<HashSet<String>> {
            let key = format!("{prefix}.{suffix}");
            let Some(val) = meta.get(&key) else {
                if required {
                    hanzo_ml::bail!("Missing required Hadamard metadata key `{key}`");
                }
                return Ok(HashSet::new());
            };
            let gguf_file::Value::Array(arr) = val else {
                hanzo_ml::bail!("Expected string array for `{key}`");
            };
            let mut set = HashSet::with_capacity(arr.len());
            for item in arr {
                let gguf_file::Value::String(s) = item else {
                    hanzo_ml::bail!("Non-string element in `{key}`: {item:?}");
                };
                if !set.insert(s.clone()) {
                    hanzo_ml::bail!("Duplicate weight entry `{s}` in `{key}`");
                }
            }
            Ok(set)
        };

        let version = get_u32("version")?;
        if version != 1 && version != 2 {
            hanzo_ml::bail!("Unsupported `{prefix}.version`: {version} (expected 1 or 2)");
        }
        let block_size = get_u32("block_size")? as usize;
        if block_size == 0 || !block_size.is_power_of_two() {
            hanzo_ml::bail!("Invalid `{prefix}.block_size`: {block_size} (must be power of 2)");
        }
        if let Some(transform) = get_opt_str("transform")? {
            if transform != "normalized-sylvester-walsh-hadamard" {
                hanzo_ml::bail!("Unsupported `{prefix}.transform`: {transform}");
            }
        }
        if let Some(axis) = get_opt_str("axis")? {
            if axis != "input-last-dimension" {
                hanzo_ml::bail!("Unsupported `{prefix}.axis`: {axis}");
            }
        }
        let sign_mode = get_opt_str("sign_mode")?.unwrap_or("identity");
        let explicit_signs = match sign_mode {
            "identity" => false,
            "explicit" => true,
            other => hanzo_ml::bail!("Unsupported `{prefix}.sign_mode`: {other}"),
        };

        let mut signs = HashMap::new();
        if explicit_signs {
            let widths = get_i32_arr("sign_widths")?;
            let values = get_i32_arr("sign_values")?;
            if widths.is_empty() {
                hanzo_ml::bail!("`{prefix}.sign_mode` is explicit but `sign_widths` is empty");
            }
            let mut offset = 0usize;
            for w in widths {
                if w <= 0 {
                    hanzo_ml::bail!("Invalid Hadamard sign width: {w}");
                }
                let width = w as usize;
                if !width.is_multiple_of(block_size) || offset + width > values.len() {
                    hanzo_ml::bail!(
                        "Invalid Hadamard sign width {width} for block_size {block_size} (values len {})",
                        values.len()
                    );
                }
                let mut vec_f32 = Vec::with_capacity(width);
                for &sv in &values[offset..offset + width] {
                    if sv != 1 && sv != -1 {
                        hanzo_ml::bail!("Hadamard sign values must be +/-1, got {sv}");
                    }
                    vec_f32.push(sv as f32);
                }
                signs.insert(width, vec_f32);
                offset += width;
            }
            if offset != values.len() {
                hanzo_ml::bail!(
                    "`{prefix}.sign_values` length mismatch: consumed {offset}, total {}",
                    values.len()
                );
            }
        }

        let weight_names = get_str_set("weight_names", true)?;
        if weight_names.is_empty() {
            hanzo_ml::bail!("`{prefix}.weight_names` is empty");
        }
        let inverse_weight_names = get_str_set("inverse_weight_names", false)?;
        for name in &inverse_weight_names {
            if name != "token_embd.weight" {
                hanzo_ml::bail!(
                    "`{prefix}.inverse_weight_names` entry `{name}` is not a supported inverse table"
                );
            }
        }
        let tied_output = get_opt_bool("tied_output")?;
        let gdn_v_grouped = get_opt_bool("gdn_v_grouped")?;

        Ok(Some(Self {
            block_size,
            explicit_signs,
            signs,
            weight_names,
            inverse_weight_names,
            tied_output,
            gdn_v_grouped,
        }))
    }

    #[inline]
    pub(crate) fn is_folded(&self, weight_name: &str) -> bool {
        self.weight_names.contains(weight_name)
    }

    #[inline]
    pub(crate) fn is_inverse(&self, weight_name: &str) -> bool {
        self.inverse_weight_names.contains(weight_name)
    }
}

const MAX_FWHT_BLOCK: usize = 8192;

#[inline]
fn can_fuse_single_row_iq_q8(x: &Tensor, block_size: usize) -> bool {
    let Some(&width) = x.dims().last() else {
        return false;
    };
    x.device().is_cuda()
        && x.rank() >= 2
        && width >= 32
        && width.is_multiple_of(32)
        && (1..=MAX_FWHT_BLOCK).contains(&block_size)
        && width.is_multiple_of(block_size)
        && x.elem_count() == width
}

fn cpu_fwht_blocks(
    data: &mut [f32],
    width: usize,
    block: usize,
    signs: Option<&[f32]>,
    inverse: bool,
) {
    let scale = 1.0f32 / (block as f32).sqrt();
    for row in data.chunks_exact_mut(width) {
        if !inverse {
            if let Some(s) = signs {
                for (v, &sv) in row.iter_mut().zip(s.iter()) {
                    *v *= sv;
                }
            }
        }
        for blk in row.chunks_exact_mut(block) {
            let mut stride = 1usize;
            while stride < block {
                let step = stride * 2;
                let mut base = 0usize;
                while base < block {
                    for j in 0..stride {
                        let a = blk[base + j];
                        let b = blk[base + j + stride];
                        blk[base + j] = a + b;
                        blk[base + j + stride] = a - b;
                    }
                    base += step;
                }
                stride = step;
            }
            for v in blk.iter_mut() {
                *v *= scale;
            }
        }
        if inverse {
            if let Some(s) = signs {
                for (v, &sv) in row.iter_mut().zip(s.iter()) {
                    *v *= sv;
                }
            }
        }
    }
}

pub(crate) struct DeviceHadamard {
    block_size: usize,
    explicit_signs: bool,
    signs: HashMap<usize, Tensor>,
    signs_host: HashMap<usize, Vec<f32>>,
}

impl DeviceHadamard {
    pub(crate) fn new(meta: &HadamardMeta, device: &Device) -> Result<Self> {
        let b = meta.block_size;
        if b == 0 || !b.is_power_of_two() {
            hanzo_ml::bail!("Hadamard block_size must be a power of two, got {b}");
        }
        let mut signs = HashMap::with_capacity(meta.signs.len());
        let mut signs_host = HashMap::with_capacity(meta.signs.len());
        for (&width, vals) in &meta.signs {
            signs.insert(width, Tensor::from_vec(vals.clone(), width, device)?);
            signs_host.insert(width, vals.clone());
        }
        Ok(Self {
            block_size: b,
            explicit_signs: meta.explicit_signs,
            signs,
            signs_host,
        })
    }

    pub(crate) fn signs_for(&self, width: usize) -> Result<Option<&Tensor>> {
        if let Some(s) = self.signs.get(&width) {
            Ok(Some(s))
        } else if self.explicit_signs {
            hanzo_ml::bail!("Missing Hadamard sign vector for width {width}");
        } else {
            Ok(None)
        }
    }

    fn transform(&self, x: &Tensor, apply_signs: bool, inverse: bool) -> Result<Tensor> {
        let orig_dtype = x.dtype();
        let orig_shape = x.shape().clone();
        let Some(&width) = orig_shape.dims().last() else {
            hanzo_ml::bail!("Hadamard transform expects rank >= 1");
        };
        if width == 0 || !width.is_multiple_of(self.block_size) {
            hanzo_ml::bail!(
                "Hadamard width {width} is not divisible by block_size {}",
                self.block_size
            );
        }
        let signs = if apply_signs {
            self.signs_for(width)?
        } else {
            None
        };
        if let Some(out) = prof::timed(x.device(), &prof::HADAMARD_NS, || {
            hanzo_ml::fwht::try_hadamard_fwht(x, signs, self.block_size, inverse)
        })? {
            return Ok(out);
        }
        if (2..=MAX_FWHT_BLOCK).contains(&self.block_size)
            && matches!(orig_dtype, DType::F32 | DType::F16 | DType::BF16)
        {
            return prof::timed(x.device(), &prof::HADAMARD_NS, || {
                hanzo_ml::quantized::hadamard_fwht(x, signs, self.block_size, inverse)
            });
        }
        let mut data = x.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
        let signs_slice = if apply_signs {
            self.signs_host.get(&width).map(Vec::as_slice)
        } else {
            None
        };
        cpu_fwht_blocks(&mut data, width, self.block_size, signs_slice, inverse);
        let out = Tensor::from_vec(data, orig_shape, x.device())?;
        if orig_dtype == DType::F32 {
            Ok(out)
        } else {
            out.to_dtype(orig_dtype)
        }
    }

    pub(crate) fn block_size(&self) -> usize {
        self.block_size
    }

    pub(crate) fn forward_iq_q8(
        &self,
        x: &Tensor,
        presigned: bool,
    ) -> Result<hanzo_ml::quantized::IqQ8Activation> {
        let Some(&width) = x.dims().last() else {
            hanzo_ml::bail!("Hadamard expects rank >= 1");
        };
        let signs = if presigned {
            None
        } else {
            self.signs_for(width)?
        };
        prof::timed(x.device(), &prof::HADAMARD_NS, || {
            hanzo_ml::quantized::QMatMul::hadamard_fwht_iq_q8(x, signs, self.block_size, false)
        })
    }

    pub(crate) fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.transform(x, true, false)
    }

    pub(crate) fn forward_presigned(&self, x: &Tensor) -> Result<Tensor> {
        self.transform(x, false, false)
    }

    pub(crate) fn inverse(&self, u: &Tensor) -> Result<Tensor> {
        self.transform(u, true, true)
    }
}

/// Token embedding table keeping quantized rows resident on CPU/CUDA/Metal (or host-gathered on
/// Vulkan/ROCm) instead of expanding large vocab tables (e.g. 248,320 x 5,120 = 5.09 GB F32 on 27B,
/// which exceeds Vulkan's 4 GiB `maxStorageBufferRange` on NVIDIA GPUs and clamps token IDs >= 209,715).
#[derive(Clone)]
enum TokEmbedding {
    Dense(Embedding),
    Quantized(Arc<QTensor>),
    QuantizedHost { q: Arc<QTensor>, target: Device },
}

impl TokEmbedding {
    fn from_gguf_qtensor(
        q: Arc<QTensor>,
        host_q: Option<Arc<QTensor>>,
        hidden_size: usize,
        device: &Device,
    ) -> Result<Self> {
        let block_aligned = hidden_size.is_multiple_of(q.dtype().block_size());
        if matches!(device, Device::Cpu | Device::Cuda(_) | Device::Metal(_)) && block_aligned {
            Ok(Self::Quantized(q))
        } else if let Some(hq) = host_q.filter(|_| block_aligned) {
            Ok(Self::QuantizedHost {
                q: hq,
                target: device.clone(),
            })
        } else {
            Ok(Self::Dense(Embedding::new(
                q.dequantize(device)?,
                hidden_size,
            )))
        }
    }

    fn forward(&self, ids: &Tensor) -> Result<Tensor> {
        match self {
            Self::Dense(e) => e.forward(ids),
            Self::Quantized(q) => q.embedding(ids),
            Self::QuantizedHost { q, target } => {
                let ids_cpu = ids.to_device(&Device::Cpu)?;
                q.embedding(&ids_cpu)?.to_device(target)
            }
        }
    }

    fn dtype(&self) -> DType {
        match self {
            Self::Dense(e) => e.embeddings().dtype(),
            Self::Quantized(_) | Self::QuantizedHost { .. } => DType::F32,
        }
    }
}

// ===================== MoE / dense FFN =====================

/// A router or gate GEMM: GGUF's QMatMul, or a quantization-aware linear.
pub(crate) enum Gemm {
    Gguf(QMatMul),
    Quant(Arc<dyn QuantMethod>),
}

impl Gemm {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        match self {
            Self::Gguf(m) => m.forward(x),
            Self::Quant(m) => m.forward(x),
        }
    }
}

/// The routed experts' gate, up and down banks.
pub(crate) enum Banks {
    /// GGUF stacked banks for the indexed matmul.
    Gguf([QMatMul; 3]),
    /// Quantized stacked banks (NVFP4 experts), run through `gather_forward`.
    Quant([Arc<dyn QuantMethod>; 3]),
}

pub(crate) struct FusedMoe {
    gate: Gemm,
    banks: Banks,
    shared_gate: Gemm,
    pub(crate) shared_gate_proj: Arc<dyn QuantMethod>,
    pub(crate) shared_up_proj: Arc<dyn QuantMethod>,
    pub(crate) shared_down_proj: Arc<dyn QuantMethod>,
    norm_topk_prob: bool,
    num_experts_per_tok: usize,
    /// The dtype router logits are rounded to before the f32 softmax: F32 for GGUF, the
    /// activation dtype (bf16) for the served bf16 router GEMM.
    router_dtype: DType,
    gate_exps_had: Option<Arc<DeviceHadamard>>,
    up_exps_had: Option<Arc<DeviceHadamard>>,
    down_exps_had: Option<Arc<DeviceHadamard>>,
    shexp_gate_had: Option<Arc<DeviceHadamard>>,
    shexp_up_had: Option<Arc<DeviceHadamard>>,
    shexp_down_had: Option<Arc<DeviceHadamard>>,
}

/// `silu(g) · u` in f32 with one rounding to `dtype` (`triton_poi_fused_mul_silu_slice_0`, and
/// FlashInfer's SwiGLU before its fc2 quantizer).
fn swiglu_rounded(g: &Tensor, u: &Tensor, dtype: DType) -> Result<Tensor> {
    let g = g.to_dtype(DType::F32)?;
    let u = u.to_dtype(DType::F32)?;
    ((g.clone() / (g.neg()?.exp()? + 1.0)?)? * u)?.to_dtype(dtype)
}

impl FusedMoe {
    /// Load the shared sparse-MoE block (routed experts + sigmoid-gated shared expert) from GGUF.
    /// Identical tensor layout for Qwen3.5-MoE and Qwen3-Next: `ffn_gate_inp`, `ffn_{gate,up,down}_exps`,
    /// and the `*_shexp` shared-expert tensors. `norm_topk_prob` is always true for both.
    pub(crate) fn from_gguf<R: std::io::Seek + std::io::Read>(
        ct: &mut Content<'_, R>,
        prefix: &str,
        dev: &Device,
        num_experts_per_tok: usize,
    ) -> Result<Self> {
        let gate = ct.tensor(&format!("{prefix}.ffn_gate_inp.weight"), dev)?;
        let gate_experts = ct.tensor(&format!("{prefix}.ffn_gate_exps.weight"), dev)?;
        let up_experts = ct.tensor(&format!("{prefix}.ffn_up_exps.weight"), dev)?;
        let down_experts = ct.tensor(&format!("{prefix}.ffn_down_exps.weight"), dev)?;
        let shared_gate = ct.tensor(&format!("{prefix}.ffn_gate_inp_shexp.weight"), dev)?;
        let shared_gate_proj =
            gguf_qmm(ct.tensor(&format!("{prefix}.ffn_gate_shexp.weight"), dev)?)?;
        let shared_up_proj = gguf_qmm(ct.tensor(&format!("{prefix}.ffn_up_shexp.weight"), dev)?)?;
        let shared_down_proj =
            gguf_qmm(ct.tensor(&format!("{prefix}.ffn_down_shexp.weight"), dev)?)?;
        Ok(Self {
            gate: Gemm::Gguf(QMatMul::from_qtensor(gate)?),
            banks: Banks::Gguf([
                QMatMul::from_qtensor(gate_experts)?,
                QMatMul::from_qtensor(up_experts)?,
                QMatMul::from_qtensor(down_experts)?,
            ]),
            // ffn_gate_inp_shexp is a hidden->1 shared-expert gate stored 1D [hidden]; the
            // matmul needs 2D, so dequantize and reshape to [1, hidden].
            shared_gate: Gemm::Gguf({
                let w = shared_gate.dequantize(dev)?;
                QMatMul::Tensor(if w.rank() == 1 { w.unsqueeze(0)? } else { w })
            }),
            shared_gate_proj,
            shared_up_proj,
            shared_down_proj,
            norm_topk_prob: true,
            num_experts_per_tok,
            router_dtype: DType::F32,
            gate_exps_had: None,
            up_exps_had: None,
            down_exps_had: None,
            shexp_gate_had: None,
            shexp_up_had: None,
            shexp_down_had: None,
        })
    }

    pub(crate) fn with_hadamard(
        mut self,
        had: &Arc<DeviceHadamard>,
        meta: &HadamardMeta,
        prefix: &str,
    ) -> Self {
        let pick = |suffix: &str| {
            meta.is_folded(&format!("{prefix}.{suffix}"))
                .then(|| Arc::clone(had))
        };
        self.gate_exps_had = pick("ffn_gate_exps.weight");
        self.up_exps_had = pick("ffn_up_exps.weight");
        self.down_exps_had = pick("ffn_down_exps.weight");
        self.shexp_gate_had = pick("ffn_gate_shexp.weight");
        self.shexp_up_had = pick("ffn_up_shexp.weight");
        self.shexp_down_had = pick("ffn_down_shexp.weight");
        self
    }

    /// The MoE block at `vb` (`...mlp`) of a safetensors checkpoint: a bf16 router `gate`
    /// `[E, h]`, NVFP4 expert banks (gate and up quantize one input, so they share its
    /// activation scale), the shared expert through `quant` (block FP8), and a bf16
    /// `shared_expert_gate` `[1, h]`.
    pub(crate) fn new(
        vb: &hanzo_quant::ShardedVarBuilder,
        props: &PropsGGUF,
        quant: &Option<hanzo_quant::QuantizedConfig>,
        dtype: DType,
    ) -> Result<Self> {
        let h = props.embedding_length;
        let e = props.num_experts.unwrap_or(0);
        let ff = props.moe_intermediate_size;
        let lin = |i: usize, o: usize, q: &Option<hanzo_quant::QuantizedConfig>, name: &str| {
            hanzo_quant::linear_no_bias(i, o, q, vb.pp(name))
        };
        let experts = vb.pp("experts");
        let bank =
            |proj: &str, share: &[&str], n: usize, k: usize| -> Result<Arc<dyn QuantMethod>> {
                Ok(Arc::new(hanzo_quant::NVFP4Layer::experts(
                    experts.clone(),
                    proj,
                    share,
                    e,
                    n,
                    k,
                    dtype,
                )?))
            };
        // The shared expert runs inside the served MoE custom op, where the
        // standalone QuantFP8 kernel quantizes its inputs (amax div.full 448): a block-FP8
        // projection takes the Eager quantizer, anything else loads through `quant`.
        let shared = |i: usize, o: usize, name: &str| -> Result<Arc<dyn QuantMethod>> {
            let p = vb.pp(name);
            if p.contains_tensor("weight_scale_inv") {
                let w =
                    p.get_with_hints_dtype((o, i), "weight", Default::default(), DType::F8E4M3)?;
                let sc = p.get_with_hints_dtype(
                    (o.div_ceil(128), i.div_ceil(128)),
                    "weight_scale_inv",
                    Default::default(),
                    DType::F32,
                )?;
                hanzo_quant::blockwise_fp8_act(
                    w,
                    sc,
                    vec![128, 128],
                    dtype,
                    hanzo_quant::quantize::Fp8Mode::Eager,
                )
            } else {
                lin(i, o, quant, name)
            }
        };
        // The shared expert's intermediate width is its gate_proj's rows.
        let sff = vb
            .clone()
            .set_device(Device::Cpu)
            .get_unchecked_dtype("shared_expert.gate_proj.weight_scale_inv", DType::F32)
            .map(|s| s.dim(0).map(|b| b * 128))
            .unwrap_or(Ok(ff))?;
        Ok(Self {
            gate: Gemm::Quant(lin(h, e, &None, "gate")?),
            banks: Banks::Quant([
                bank("gate_proj", &["up_proj"], ff, h)?,
                bank("up_proj", &["gate_proj"], ff, h)?,
                bank("down_proj", &[], h, ff)?,
            ]),
            shared_gate: Gemm::Quant(lin(h, 1, &None, "shared_expert_gate")?),
            shared_gate_proj: shared(h, sff, "shared_expert.gate_proj")?,
            shared_up_proj: shared(h, sff, "shared_expert.up_proj")?,
            shared_down_proj: shared(sff, h, "shared_expert.down_proj")?,
            norm_topk_prob: true,
            num_experts_per_tok: props.num_experts_per_tok,
            router_dtype: dtype,
            gate_exps_had: None,
            up_exps_had: None,
            down_exps_had: None,
            shexp_gate_had: None,
            shexp_up_had: None,
            shexp_down_had: None,
        })
    }

    /// Router logits `[t, E]`, rounded to the router dtype.
    pub(crate) fn logits(&self, xs: &Tensor) -> Result<Tensor> {
        match self.router_dtype {
            DType::F32 => self.gate.forward(&xs.to_dtype(DType::F32)?),
            d => self.gate.forward(&xs.to_dtype(d)?),
        }
    }

    /// `(ids [t, k], weights [t, k] f32)`: softmax in f32 over the logits, the top k by
    /// probability (ties to the lower index), renormalized in f32.
    pub(crate) fn route(&self, logits: &Tensor) -> Result<(Tensor, Tensor)> {
        let routing_weights = hanzo_nn::ops::softmax_last_dim(&logits.to_dtype(DType::F32)?)?;
        let TopKOutput {
            values: mut scores,
            indices,
        } = routing_weights.topk(self.num_experts_per_tok)?;
        if self.norm_topk_prob {
            scores = scores.broadcast_div(&scores.sum_keepdim(D::Minus1)?)?;
        }
        Ok((indices, scores))
    }

    /// The routed experts over `xs` `[t, h]` for `ids`/`weights` `[t, k]`: `[t, h]` in `xs`'s dtype.
    pub(crate) fn experts(&self, xs: &Tensor, ids: &Tensor, weights: &Tensor) -> Result<Tensor> {
        let xs_rot = match self.gate_exps_had.as_ref().or(self.up_exps_had.as_ref()) {
            Some(had) => Some(had.forward(xs)?),
            None => None,
        };
        self.experts_with_rot(xs, xs_rot.as_ref(), ids, weights)
    }

    fn experts_with_rot(
        &self,
        xs: &Tensor,
        xs_rot: Option<&Tensor>,
        ids: &Tensor,
        weights: &Tensor,
    ) -> Result<Tensor> {
        let (num_tokens, hidden_dim) = xs.dims2()?;
        let xs_e = xs.reshape((num_tokens, 1, hidden_dim))?;
        let xs_rot_e = match xs_rot {
            Some(r) => Some(r.reshape((num_tokens, 1, hidden_dim))?),
            None => None,
        };
        let gate_in = if self.gate_exps_had.is_some() {
            xs_rot_e.as_ref().unwrap()
        } else {
            &xs_e
        };
        let up_in = if self.up_exps_had.is_some() {
            xs_rot_e.as_ref().unwrap()
        } else {
            &xs_e
        };
        let routed = match &self.banks {
            Banks::Gguf([gate, up, down]) => {
                let g = gate.indexed_moe_forward(gate_in, ids)?;
                let u = up.indexed_moe_forward(up_in, ids)?;
                let act = crate::ops::mul_and_act(&g, &u, crate::layers::Activation::Silu)?;
                let act = match &self.down_exps_had {
                    Some(had) => had.forward(&act)?,
                    None => act,
                };
                down.indexed_moe_forward(&act, ids)?
            }
            Banks::Quant([gate, up, down]) => {
                let ids = ids.to_dtype(DType::U32)?;
                let g = gate.gather_forward(gate_in, &ids)?;
                let u = up.gather_forward(up_in, &ids)?;
                let act = swiglu_rounded(&g, &u, xs.dtype())?;
                let act = match &self.down_exps_had {
                    Some(had) => had.forward(&act)?,
                    None => act,
                };
                down.gather_forward(&act, &ids)?
            }
        };
        // Weighted expert-combine: out[t,:] = sum_e scores[t,e] * routed[t,e,:], accumulated in
        // f32 and rounded once. The fused CUDA kernel does it in one coalesced pass.
        #[cfg(feature = "cuda")]
        if routed.device().is_cuda() {
            return crate::cuda::moe::moe_combine_cuda(&routed, weights);
        }
        routed
            .to_dtype(DType::F32)?
            .broadcast_mul(&weights.to_dtype(DType::F32)?.unsqueeze(D::Minus1)?)?
            .sum(D::Minus2)?
            .to_dtype(routed.dtype())
    }

    /// The sigmoid-gated shared expert over `xs` `[t, h]`.
    pub(crate) fn shared(&self, xs: &Tensor) -> Result<Tensor> {
        let xs_rot = match self.shexp_gate_had.as_ref().or(self.shexp_up_had.as_ref()) {
            Some(had) => Some(had.forward(xs)?),
            None => None,
        };
        self.shared_with_rot(xs, xs_rot.as_ref())
    }

    fn shared_with_rot(&self, xs: &Tensor, xs_rot: Option<&Tensor>) -> Result<Tensor> {
        let gate_in = if self.shexp_gate_had.is_some() {
            xs_rot.unwrap()
        } else {
            xs
        };
        let up_in = if self.shexp_up_had.is_some() {
            xs_rot.unwrap()
        } else {
            xs
        };
        let g = self.shared_gate_proj.forward(gate_in)?;
        let u = self.shared_up_proj.forward(up_in)?;
        if self.router_dtype == DType::F32 {
            // GGUF: matching qwen3_next SparseMoeBlock.
            let act = crate::ops::mul_and_act(&g, &u, crate::layers::Activation::Silu)?;
            let act = match &self.shexp_down_had {
                Some(had) => had.forward(&act)?,
                None => act,
            };
            let out = self.shared_down_proj.forward(&act)?;
            let gate = sigmoid(&self.shared_gate.forward(&xs.to_dtype(DType::F32)?)?)?
                .to_dtype(out.dtype())?;
            return out.broadcast_mul(&gate);
        }
        // Served: SiluAndMul in f32 with one rounding; `sigmoid(gate)` and its product with the
        // expert each rounded, as the eager module inside moe_forward_shared computes them.
        let dtype = xs.dtype();
        let act = swiglu_rounded(&g.to_dtype(dtype)?, &u.to_dtype(dtype)?, dtype)?;
        let act = match &self.shexp_down_had {
            Some(had) => had.forward(&act)?,
            None => act,
        };
        let out = self.shared_down_proj.forward(&act)?.to_dtype(dtype)?;
        let gate =
            sigmoid(&self.shared_gate.forward(xs)?.to_dtype(DType::F32)?)?.to_dtype(dtype)?;
        out.broadcast_mul(&gate)
    }

    pub(crate) fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let (batch, seq_len, hidden_dim) = xs.dims3()?;
        let xs = xs.reshape(((), hidden_dim))?;
        let original_dtype = xs.dtype();
        let xs_rot = match self
            .gate_exps_had
            .as_ref()
            .or(self.up_exps_had.as_ref())
            .or(self.shexp_gate_had.as_ref())
            .or(self.shexp_up_had.as_ref())
        {
            Some(had) => Some(had.forward(&xs)?),
            None => None,
        };
        let (ids, weights) = self.route(&self.logits(&xs)?)?;
        let routed = self.experts_with_rot(&xs, xs_rot.as_ref(), &ids, &weights)?;
        let shared = self.shared_with_rot(&xs, xs_rot.as_ref())?;
        let out = if self.router_dtype == DType::F32 {
            (routed + shared)?
        } else {
            // routed + shared in f32, one rounding (the compiled add kernel)
            (routed.to_dtype(DType::F32)? + shared.to_dtype(DType::F32)?)?
        };
        out.reshape((batch, seq_len, hidden_dim))?
            .to_dtype(original_dtype)
    }
}

pub(crate) struct DenseMlp {
    gate: Arc<dyn QuantMethod>,
    up: Arc<dyn QuantMethod>,
    down: Arc<dyn QuantMethod>,
    gate_had: Option<Arc<DeviceHadamard>>,
    up_had: Option<Arc<DeviceHadamard>>,
    down_had: Option<Arc<DeviceHadamard>>,
    presigned_input: bool,
}

impl DenseMlp {
    pub(crate) fn load<R: std::io::Seek + std::io::Read>(
        ct: &mut Content<'_, R>,
        prefix: &str,
        device: &Device,
    ) -> Result<Self> {
        Ok(Self {
            gate: gguf_qmm(ct.tensor(&format!("{prefix}.ffn_gate.weight"), device)?)?,
            up: gguf_qmm(ct.tensor(&format!("{prefix}.ffn_up.weight"), device)?)?,
            down: gguf_qmm(ct.tensor(&format!("{prefix}.ffn_down.weight"), device)?)?,
            gate_had: None,
            up_had: None,
            down_had: None,
            presigned_input: false,
        })
    }

    pub(crate) fn with_hadamard(
        mut self,
        had: &Arc<DeviceHadamard>,
        meta: &HadamardMeta,
        prefix: &str,
    ) -> Self {
        let pick = |suffix: &str| {
            meta.is_folded(&format!("{prefix}.{suffix}"))
                .then(|| Arc::clone(had))
        };
        self.gate_had = pick("ffn_gate.weight");
        self.up_had = pick("ffn_up.weight");
        self.down_had = pick("ffn_down.weight");
        self
    }

    pub(crate) fn input_hadamard_for_norm_fold(&self) -> Option<&Arc<DeviceHadamard>> {
        if self.gate_had.is_some() && self.up_had.is_some() {
            self.gate_had.as_ref()
        } else {
            None
        }
    }

    pub(crate) fn set_presigned_input(&mut self) {
        self.presigned_input = true;
    }

    pub(crate) fn can_fuse_norm_fwht_q8(&self) -> Option<usize> {
        if self.presigned_input
            && self.gate.supports_shared_iq_q8()
            && self.up.supports_shared_iq_q8()
        {
            self.gate_had.as_ref().map(|h| h.block_size())
        } else {
            None
        }
    }

    pub(crate) fn forward_iq_q8(
        &self,
        q8: &hanzo_ml::quantized::IqQ8Activation,
        out_dtype: DType,
    ) -> Result<Tensor> {
        let dev = self.gate.dtype_and_device().1;
        let (gate, up) = prof::timed(&dev, &prof::QPROJ_MLP_NS, || {
            let gate = self.gate.forward_iq_q8_dtype(q8, out_dtype)?;
            let up = self.up.forward_iq_q8_dtype(q8, out_dtype)?;
            Ok((gate, up))
        })?;
        self.finish_down(gate, up, out_dtype)
    }

    fn finish_down(&self, gate: Tensor, up: Tensor, out_dtype: DType) -> Result<Tensor> {
        let y = crate::ops::mul_and_act(&gate, &up, crate::layers::Activation::Silu)?;
        if let Some(ref had) = self.down_had {
            if self.down.supports_shared_iq_q8() && can_fuse_single_row_iq_q8(&y, had.block_size())
            {
                let q8 = had.forward_iq_q8(&y, false)?;
                return prof::timed(y.device(), &prof::QPROJ_MLP_NS, || {
                    self.down.forward_iq_q8_dtype(&q8, out_dtype)
                });
            }
            let y_rot = had.forward(&y)?;
            return prof::timed(y.device(), &prof::QPROJ_MLP_NS, || {
                self.down.forward(&y_rot)
            });
        }
        prof::timed(y.device(), &prof::QPROJ_MLP_NS, || self.down.forward(&y))
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        if let (Some(ref g_had), true, true) = (
            &self.gate_had,
            self.up_had.is_some(),
            self.gate.supports_shared_iq_q8() && self.up.supports_shared_iq_q8(),
        ) {
            if can_fuse_single_row_iq_q8(xs, g_had.block_size()) {
                let q8 = g_had.forward_iq_q8(xs, self.presigned_input)?;
                return self.forward_iq_q8(&q8, xs.dtype());
            }
        }
        let xs_rot = match self.gate_had.as_ref().or(self.up_had.as_ref()) {
            Some(had) if self.presigned_input => Some(had.forward_presigned(xs)?),
            Some(had) => Some(had.forward(xs)?),
            None => None,
        };
        let gate_in = if self.gate_had.is_some() {
            xs_rot.as_ref().unwrap()
        } else {
            xs
        };
        let up_in = if self.up_had.is_some() {
            xs_rot.as_ref().unwrap()
        } else {
            xs
        };
        let (gate, up) = prof::timed(xs.device(), &prof::QPROJ_MLP_NS, || {
            let gate = self.gate.forward(gate_in)?;
            let up = self.up.forward(up_in)?;
            Ok((gate, up))
        })?;
        self.finish_down(gate, up, xs.dtype())
    }
}

pub(crate) enum MoeOrMlp {
    FusedMoe(Box<FusedMoe>),
    Mlp(DenseMlp),
}

impl MoeOrMlp {
    pub(crate) fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        match self {
            Self::Mlp(m) => m.forward(xs),
            Self::FusedMoe(m) => m.forward(xs),
        }
    }
}

// ===================== Gated full-attention layer =====================

pub(crate) struct GatedFullAttention {
    // q_proj output is doubled: first head_dim is q, second head_dim is the output gate.
    attn_q: Arc<dyn QuantMethod>,
    attn_k: Arc<dyn QuantMethod>,
    attn_v: Arc<dyn QuantMethod>,
    attn_o: Arc<dyn QuantMethod>,
    q_norm: QRmsNorm,
    k_norm: QRmsNorm,
    n_head: usize,
    n_kv_head: usize,
    head_dim: usize,
    rotary: Arc<Qwen3VLRotaryEmbedding>,
    paged_attn: Option<PagedAttention>,
    sdpa_params: SdpaParams,
    dtype: DType,
    /// Attend with the QSA kernel's math (zen6) instead of SDPA.
    qsa: bool,
    q_had: Option<Arc<DeviceHadamard>>,
    k_had: Option<Arc<DeviceHadamard>>,
    v_had: Option<Arc<DeviceHadamard>>,
    o_had: Option<Arc<DeviceHadamard>>,
    presigned_input: bool,
}

impl GatedFullAttention {
    /// The attention at `vb` (`...self_attn`) of a safetensors checkpoint: `q_proj` `[2·hq·d, h]`
    /// with the per-head `[q | gate]` layout GGUF `attn_q` uses, `k_proj`/`v_proj` `[hkv·d, h]`,
    /// `o_proj` `[h, hq·d]`, all through `quant` (block FP8 in a ModelOpt checkpoint), and the
    /// Gemma q/k norms as `1 + w` in f32. Attends with the QSA kernel's math.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        vb: &hanzo_quant::ShardedVarBuilder,
        props: &PropsGGUF,
        quant: &Option<hanzo_quant::QuantizedConfig>,
        rotary: Arc<Qwen3VLRotaryEmbedding>,
        paged_attn: Option<PagedAttention>,
        device: &Device,
        dtype: DType,
    ) -> Result<Self> {
        let (h, hq, hkv, d) = (
            props.embedding_length,
            props.head_count,
            props.head_count_kv,
            props.head_dim,
        );
        let lin =
            |i: usize, o: usize, name: &str| hanzo_quant::linear_no_bias(i, o, quant, vb.pp(name));
        let norm = |name: &str| -> Result<QRmsNorm> {
            let w = vb.get_with_hints_dtype(
                (d,),
                &format!("{name}.weight"),
                Default::default(),
                DType::F32,
            )?;
            Ok(QRmsNorm::from_weight(
                (w.to_device(device)? + 1.0)?,
                props.rms_norm_eps as f64,
            ))
        };
        Ok(Self {
            attn_q: lin(h, 2 * hq * d, "q_proj")?,
            attn_k: lin(h, hkv * d, "k_proj")?,
            attn_v: lin(h, hkv * d, "v_proj")?,
            attn_o: lin(hq * d, h, "o_proj")?,
            q_norm: norm("q_norm")?,
            k_norm: norm("k_norm")?,
            n_head: hq,
            n_kv_head: hkv,
            head_dim: d,
            rotary,
            paged_attn,
            sdpa_params: SdpaParams {
                n_kv_groups: hq / hkv,
                softcap: None,
                softmax_scale: 1.0 / (d as f32).sqrt(),
                sliding_window: None,
                sinks: None,
            },
            dtype,
            qsa: true,
            q_had: None,
            k_had: None,
            v_had: None,
            o_had: None,
            presigned_input: false,
        })
    }

    /// Attend with the QSA kernel's math (zen6's attention in both formats).
    pub(crate) fn qsa(self) -> Self {
        Self { qsa: true, ..self }
    }

    pub(crate) fn with_hadamard(
        mut self,
        had: &Arc<DeviceHadamard>,
        meta: &HadamardMeta,
        prefix: &str,
    ) -> Self {
        let pick = |suffix: &str| {
            meta.is_folded(&format!("{prefix}.{suffix}"))
                .then(|| Arc::clone(had))
        };
        self.q_had = pick("attn_q.weight");
        self.k_had = pick("attn_k.weight");
        self.v_had = pick("attn_v.weight");
        self.o_had = pick("attn_output.weight");
        self
    }

    pub(crate) fn input_hadamard_for_norm_fold(&self) -> Option<&Arc<DeviceHadamard>> {
        if self.q_had.is_some() && self.k_had.is_some() && self.v_had.is_some() {
            self.q_had.as_ref()
        } else {
            None
        }
    }

    pub(crate) fn set_presigned_input(&mut self) {
        self.presigned_input = true;
    }

    pub(crate) fn can_fuse_norm_fwht_q8(&self) -> Option<usize> {
        if self.presigned_input
            && self.attn_q.supports_shared_iq_q8()
            && self.attn_k.supports_shared_iq_q8()
            && self.attn_v.supports_shared_iq_q8()
        {
            self.q_had.as_ref().map(|h| h.block_size())
        } else {
            None
        }
    }

    /// The attention of one full-attention block at `prefix`. `paged_attn` is `None` for a block
    /// that keeps its own `KvCache` instead of a slot in the paged cache.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn load<R: std::io::Seek + std::io::Read>(
        ct: &mut Content<'_, R>,
        prefix: &str,
        props: &PropsGGUF,
        rotary: Arc<Qwen3VLRotaryEmbedding>,
        paged_attn: Option<PagedAttention>,
        device: &Device,
        dtype: DType,
    ) -> Result<Self> {
        Ok(Self {
            attn_q: gguf_qmm(ct.tensor(&format!("{prefix}.attn_q.weight"), device)?)?,
            attn_k: gguf_qmm(ct.tensor(&format!("{prefix}.attn_k.weight"), device)?)?,
            attn_v: gguf_qmm(ct.tensor(&format!("{prefix}.attn_v.weight"), device)?)?,
            attn_o: gguf_qmm(ct.tensor(&format!("{prefix}.attn_output.weight"), device)?)?,
            q_norm: QRmsNorm::new(
                ct.tensor(&format!("{prefix}.attn_q_norm.weight"), device)?,
                props.rms_norm_eps,
            )?,
            k_norm: QRmsNorm::new(
                ct.tensor(&format!("{prefix}.attn_k_norm.weight"), device)?,
                props.rms_norm_eps,
            )?,
            n_head: props.head_count,
            n_kv_head: props.head_count_kv,
            head_dim: props.head_dim,
            rotary,
            paged_attn,
            sdpa_params: SdpaParams {
                n_kv_groups: props.head_count / props.head_count_kv,
                softcap: None,
                softmax_scale: 1.0 / (props.head_dim as f32).sqrt(),
                sliding_window: None,
                sinks: None,
            },
            dtype,
            qsa: false,
            q_had: None,
            k_had: None,
            v_had: None,
            o_had: None,
            presigned_input: false,
        })
    }

    /// Query heads, K/V heads and head width.
    pub(crate) fn heads(&self) -> (usize, usize, usize) {
        (self.n_head, self.n_kv_head, self.head_dim)
    }

    /// Writes rows `[tokens, heads, dim]` into the paged `(K, V)` entry `cache` at the forward's
    /// slots, the write this block's paged attention makes before it attends.
    pub(crate) fn write_paged(
        &self,
        k: &Tensor,
        v: &Tensor,
        cache: &(Tensor, Tensor),
        meta: &PagedAttentionInputMetadata,
    ) -> Result<()> {
        let Some(paged) = &self.paged_attn else {
            hanzo_ml::bail!("a paged cache write from a block without paged attention");
        };
        paged.write(k, v, &cache.0, &cache.1, meta)
    }

    /// This block's MRoPE cos/sin for `[3, batch, seq]` position ids.
    pub(crate) fn rotary_cos_sin(
        &self,
        positions: &Tensor,
        dtype: DType,
    ) -> Result<(Tensor, Tensor)> {
        self.rotary.compute_cos_sin(positions, dtype)
    }

    /// Project, attend, then gate and project out.
    pub(crate) fn forward(
        &self,
        x: &Tensor,
        mask: &AttentionMask,
        cos_sin: &(Tensor, Tensor),
        kv_cache: &mut KvCache,
        paged: Option<((Tensor, Tensor), &PagedAttentionInputMetadata)>,
    ) -> Result<Tensor> {
        let (q, k, v, gate) = self.project(x, cos_sin)?;
        let y = prof::timed(x.device(), &prof::FULL_ATTN_NS, || {
            self.attend(&q, &k, &v, mask, kv_cache, paged)
        })?;
        self.output(&y, &gate, x.dtype())
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn forward_with_q8(
        &self,
        q8: &hanzo_ml::quantized::IqQ8Activation,
        out_dtype: DType,
        b_sz: usize,
        seq_len: usize,
        dev: &Device,
        mask: &AttentionMask,
        cos_sin: &(Tensor, Tensor),
        kv_cache: &mut KvCache,
        paged: Option<((Tensor, Tensor), &PagedAttentionInputMetadata)>,
    ) -> Result<Tensor> {
        let (q_gate, k, v) = prof::timed(dev, &prof::QPROJ_ATTN_NS, || {
            let q_gate = self.attn_q.forward_iq_q8_dtype(q8, out_dtype)?;
            let k = self.attn_k.forward_iq_q8_dtype(q8, out_dtype)?;
            let v = self.attn_v.forward_iq_q8_dtype(q8, out_dtype)?;
            Ok((q_gate, k, v))
        })?;
        let (q, k, v, gate) =
            self.split_norm_rope_qkv(q_gate, k, v, b_sz, seq_len, dev, cos_sin)?;
        let y = prof::timed(dev, &prof::FULL_ATTN_NS, || {
            self.attend(&q, &k, &v, mask, kv_cache, paged)
        })?;
        self.output(&y, &gate, out_dtype)
    }

    #[allow(clippy::too_many_arguments)]
    fn split_norm_rope_qkv(
        &self,
        q_gate: Tensor,
        k: Tensor,
        v: Tensor,
        b_sz: usize,
        seq_len: usize,
        dev: &Device,
        cos_sin: &(Tensor, Tensor),
    ) -> Result<(Tensor, Tensor, Tensor, Tensor)> {
        prof::timed(dev, &prof::FULL_ATTN_NS, || {
            let q_gate = q_gate.reshape((b_sz, seq_len, self.n_head, self.head_dim * 2))?;
            let q = q_gate.narrow(D::Minus1, 0, self.head_dim)?;
            let gate = q_gate.narrow(D::Minus1, self.head_dim, self.head_dim)?;
            let gate = gate.reshape((b_sz, seq_len, self.n_head * self.head_dim))?;

            let (q, k, v) = if seq_len != 1 {
                let q = q.transpose(1, 2)?;
                let k = k
                    .reshape((b_sz, seq_len, self.n_kv_head, self.head_dim))?
                    .transpose(1, 2)?;
                let v = v
                    .reshape((b_sz, seq_len, self.n_kv_head, self.head_dim))?
                    .transpose(1, 2)?;
                (q, k, v)
            } else {
                let q = q.reshape((b_sz, self.n_head, seq_len, self.head_dim))?;
                let k = k.reshape((b_sz, self.n_kv_head, seq_len, self.head_dim))?;
                let v = v.reshape((b_sz, self.n_kv_head, seq_len, self.head_dim))?;
                (q, k, v)
            };

            let (q, k) = self.rotary.forward_qk_norm_served(
                cos_sin,
                &q.contiguous()?,
                &k.contiguous()?,
                self.q_norm.weight(),
                self.k_norm.weight(),
                self.q_norm.eps(),
            )?;

            Ok((
                q.to_dtype(self.dtype)?,
                k.to_dtype(self.dtype)?,
                v.to_dtype(self.dtype)?,
                gate,
            ))
        })
    }

    /// `(q, k, v, gate)`: q, k and v as `[batch, heads, seq, head_dim]` in the compute dtype, with
    /// q and k normed and roped, and the pre-sigmoid output gate as `[batch, seq, heads·head_dim]`.
    pub(crate) fn project(
        &self,
        x: &Tensor,
        cos_sin: &(Tensor, Tensor),
    ) -> Result<(Tensor, Tensor, Tensor, Tensor)> {
        let (b_sz, seq_len, _) = x.dims3()?;

        if let (Some(ref q_had), true, true) = (
            &self.q_had,
            self.k_had.is_some() && self.v_had.is_some(),
            self.attn_q.supports_shared_iq_q8()
                && self.attn_k.supports_shared_iq_q8()
                && self.attn_v.supports_shared_iq_q8(),
        ) {
            if can_fuse_single_row_iq_q8(x, q_had.block_size()) {
                let q8 = q_had.forward_iq_q8(x, self.presigned_input)?;
                let out_dtype = x.dtype();
                let (q_gate, k, v) = prof::timed(x.device(), &prof::QPROJ_ATTN_NS, || {
                    let q_gate = self.attn_q.forward_iq_q8_dtype(&q8, out_dtype)?;
                    let k = self.attn_k.forward_iq_q8_dtype(&q8, out_dtype)?;
                    let v = self.attn_v.forward_iq_q8_dtype(&q8, out_dtype)?;
                    Ok((q_gate, k, v))
                })?;
                return self.split_norm_rope_qkv(q_gate, k, v, b_sz, seq_len, x.device(), cos_sin);
            }
        }

        let x_rot = match self
            .q_had
            .as_ref()
            .or(self.k_had.as_ref())
            .or(self.v_had.as_ref())
        {
            Some(had) if self.presigned_input => Some(had.forward_presigned(x)?),
            Some(had) => Some(had.forward(x)?),
            None => None,
        };
        let q_in = if self.q_had.is_some() {
            x_rot.as_ref().unwrap()
        } else {
            x
        };
        let k_in = if self.k_had.is_some() {
            x_rot.as_ref().unwrap()
        } else {
            x
        };
        let v_in = if self.v_had.is_some() {
            x_rot.as_ref().unwrap()
        } else {
            x
        };

        let (q_gate, k, v) = prof::timed(x.device(), &prof::QPROJ_ATTN_NS, || {
            let q_gate = self.attn_q.forward(q_in)?;
            let k = self.attn_k.forward(k_in)?;
            let v = self.attn_v.forward(v_in)?;
            Ok((q_gate, k, v))
        })?;

        self.split_norm_rope_qkv(q_gate, k, v, b_sz, seq_len, x.device(), cos_sin)
    }

    /// `project`'s q attended over this block's cache once k and v are written to it, as
    /// `[batch, seq, heads·head_dim]`.
    pub(crate) fn attend(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        mask: &AttentionMask,
        kv_cache: &mut KvCache,
        paged: Option<((Tensor, Tensor), &PagedAttentionInputMetadata)>,
    ) -> Result<Tensor> {
        let (b_sz, _, seq_len, _) = q.dims4()?;

        // PagedAttention keeps the decode KV write/read position-invariant (device slot_mappings +
        // bucketed context_lens), which is what makes the whole decode forward capturable by a CUDA
        // graph. The plain hybrid `KvCache::append` path (host-offset write, frozen under capture) is
        // kept only for the eager / no-paged (CPU) case. Mirrors quantized_qwen3_moe / qwen3_next.
        let y = match (&self.paged_attn, paged) {
            (Some(paged_attn), Some(((key_cache, value_cache), input_metadata))) => paged_attn
                .forward(
                    q,
                    k,
                    v,
                    mask,
                    Some(key_cache),
                    Some(value_cache),
                    input_metadata,
                    &self.sdpa_params,
                    None,
                )?,
            (Some(paged_attn), None) => {
                let input_metadata = PagedAttentionInputMetadata::dummy(q.device())?;
                paged_attn.forward(
                    q,
                    k,
                    v,
                    mask,
                    None,
                    None,
                    &input_metadata,
                    &self.sdpa_params,
                    None,
                )?
            }
            (None, _) if self.qsa => {
                let (k, v) = kv_cache.append(k, v)?;
                // [b, hq, s, d] -> [b, s, hq·d] below, as SDPA's output
                crate::models::qsa::attend(q, &k, &v)?.transpose(1, 2)?
            }
            (None, _) => {
                let (k, v) = kv_cache.append(k, v)?;
                Sdpa.run_attention(q, &k, &v, mask, None, &self.sdpa_params)?
            }
        };
        if self.qsa && self.paged_attn.is_none() {
            return y.contiguous()?.reshape((b_sz, seq_len, ()));
        }

        if mask.is_custom() {
            y.transpose(1, 2)?.reshape((b_sz, seq_len, ()))
        } else {
            y.reshape((b_sz, seq_len, ()))
        }
    }

    /// `o_proj(y * sigmoid(gate))` in `dtype`, the residual stream's. The product is formed in f32
    /// and handed over unrounded: the served graph fuses it into o_proj's activation quantizer
    /// (`…mul_permute_sigmoid…`), which reads it from registers.
    pub(crate) fn output(&self, y: &Tensor, gate: &Tensor, dtype: DType) -> Result<Tensor> {
        let y = prof::timed(y.device(), &prof::FULL_ATTN_NS, || {
            let gate = sigmoid(&gate.to_dtype(DType::F32)?)?;
            y.to_dtype(DType::F32)?.broadcast_mul(&gate)
        })?;
        if let Some(ref had) = self.o_had {
            if self.attn_o.supports_shared_iq_q8()
                && can_fuse_single_row_iq_q8(&y, had.block_size())
            {
                let q8 = had.forward_iq_q8(&y, false)?;
                return prof::timed(y.device(), &prof::QPROJ_ATTN_NS, || {
                    self.attn_o.forward_iq_q8_dtype(&q8, dtype)
                });
            }
            let y_rot = had.forward(&y)?;
            return prof::timed(y.device(), &prof::QPROJ_ATTN_NS, || {
                self.attn_o.forward(&y_rot)?.to_dtype(dtype)
            });
        }
        prof::timed(y.device(), &prof::QPROJ_ATTN_NS, || {
            self.attn_o.forward(&y)?.to_dtype(dtype)
        })
    }
}

// ===================== Gated DeltaNet (linear-attention) layer =====================

pub(crate) struct QGatedDeltaNet {
    in_proj_qkv: Arc<dyn QuantMethod>, // merged q,k,v (no z) -> attn_qkv
    in_proj_z: Arc<dyn QuantMethod>,   // z gate -> attn_gate
    in_proj_b: Arc<dyn QuantMethod>,   // beta -> ssm_beta
    in_proj_a: Arc<dyn QuantMethod>,   // alpha -> ssm_alpha
    conv1d_weight: Tensor,             // (conv_dim, kernel) f32
    dt_bias: Tensor,                   // (num_v_heads,) f32
    a: Tensor,                         // -exp(A_log), (num_v_heads,) f32, precomputed in GGUF
    norm: RmsNormGated,
    out_proj: Arc<dyn QuantMethod>,
    num_k_heads: usize,
    num_v_heads: usize,
    head_k_dim: usize,
    head_v_dim: usize,
    conv_kernel_size: usize,
    key_dim: usize,
    value_dim: usize,
    qkv_had: Option<Arc<DeviceHadamard>>,
    z_had: Option<Arc<DeviceHadamard>>,
    b_had: Option<Arc<DeviceHadamard>>,
    a_had: Option<Arc<DeviceHadamard>>,
    out_had: Option<Arc<DeviceHadamard>>,
    out_v_grouped: bool,
}

/// Rows produced by the conv state spliced onto the left of a continuation are context, not output.
fn trim_carried(out: &Tensor, carried: usize) -> Result<Tensor> {
    if carried == 0 {
        return Ok(out.clone());
    }
    let len = out.dim(1)?;
    out.narrow(1, carried, len - carried)
}

impl QGatedDeltaNet {
    /// The GDN block at `prefix`. Its output norm gates with silu(z), the default when the config
    /// sets no `output_gate_type`.
    pub(crate) fn load<R: std::io::Seek + std::io::Read>(
        ct: &mut Content<'_, R>,
        prefix: &str,
        props: &PropsGGUF,
        dev: &Device,
    ) -> Result<Self> {
        let in_proj_qkv = gguf_qmm(ct.tensor(&format!("{prefix}.attn_qkv.weight"), dev)?)?;
        let in_proj_z = gguf_qmm(ct.tensor(&format!("{prefix}.attn_gate.weight"), dev)?)?;
        let in_proj_b = gguf_qmm(ct.tensor(&format!("{prefix}.ssm_beta.weight"), dev)?)?;
        let in_proj_a = gguf_qmm(ct.tensor(&format!("{prefix}.ssm_alpha.weight"), dev)?)?;
        let out_proj = gguf_qmm(ct.tensor(&format!("{prefix}.ssm_out.weight"), dev)?)?;

        // conv1d / dt / a are small f32 params kept dequantized.
        let mut conv1d_weight = ct
            .tensor(&format!("{prefix}.ssm_conv1d.weight"), dev)?
            .dequantize(dev)?;
        // GGUF squeezes conv1d to 2D (conv_dim, kernel); ensure 2D.
        if conv1d_weight.rank() == 3 {
            conv1d_weight = conv1d_weight.squeeze(1)?;
        }
        // GGUF conversions name this `ssm_dt.bias` or `ssm_dt`; accept both.
        let dt_bias = ct
            .tensor(&format!("{prefix}.ssm_dt.bias"), dev)
            .or_else(|_| ct.tensor(&format!("{prefix}.ssm_dt"), dev))?
            .dequantize(dev)?
            .to_dtype(DType::F32)?;
        let a = ct
            .tensor(&format!("{prefix}.ssm_a"), dev)?
            .dequantize(dev)?
            .to_dtype(DType::F32)?;

        let ssm_norm_w = ct
            .tensor(&format!("{prefix}.ssm_norm.weight"), dev)?
            .dequantize(dev)?;
        let norm = RmsNormGated::from_weight(ssm_norm_w, props.rms_norm_eps as f64);

        Ok(Self {
            in_proj_qkv,
            in_proj_z,
            in_proj_b,
            in_proj_a,
            conv1d_weight,
            dt_bias,
            a,
            norm,
            out_proj,
            num_k_heads: props.num_k_heads,
            num_v_heads: props.num_v_heads,
            head_k_dim: props.head_k_dim,
            head_v_dim: props.head_v_dim,
            conv_kernel_size: props.conv_kernel,
            key_dim: props.num_k_heads * props.head_k_dim,
            value_dim: props.num_v_heads * props.head_v_dim,
            qkv_had: None,
            z_had: None,
            b_had: None,
            a_had: None,
            out_had: None,
            out_v_grouped: false,
        })
    }

    pub(crate) fn with_hadamard(
        mut self,
        had: &Arc<DeviceHadamard>,
        meta: &HadamardMeta,
        prefix: &str,
    ) -> Self {
        let pick = |suffix: &str| {
            meta.is_folded(&format!("{prefix}.{suffix}"))
                .then(|| Arc::clone(had))
        };
        self.qkv_had = pick("attn_qkv.weight");
        self.z_had = pick("attn_gate.weight");
        self.b_had = pick("ssm_beta.weight");
        self.a_had = pick("ssm_alpha.weight");
        self.out_had = pick("ssm_out.weight");
        self.out_v_grouped = self.out_had.is_some() && meta.gdn_v_grouped;
        self
    }

    /// The GDN block at `vb` (`...linear_attn`) of a safetensors checkpoint, in the tiled V-head
    /// order the recurrence consumes.
    ///
    /// The checkpoint keeps V heads grouped by key head (`[K0_v0..v{g-1}, K1_v0.., ...]`); tiled
    /// order puts tiled head `r·K + k` where grouped head `k·g + r` was (π(r·K+k) = k·g+r, K key
    /// heads, g V heads per key head), so V head j reads key head j % K as the q/k tiling does.
    /// The permutation moves whole 128-row heads, which are whole FP8 blocks: it applies to the V
    /// rows of `in_proj_qkv` (rows ≥ 2·key_dim) and all rows of `in_proj_z` with their
    /// `weight_scale_inv` rows, to `out_proj`'s columns and scale columns, to `in_proj_a`/`b`
    /// rows, the conv's V channels, `A_log` and `dt_bias`. `norm` is used as-is (no `+1`).
    pub(crate) fn new(
        vb: &hanzo_quant::ShardedVarBuilder,
        props: &PropsGGUF,
        quant: &Option<hanzo_quant::QuantizedConfig>,
    ) -> Result<Self> {
        let dev = vb.device().clone();
        let host = vb.clone().set_device(Device::Cpu);
        let (kh, vh, dk, dv) = (
            props.num_k_heads,
            props.num_v_heads,
            props.head_k_dim,
            props.head_v_dim,
        );
        let (key_dim, value_dim, h) = (kh * dk, vh * dv, props.embedding_length);
        let g = vh / kh;
        // tiled head p reads grouped head perm[p]
        let perm: Vec<usize> = (0..vh).map(|p| (p % kh) * g + p / kh).collect();
        let heads = |t: &Tensor, dim: usize, start: usize, width: usize| -> Result<Tensor> {
            let parts = perm
                .iter()
                .map(|&j| t.narrow(dim, start + j * width, width))
                .collect::<Result<Vec<_>>>()?;
            Tensor::cat(&parts, dim)
        };
        let rows = |name: &str| host.get_unchecked_dtype(name, DType::F32);
        // A projection with its V heads moved to tiled order: rows (dim 0) from `start` for the
        // input projections, columns (dim 1) for out_proj. Block FP8 when the checkpoint carries
        // `weight_scale_inv` (one scale row or column per 128-wide head), else a float linear.
        let proj = |name: &str, dim: usize, start: usize| -> Result<Arc<dyn QuantMethod>> {
            let permute = |t: &Tensor, width: usize, start: usize| -> Result<Tensor> {
                let len = t.dim(dim)?;
                let lead = t.narrow(dim, 0, start)?;
                let moved = heads(t, dim, start, width)?;
                if start == 0 {
                    Ok(moved)
                } else {
                    debug_assert_eq!(start + vh * width, len);
                    Tensor::cat(&[lead, moved], dim)
                }
            };
            if host.contains_tensor(&format!("{name}.weight_scale_inv")) {
                if dv != 128 || start % 128 != 0 {
                    hanzo_ml::bail!("{name}: tiling block-FP8 heads needs 128-wide V heads");
                }
                let w = host.get_unchecked_dtype(&format!("{name}.weight"), DType::F8E4M3)?;
                let sc = rows(&format!("{name}.weight_scale_inv"))?;
                hanzo_quant::blockwise_fp8_moe(
                    permute(&w, dv, start)?.to_device(&dev)?,
                    permute(&sc, 1, start / 128)?.to_device(&dev)?,
                    vec![128, 128],
                    DType::BF16,
                )
            } else {
                let w = host.get_unchecked_dtype(&format!("{name}.weight"), vb.dtype())?;
                let w = permute(&w, dv, start)?.to_device(&dev)?;
                Ok(Arc::new(hanzo_quant::UnquantLinear::new(
                    QuantMethodConfig::Unquantized(hanzo_nn::Linear::new(w, None)),
                )?))
            }
        };
        let _ = (quant, h);
        let (in_proj_qkv, in_proj_z, out_proj) = (
            proj("in_proj_qkv", 0, 2 * key_dim)?,
            proj("in_proj_z", 0, 0)?,
            proj("out_proj", 1, 0)?,
        );
        // in_proj_a / in_proj_b run on the f32-lifted input: f32 weights holding the bf16 values
        let f32_lin = |name: &str| -> Result<Arc<dyn QuantMethod>> {
            let w = heads(&rows(&format!("{name}.weight"))?, 0, 0, 1)?.to_device(&dev)?;
            Ok(Arc::new(hanzo_quant::UnquantLinear::new(
                QuantMethodConfig::Unquantized(hanzo_nn::Linear::new(w, None)),
            )?))
        };
        let conv = rows("conv1d.weight")?.squeeze(1)?;
        let conv = Tensor::cat(
            &[
                conv.narrow(0, 0, 2 * key_dim)?,
                heads(&conv, 0, 2 * key_dim, dv)?,
            ],
            0,
        )?;
        let a_log = heads(&rows("A_log")?, 0, 0, 1)?;
        Ok(Self {
            in_proj_qkv,
            in_proj_z,
            in_proj_b: f32_lin("in_proj_b")?,
            in_proj_a: f32_lin("in_proj_a")?,
            conv1d_weight: conv.to_device(&dev)?,
            dt_bias: heads(&rows("dt_bias")?, 0, 0, 1)?.to_device(&dev)?,
            a: a_log.exp()?.neg()?.to_device(&dev)?,
            norm: RmsNormGated::from_weight(
                rows("norm.weight")?.to_device(&dev)?,
                props.rms_norm_eps as f64,
            )
            .sigmoid(),
            out_proj,
            num_k_heads: kh,
            num_v_heads: vh,
            head_k_dim: dk,
            head_v_dim: dv,
            conv_kernel_size: props.conv_kernel,
            key_dim,
            value_dim,
            qkv_had: None,
            z_had: None,
            b_had: None,
            a_had: None,
            out_had: None,
            out_v_grouped: false,
        })
    }

    /// Gate the output norm with sigmoid(z), `output_gate_type = "sigmoid"`.
    pub(crate) fn sigmoid(self) -> Self {
        Self {
            norm: self.norm.sigmoid(),
            ..self
        }
    }

    /// The block over `x` `[b, s, h]`. Arithmetic is f32, rounded to `x`'s dtype at the six points
    /// the served prefill path stores: the projection outputs (qkv and z, then b and a),
    /// the causal conv + silu, the l2-normed q and k (and v), the recurrence output, and the gated
    /// norm. g, beta and the state stay f32. With f32 input every rounding is a no-op.
    pub(crate) fn forward(&self, x: &Tensor, cache: &mut GdnLayerCache) -> Result<Tensor> {
        self.forward_probed(x, cache, None)
    }

    /// [`Self::forward`], noting each stored stage in `probe` when given (for the goldens).
    pub(crate) fn forward_probed(
        &self,
        x: &Tensor,
        cache: &mut GdnLayerCache,
        mut probe: Option<&mut Vec<(&'static str, Tensor)>>,
    ) -> Result<Tensor> {
        let is_probed = probe.is_some();
        let mut note = |name: &'static str, t: &Tensor| {
            if let Some(p) = probe.as_mut() {
                p.push((name, t.clone()));
            }
        };
        let rd = x.dtype();
        let fast_cuda = !is_probed && x.device().is_cuda();
        let store = |t: Tensor| -> Result<Tensor> {
            if fast_cuda {
                if t.dtype() == DType::F32 {
                    Ok(t)
                } else {
                    t.to_dtype(DType::F32)
                }
            } else {
                t.to_dtype(rd)?.to_dtype(DType::F32)
            }
        };
        let x_f32_buf;
        let x: &Tensor = if fast_cuda {
            x
        } else {
            x_f32_buf = x.to_dtype(DType::F32)?;
            &x_f32_buf
        };
        let (batch_size, seq_len, _hidden) = x.dims3()?;
        let v_per_group = self.num_v_heads / self.num_k_heads;

        // 1-2. Projections: qkv is merged [q | k | v]; z is the gate; b and a are bf16 GEMMs.
        let proj_out_dt = if fast_cuda { DType::F32 } else { rd };
        let (mixed_qkv, z, b, a) = if let (Some(ref qkv_had), true, true) = (
            &self.qkv_had,
            self.z_had.is_some() && self.b_had.is_none() && self.a_had.is_none(),
            self.in_proj_qkv.supports_shared_iq_q8()
                && self.in_proj_z.supports_shared_iq_q8()
                && can_fuse_single_row_iq_q8(x, self.qkv_had.as_ref().unwrap().block_size()),
        ) {
            let q8 = qkv_had.forward_iq_q8(x, false)?;
            prof::timed(x.device(), &prof::QPROJ_GDN_NS, || {
                let mixed_qkv = store(self.in_proj_qkv.forward_iq_q8_dtype(&q8, proj_out_dt)?)?;
                let z = store(self.in_proj_z.forward_iq_q8_dtype(&q8, proj_out_dt)?)?;
                let b = store(self.in_proj_b.forward(x)?)?;
                let a = store(self.in_proj_a.forward(x)?)?;
                Ok((mixed_qkv, z, b, a))
            })?
        } else {
            let x_rot = match self
                .qkv_had
                .as_ref()
                .or(self.z_had.as_ref())
                .or(self.b_had.as_ref())
                .or(self.a_had.as_ref())
            {
                Some(had) => Some(had.forward(x)?),
                None => None,
            };
            let qkv_in = if self.qkv_had.is_some() {
                x_rot.as_ref().unwrap()
            } else {
                x
            };
            let z_in = if self.z_had.is_some() {
                x_rot.as_ref().unwrap()
            } else {
                x
            };
            let b_in = if self.b_had.is_some() {
                x_rot.as_ref().unwrap()
            } else {
                x
            };
            let a_in = if self.a_had.is_some() {
                x_rot.as_ref().unwrap()
            } else {
                x
            };
            prof::timed(x.device(), &prof::QPROJ_GDN_NS, || {
                let mixed_qkv = store(self.in_proj_qkv.forward(qkv_in)?)?;
                let z = store(self.in_proj_z.forward(z_in)?)?;
                let b = store(self.in_proj_b.forward(b_in)?)?;
                let a = store(self.in_proj_a.forward(a_in)?)?;
                Ok((mixed_qkv, z, b, a))
            })?
        };
        let prof_gdn_t0 = if prof::enabled() {
            x.device().synchronize()?;
            Some(std::time::Instant::now())
        } else {
            None
        };

        let z = z.reshape((batch_size, seq_len, self.num_v_heads, self.head_v_dim))?;
        let b = b.reshape((batch_size, seq_len, self.num_v_heads))?;
        let a = a.reshape((batch_size, seq_len, self.num_v_heads))?;

        cache.trail_conv(&mixed_qkv)?;

        // 3. Causal conv1d over the concatenated qkv, then silu.
        let mixed_qkv = if fast_cuda || rd == DType::F32 {
            self.conv(&mixed_qkv, cache)?
        } else {
            self.causal_conv1d_served(&mixed_qkv, cache, rd)?
        };
        let mixed_qkv = store(mixed_qkv)?;
        note("conv", &mixed_qkv);

        #[cfg(feature = "rocm")]
        let use_rocm_mixer = mixed_qkv.device().is_rocm()
            && cache.trail.is_none()
            && self.head_k_dim == 128
            && self.head_v_dim == 128;
        #[cfg(not(feature = "rocm"))]
        let use_rocm_mixer = false;

        let y = if use_rocm_mixer {
            #[cfg(feature = "rocm")]
            {
                let f32s = |t: &Tensor| t.to_dtype(DType::F32)?.contiguous();
                let y = hanzo_ml::rocm_backend::gdn_mixer_rocm(
                    &mixed_qkv,
                    &f32s(&a)?,
                    &f32s(&b)?,
                    &f32s(&z.reshape((batch_size, seq_len, self.value_dim))?)?,
                    &self.a,
                    &self.dt_bias,
                    &self.norm.weight,
                    &cache.recurrent_state,
                    self.num_k_heads,
                    self.norm.eps() as f32,
                    self.norm.gates_with_sigmoid(),
                )?;
                cache.seqlen_offset += seq_len;
                y
            }
            #[cfg(not(feature = "rocm"))]
            {
                unreachable!()
            }
        } else {
            let q = mixed_qkv.narrow(D::Minus1, 0, self.key_dim)?;
            let k = mixed_qkv.narrow(D::Minus1, self.key_dim, self.key_dim)?;
            let v = mixed_qkv.narrow(D::Minus1, self.key_dim * 2, self.value_dim)?;
            let q = q.reshape((batch_size, seq_len, self.num_k_heads, self.head_k_dim))?;
            let k = k.reshape((batch_size, seq_len, self.num_k_heads, self.head_k_dim))?;
            let v = v.reshape((batch_size, seq_len, self.num_v_heads, self.head_v_dim))?;

            let (beta, g) = {
                #[cfg(feature = "cuda")]
                {
                    if fast_cuda {
                        let b_flat = b.contiguous()?.flatten_all()?;
                        let a_flat = a.contiguous()?.flatten_all()?;
                        let (beta_flat, g_flat) = crate::cuda::gdn::fused_gdn_gating_cuda(
                            &b_flat,
                            &a_flat,
                            &self.a,
                            &self.dt_bias,
                        )?;
                        let shape = b.shape();
                        (beta_flat.reshape(shape)?, g_flat.reshape(shape)?)
                    } else {
                        let beta = sigmoid(&b)?;
                        let dt_bias = self.dt_bias.unsqueeze(0)?.unsqueeze(0)?;
                        let xa = a.broadcast_add(&dt_bias)?;
                        let sp = {
                            let pos = (xa.clone() + (xa.neg()?.exp()? + 1.0)?.log()?)?;
                            let neg = (xa.exp()? + 1.0)?.log()?;
                            let zero = xa.zeros_like()?;
                            let sp = xa.gt(&zero)?.where_cond(&pos, &neg)?;
                            xa.le(20.0)?.where_cond(&sp, &xa)?
                        };
                        let g = self.a.unsqueeze(0)?.unsqueeze(0)?.broadcast_mul(&sp)?;
                        (beta, g)
                    }
                }
                #[cfg(not(feature = "cuda"))]
                {
                    let beta = sigmoid(&b)?;
                    let dt_bias = self.dt_bias.unsqueeze(0)?.unsqueeze(0)?;
                    let xa = a.broadcast_add(&dt_bias)?;
                    let sp = {
                        let pos = (xa.clone() + (xa.neg()?.exp()? + 1.0)?.log()?)?;
                        let neg = (xa.exp()? + 1.0)?.log()?;
                        let zero = xa.zeros_like()?;
                        let sp = xa.gt(&zero)?.where_cond(&pos, &neg)?;
                        xa.le(20.0)?.where_cond(&sp, &xa)?
                    };
                    let g = self.a.unsqueeze(0)?.unsqueeze(0)?.broadcast_mul(&sp)?;
                    (beta, g)
                }
            };
            note("g", &g);
            note("beta", &beta);

            let (q, k) = if fast_cuda {
                let q = l2_norm(&q, L2_NORM_EPS)?;
                let k = l2_norm(&k, L2_NORM_EPS)?;
                if v_per_group > 1 {
                    let tile = |t: &Tensor| {
                        t.unsqueeze(2)?.repeat((1, 1, v_per_group, 1, 1))?.reshape((
                            batch_size,
                            seq_len,
                            self.num_v_heads,
                            self.head_k_dim,
                        ))
                    };
                    (tile(&q)?, tile(&k)?)
                } else {
                    (q, k)
                }
            } else {
                let (q, k) = if v_per_group > 1 {
                    let tile = |t: &Tensor| {
                        t.unsqueeze(2)?.repeat((1, 1, v_per_group, 1, 1))?.reshape((
                            batch_size,
                            seq_len,
                            self.num_v_heads,
                            self.head_k_dim,
                        ))
                    };
                    (tile(&q)?, tile(&k)?)
                } else {
                    (q, k)
                };
                (
                    store(l2_norm(&q, L2_NORM_EPS)?)?,
                    store(l2_norm(&k, L2_NORM_EPS)?)?,
                )
            };
            note("q", &q);
            note("k", &k);
            note("v", &v);

            let y = store(cache.recurrence(&q, &k, &v, &g, &beta)?)?;
            note("core", &y);
            cache.seqlen_offset += seq_len;

            let z_shape = z.shape().clone();
            let y = y.reshape(((), self.head_v_dim))?;
            let z = z.reshape(((), self.head_v_dim))?;
            let y = store(self.norm.forward(&y, &z)?)?;
            let y = y
                .reshape(z_shape)?
                .reshape((batch_size, seq_len, self.value_dim))?;
            note("normed", &y);
            y
        };

        // When `ssm_out.weight` was Hadamard-folded in HF grouped V-head order (`gdn_v_grouped`),
        // transpose tiled `[batch, seq, v_per_group, num_k_heads, head_v_dim]` back to grouped
        // `[batch, seq, num_k_heads, v_per_group, head_v_dim]` before `signs` + `H_B`.
        let y = if self.out_v_grouped && v_per_group > 1 {
            y.reshape((
                batch_size,
                seq_len,
                v_per_group,
                self.num_k_heads,
                self.head_v_dim,
            ))?
            .transpose(2, 3)?
            .contiguous()?
            .reshape((batch_size, seq_len, self.value_dim))?
        } else {
            y
        };
        if let Some(t0) = prof_gdn_t0 {
            x.device().synchronize()?;
            prof::GDN_RECURR_NS.fetch_add(
                t0.elapsed().as_nanos() as u64,
                std::sync::atomic::Ordering::Relaxed,
            );
        }
        if let Some(ref had) = self.out_had {
            if self.out_proj.supports_shared_iq_q8()
                && can_fuse_single_row_iq_q8(&y, had.block_size())
            {
                let q8 = had.forward_iq_q8(&y, false)?;
                return prof::timed(x.device(), &prof::QPROJ_GDN_NS, || {
                    self.out_proj.forward_iq_q8_dtype(&q8, rd)
                });
            }
            let y_rot = had.forward(&y)?;
            return prof::timed(x.device(), &prof::QPROJ_GDN_NS, || {
                self.out_proj.forward(&y_rot)?.to_dtype(rd)
            });
        }

        prof::timed(x.device(), &prof::QPROJ_GDN_NS, || {
            self.out_proj.forward(&y)?.to_dtype(rd)
        })
    }

    /// The served causal conv: each tap's product rounded to `rd` (the
    /// kernel multiplies two bf16 tensors), accumulated in f32 over the four taps in order, then
    /// silu in f32. The conv state (the last `kernel` inputs) advances as in the other paths.
    fn causal_conv1d_served(
        &self,
        x: &Tensor,
        cache: &mut GdnLayerCache,
        rd: DType,
    ) -> Result<Tensor> {
        let (_, seq_len, _) = x.dims3()?;
        let kw = self.conv_kernel_size;
        let x_t = x.transpose(1, 2)?.contiguous()?;
        let left = if cache.seqlen_offset > 0 {
            cache
                .conv_state
                .narrow(D::Minus1, 1, kw - 1)?
                .to_dtype(DType::F32)?
        } else {
            let (b, c, _) = cache.conv_state.dims3()?;
            Tensor::zeros((b, c, kw - 1), DType::F32, x.device())?
        };
        let hist = Tensor::cat(&[&left, &x_t], D::Minus1)?.contiguous()?;
        let total = hist.dim(D::Minus1)?;
        cache.conv_state = hist
            .narrow(D::Minus1, total - kw, kw)?
            .to_dtype(cache.conv_state.dtype())?;
        let weight = self.conv1d_weight.to_dtype(DType::F32)?;
        let mut acc: Option<Tensor> = None;
        for j in 0..kw {
            let tap = hist
                .narrow(D::Minus1, j, seq_len)?
                .broadcast_mul(&weight.narrow(1, j, 1)?.unsqueeze(0)?)?
                .to_dtype(rd)?
                .to_dtype(DType::F32)?;
            acc = Some(match acc {
                None => tap,
                Some(a) => (a + tap)?,
            });
        }
        let acc = acc.expect("kernel width >= 1");
        let out = (acc.clone() / (acc.neg()?.exp()? + 1.0)?)?;
        out.transpose(1, 2)
    }

    /// The causal conv1d with silu over the projected q|k|v, advancing the conv state. On ROCm one
    /// launch serves prefill and decode, the state rewritten in place.
    fn conv(&self, x: &Tensor, cache: &mut GdnLayerCache) -> Result<Tensor> {
        #[cfg(feature = "rocm")]
        if x.device().is_rocm() && self.conv_kernel_size == 4 {
            return hanzo_ml::rocm_backend::gdn_conv_rocm(
                &x.contiguous()?,
                &self.conv1d_weight,
                &cache.conv_state,
                cache.seqlen_offset == 0,
            );
        }
        if cache.seqlen_offset > 0 && x.dim(1)? == 1 {
            self.causal_conv1d_update(x, cache)
        } else {
            self.causal_conv1d_full(x, cache)
        }
    }

    fn causal_conv1d_update(&self, x: &Tensor, cache: &mut GdnLayerCache) -> Result<Tensor> {
        let (_batch, seq_len, _conv_dim) = x.dims3()?;

        #[cfg(feature = "cuda")]
        if x.device().is_cuda() && seq_len == 1 {
            let cs_dtype = cache.conv_state.dtype();
            let x_t = x.to_dtype(cs_dtype)?.transpose(1, 2)?.contiguous()?;
            let weight = self.conv1d_weight.to_dtype(cs_dtype)?.contiguous()?;
            let (output, new_conv_state) = crate::cuda::gdn::causal_conv1d_cuda(
                &x_t,
                &weight,
                &cache.conv_state,
                self.conv_kernel_size,
                true,
            )?;
            cache.conv_state = new_conv_state;
            return output.transpose(1, 2);
        }

        let x_t = x.transpose(1, 2)?.contiguous()?;

        let state_len = cache.conv_state.dim(2)?;
        let conv_state = cache.conv_state.to_dtype(x_t.dtype())?;
        let hidden_new = Tensor::cat(&[conv_state, x_t], 2)?;
        let new_len = hidden_new.dim(2)?;
        cache.conv_state = hidden_new.narrow(2, new_len - state_len, state_len)?;

        let weight = self.conv1d_weight.to_dtype(hidden_new.dtype())?;
        let mut conv_outputs = Vec::with_capacity(seq_len);
        let total_len = hidden_new.dim(2)?;
        for i in (total_len - seq_len)..total_len {
            let window =
                hidden_new.narrow(2, i + 1 - self.conv_kernel_size, self.conv_kernel_size)?;
            let out = window
                .broadcast_mul(&weight.unsqueeze(0)?)?
                .sum(D::Minus1)?;
            conv_outputs.push(out);
        }
        let out = Tensor::stack(&conv_outputs, 2)?;
        let out = hanzo_nn::ops::silu(&out)?;
        out.transpose(1, 2)
    }

    // Single decode step (seq_len==1, batch==1) of the causal conv1d on Vulkan. conv_state is
    // (1, conv_dim, k) -- it stores k columns; the step drops the oldest and appends x, exactly as
    // the CPU causal_conv1d_update does. conv_state is updated in place in VRAM (aliases the pool
    // buffer); x is (1, 1, conv_dim). Returns silu(conv) as (1, 1, conv_dim). The GGUF conv1d_weight
    // is (conv_dim, k) with no bias.
    #[allow(dead_code)]
    fn causal_conv1d_update_vulkan(&self, x: &Tensor, cache: &mut GdnLayerCache) -> Result<Tensor> {
        let conv_dim = self.conv1d_weight.dim(0)?;
        let x_flat = x.reshape(conv_dim)?.to_dtype(DType::F32)?.contiguous()?;
        let weight = self.conv1d_weight.to_dtype(DType::F32)?.contiguous()?;
        let mut conv_state = cache
            .conv_state
            .reshape((conv_dim, self.conv_kernel_size))?;
        let out = crate::vulkan::gdn::gdn_conv1d_step_vulkan(&mut conv_state, &x_flat, &weight)?;
        cache.conv_state = conv_state.reshape((1, conv_dim, self.conv_kernel_size))?;
        out.reshape((1, 1, conv_dim))
    }

    fn causal_conv1d_full(&self, x: &Tensor, cache: &mut GdnLayerCache) -> Result<Tensor> {
        let (batch_size, seq_len, conv_dim) = x.dims3()?;
        // The full kernel has no conv_state argument and zero-pads its left edge, which is right
        // only at the start of a sequence. A continuation (a later prefill chunk, or a speculative
        // replay) must see the previous tokens, so splice them on and drop their outputs after.
        let carried = if cache.seqlen_offset > 0 {
            self.conv_kernel_size - 1
        } else {
            0
        };
        let x_t = if carried > 0 {
            let left = cache
                .conv_state
                .narrow(D::Minus1, 1, carried)?
                .to_dtype(x.dtype())?;
            Tensor::cat(&[&left, &x.transpose(1, 2)?], D::Minus1)?.contiguous()?
        } else {
            x.transpose(1, 2)?.contiguous()?
        };
        let seq_len = seq_len + carried;

        #[cfg(feature = "cuda")]
        if x_t.device().is_cuda() {
            let cs_dtype = cache.conv_state.dtype();
            let x_cs = if x_t.dtype() == cs_dtype {
                x_t
            } else {
                x_t.to_dtype(cs_dtype)?.contiguous()?
            };
            let weight = self.conv1d_weight.to_dtype(cs_dtype)?.contiguous()?;
            let (output, new_conv_state) = crate::cuda::gdn::causal_conv1d_cuda(
                &x_cs,
                &weight,
                &cache.conv_state,
                self.conv_kernel_size,
                false,
            )?;
            cache.conv_state = new_conv_state;
            return trim_carried(&output.transpose(1, 2)?, carried);
        }

        let pad_width = self.conv_kernel_size.saturating_sub(seq_len);
        cache.conv_state = if pad_width > 0 {
            let zeros =
                Tensor::zeros((batch_size, conv_dim, pad_width), x_t.dtype(), x_t.device())?;
            Tensor::cat(&[zeros, x_t.clone()], 2)?
        } else {
            x_t.narrow(2, seq_len - self.conv_kernel_size, self.conv_kernel_size)?
        };

        let padded_t = Tensor::cat(
            &[
                Tensor::zeros(
                    (batch_size, conv_dim, self.conv_kernel_size - 1),
                    x_t.dtype(),
                    x_t.device(),
                )?,
                x_t,
            ],
            2,
        )?;

        let weight = self.conv1d_weight.to_dtype(padded_t.dtype())?;
        let mut conv_outputs = Vec::with_capacity(seq_len);
        for i in 0..seq_len {
            let window = padded_t.narrow(2, i, self.conv_kernel_size)?;
            let out = window
                .broadcast_mul(&weight.unsqueeze(0)?)?
                .sum(D::Minus1)?;
            conv_outputs.push(out);
        }
        let out = Tensor::stack(&conv_outputs, 2)?;
        let out = hanzo_nn::ops::silu(&out)?;
        trim_carried(&out.transpose(1, 2)?, carried)
    }
}

// ===================== Decoder layer =====================

pub(crate) enum LayerImpl {
    FullAttention(GatedFullAttention),
    LinearAttention(QGatedDeltaNet),
}

pub(crate) struct DecoderLayer {
    pub(crate) layer_impl: LayerImpl,
    pub(crate) input_layernorm: QRmsNorm,
    pub(crate) post_attention_layernorm: QRmsNorm,
    pub(crate) mlp: MoeOrMlp,
}

impl DecoderLayer {
    pub(crate) fn with_folded_norm_signs(mut self, hidden_size: usize) -> Result<Self> {
        if let LayerImpl::FullAttention(ref mut attn) = self.layer_impl {
            if let Some(had) = attn.input_hadamard_for_norm_fold().cloned() {
                if let Some(signs) = had.signs_for(hidden_size)? {
                    self.input_layernorm = self.input_layernorm.mul_weight(signs)?;
                }
                attn.set_presigned_input();
            }
        }
        if let MoeOrMlp::Mlp(ref mut dense) = self.mlp {
            if let Some(had) = dense.input_hadamard_for_norm_fold().cloned() {
                if let Some(signs) = had.signs_for(hidden_size)? {
                    self.post_attention_layernorm =
                        self.post_attention_layernorm.mul_weight(signs)?;
                }
                dense.set_presigned_input();
            }
        }
        Ok(self)
    }

    /// This block's MRoPE cos/sin for `[3, batch, seq]` position ids.
    pub(crate) fn rotary_cos_sin(
        &self,
        positions: &Tensor,
        dtype: DType,
    ) -> Result<(Tensor, Tensor)> {
        let LayerImpl::FullAttention(attn) = &self.layer_impl else {
            hanzo_ml::bail!("expected a full-attention block");
        };
        attn.rotary_cos_sin(positions, dtype)
    }

    /// One full-attention block over its own `KvCache`: attention, then the MLP, both residual.
    pub(crate) fn forward_attention(
        &self,
        x: &Tensor,
        mask: &AttentionMask,
        cos_sin: &(Tensor, Tensor),
        kv_cache: &mut KvCache,
    ) -> Result<Tensor> {
        let LayerImpl::FullAttention(attn) = &self.layer_impl else {
            hanzo_ml::bail!("expected a full-attention block");
        };
        let (b_sz, seq_len, _) = x.dims3()?;
        let residual = x.clone();
        let attn_out = if let Some(b) = attn
            .can_fuse_norm_fwht_q8()
            .filter(|&b| can_fuse_single_row_iq_q8(x, b))
        {
            let q8 = prof::timed(x.device(), &prof::HADAMARD_NS, || {
                hanzo_ml::quantized::QMatMul::rms_norm_hadamard_fwht_iq_q8(
                    x,
                    self.input_layernorm.weight(),
                    self.input_layernorm.eps() as f32,
                    None,
                    b,
                )
            })?;
            attn.forward_with_q8(
                &q8,
                x.dtype(),
                b_sz,
                seq_len,
                x.device(),
                mask,
                cos_sin,
                kv_cache,
                None,
            )?
        } else {
            let normed = self.input_layernorm.forward(x)?;
            attn.forward(&normed, mask, cos_sin, kv_cache, None)?
        };
        if let MoeOrMlp::Mlp(ref dense) = self.mlp {
            if let Some(b) = dense
                .can_fuse_norm_fwht_q8()
                .filter(|&b| can_fuse_single_row_iq_q8(&attn_out, b))
            {
                let (x_mid, q8) = prof::timed(x.device(), &prof::HADAMARD_NS, || {
                    hanzo_ml::quantized::QMatMul::add_rms_norm_hadamard_fwht_iq_q8(
                        &attn_out,
                        &residual,
                        self.post_attention_layernorm.weight(),
                        self.post_attention_layernorm.eps() as f32,
                        None,
                        b,
                    )
                })?;
                return dense.forward_iq_q8(&q8, x_mid.dtype())? + x_mid;
            }
        }
        let (x_mid, normed) = self
            .post_attention_layernorm
            .forward_of_sum(&attn_out, &residual)?;
        self.mlp.forward(&normed)? + x_mid
    }
}

// ===================== Config extraction =====================

#[allow(dead_code)]
pub(crate) struct PropsGGUF {
    pub(crate) head_count: usize,
    pub(crate) head_count_kv: usize,
    pub(crate) block_count: usize,
    pub(crate) embedding_length: usize,
    pub(crate) rms_norm_eps: f32,
    pub(crate) max_seq_len: usize,
    pub(crate) rope_freq_base: f32,
    pub(crate) head_dim: usize,
    pub(crate) rot_dim: usize,
    pub(crate) mrope_section: Vec<usize>,
    pub(crate) full_attention_interval: usize,
    // GDN
    pub(crate) conv_kernel: usize,
    pub(crate) head_k_dim: usize,
    pub(crate) head_v_dim: usize,
    pub(crate) num_k_heads: usize,
    pub(crate) num_v_heads: usize,
    // MoE (None for dense)
    pub(crate) num_experts: Option<usize>,
    pub(crate) num_experts_per_tok: usize,
    pub(crate) moe_intermediate_size: usize,
    pub(crate) is_moe: bool,
    /// Trailing multi-token-prediction blocks, excluded from `block_count`; the first sits at
    /// `blk.{block_count}`.
    pub(crate) nextn_predict_layers: usize,
}

/// The file's `general.architecture`, which must be one of `allowed`.
pub(crate) fn verify_arch(
    metadata: &HashMap<String, hanzo_ml::quantized::gguf_file::Value>,
    allowed: &[&str],
) -> Result<String> {
    use crate::utils::gguf_metadata::TryValueInto;
    let actual_arch: String = metadata
        .get("general.architecture")
        .cloned()
        .try_value_into()?;
    if !allowed.contains(&actual_arch.as_str()) {
        let expected = allowed
            .iter()
            .map(|arch| format!("`{arch}`"))
            .collect::<Vec<_>>()
            .join("/");
        hanzo_ml::bail!("Expected {expected} architecture, got `{actual_arch}`.");
    }
    Ok(actual_arch)
}

impl PropsGGUF {
    pub(crate) fn try_from(c: &ContentMetadata, is_moe: bool) -> Result<Self> {
        let required = [
            "attention.head_count",
            "attention.head_count_kv",
            "block_count",
            "embedding_length",
            "attention.layer_norm_rms_epsilon",
            "ssm.conv_kernel",
            "ssm.state_size",
            "ssm.group_count",
            "ssm.time_step_rank",
        ];
        c.has_required_keys(&required)
            .map_err(|e| hanzo_ml::Error::Msg(format!("{e}")))?;

        let embed_len = c
            .get_value::<u32>("embedding_length")
            .map_err(|e| hanzo_ml::Error::Msg(format!("{e}")))? as usize;
        let head_count = c
            .get_value::<u32>("attention.head_count")
            .map_err(|e| hanzo_ml::Error::Msg(format!("{e}")))? as usize;

        let head_dim = c
            .get_value::<u32>("attention.key_length")
            .ok()
            .map(|x| x as usize)
            .unwrap_or(embed_len / head_count);

        // rope.dimension_count is the rotary (partial) dim; fall back to head_dim * 0.25.
        let rot_dim = c
            .get_value::<u32>("rope.dimension_count")
            .ok()
            .map(|x| x as usize)
            .unwrap_or((head_dim as f64 * DEFAULT_PARTIAL_ROTARY_FACTOR) as usize);

        // An INT32 array in the file (the converter writes Python ints).
        let mrope_section = c
            .get_value::<Vec<i32>>("rope.dimension_sections")
            .ok()
            .map(|v| v.into_iter().map(|x| x.max(0) as usize).collect::<Vec<_>>())
            // mrope_section sums to rot_dim/2; the default [t,h,w,0] is [11,11,10,0].
            .unwrap_or_else(|| vec![11, 11, 10, 0]);

        let head_k_dim = c
            .get_value::<u32>("ssm.state_size")
            .map_err(|e| hanzo_ml::Error::Msg(format!("{e}")))? as usize;
        let num_k_heads = c
            .get_value::<u32>("ssm.group_count")
            .map_err(|e| hanzo_ml::Error::Msg(format!("{e}")))? as usize;
        let num_v_heads = c
            .get_value::<u32>("ssm.time_step_rank")
            .map_err(|e| hanzo_ml::Error::Msg(format!("{e}")))? as usize;
        // head_v_dim: ssm.inner_size / num_v_heads, else equal to head_k_dim (state_size).
        let head_v_dim = c
            .get_value::<u32>("ssm.inner_size")
            .ok()
            .map(|x| x as usize / num_v_heads)
            .unwrap_or(head_k_dim);

        let (num_experts, num_experts_per_tok, moe_intermediate_size) = if is_moe {
            (
                Some(
                    c.get_value::<u32>("expert_count")
                        .map_err(|e| hanzo_ml::Error::Msg(format!("{e}")))?
                        as usize,
                ),
                c.get_value::<u32>("expert_used_count")
                    .map_err(|e| hanzo_ml::Error::Msg(format!("{e}")))? as usize,
                c.get_value::<u32>("expert_feed_forward_length")
                    .map_err(|e| hanzo_ml::Error::Msg(format!("{e}")))? as usize,
            )
        } else {
            (None, 0, 0)
        };

        Ok(Self {
            head_count,
            head_count_kv: {
                // hybrid layers store head_count_kv as a per-layer array; take the max (attention layers)
                let key = "attention.head_count_kv";
                c.get_value::<u32>(key)
                    .map(|n| n as usize)
                    .or_else(|_| {
                        c.get_value::<Vec<u32>>(key)
                            .map(|v| v.into_iter().max().unwrap_or(0) as usize)
                    })
                    .map_err(|e| hanzo_ml::Error::Msg(format!("{e}")))?
            },
            // block_count includes the trailing multi-token-prediction blocks, so the transformer
            // depth is block_count - nextn_predict_layers; the head loads from `blk.{depth}` when
            // `--mtp-model` asks for it.
            block_count: (c
                .get_value::<u32>("block_count")
                .map_err(|e| hanzo_ml::Error::Msg(format!("{e}")))?
                .saturating_sub(c.get_value::<u32>("nextn_predict_layers").unwrap_or(0)))
                as usize,
            embedding_length: embed_len,
            rms_norm_eps: c
                .get_value("attention.layer_norm_rms_epsilon")
                .map_err(|e| hanzo_ml::Error::Msg(format!("{e}")))?,
            max_seq_len: c
                .get_value::<u64>("context_length")
                .ok()
                .unwrap_or(DEFAULT_MAX_SEQ_LEN as u64) as usize,
            rope_freq_base: c.get_value("rope.freq_base").ok().unwrap_or(10_000_000_f32),
            head_dim,
            rot_dim,
            mrope_section,
            full_attention_interval: c
                .get_value::<u32>("full_attention_interval")
                .ok()
                .filter(|i| *i > 0)
                .map(|x| x as usize)
                .unwrap_or(DEFAULT_FULL_ATTENTION_INTERVAL),
            conv_kernel: c
                .get_value::<u32>("ssm.conv_kernel")
                .map_err(|e| hanzo_ml::Error::Msg(format!("{e}")))?
                as usize,
            head_k_dim,
            head_v_dim,
            num_k_heads,
            num_v_heads,
            num_experts,
            num_experts_per_tok,
            moe_intermediate_size,
            is_moe,
            nextn_predict_layers: c.get_value::<u32>("nextn_predict_layers").unwrap_or(0) as usize,
        })
    }
}

// ===================== Model =====================

pub struct ModelWeights {
    tok_embeddings: TokEmbedding,
    embed_hadamard: Option<Arc<DeviceHadamard>>,
    hadamard_meta: Option<HadamardMeta>,
    root_hadamard: Option<Arc<DeviceHadamard>>,
    layers: Vec<DecoderLayer>,
    layer_types: Vec<LayerType>,
    norm: QRmsNorm,
    output: Arc<dyn QuantMethod>,
    output_hadamard: Option<Arc<DeviceHadamard>>,
    rotary: Arc<Qwen3VLRotaryEmbedding>,
    pub device: Device,
    pub cache: EitherCache,
    pub max_seq_len: usize,
    mapper: Option<Box<dyn DeviceMapper + Send + Sync>>,
    dtype: DType,
    props: PropsGGUF,
    /// Hidden-state capture for a parallel-block draft (DFlash). Off until a draft names layers.
    pub(crate) spec_capture: crate::speculative::HiddenPrefixCapture,
    /// What the MTP head reads from the last forward. Off by default.
    last_spec: Mutex<Option<SpecRows>>,
    store_spec: AtomicBool,
    /// The MRoPE positions of the target rows the MTP head drafts from, handed over as those
    /// rows are selected.
    mtp_anchors: crate::models::qwen3_5_mtp::AnchorPositions,
}

/// The rows the last forward took its logits from: their final-norm hidden state,
/// `[batch, rows, hidden]`, and the absolute position of each.
pub(crate) struct SpecRows {
    pub(crate) hidden: Tensor,
    pub(crate) positions: Vec<Vec<usize>>,
}

pub(crate) fn gguf_qmm(q: QTensor) -> Result<Arc<dyn QuantMethod>> {
    gguf_qmm_arc(Arc::new(q))
}

fn gguf_qmm_arc(q_weight: Arc<QTensor>) -> Result<Arc<dyn QuantMethod>> {
    Ok(Arc::new(GgufMatMul::new(QuantMethodConfig::Gguf {
        q_weight,
        b: None,
    })?))
}

impl ModelConfig::FromGGUF for ModelWeights {
    fn from_gguf<R: std::io::Seek + std::io::Read>(
        mut ct: Content<'_, R>,
        device: &Device,
        mapper: Box<dyn DeviceMapper + Send + Sync>,
        attention_mechanism: AttentionImplementation,
        dtype: DType,
    ) -> Result<Self> {
        let meta = ct.get_metadata();
        let actual_arch = verify_arch(meta, &["qwen35moe", "qwen35"])?;
        let is_moe = actual_arch == "qwen35moe";
        let hadamard_meta = HadamardMeta::from_gguf_metadata(meta)?;

        let metadata = ContentMetadata {
            path_prefix: &actual_arch,
            metadata: meta,
        };
        let props = PropsGGUF::try_from(&metadata, is_moe)?;

        let key_dim = props.num_k_heads * props.head_k_dim;
        let value_dim = props.num_v_heads * props.head_v_dim;
        let conv_dim = key_dim * 2 + value_dim;

        // GGUF stores V heads in tiled order (converter's _LinearAttentionVReorderBase). The K==V
        // case is a no-op reorder; for V = m*K we consume tiled order directly (QGatedDeltaNet
        // tiles q/k to match). Only a non-integer V/K split is genuinely unsupported.
        if props.num_v_heads % props.num_k_heads != 0 {
            hanzo_ml::bail!(
                "qwen35 GGUF GDN requires num_v_heads ({}) to be a multiple of num_k_heads ({}).",
                props.num_v_heads,
                props.num_k_heads
            );
        }

        let layer_types: Vec<LayerType> = (0..props.block_count)
            .map(|i| {
                if (i + 1) % props.full_attention_interval == 0 {
                    LayerType::FullAttention
                } else {
                    LayerType::LinearAttention
                }
            })
            .collect();

        // Precompute one `DeviceHadamard` per device location when Hadamard metadata is present.
        let mut hadamards = HashMap::new();
        if let Some(ref hm) = hadamard_meta {
            hadamards.insert(
                device.location(),
                Arc::new(DeviceHadamard::new(hm, device)?),
            );
            for layer_idx in 0..props.block_count {
                let dev = mapper.device_for(layer_idx, false).unwrap_or(device);
                if let std::collections::hash_map::Entry::Vacant(e) =
                    hadamards.entry(dev.location())
                {
                    e.insert(Arc::new(DeviceHadamard::new(hm, dev)?));
                }
            }
        }

        let has_output_weight = ct.has_tensor("output.weight");
        let host_qtok_embeddings =
            if !matches!(device, Device::Cpu | Device::Cuda(_) | Device::Metal(_)) {
                Some(Arc::new(ct.tensor("token_embd.weight", &Device::Cpu)?))
            } else {
                None
            };
        let qtok_embeddings = if has_output_weight && host_qtok_embeddings.is_some() {
            Arc::clone(host_qtok_embeddings.as_ref().unwrap())
        } else {
            Arc::new(ct.tensor("token_embd.weight", device)?)
        };
        let tok_embeddings = TokEmbedding::from_gguf_qtensor(
            Arc::clone(&qtok_embeddings),
            host_qtok_embeddings,
            props.embedding_length,
            device,
        )?;
        let norm = QRmsNorm::new(ct.tensor("output_norm.weight", device)?, props.rms_norm_eps)?;
        let output = if has_output_weight {
            gguf_qmm(ct.tensor("output.weight", device)?)?
        } else {
            gguf_qmm_arc(qtok_embeddings)?
        };

        let (embed_hadamard, output_hadamard) = match &hadamard_meta {
            Some(hm) => {
                let dev_had = hadamards
                    .get(&device.location())
                    .expect("Hadamard initialized for root device");
                let emb_had = hm
                    .is_inverse("token_embd.weight")
                    .then(|| Arc::clone(dev_had));
                let out_folded =
                    hm.is_folded("output.weight") || (!has_output_weight && hm.tied_output);
                let out_had = out_folded.then(|| Arc::clone(dev_had));
                (emb_had, out_had)
            }
            None => (None, None),
        };

        // One mRoPE per device location. head_dim arg = rot_dim so cos/sin width = rot_dim/2.
        let mut ropes = HashMap::new();
        for layer_idx in 0..props.block_count {
            let dev = mapper.device_for(layer_idx, false).unwrap_or(device);
            if let std::collections::hash_map::Entry::Vacant(e) = ropes.entry(dev.location()) {
                e.insert(Arc::new(Qwen3VLRotaryEmbedding::new(
                    props.rope_freq_base,
                    props.rot_dim,
                    dev,
                    props.mrope_section.clone(),
                )?));
            }
        }
        let default_rotary = ropes
            .get(&device.location())
            .cloned()
            .unwrap_or_else(|| ropes.values().next().unwrap().clone());

        let mut layers = Vec::with_capacity(props.block_count);
        for layer_idx in NiceProgressBar::<_, 'b'>(
            0..props.block_count,
            "Loading repeating layers",
            &new_multi_progress(),
        ) {
            let prefix = format!("blk.{layer_idx}");
            let dev = mapper.device_for(layer_idx, false).unwrap_or(device);
            let rotary = ropes
                .get(&dev.location())
                .expect("No RoPE for device location!")
                .clone();
            let layer_had = hadamard_meta
                .as_ref()
                .map(|hm| (hadamards.get(&dev.location()).unwrap(), hm));

            let input_layernorm = QRmsNorm::new(
                ct.tensor(&format!("{prefix}.attn_norm.weight"), dev)?,
                props.rms_norm_eps,
            )?;
            let post_attention_layernorm = QRmsNorm::new(
                ct.tensor(&format!("{prefix}.post_attention_norm.weight"), dev)?,
                props.rms_norm_eps,
            )?;

            let layer_impl = match layer_types[layer_idx] {
                LayerType::FullAttention => {
                    let paged_attn = match attention_mechanism {
                        AttentionImplementation::PagedAttention => {
                            Some(PagedAttention::new(props.head_dim, dev, None)?)
                        }
                        AttentionImplementation::Eager => None,
                    };
                    let mut attn = GatedFullAttention::load(
                        &mut ct, &prefix, &props, rotary, paged_attn, dev, dtype,
                    )?;
                    if let Some((had, hm)) = layer_had {
                        attn = attn.with_hadamard(had, hm, &prefix);
                    }
                    LayerImpl::FullAttention(attn)
                }
                LayerType::LinearAttention => {
                    let mut gdn = QGatedDeltaNet::load(&mut ct, &prefix, &props, dev)?;
                    if let Some((had, hm)) = layer_had {
                        gdn = gdn.with_hadamard(had, hm, &prefix);
                    }
                    LayerImpl::LinearAttention(gdn)
                }
            };

            let mlp = if is_moe {
                let mut moe =
                    FusedMoe::from_gguf(&mut ct, &prefix, dev, props.num_experts_per_tok)?;
                if let Some((had, hm)) = layer_had {
                    moe = moe.with_hadamard(had, hm, &prefix);
                }
                MoeOrMlp::FusedMoe(Box::new(moe))
            } else {
                let mut dense = DenseMlp::load(&mut ct, &prefix, dev)?;
                if let Some((had, hm)) = layer_had {
                    dense = dense.with_hadamard(had, hm, &prefix);
                }
                MoeOrMlp::Mlp(dense)
            };

            layers.push(
                DecoderLayer {
                    layer_impl,
                    input_layernorm,
                    post_attention_layernorm,
                    mlp,
                }
                .with_folded_norm_signs(props.embedding_length)?,
            );
        }

        let root_hadamard = hadamards.get(&device.location()).cloned();

        // Pipeline hybrid cache (recurrent pool for GDN layers + KV cache for attention layers).
        let pipeline_layer_types: Vec<HybridLayerType> = layer_types
            .iter()
            .map(|lt| match lt {
                LayerType::FullAttention => HybridLayerType::Attention,
                LayerType::LinearAttention => HybridLayerType::Recurrent,
            })
            .collect();
        let hybrid_cache_config = HybridCacheConfig::uniform(
            pipeline_layer_types,
            props.max_seq_len,
            RecurrentLayerConfig {
                conv_dim,
                conv_width: props.conv_kernel,
                state_dims: vec![props.num_v_heads, props.head_k_dim, props.head_v_dim],
                conv_dtype: dtype,
                // The CUDA and ROCm fused GDN mixers mutate recurrent state in place as f32.
                state_dtype: if device.is_rocm() || device.is_cuda() {
                    DType::F32
                } else {
                    dtype
                },
            },
        );
        let pipeline_cache = Arc::new(Mutex::new(
            HybridCache::new(hybrid_cache_config, device)
                .map_err(|e| hanzo_ml::Error::Msg(format!("Failed to create hybrid cache: {e}")))?,
        ));

        Ok(Self {
            tok_embeddings,
            embed_hadamard,
            hadamard_meta,
            root_hadamard,
            layers,
            layer_types,
            norm,
            output,
            output_hadamard,
            rotary: default_rotary,
            device: device.clone(),
            cache: EitherCache::Hybrid(pipeline_cache),
            max_seq_len: props.max_seq_len,
            mapper: Some(mapper),
            dtype,
            props,
            spec_capture: crate::speculative::HiddenPrefixCapture::default(),
            last_spec: Mutex::new(None),
            store_spec: AtomicBool::new(false),
            mtp_anchors: Arc::new(Mutex::new(None)),
        })
    }
}

impl ModelWeights {
    pub fn forward(
        &self,
        input_ids: &Tensor,
        seqlen_offsets: &[usize],
        context_lens: Vec<(usize, usize)>,
        metadata: Option<(Vec<(Tensor, Tensor)>, &PagedAttentionInputMetadata)>,
    ) -> Result<Tensor> {
        let (b_sz, seq_len) = input_ids.dims2()?;
        let prof_t0 = if prof::enabled() {
            if seq_len > 1 {
                prof::reset();
            }
            self.device.synchronize()?;
            Some(std::time::Instant::now())
        } else {
            None
        };
        let mut x = self.tok_embeddings.forward(input_ids)?;
        if let Some(ref had) = self.embed_hadamard {
            x = had.inverse(&x)?;
        }

        let mut hybrid_cache = self.cache.hybrid();
        let trail = hybrid_cache.records_trail(x.dim(1)?);
        let state_indices = hybrid_cache.state_indices().cloned();
        let state_indices_host: Option<Vec<u32>> =
            hybrid_cache.state_indices_host().map(|s| s.to_vec());
        if self
            .layer_types
            .iter()
            .any(|lt| matches!(lt, LayerType::LinearAttention))
            && state_indices.is_none()
        {
            hanzo_ml::bail!(
                "Hybrid recurrent state indices are required for linear-attention layers."
            );
        }

        // With PagedAttention the running context lives in the paged KV pool, so the past-kv length
        // comes from the host `seqlen_offsets`, not the (unused) hybrid attention KvCache; decode
        // (non-first chunk) needs no mask, as paged attention enforces causality via context_lens.
        // Without paging (CPU/eager) fall back to the hybrid cache. Mirrors quantized_qwen3_moe.
        let mask = CausalMasker.make_causal_mask(
            input_ids,
            match metadata.as_ref() {
                Some(_) => &seqlen_offsets as &dyn PastKvLenCache,
                None => &*hybrid_cache as &dyn PastKvLenCache,
            },
            self.dtype,
            &CausalMaskConfig::gguf(),
        )?;
        let mask = crate::layers_masker::paged_chunk_mask(
            mask,
            metadata.as_ref().map(|(_, meta)| *meta),
            input_ids,
        )?;
        let mask = if let Some(ref mapper) = self.mapper {
            DeviceMappedMask::new(mask, &**mapper)?
        } else {
            DeviceMappedMask::from_single(mask)
        };

        // Text-only 3D mRoPE position ids: all three rows equal the linear position per sequence.
        // The decode-graph path threads a STABLE device `rope_positions` buffer (refreshed in place
        // by the graph runner) so the captured cos/sin advance with the replayed token instead of
        // freezing at the warmup position; the eager path synthesizes them from `seqlen_offsets`.
        let rope_positions = metadata
            .as_ref()
            .and_then(|(_, meta)| meta.rope_positions.as_ref())
            .and_then(|rp| rp.get(&self.device.location()));
        let cos_sin = text_mrope(
            &self.rotary,
            &self.device,
            seqlen_offsets,
            seq_len,
            x.dtype(),
            rope_positions,
        )?;

        let capture_layers = self.spec_capture.layers_for(b_sz);
        let mut captured: Vec<Tensor> = Vec::with_capacity(capture_layers.len());
        // The paged cache holds one K/V pair per attention layer, so an attention layer reads it
        // at its ordinal among attention layers, not at its decoder index.
        let mut kv_layer = 0;
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            if let Some(ref mapper) = self.mapper {
                x = mapper.map(x, layer_idx)?;
            }
            let residual = x.clone();
            let attn_out = match &layer.layer_impl {
                LayerImpl::FullAttention(attn) => {
                    let paged = metadata
                        .as_ref()
                        .map(|(kv_cache, meta)| (kv_cache[kv_layer].clone(), *meta));
                    kv_layer += 1;
                    let Some(HybridLayerCache::Attention(kv_cache)) =
                        hybrid_cache.get_mut(layer_idx)
                    else {
                        hanzo_ml::bail!("Hybrid cache layer {layer_idx} not attention.");
                    };
                    if let Some(b) = attn
                        .can_fuse_norm_fwht_q8()
                        .filter(|&b| can_fuse_single_row_iq_q8(&x, b))
                    {
                        let q8 = prof::timed(x.device(), &prof::HADAMARD_NS, || {
                            hanzo_ml::quantized::QMatMul::rms_norm_hadamard_fwht_iq_q8(
                                &x,
                                layer.input_layernorm.weight(),
                                layer.input_layernorm.eps() as f32,
                                None,
                                b,
                            )
                        })?;
                        attn.forward_with_q8(
                            &q8,
                            x.dtype(),
                            b_sz,
                            seq_len,
                            x.device(),
                            &mask.get(x.device()),
                            &cos_sin,
                            kv_cache,
                            paged,
                        )?
                    } else {
                        let normed = layer.input_layernorm.forward(&x)?;
                        attn.forward(
                            &normed,
                            &mask.get(normed.device()),
                            &cos_sin,
                            kv_cache,
                            paged,
                        )?
                    }
                }
                LayerImpl::LinearAttention(gdn) => {
                    let normed = layer.input_layernorm.forward(&x)?;
                    let Some(HybridLayerCache::Recurrent(pool)) = hybrid_cache.get_mut(layer_idx)
                    else {
                        hanzo_ml::bail!("Hybrid cache layer {layer_idx} not recurrent.");
                    };
                    if b_sz == 1 {
                        // Single sequence: one slot, so gather/scatter is constant-offset
                        // `narrow`/`slice_set` on the HOST slot with no `to_vec1` sync. That sync-free
                        // form is what makes the decode step CUDA-graph capturable (constant baked slot,
                        // recurrence evolves the live pool state in place across replays). b_sz>1 gathers
                        // below.
                        let slot = state_indices_host
                            .as_ref()
                            .and_then(|s| s.first().copied())
                            .ok_or_else(|| {
                                hanzo_ml::Error::msg("missing host recurrent state index")
                            })? as usize;
                        let slots = PoolSlots::One {
                            slot,
                            offset: seqlen_offsets.first().copied().unwrap_or(0),
                        };
                        forward_pooled(pool, slots, layer_idx, trail, |cache| {
                            gdn.forward(&normed, cache)
                        })?
                    } else {
                        let indices = state_indices
                            .as_ref()
                            .expect("checked above: recurrent indices required");
                        forward_pooled(pool, PoolSlots::Many(indices), layer_idx, trail, |cache| {
                            gdn.forward(&normed, cache)
                        })?
                    }
                }
            };

            x = if let Some((dense, b)) = match &layer.mlp {
                MoeOrMlp::Mlp(dense) => dense
                    .can_fuse_norm_fwht_q8()
                    .filter(|&b| can_fuse_single_row_iq_q8(&attn_out, b))
                    .map(|b| (dense, b)),
                MoeOrMlp::FusedMoe(_) => None,
            } {
                let (x_mid, q8) = prof::timed(attn_out.device(), &prof::HADAMARD_NS, || {
                    hanzo_ml::quantized::QMatMul::add_rms_norm_hadamard_fwht_iq_q8(
                        &attn_out,
                        &residual,
                        layer.post_attention_layernorm.weight(),
                        layer.post_attention_layernorm.eps() as f32,
                        None,
                        b,
                    )
                })?;
                let ffn_out = dense.forward_iq_q8(&q8, x_mid.dtype())?;
                (ffn_out + &x_mid)?
            } else {
                let (x_mid, normed) = layer
                    .post_attention_layernorm
                    .forward_of_sum(&attn_out, &residual)?;
                let ffn_out = layer.mlp.forward(&normed)?;
                (ffn_out + &x_mid)?
            };

            // Metal prefill hazard drain. The GDN + generic-MoE prefill path churns many
            // short-lived pooled Metal buffers, and the device buffer pool recycles on
            // `Arc::strong_count == 1` with no GPU-completion check (find_available_buffer): a
            // buffer can be handed to a later op while an earlier, recorded-but-not-yet-executed
            // op still needs to read its previous contents -> corrupted intermediates -> NaN
            // logits (which surfaced downstream as the sampler's bad Metal top-k normalizer). A
            // completion drain per prefill layer forces each layer's ops to finish before its
            // buffers can be recycled. Gated to prefill (seq_len > 1) and Metal only: the
            // single-token decode step is short and race-free, so decode keeps the fast resident
            // MoE path with no per-token drain, and CUDA/other backends are untouched.
            if seq_len > 1 && x.device().is_metal() {
                x.device().synchronize()?;
            }
            if capture_layers.contains(&layer_idx) {
                captured.push(x.clone());
            }
        }
        if !capture_layers.is_empty() {
            let start_pos = seqlen_offsets.first().copied().unwrap_or(0);
            self.spec_capture.fold(start_pos, captured)?;
        }

        let x = x.to_device(&self.device)?;
        let x = extract_logits(&x, context_lens.clone())?;
        let store_spec = self.store_spec.load(Ordering::Relaxed);
        if !store_spec {
            if let Some(ref had) = self.output_hadamard {
                if self.output.supports_shared_iq_q8()
                    && can_fuse_single_row_iq_q8(&x, had.block_size())
                {
                    let signs = had.signs_for(self.props.embedding_length)?;
                    let q8 = prof::timed(&self.device, &prof::HADAMARD_NS, || {
                        hanzo_ml::quantized::QMatMul::rms_norm_hadamard_fwht_iq_q8(
                            &x,
                            self.norm.weight(),
                            self.norm.eps() as f32,
                            signs,
                            had.block_size(),
                        )
                    })?;
                    let out = prof::timed(&self.device, &prof::LM_HEAD_NS, || {
                        self.output.forward_iq_q8_dtype(&q8, x.dtype())
                    })?;
                    if let Some(t0) = prof_t0 {
                        self.device.synchronize()?;
                        prof::TOTAL_NS.fetch_add(
                            t0.elapsed().as_nanos() as u64,
                            std::sync::atomic::Ordering::Relaxed,
                        );
                        if seq_len > 1 {
                            prof::dump("prefill", seq_len, 1);
                            prof::DECODE_STEPS.store(0, std::sync::atomic::Ordering::Relaxed);
                            prof::reset();
                        } else {
                            let s = prof::DECODE_STEPS
                                .fetch_add(1, std::sync::atomic::Ordering::Relaxed)
                                + 1;
                            if s == 1 || s % 16 == 0 {
                                prof::dump("decode", seq_len, s);
                            }
                        }
                    }
                    return Ok(out);
                }
            }
        }
        let x = self.norm.forward(&x)?;
        if store_spec {
            // The rows `extract_logits` kept, and where each sits in its sequence, so a row index
            // means the same position in both.
            let positions = context_lens
                .iter()
                .enumerate()
                .map(|(seq, (start, len))| {
                    let offset = seqlen_offsets.get(seq).copied().unwrap_or(0) + start;
                    (offset..offset + len).collect()
                })
                .collect();
            if let Ok(mut slot) = self.last_spec.lock() {
                *slot = Some(SpecRows {
                    hidden: x.clone(),
                    positions,
                });
            }
        }
        let x = match &self.output_hadamard {
            Some(had) => had.forward(&x)?,
            None => x,
        };
        let x_in = x.contiguous()?;
        let out = prof::timed(&self.device, &prof::LM_HEAD_NS, || {
            self.output.forward(&x_in)
        })?;
        if let Some(t0) = prof_t0 {
            self.device.synchronize()?;
            prof::TOTAL_NS.fetch_add(
                t0.elapsed().as_nanos() as u64,
                std::sync::atomic::Ordering::Relaxed,
            );
            if seq_len > 1 {
                prof::dump("prefill", seq_len, 1);
                prof::DECODE_STEPS.store(0, std::sync::atomic::Ordering::Relaxed);
                prof::reset();
            } else {
                let s = prof::DECODE_STEPS.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
                if s == 1 || s % 16 == 0 {
                    prof::dump("decode", seq_len, s);
                }
            }
        }
        Ok(out)
    }

    pub(crate) fn mtp_anchors(&self) -> crate::models::qwen3_5_mtp::AnchorPositions {
        self.mtp_anchors.clone()
    }

    pub(crate) fn props(&self) -> &PropsGGUF {
        &self.props
    }

    pub(crate) fn rotary(&self) -> Arc<Qwen3VLRotaryEmbedding> {
        self.rotary.clone()
    }

    pub(crate) fn compute_dtype(&self) -> DType {
        self.dtype
    }

    pub(crate) fn hadamard_for_mtp(&self) -> Option<(&Arc<DeviceHadamard>, &HadamardMeta)> {
        match (&self.root_hadamard, &self.hadamard_meta) {
            (Some(had), Some(hm)) => Some((had, hm)),
            _ => None,
        }
    }

    pub(crate) fn has_embed_hadamard(&self) -> bool {
        self.embed_hadamard.is_some()
    }

    /// Stash the final-norm hidden state and positions of each forward for the MTP head.
    pub(crate) fn set_store_spec(&self, store: bool) {
        self.store_spec.store(store, Ordering::Relaxed);
        if !store {
            if let Ok(mut slot) = self.last_spec.lock() {
                *slot = None;
            }
        }
    }

    /// What the last forward stashed, if any.
    pub(crate) fn last_spec(&self) -> Option<(Tensor, Vec<Vec<usize>>)> {
        let slot = self.last_spec.lock().ok()?;
        slot.as_ref()
            .map(|rows| (rows.hidden.clone(), rows.positions.clone()))
    }

    /// The embedding and output head, lent to a draft that carries neither. The head takes the
    /// residual stream's dtype, which is the embedding's.
    pub(crate) fn shared_heads(&self) -> crate::speculative::SpeculativeSharedHeads {
        let embed_tokens = self.tok_embeddings.clone();
        let embed_had = self.embed_hadamard.clone();
        let output = Arc::clone(&self.output);
        let output_had = self.output_hadamard.clone();
        let dtype = self.tok_embeddings.dtype();
        crate::speculative::SpeculativeSharedHeads {
            embed: Arc::new(move |ids: &Tensor| {
                let z = embed_tokens.forward(ids)?;
                match &embed_had {
                    Some(had) => had.inverse(&z),
                    None => Ok(z),
                }
            }),
            lm_head: Arc::new(move |hidden: &Tensor| {
                let h = hidden.to_dtype(dtype)?;
                let h = match &output_had {
                    Some(had) => had.forward(&h)?,
                    None => h,
                };
                output.forward(&h.contiguous()?)
            }),
        }
    }
}

/// Build text-only mRoPE cos/sin. position_ids shape (3, batch, seq) with all three temporal/
/// height/width rows equal to the absolute token position; this collapses interleaved mRoPE to
/// plain partial RoPE, which is correct for text-only generation.
pub(crate) fn text_mrope(
    rotary: &Qwen3VLRotaryEmbedding,
    device: &Device,
    seqlen_offsets: &[usize],
    seq_len: usize,
    dtype: DType,
    rope_positions: Option<&Tensor>,
) -> Result<(Tensor, Tensor)> {
    // (3, batch, seq) position ids -> interleaved mRoPE collapses to partial RoPE for text.
    let position_ids = match rope_positions {
        // Decode-graph path: read the advancing position from the stable device buffer the graph
        // runner refreshes in place (no host Tensor::from_vec, so the captured cos/sin advance).
        Some(rp) if seq_len == 1 => {
            let batch = rp.dim(0)?;
            let pos = rp.reshape((1, batch, 1))?;
            Tensor::cat(&[&pos, &pos, &pos], 0)?
        }
        _ => {
            let batch = seqlen_offsets.len().max(1);
            let mut positions = Vec::with_capacity(batch * seq_len);
            for &off in seqlen_offsets.iter() {
                for p in 0..seq_len {
                    positions.push((off + p) as u32);
                }
            }
            if seqlen_offsets.is_empty() {
                for p in 0..seq_len {
                    positions.push(p as u32);
                }
            }
            let pos_1d = Tensor::from_vec(positions, (batch, seq_len), device)?;
            Tensor::stack(&[&pos_1d, &pos_1d, &pos_1d], 0)?
        }
    };
    rotary.compute_cos_sin(&position_ids, dtype)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::gdn::{GatedDeltaNet, GdnInProj};
    use hanzo_ml::quantized::{gguf_file, GgmlDType, QTensor};
    use hanzo_nn::Linear;
    use hanzo_quant::UnquantLinear;
    use rand::{rngs::StdRng, Rng, SeedableRng};

    const HIDDEN: usize = 12;
    const HEADS: usize = 2;
    const KV_HEADS: usize = 1;
    const HEAD_DIM: usize = 16;
    const ROT_DIM: usize = 8;
    const THETA: f32 = 10_000.0;
    const EPS: f32 = 1e-6;
    const GDN_HEADS: usize = 2;
    const GDN_DIM: usize = 4;
    const CONV: usize = 4;

    fn props() -> PropsGGUF {
        PropsGGUF {
            head_count: HEADS,
            head_count_kv: KV_HEADS,
            block_count: 4,
            embedding_length: HIDDEN,
            rms_norm_eps: EPS,
            max_seq_len: 64,
            rope_freq_base: THETA,
            head_dim: HEAD_DIM,
            rot_dim: ROT_DIM,
            mrope_section: vec![2, 1, 1],
            full_attention_interval: 4,
            conv_kernel: CONV,
            head_k_dim: GDN_DIM,
            head_v_dim: GDN_DIM,
            num_k_heads: GDN_HEADS,
            num_v_heads: GDN_HEADS,
            num_experts: None,
            num_experts_per_tok: 0,
            moe_intermediate_size: 0,
            is_moe: false,
            nextn_predict_layers: 0,
        }
    }

    /// Seeded tensors in torch layout (`[out, in]`, or `[n]` for a vector), named as in the GGUF.
    fn draw(shapes: &[(&str, &[usize])], seed: u64) -> Vec<(String, Vec<usize>, Vec<f32>)> {
        let mut rng = StdRng::seed_from_u64(seed);
        shapes
            .iter()
            .map(|(name, shape)| {
                let n: usize = shape.iter().product();
                // Vectors sit around 1, as a trained norm weight does.
                let shift = if shape.len() == 1 { 1.0 } else { 0.0 };
                let data = (0..n)
                    .map(|_| shift + rng.random_range(-0.5f32..0.5))
                    .collect();
                (format!("blk.0.{name}"), shape.to_vec(), data)
            })
            .collect()
    }

    fn open(
        dir: &std::path::Path,
        tensors: &[(String, Vec<usize>, Vec<f32>)],
    ) -> Result<std::fs::File> {
        let path = dir.join("block.gguf");
        let owned = tensors
            .iter()
            .map(|(name, shape, data)| {
                let t = Tensor::from_vec(data.clone(), shape.as_slice(), &Device::Cpu)?;
                Ok((name.as_str(), QTensor::quantize(&t, GgmlDType::F32)?))
            })
            .collect::<Result<Vec<_>>>()?;
        let refs = owned.iter().map(|(n, t)| (*n, t)).collect::<Vec<_>>();
        let arch = gguf_file::Value::String("qwen35".to_string());
        let mut file = std::fs::File::create(&path).map_err(hanzo_ml::Error::msg)?;
        gguf_file::write(&mut file, &[("general.architecture", &arch)], &refs)?;
        std::fs::File::open(&path).map_err(hanzo_ml::Error::msg)
    }

    fn metadata(arch: &str) -> HashMap<String, gguf_file::Value> {
        HashMap::from([(
            "general.architecture".to_string(),
            gguf_file::Value::String(arch.to_string()),
        )])
    }

    #[test]
    fn verify_arch_accepts_only_the_listed() -> Result<()> {
        let allowed = ["qwen35moe", "qwen35"];
        assert_eq!(verify_arch(&metadata("qwen35"), &allowed)?, "qwen35");
        let err = verify_arch(&metadata("zen6"), &allowed)
            .expect_err("zen6 is not listed")
            .to_string();
        assert!(
            err.contains("Expected `qwen35moe`/`qwen35` architecture, got `zen6`."),
            "{err}"
        );
        assert_eq!(verify_arch(&metadata("zen6"), &["zen6"])?, "zen6");
        Ok(())
    }

    /// One position of the full-attention block in f64: `q`, `k`, `v` and `gate` are the
    /// projections (as `GatedFullAttention::project` returns them), `y` is the attention output
    /// and `out` is `o_proj(y * sigmoid(gate))`.
    struct Row {
        q: Vec<f64>,
        k: Vec<f64>,
        v: Vec<f64>,
        gate: Vec<f64>,
        y: Vec<f64>,
        out: Vec<f64>,
    }

    /// The reference over one sequence at positions `0..`: per-head q/k RMSNorm, NeoX RoPE on the
    /// first `ROT_DIM` dims (text MRoPE has equal planes, so it is plain partial RoPE), and causal
    /// `softmax(q·k/√d)·v` with each KV head serving `HEADS / KV_HEADS` query heads.
    fn attention_reference(x: &[f64], w: &HashMap<String, Vec<f64>>) -> Vec<Row> {
        let linear = |m: &[f64], u: &[f64]| -> Vec<f64> {
            m.chunks(u.len())
                .map(|row| row.iter().zip(u).map(|(a, b)| a * b).sum())
                .collect()
        };
        let norm = |u: &mut [f64], g: &[f64]| {
            let mean = u.iter().map(|a| a * a).sum::<f64>() / u.len() as f64;
            let inv = 1.0 / (mean + f64::from(EPS)).sqrt();
            for (a, g) in u.iter_mut().zip(g) {
                *a *= inv * g;
            }
        };
        let rope = |u: &mut [f64], p: usize| {
            let (lo, hi) = u[..ROT_DIM].split_at_mut(ROT_DIM / 2);
            for (i, (a, c)) in lo.iter_mut().zip(hi).enumerate() {
                let theta = p as f64 * f64::from(THETA).powf(-((2 * i) as f64) / ROT_DIM as f64);
                let (x0, x1) = (*a, *c);
                *a = x0 * theta.cos() - x1 * theta.sin();
                *c = x1 * theta.cos() + x0 * theta.sin();
            }
        };
        let mut rows = x
            .chunks(HIDDEN)
            .enumerate()
            .map(|(p, xt)| {
                // q_proj interleaves `[q_h | gate_h]` per head.
                let qg = linear(&w["blk.0.attn_q.weight"], xt);
                let (mut q, mut gate) = (vec![], vec![]);
                for head in qg.chunks(2 * HEAD_DIM) {
                    let mut qh = head[..HEAD_DIM].to_vec();
                    norm(&mut qh, &w["blk.0.attn_q_norm.weight"]);
                    rope(&mut qh, p);
                    q.extend(qh);
                    gate.extend_from_slice(&head[HEAD_DIM..]);
                }
                let mut k = linear(&w["blk.0.attn_k.weight"], xt);
                for kh in k.chunks_mut(HEAD_DIM) {
                    norm(kh, &w["blk.0.attn_k_norm.weight"]);
                    rope(kh, p);
                }
                let v = linear(&w["blk.0.attn_v.weight"], xt);
                Row {
                    q,
                    k,
                    v,
                    gate,
                    y: vec![],
                    out: vec![],
                }
            })
            .collect::<Vec<_>>();
        for t in 0..rows.len() {
            let mut y = vec![0f64; HEADS * HEAD_DIM];
            for h in 0..HEADS {
                let kv = (h / (HEADS / KV_HEADS)) * HEAD_DIM;
                let qh = &rows[t].q[h * HEAD_DIM..(h + 1) * HEAD_DIM];
                let scores = rows[..=t]
                    .iter()
                    .map(|row| {
                        let dot: f64 = qh.iter().zip(&row.k[kv..]).map(|(a, b)| a * b).sum();
                        dot / (HEAD_DIM as f64).sqrt()
                    })
                    .collect::<Vec<_>>();
                let top = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let e = scores.iter().map(|s| (s - top).exp()).collect::<Vec<_>>();
                let z: f64 = e.iter().sum();
                for (row, e) in rows[..=t].iter().zip(&e) {
                    for (yj, vj) in y[h * HEAD_DIM..(h + 1) * HEAD_DIM]
                        .iter_mut()
                        .zip(&row.v[kv..])
                    {
                        *yj += e / z * vj;
                    }
                }
            }
            let gated = y
                .iter()
                .zip(&rows[t].gate)
                .map(|(a, g)| a / (1.0 + (-g).exp()))
                .collect::<Vec<_>>();
            rows[t].out = linear(&w["blk.0.attn_output.weight"], &gated);
            rows[t].y = y;
        }
        rows
    }

    /// `got` against `want` at position `at`, each value relative to its reference or, where that
    /// is below 0.1, to 0.1: the values are sums that can cancel to near zero, where the f32
    /// rounding of their O(1) terms is no longer small relative to the result.
    fn check(stage: &str, at: usize, got: &[f32], want: &[f64], tol: f64) {
        assert_eq!(got.len(), want.len(), "{stage} at {at}");
        let worst = got
            .iter()
            .zip(want)
            .map(|(g, r)| (f64::from(*g) - r).abs() / r.abs().max(0.1))
            .fold(0f64, f64::max);
        assert!(
            worst < tol,
            "{stage} at {at}: max relative error {worst:.3e}"
        );
    }

    /// Position `i` of head `h` in the values of a `[1, heads, len, HEAD_DIM]` tensor.
    fn head(t: &[f32], len: usize, h: usize, i: usize) -> &[f32] {
        &t[(h * len + i) * HEAD_DIM..][..HEAD_DIM]
    }

    /// A prompt, then one decode step over the block's own `KvCache`, each through `project`,
    /// `attend` and `output`: every stage equals the reference over the whole sequence.
    #[test]
    fn gated_attention_matches_reference() -> Result<()> {
        // The CPU's eager SDPA, which serves the prompt, multiplies in f16 (hanzo-quant
        // `MatMul::matmul` without `accelerate`): 2^-11 rounding per operand through the score and
        // value products. The single-query kernel that serves the decode step is f32.
        const F16_ATTENTION: f64 = 1e-2;
        let dev = Device::Cpu;
        let tensors = draw(
            &[
                ("attn_q.weight", &[2 * HEADS * HEAD_DIM, HIDDEN]),
                ("attn_k.weight", &[KV_HEADS * HEAD_DIM, HIDDEN]),
                ("attn_v.weight", &[KV_HEADS * HEAD_DIM, HIDDEN]),
                ("attn_output.weight", &[HIDDEN, HEADS * HEAD_DIM]),
                ("attn_q_norm.weight", &[HEAD_DIM]),
                ("attn_k_norm.weight", &[HEAD_DIM]),
            ],
            0x6174_746e,
        );
        let dir = tempfile::tempdir().map_err(hanzo_ml::Error::msg)?;
        let mut files = [open(dir.path(), &tensors)?];
        let mut readers: Vec<&mut std::fs::File> = files.iter_mut().collect();
        let mut ct = Content::from_readers(&mut readers)?;
        let props = props();
        let rotary = Arc::new(Qwen3VLRotaryEmbedding::new(
            THETA,
            ROT_DIM,
            &dev,
            props.mrope_section.clone(),
        )?);
        let attn =
            GatedFullAttention::load(&mut ct, "blk.0", &props, rotary, None, &dev, DType::F32)?;
        let w: HashMap<String, Vec<f64>> = tensors
            .iter()
            .map(|(name, _, data)| (name.clone(), data.iter().map(|&a| f64::from(a)).collect()))
            .collect();

        let (prompt, total) = (5usize, 6usize);
        let mut rng = StdRng::seed_from_u64(0x7870);
        let x: Vec<f32> = (0..total * HIDDEN)
            .map(|_| rng.random_range(-1f32..1.0))
            .collect();
        let want = attention_reference(&x.iter().map(|&a| f64::from(a)).collect::<Vec<_>>(), &w);
        let xs = Tensor::from_vec(x, (1, total, HIDDEN), &dev)?;

        let mut cache = KvCache::new_normal(2, total, total);
        for (start, len) in [(0, prompt), (prompt, 1)] {
            let ids = Tensor::zeros((1, len), DType::U32, &dev)?;
            let offsets: &[usize] = &[start];
            let mask = CausalMasker.make_causal_mask(
                &ids,
                &offsets,
                DType::F32,
                &CausalMaskConfig::gguf(),
            )?;
            let pos = Tensor::arange(start as u32, (start + len) as u32, &dev)?.unsqueeze(0)?;
            let cos_sin =
                attn.rotary_cos_sin(&Tensor::stack(&[&pos, &pos, &pos], 0)?, DType::F32)?;
            let chunk = xs.narrow(1, start, len)?.contiguous()?;
            let (q, k, v, gate) = attn.project(&chunk, &cos_sin)?;
            let y = attn.attend(&q, &k, &v, &mask, &mut cache, None)?;
            let out = attn.output(&y, &gate, DType::F32)?;

            let tol = if len > 1 { F16_ATTENTION } else { 1e-5 };
            let values = |t: &Tensor| t.flatten_all()?.to_vec1::<f32>();
            let (q, k, v) = (values(&q)?, values(&k)?, values(&v)?);
            let (gate, y, out) = (values(&gate)?, values(&y)?, values(&out)?);
            let width = HEADS * HEAD_DIM;
            for (i, row) in want[start..start + len].iter().enumerate() {
                let at = start + i;
                for h in 0..HEADS {
                    let own = h * HEAD_DIM..(h + 1) * HEAD_DIM;
                    check("q", at, head(&q, len, h, i), &row.q[own], 1e-5);
                }
                for h in 0..KV_HEADS {
                    let own = h * HEAD_DIM..(h + 1) * HEAD_DIM;
                    check("k", at, head(&k, len, h, i), &row.k[own.clone()], 1e-5);
                    check("v", at, head(&v, len, h, i), &row.v[own], 1e-5);
                }
                check("gate", at, &gate[i * width..][..width], &row.gate, 1e-5);
                check("y", at, &y[i * width..][..width], &row.y, tol);
                check("out", at, &out[i * HIDDEN..][..HIDDEN], &row.out, tol);
            }
        }
        Ok(())
    }

    /// `QGatedDeltaNet::load` reads a Qwen3.5 GGUF block into the layer the safetensors
    /// `GatedDeltaNet` computes from the same weights. With as many V heads as K heads the GGUF's
    /// tiled V order is the checkpoint's grouped order, so the two agree over a prompt and a step.
    #[test]
    fn gdn_load_matches_safetensors_layer() -> Result<()> {
        let dev = Device::Cpu;
        let (key_dim, value_dim) = (GDN_HEADS * GDN_DIM, GDN_HEADS * GDN_DIM);
        let conv_dim = 2 * key_dim + value_dim;
        let mut tensors = draw(
            &[
                ("attn_qkv.weight", &[conv_dim, HIDDEN]),
                ("attn_gate.weight", &[value_dim, HIDDEN]),
                ("ssm_beta.weight", &[GDN_HEADS, HIDDEN]),
                ("ssm_alpha.weight", &[GDN_HEADS, HIDDEN]),
                ("ssm_out.weight", &[HIDDEN, value_dim]),
                ("ssm_conv1d.weight", &[conv_dim, CONV]),
                ("ssm_dt.bias", &[GDN_HEADS]),
                ("ssm_norm.weight", &[GDN_DIM]),
            ],
            0x6764_6e6c,
        );
        let a_log = Tensor::new(&[-0.3f32, 0.4], &dev)?;
        // The converter stores -exp(A_log).
        let ssm_a = a_log.exp()?.neg()?;
        tensors.push(("blk.0.ssm_a".to_string(), vec![GDN_HEADS], ssm_a.to_vec1()?));
        let tensor = |name: &str| -> Result<Tensor> {
            let (_, shape, data) = tensors
                .iter()
                .find(|(n, _, _)| n == &format!("blk.0.{name}"))
                .expect("drawn above");
            Tensor::from_vec(data.clone(), shape.as_slice(), &dev)
        };
        let dense = |name: &str| -> Result<Arc<dyn QuantMethod>> {
            Ok(Arc::new(UnquantLinear::new(
                QuantMethodConfig::Unquantized(Linear::new(tensor(name)?, None)),
            )?))
        };
        let dt_bias = tensor("ssm_dt.bias")?;
        let dt_bias_f32 = dt_bias.to_dtype(DType::F32)?.contiguous()?;
        let neg_exp_a = a_log.to_dtype(DType::F32)?.exp()?.neg()?.contiguous()?;
        let twin = GatedDeltaNet {
            in_proj: GdnInProj::Split {
                qkv: dense("attn_qkv.weight")?,
                z: dense("attn_gate.weight")?,
                b: dense("ssm_beta.weight")?,
                a: dense("ssm_alpha.weight")?,
            },
            conv1d_weight: tensor("ssm_conv1d.weight")?.unsqueeze(1)?,
            dt_bias,
            dt_bias_f32,
            a_log,
            neg_exp_a,
            norm: RmsNormGated::from_weight(tensor("ssm_norm.weight")?, f64::from(EPS)),
            out_proj: dense("ssm_out.weight")?,
            num_k_heads: GDN_HEADS,
            num_v_heads: GDN_HEADS,
            head_k_dim: GDN_DIM,
            head_v_dim: GDN_DIM,
            conv_kernel_size: CONV,
            key_dim,
            value_dim,
        };

        let dir = tempfile::tempdir().map_err(hanzo_ml::Error::msg)?;
        let mut files = [open(dir.path(), &tensors)?];
        let mut readers: Vec<&mut std::fs::File> = files.iter_mut().collect();
        let mut ct = Content::from_readers(&mut readers)?;
        let gdn = QGatedDeltaNet::load(&mut ct, "blk.0", &props(), &dev)?;

        let fresh = || -> Result<GdnLayerCache> {
            Ok(GdnLayerCache {
                conv_state: Tensor::zeros((1, conv_dim, CONV), DType::F32, &dev)?,
                recurrent_state: Tensor::zeros((1, GDN_HEADS, GDN_DIM, GDN_DIM), DType::F32, &dev)?,
                seqlen_offset: 0,
                trail: None,
            })
        };
        let (mut ours, mut theirs) = (fresh()?, fresh()?);
        let mut rng = StdRng::seed_from_u64(0x6764_6e78);
        let mut at = 0;
        for len in [5usize, 1] {
            let x: Vec<f32> = (0..len * HIDDEN)
                .map(|_| rng.random_range(-1f32..1.0))
                .collect();
            let x = Tensor::from_vec(x, (1, len, HIDDEN), &dev)?;
            let got = gdn
                .forward(&x, &mut ours)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            let want = twin
                .forward(&x, &mut theirs)?
                .flatten_all()?
                .to_dtype(DType::F64)?
                .to_vec1::<f64>()?;
            check("gdn", at, &got, &want, 1e-5);
            at += len;
        }
        Ok(())
    }

    #[test]
    fn hadamard_meta_and_folded_linear_equivalence() -> Result<()> {
        let dev = Device::Cpu;
        for prefix in ["prism.hadamard", "zen.hadamard"] {
            let mut kv = HashMap::new();
            kv.insert(
                format!("{prefix}.version"),
                gguf_file::Value::U32(if prefix == "prism.hadamard" { 1 } else { 2 }),
            );
            kv.insert(format!("{prefix}.block_size"), gguf_file::Value::U32(8));
            kv.insert(
                format!("{prefix}.transform"),
                gguf_file::Value::String("normalized-sylvester-walsh-hadamard".to_string()),
            );
            kv.insert(
                format!("{prefix}.axis"),
                gguf_file::Value::String("input-last-dimension".to_string()),
            );
            kv.insert(
                format!("{prefix}.sign_mode"),
                gguf_file::Value::String("explicit".to_string()),
            );
            kv.insert(
                format!("{prefix}.sign_widths"),
                gguf_file::Value::Array(vec![gguf_file::Value::I32(16)]),
            );
            let mut rng = StdRng::seed_from_u64(0x4841_4441);
            let signs_i32: Vec<i32> = (0..16)
                .map(|_| if rng.random_bool(0.5) { 1 } else { -1 })
                .collect();
            kv.insert(
                format!("{prefix}.sign_values"),
                gguf_file::Value::Array(
                    signs_i32
                        .iter()
                        .copied()
                        .map(gguf_file::Value::I32)
                        .collect(),
                ),
            );
            kv.insert(
                format!("{prefix}.weight_names"),
                gguf_file::Value::Array(vec![gguf_file::Value::String(
                    "blk.0.ffn_gate.weight".to_string(),
                )]),
            );
            kv.insert(
                format!("{prefix}.inverse_weight_names"),
                gguf_file::Value::Array(vec![gguf_file::Value::String(
                    "token_embd.weight".to_string(),
                )]),
            );
            kv.insert(
                format!("{prefix}.tied_output"),
                gguf_file::Value::Bool(prefix == "zen.hadamard"),
            );
            kv.insert(
                format!("{prefix}.gdn_v_grouped"),
                gguf_file::Value::Bool(true),
            );

            let hm = HadamardMeta::from_gguf_metadata(&kv)?.expect("parsed hadamard");
            assert_eq!(hm.block_size, 8);
            assert!(hm.is_folded("blk.0.ffn_gate.weight"));
            assert!(hm.is_inverse("token_embd.weight"));
            assert_eq!(hm.tied_output, prefix == "zen.hadamard");
            assert!(hm.gdn_v_grouped);

            let had = DeviceHadamard::new(&hm, &dev)?;
            // 1) Roundtrip T^{-1}(T(x)) == x on 3D activations [2, 3, 16]
            let x_data: Vec<f32> = (0..2 * 3 * 16)
                .map(|_| rng.random_range(-1f32..1.0))
                .collect();
            let x = Tensor::from_vec(x_data.clone(), (2, 3, 16), &dev)?;
            let x_rt = had.inverse(&had.forward(&x)?)?;
            let got_rt = x_rt.flatten_all()?.to_vec1::<f32>()?;
            let want_rt: Vec<f64> = x_data.iter().map(|&v| f64::from(v)).collect();
            check("hadamard_roundtrip", 0, &got_rt, &want_rt, 1e-5);

            // 2) Folded linear equivalence: T(x) @ T(W)^T == x @ W^T
            let w_data: Vec<f32> = (0..10 * 16).map(|_| rng.random_range(-1f32..1.0)).collect();
            let w = Tensor::from_vec(w_data, (10, 16), &dev)?;
            let w_rot = had.forward(&w)?;
            let x_2d = x.reshape((6, 16))?;
            let y_primal = x_2d
                .matmul(&w.t()?)?
                .flatten_all()?
                .to_dtype(DType::F64)?
                .to_vec1::<f64>()?;
            let y_folded = had
                .forward(&x)?
                .reshape((6, 16))?
                .matmul(&w_rot.t()?)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            check("hadamard_linear", 0, &y_folded, &y_primal, 1e-5);
        }
        Ok(())
    }

    #[test]
    fn spec_capture_pause_and_resume_across_stand_down() -> Result<()> {
        let cap = crate::speculative::HiddenPrefixCapture::default();
        cap.request(crate::speculative::CaptureRequest {
            layers: vec![0, 1],
            retain: Some(8),
            dtype: None,
        });
        cap.note_forward(&[1]);
        assert_eq!(cap.layers_for(1), vec![0, 1]);
        let row = |val: f32, n: usize| Tensor::full(val, (1, n, 4), &Device::Cpu);
        cap.fold(0, vec![row(1.0, 3)?, row(2.0, 3)?])?;
        assert!(cap.hiddens().is_some());

        // Stand down: pausing disables layer capture immediately without corrupting owner tracking.
        cap.set_forward_paused(true);
        assert!(cap.is_forward_paused());
        assert!(cap.layers_for(1).is_empty());

        // Resuming on a new sequence reseeds cleanly without mixing seq 1 and seq 2 rows.
        cap.set_forward_paused(false);
        assert!(!cap.is_forward_paused());
        cap.note_forward(&[2]);
        assert_eq!(cap.layers_for(1), vec![0, 1]);
        cap.fold(0, vec![row(3.0, 2)?, row(4.0, 2)?])?;
        let win = cap.hiddens().expect("resumed capture window");
        assert_eq!(win.start, 0);
        assert_eq!(win.end()?, 2);
        let got: Vec<f32> = win.layers[0].flatten_all()?.to_vec1()?;
        assert!(got.iter().all(|v| (*v - 3.0).abs() < 1e-6));
        Ok(())
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_qwen35_stand_down_partial_rewind_kv_gdn_and_cuda_graph_parity() -> Result<()> {
        if !std::env::args().any(|a| a == "--exact")
            && crate::engine_env_os(
                "ENGINE_CUDA_CAPTURE_ISOLATED",
                "HANZO_CUDA_CAPTURE_ISOLATED",
            )
            .is_none()
        {
            let exe = std::env::current_exe().map_err(hanzo_ml::Error::wrap)?;
            let out = std::process::Command::new(exe)
                .env("ENGINE_CUDA_CAPTURE_ISOLATED", "1")
                .args([
                    "--exact",
                    "models::quantized_qwen3_5_moe::tests::cuda_qwen35_stand_down_partial_rewind_kv_gdn_and_cuda_graph_parity",
                    "--nocapture",
                ])
                .output()
                .map_err(hanzo_ml::Error::wrap)?;
            let stdout = String::from_utf8_lossy(&out.stdout);
            let stderr = String::from_utf8_lossy(&out.stderr);
            assert!(
                out.status.success() && stdout.contains("1 passed"),
                "isolated cuda_qwen35 capture test failed:\n{stdout}\n{stderr}"
            );
            return Ok(());
        }
        use crate::pipeline::cuda_graph::{
            capture_cuda_graph_with_output, CudaDecodeGraphMetadataBuffers,
        };
        use crate::utils::model_config::FromGGUF;
        use crate::DeviceMapSetting;
        use hanzo_ml::{IndexOp, Var};

        const Q_HIDDEN: usize = 64;
        const Q_HEADS: usize = 2;
        const Q_KV_HEADS: usize = 1;
        const Q_HEAD_DIM: usize = 64;
        const Q_ROT_DIM: usize = 16;
        const Q_GDN_HEADS: usize = 2;
        const Q_GDN_DIM: usize = 16;
        const Q_CONV: usize = 4;
        const Q_FFN: usize = 64;
        const Q_VOCAB: usize = 32;
        const BLOCK_SIZE: usize = 16;
        const NUM_BLOCKS: usize = 4;

        let dev = Device::new_cuda_with_stream(0)?;
        if let Device::Cuda(d) = &dev {
            unsafe { d.disable_event_tracking() };
        }
        let conv_dim = 2 * Q_GDN_HEADS * Q_GDN_DIM + Q_GDN_HEADS * Q_GDN_DIM;
        let mut rng = StdRng::seed_from_u64(0x5157_3501);
        let mut specs: Vec<(String, Vec<usize>, Vec<f32>, GgmlDType)> = Vec::new();
        let mut push_t = |name: String, shape: Vec<usize>, dt: GgmlDType| {
            let n: usize = shape.iter().product();
            let is_vec = shape.len() == 1;
            let data: Vec<f32> = (0..n)
                .map(|_| {
                    if is_vec {
                        if name.ends_with(".ssm_a") {
                            -rng.random_range(0.05f32..0.4)
                        } else if name.ends_with(".ssm_dt.bias") {
                            rng.random_range(-0.3f32..0.3)
                        } else {
                            1.0 + rng.random_range(-0.1f32..0.1)
                        }
                    } else {
                        rng.random_range(-0.15f32..0.15)
                    }
                })
                .collect();
            specs.push((name, shape, data, dt));
        };

        push_t(
            "token_embd.weight".into(),
            vec![Q_VOCAB, Q_HIDDEN],
            GgmlDType::F16,
        );
        push_t("output_norm.weight".into(), vec![Q_HIDDEN], GgmlDType::F32);
        push_t(
            "output.weight".into(),
            vec![Q_VOCAB, Q_HIDDEN],
            GgmlDType::F16,
        );
        for blk in 0..2 {
            let p = format!("blk.{blk}");
            push_t(
                format!("{p}.attn_norm.weight"),
                vec![Q_HIDDEN],
                GgmlDType::F32,
            );
            push_t(
                format!("{p}.post_attention_norm.weight"),
                vec![Q_HIDDEN],
                GgmlDType::F32,
            );
            push_t(
                format!("{p}.ffn_gate.weight"),
                vec![Q_FFN, Q_HIDDEN],
                GgmlDType::F16,
            );
            push_t(
                format!("{p}.ffn_up.weight"),
                vec![Q_FFN, Q_HIDDEN],
                GgmlDType::F16,
            );
            push_t(
                format!("{p}.ffn_down.weight"),
                vec![Q_HIDDEN, Q_FFN],
                GgmlDType::F16,
            );
            if blk == 0 {
                push_t(
                    format!("{p}.attn_qkv.weight"),
                    vec![conv_dim, Q_HIDDEN],
                    GgmlDType::F16,
                );
                push_t(
                    format!("{p}.attn_gate.weight"),
                    vec![Q_GDN_HEADS * Q_GDN_DIM, Q_HIDDEN],
                    GgmlDType::F16,
                );
                push_t(
                    format!("{p}.ssm_beta.weight"),
                    vec![Q_GDN_HEADS, Q_HIDDEN],
                    GgmlDType::F16,
                );
                push_t(
                    format!("{p}.ssm_alpha.weight"),
                    vec![Q_GDN_HEADS, Q_HIDDEN],
                    GgmlDType::F16,
                );
                push_t(
                    format!("{p}.ssm_out.weight"),
                    vec![Q_HIDDEN, Q_GDN_HEADS * Q_GDN_DIM],
                    GgmlDType::F16,
                );
                push_t(
                    format!("{p}.ssm_conv1d.weight"),
                    vec![conv_dim, Q_CONV],
                    GgmlDType::F32,
                );
                push_t(
                    format!("{p}.ssm_dt.bias"),
                    vec![Q_GDN_HEADS],
                    GgmlDType::F32,
                );
                push_t(format!("{p}.ssm_a"), vec![Q_GDN_HEADS], GgmlDType::F32);
                push_t(
                    format!("{p}.ssm_norm.weight"),
                    vec![Q_GDN_DIM],
                    GgmlDType::F32,
                );
            } else {
                push_t(
                    format!("{p}.attn_q.weight"),
                    vec![2 * Q_HEADS * Q_HEAD_DIM, Q_HIDDEN],
                    GgmlDType::F16,
                );
                push_t(
                    format!("{p}.attn_k.weight"),
                    vec![Q_KV_HEADS * Q_HEAD_DIM, Q_HIDDEN],
                    GgmlDType::F16,
                );
                push_t(
                    format!("{p}.attn_v.weight"),
                    vec![Q_KV_HEADS * Q_HEAD_DIM, Q_HIDDEN],
                    GgmlDType::F16,
                );
                push_t(
                    format!("{p}.attn_output.weight"),
                    vec![Q_HIDDEN, Q_HEADS * Q_HEAD_DIM],
                    GgmlDType::F16,
                );
                push_t(
                    format!("{p}.attn_q_norm.weight"),
                    vec![Q_HEAD_DIM],
                    GgmlDType::F32,
                );
                push_t(
                    format!("{p}.attn_k_norm.weight"),
                    vec![Q_HEAD_DIM],
                    GgmlDType::F32,
                );
            }
        }

        let dir = tempfile::tempdir().map_err(hanzo_ml::Error::msg)?;
        let gguf_path = dir.path().join("qwen35_tiny.gguf");
        let owned_qt = specs
            .iter()
            .map(|(name, shape, data, dt)| {
                let t = Tensor::from_vec(data.clone(), shape.as_slice(), &Device::Cpu)?;
                Ok((name.as_str(), QTensor::quantize(&t, *dt)?))
            })
            .collect::<Result<Vec<_>>>()?;
        let qt_refs: Vec<(&str, &QTensor)> = owned_qt.iter().map(|(n, q)| (*n, q)).collect();
        let meta_kv = [
            (
                "general.architecture",
                gguf_file::Value::String("qwen35".into()),
            ),
            ("qwen35.block_count", gguf_file::Value::U32(2)),
            ("qwen35.full_attention_interval", gguf_file::Value::U32(2)),
            (
                "qwen35.embedding_length",
                gguf_file::Value::U32(Q_HIDDEN as u32),
            ),
            (
                "qwen35.attention.head_count",
                gguf_file::Value::U32(Q_HEADS as u32),
            ),
            (
                "qwen35.attention.head_count_kv",
                gguf_file::Value::U32(Q_KV_HEADS as u32),
            ),
            (
                "qwen35.attention.key_length",
                gguf_file::Value::U32(Q_HEAD_DIM as u32),
            ),
            (
                "qwen35.rope.dimension_count",
                gguf_file::Value::U32(Q_ROT_DIM as u32),
            ),
            (
                "qwen35.rope.dimension_sections",
                gguf_file::Value::Array(vec![
                    gguf_file::Value::I32(4),
                    gguf_file::Value::I32(2),
                    gguf_file::Value::I32(2),
                    gguf_file::Value::I32(0),
                ]),
            ),
            (
                "qwen35.attention.layer_norm_rms_epsilon",
                gguf_file::Value::F32(1e-6),
            ),
            ("qwen35.context_length", gguf_file::Value::U64(32)),
            (
                "qwen35.ssm.conv_kernel",
                gguf_file::Value::U32(Q_CONV as u32),
            ),
            (
                "qwen35.ssm.state_size",
                gguf_file::Value::U32(Q_GDN_DIM as u32),
            ),
            (
                "qwen35.ssm.group_count",
                gguf_file::Value::U32(Q_GDN_HEADS as u32),
            ),
            (
                "qwen35.ssm.time_step_rank",
                gguf_file::Value::U32(Q_GDN_HEADS as u32),
            ),
            (
                "qwen35.ssm.inner_size",
                gguf_file::Value::U32((Q_GDN_HEADS * Q_GDN_DIM) as u32),
            ),
        ];
        {
            let mut out_f = std::fs::File::create(&gguf_path).map_err(hanzo_ml::Error::msg)?;
            let meta_refs: Vec<(&str, &gguf_file::Value)> =
                meta_kv.iter().map(|(k, v)| (*k, v)).collect();
            gguf_file::write(&mut out_f, &meta_refs, &qt_refs)?;
        }

        let load_model = || -> Result<(ModelWeights, Vec<(Tensor, Tensor)>, usize)> {
            let mut f = std::fs::File::open(&gguf_path).map_err(hanzo_ml::Error::msg)?;
            let mut readers = [&mut f];
            let ct = Content::from_readers(&mut readers)?;
            let mapper = DeviceMapSetting::dummy().into_mapper(2, &dev, None, &[dev.clone()])?;
            let model = ModelWeights::from_gguf(
                ct,
                &dev,
                mapper,
                AttentionImplementation::PagedAttention,
                DType::F16,
            )?;
            let slot = {
                let mut hybrid = model.cache.hybrid();
                let s = hybrid.allocate_seq().expect("recurrent slot");
                hybrid.set_state_indices(Some(Tensor::new(&[s as u32], &dev)?));
                hybrid.set_state_indices_host(Some(vec![s as u32]));
                s
            };
            let x = 16 / DType::F16.size_in_bytes();
            let k_cache = Tensor::zeros(
                (NUM_BLOCKS, Q_KV_HEADS, Q_HEAD_DIM / x, BLOCK_SIZE, x),
                DType::F16,
                &dev,
            )?;
            let v_cache = Tensor::zeros(
                (NUM_BLOCKS, Q_KV_HEADS, Q_HEAD_DIM, BLOCK_SIZE),
                DType::F16,
                &dev,
            )?;
            Ok((model, vec![(k_cache, v_cache)], slot))
        };

        let make_prefill_meta =
            |slots: &[i64], ctx_len: usize| -> Result<PagedAttentionInputMetadata> {
                let mut m = PagedAttentionInputMetadata::dummy(&dev)?;
                m.slot_mappings = HashMap::from([(
                    dev.location(),
                    Tensor::from_vec(slots.to_vec(), (slots.len(),), &dev)?,
                )]);
                m.block_tables = Some(HashMap::from([(
                    dev.location(),
                    Tensor::from_vec(vec![0u32, 1u32], (1, 2), &dev)?,
                )]));
                m.context_lens = Some(HashMap::from([(
                    dev.location(),
                    Tensor::from_vec(vec![ctx_len as u32], (1,), &dev)?,
                )]));
                m.block_size = Some(BLOCK_SIZE);
                m.max_context_len = Some(ctx_len);
                m.paged_context_lens_cpu = Some(vec![ctx_len]);
                m.is_first_prompt_chunk = true;
                m.block_table_signature = Some(vec![0u64, 1u64]);
                Ok(m)
            };
        let make_verify_meta =
            |slots: &[i64], cached: usize, q_len: usize| -> Result<PagedAttentionInputMetadata> {
                let total = cached + q_len;
                let mut m = make_prefill_meta(slots, total)?;
                m.is_first_prompt_chunk = false;
                m.num_cached_tokens = Some(vec![cached]);
                m.query_lens = Some(vec![q_len]);
                m.cu_seqlens_q = Some(HashMap::from([(
                    dev.location(),
                    Tensor::from_vec(vec![0u32, q_len as u32], (2,), &dev)?,
                )]));
                m.cu_seqlens_kv = Some(HashMap::from([(
                    dev.location(),
                    Tensor::from_vec(vec![0u32, total as u32], (2,), &dev)?,
                )]));
                Ok(m)
            };
        let make_decode_meta = |pos: usize| -> Result<PagedAttentionInputMetadata> {
            let mut m = make_prefill_meta(&[pos as i64], pos + 1)?;
            m.is_first_prompt_chunk = false;
            Ok(m)
        };

        let (spec_model, spec_kv, spec_slot) = load_model()?;
        let (plain_model, plain_kv, plain_slot) = load_model()?;

        // 1) Shared prompt [1, 2, 3, 4, 5] (5 tokens)
        let prompt_ids: Vec<u32> = vec![1, 2, 3, 4, 5];
        let prompt_tensor = Tensor::from_vec(prompt_ids.clone(), (1, prompt_ids.len()), &dev)?;
        let prefill_slots: Vec<i64> = (0..prompt_ids.len() as i64).collect();
        let prefill_meta = make_prefill_meta(&prefill_slots, prompt_ids.len())?;

        spec_model
            .spec_capture
            .request(crate::speculative::CaptureRequest {
                layers: vec![0, 1],
                retain: Some(16),
                dtype: None,
            });
        spec_model.spec_capture.note_forward(&[1]);
        let spec_prefill_logits = spec_model.forward(
            &prompt_tensor,
            &[0],
            vec![(prompt_ids.len() - 1, 1)],
            Some((spec_kv.clone(), &prefill_meta)),
        )?;
        let plain_prefill_logits = plain_model.forward(
            &prompt_tensor,
            &[0],
            vec![(prompt_ids.len() - 1, 1)],
            Some((plain_kv.clone(), &prefill_meta)),
        )?;
        assert_eq!(
            spec_prefill_logits
                .argmax(D::Minus1)?
                .flatten_all()?
                .to_vec1::<u32>()?,
            plain_prefill_logits
                .argmax(D::Minus1)?
                .flatten_all()?
                .to_vec1::<u32>()?
        );

        // 2) Speculative verify 4 tokens [6, 10, 20, 21], accept only 2 ([6, 10]) -> reject 2.
        let verify_tokens: Vec<u32> = vec![6, 10, 20, 21];
        let verify_tensor =
            Tensor::from_vec(verify_tokens.clone(), (1, verify_tokens.len()), &dev)?;
        let verify_slots: Vec<i64> = (5..9).collect();
        let verify_meta = make_verify_meta(&verify_slots, 5, 4)?;
        spec_model.cache.hybrid().expect_verify(Some(4));
        let spec_verify_logits = spec_model.forward(
            &verify_tensor,
            &[5],
            vec![(0, 4)],
            Some((spec_kv.clone(), &verify_meta)),
        )?;
        spec_model.cache.hybrid().expect_verify(None);
        assert_eq!(
            spec_model.spec_capture.hiddens().expect("captured").end()?,
            9
        );

        // Partial verification rollback on GDN recurrent state (rejecting tokens 20, 21 -> keep offset 7).
        {
            let mut hybrid = spec_model.cache.hybrid();
            hybrid.rewind_recurrent(spec_slot, 2)?;
            assert_eq!(hybrid.recurrent_offset(spec_slot), Some(7));
        }

        // Plain reference executes the 2 accepted tokens step-by-step at offsets 5 and 6.
        let mut plain_accepted_logits = Vec::new();
        for (i, &tok) in verify_tokens[..2].iter().enumerate() {
            let pos = 5 + i;
            let ids = Tensor::from_vec(vec![tok], (1, 1), &dev)?;
            let meta = make_decode_meta(pos)?;
            let logits =
                plain_model.forward(&ids, &[pos], vec![(0, 1)], Some((plain_kv.clone(), &meta)))?;
            plain_accepted_logits.push(logits);
        }

        for i in 0..2usize {
            let spec_row = spec_verify_logits
                .narrow(1, i, 1)?
                .to_dtype(DType::F32)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            let plain_row = plain_accepted_logits[i]
                .to_dtype(DType::F32)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            let max_diff = spec_row
                .iter()
                .zip(&plain_row)
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            assert!(
                max_diff < 2e-2,
                "verify kept row {i} vs plain decode max_diff={max_diff}"
            );
        }

        // 3) Stand down: pause capture so plain decode graph path applies.
        // Run eager warmup on pos=7 (mirroring `try_cuda_decode_graph_forward`), capture the CUDA
        // decode graph, then replay pos=8..12 through `cuGraphLaunch` and compare every step
        // against plain stepwise decode.
        spec_model.spec_capture.set_forward_paused(true);
        assert!(spec_model.spec_capture.is_forward_paused());
        assert!(spec_model.spec_capture.layers_for(1).is_empty());

        let Device::Cuda(cuda_dev) = &dev else {
            unreachable!();
        };
        let post_stand_down_tokens: Vec<u32> = vec![14, 7, 19, 3, 25];
        let step7_meta = make_decode_meta(7)?;
        let step7_ids = Tensor::from_vec(vec![post_stand_down_tokens[0]], (1, 1), &dev)?;
        let (graph_input_var, mut meta_buffers, graph_logits, graph) = {
            let _htod_guard = cuda_dev.enable_cuda_graph_htod_cache();
            let warmup_logits = spec_model.forward(
                &step7_ids,
                &[7],
                vec![(0, 1)],
                Some((spec_kv.clone(), &step7_meta)),
            )?;
            dev.synchronize()?;
            let plain_step7 = plain_model.forward(
                &step7_ids,
                &[7],
                vec![(0, 1)],
                Some((plain_kv.clone(), &step7_meta)),
            )?;
            dev.synchronize()?;
            assert_eq!(
                warmup_logits
                    .argmax(D::Minus1)?
                    .flatten_all()?
                    .to_vec1::<u32>()?,
                plain_step7
                    .argmax(D::Minus1)?
                    .flatten_all()?
                    .to_vec1::<u32>()?,
                "post-stand-down warmup token mismatch at pos 7"
            );

            let graph_input_var = Var::from_tensor(&step7_ids)?;
            let (meta_buffers, rebound_meta) =
                CudaDecodeGraphMetadataBuffers::new(&step7_meta, &[7], BLOCK_SIZE)?;
            let graph_input_ids = graph_input_var.as_detached_tensor();

            let (graph, graph_logits) =
                capture_cuda_graph_with_output(&dev, &warmup_logits, || {
                    spec_model.forward(
                        &graph_input_ids,
                        &[7],
                        vec![(0, 1)],
                        Some((spec_kv.clone(), &rebound_meta)),
                    )
                })?;
            (graph_input_var, meta_buffers, graph_logits, graph)
        };

        for (step, &tok) in post_stand_down_tokens.iter().enumerate().skip(1) {
            let pos = 7 + step;
            let meta = make_decode_meta(pos)?;
            let next_ids = Tensor::from_vec(vec![tok], (1, 1), &dev)?;
            graph_input_var.set(&next_ids)?;
            meta_buffers.copy_from(&meta, &[pos])?;
            graph.launch()?;
            dev.synchronize()?;

            let plain_ids = Tensor::from_vec(vec![tok], (1, 1), &dev)?;
            let plain_out = plain_model.forward(
                &plain_ids,
                &[pos],
                vec![(0, 1)],
                Some((plain_kv.clone(), &meta)),
            )?;
            dev.synchronize()?;

            let got_tok = graph_logits
                .argmax(D::Minus1)?
                .flatten_all()?
                .to_vec1::<u32>()?;
            let want_tok = plain_out
                .argmax(D::Minus1)?
                .flatten_all()?
                .to_vec1::<u32>()?;
            assert_eq!(
                got_tok, want_tok,
                "post-stand-down CUDA graph token mismatch at pos {pos}"
            );
            let got_v = graph_logits
                .to_dtype(DType::F32)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            let want_v = plain_out
                .to_dtype(DType::F32)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            let max_diff = got_v
                .iter()
                .zip(&want_v)
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            assert!(
                max_diff < 2e-2,
                "post-stand-down CUDA graph logits mismatch at pos {pos}: max_diff={max_diff}"
            );
        }

        // Verify final GDN recurrent/conv state parity against plain stepwise decode.
        let (spec_conv, spec_rec) = {
            let hybrid = spec_model.cache.hybrid();
            let Some(crate::kv_cache::HybridLayerCache::Recurrent(pool)) = hybrid.get(0) else {
                hanzo_ml::bail!("expected blk.0 to be GDN recurrent pool");
            };
            (
                pool.conv_state.i(spec_slot)?,
                pool.recurrent_state.i(spec_slot)?,
            )
        };
        let (plain_conv, plain_rec) = {
            let hybrid = plain_model.cache.hybrid();
            let Some(crate::kv_cache::HybridLayerCache::Recurrent(pool)) = hybrid.get(0) else {
                hanzo_ml::bail!("expected blk.0 to be GDN recurrent pool");
            };
            (
                pool.conv_state.i(plain_slot)?,
                pool.recurrent_state.i(plain_slot)?,
            )
        };
        let conv_diff = (&spec_conv - &plain_conv)?
            .to_dtype(DType::F32)?
            .abs()?
            .flatten_all()?
            .max(0)?
            .to_scalar::<f32>()?;
        let rec_diff = (&spec_rec - &plain_rec)?
            .to_dtype(DType::F32)?
            .abs()?
            .flatten_all()?
            .max(0)?
            .to_scalar::<f32>()?;
        assert!(
            conv_diff < 1e-2 && rec_diff < 1e-2,
            "final Qwen3.5 GDN state mismatch after stand-down+graph: conv={conv_diff}, rec={rec_diff}"
        );
        Ok(())
    }

    #[test]
    fn qwen35_rmsnorm_sign_folding_is_bit_exact() -> Result<()> {
        let mut devices = vec![Device::Cpu];
        #[cfg(feature = "cuda")]
        if let Ok(cuda_dev) = Device::new_cuda(0) {
            devices.push(cuda_dev);
        }

        for dev in devices {
            for &block in &[1024usize, 4096] {
                let width = block;
                let sign_vals: Vec<i32> = (0..width)
                    .map(|i| if (i * 11 + 5) % 2 == 0 { 1 } else { -1 })
                    .collect();
                let mut kv = HashMap::new();
                kv.insert("prism.hadamard.version".into(), gguf_file::Value::U32(1));
                kv.insert(
                    "prism.hadamard.block_size".into(),
                    gguf_file::Value::U32(block as u32),
                );
                kv.insert(
                    "prism.hadamard.transform".into(),
                    gguf_file::Value::String("normalized-sylvester-walsh-hadamard".into()),
                );
                kv.insert(
                    "prism.hadamard.axis".into(),
                    gguf_file::Value::String("input-last-dimension".into()),
                );
                kv.insert(
                    "prism.hadamard.sign_mode".into(),
                    gguf_file::Value::String("explicit".into()),
                );
                kv.insert(
                    "prism.hadamard.sign_widths".into(),
                    gguf_file::Value::Array(vec![gguf_file::Value::I32(width as i32)]),
                );
                kv.insert(
                    "prism.hadamard.sign_values".into(),
                    gguf_file::Value::Array(
                        sign_vals
                            .iter()
                            .map(|&v| gguf_file::Value::I8(v as i8))
                            .collect(),
                    ),
                );
                kv.insert(
                    "prism.hadamard.weight_names".into(),
                    gguf_file::Value::Array(vec![gguf_file::Value::String(
                        "blk.0.ffn_gate.weight".into(),
                    )]),
                );

                let hm = HadamardMeta::from_gguf_metadata(&kv)?.unwrap();
                let had = DeviceHadamard::new(&hm, &dev)?;
                let signs = had.signs_for(width)?.unwrap();

                let norm_w_vals: Vec<f32> = (0..width)
                    .map(|i| 0.75 + ((i as f32) * 0.013).sin() * 0.4)
                    .collect();
                let norm =
                    QRmsNorm::from_weight(Tensor::from_vec(norm_w_vals, (width,), &dev)?, 1e-6);
                let signed_norm = norm.mul_weight(signs)?;

                let x_vals: Vec<f32> = (0..(2 * width))
                    .map(|i| ((i as f32) * 0.021 - 1.1).cos())
                    .collect();
                let x_f32 = Tensor::from_vec(x_vals, (2, width), &dev)?;

                let dtypes: &[DType] = if dev.is_cuda() {
                    &[DType::F32, DType::F16, DType::BF16]
                } else {
                    &[DType::F32]
                };
                for &dt in dtypes {
                    let x = x_f32.to_dtype(dt)?;
                    let unfolded = had.forward(&norm.forward(&x)?)?;
                    let folded = had.forward_presigned(&signed_norm.forward(&x)?)?;
                    let u_v = unfolded
                        .to_dtype(DType::F32)?
                        .flatten_all()?
                        .to_vec1::<f32>()?;
                    let f_v = folded
                        .to_dtype(DType::F32)?
                        .flatten_all()?
                        .to_vec1::<f32>()?;
                    assert_eq!(
                        u_v, f_v,
                        "sign-folded RMSNorm + FWHT must be IEEE-754 bit-exact (block={block}, dt={dt:?}, dev={dev:?})"
                    );
                }
            }
        }
        Ok(())
    }
}
