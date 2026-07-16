#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

//! Mamba (selective state-space) decoder — zero-config loader for `MambaForCausalLM`
//! (e.g. `state-spaces/mamba-130m-hf`).
//!
//! Pure Mamba-1: a per-channel transition `A: (d_inner, d_state)`, `B`/`C: (d_state)` shared
//! across channels, and `dt` from `dt_proj`. No attention, no rotary — a depthwise causal
//! conv1d feeds a selective scan, gated by `silu(z)`. The scan is the reference
//! `slow_forward` from HF `modeling_mamba.py`, carried out per timestep in plain `hanzo_ml`
//! ops.
//!
//! The recurrence is stateful: `conv_state` + `ssm_state` persist across forward calls
//! (interior-mutable, like the KV cache) and reset at the start of a sequence
//! (`seqlen_offset == 0`), so incremental decode stays correct. CPU + F32 is the tested path.

use std::sync::{Arc, Mutex};

use hanzo_ml::{DType, Device, Module, Result, Tensor};
use hanzo_nn::Embedding;
use hanzo_quant::{
    QuantMethod, QuantMethodConfig, QuantizedConfig, ShardedVarBuilder, UnquantLinear,
};
use serde::{Deserialize, Serialize};

use crate::{
    amoe::AnyMoeBaseModelMixin,
    device_map::DeviceMapper,
    layers::{embedding, Activation, RmsNorm},
    models::gdn::softplus,
    paged_attention::{AttentionImplementation, ModelConfigMetadata},
    pipeline::{
        text_models_inputs_processor::FlashParams, EitherCache, IsqModel, NormalCache,
        NormalLoadingMetadata, NormalModel,
    },
    utils::{progress::NiceProgressBar, unvarbuilder::UnVarBuilder},
};

/// Mamba has no positional bound; this only caps the (unused) dummy cache and scheduler.
const MAX_SEQ_LEN: usize = 8192;

fn default_state_size() -> usize {
    16
}
fn default_conv_kernel() -> usize {
    4
}
fn default_expand() -> usize {
    2
}
fn default_layer_norm_epsilon() -> f64 {
    1e-5
}
fn default_hidden_act() -> Activation {
    Activation::Silu
}
fn default_use_conv_bias() -> bool {
    true
}
fn default_tie_word_embeddings() -> bool {
    true
}

/// `time_step_rank` is `"auto"` in some configs and an integer in saved checkpoints.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(untagged)]
pub enum TimeStepRank {
    Auto(String),
    Value(usize),
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    #[serde(default = "default_state_size")]
    pub state_size: usize,
    pub num_hidden_layers: usize,
    #[serde(default = "default_conv_kernel")]
    pub conv_kernel: usize,
    #[serde(default = "default_expand")]
    pub expand: usize,
    pub intermediate_size: Option<usize>,
    pub time_step_rank: Option<TimeStepRank>,
    #[serde(default = "default_layer_norm_epsilon")]
    pub layer_norm_epsilon: f64,
    #[serde(default = "default_hidden_act")]
    pub hidden_act: Activation,
    #[serde(default = "default_use_conv_bias")]
    pub use_conv_bias: bool,
    #[serde(default)]
    pub use_bias: bool,
    #[serde(default = "default_tie_word_embeddings")]
    pub tie_word_embeddings: bool,
    pub quantization_config: Option<QuantizedConfig>,
}

impl Config {
    /// Inner (expanded) width the mixer runs at.
    pub fn d_inner(&self) -> usize {
        self.intermediate_size
            .unwrap_or(self.expand * self.hidden_size)
    }

    /// Rank of the low-rank `dt` projection; `"auto"` resolves to `ceil(hidden_size / 16)`.
    pub fn dt_rank(&self) -> usize {
        match &self.time_step_rank {
            Some(TimeStepRank::Value(v)) => *v,
            _ => self.hidden_size.div_ceil(16),
        }
    }
}

/// A plain (unquantized) linear from an HF `[out, in]` weight, wrapped so ISQ still applies.
fn linear(
    in_f: usize,
    out_f: usize,
    bias: bool,
    vb: ShardedVarBuilder,
) -> Result<Arc<dyn QuantMethod>> {
    let w = vb.get((out_f, in_f), "weight")?;
    let b = if bias {
        Some(vb.get(out_f, "bias")?)
    } else {
        None
    };
    Ok(Arc::new(UnquantLinear::new(
        QuantMethodConfig::Unquantized(hanzo_nn::Linear::new(w, b)),
    )?))
}

/// Per-layer recurrent state: the trailing `conv_kernel-1` conv inputs and the
/// `d_inner × d_state` SSM state. Zero-initialized at the start of a sequence.
#[derive(Clone)]
struct MambaLayerState {
    conv: Tensor, // (b, d_inner, conv_kernel - 1)
    ssm: Tensor,  // (b, d_inner, d_state)
}

impl MambaLayerState {
    fn zeros(
        b: usize,
        d_inner: usize,
        d_state: usize,
        conv_kernel: usize,
        dtype: DType,
        device: &Device,
    ) -> Result<Self> {
        Ok(Self {
            conv: Tensor::zeros((b, d_inner, conv_kernel - 1), dtype, device)?,
            ssm: Tensor::zeros((b, d_inner, d_state), dtype, device)?,
        })
    }
}

struct MambaMixer {
    in_proj: Arc<dyn QuantMethod>,
    conv1d_weight: Tensor, // (d_inner, 1, conv_kernel), as stored
    conv1d_bias: Option<Tensor>,
    x_proj: Arc<dyn QuantMethod>,
    dt_proj: Arc<dyn QuantMethod>,
    a: Tensor,     // (d_inner, d_state) = -exp(A_log), for the scan
    a_log: Tensor, // (d_inner, d_state), as stored, for serialization
    d: Tensor,     // (d_inner)
    out_proj: Arc<dyn QuantMethod>,
    d_inner: usize,
    dt_rank: usize,
    d_state: usize,
    conv_kernel: usize,
    act: Activation,
}

impl MambaMixer {
    fn new(cfg: &Config, vb: ShardedVarBuilder) -> Result<Self> {
        let d_inner = cfg.d_inner();
        let dt_rank = cfg.dt_rank();
        let d_state = cfg.state_size;
        let in_proj = linear(cfg.hidden_size, 2 * d_inner, cfg.use_bias, vb.pp("in_proj"))?;
        let conv1d_weight = vb.get((d_inner, 1, cfg.conv_kernel), "conv1d.weight")?;
        let conv1d_bias = if cfg.use_conv_bias {
            Some(vb.get(d_inner, "conv1d.bias")?)
        } else {
            None
        };
        let x_proj = linear(d_inner, dt_rank + 2 * d_state, false, vb.pp("x_proj"))?;
        let dt_proj = linear(dt_rank, d_inner, true, vb.pp("dt_proj"))?;
        // `A` is parameterized in log space; the effective transition is `-exp(A_log)`.
        let a_log = vb.get((d_inner, d_state), "A_log")?;
        let a = a_log.exp()?.neg()?;
        let d = vb.get(d_inner, "D")?;
        let out_proj = linear(d_inner, cfg.hidden_size, cfg.use_bias, vb.pp("out_proj"))?;
        Ok(Self {
            in_proj,
            conv1d_weight,
            conv1d_bias,
            x_proj,
            dt_proj,
            a,
            a_log,
            d,
            out_proj,
            d_inner,
            dt_rank,
            d_state,
            conv_kernel: cfg.conv_kernel,
            act: cfg.hidden_act,
        })
    }

    /// Depthwise causal conv1d seeded from `state.conv`, then updates it to the trailing
    /// window. `x`: (b, d_inner, l) -> (b, d_inner, l). Prepending the saved state and doing
    /// a valid (unpadded) conv is exactly HF's left-padded causal conv, and continues it
    /// across chunks/decode steps.
    fn causal_conv(&self, x: &Tensor, state: &mut MambaLayerState) -> Result<Tensor> {
        let l = x.dim(2)?;
        let k = self.conv_kernel;
        let w = self.conv1d_weight.reshape((self.d_inner, k))?;
        let xc = Tensor::cat(&[&state.conv, x], 2)?; // (b, d_inner, k-1 + l)
        let mut acc: Option<Tensor> = None;
        for j in 0..k {
            let wj = w.narrow(1, j, 1)?.reshape((1, self.d_inner, 1))?;
            let term = xc.narrow(2, j, l)?.broadcast_mul(&wj)?;
            acc = Some(match acc {
                Some(prev) => (prev + term)?,
                None => term,
            });
        }
        let mut y = acc.unwrap();
        if let Some(bias) = &self.conv1d_bias {
            y = y.broadcast_add(&bias.reshape((1, self.d_inner, 1))?)?;
        }
        state.conv = xc.narrow(2, l, k - 1)?.contiguous()?;
        Ok(y)
    }

    /// The selective scan (HF `slow_forward`), per timestep, seeded from `state.ssm`.
    /// `u`: (b, d_inner, l); `dt`: (b, l, d_inner); `b_mat`/`c_mat`: (b, l, d_state).
    /// Returns (b, d_inner, l).
    fn selective_scan(
        &self,
        u: &Tensor,
        dt: &Tensor,
        b_mat: &Tensor,
        c_mat: &Tensor,
        state: &mut MambaLayerState,
    ) -> Result<Tensor> {
        let (b, _, l) = u.dims3()?;
        let dt = dt.transpose(1, 2)?.contiguous()?; // (b, d_inner, l)
        let a = self.a.reshape((1, self.d_inner, self.d_state))?;
        let mut ys = Vec::with_capacity(l);
        for t in 0..l {
            let dt_t = dt.narrow(2, t, 1)?.reshape((b, self.d_inner, 1))?;
            let u_t = u.narrow(2, t, 1)?.reshape((b, self.d_inner, 1))?;
            let b_t = b_mat.narrow(1, t, 1)?.reshape((b, 1, self.d_state))?;
            let c_t = c_mat.narrow(1, t, 1)?.reshape((b, 1, self.d_state))?;
            // discrete_A = exp(dt · A); discrete_B·u = dt · B · u.
            let da = dt_t.broadcast_mul(&a)?.exp()?; // (b, d_inner, d_state)
            let dbu = dt_t.broadcast_mul(&b_t)?.broadcast_mul(&u_t)?; // (b, d_inner, d_state)
            state.ssm = (state.ssm.mul(&da)? + dbu)?;
            // y_t = (state · C) contracted over the state dim.
            ys.push(state.ssm.broadcast_mul(&c_t)?.sum_keepdim(2)?); // (b, d_inner, 1)
        }
        Tensor::cat(&ys, 2)
    }

    fn forward(&self, xs: &Tensor, state: &mut MambaLayerState) -> Result<Tensor> {
        let proj = self.in_proj.forward(xs)?; // (b, l, 2*d_inner)
        let hidden = proj.narrow(2, 0, self.d_inner)?;
        let gate = proj.narrow(2, self.d_inner, self.d_inner)?;

        // Conv over time -> silu.
        let hidden = hidden.transpose(1, 2)?.contiguous()?; // (b, d_inner, l)
        let hidden = self.causal_conv(&hidden, state)?.apply(&self.act)?;

        // Project to (dt, B, C).
        let ssm_params = self
            .x_proj
            .forward(&hidden.transpose(1, 2)?.contiguous()?)?;
        let dt = ssm_params.narrow(2, 0, self.dt_rank)?;
        let b_mat = ssm_params.narrow(2, self.dt_rank, self.d_state)?;
        let c_mat = ssm_params.narrow(2, self.dt_rank + self.d_state, self.d_state)?;
        let dt = softplus(&self.dt_proj.forward(&dt)?)?; // (b, l, d_inner)

        // Scan, skip connection D·u, output gate silu(z).
        let y = self.selective_scan(&hidden, &dt, &b_mat, &c_mat, state)?; // (b, d_inner, l)
        let d = self.d.reshape((1, self.d_inner, 1))?;
        let y = (y + hidden.broadcast_mul(&d)?)?;
        let gate = gate.transpose(1, 2)?.contiguous()?; // (b, d_inner, l)
        let y = (y * gate.apply(&self.act)?)?;

        self.out_proj.forward(&y.transpose(1, 2)?.contiguous()?)
    }
}

struct Block {
    norm: RmsNorm,
    mixer: MambaMixer,
}

impl Block {
    fn new(
        cfg: &Config,
        vb: ShardedVarBuilder,
        mapper: &dyn DeviceMapper,
        layer_idx: usize,
        loading_isq: bool,
    ) -> Result<Self> {
        let norm = RmsNorm::new(
            cfg.hidden_size,
            cfg.layer_norm_epsilon,
            mapper.set_device(layer_idx, vb.pp("norm"), false),
        )?;
        let mixer = MambaMixer::new(
            cfg,
            mapper.set_device(layer_idx, vb.pp("mixer"), loading_isq),
        )?;
        Ok(Self { norm, mixer })
    }

    fn forward(&self, xs: &Tensor, state: &mut MambaLayerState) -> Result<Tensor> {
        let residual = xs;
        let xs = self.norm.forward(xs)?;
        let xs = self.mixer.forward(&xs, state)?;
        residual + xs
    }
}

pub struct Model {
    embeddings: Embedding,
    blocks: Vec<Block>,
    norm_f: RmsNorm,
    lm_head: Arc<dyn QuantMethod>,
    state: Mutex<Vec<MambaLayerState>>,
    d_inner: usize,
    d_state: usize,
    conv_kernel: usize,
    dtype: DType,
    device: Device,
    cache: EitherCache,
    max_seq_len: usize,
    mapper: Box<dyn DeviceMapper + Send + Sync>,
    cfg: ModelConfigMetadata,
}

impl Model {
    pub fn new(
        cfg: &Config,
        vb: ShardedVarBuilder,
        _is_gptx: bool,
        normal_loading_metadata: NormalLoadingMetadata,
        _attention_mechanism: AttentionImplementation,
    ) -> Result<Self> {
        if let Some(ref quant_cfg) = &cfg.quantization_config {
            tracing::info!(
                "Using {} quantization: {}.",
                quant_cfg.name(),
                quant_cfg.get_bits_name(&vb)
            );
        }
        let dtype = vb.dtype();
        let mapper = normal_loading_metadata.mapper;
        let vb_m = vb.pp("backbone");

        let embeddings = embedding(
            cfg.vocab_size,
            cfg.hidden_size,
            mapper.set_nm_device(vb_m.pp("embeddings"), false),
            &cfg.quantization_config,
        )?;

        let vb_l = vb_m.pp("layers");
        let blocks: Vec<Block> = NiceProgressBar::<_, 'b'>(
            0..cfg.num_hidden_layers,
            "Loading repeating layers",
            &normal_loading_metadata.multi_progress,
        )
        .par_iter_if_isq(|layer_idx| {
            Block::new(
                cfg,
                vb_l.pp(layer_idx),
                &*mapper,
                layer_idx,
                normal_loading_metadata.loading_isq,
            )
        })?;

        let norm_f = RmsNorm::new(
            cfg.hidden_size,
            cfg.layer_norm_epsilon,
            mapper.set_nm_device(vb_m.pp("norm_f"), false),
        )?;

        // The head ties to the token embedding unless the checkpoint carries its own.
        let lm_head = if cfg.tie_word_embeddings {
            Arc::new(UnquantLinear::new(QuantMethodConfig::Unquantized(
                hanzo_nn::Linear::new(
                    mapper.cast_nm_device(
                        embeddings.embeddings(),
                        normal_loading_metadata.loading_isq,
                    )?,
                    None,
                ),
            ))?)
        } else {
            linear(
                cfg.hidden_size,
                cfg.vocab_size,
                false,
                mapper.set_nm_device(vb.pp("lm_head"), normal_loading_metadata.loading_isq),
            )?
        };

        Ok(Self {
            embeddings,
            blocks,
            norm_f,
            lm_head,
            state: Mutex::new(Vec::new()),
            d_inner: cfg.d_inner(),
            d_state: cfg.state_size,
            conv_kernel: cfg.conv_kernel,
            dtype,
            device: normal_loading_metadata.real_device.clone(),
            cache: EitherCache::Normal(NormalCache::new(cfg.num_hidden_layers, MAX_SEQ_LEN)),
            max_seq_len: MAX_SEQ_LEN,
            cfg: ModelConfigMetadata {
                max_seq_len: MAX_SEQ_LEN,
                num_layers: cfg.num_hidden_layers,
                hidden_size: cfg.hidden_size,
                // No attention: placeholders kept non-zero for memory/device-map math only.
                num_attn_heads: 1,
                num_kv_heads: 1,
                sliding_window: None,
                k_head_dim: cfg.hidden_size,
                v_head_dim: cfg.hidden_size,
                kv_cache_layout: crate::paged_attention::KvCacheLayout::Standard,
            },
            mapper,
        })
    }

    pub fn forward(
        &self,
        input_ids: &Tensor,
        ctx: &mut crate::pipeline::ModelForwardContext<'_>,
    ) -> Result<Tensor> {
        let (b, _l) = input_ids.dims2()?;
        let mut xs = self.embeddings.forward(input_ids)?;

        let n = self.blocks.len();
        // A sequence starts at offset 0; reset the recurrent state then (also if the batch
        // width changed) so decode continues from carried state rather than recomputing.
        let fresh = ctx.seqlen_offsets().first().copied().unwrap_or(0) == 0;
        let mut states = self.state.lock().expect("mamba state poisoned");
        let need_reset = fresh
            || states.len() != n
            || states.first().map(|s| s.ssm.dim(0).unwrap_or(0)) != Some(b);
        if need_reset {
            let mut fresh_states = Vec::with_capacity(n);
            for i in 0..n {
                let device = self.mapper.device_for(i, false).unwrap_or(&self.device);
                fresh_states.push(MambaLayerState::zeros(
                    b,
                    self.d_inner,
                    self.d_state,
                    self.conv_kernel,
                    self.dtype,
                    device,
                )?);
            }
            *states = fresh_states;
        }

        for (i, block) in self.blocks.iter().enumerate() {
            xs = self.mapper.map(xs, i)?;
            xs = block.forward(&xs, &mut states[i])?;
        }
        drop(states);

        let xs = xs.to_device(&self.device)?.apply(&self.norm_f)?;
        let xs = ctx.logits(&xs)?;
        self.lm_head.forward(&xs)
    }
}

impl IsqModel for Model {
    fn get_layers(
        &mut self,
    ) -> (
        Vec<(&mut Arc<dyn QuantMethod>, Option<usize>)>,
        &dyn DeviceMapper,
    ) {
        let mut tensors = Vec::new();
        tensors.push((&mut self.lm_head, None));
        for (i, block) in self.blocks.iter_mut().enumerate() {
            tensors.push((&mut block.mixer.in_proj, Some(i)));
            tensors.push((&mut block.mixer.x_proj, Some(i)));
            tensors.push((&mut block.mixer.dt_proj, Some(i)));
            tensors.push((&mut block.mixer.out_proj, Some(i)));
        }
        (tensors, &*self.mapper)
    }

    fn residual_tensors(&self) -> Vec<(String, Tensor)> {
        let uvb = UnVarBuilder::new();
        let uvb_m = uvb.pp("backbone");
        uvb_m.pp("embeddings").add(&self.embeddings);
        uvb_m.pp("norm_f").add(&self.norm_f);
        for (i, block) in self.blocks.iter().enumerate() {
            let uvb_l = uvb_m.pp("layers").pp(i);
            uvb_l.pp("norm").add(&block.norm);
            let mixer = uvb_l.pp("mixer");
            mixer.add_tensor("conv1d.weight", block.mixer.conv1d_weight.clone());
            if let Some(bias) = &block.mixer.conv1d_bias {
                mixer.add_tensor("conv1d.bias", bias.clone());
            }
            mixer.add_tensor("A_log", block.mixer.a_log.clone());
            mixer.add_tensor("D", block.mixer.d.clone());
        }
        uvb.to_safetensors()
    }
}

impl crate::speculative::SpeculativeTargetMixin for Model {}

impl NormalModel for Model {
    fn forward(
        &self,
        input_ids: &Tensor,
        ctx: &mut crate::pipeline::ModelForwardContext<'_>,
    ) -> Result<Tensor> {
        self.forward(input_ids, ctx)
    }
    fn xlora_forward(
        &self,
        _input_ids: &Tensor,
        _input_ids_full: &Tensor,
        _seqlen_offsets: &[usize],
        _seqlen_offsets_full: &[usize],
        _no_kv_cache: bool,
        _non_granular_state: &Option<crate::xlora_models::NonGranularState>,
        _context_lens: Vec<(usize, usize)>,
        _position_ids: Vec<usize>,
        _flash_params: &FlashParams,
        _flash_params_full: &FlashParams,
    ) -> Result<Tensor> {
        unimplemented!()
    }
    fn cache(&self) -> &EitherCache {
        &self.cache
    }
    fn cache_mut(&mut self) -> &mut EitherCache {
        &mut self.cache
    }
    fn device(&self) -> &Device {
        &self.device
    }
    fn is_xlora(&self) -> bool {
        false
    }
    fn max_seq_len(&self) -> usize {
        self.max_seq_len
    }
    fn config(&self) -> &ModelConfigMetadata {
        &self.cfg
    }
}

impl AnyMoeBaseModelMixin for Model {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::ModelForwardContext;
    use crate::DeviceMapSetting;
    use hanzo_ml::{safetensors, DType};
    use indicatif::MultiProgress;
    use rand::{rngs::StdRng, Rng, SeedableRng};

    // The arch block of the real `state-spaces/mamba-130m-hf` config.json. Embedded for a
    // hermetic parse test.
    const REAL_CONFIG: &str = r#"{
        "architectures": ["MambaForCausalLM"],
        "conv_kernel": 4,
        "expand": 2,
        "hidden_act": "silu",
        "hidden_size": 768,
        "intermediate_size": 1536,
        "layer_norm_epsilon": 1e-05,
        "model_type": "mamba",
        "num_hidden_layers": 24,
        "state_size": 16,
        "time_step_rank": 48,
        "use_bias": false,
        "use_conv_bias": true,
        "vocab_size": 50280
    }"#;

    const TOY_CONFIG: &str = r#"{
        "hidden_act": "silu",
        "hidden_size": 32,
        "state_size": 8,
        "num_hidden_layers": 2,
        "conv_kernel": 4,
        "expand": 2,
        "time_step_rank": 2,
        "layer_norm_epsilon": 1e-05,
        "use_bias": false,
        "use_conv_bias": true,
        "vocab_size": 64
    }"#;

    fn loading_metadata(device: &Device, num_layers: usize) -> Result<NormalLoadingMetadata> {
        let mapper = DeviceMapSetting::dummy().into_mapper(
            num_layers,
            device,
            None,
            std::slice::from_ref(device),
        )?;
        Ok(NormalLoadingMetadata {
            mapper,
            loading_isq: false,
            real_device: device.clone(),
            multi_progress: Arc::new(MultiProgress::new()),
            matformer_slicing_config: None,
        })
    }

    // Random-init checkpoint with the exact real Mamba tensor keys (backbone prefix, the
    // `(d_inner, 1, conv_kernel)` conv layout, `A_log`/`D`), tied head.
    fn write_checkpoint(cfg: &Config, dir: &std::path::Path) -> Result<std::path::PathBuf> {
        let mut rng = StdRng::seed_from_u64(0x6d616d62);
        let mut t = |shape: &[usize]| -> Result<Tensor> {
            let n: usize = shape.iter().product();
            let data: Vec<f32> = (0..n).map(|_| rng.random_range(-0.08f32..0.08)).collect();
            Tensor::from_vec(data, shape, &Device::Cpu)
        };
        let one = |n: usize| Tensor::ones((n,), DType::F32, &Device::Cpu);
        let zero = |n: usize| Tensor::zeros((n,), DType::F32, &Device::Cpu);
        let h = cfg.hidden_size;
        let di = cfg.d_inner();
        let dr = cfg.dt_rank();
        let ds = cfg.state_size;
        let k = cfg.conv_kernel;
        let mut ts = std::collections::HashMap::new();
        ts.insert(
            "backbone.embeddings.weight".to_string(),
            t(&[cfg.vocab_size, h])?,
        );
        ts.insert("backbone.norm_f.weight".to_string(), one(h)?);
        for i in 0..cfg.num_hidden_layers {
            let p = format!("backbone.layers.{i}");
            ts.insert(format!("{p}.norm.weight"), one(h)?);
            ts.insert(format!("{p}.mixer.in_proj.weight"), t(&[2 * di, h])?);
            ts.insert(format!("{p}.mixer.conv1d.weight"), t(&[di, 1, k])?);
            ts.insert(format!("{p}.mixer.conv1d.bias"), zero(di)?);
            ts.insert(format!("{p}.mixer.x_proj.weight"), t(&[dr + 2 * ds, di])?);
            ts.insert(format!("{p}.mixer.dt_proj.weight"), t(&[di, dr])?);
            ts.insert(format!("{p}.mixer.dt_proj.bias"), zero(di)?);
            // A_log = 0 -> A = -1: a stable, well-conditioned toy transition.
            ts.insert(
                format!("{p}.mixer.A_log"),
                Tensor::zeros((di, ds), DType::F32, &Device::Cpu)?,
            );
            ts.insert(format!("{p}.mixer.D"), one(di)?);
            ts.insert(format!("{p}.mixer.out_proj.weight"), t(&[h, di])?);
        }
        let path = dir.join("model.safetensors");
        safetensors::save(&ts, &path)?;
        Ok(path)
    }

    fn load_toy(path: &std::path::Path, cfg: &Config, device: &Device) -> Result<Model> {
        let vb = crate::utils::varbuilder_utils::from_mmaped_safetensors(
            vec![path.to_path_buf()],
            Vec::new(),
            Some(DType::F32),
            device,
            vec![None],
            true,
            None,
            |_| true,
            Arc::new(|_| crate::utils::varbuilder_utils::DeviceForLoadTensor::Base),
        )?;
        let metadata = loading_metadata(device, cfg.num_hidden_layers)?;
        Model::new(cfg, vb, false, metadata, AttentionImplementation::Eager)
    }

    fn forward_ids(model: &Model, ids: &[u32], device: &Device) -> Result<Tensor> {
        let seq_len = ids.len();
        let input_ids = Tensor::from_vec(ids.to_vec(), (1, seq_len), device)?;
        let seqlen_offsets = vec![0usize];
        let context_lens = vec![(0usize, seq_len)];
        let position_ids: Vec<usize> = (0..seq_len).collect();
        let flash_params = FlashParams::empty(true);
        let mut ctx = ModelForwardContext::new(
            &seqlen_offsets,
            &context_lens,
            &position_ids,
            None,
            &flash_params,
        );
        model.forward(&input_ids, &mut ctx)
    }

    #[test]
    fn mamba_config_parses() {
        let cfg: Config = serde_json::from_str(REAL_CONFIG).unwrap();
        assert_eq!(cfg.num_hidden_layers, 24);
        assert_eq!(cfg.hidden_size, 768);
        assert_eq!(cfg.state_size, 16);
        assert_eq!(cfg.vocab_size, 50280);
        assert_eq!(cfg.d_inner(), 1536);
        assert_eq!(cfg.dt_rank(), 48);
        assert!(matches!(cfg.hidden_act, Activation::Silu));
    }

    #[test]
    fn mamba_registry_dispatch() {
        use crate::pipeline::NormalLoaderType;
        assert_eq!(
            NormalLoaderType::from_causal_lm_name("MambaForCausalLM").unwrap(),
            NormalLoaderType::Mamba
        );
    }

    #[test]
    fn mamba_toy_forward_causal() -> Result<()> {
        let device = Device::Cpu;
        let cfg: Config = serde_json::from_str(TOY_CONFIG).unwrap();
        let dir = tempfile::tempdir().map_err(hanzo_ml::Error::wrap)?;
        let path = write_checkpoint(&cfg, dir.path())?;

        let model = load_toy(&path, &cfg, &device)?;
        let logits = forward_ids(&model, &[1u32, 2, 3, 4, 5], &device)?;
        assert_eq!(logits.dims3()?, (1, 5, cfg.vocab_size));
        let flat = logits.flatten_all()?.to_vec1::<f32>()?;
        assert!(flat.iter().all(|x| x.is_finite()), "logits must be finite");

        // Causality by construction: changing a LATER token must not change EARLIER logits.
        let model_a = load_toy(&path, &cfg, &device)?;
        let a = forward_ids(&model_a, &[1u32, 2, 3, 4, 5], &device)?;
        let model_b = load_toy(&path, &cfg, &device)?;
        let b = forward_ids(&model_b, &[1u32, 2, 3, 4, 9], &device)?;
        let a_past = a.narrow(1, 0, 4)?.flatten_all()?.to_vec1::<f32>()?;
        let b_past = b.narrow(1, 0, 4)?.flatten_all()?.to_vec1::<f32>()?;
        for (x, y) in a_past.iter().zip(b_past.iter()) {
            assert!((x - y).abs() < 1e-4, "past logits changed: {x} vs {y}");
        }
        Ok(())
    }
}
