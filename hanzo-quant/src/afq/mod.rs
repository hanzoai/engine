use std::{
    borrow::Cow,
    io::Cursor,
    sync::{atomic::AtomicUsize, Arc},
};

use byteorder::{LittleEndian, ReadBytesExt};
use hanzo_ml::{DType, Device, Result, Tensor, D};
use hanzo_nn::Linear;

use crate::{
    utils::{
        deserialize_tensor, fake_deserialize_tensor, serialize_tensor, version_is_compatible,
        UQFF_VERSION,
    },
    Comm, IsqType, QuantMethod, QuantMethodConfig, QuantizeOntoGuard, QuantizedConfig,
    QuantizedSerde, QuantizedSerdeType, ShardedVarBuilder, UnquantLinear,
};

pub mod ops;

#[cfg(feature = "cuda")]
pub(crate) mod ffi;

#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AfqBits {
    Two = 2,
    Three = 3,
    Four = 4,
    Six = 6,
    Eight = 8,
    Mxfp4 = 40,
}

impl TryFrom<usize> for AfqBits {
    type Error = hanzo_ml::Error;
    fn try_from(value: usize) -> Result<Self> {
        match value {
            2 => Ok(Self::Two),
            3 => Ok(Self::Three),
            4 => Ok(Self::Four),
            6 => Ok(Self::Six),
            8 => Ok(Self::Eight),
            40 => Ok(Self::Mxfp4),
            x => hanzo_ml::bail!("Invalid AFQ bits {x}."),
        }
    }
}

impl TryFrom<u8> for AfqBits {
    type Error = hanzo_ml::Error;
    fn try_from(value: u8) -> Result<Self> {
        Self::try_from(value as usize)
    }
}

#[repr(u8)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum AfqGroupSize {
    Low = 32,
    #[default]
    Med = 64,
    High = 128,
}

impl TryFrom<usize> for AfqGroupSize {
    type Error = hanzo_ml::Error;
    fn try_from(value: usize) -> Result<Self> {
        match value {
            32 => Ok(Self::Low),
            64 => Ok(Self::Med),
            128 => Ok(Self::High),
            x => hanzo_ml::bail!("Invalid AFQ group size {x}."),
        }
    }
}

impl TryFrom<u8> for AfqGroupSize {
    type Error = hanzo_ml::Error;
    fn try_from(value: u8) -> Result<Self> {
        Self::try_from(value as usize)
    }
}

const MAX_CUDA_FWHT_BLOCK: usize = 8192;

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

fn fwht_typed(
    x: &Tensor,
    block: usize,
    signs: &Tensor,
    inverse: bool,
    out_dtype: DType,
) -> Result<Tensor> {
    if block == 0 || !block.is_power_of_two() {
        hanzo_ml::bail!("Hadamard block size must be a power of two, got {block}");
    }
    let orig_shape = x.shape().clone();
    let last_dim = x.dim(D::Minus1)?;
    if last_dim == 0 || !last_dim.is_multiple_of(block) {
        hanzo_ml::bail!("fwht expects last dim ({last_dim}) divisible by block ({block})");
    }
    if (2..=MAX_CUDA_FWHT_BLOCK).contains(&block) {
        let xin = match x.dtype() {
            DType::F32 | DType::F16 | DType::BF16 => Cow::Borrowed(x),
            _ => Cow::Owned(x.to_dtype(DType::F32)?),
        };
        let out = hanzo_ml::quantized::hadamard_fwht(xin.as_ref(), Some(signs), block, inverse)?;
        return if out.dtype() != out_dtype {
            out.to_dtype(out_dtype)
        } else {
            Ok(out)
        };
    }
    let mut data = x.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
    let signs_vec = signs
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    if signs_vec.len() != last_dim {
        hanzo_ml::bail!(
            "fwht sign width ({}) does not match last dim ({last_dim})",
            signs_vec.len()
        );
    }
    cpu_fwht_blocks(&mut data, last_dim, block, Some(&signs_vec), inverse);
    let out = Tensor::from_vec(data, orig_shape, x.device())?;
    if out.dtype() != out_dtype {
        out.to_dtype(out_dtype)
    } else {
        Ok(out)
    }
}

#[derive(Clone, Copy)]
struct AfqLinearSpec {
    in_dim: usize,
    out_dim: usize,
    bits: usize,
    group_size: usize,
    hadamard_block: usize,
    bias: bool,
}

#[derive(Debug)]
pub struct AfqLayer {
    w_q: Tensor,
    scales: Tensor,
    biases: Tensor,
    bias: Option<Tensor>,
    signs: Option<Tensor>,
    hadamard_block: usize,
    bits: AfqBits,
    group_size: AfqGroupSize,
}

/// View of an AfqLayer's storage tensors, used by fused QKV/gate-up paths.
#[derive(Clone)]
pub struct AfqInner<'a> {
    pub w_q: &'a Tensor,
    pub scales: &'a Tensor,
    pub biases: &'a Tensor,
    pub bias: Option<&'a Tensor>,
    pub bits: AfqBits,
    pub group_size: AfqGroupSize,
}

impl QuantMethod for AfqLayer {
    fn new(method: QuantMethodConfig) -> hanzo_ml::Result<Self>
    where
        Self: Sized,
    {
        match method {
            QuantMethodConfig::Gguf { .. }
            | QuantMethodConfig::GptqAwq { .. }
            | QuantMethodConfig::Hqq { .. }
            | QuantMethodConfig::Dummy
            | QuantMethodConfig::FP8 { .. }
            | QuantMethodConfig::Bnb { .. }
            | QuantMethodConfig::BlockwiseFP8 { .. }
            | QuantMethodConfig::PerTensorFP8 { .. }
            | QuantMethodConfig::Unquantized(_)
            | QuantMethodConfig::MXFP4 { .. }
            | QuantMethodConfig::NVFP4 { .. } => unreachable!(),
            QuantMethodConfig::Afq {
                weight,
                bias,
                bits,
                group_size,
            } => {
                let (w_q, scales, biases) = ops::afq_quantize_op(&weight, group_size, bits)?;
                let mm_dtype = scales.dtype();
                let bias = bias
                    .map(|b| {
                        if b.dtype() == mm_dtype {
                            Ok(b)
                        } else {
                            b.to_dtype(mm_dtype)
                        }
                    })
                    .transpose()?;

                Ok(Self {
                    w_q,
                    scales,
                    biases,
                    bias,
                    signs: None,
                    hadamard_block: 0,
                    bits,
                    group_size,
                })
            }
        }
    }

    fn dequantize_w(&self) -> Result<hanzo_ml::Tensor> {
        let w = ops::afq_dequantize_op(
            &self.w_q,
            &self.scales,
            &self.biases,
            self.group_size,
            self.bits,
        )?;
        if let Some(ref signs) = self.signs {
            fwht_typed(&w, self.hadamard_block, signs, true, w.dtype())
        } else {
            Ok(w)
        }
    }

    fn forward_raw(&self, x: &Tensor) -> Result<Tensor> {
        let orig_dtype = x.dtype();
        let mm_dtype = self.scales.dtype();
        let x_mm = if let Some(ref signs) = self.signs {
            fwht_typed(x, self.hadamard_block, signs, false, mm_dtype)?
        } else if x.dtype() != mm_dtype {
            x.to_dtype(mm_dtype)?
        } else {
            x.clone()
        };

        let mut out = if self.w_q.rank() == 2 && x_mm.rank() > 2 {
            let k_in = x_mm.dim(D::Minus1)?;
            let mut out_dims = x_mm.dims()[..x_mm.rank() - 1].to_vec();
            let x_2d = x_mm.reshape(((), k_in))?;
            let out_2d = ops::afq_mm_op(
                &x_2d,
                &self.w_q,
                &self.scales,
                &self.biases,
                None,
                None,
                self.group_size,
                self.bits,
                true,
            )?;
            out_dims.push(out_2d.dim(D::Minus1)?);
            out_2d.reshape(out_dims)?
        } else {
            ops::afq_mm_op(
                &x_mm,
                &self.w_q,
                &self.scales,
                &self.biases,
                None,
                None,
                self.group_size,
                self.bits,
                true,
            )?
        };

        if let Some(ref b) = self.bias {
            if b.dtype() == out.dtype() {
                out = out.broadcast_add(b)?;
            } else {
                out = out.broadcast_add(&b.to_dtype(out.dtype())?)?;
            }
        }
        if out.dtype() != orig_dtype {
            out.to_dtype(orig_dtype)
        } else {
            Ok(out)
        }
    }

    fn gather_forward_raw(&self, x: &Tensor, indices: &Tensor) -> Result<Tensor> {
        let orig_dtype = x.dtype();
        let mm_dtype = self.scales.dtype();
        let x_mm = if let Some(ref signs) = self.signs {
            fwht_typed(x, self.hadamard_block, signs, false, mm_dtype)?
        } else if x.dtype() != mm_dtype {
            x.to_dtype(mm_dtype)?
        } else {
            x.clone()
        };
        let out = ops::afq_mm_op(
            &x_mm,
            &self.w_q,
            &self.scales,
            &self.biases,
            None,
            Some(indices),
            self.group_size,
            self.bits,
            true,
        )?;
        if out.dtype() != orig_dtype {
            out.to_dtype(orig_dtype)
        } else {
            Ok(out)
        }
    }

    fn quantized_act_type(&self) -> Option<DType> {
        None
    }

    fn afq_inner(&self) -> Option<crate::AfqInner<'_>> {
        if self.signs.is_some() {
            return None;
        }
        Some(crate::AfqInner {
            w_q: &self.w_q,
            scales: &self.scales,
            biases: &self.biases,
            bias: self.bias.as_ref(),
            bits: self.bits,
            group_size: self.group_size,
        })
    }

    fn add_delta_w(&self, delta: &Tensor) -> Result<Arc<dyn QuantMethod>> {
        let dequant = self.dequantize_w()?;
        Ok(Arc::new(Self::new(QuantMethodConfig::Afq {
            weight: (dequant + delta)?,
            bias: self.bias.clone(),
            bits: self.bits,
            group_size: self.group_size,
        })?))
    }

    fn dtype_and_device(&self) -> (DType, hanzo_ml::Device) {
        (self.scales.dtype(), self.scales.device().clone())
    }

    fn apply_isq(
        self: Arc<Self>,
        dtype: Option<IsqType>,
        device: Device,
        _n_quantized: &AtomicUsize,
        _imatrix_weight: Option<Vec<f32>>,
        guard: QuantizeOntoGuard,
    ) -> Result<Arc<dyn QuantMethod>> {
        match dtype {
            Some(IsqType::F8Q8) => {
                let _acquired_quantize_guard = guard.acquire(&device);
                let w = self.dequantize_w()?.to_device(&device)?;
                let b = self
                    .bias
                    .as_ref()
                    .map(|b| b.to_device(&device))
                    .transpose()?;
                Ok(Arc::new(crate::F8Q8Linear::from_weight(&w, b)?))
            }
            _ => todo!(),
        }
    }
}

impl AfqLayer {
    pub fn get_isq_type_from_uqff(data: Cow<[u8]>) -> Result<IsqType> {
        let mut buffer = Cursor::new(data.to_vec());

        let version = buffer.read_u32::<LittleEndian>()?;
        if let Err(e) = version_is_compatible(version) {
            return Err(hanzo_ml::Error::wrap(e));
        }

        let isq_type = buffer.read_u8()? as usize;
        if isq_type != QuantizedSerdeType::Afq as usize {
            hanzo_ml::bail!(
                "ISQ type ({isq_type}) doesn't match expected type {}",
                QuantizedSerdeType::Afq as usize
            );
        }

        let has_bias = buffer.read_u8()? != 0;

        // Weight, scales, biases
        fake_deserialize_tensor(&mut buffer)?;
        fake_deserialize_tensor(&mut buffer)?;
        fake_deserialize_tensor(&mut buffer)?;

        // Bits and group size
        let bits: AfqBits = buffer.read_u8()?.try_into()?;
        let _group_size: AfqGroupSize = buffer.read_u8()?.try_into()?;

        if has_bias {
            fake_deserialize_tensor(&mut buffer)?
        }

        match bits {
            AfqBits::Two => Ok(IsqType::AFQ2),
            AfqBits::Three => Ok(IsqType::AFQ3),
            AfqBits::Four => Ok(IsqType::AFQ4),
            AfqBits::Six => Ok(IsqType::AFQ6),
            AfqBits::Eight => Ok(IsqType::AFQ8),
            AfqBits::Mxfp4 => hanzo_ml::bail!("mxfp4 is not supported as an ISQ type"),
        }
    }

    pub fn hadamard_block(&self) -> usize {
        self.hadamard_block
    }

    pub fn afq_linear_b(
        in_dim: usize,
        out_dim: usize,
        config: &QuantizedConfig,
        bias: bool,
        vb: ShardedVarBuilder,
    ) -> Result<Arc<dyn QuantMethod>> {
        if !matches!(config, QuantizedConfig::Afq { .. }) {
            hanzo_ml::bail!("Unexpected quantization config.")
        }
        // Unquantized fallback when either the per-module recipe excludes this layer or the
        // checkpoint left this module unquantized (e.g., GDN `in_proj_a` / `in_proj_b`).
        let prefix = vb.prefix();
        let Some((bits, group_size)) = config
            .afq_at(&prefix)
            .filter(|_| vb.contains_tensor("scales"))
        else {
            let weight = vb.get_with_hints((out_dim, in_dim), "weight", Default::default())?;
            let bias = if bias {
                Some(vb.get((out_dim,), "bias")?)
            } else {
                None
            };
            return Ok(Arc::new(UnquantLinear::new(
                QuantMethodConfig::Unquantized(Linear::new(weight, bias)),
            )?));
        };
        let hadamard_block = config.afq_hadamard_block_at(&prefix).unwrap_or(0);

        Ok(Self::load_typed(
            AfqLinearSpec {
                in_dim,
                out_dim,
                bits,
                group_size,
                hadamard_block,
                bias,
            },
            vb,
        )?)
    }

    pub fn afq_embed(
        vocab_size: usize,
        hidden_size: usize,
        config: &QuantizedConfig,
        vb: ShardedVarBuilder,
    ) -> Result<Arc<Self>> {
        let prefix = vb.prefix();
        let Some((bits, group_size)) = config.afq_at(&prefix) else {
            hanzo_ml::bail!("AFQ embedding {} excluded from quantization recipe", prefix);
        };
        let hadamard_block = config.afq_hadamard_block_at(&prefix).unwrap_or(0);
        Self::load_typed(
            AfqLinearSpec {
                in_dim: hidden_size,
                out_dim: vocab_size,
                bits,
                group_size,
                hadamard_block,
                bias: false,
            },
            vb,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn afq_linear_typed(
        in_dim: usize,
        out_dim: usize,
        bits: usize,
        group_size: usize,
        bias: bool,
        vb: ShardedVarBuilder,
    ) -> Result<Arc<Self>> {
        Self::load_typed(
            AfqLinearSpec {
                in_dim,
                out_dim,
                bits,
                group_size,
                hadamard_block: 0,
                bias,
            },
            vb,
        )
    }

    fn load_typed(spec: AfqLinearSpec, vb: ShardedVarBuilder) -> Result<Arc<Self>> {
        let AfqLinearSpec {
            in_dim,
            out_dim,
            bits,
            group_size,
            hadamard_block,
            bias,
        } = spec;
        let w_q = vb.get_with_hints_dtype(
            (out_dim, in_dim * bits / 32),
            "weight",
            Default::default(),
            DType::U32,
        )?;
        let signs = if vb.contains_tensor("signs") {
            Some(vb.get_with_hints_dtype((in_dim,), "signs", Default::default(), DType::F32)?)
        } else {
            None
        };
        if signs.is_some() && hadamard_block == 0 {
            hanzo_ml::bail!(
                "AFQ layer {} carries Hadamard signs on disk while hadamard_block is 0",
                vb.prefix()
            );
        }
        if hadamard_block > 0 {
            if !hadamard_block.is_power_of_two() {
                hanzo_ml::bail!(
                    "AFQ layer {} has non-power-of-two hadamard_block {hadamard_block}",
                    vb.prefix()
                );
            }
            if !in_dim.is_multiple_of(hadamard_block) {
                hanzo_ml::bail!(
                    "AFQ layer {} input dim {in_dim} is not divisible by hadamard_block {hadamard_block}",
                    vb.prefix()
                );
            }
            if signs.is_none() {
                hanzo_ml::bail!(
                    "AFQ layer {} specifies hadamard_block {hadamard_block} without a signs tensor",
                    vb.prefix()
                );
            }
        }
        let (scales, biases) = if signs.is_some() {
            (
                vb.get_with_hints_dtype(
                    (out_dim, in_dim / group_size),
                    "scales",
                    Default::default(),
                    DType::F16,
                )?,
                vb.get_with_hints_dtype(
                    (out_dim, in_dim / group_size),
                    "biases",
                    Default::default(),
                    DType::F16,
                )?,
            )
        } else {
            (
                vb.get_with_hints((out_dim, in_dim / group_size), "scales", Default::default())?,
                vb.get_with_hints((out_dim, in_dim / group_size), "biases", Default::default())?,
            )
        };

        let mm_dtype = scales.dtype();
        let bias = if bias {
            let b = vb.get((out_dim,), "bias")?;
            Some(if b.dtype() == mm_dtype {
                b
            } else {
                b.to_dtype(mm_dtype)?
            })
        } else {
            None
        };

        Ok(Arc::new(Self {
            w_q,
            scales,
            bias,
            biases,
            signs,
            hadamard_block,
            bits: AfqBits::try_from(bits)?,
            group_size: AfqGroupSize::try_from(group_size)?,
        }))
    }

    /// Token embedding lookup directly from quantized rows (without dequantizing the full vocab).
    pub fn embed(&self, ids: &Tensor, out_dtype: DType) -> Result<Tensor> {
        let orig_dims = ids.dims().to_vec();
        let flat_ids = ids.flatten_all()?;
        let flat_ids = match flat_ids.dtype() {
            DType::U32 | DType::I64 => flat_ids,
            _ => flat_ids.to_dtype(DType::U32)?,
        };
        let w_sel = self.w_q.index_select(&flat_ids, 0)?;
        let s_sel = self.scales.index_select(&flat_ids, 0)?;
        let b_sel = self.biases.index_select(&flat_ids, 0)?;
        let deq = ops::afq_dequantize_op(&w_sel, &s_sel, &b_sel, self.group_size, self.bits)?;
        let emb = if let Some(ref signs) = self.signs {
            fwht_typed(&deq, self.hadamard_block, signs, true, out_dtype)?
        } else if deq.dtype() != out_dtype {
            deq.to_dtype(out_dtype)?
        } else {
            deq
        };
        let mut out_dims = orig_dims;
        out_dims.push(emb.dim(D::Minus1)?);
        emb.reshape(out_dims)
    }

    pub fn afq_packed_linear_b(
        num_local_experts: usize,
        in_dim: usize,
        out_dim: usize,
        config: &QuantizedConfig,
        bias: bool,
        vb: ShardedVarBuilder,
    ) -> Result<Arc<dyn QuantMethod>> {
        let Some((bits, group_size)) = config.afq_at(&vb.prefix()) else {
            hanzo_ml::bail!("{} is not an AFQ-quantized expert bank", vb.prefix())
        };

        let w_q = vb.get_with_hints_dtype(
            (num_local_experts, out_dim, in_dim * bits / 32),
            "weight",
            Default::default(),
            DType::U32,
        )?;
        let scales = vb.get_with_hints(
            (num_local_experts, out_dim, in_dim / group_size),
            "scales",
            Default::default(),
        )?;
        let biases = vb.get_with_hints(
            (num_local_experts, out_dim, in_dim / group_size),
            "biases",
            Default::default(),
        )?;

        let bias = if bias {
            Some(vb.get((num_local_experts, out_dim), "bias")?)
        } else {
            None
        };

        Ok(Arc::new(Self {
            w_q,
            scales,
            bias,
            biases,
            signs: None,
            hadamard_block: 0,
            bits: AfqBits::try_from(bits)?,
            group_size: AfqGroupSize::try_from(group_size)?,
        }))
    }
}

impl QuantizedSerde for AfqLayer {
    fn name(&self) -> &'static str {
        "afq-layer"
    }
    fn isq_serde_supported(&self) -> bool {
        true
    }
    fn serialize(&self) -> Result<Cow<'_, [u8]>> {
        self.serialize_with_bias(self.bias.clone())
    }
    fn serialize_with_bias(&self, bias: Option<Tensor>) -> Result<Cow<'_, [u8]>> {
        let mut buffer = Vec::new();

        // Version is always first!
        buffer.extend(&UQFF_VERSION.to_le_bytes());

        // ISQ type for afq is 4
        buffer.push(QuantizedSerdeType::Afq as u8);

        // Has bias
        buffer.push(bias.is_some() as u8);

        // Weight, scales, biases
        serialize_tensor(&mut buffer, &self.w_q)?;
        serialize_tensor(&mut buffer, &self.scales)?;
        serialize_tensor(&mut buffer, &self.biases)?;

        // Bits and group size
        buffer.push(self.bits as u8);
        buffer.push(self.group_size as u8);

        if let Some(bias) = &bias {
            // Bias
            serialize_tensor(&mut buffer, bias)?;
        }

        Ok(Cow::from(buffer))
    }
    fn deserialize(
        data: Cow<[u8]>,
        device: &Device,
        _comm: &Arc<Comm>,
        guard: QuantizeOntoGuard,
    ) -> Result<Arc<dyn QuantMethod>>
    where
        Self: Sized,
    {
        let mut buffer = Cursor::new(data);

        let version = buffer.read_u32::<LittleEndian>()?;
        if let Err(e) = version_is_compatible(version) {
            return Err(hanzo_ml::Error::wrap(e));
        }

        let isq_type = buffer.read_u8()? as usize;
        if isq_type != QuantizedSerdeType::Afq as usize {
            hanzo_ml::bail!(
                "ISQ type ({isq_type}) doesn't match expected type {}",
                QuantizedSerdeType::Afq as usize
            );
        }

        let has_bias = buffer.read_u8()? != 0;

        let _acquired_load_guard = guard.acquire(device);
        // Weight, scales, biases
        let w_q = deserialize_tensor(&mut buffer, device)?;
        let scales = deserialize_tensor(&mut buffer, device)?;
        let biases = deserialize_tensor(&mut buffer, device)?;

        // Bits and group size
        let bits: AfqBits = buffer.read_u8()?.try_into()?;
        let group_size: AfqGroupSize = buffer.read_u8()?.try_into()?;

        let b = if has_bias {
            Some(deserialize_tensor(&mut buffer, device)?)
        } else {
            None
        };

        Ok(Arc::new(Self {
            w_q,
            scales,
            bias: b,
            biases,
            signs: None,
            hadamard_block: 0,
            bits,
            group_size,
        }))
    }
    fn deserialize_ext_bias(
        data: Cow<[u8]>,
        device: &Device,
        guard: QuantizeOntoGuard,
    ) -> Result<(Arc<dyn QuantMethod>, Option<Tensor>)>
    where
        Self: Sized,
    {
        let mut buffer = Cursor::new(data);

        let version = buffer.read_u32::<LittleEndian>()?;
        if let Err(e) = version_is_compatible(version) {
            return Err(hanzo_ml::Error::wrap(e));
        }

        let isq_type = buffer.read_u8()? as usize;
        if isq_type != QuantizedSerdeType::Afq as usize {
            hanzo_ml::bail!(
                "ISQ type ({isq_type}) doesn't match expected type {}",
                QuantizedSerdeType::Afq as usize
            );
        }

        let has_bias = buffer.read_u8()? != 0;

        let _acquired_load_guard = guard.acquire(device);
        // Weight, scales, biases
        let w_q = deserialize_tensor(&mut buffer, device)?;
        let scales = deserialize_tensor(&mut buffer, device)?;
        let biases = deserialize_tensor(&mut buffer, device)?;

        // Bits and group size
        let bits: AfqBits = buffer.read_u8()?.try_into()?;
        let group_size: AfqGroupSize = buffer.read_u8()?.try_into()?;

        let b = if has_bias {
            Some(deserialize_tensor(&mut buffer, device)?)
        } else {
            None
        };

        Ok((
            Arc::new(Self {
                w_q,
                scales,
                bias: None,
                biases,
                signs: None,
                hadamard_block: 0,
                bits,
                group_size,
            }),
            b,
        ))
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;
    use crate::{AfqModuleSpec, ShardedSafeTensors};

    fn dense_sylvester_fwht(
        x: &[f32],
        rows: usize,
        width: usize,
        block: usize,
        signs: &[f32],
        inverse: bool,
    ) -> Vec<f32> {
        let scale = 1.0f32 / (block as f32).sqrt();
        let mut out = vec![0.0f32; rows * width];
        for r in 0..rows {
            let src = &x[r * width..(r + 1) * width];
            let dst = &mut out[r * width..(r + 1) * width];
            for b0 in (0..width).step_by(block) {
                for i in 0..block {
                    let mut sum = 0.0f32;
                    for j in 0..block {
                        let v = if inverse {
                            src[b0 + j]
                        } else {
                            src[b0 + j] * signs[b0 + j]
                        };
                        let h = if ((i & j).count_ones() & 1) == 0 {
                            scale
                        } else {
                            -scale
                        };
                        sum += v * h;
                    }
                    dst[b0 + i] = if inverse { sum * signs[b0 + i] } else { sum };
                }
            }
        }
        out
    }

    #[test]
    fn fwht_typed_forward_and_inverse_match_sylvester_reference() -> Result<()> {
        for &block in &[64usize, 1024, 4096] {
            let width = if block == 64 { 128 } else { block };
            let rows = 3usize;
            let x_host: Vec<f32> = (0..(rows * width))
                .map(|i| ((i as f32) * 0.0137 - 1.7).sin())
                .collect();
            let signs_host: Vec<f32> = (0..width)
                .map(|i| if (i * 7 + 3) % 2 == 0 { 1.0 } else { -1.0 })
                .collect();
            let want_fwd = if block <= 1024 {
                Some(dense_sylvester_fwht(
                    &x_host,
                    rows,
                    width,
                    block,
                    &signs_host,
                    false,
                ))
            } else {
                None
            };

            let mut devices = vec![Device::Cpu];
            #[cfg(feature = "cuda")]
            if let Ok(cuda_dev) = Device::new_cuda(0) {
                devices.push(cuda_dev);
            }

            let mut cpu_fwd_out: Option<Vec<f32>> = None;
            let mut cpu_inv_out: Option<Vec<f32>> = None;

            for dev in &devices {
                let x = Tensor::from_vec(x_host.clone(), (rows, width), dev)?;
                let signs = Tensor::from_vec(signs_host.clone(), (width,), dev)?;
                let fwd = fwht_typed(&x, block, &signs, false, DType::F32)?;
                let roundtrip = fwht_typed(&fwd, block, &signs, true, DType::F32)?;
                let fwd_v = fwd.flatten_all()?.to_vec1::<f32>()?;
                let rt_v = roundtrip.flatten_all()?.to_vec1::<f32>()?;

                if let Some(ref want) = want_fwd {
                    let max_err = fwd_v
                        .iter()
                        .zip(want.iter())
                        .map(|(a, b)| (a - b).abs())
                        .fold(0.0f32, f32::max);
                    assert!(max_err < 1e-4, "fwht block={block} max_err={max_err}");
                }
                let rt_err = rt_v
                    .iter()
                    .zip(x_host.iter())
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0f32, f32::max);
                assert!(
                    rt_err < 1e-4,
                    "fwht roundtrip block={block} rt_err={rt_err}"
                );

                if dev.is_cpu() {
                    cpu_fwd_out = Some(fwd_v);
                    cpu_inv_out = Some(
                        fwht_typed(&x, block, &signs, true, DType::F32)?
                            .flatten_all()?
                            .to_vec1::<f32>()?,
                    );
                } else {
                    let cf = cpu_fwd_out.as_ref().unwrap();
                    let ci = cpu_inv_out.as_ref().unwrap();
                    let cuda_inv = fwht_typed(&x, block, &signs, true, DType::F32)?
                        .flatten_all()?
                        .to_vec1::<f32>()?;
                    let fwd_diff = fwd_v
                        .iter()
                        .zip(cf.iter())
                        .map(|(a, b)| (a - b).abs())
                        .fold(0.0f32, f32::max);
                    let inv_diff = cuda_inv
                        .iter()
                        .zip(ci.iter())
                        .map(|(a, b)| (a - b).abs())
                        .fold(0.0f32, f32::max);
                    assert!(fwd_diff < 1e-4 && inv_diff < 1e-4);

                    for dt in [DType::F16, DType::BF16] {
                        let x_h = x.to_dtype(dt)?;
                        let y_h = fwht_typed(&x_h, block, &signs, false, dt)?;
                        assert_eq!(y_h.dtype(), dt);
                        let z_h = fwht_typed(&y_h, block, &signs, true, dt)?;
                        let err = (&z_h.to_dtype(DType::F32)? - &x_h.to_dtype(DType::F32)?)?
                            .abs()?
                            .flatten_all()?
                            .max(0)?
                            .to_scalar::<f32>()?;
                        assert!(
                            err < 3e-2,
                            "half roundtrip block={block} dt={dt:?} err={err}"
                        );
                    }
                }
            }
        }
        Ok(())
    }

    #[test]
    fn afq_signs_fail_closed_when_hadamard_block_zero_and_succeed_with_spec() -> Result<()> {
        let mut devices = vec![Device::Cpu];
        #[cfg(feature = "cuda")]
        if let Ok(cuda_dev) = Device::new_cuda(0) {
            devices.push(cuda_dev);
        }

        for dev in devices {
            for &block in &[1024usize, 4096] {
                let in_dim = block;
                let out_dim = 8usize;
                let gs = 128usize;
                let packed_k = in_dim * 2 / 32;
                let groups = in_dim / gs;

                let w_vals: Vec<u32> = (0..(out_dim * packed_k))
                    .map(|i| {
                        0x5555_5555u32 ^ ((i as u32).wrapping_mul(0x0101_0101) & 0xaaaa_aaaau32)
                    })
                    .collect();
                let s_vals: Vec<f32> = (0..(out_dim * groups))
                    .map(|i| 0.05 + ((i % 7) as f32) * 0.01)
                    .collect();
                let b_vals: Vec<f32> = s_vals.iter().map(|s| -s).collect();
                let sign_vals: Vec<f32> = (0..in_dim)
                    .map(|i| if i % 3 == 0 { -1.0 } else { 1.0 })
                    .collect();

                let mut map = HashMap::new();
                map.insert(
                    "language_model.lm_head.weight".to_string(),
                    Tensor::from_vec(w_vals, (out_dim, packed_k), &dev)?,
                );
                map.insert(
                    "language_model.lm_head.scales".to_string(),
                    Tensor::from_vec(s_vals, (out_dim, groups), &dev)?.to_dtype(DType::F16)?,
                );
                map.insert(
                    "language_model.lm_head.biases".to_string(),
                    Tensor::from_vec(b_vals, (out_dim, groups), &dev)?.to_dtype(DType::F16)?,
                );
                map.insert(
                    "language_model.lm_head.signs".to_string(),
                    Tensor::from_vec(sign_vals, (in_dim,), &dev)?,
                );

                let vb = ShardedSafeTensors::wrap_with_dummy_regexes(
                    Box::new(map),
                    DType::F16,
                    dev.clone(),
                    None,
                );

                let missing_block_cfg: QuantizedConfig =
                    serde_json::from_str(r#"{"bits": 2, "group_size": 128, "mode": "affine"}"#)
                        .unwrap();
                assert!(
                    AfqLayer::afq_linear_b(
                        in_dim,
                        out_dim,
                        &missing_block_cfg,
                        false,
                        vb.pp("language_model.lm_head")
                    )
                    .is_err(),
                    "must fail closed when signs exist and hadamard_block is 0"
                );
                assert!(AfqLayer::afq_embed(
                    out_dim,
                    in_dim,
                    &missing_block_cfg,
                    vb.pp("language_model.lm_head")
                )
                .is_err());
                assert!(AfqLayer::afq_linear_typed(
                    in_dim,
                    out_dim,
                    2,
                    128,
                    false,
                    vb.pp("language_model.lm_head")
                )
                .is_err());

                let mut cfg = missing_block_cfg;
                cfg.with_afq_module_specs(&[AfqModuleSpec {
                    path: "lm_head".to_string(),
                    block,
                    embedding: false,
                    dtype: Some("float16".to_string()),
                }]);
                let lin = AfqLayer::afq_linear_b(
                    in_dim,
                    out_dim,
                    &cfg,
                    false,
                    vb.pp("language_model.lm_head"),
                )?;
                let emb =
                    AfqLayer::afq_embed(out_dim, in_dim, &cfg, vb.pp("language_model.lm_head"))?;
                assert_eq!(emb.hadamard_block(), block);

                let w_deq_raw = lin.dequantize_w()?;
                let w_deq = w_deq_raw.to_dtype(DType::F32)?;
                let ids = Tensor::from_vec(vec![0u32, 3, 7], (3,), &dev)?;
                let emb_f16 = emb.embed(&ids, w_deq_raw.dtype())?.to_dtype(DType::F32)?;
                let ref_rows = w_deq.index_select(&ids, 0)?;
                let emb_f16_err = (&emb_f16 - &ref_rows)?
                    .abs()?
                    .flatten_all()?
                    .max(0)?
                    .to_scalar::<f32>()?;
                assert!(
                    emb_f16_err < 1e-6,
                    "embed f16 mismatch block={block} err={emb_f16_err}"
                );
                let emb_rows = emb.embed(&ids, DType::F32)?;
                let emb_err = (&emb_rows - &ref_rows)?
                    .abs()?
                    .flatten_all()?
                    .max(0)?
                    .to_scalar::<f32>()?;
                assert!(
                    emb_err < 2.5e-3,
                    "embed f32 mismatch block={block} err={emb_err}"
                );

                for &m in &[4usize, 40] {
                    let x_vals: Vec<f32> = (0..(m * in_dim))
                        .map(|i| ((i as f32) * 0.019 - 0.4).cos() * 0.5)
                        .collect();
                    let x = Tensor::from_vec(x_vals, (m, in_dim), &dev)?.to_dtype(DType::F16)?;
                    let y = lin.forward_raw(&x)?.to_dtype(DType::F32)?;
                    let y_ref = x.to_dtype(DType::F32)?.matmul(&w_deq.t()?)?;
                    let mm_err = (&y - &y_ref)?
                        .abs()?
                        .flatten_all()?
                        .max(0)?
                        .to_scalar::<f32>()?;
                    assert!(
                        mm_err < 5e-2,
                        "afq forward mismatch m={m} block={block} dev={dev:?} err={mm_err}"
                    );
                }
            }
        }
        Ok(())
    }
}
