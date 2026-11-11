use std::{
    borrow::Cow,
    collections::HashMap,
    io::Cursor,
    sync::{atomic::AtomicUsize, Arc, Mutex, OnceLock},
};

use byteorder::{LittleEndian, ReadBytesExt};
use hanzo_ml::{DType, Device, DeviceLocation, Result, Tensor, D};
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

static HADAMARD_CACHE: OnceLock<Mutex<HashMap<(DeviceLocation, usize), Tensor>>> = OnceLock::new();

fn hadamard_matrix(block: usize, device: &Device) -> Result<Tensor> {
    if block == 0 || !block.is_power_of_two() {
        hanzo_ml::bail!("Hadamard block size must be a power of two, got {block}");
    }
    let cache = HADAMARD_CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    let key = (device.location(), block);
    if let Some(h) = cache.lock().unwrap().get(&key) {
        return Ok(h.clone());
    }
    let scale = 1.0f32 / (block as f32).sqrt();
    let mut data = vec![0.0f32; block * block];
    for i in 0..block {
        for j in 0..block {
            let parity = (i & j).count_ones() & 1;
            data[i * block + j] = if parity == 0 { scale } else { -scale };
        }
    }
    let h = Tensor::from_vec(data, (block, block), device)?;
    cache.lock().unwrap().insert(key, h.clone());
    Ok(h)
}

fn fwht_typed(
    x: &Tensor,
    block: usize,
    signs: &Tensor,
    inverse: bool,
    out_dtype: DType,
) -> Result<Tensor> {
    let orig_shape = x.shape().clone();
    let last_dim = x.dim(D::Minus1)?;
    if last_dim % block != 0 {
        hanzo_ml::bail!("fwht expects last dim ({last_dim}) divisible by block ({block})");
    }
    let mut y = x.to_dtype(DType::F32)?;
    if !inverse {
        y = y.broadcast_mul(signs)?;
    }
    let h = hadamard_matrix(block, x.device())?;
    let mut out = y.reshape(((), block))?.matmul(&h)?.reshape(orig_shape)?;
    if inverse {
        out = out.broadcast_mul(signs)?;
    }
    if out.dtype() != out_dtype {
        out.to_dtype(out_dtype)
    } else {
        Ok(out)
    }
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
            out = out.broadcast_add(&b.to_dtype(out.dtype())?)?;
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
        let Some((bits, group_size)) = config
            .afq_at(&vb.prefix())
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

        Ok(Self::afq_linear_typed(
            in_dim, out_dim, bits, group_size, bias, vb,
        )?)
    }

    pub fn afq_linear_typed(
        in_dim: usize,
        out_dim: usize,
        bits: usize,
        group_size: usize,
        bias: bool,
        vb: ShardedVarBuilder,
    ) -> Result<Arc<Self>> {
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
        let hadamard_block = if signs.is_some() { 512 } else { 0 };
        // Preserve 10-bit F16 mantissa precision on Prism/Bonsai-2 Hadamard checkpoints.
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

        let bias = if bias {
            Some(vb.get((out_dim,), "bias")?)
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
