use std::io::Write;
use std::sync::OnceLock;

use hanzo_ml::{backend::BackendStorage, DType, Device, Result, Storage, Tensor};

use super::wire::{cpu_to_f32, get_ring_streams, read_bytes, write_bytes, SharedTcpStream};
use super::RingConfig;

pub const PP_OP_FORWARD: u8 = 1;
pub const PP_OP_TERMINATE: u8 = 0;

// Per-hop wire header: the op tag lets the protocol grow (e.g. KV-clear on new sequence),
// dims carry the activation shape, offsets carry each sequence's past-kv length so a
// stateless worker can rebuild RoPE positions and the causal mask without seeing tokens.
#[derive(Debug, Clone)]
pub struct PpHeader {
    pub op: u8,
    pub dims: Vec<usize>,
    pub offsets: Vec<usize>,
}

/// Ring pipeline transport. Sends activations to the right neighbour and receives them from the
/// left, sharing the same socket pair the collectives use. Activations cross as canonical f32.
#[derive(Debug)]
pub struct RingPipeline {
    left: SharedTcpStream,
    right: SharedTcpStream,
    rank: usize,
    world_size: usize,
}

impl RingPipeline {
    pub fn from_config(config: &RingConfig) -> Self {
        let (left, right) = get_ring_streams(config);
        Self {
            left,
            right,
            rank: config.rank,
            world_size: config.world_size,
        }
    }

    pub fn is_head(&self) -> bool {
        self.rank == 0
    }

    pub fn rank(&self) -> usize {
        self.rank
    }

    pub fn world_size(&self) -> usize {
        self.world_size
    }

    pub fn send_right(&self, h: &Tensor, header: &PpHeader) -> Result<()> {
        let data = tensor_to_f32(h)?;
        let hdr = encode_header(header);
        let mut guard = self
            .right
            .lock()
            .map_err(|e| hanzo_ml::Error::msg(format!("lock right: {e:?}")))?;
        write_bytes(&mut guard, &hdr)?;
        let payload = unsafe {
            std::slice::from_raw_parts(data.as_ptr() as *const u8, std::mem::size_of_val(&data[..]))
        };
        write_bytes(&mut guard, payload)?;
        guard
            .flush()
            .map_err(|e| hanzo_ml::Error::msg(format!("flush right: {e:?}")))?;
        Ok(())
    }

    /// Blocking receive from the left neighbour. Returns `(None, header)` when the header op is
    /// `PP_OP_TERMINATE` (no payload follows), else `(Some(activation), header)`.
    pub fn recv_left(&self, device: &Device, dtype: DType) -> Result<(Option<Tensor>, PpHeader)> {
        let mut guard = self
            .left
            .lock()
            .map_err(|e| hanzo_ml::Error::msg(format!("lock left: {e:?}")))?;
        let header = decode_header(&mut guard)?;
        if header.op == PP_OP_TERMINATE {
            return Ok((None, header));
        }
        let n: usize = header.dims.iter().product();
        let mut buf = vec![0f32; n];
        let raw = unsafe {
            std::slice::from_raw_parts_mut(
                buf.as_mut_ptr() as *mut u8,
                n * std::mem::size_of::<f32>(),
            )
        };
        read_bytes(&mut guard, raw)?;
        drop(guard);
        let t = Tensor::from_vec(buf, header.dims.clone(), device)?.to_dtype(dtype)?;
        Ok((Some(t), header))
    }

    pub fn send_terminate(&self) -> Result<()> {
        let hdr = encode_header(&PpHeader {
            op: PP_OP_TERMINATE,
            dims: vec![],
            offsets: vec![],
        });
        let mut guard = self
            .right
            .lock()
            .map_err(|e| hanzo_ml::Error::msg(format!("lock right: {e:?}")))?;
        write_bytes(&mut guard, &hdr)?;
        guard
            .flush()
            .map_err(|e| hanzo_ml::Error::msg(format!("flush right: {e:?}")))?;
        Ok(())
    }
}

fn tensor_to_f32(xs: &Tensor) -> Result<Vec<f32>> {
    let xs = xs.contiguous()?;
    let storage = xs.storage_and_layout().0;
    match &*storage {
        Storage::Cpu(s) => cpu_to_f32(s),
        Storage::Cuda(s) => cpu_to_f32(&s.to_cpu_storage()?),
        Storage::Metal(s) => cpu_to_f32(&s.to_cpu_storage()?),
        #[cfg(feature = "rocm")]
        Storage::Rocm(s) => cpu_to_f32(&s.to_cpu_storage()?),
    }
}

fn encode_header(header: &PpHeader) -> Vec<u8> {
    let mut out = Vec::with_capacity(1 + 8 + 4 * (header.dims.len() + header.offsets.len()));
    out.push(header.op);
    if header.op == PP_OP_TERMINATE {
        return out;
    }
    put_u32(&mut out, header.dims.len() as u32);
    for &d in &header.dims {
        put_u32(&mut out, d as u32);
    }
    put_u32(&mut out, header.offsets.len() as u32);
    for &o in &header.offsets {
        put_u32(&mut out, o as u32);
    }
    out
}

fn decode_header(stream: &mut std::net::TcpStream) -> Result<PpHeader> {
    let mut op = [0u8; 1];
    read_bytes(stream, &mut op)?;
    if op[0] == PP_OP_TERMINATE {
        return Ok(PpHeader {
            op: PP_OP_TERMINATE,
            dims: vec![],
            offsets: vec![],
        });
    }
    let ndims = read_u32(stream)? as usize;
    let mut dims = Vec::with_capacity(ndims);
    for _ in 0..ndims {
        dims.push(read_u32(stream)? as usize);
    }
    let noff = read_u32(stream)? as usize;
    let mut offsets = Vec::with_capacity(noff);
    for _ in 0..noff {
        offsets.push(read_u32(stream)? as usize);
    }
    Ok(PpHeader {
        op: op[0],
        dims,
        offsets,
    })
}

fn put_u32(out: &mut Vec<u8>, v: u32) {
    out.extend_from_slice(&v.to_le_bytes());
}

fn read_u32(stream: &mut std::net::TcpStream) -> Result<u32> {
    let mut b = [0u8; 4];
    read_bytes(stream, &mut b)?;
    Ok(u32::from_le_bytes(b))
}

static TP_ENDPOINTS: OnceLock<(SharedTcpStream, SharedTcpStream, usize)> = OnceLock::new();

fn tp_endpoints() -> (SharedTcpStream, SharedTcpStream, usize) {
    TP_ENDPOINTS
        .get_or_init(|| {
            let config = RingConfig::load();
            let (left, right) = get_ring_streams(&config);
            (left, right, config.rank)
        })
        .clone()
}

/// Ring tensor-parallel token sync: rank 0 broadcasts its sampled tokens around the ring so no
/// rank diverges on a near-tie argmax (which would silently desync the replicated KV caches).
/// Head writes right then drains the wrap-around from left; workers read from left (overwriting
/// `values` in place) and forward right. One hop per rank keeps the shared streams balanced.
pub fn broadcast_head_tokens(values: &mut [u32]) -> Result<()> {
    let (left, right, rank) = tp_endpoints();
    let nbytes = std::mem::size_of_val(values);
    let lock_err = |e| hanzo_ml::Error::msg(format!("lock: {e:?}"));
    if rank == 0 {
        {
            let mut g = right.lock().map_err(lock_err)?;
            let raw = unsafe { std::slice::from_raw_parts(values.as_ptr() as *const u8, nbytes) };
            write_bytes(&mut g, raw)?;
            g.flush()
                .map_err(|e| hanzo_ml::Error::msg(format!("flush: {e:?}")))?;
        }
        let mut lg = left.lock().map_err(lock_err)?;
        let mut discard = vec![0u8; nbytes];
        read_bytes(&mut lg, &mut discard)?;
    } else {
        {
            let mut lg = left.lock().map_err(lock_err)?;
            let raw =
                unsafe { std::slice::from_raw_parts_mut(values.as_mut_ptr() as *mut u8, nbytes) };
            read_bytes(&mut lg, raw)?;
        }
        let mut g = right.lock().map_err(lock_err)?;
        let raw = unsafe { std::slice::from_raw_parts(values.as_ptr() as *const u8, nbytes) };
        write_bytes(&mut g, raw)?;
        g.flush()
            .map_err(|e| hanzo_ml::Error::msg(format!("flush: {e:?}")))?;
    }
    Ok(())
}
