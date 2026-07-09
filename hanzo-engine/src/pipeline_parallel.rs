use std::ops::Range;
use std::sync::Arc;

use hanzo_ml::{DType, Device, Result, Tensor};
use hanzo_quant::{PpHeader, RingConfig, RingPipeline, PP_OP_FORWARD};

// PP is the ring mode for the quantized/gguf path (which has no tensor-parallel support).
// Opt out with RING_PP=0 to keep the legacy safetensors ring TP behaviour.
pub fn use_pipeline_parallel() -> bool {
    hanzo_quant::distributed::use_ring() && std::env::var("RING_PP").map_or(true, |v| v != "0")
}

/// Contiguous transformer-layer split across the ring. Each rank owns `ranges[rank]` of the
/// `n_layers` blocks; rank 0 additionally owns the embedding, final norm and lm_head.
#[derive(Debug, Clone)]
pub struct RingLayout {
    pub rank: usize,
    pub world_size: usize,
    pub ranges: Vec<Range<usize>>,
}

impl RingLayout {
    pub fn new(n_layers: usize) -> Result<Self> {
        let config = RingConfig::load();
        let ranges = split_layers(n_layers, config.world_size)?;
        Ok(Self {
            rank: config.rank,
            world_size: config.world_size,
            ranges,
        })
    }

    pub fn local(&self) -> Range<usize> {
        self.ranges[self.rank].clone()
    }

    pub fn is_head(&self) -> bool {
        self.rank == 0
    }
}

fn ranges_from_counts(counts: &[usize]) -> Vec<Range<usize>> {
    let mut ranges = Vec::with_capacity(counts.len());
    let mut start = 0;
    for &c in counts {
        ranges.push(start..start + c);
        start += c;
    }
    ranges
}

// RING_LAYER_SPLIT="a,b,.." forces exact per-rank block counts. Otherwise distribute
// proportionally to RING_MEM_GIB="g0,g1,.." (per-rank GPU budget), falling back to an even split.
fn split_layers(n_layers: usize, world_size: usize) -> Result<Vec<Range<usize>>> {
    if let Ok(spec) = std::env::var("RING_LAYER_SPLIT") {
        let counts = spec
            .split(',')
            .map(|x| x.trim().parse::<usize>())
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(|e| hanzo_ml::Error::msg(format!("RING_LAYER_SPLIT parse: {e}")))?;
        if counts.len() != world_size {
            hanzo_ml::bail!(
                "RING_LAYER_SPLIT has {} entries but world_size is {world_size}",
                counts.len()
            );
        }
        if counts.iter().sum::<usize>() != n_layers {
            hanzo_ml::bail!(
                "RING_LAYER_SPLIT sums to {} but model has {n_layers} layers",
                counts.iter().sum::<usize>()
            );
        }
        return Ok(ranges_from_counts(&counts));
    }

    let weights: Vec<f64> = match std::env::var("RING_MEM_GIB") {
        Ok(spec) => {
            let w = spec
                .split(',')
                .map(|x| x.trim().parse::<f64>())
                .collect::<std::result::Result<Vec<_>, _>>()
                .map_err(|e| hanzo_ml::Error::msg(format!("RING_MEM_GIB parse: {e}")))?;
            if w.len() != world_size {
                hanzo_ml::bail!(
                    "RING_MEM_GIB has {} entries but world_size is {world_size}",
                    w.len()
                );
            }
            w
        }
        Err(_) => vec![1.0; world_size],
    };

    let total: f64 = weights.iter().sum();
    let mut counts: Vec<usize> = weights
        .iter()
        .map(|w| ((w / total) * n_layers as f64).floor() as usize)
        .collect();
    let mut assigned: usize = counts.iter().sum();
    let mut r = 0;
    while assigned < n_layers {
        counts[r % world_size] += 1;
        assigned += 1;
        r += 1;
    }
    Ok(ranges_from_counts(&counts))
}

/// Implemented by each model that participates in pipeline parallelism. The transport and the
/// forward/worker drivers live here once; each model contributes only its embed, local-layer
/// loop, and final norm+head, reusing its existing single-node code.
pub trait PipelineParallelModel {
    fn ring(&self) -> &Arc<RingPipeline>;
    fn pp_device(&self) -> &Device;
    fn pp_dtype(&self) -> DType;
    fn pp_embed(&self, tokens: &Tensor) -> Result<Tensor>;
    fn pp_run_local(&self, h: &Tensor, offsets: &[usize]) -> Result<Tensor>;
    fn pp_norm_head(&self, h: &Tensor, context_lens: Vec<(usize, usize)>) -> Result<Tensor>;
}

/// Head (rank 0) forward: embed the tokens, run the head's local layers, send the activation
/// around the ring, receive the post-final-layer activation back from the left neighbour, then
/// run the final norm + lm_head to produce logits.
pub fn pp_head_forward<M: PipelineParallelModel>(
    m: &M,
    tokens: &Tensor,
    offsets: &[usize],
    context_lens: Vec<(usize, usize)>,
) -> Result<Tensor> {
    let h = m.pp_embed(tokens)?;
    let h = m.pp_run_local(&h, offsets)?;
    let ring = m.ring();
    let header = PpHeader {
        op: PP_OP_FORWARD,
        dims: h.dims().to_vec(),
        offsets: offsets.to_vec(),
    };
    ring.send_right(&h, &header)?;
    let (final_h, _) = ring.recv_left(m.pp_device(), m.pp_dtype())?;
    let final_h =
        final_h.ok_or_else(|| hanzo_ml::Error::msg("PP head received terminate mid-generation"))?;
    m.pp_norm_head(&final_h, context_lens)
}

/// One worker (rank != 0) step: receive an activation from the left, run this rank's local
/// layers, and forward the result to the right. Returns `false` on a terminate frame.
pub fn pp_worker_step<M: PipelineParallelModel>(m: &M) -> Result<bool> {
    let ring = m.ring();
    let (h, header) = ring.recv_left(m.pp_device(), m.pp_dtype())?;
    let Some(h) = h else { return Ok(false) };
    let h = m.pp_run_local(&h, &header.offsets)?;
    let out = PpHeader {
        op: PP_OP_FORWARD,
        dims: h.dims().to_vec(),
        offsets: header.offsets,
    };
    ring.send_right(&h, &out)?;
    Ok(true)
}
