// Vulkan copy_blocks / swap_blocks: STUBS. Block-level cache movement used for CPU<->GPU swapping
// and copy-on-write block sharing (CUDA copy_blocks_kernel.cu / Metal cache.rs). The current
// hanzo-engine cache_engine (vLLM-v1 block pool) does not call these, so they're provided only to
// match the cuda/metal export surface; both bail if reached.
//
// TODO (continuous batching / swapping): copy_blocks = gather/scatter of (src_block -> dst_blocks)
// per layer via a small kernel or queued buffer-to-buffer copies; swap_blocks = device<->host block
// transfers (a staging buffer + vkCmdCopyBuffer per block range).

use std::collections::HashMap;

use hanzo_ml::{Result, Tensor};

pub fn copy_blocks(
    _key_caches: Vec<&mut Tensor>,
    _value_caches: Vec<&mut Tensor>,
    _block_mapping: &HashMap<usize, Vec<usize>>,
) -> Result<()> {
    hanzo_ml::bail!("vulkan copy_blocks is not yet implemented")
}

/// # Safety
/// Matches the cuda/metal signature; this stub touches nothing and immediately bails.
pub unsafe fn swap_blocks(
    _src: Tensor,
    _dst: &Tensor,
    _block_mapping: HashMap<usize, usize>,
) -> Result<()> {
    hanzo_ml::bail!("vulkan swap_blocks is not yet implemented")
}
