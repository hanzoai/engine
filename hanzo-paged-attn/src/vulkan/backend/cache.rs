// Vulkan copy_blocks / swap_blocks: block-level cache movement used for CPU<->GPU swapping and
// copy-on-write block sharing (mirrors CUDA copy_blocks_kernel.cu / Metal cache.rs). The current
// hanzo-engine cache_engine (vLLM-v1 block pool) does not call these, but they are implemented (not
// stubbed) to match the cuda/metal export surface and to be correct if a swapping scheduler is
// wired up. Both are expressed with high-level Tensor ops (narrow + slice_set), which the Vulkan
// backend services on-GPU via its copy2d / copy_strided_src kernels -- no new compute kernel needed.

use std::collections::HashMap;
use std::iter::zip;

use hanzo_ml::{Device, IndexOp, Result, Tensor};

// Copy a single [.., block, ..] slot along dim 0 from `src` block index to `dst` block index, where
// both alias the same cache tensor. We materialize a contiguous copy of the source block first so it
// no longer shares storage with the destination (slice_set rejects aliasing src/dst).
fn copy_one_block(cache: &Tensor, src_block: usize, dst_block: usize) -> Result<()> {
    if src_block == dst_block {
        return Ok(());
    }
    let block = cache.narrow(0, src_block, 1)?.contiguous()?;
    cache.slice_set(&block, 0, dst_block)
}

pub fn copy_blocks(
    key_caches: Vec<&mut Tensor>,
    value_caches: Vec<&mut Tensor>,
    block_mapping: &HashMap<usize, Vec<usize>>,
) -> Result<()> {
    if key_caches.is_empty() {
        return Ok(());
    }
    let cache_dev = key_caches.first().unwrap().device();
    if !matches!(cache_dev, Device::Vulkan(_)) {
        hanzo_ml::bail!("vulkan copy_blocks: caches must be on a vulkan device");
    }
    if !cache_dev.same_device(value_caches.first().unwrap().device()) {
        hanzo_ml::bail!(
            "`key` and `value` caches have different devices, got {:?} and {:?} respectively.",
            cache_dev,
            value_caches.first().unwrap().device()
        );
    }
    if key_caches.first().unwrap().dtype() != value_caches.first().unwrap().dtype() {
        hanzo_ml::bail!(
            "Key and value caches have different types, got {:?} and {:?}.",
            key_caches.first().unwrap().dtype(),
            value_caches.first().unwrap().dtype()
        );
    }

    for (key_cache, value_cache) in zip(&key_caches, &value_caches) {
        for (src_block, dst_blocks) in block_mapping {
            for dst_block in dst_blocks {
                copy_one_block(key_cache, *src_block, *dst_block)?;
                copy_one_block(value_cache, *src_block, *dst_block)?;
            }
        }
    }

    Ok(())
}

// `dst` REALLY should be &mut. That's the only reason this is unsafe.
/// # Safety
/// `dst` is the only shared reference and upholds the `&mut` aliasing guarantee.
pub unsafe fn swap_blocks(
    src: Tensor,
    dst: &Tensor,
    block_mapping: HashMap<usize, usize>,
) -> Result<()> {
    if src.dtype() != dst.dtype() {
        hanzo_ml::bail!(
            "swap_blocks: src/dst dtype mismatch, got {:?} and {:?}",
            src.dtype(),
            dst.dtype()
        );
    }
    // The engine indexes blocks along dim 0 (block size == product of the trailing dims), matching
    // the cuda/metal byte-offset math (block stride == src.dims()[0] elems for the flattened cache).
    let src = src.contiguous()?;
    let num_blocks = src.dims()[0];
    for (src_block, dst_block) in block_mapping {
        if src_block >= num_blocks {
            hanzo_ml::bail!("swap_blocks: src block {src_block} out of range ({num_blocks})");
        }
        let block = src.i(src_block)?.unsqueeze(0)?.contiguous()?;
        let block = match dst.device().location() {
            loc if loc == src.device().location() => block,
            _ => block.to_device(dst.device())?,
        };
        dst.slice_set(&block, 0, dst_block)?;
    }
    Ok(())
}
