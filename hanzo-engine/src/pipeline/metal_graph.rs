//! Metal decode-graph: the Metal-backend analog of [`crate::pipeline::cuda_graph`].
//!
//! # Why this is structured like the CUDA decode-graph but does *not* replay a graph
//!
//! On CUDA the decode fast-path captures the per-token forward into a `cudaGraph`
//! (via stream capture) and replays it with `cuGraphLaunch`, collapsing all the
//! per-kernel launch overhead into a single graph launch. Metal has **no 1:1
//! equivalent** of capture/replay for an arbitrary, already-encoded sequence of
//! compute dispatches:
//!
//! * An `MTLIndirectCommandBuffer` (ICB) *can* be replayed with
//!   `executeCommandsInBuffer`, but only for commands explicitly encoded into the
//!   ICB through `MTLIndirectComputeCommand`. In this codebase every kernel — the
//!   candle matmul/softmax/RoPE/norm ops and the `hanzo-metal-kernels` paged-attn
//!   kernels — dispatches into the shared auto-committing compute encoder
//!   (`hanzo_metal_kernels::metal::Commands`), *not* into an ICB. Re-targeting all
//!   of them to an ICB would be an engine-wide rewrite of candle + the kernel crate
//!   + paged-attn, and ICBs additionally cannot express the per-dispatch
//!   `setThreadgroupMemoryLength` that paged-attn v1/v2 require. ICB is therefore
//!   the wrong tool for a decode-graph *port*.
//! * A retained `MTLCommandBuffer` cannot be replayed either: once committed it is
//!   consumed, and Metal exposes no "re-commit" of a finished command buffer.
//!
//! What *does* port cleanly — and is the load-bearing half of the CUDA design — is
//! the **invariance machinery**: the per-token forward must be made position- and
//! length-invariant so that the work encoded for token N is byte-for-byte the same
//! dispatch sequence as token N-1, differing only in the *contents* of a fixed set
//! of device buffers (slot mappings, context lengths, block tables, RoPE positions).
//! The CUDA graph only works because that invariance already holds (bucketed
//! `max_context_len`, stable `Var` buffers updated in place via [`Var::set`], paged
//! attention reading context length from a device buffer). On Metal we reuse exactly
//! that machinery so the decode step re-encodes an identical command stream into the
//! existing `Commands` pipeline; with stable buffers the per-token CPU work is the
//! cheap, identical re-encode rather than rebuilding metadata tensors, and the
//! command-buffer batching (`CANDLE_METAL_COMPUTE_PER_BUFFER`) is what coalesces the
//! dispatches per step.
//!
//! Concretely this module provides the Metal counterparts of the CUDA types:
//!
//! * [`MetalDecodeGraphKey`]   ~ `CudaDecodeGraphKey`
//! * [`MetalDecodeGraphMetadataBuffers`] ~ `CudaDecodeGraphMetadataBuffers`
//! * [`metal_decode_graphs_enabled`] (gated on `HANZO_METAL_GRAPHS`)
//!
//! The pipeline gate in `normal.rs` mirrors the CUDA gate one-for-one behind
//! `#[cfg(feature = "metal")]`: `metal_decode_graphs_enabled()` +
//! `supports_metal_decode_graphs()` + `q_len == 1` + paged metadata present +
//! `is_metal()`. The CUDA and ROCm paths are untouched.
//!
//! NOTE (NEEDS MAC VALIDATION): this file is a code port authored on a non-Metal
//! host. It cannot be compiled or run against Metal here. See the module-level
//! review notes at the bottom of the file.

use std::collections::HashMap;

use hanzo_ml::{DType, Device, DeviceLocation, Tensor, Var};

use crate::pipeline::text_models_inputs_processor::PagedAttentionInputMetadata;

/// Cache capacity for compiled/warmed decode-graph entries, mirroring
/// `CUDA_DECODE_GRAPH_CACHE_CAPACITY` so the bucketed-key LRU behaves identically.
pub(crate) const METAL_DECODE_GRAPH_CACHE_CAPACITY: usize = 32;

/// Identity key for a warmed Metal decode step.
///
/// Two decode steps that produce the same key encode an identical command stream
/// (same shapes, dtypes, device placement, and bucketed `max_context_len`), so a
/// cached entry's stable metadata buffers can be reused by updating their contents
/// in place. This mirrors [`crate::pipeline::cuda_graph::CudaDecodeGraphKey`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct MetalDecodeGraphKey {
    device: DeviceLocation,
    input_shape: Vec<usize>,
    input_dtype: DType,
    max_context_len: Option<usize>,
    full_max_context_len: Option<usize>,
    tensors: Vec<MetalGraphTensorKey>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct MetalGraphTensorKey {
    name: &'static str,
    location: DeviceLocation,
    shape: Vec<usize>,
    dtype: DType,
}

type MetalGraphVarMap = HashMap<DeviceLocation, Var>;

/// Stable per-step device buffers, updated in place between decode steps.
///
/// Each field is a map from device location to a [`Var`] whose backing Metal buffer
/// is allocated once and then overwritten via [`Var::set`] (an in-place
/// `copy_strided_src` into the same buffer — the buffer identity is preserved). The
/// paged-attention kernels read context lengths / block tables / slot mappings from
/// these device buffers, so the encoded launch geometry stays fixed (sized for the
/// bucketed `max_context_len`) while the actual attention math follows the live
/// device contents.
///
/// This is a field-for-field mirror of
/// [`crate::pipeline::cuda_graph::CudaDecodeGraphMetadataBuffers`]; keeping the shape
/// identical means the inputs-processor padding (`cuda_graph_block_table_len`, the
/// bucketing in `text_models_inputs_processor`) applies unchanged on Metal.
pub(crate) struct MetalDecodeGraphMetadataBuffers {
    slot_mappings: MetalGraphVarMap,
    block_tables: Option<MetalGraphVarMap>,
    context_lens: Option<MetalGraphVarMap>,
    full_block_tables: Option<MetalGraphVarMap>,
    full_context_lens: Option<MetalGraphVarMap>,
    paged_kv_indptr: Option<MetalGraphVarMap>,
    paged_kv_indices: Option<MetalGraphVarMap>,
    paged_kv_last_page_len: Option<MetalGraphVarMap>,
    full_paged_kv_indptr: Option<MetalGraphVarMap>,
    full_paged_kv_indices: Option<MetalGraphVarMap>,
    full_paged_kv_last_page_len: Option<MetalGraphVarMap>,
    paged_kv_request_indices: Option<MetalGraphVarMap>,
    paged_kv_tile_indices: Option<MetalGraphVarMap>,
    paged_kv_o_indptr: Option<MetalGraphVarMap>,
    paged_kv_chunk_size: Option<MetalGraphVarMap>,
    paged_kv_block_valid_mask: Option<MetalGraphVarMap>,
    full_paged_kv_request_indices: Option<MetalGraphVarMap>,
    full_paged_kv_tile_indices: Option<MetalGraphVarMap>,
    full_paged_kv_o_indptr: Option<MetalGraphVarMap>,
    full_paged_kv_chunk_size: Option<MetalGraphVarMap>,
    full_paged_kv_block_valid_mask: Option<MetalGraphVarMap>,
    rope_positions: MetalGraphVarMap,
    block_table_signature: Option<Vec<u64>>,
    full_block_table_signature: Option<Vec<u64>>,
}

impl MetalDecodeGraphKey {
    pub(crate) fn new(
        input_ids: &Tensor,
        metadata: &PagedAttentionInputMetadata,
        block_size: usize,
    ) -> hanzo_ml::Result<Self> {
        let mut tensors = Vec::new();
        push_graph_tensor_keys("slot_mappings", Some(&metadata.slot_mappings), &mut tensors);
        push_graph_tensor_keys("block_tables", metadata.block_tables.as_ref(), &mut tensors);
        push_graph_tensor_keys("context_lens", metadata.context_lens.as_ref(), &mut tensors);
        push_graph_tensor_keys(
            "full_block_tables",
            metadata.full_block_tables.as_ref(),
            &mut tensors,
        );
        push_graph_tensor_keys(
            "full_context_lens",
            metadata.full_context_lens.as_ref(),
            &mut tensors,
        );
        push_graph_tensor_keys(
            "paged_kv_indptr",
            metadata.paged_kv_indptr.as_ref(),
            &mut tensors,
        );
        push_graph_tensor_keys(
            "paged_kv_indices",
            metadata.paged_kv_indices.as_ref(),
            &mut tensors,
        );
        push_graph_tensor_keys(
            "paged_kv_last_page_len",
            metadata.paged_kv_last_page_len.as_ref(),
            &mut tensors,
        );
        push_graph_tensor_keys(
            "full_paged_kv_indptr",
            metadata.full_paged_kv_indptr.as_ref(),
            &mut tensors,
        );
        push_graph_tensor_keys(
            "full_paged_kv_indices",
            metadata.full_paged_kv_indices.as_ref(),
            &mut tensors,
        );
        push_graph_tensor_keys(
            "full_paged_kv_last_page_len",
            metadata.full_paged_kv_last_page_len.as_ref(),
            &mut tensors,
        );
        push_graph_tensor_keys(
            "paged_kv_request_indices",
            metadata.paged_kv_request_indices.as_ref(),
            &mut tensors,
        );
        push_graph_tensor_keys(
            "paged_kv_tile_indices",
            metadata.paged_kv_tile_indices.as_ref(),
            &mut tensors,
        );
        push_graph_tensor_keys(
            "paged_kv_o_indptr",
            metadata.paged_kv_o_indptr.as_ref(),
            &mut tensors,
        );
        push_graph_tensor_keys(
            "paged_kv_chunk_size",
            metadata.paged_kv_chunk_size.as_ref(),
            &mut tensors,
        );
        push_graph_tensor_keys(
            "paged_kv_block_valid_mask",
            metadata.paged_kv_block_valid_mask.as_ref(),
            &mut tensors,
        );
        push_graph_tensor_keys(
            "full_paged_kv_request_indices",
            metadata.full_paged_kv_request_indices.as_ref(),
            &mut tensors,
        );
        push_graph_tensor_keys(
            "full_paged_kv_tile_indices",
            metadata.full_paged_kv_tile_indices.as_ref(),
            &mut tensors,
        );
        push_graph_tensor_keys(
            "full_paged_kv_o_indptr",
            metadata.full_paged_kv_o_indptr.as_ref(),
            &mut tensors,
        );
        push_graph_tensor_keys(
            "full_paged_kv_chunk_size",
            metadata.full_paged_kv_chunk_size.as_ref(),
            &mut tensors,
        );
        push_graph_tensor_keys(
            "full_paged_kv_block_valid_mask",
            metadata.full_paged_kv_block_valid_mask.as_ref(),
            &mut tensors,
        );
        tensors.sort_by(|a, b| {
            a.name.cmp(b.name).then_with(|| {
                device_location_sort_key(&a.location).cmp(&device_location_sort_key(&b.location))
            })
        });

        Ok(Self {
            device: input_ids.device().location(),
            input_shape: input_ids.dims().to_vec(),
            input_dtype: input_ids.dtype(),
            max_context_len: bucket_context_len(metadata.block_tables.as_ref(), block_size),
            full_max_context_len: bucket_context_len(
                metadata.full_block_tables.as_ref(),
                block_size,
            ),
            tensors,
        })
    }
}

impl MetalDecodeGraphMetadataBuffers {
    pub(crate) fn new(
        metadata: &PagedAttentionInputMetadata,
        seqlen_offsets: &[usize],
        block_size: usize,
    ) -> hanzo_ml::Result<(Self, PagedAttentionInputMetadata)> {
        let slot_mappings = var_map_from_tensor_map(&metadata.slot_mappings)?;
        let rope_positions = rope_positions_var_map(&metadata.slot_mappings, seqlen_offsets)?;
        let buffers = Self {
            slot_mappings,
            block_tables: option_var_map_from_tensor_map(metadata.block_tables.as_ref())?,
            context_lens: option_var_map_from_tensor_map(metadata.context_lens.as_ref())?,
            full_block_tables: option_var_map_from_tensor_map(metadata.full_block_tables.as_ref())?,
            full_context_lens: option_var_map_from_tensor_map(metadata.full_context_lens.as_ref())?,
            paged_kv_indptr: option_var_map_from_tensor_map(metadata.paged_kv_indptr.as_ref())?,
            paged_kv_indices: option_var_map_from_tensor_map(metadata.paged_kv_indices.as_ref())?,
            paged_kv_last_page_len: option_var_map_from_tensor_map(
                metadata.paged_kv_last_page_len.as_ref(),
            )?,
            full_paged_kv_indptr: option_var_map_from_tensor_map(
                metadata.full_paged_kv_indptr.as_ref(),
            )?,
            full_paged_kv_indices: option_var_map_from_tensor_map(
                metadata.full_paged_kv_indices.as_ref(),
            )?,
            full_paged_kv_last_page_len: option_var_map_from_tensor_map(
                metadata.full_paged_kv_last_page_len.as_ref(),
            )?,
            paged_kv_request_indices: option_var_map_from_tensor_map(
                metadata.paged_kv_request_indices.as_ref(),
            )?,
            paged_kv_tile_indices: option_var_map_from_tensor_map(
                metadata.paged_kv_tile_indices.as_ref(),
            )?,
            paged_kv_o_indptr: option_var_map_from_tensor_map(metadata.paged_kv_o_indptr.as_ref())?,
            paged_kv_chunk_size: option_var_map_from_tensor_map(
                metadata.paged_kv_chunk_size.as_ref(),
            )?,
            paged_kv_block_valid_mask: option_var_map_from_tensor_map(
                metadata.paged_kv_block_valid_mask.as_ref(),
            )?,
            full_paged_kv_request_indices: option_var_map_from_tensor_map(
                metadata.full_paged_kv_request_indices.as_ref(),
            )?,
            full_paged_kv_tile_indices: option_var_map_from_tensor_map(
                metadata.full_paged_kv_tile_indices.as_ref(),
            )?,
            full_paged_kv_o_indptr: option_var_map_from_tensor_map(
                metadata.full_paged_kv_o_indptr.as_ref(),
            )?,
            full_paged_kv_chunk_size: option_var_map_from_tensor_map(
                metadata.full_paged_kv_chunk_size.as_ref(),
            )?,
            full_paged_kv_block_valid_mask: option_var_map_from_tensor_map(
                metadata.full_paged_kv_block_valid_mask.as_ref(),
            )?,
            rope_positions,
            block_table_signature: metadata.block_table_signature.clone(),
            full_block_table_signature: metadata.full_block_table_signature.clone(),
        };
        let metadata = buffers.metadata_from(metadata, block_size);
        Ok((buffers, metadata))
    }

    /// Update the stable device buffers in place for a new decode step.
    ///
    /// Mirrors `CudaDecodeGraphMetadataBuffers::copy_from`: only the buffers whose
    /// contents change every step (slot mappings, context lengths, last-page lengths,
    /// RoPE positions) are written unconditionally; the block-table-derived buffers
    /// are only re-uploaded when the block-table signature changes (which on Metal
    /// also matters because `Var::set` issues a blit/copy that would otherwise serialize
    /// the encoder).
    pub(crate) fn copy_from(
        &mut self,
        metadata: &PagedAttentionInputMetadata,
        seqlen_offsets: &[usize],
    ) -> hanzo_ml::Result<()> {
        let block_tables_changed =
            signature_changed(&self.block_table_signature, &metadata.block_table_signature);
        let full_block_tables_changed = signature_changed(
            &self.full_block_table_signature,
            &metadata.full_block_table_signature,
        );

        copy_var_map(&self.slot_mappings, &metadata.slot_mappings, "slot_mappings")?;
        copy_option_var_map(
            &self.context_lens,
            metadata.context_lens.as_ref(),
            "context_lens",
        )?;
        copy_option_var_map(
            &self.full_context_lens,
            metadata.full_context_lens.as_ref(),
            "full_context_lens",
        )?;
        copy_option_var_map(
            &self.paged_kv_last_page_len,
            metadata.paged_kv_last_page_len.as_ref(),
            "paged_kv_last_page_len",
        )?;
        copy_option_var_map(
            &self.full_paged_kv_last_page_len,
            metadata.full_paged_kv_last_page_len.as_ref(),
            "full_paged_kv_last_page_len",
        )?;
        if block_tables_changed {
            copy_option_var_map(
                &self.block_tables,
                metadata.block_tables.as_ref(),
                "block_tables",
            )?;
            copy_option_var_map(
                &self.paged_kv_indptr,
                metadata.paged_kv_indptr.as_ref(),
                "paged_kv_indptr",
            )?;
            copy_option_var_map(
                &self.paged_kv_indices,
                metadata.paged_kv_indices.as_ref(),
                "paged_kv_indices",
            )?;
            copy_option_var_map(
                &self.paged_kv_request_indices,
                metadata.paged_kv_request_indices.as_ref(),
                "paged_kv_request_indices",
            )?;
            copy_option_var_map(
                &self.paged_kv_tile_indices,
                metadata.paged_kv_tile_indices.as_ref(),
                "paged_kv_tile_indices",
            )?;
            copy_option_var_map(
                &self.paged_kv_o_indptr,
                metadata.paged_kv_o_indptr.as_ref(),
                "paged_kv_o_indptr",
            )?;
            copy_option_var_map(
                &self.paged_kv_chunk_size,
                metadata.paged_kv_chunk_size.as_ref(),
                "paged_kv_chunk_size",
            )?;
            copy_option_var_map(
                &self.paged_kv_block_valid_mask,
                metadata.paged_kv_block_valid_mask.as_ref(),
                "paged_kv_block_valid_mask",
            )?;
            self.block_table_signature = metadata.block_table_signature.clone();
        }
        if full_block_tables_changed {
            copy_option_var_map(
                &self.full_block_tables,
                metadata.full_block_tables.as_ref(),
                "full_block_tables",
            )?;
            copy_option_var_map(
                &self.full_paged_kv_indptr,
                metadata.full_paged_kv_indptr.as_ref(),
                "full_paged_kv_indptr",
            )?;
            copy_option_var_map(
                &self.full_paged_kv_indices,
                metadata.full_paged_kv_indices.as_ref(),
                "full_paged_kv_indices",
            )?;
            copy_option_var_map(
                &self.full_paged_kv_request_indices,
                metadata.full_paged_kv_request_indices.as_ref(),
                "full_paged_kv_request_indices",
            )?;
            copy_option_var_map(
                &self.full_paged_kv_tile_indices,
                metadata.full_paged_kv_tile_indices.as_ref(),
                "full_paged_kv_tile_indices",
            )?;
            copy_option_var_map(
                &self.full_paged_kv_o_indptr,
                metadata.full_paged_kv_o_indptr.as_ref(),
                "full_paged_kv_o_indptr",
            )?;
            copy_option_var_map(
                &self.full_paged_kv_chunk_size,
                metadata.full_paged_kv_chunk_size.as_ref(),
                "full_paged_kv_chunk_size",
            )?;
            copy_option_var_map(
                &self.full_paged_kv_block_valid_mask,
                metadata.full_paged_kv_block_valid_mask.as_ref(),
                "full_paged_kv_block_valid_mask",
            )?;
            self.full_block_table_signature = metadata.full_block_table_signature.clone();
        }
        copy_rope_positions(&self.rope_positions, seqlen_offsets)?;
        Ok(())
    }

    /// Build a [`PagedAttentionInputMetadata`] that references the stable device
    /// buffers instead of the freshly-built per-step tensors, so the warmed forward
    /// reads from the buffers that [`Self::copy_from`] updates in place.
    fn metadata_from(
        &self,
        metadata: &PagedAttentionInputMetadata,
        block_size: usize,
    ) -> PagedAttentionInputMetadata {
        PagedAttentionInputMetadata {
            block_tables: option_tensor_map_from_var_map(&self.block_tables),
            context_lens: option_tensor_map_from_var_map(&self.context_lens),
            block_size: metadata.block_size,
            paged_context_lens_cpu: metadata.paged_context_lens_cpu.clone(),
            full_paged_context_lens_cpu: metadata.full_paged_context_lens_cpu.clone(),
            slot_mappings: tensor_map_from_var_map(&self.slot_mappings),
            max_context_len: bucket_context_len_from_vars(&self.block_tables, block_size),
            full_block_tables: option_tensor_map_from_var_map(&self.full_block_tables),
            full_context_lens: option_tensor_map_from_var_map(&self.full_context_lens),
            full_max_context_len: bucket_context_len_from_vars(&self.full_block_tables, block_size),
            is_first_prompt_chunk: metadata.is_first_prompt_chunk,
            disable_cuda_graphs: metadata.disable_cuda_graphs,
            paged_kv_indptr: option_tensor_map_from_var_map(&self.paged_kv_indptr),
            paged_kv_indices: option_tensor_map_from_var_map(&self.paged_kv_indices),
            paged_kv_last_page_len: option_tensor_map_from_var_map(&self.paged_kv_last_page_len),
            full_paged_kv_indptr: option_tensor_map_from_var_map(&self.full_paged_kv_indptr),
            full_paged_kv_indices: option_tensor_map_from_var_map(&self.full_paged_kv_indices),
            full_paged_kv_last_page_len: option_tensor_map_from_var_map(
                &self.full_paged_kv_last_page_len,
            ),
            paged_kv_q_indptr: metadata.paged_kv_q_indptr.clone(),
            paged_kv_qo_tile_indices: metadata.paged_kv_qo_tile_indices.clone(),
            paged_kv_request_indices: option_tensor_map_from_var_map(&self.paged_kv_request_indices),
            paged_kv_tile_indices: option_tensor_map_from_var_map(&self.paged_kv_tile_indices),
            paged_kv_o_indptr: option_tensor_map_from_var_map(&self.paged_kv_o_indptr),
            paged_kv_chunk_size: option_tensor_map_from_var_map(&self.paged_kv_chunk_size),
            paged_kv_block_valid_mask: option_tensor_map_from_var_map(&self.paged_kv_block_valid_mask),
            block_table_signature: self.block_table_signature.clone(),
            full_paged_kv_q_indptr: metadata.full_paged_kv_q_indptr.clone(),
            full_paged_kv_qo_tile_indices: metadata.full_paged_kv_qo_tile_indices.clone(),
            full_paged_kv_request_indices: option_tensor_map_from_var_map(
                &self.full_paged_kv_request_indices,
            ),
            full_paged_kv_tile_indices: option_tensor_map_from_var_map(
                &self.full_paged_kv_tile_indices,
            ),
            full_paged_kv_o_indptr: option_tensor_map_from_var_map(&self.full_paged_kv_o_indptr),
            full_paged_kv_chunk_size: option_tensor_map_from_var_map(&self.full_paged_kv_chunk_size),
            full_paged_kv_block_valid_mask: option_tensor_map_from_var_map(
                &self.full_paged_kv_block_valid_mask,
            ),
            full_block_table_signature: self.full_block_table_signature.clone(),
            rope_positions: Some(tensor_map_from_var_map(&self.rope_positions)),
            num_cached_tokens: metadata.num_cached_tokens.clone(),
            query_lens: metadata.query_lens.clone(),
            cu_seqlens_q: metadata.cu_seqlens_q.clone(),
            cu_seqlens_kv: metadata.cu_seqlens_kv.clone(),
        }
    }
}

/// Whether Metal decode graphs are enabled. Gated on `HANZO_METAL_GRAPHS`
/// (default on), mirroring `cuda_decode_graphs_enabled`.
pub(crate) fn metal_decode_graphs_enabled() -> bool {
    crate::perf_flags::metal_graphs_enabled()
}

fn device_location_sort_key(location: &DeviceLocation) -> (u8, usize) {
    match location {
        DeviceLocation::Cpu => (0, 0),
        DeviceLocation::Cuda { gpu_id } => (1, *gpu_id),
        DeviceLocation::Metal { gpu_id } => (2, *gpu_id),
    }
}

fn push_graph_tensor_keys(
    name: &'static str,
    map: Option<&HashMap<DeviceLocation, Tensor>>,
    keys: &mut Vec<MetalGraphTensorKey>,
) {
    if let Some(map) = map {
        keys.extend(map.iter().map(|(location, tensor)| MetalGraphTensorKey {
            name,
            location: *location,
            shape: tensor.dims().to_vec(),
            dtype: tensor.dtype(),
        }));
    }
}

fn bucket_context_len(
    map: Option<&HashMap<DeviceLocation, Tensor>>,
    block_size: usize,
) -> Option<usize> {
    map.and_then(|map| map.values().next())
        .and_then(|tensor| tensor.dims().last().copied())
        .map(|blocks| blocks * block_size)
}

fn bucket_context_len_from_vars(map: &Option<MetalGraphVarMap>, block_size: usize) -> Option<usize> {
    map.as_ref()
        .and_then(|map| map.values().next())
        .and_then(|var| var.as_tensor().dims().last().copied())
        .map(|blocks| blocks * block_size)
}

fn signature_changed(previous: &Option<Vec<u64>>, current: &Option<Vec<u64>>) -> bool {
    match (previous, current) {
        (Some(previous), Some(current)) => previous != current,
        _ => true,
    }
}

fn var_map_from_tensor_map(
    map: &HashMap<DeviceLocation, Tensor>,
) -> hanzo_ml::Result<MetalGraphVarMap> {
    map.iter()
        .map(|(location, tensor)| Ok((*location, Var::from_tensor(tensor)?)))
        .collect()
}

fn option_var_map_from_tensor_map(
    map: Option<&HashMap<DeviceLocation, Tensor>>,
) -> hanzo_ml::Result<Option<MetalGraphVarMap>> {
    map.map(var_map_from_tensor_map).transpose()
}

fn tensor_map_from_var_map(map: &MetalGraphVarMap) -> HashMap<DeviceLocation, Tensor> {
    map.iter()
        .map(|(location, var)| (*location, var.as_detached_tensor()))
        .collect()
}

fn option_tensor_map_from_var_map(
    map: &Option<MetalGraphVarMap>,
) -> Option<HashMap<DeviceLocation, Tensor>> {
    map.as_ref().map(tensor_map_from_var_map)
}

fn copy_var_map(
    dst: &MetalGraphVarMap,
    src: &HashMap<DeviceLocation, Tensor>,
    name: &str,
) -> hanzo_ml::Result<()> {
    if dst.len() != src.len() {
        hanzo_ml::bail!("{name} device count changed during Metal decode-graph replay");
    }
    for (location, dst) in dst {
        let src = src
            .get(location)
            .ok_or_else(|| hanzo_ml::Error::msg(format!("{name} missing {location:?}")))?;
        dst.set(src)?;
    }
    Ok(())
}

fn copy_option_var_map(
    dst: &Option<MetalGraphVarMap>,
    src: Option<&HashMap<DeviceLocation, Tensor>>,
    name: &str,
) -> hanzo_ml::Result<()> {
    match (dst, src) {
        (Some(dst), Some(src)) => copy_var_map(dst, src, name),
        (None, None) => Ok(()),
        _ => hanzo_ml::bail!("{name} changed optional state during Metal decode-graph replay"),
    }
}

fn rope_positions_tensor(seqlen_offsets: &[usize], device: &Device) -> hanzo_ml::Result<Tensor> {
    let mut positions = Vec::with_capacity(seqlen_offsets.len());
    for offset in seqlen_offsets {
        positions.push(u32::try_from(*offset).map_err(hanzo_ml::Error::wrap)?);
    }
    Tensor::from_vec(positions, (seqlen_offsets.len(),), device)
}

fn rope_positions_var_map(
    slot_mappings: &HashMap<DeviceLocation, Tensor>,
    seqlen_offsets: &[usize],
) -> hanzo_ml::Result<MetalGraphVarMap> {
    slot_mappings
        .iter()
        .map(|(location, tensor)| {
            let positions = rope_positions_tensor(seqlen_offsets, tensor.device())?;
            Ok((*location, Var::from_tensor(&positions)?))
        })
        .collect()
}

fn copy_rope_positions(dst: &MetalGraphVarMap, seqlen_offsets: &[usize]) -> hanzo_ml::Result<()> {
    for dst in dst.values() {
        let positions = rope_positions_tensor(seqlen_offsets, dst.device())?;
        dst.set(&positions)?;
    }
    Ok(())
}

// =============================================================================
// NEEDS MAC VALIDATION — self-review notes (authored on a non-Metal host)
// =============================================================================
//
// This module compiles only under `--features metal` and was *not* built or run
// here (Windows/AMD box). The following items must be validated on an Apple Silicon
// Mac:
//
// 1. INVARIANCE (verified by reading the kernel, must be confirmed on-device):
//    `pagedattention.metal` reads the per-sequence length from the device buffer
//    `context_lens[seq_idx]` (lines ~808 and ~1176) and bounds every block/token
//    loop by it (`num_context_blocks = DIVIDE_ROUND_UP(context_len, BLOCK_SIZE)`,
//    masking at `token_idx >= context_len`). The host scalar `max_context_len`
//    only sets launch geometry: `max_num_partitions` (grid `.z`), the v1-vs-v2
//    choice, and `setThreadgroupMemoryLength`. Because `MetalDecodeGraphKey` buckets
//    `max_context_len` to a block multiple, those launch dims stay fixed for a key,
//    so re-encoding the decode step for a new token produces an identical dispatch
//    sequence while the live `context_lens` buffer drives the math. This is the same
//    contract the CUDA graph relies on. CONFIRM the kernel never uses the host
//    `max_context_len` to index/iterate (only for geometry) on the validated commit.
//
// 2. `Var::set` semantics: `variable.rs::set` performs `copy_strided_src` into the
//    existing storage (in-place, buffer identity preserved). On Metal that lowers to
//    a blit/compute copy through the shared `Commands` encoder, so updating the
//    stable buffers does NOT reallocate. Confirm there is no hidden realloc for the
//    Metal storage path and that contiguity holds (set bails on non-contiguous).
//
// 3. The pipeline gate (in `normal.rs`, behind `#[cfg(feature = "metal")]`) must
//    additionally warm the step with a normal forward + `device.synchronize()`
//    BEFORE caching the entry, exactly like the CUDA path issues a warmup forward
//    before capture. There is no `cudaGraphLaunch` analog to call on replay: replay
//    is simply re-running the model forward with the cached metadata buffers (whose
//    contents were refreshed by `copy_from`). The win comes from (a) not rebuilding
//    metadata tensors per token and (b) the command-buffer batching coalescing the
//    per-step dispatches; it is NOT a single-dispatch graph launch. This must be
//    benchmarked on-device to confirm the overhead actually drops vs the plain path;
//    if it does not beat the plain forward, the gate should default the env flag off.
//
// 4. If on-device profiling shows the remaining win requires fewer command-buffer
//    commits per step, the follow-up (out of scope for this port) is to raise
//    `CANDLE_METAL_COMPUTE_PER_BUFFER` for the decode step or to expose a
//    non-blocking `Commands::flush` on `MetalDevice` so the step commits once.
//    Neither requires changing this module's types.
