//! ROCm/HIP decode graph capture + replay, mirroring `cuda_graph.rs` 1:1 for the
//! position-invariant paged decode forward.
//!
//! Stages 1+2 made the ROCm GGUF decode step position-invariant: the paged
//! attention kernel reads `context_lens` from device memory and takes only the
//! bucketed `max_context_len` (a shape/capacity bound) as a host scalar, and the
//! decode RoPE indexes a device `positions` tensor. The only per-token-varying
//! inputs are therefore the *contents* of a fixed set of device buffers
//! (`input_ids`, `slot_mappings`, `context_lens`, `rope_positions`). This module
//! captures the whole decode forward into a `hipGraph` once per bucket and, on
//! every subsequent token, refreshes those buffers in place via [`Var::set`]
//! (an in-place strided device copy into stable storage) and replays the graph
//! with a single `hipGraphLaunch`, collapsing all per-token launch/scheduling
//! overhead.
//!
//! Unlike the CUDA path this drops the FlashInfer `full_*`/`paged_kv_*` metadata
//! (the ROCm v1 backend does not use them) and keeps only the four buffers the
//! standard ROCm paged decode consumes.

use std::collections::HashMap;

use hanzo_ml::rocm_backend::rocm_rs::hip::bindings as hip;
use hanzo_ml::{DType, Device, DeviceLocation, RocmDevice, Tensor, Var};

use crate::pipeline::text_models_inputs_processor::PagedAttentionInputMetadata;

/// Instantiate flags = 0 (no AutoFreeOnLaunch).
///
/// `hipGraphInstantiateFlagAutoFreeOnLaunch` is correct ONLY when paired with a
/// graph-ordered (stream-ordered) allocator, where per-launch temporaries are
/// graph alloc-nodes the flag deterministically re-frees/re-allocates each
/// launch (the CUDA path). The ROCm backend has NO graph-ordered allocator —
/// every buffer is a plain `hipMalloc` whose fixed pointer is baked into the
/// kernel nodes and reserved out of the caching pool for the graph's lifetime
/// (see `wrappers::PoolInner`). Under that contract AutoFreeOnLaunch makes the
/// runtime recycle the reserved scratch/logits across launches, so the cached
/// `logits` read back frozen data. flags = 0 keeps the baked pointers stable
/// across replays, which is exactly the buffer-address-stability invariant this
/// path depends on.
const ROCM_GRAPH_INSTANTIATE_FLAGS: ::std::os::raw::c_ulonglong = 0;

/// Relaxed capture mode: tolerates the rocBLAS / JIT-module launches the decode
/// forward performs without aborting capture (mirrors CUDA's RELAXED mode).
const ROCM_CAPTURE_MODE_RELAXED: hip::hipStreamCaptureMode =
    hip::hipStreamCaptureMode_hipStreamCaptureModeRelaxed;

pub(crate) const ROCM_DECODE_GRAPH_CACHE_CAPACITY: usize = 32;

/// Owns a captured + instantiated HIP graph for one decode bucket. Mirrors
/// `CudaGraphHandle`.
pub(crate) struct RocmGraphHandle {
    graph: hip::hipGraph_t,
    exec: hip::hipGraphExec_t,
    /// Kept alive so the captured stream outlives the graph; the device owns the
    /// real stream, this is the handle we replay onto.
    device: RocmDevice,
}

// The raw HIP graph/exec pointers are only ever touched on the owning device's
// stream; the handle is moved between the capture and the cached entry. Mirrors
// `unsafe impl Send for CudaGraphHandle`.
unsafe impl Send for RocmGraphHandle {}

impl Drop for RocmGraphHandle {
    fn drop(&mut self) {
        // Quiesce the stream before tearing down so no in-flight replay reads the
        // graph we are about to destroy (mirrors cuda_graph.rs Drop).
        let _ = self.device.synchronize();
        if !self.exec.is_null() {
            unsafe {
                let _ = hip::hipGraphExecDestroy(self.exec);
            }
            self.exec = std::ptr::null_mut();
        }
        if !self.graph.is_null() {
            unsafe {
                let _ = hip::hipGraphDestroy(self.graph);
            }
            self.graph = std::ptr::null_mut();
        }
    }
}

impl RocmGraphHandle {
    /// Ends capture on the device stream and instantiates the executable graph.
    /// Returns `Ok(None)` if capture produced no graph (caller falls back to
    /// eager). Mirrors `CudaGraphHandle::end_capture`.
    pub(crate) fn end_capture(device: &RocmDevice) -> hanzo_ml::Result<Option<Self>> {
        let stream = device.stream().as_raw();
        let mut graph: hip::hipGraph_t = std::ptr::null_mut();
        let result = unsafe { hip::hipStreamEndCapture(stream, &mut graph) };
        if result != hip::hipError_t_hipSuccess {
            return Err(hanzo_ml::Error::msg(format!("{result:?}"))
                .context("ROCm graph stream end capture failed"));
        }
        if graph.is_null() {
            return Ok(None);
        }

        let mut exec: hip::hipGraphExec_t = std::ptr::null_mut();
        let result = unsafe {
            hip::hipGraphInstantiateWithFlags(&mut exec, graph, ROCM_GRAPH_INSTANTIATE_FLAGS)
        };
        if result != hip::hipError_t_hipSuccess {
            unsafe {
                let _ = hip::hipGraphDestroy(graph);
            }
            return Err(hanzo_ml::Error::msg(format!("{result:?}"))
                .context("ROCm graph instantiate failed"));
        }

        Ok(Some(Self {
            graph,
            exec,
            device: device.clone(),
        }))
    }

    /// Uploads the executable graph to the device ahead of the first replay so the
    /// first `launch` does not pay the upload cost. Mirrors `CudaGraphHandle::upload`.
    pub(crate) fn upload(&self) -> hanzo_ml::Result<()> {
        let stream = self.device.stream().as_raw();
        let result = unsafe { hip::hipGraphUpload(self.exec, stream) };
        if result != hip::hipError_t_hipSuccess {
            return Err(
                hanzo_ml::Error::msg(format!("{result:?}")).context("ROCm graph upload failed")
            );
        }
        Ok(())
    }

    /// Replays the captured decode forward in one launch. Mirrors
    /// `CudaGraphHandle::launch`.
    pub(crate) fn launch(&self) -> hanzo_ml::Result<()> {
        let stream = self.device.stream().as_raw();
        let result = unsafe { hip::hipGraphLaunch(self.exec, stream) };
        if result != hip::hipError_t_hipSuccess {
            return Err(
                hanzo_ml::Error::msg(format!("{result:?}")).context("ROCm graph launch failed")
            );
        }
        Ok(())
    }
}

/// Bucket key for a captured decode graph. Two decode steps share a graph iff
/// they agree on device, input shape/dtype, the bucketed context capacity, and
/// the shape/dtype/location of every rebindable metadata tensor. Mirrors
/// `CudaDecodeGraphKey` minus the FlashInfer/full_* fields.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct RocmDecodeGraphKey {
    device: DeviceLocation,
    input_shape: Vec<usize>,
    input_dtype: DType,
    max_context_len: Option<usize>,
    tensors: Vec<RocmGraphTensorKey>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct RocmGraphTensorKey {
    name: &'static str,
    location: DeviceLocation,
    shape: Vec<usize>,
    dtype: DType,
}

type RocmGraphVarMap = HashMap<DeviceLocation, Var>;

/// Stable device buffers backing a captured decode graph. The graph's kernels
/// were captured reading these exact allocations; [`Self::copy_from`] refreshes
/// their contents in place between replays. Mirrors `CudaDecodeGraphMetadataBuffers`.
pub(crate) struct RocmDecodeGraphMetadataBuffers {
    slot_mappings: RocmGraphVarMap,
    block_tables: Option<RocmGraphVarMap>,
    context_lens: Option<RocmGraphVarMap>,
    /// Full-attention block tables / context lens. Models without a sliding
    /// window (e.g. Qwen3) take the `use_full` decode path in
    /// `paged_attention::layers::paged_attention`, where the paged-attn kernel
    /// reads `full_context_lens` / `full_block_tables` instead of the windowed
    /// ones. These MUST be refreshed every token exactly like their non-full
    /// counterparts — otherwise the captured kernel attends a FROZEN context
    /// span (the warmup length) and decode degenerates. Mirrors CUDA.
    full_block_tables: Option<RocmGraphVarMap>,
    full_context_lens: Option<RocmGraphVarMap>,
    rope_positions: RocmGraphVarMap,
    /// Last-seen block-table contents; block tables only change when the paging
    /// layout shifts, so we skip the copy otherwise (mirrors CUDA's signature).
    block_table_signature: Option<Vec<u64>>,
    full_block_table_signature: Option<Vec<u64>>,
}

impl RocmDecodeGraphKey {
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
            "rope_positions",
            metadata.rope_positions.as_ref(),
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
            tensors,
        })
    }
}

impl RocmDecodeGraphMetadataBuffers {
    /// Allocates the stable buffers from the warmup metadata and returns a rebound
    /// [`PagedAttentionInputMetadata`] whose tensors point at those buffers, so the
    /// captured forward reads the stable allocations. Mirrors
    /// `CudaDecodeGraphMetadataBuffers::new`.
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
            rope_positions,
            block_table_signature: metadata.block_table_signature.clone(),
            full_block_table_signature: metadata.full_block_table_signature.clone(),
        };
        let metadata = buffers.metadata_from(metadata, block_size);
        Ok((buffers, metadata))
    }

    /// Refreshes the stable buffers in place with the contents for the current
    /// token before replaying. Mirrors `CudaDecodeGraphMetadataBuffers::copy_from`.
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

        // slot_mappings and (full_)context_lens advance every token (new KV slot
        // + the device-resident running context length), so always refresh them.
        // Models without a sliding window (Qwen3) read full_context_lens in the
        // paged-attn kernel via the `use_full` path, so it MUST advance too —
        // freezing it pins attention to the warmup span and decode degenerates.
        copy_var_map(
            &self.slot_mappings,
            &metadata.slot_mappings,
            "slot_mappings",
        )?;
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
        // block_tables only change when the paging layout shifts (new block
        // allocated); skip the copy otherwise. The bucketed key guarantees the
        // shape is unchanged within a graph.
        if block_tables_changed {
            copy_option_var_map(
                &self.block_tables,
                metadata.block_tables.as_ref(),
                "block_tables",
            )?;
            self.block_table_signature = metadata.block_table_signature.clone();
        }
        if full_block_tables_changed {
            copy_option_var_map(
                &self.full_block_tables,
                metadata.full_block_tables.as_ref(),
                "full_block_tables",
            )?;
            self.full_block_table_signature = metadata.full_block_table_signature.clone();
        }
        // RoPE position advances by one per token; recompute on device in place.
        copy_rope_positions(&self.rope_positions, seqlen_offsets)?;
        Ok(())
    }

    /// Builds a [`PagedAttentionInputMetadata`] whose rebindable tensors alias the
    /// stable buffers (so the captured forward and every replay read the same
    /// allocations), copying through the scalar / passthrough fields verbatim.
    /// Mirrors `CudaDecodeGraphMetadataBuffers::metadata_from` for the ROCm subset.
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
            // Alias the STABLE full buffers (refreshed in copy_from) so the
            // captured `use_full` paged-attn reads the advancing context span,
            // not a frozen warmup clone. full_max_context_len is the bucketed
            // capacity (position-invariant), matching CUDA.
            full_block_tables: option_tensor_map_from_var_map(&self.full_block_tables),
            full_context_lens: option_tensor_map_from_var_map(&self.full_context_lens),
            full_max_context_len: bucket_context_len_from_vars(&self.full_block_tables, block_size),
            is_first_prompt_chunk: metadata.is_first_prompt_chunk,
            disable_cuda_graphs: metadata.disable_cuda_graphs,
            paged_kv_indptr: metadata.paged_kv_indptr.clone(),
            paged_kv_indices: metadata.paged_kv_indices.clone(),
            paged_kv_last_page_len: metadata.paged_kv_last_page_len.clone(),
            full_paged_kv_indptr: metadata.full_paged_kv_indptr.clone(),
            full_paged_kv_indices: metadata.full_paged_kv_indices.clone(),
            full_paged_kv_last_page_len: metadata.full_paged_kv_last_page_len.clone(),
            paged_kv_q_indptr: metadata.paged_kv_q_indptr.clone(),
            paged_kv_qo_tile_indices: metadata.paged_kv_qo_tile_indices.clone(),
            paged_kv_request_indices: metadata.paged_kv_request_indices.clone(),
            paged_kv_tile_indices: metadata.paged_kv_tile_indices.clone(),
            paged_kv_o_indptr: metadata.paged_kv_o_indptr.clone(),
            paged_kv_chunk_size: metadata.paged_kv_chunk_size.clone(),
            paged_kv_block_valid_mask: metadata.paged_kv_block_valid_mask.clone(),
            block_table_signature: self.block_table_signature.clone(),
            full_paged_kv_q_indptr: metadata.full_paged_kv_q_indptr.clone(),
            full_paged_kv_qo_tile_indices: metadata.full_paged_kv_qo_tile_indices.clone(),
            full_paged_kv_request_indices: metadata.full_paged_kv_request_indices.clone(),
            full_paged_kv_tile_indices: metadata.full_paged_kv_tile_indices.clone(),
            full_paged_kv_o_indptr: metadata.full_paged_kv_o_indptr.clone(),
            full_paged_kv_chunk_size: metadata.full_paged_kv_chunk_size.clone(),
            full_paged_kv_block_valid_mask: metadata.full_paged_kv_block_valid_mask.clone(),
            full_block_table_signature: metadata.full_block_table_signature.clone(),
            rope_positions: Some(tensor_map_from_var_map(&self.rope_positions)),
            num_cached_tokens: metadata.num_cached_tokens.clone(),
            query_lens: metadata.query_lens.clone(),
            cu_seqlens_q: metadata.cu_seqlens_q.clone(),
            cu_seqlens_kv: metadata.cu_seqlens_kv.clone(),
        }
    }
}

pub(crate) fn rocm_decode_graphs_enabled() -> bool {
    crate::perf_flags::rocm_graphs_enabled()
}

/// If the stream is mid-capture, end (and discard) the capture so a failed
/// partial capture does not leave the stream in capturing state, then drain the
/// stream so any error latched during capture (e.g. `hipErrorStreamCaptureImplicit`)
/// is cleared before the caller falls back to eager execution. Without the final
/// drain a failed capture leaves the stream invalidated and the next eager launch
/// fails with `hipErrorStreamCaptureInvalidated`. Mirrors `end_cuda_capture_discard`
/// but adds the explicit reset the ROCm runtime needs.
pub(crate) fn end_rocm_capture_discard(device: &RocmDevice) {
    let stream = device.stream().as_raw();
    unsafe {
        // A capture invalidated mid-flight (e.g. by hipErrorStreamCaptureImplicit)
        // stays in the capturing state until `hipStreamEndCapture` terminates it.
        // The terminating call itself returns the invalidation error and may need
        // to be repeated until the stream reports it is no longer capturing, so we
        // loop (bounded) rather than calling it once.
        for _ in 0..8 {
            let mut status: hip::hipStreamCaptureStatus =
                hip::hipStreamCaptureStatus_hipStreamCaptureStatusNone;
            let info = hip::hipStreamIsCapturing(stream, &mut status);
            if info != hip::hipError_t_hipSuccess
                || status == hip::hipStreamCaptureStatus_hipStreamCaptureStatusNone
            {
                break;
            }
            let mut graph: hip::hipGraph_t = std::ptr::null_mut();
            let _ = hip::hipStreamEndCapture(stream, &mut graph);
            if !graph.is_null() {
                let _ = hip::hipGraphDestroy(graph);
            }
        }
        // Drain the whole device and clear the sticky/last error the failed capture
        // latched, so the next (eager) launch starts from a clean state. A plain
        // stream sync is insufficient here: the device-wide sync resets the capture
        // bookkeeping the runtime keyed to the aborted sequence.
        let _ = hip::hipDeviceSynchronize();
        let _ = hip::hipGetLastError();
    }
    let _ = device.synchronize();
    unsafe {
        let _ = hip::hipGetLastError();
    }
}

/// Begins capture on the device stream in RELAXED mode.
pub(crate) fn begin_rocm_capture(device: &RocmDevice) -> hanzo_ml::Result<()> {
    let stream = device.stream().as_raw();
    let result = unsafe { hip::hipStreamBeginCapture(stream, ROCM_CAPTURE_MODE_RELAXED) };
    if result != hip::hipError_t_hipSuccess {
        return Err(
            hanzo_ml::Error::msg(format!("{result:?}")).context("ROCm graph begin capture failed")
        );
    }
    Ok(())
}

/// Extracts the [`RocmDevice`] from a hanzo-ml [`Device`], or errors if it is not
/// a ROCm device.
pub(crate) fn rocm_device(device: &Device) -> hanzo_ml::Result<RocmDevice> {
    match device {
        Device::Rocm(d) => Ok(d.clone()),
        _ => hanzo_ml::bail!("ROCm decode graph expected a ROCm device"),
    }
}

fn device_location_sort_key(location: &DeviceLocation) -> (u8, usize) {
    match location {
        DeviceLocation::Cpu => (0, 0),
        DeviceLocation::Cuda { gpu_id } => (1, *gpu_id),
        DeviceLocation::Metal { gpu_id } => (2, *gpu_id),
        #[cfg(feature = "rocm")]
        DeviceLocation::Rocm { gpu_id } => (3, *gpu_id),
        #[allow(unreachable_patterns)]
        _ => (u8::MAX, 0),
    }
}

fn push_graph_tensor_keys(
    name: &'static str,
    map: Option<&HashMap<DeviceLocation, Tensor>>,
    keys: &mut Vec<RocmGraphTensorKey>,
) {
    if let Some(map) = map {
        keys.extend(map.iter().map(|(location, tensor)| RocmGraphTensorKey {
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

fn bucket_context_len_from_vars(map: &Option<RocmGraphVarMap>, block_size: usize) -> Option<usize> {
    map.as_ref()
        .and_then(|map| map.values().next())
        .and_then(|var| var.as_detached_tensor().dims().last().copied())
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
) -> hanzo_ml::Result<RocmGraphVarMap> {
    map.iter()
        .map(|(location, tensor)| Ok((*location, Var::from_tensor(tensor)?)))
        .collect()
}

fn option_var_map_from_tensor_map(
    map: Option<&HashMap<DeviceLocation, Tensor>>,
) -> hanzo_ml::Result<Option<RocmGraphVarMap>> {
    map.map(var_map_from_tensor_map).transpose()
}

fn tensor_map_from_var_map(map: &RocmGraphVarMap) -> HashMap<DeviceLocation, Tensor> {
    map.iter()
        .map(|(location, var)| (*location, var.as_detached_tensor()))
        .collect()
}

fn option_tensor_map_from_var_map(
    map: &Option<RocmGraphVarMap>,
) -> Option<HashMap<DeviceLocation, Tensor>> {
    map.as_ref().map(tensor_map_from_var_map)
}

fn copy_var_map(
    dst: &RocmGraphVarMap,
    src: &HashMap<DeviceLocation, Tensor>,
    name: &str,
) -> hanzo_ml::Result<()> {
    if dst.len() != src.len() {
        hanzo_ml::bail!("{name} device count changed during ROCm graph replay");
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
    dst: &Option<RocmGraphVarMap>,
    src: Option<&HashMap<DeviceLocation, Tensor>>,
    name: &str,
) -> hanzo_ml::Result<()> {
    match (dst, src) {
        (Some(dst), Some(src)) => copy_var_map(dst, src, name),
        (None, None) => Ok(()),
        _ => hanzo_ml::bail!("{name} changed optional state during ROCm graph replay"),
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
) -> hanzo_ml::Result<RocmGraphVarMap> {
    slot_mappings
        .iter()
        .map(|(location, tensor)| {
            let positions = rope_positions_tensor(seqlen_offsets, tensor.device())?;
            Ok((*location, Var::from_tensor(&positions)?))
        })
        .collect()
}

fn copy_rope_positions(dst: &RocmGraphVarMap, seqlen_offsets: &[usize]) -> hanzo_ml::Result<()> {
    for dst in dst.values() {
        let positions = rope_positions_tensor(seqlen_offsets, dst.device())?;
        dst.set(&positions)?;
    }
    Ok(())
}
