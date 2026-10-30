// Rust FFI for the ROCm/HIP PagedAttention kernels.
//
// Mirrors src/cuda/ffi.rs for the CORE decode path (paged_attention_v1_* and
// _v2_*, and reshape_and_cache). The attention launchers return the HIP status of
// their own launch, which the caller must read: a launch the device refuses is
// otherwise reported by whatever kernel runs next, under that kernel's name.
//
// The HIP launchers take the stream as an opaque
// `int64_t` (a `hipStream_t` reinterpret-cast), exactly like the engine's
// src/rocm/sort.hip.cpp, so `stream` here is `i64` rather than a typed handle.

use core::ffi::{c_int, c_long, c_void};

extern "C" {
    pub fn reshape_and_cache(
        key: *const c_void,
        value: *const c_void,
        key_cache: *const c_void,
        value_cache: *const c_void,
        slot_mapping: *const c_long,

        num_tokens: c_int,
        num_heads: c_int,
        head_size: c_int,
        block_size: c_int,
        x: c_int,
        key_stride: c_int,
        value_stride: c_int,
        stream: i64,

        dtype: u32,
        cache_dtype: u32,
        k_scale: *const f32,
        v_scale: *const f32,
    );

    pub fn paged_attention_v1_f16(
        out: *const c_void,
        query: *const c_void,
        key_cache: *const c_void,
        value_cache: *const c_void,
        alibi_slopes: *const c_void,
        num_kv_heads: c_int,
        scale: f32,
        softcapping: f32,
        block_tables: *const c_int,
        context_lens: *const c_int,
        block_size: c_int,
        max_context_len: c_int,
        num_seqs: c_int,
        num_heads: c_int,
        head_size: c_int,
        max_num_blocks_per_seq: c_int,
        q_stride: c_int,
        kv_block_stride: c_int,
        kv_head_stride: c_int,
        stream: i64,
        cache_dtype: u32,
        k_scale: *const f32,
        v_scale: *const f32,
        sinks: *const f32,
    ) -> c_int;

    pub fn paged_attention_v1_bf16(
        out: *const c_void,
        query: *const c_void,
        key_cache: *const c_void,
        value_cache: *const c_void,
        alibi_slopes: *const c_void,
        num_kv_heads: c_int,
        scale: f32,
        softcapping: f32,
        block_tables: *const c_int,
        context_lens: *const c_int,
        block_size: c_int,
        max_context_len: c_int,
        num_seqs: c_int,
        num_heads: c_int,
        head_size: c_int,
        max_num_blocks_per_seq: c_int,
        q_stride: c_int,
        kv_block_stride: c_int,
        kv_head_stride: c_int,
        stream: i64,
        cache_dtype: u32,
        k_scale: *const f32,
        v_scale: *const f32,
        sinks: *const f32,
    ) -> c_int;

    /// 1 when v1's shared memory for `max_context_len` fits one workgroup on the
    /// current device, 0 when it does not or cannot be established.
    pub fn paged_attention_v1_fits(max_context_len: c_int, block_size: c_int) -> c_int;

    pub fn paged_attention_v2_f16(
        out: *const c_void,
        exp_sums: *const c_void,
        max_logits: *const c_void,
        tmp_out: *const c_void,
        query: *const c_void,
        key_cache: *const c_void,
        value_cache: *const c_void,
        alibi_slopes: *const c_void,
        num_kv_heads: c_int,
        scale: f32,
        softcapping: f32,
        block_tables: *const c_int,
        context_lens: *const c_int,
        block_size: c_int,
        max_context_len: c_int,
        num_seqs: c_int,
        num_heads: c_int,
        head_size: c_int,
        max_num_blocks_per_seq: c_int,
        q_stride: c_int,
        kv_block_stride: c_int,
        kv_head_stride: c_int,
        stream: i64,
        cache_dtype: u32,
        sinks: *const f32,
    ) -> c_int;

    pub fn paged_attention_v2_bf16(
        out: *const c_void,
        exp_sums: *const c_void,
        max_logits: *const c_void,
        tmp_out: *const c_void,
        query: *const c_void,
        key_cache: *const c_void,
        value_cache: *const c_void,
        alibi_slopes: *const c_void,
        num_kv_heads: c_int,
        scale: f32,
        softcapping: f32,
        block_tables: *const c_int,
        context_lens: *const c_int,
        block_size: c_int,
        max_context_len: c_int,
        num_seqs: c_int,
        num_heads: c_int,
        head_size: c_int,
        max_num_blocks_per_seq: c_int,
        q_stride: c_int,
        kv_block_stride: c_int,
        kv_head_stride: c_int,
        stream: i64,
        cache_dtype: u32,
        sinks: *const f32,
    ) -> c_int;
}
