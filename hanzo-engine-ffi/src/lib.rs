//! C ABI over the Hanzo native inference engine — full parity with the Rust VM.
//!
//! Exposes both text generation (`hanzo_ffi_infer`) and embeddings
//! (`hanzo_ffi_embed`), routed by model name through the engine's NATIVE
//! multi-model support, so non-Rust callers (the Go EVM AI precompile via cgo)
//! can use ANY loaded zen / zen-embedding model in-process. The engine is built
//! once (lazy, from env) holding every configured model.
//!
//! Config (read on first call): `FFI_MODELS` = `;`-separated list of
//! `name=kind:source`, where `kind` ∈ {`gguf`,`plain`,`embedding`}:
//!   - `gguf:/abs/path/model.gguf`  (causal GGUF; tokenizer from the file's dir
//!      or `FFI_TOK_DIR` if set)
//!   - `plain:<hf-repo-or-dir>`     (causal safetensors)
//!   - `embedding:<hf-repo-or-dir>` (embedding model; arch auto-detected)
//! The first entry is the default model. Example:
//!   `FFI_MODELS="zen-nano=gguf:/tmp/zen5-weights/zen-5-flash.gguf;zen-embed=embedding:/tmp/zen-embedding-0.6B"`

#![allow(clippy::doc_lazy_continuation, clippy::doc_overindented_list_items)]

use std::ffi::c_int;
use std::sync::OnceLock;

use hanzo_server_core::server::{load_model_at_runtime, ModelConfig, ServerBuilder};

/// Lazy one-time multi-model engine build. `Ok(())` once registered.
fn ensure_engine() -> Result<(), String> {
    static INIT: OnceLock<Result<(), String>> = OnceLock::new();
    INIT.get_or_init(|| {
        if hanzo_engine::inference_engine_registered() {
            return Ok(());
        }
        let spec = std::env::var("FFI_MODELS")
            .map_err(|_| "set FFI_MODELS (name=kind:source;...)".to_string())?;
        let tok_dir = std::env::var("FFI_TOK_DIR").ok();
        let configs = hanzo_engine::parse_model_spec(&spec, tok_dir.as_deref())?;
        let default_id = configs[0].0.clone();
        let mut builder = ServerBuilder::new();
        for (name, model) in configs {
            // alias == name makes `name` the routable model id (get_sender).
            builder =
                builder.add_model_config(ModelConfig::new(name.clone(), model).with_alias(name));
        }
        builder = builder.with_default_model_id(default_id);

        let rt = tokio::runtime::Runtime::new().map_err(|e| e.to_string())?;
        let hanzo = rt
            .block_on(builder.build())
            .map_err(|e| format!("engine build failed: {e}"))?;
        hanzo_engine::register_engine(hanzo);
        Ok(())
    })
    .clone()
}

/// Trim a NUL-padded / whitespace model name; empty → "" (engine default).
unsafe fn model_str(ptr: *const u8, len: usize) -> String {
    if ptr.is_null() || len == 0 {
        return String::new();
    }
    String::from_utf8_lossy(std::slice::from_raw_parts(ptr, len))
        .trim_matches('\0')
        .trim()
        .to_string()
}

/// Load all configured models if needed. Returns 1 when ready, 0 on failure.
#[no_mangle]
pub extern "C" fn hanzo_ffi_ready() -> c_int {
    match ensure_engine() {
        Ok(()) => 1,
        Err(_) => 0,
    }
}

/// Text generation on model `model` (NUL/empty = default). On success (0) writes
/// a heap UTF-8 buffer to `*out`/`*out_len`; free with [`hanzo_ffi_free`].
/// Errors: -1 bad args, -2 engine unavailable, -3 inference failed.
///
/// # Safety
/// Pointers must be valid for their lengths; `out`/`out_len` writable.
#[no_mangle]
pub unsafe extern "C" fn hanzo_ffi_infer(
    model: *const u8,
    model_len: usize,
    prompt: *const u8,
    prompt_len: usize,
    out: *mut *mut u8,
    out_len: *mut usize,
) -> c_int {
    if prompt.is_null() || out.is_null() || out_len.is_null() {
        return -1;
    }
    if ensure_engine().is_err() {
        return -2;
    }
    let name = model_str(model, model_len);
    let prompt_slice = std::slice::from_raw_parts(prompt, prompt_len);
    match hanzo_engine::infer(&name, prompt_slice) {
        Ok(mut bytes) => {
            bytes.shrink_to_fit();
            let len = bytes.len();
            let ptr = bytes.as_mut_ptr();
            std::mem::forget(bytes);
            *out = ptr;
            *out_len = len;
            0
        }
        Err(_) => -3,
    }
}

/// Embedding on model `model` (NUL/empty = default). On success (0) writes a
/// heap `f32` buffer to `*out`/`*out_count` (the model's native dimension);
/// free with [`hanzo_ffi_free_f32`]. Errors: -1 bad args, -2 engine
/// unavailable, -3 embedding failed.
///
/// # Safety
/// Pointers must be valid for their lengths; `out`/`out_count` writable.
#[no_mangle]
pub unsafe extern "C" fn hanzo_ffi_embed(
    model: *const u8,
    model_len: usize,
    text: *const u8,
    text_len: usize,
    out: *mut *mut f32,
    out_count: *mut usize,
) -> c_int {
    if text.is_null() || out.is_null() || out_count.is_null() {
        return -1;
    }
    if ensure_engine().is_err() {
        return -2;
    }
    let name = model_str(model, model_len);
    let text_slice = std::slice::from_raw_parts(text, text_len);
    match hanzo_engine::embed(&name, text_slice) {
        Ok(mut v) => {
            v.shrink_to_fit();
            let count = v.len();
            let ptr = v.as_mut_ptr();
            std::mem::forget(v);
            *out = ptr;
            *out_count = count;
            0
        }
        Err(_) => -3,
    }
}

/// Load one model into the LIVE engine at runtime, routable immediately by
/// `name`. `kind` ∈ {`gguf`,`plain`,`embedding`}; `source` is the same value the
/// startup `FFI_MODELS` spec uses (abs `.gguf` path, or HF repo / local
/// dir). The engine must already be up (`hanzo_ffi_ready`); the new model is
/// added incrementally via `Hanzo::add_model` without disturbing loaded models.
/// GGUF tokenizer resolution honors `FFI_TOK_DIR` exactly as at startup.
///
/// Returns 0 on success. Errors: -1 bad args, -2 engine unavailable,
/// -3 load failed (bad spec, missing weights, or name/alias conflict).
///
/// # Safety
/// Each `(ptr, len)` pair must be a valid readable buffer (or len 0).
#[no_mangle]
pub unsafe extern "C" fn hanzo_ffi_load(
    name: *const u8,
    name_len: usize,
    kind: *const u8,
    kind_len: usize,
    source: *const u8,
    source_len: usize,
) -> c_int {
    let name = model_str(name, name_len);
    let kind = model_str(kind, kind_len);
    let source = model_str(source, source_len);
    if name.is_empty() || kind.is_empty() || source.is_empty() {
        return -1;
    }
    if ensure_engine().is_err() {
        return -2;
    }
    let Some(hanzo) = hanzo_engine::engine_handle() else {
        return -2;
    };

    // Reuse the one spec format: `name=kind:source` parses to exactly one config.
    let spec = format!("{name}={kind}:{source}");
    let tok_dir = std::env::var("FFI_TOK_DIR").ok();
    let configs = match hanzo_engine::parse_model_spec(&spec, tok_dir.as_deref()) {
        Ok(c) => c,
        Err(_) => return -3,
    };
    let Some((cfg_name, model)) = configs.into_iter().next() else {
        return -3;
    };

    // Fail-secure pre-flight: reject a duplicate id at the boundary, before any
    // weights load or engine thread spawns. `Hanzo::add_model` also rejects
    // duplicates, but only after building the pipeline; catching it here avoids
    // the wasted load and keeps the engine's model set unperturbed on a
    // mistaken re-load (the common control-plane error).
    match hanzo.list_models() {
        Ok(ids) if ids.iter().any(|id| id == &cfg_name) => return -3,
        Ok(_) => {}
        Err(_) => return -2,
    }

    // alias == name makes `name` the routable id (parity with startup build).
    let config = ModelConfig::new(cfg_name.clone(), model).with_alias(cfg_name);

    // Drive the async load on a transient runtime, exactly as the startup build
    // does. The engine runs each model on its own thread, so this runtime is
    // disposable once `add_model` returns.
    let rt = match tokio::runtime::Runtime::new() {
        Ok(rt) => rt,
        Err(_) => return -3,
    };
    match rt.block_on(load_model_at_runtime(&hanzo, config)) {
        Ok(_) => 0,
        Err(_) => -3,
    }
}

/// Unload model `name` from the LIVE engine (`Hanzo::remove_model`). Returns 0 on
/// success. Errors: -1 bad args, -2 engine unavailable, -3 unload failed (model
/// not found, or it is the last remaining model — the engine refuses to go
/// empty). Synchronous; safe to call from a non-async (Go) thread.
///
/// # Safety
/// `(name, name_len)` must be a valid readable buffer (or len 0).
#[no_mangle]
pub unsafe extern "C" fn hanzo_ffi_unload(name: *const u8, name_len: usize) -> c_int {
    let name = model_str(name, name_len);
    if name.is_empty() {
        return -1;
    }
    if ensure_engine().is_err() {
        return -2;
    }
    let Some(hanzo) = hanzo_engine::engine_handle() else {
        return -2;
    };
    match hanzo.remove_model(&name) {
        Ok(()) => 0,
        Err(_) => -3,
    }
}

/// List the routable model ids of the LIVE engine, newline-joined, into a freshly
/// allocated UTF-8 buffer written to `*out` / `*out_len` (free with
/// [`hanzo_ffi_free`]). An empty engine yields an empty buffer (`*out_len = 0`,
/// `*out` may be null). Returns 0 on success. Errors: -1 bad args, -2 engine
/// unavailable, -3 list failed.
///
/// # Safety
/// `out`/`out_len` must be writable; the returned buffer is freed via
/// `hanzo_ffi_free(*out, *out_len)`.
#[no_mangle]
pub unsafe extern "C" fn hanzo_ffi_list(out: *mut *mut u8, out_len: *mut usize) -> c_int {
    if out.is_null() || out_len.is_null() {
        return -1;
    }
    if ensure_engine().is_err() {
        return -2;
    }
    let Some(hanzo) = hanzo_engine::engine_handle() else {
        return -2;
    };
    let ids = match hanzo.list_models() {
        Ok(ids) => ids,
        Err(_) => return -3,
    };
    let mut bytes = ids.join("\n").into_bytes();
    bytes.shrink_to_fit();
    let len = bytes.len();
    if len == 0 {
        *out = std::ptr::null_mut();
        *out_len = 0;
        return 0;
    }
    let ptr = bytes.as_mut_ptr();
    std::mem::forget(bytes);
    *out = ptr;
    *out_len = len;
    0
}

/// Release a buffer returned by [`hanzo_ffi_infer`].
///
/// # Safety
/// `ptr`/`len` must be exactly what `hanzo_ffi_infer` returned, once.
#[no_mangle]
pub unsafe extern "C" fn hanzo_ffi_free(ptr: *mut u8, len: usize) {
    if !ptr.is_null() && len != 0 {
        drop(Vec::from_raw_parts(ptr, len, len));
    }
}

/// Release an `f32` buffer returned by [`hanzo_ffi_embed`].
///
/// # Safety
/// `ptr`/`count` must be exactly what `hanzo_ffi_embed` returned, once.
#[no_mangle]
pub unsafe extern "C" fn hanzo_ffi_free_f32(ptr: *mut f32, count: usize) {
    if !ptr.is_null() && count != 0 {
        drop(Vec::from_raw_parts(ptr, count, count));
    }
}
