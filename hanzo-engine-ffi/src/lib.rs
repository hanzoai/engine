//! C ABI over the Hanzo native inference engine — full parity with the Rust VM.
//!
//! Exposes both text generation (`hanzo_ffi_infer`) and embeddings
//! (`hanzo_ffi_embed`), routed by model name through the engine's NATIVE
//! multi-model support, so non-Rust callers (the Go EVM AI precompile via cgo)
//! can use ANY loaded zen / zen-embedding model in-process. The engine is built
//! once (lazy, from env) holding every configured model.
//!
//! Config (read on first call): `ENGINE_FFI_MODELS` (or `FFI_MODELS`) = `;`-separated list of
//! `name=kind:source`, where `kind` in {`gguf`,`plain`,`embedding`}:
//!   - `gguf:/abs/path/model.gguf`  (causal GGUF; tokenizer from the file's dir
//!      or `ENGINE_FFI_TOK_DIR` if set)
//!   - `plain:<hf-repo-or-dir>`     (causal safetensors)
//!   - `embedding:<hf-repo-or-dir>` (embedding model; arch auto-detected)
//! The first entry is the default model. Example:
//!   `ENGINE_FFI_MODELS="zen-nano=gguf:/tmp/zen5-weights/zen-5-flash.gguf;zen-embed=embedding:/tmp/zen-embedding-0.6B"`

#![allow(clippy::doc_lazy_continuation, clippy::doc_overindented_list_items)]

use std::ffi::{c_int, c_void};
use std::sync::OnceLock;

use either::Either;
use hanzo_engine::{
    MessageContent, NormalRequest, Request, RequestMessage, Response, SamplingParams,
};
use hanzo_server_core::server::{load_model_at_runtime, ModelConfig, ServerBuilder};
use indexmap::IndexMap;

/// Why the last call failed, for [`hanzo_ffi_error`].
static LAST_ERROR: std::sync::Mutex<String> = std::sync::Mutex::new(String::new());

fn fail<T>(e: String) -> Result<T, String> {
    *LAST_ERROR.lock().unwrap_or_else(|p| p.into_inner()) = e.clone();
    Err(e)
}

/// Lazy one-time multi-model engine build. `Ok(())` once registered.
fn ensure_engine() -> Result<(), String> {
    static INIT: OnceLock<Result<(), String>> = OnceLock::new();
    INIT.get_or_init(|| {
        if hanzo_engine::inference_engine_registered() {
            return Ok(());
        }
        let spec = hanzo_engine::engine_env_var("ENGINE_FFI_MODELS", "FFI_MODELS")
            .or_else(|_| std::env::var("HANZO_FFI_MODELS"))
            .map_err(|_| "set ENGINE_FFI_MODELS (name=kind:source;...)".to_string())?;
        let tok_dir = hanzo_engine::engine_env_var("ENGINE_FFI_TOK_DIR", "FFI_TOK_DIR")
            .or_else(|_| std::env::var("HANZO_FFI_TOK_DIR"))
            .ok();
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
            .map_err(|e| format!("engine build failed: {e:#}"));
        let hanzo = match hanzo {
            Ok(h) => h,
            Err(e) => return fail(e),
        };
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
    let tok_dir = hanzo_engine::engine_env_var("ENGINE_FFI_TOK_DIR", "FFI_TOK_DIR")
        .or_else(|_| std::env::var("HANZO_FFI_TOK_DIR"))
        .ok();
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

/// Called once per generated text piece. Return non-zero to stop generating.
pub type TokenCallback = extern "C" fn(text: *const u8, len: usize, ctx: *mut c_void) -> c_int;

/// Chat on model `model` (NUL/empty = default), streaming. `messages` is a JSON
/// array of `{"role": "...", "content": "..."}`, rendered with the model's own
/// chat template. `on_token` receives each text piece as it is generated; at
/// most `max_tokens` are produced (0 = 1024). On return `*usage` / `*usage_len`
/// hold a JSON object `{prompt_tokens, completion_tokens, prompt_tok_per_sec,
/// tok_per_sec, seconds}` (free with [`hanzo_ffi_free`]; null if the run was
/// cut before the engine reported it).
///
/// Returns 0 when generation ended, by the model, by `max_tokens` or by the
/// callback. Errors: -1 bad args, -2 engine unavailable, -3 generation failed.
///
/// # Safety
/// Pointers must be valid for their lengths; `usage`/`usage_len` writable or
/// null; `ctx` is passed to `on_token` untouched. Blocks the calling thread.
#[no_mangle]
pub unsafe extern "C" fn hanzo_ffi_chat(
    model: *const u8,
    model_len: usize,
    messages: *const u8,
    messages_len: usize,
    max_tokens: u32,
    on_token: Option<TokenCallback>,
    ctx: *mut c_void,
    usage: *mut *mut u8,
    usage_len: *mut usize,
) -> c_int {
    let Some(on_token) = on_token else { return -1 };
    if messages.is_null() {
        return -1;
    }
    let Ok(turns) = serde_json::from_slice::<Vec<serde_json::Value>>(std::slice::from_raw_parts(
        messages,
        messages_len,
    )) else {
        return -1;
    };
    let mut chat: Vec<IndexMap<String, MessageContent>> = Vec::with_capacity(turns.len());
    for t in &turns {
        let (Some(role), Some(content)) = (
            t.get("role").and_then(|v| v.as_str()),
            t.get("content").and_then(|v| v.as_str()),
        ) else {
            return -1;
        };
        let mut m: IndexMap<String, MessageContent> = IndexMap::new();
        m.insert("role".to_string(), Either::Left(role.to_string()));
        m.insert("content".to_string(), Either::Left(content.to_string()));
        chat.push(m);
    }
    if chat.is_empty() {
        return -1;
    }
    if ensure_engine().is_err() {
        return -2;
    }
    let Some(hanzo) = hanzo_engine::engine_handle() else {
        return -2;
    };
    let name = model_str(model, model_len);
    let model_id = if name.is_empty() || name == "default" {
        None
    } else {
        Some(name)
    };

    let mut sampling = SamplingParams::deterministic();
    sampling.temperature = Some(0.7);
    sampling.top_k = Some(40);
    sampling.top_p = Some(0.9);
    sampling.max_len = Some(if max_tokens == 0 {
        1024
    } else {
        max_tokens as usize
    });
    let request = RequestMessage::Chat {
        messages: chat,
        enable_thinking: Some(false),
        reasoning_effort: None,
        preserve_thinking: None,
    };
    let (tx, mut rx) = tokio::sync::mpsc::channel(64);
    let mut req = NormalRequest::new_simple(request, sampling, tx, 0, None, None);
    req.is_streaming = true;
    req.model_id = model_id;
    if hanzo.send_request(Request::Normal(Box::new(req))).is_err() {
        return -2;
    }

    let mut report: Option<String> = None;
    loop {
        match rx.blocking_recv() {
            Some(Response::Chunk(chunk)) => {
                if let Some(text) = chunk
                    .choices
                    .first()
                    .and_then(|c| c.delta.content.as_deref())
                {
                    if !text.is_empty() && on_token(text.as_ptr(), text.len(), ctx) != 0 {
                        // Dropping the receiver ends the request on the engine side.
                        break;
                    }
                }
                if let Some(u) = chunk.usage {
                    report = Some(
                        serde_json::json!({
                            "prompt_tokens": u.prompt_tokens,
                            "completion_tokens": u.completion_tokens,
                            "prompt_tok_per_sec": u.avg_prompt_tok_per_sec,
                            "tok_per_sec": u.avg_compl_tok_per_sec,
                            "seconds": u.total_time_sec,
                        })
                        .to_string(),
                    );
                }
                if chunk
                    .choices
                    .first()
                    .is_some_and(|c| c.finish_reason.is_some())
                    && report.is_some()
                {
                    break;
                }
            }
            Some(Response::Done(_)) | None => break,
            Some(Response::ModelError(msg, _)) => {
                let _: Result<(), String> = fail(msg);
                return -3;
            }
            Some(Response::ValidationError(e)) | Some(Response::InternalError(e)) => {
                let _: Result<(), String> = fail(e.to_string());
                return -3;
            }
            Some(_) => {
                let _: Result<(), String> = fail("unexpected engine response".to_string());
                return -3;
            }
        }
    }
    if !usage.is_null() && !usage_len.is_null() {
        match report {
            Some(r) => {
                let mut bytes = r.into_bytes();
                bytes.shrink_to_fit();
                *usage_len = bytes.len();
                *usage = bytes.as_mut_ptr();
                std::mem::forget(bytes);
            }
            None => {
                *usage = std::ptr::null_mut();
                *usage_len = 0;
            }
        }
    }
    0
}

/// Why the last failing call failed, as UTF-8 text in a fresh buffer written to
/// `*out` / `*out_len` (free with [`hanzo_ffi_free`]); empty if nothing has failed.
///
/// # Safety
/// `out`/`out_len` must be writable.
#[no_mangle]
pub unsafe extern "C" fn hanzo_ffi_error(out: *mut *mut u8, out_len: *mut usize) -> c_int {
    if out.is_null() || out_len.is_null() {
        return -1;
    }
    let mut bytes = LAST_ERROR
        .lock()
        .unwrap_or_else(|p| p.into_inner())
        .clone()
        .into_bytes();
    bytes.shrink_to_fit();
    *out_len = bytes.len();
    if bytes.is_empty() {
        *out = std::ptr::null_mut();
    } else {
        *out = bytes.as_mut_ptr();
        std::mem::forget(bytes);
    }
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
