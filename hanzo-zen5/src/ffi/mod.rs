//! FFI wrapper around the vendored `zen5-engine` C runtime (`ds4.h`).
//!
//! [`sys`] re-exports the raw bindgen output. The rest of the module is an
//! ergonomic, panic-safe Rust wrapper that satisfies the [`Zen5Engine`]
//! trait.
//!
//! ## Lifetimes & threading
//! `ds4_engine` is reference-counted on the C side and the public functions
//! take `*mut ds4_engine` (interior mutability under a mutex in the C
//! runtime). We hold the pointer in an `Arc` and mark [`Engine`]
//! `Send + Sync`. `ds4_session` is NOT thread-safe — each in-flight request
//! gets its own session inside [`Engine::complete`].
//!
//! ## Backends
//! The backend is chosen at build time (`metal` / `cuda` cargo features) and
//! surfaced at load time via [`EngineOptions::backend`]. A mismatch returns
//! `Zen5Error::Backend` instead of crashing.

pub mod sys;

use std::ffi::CString;
use std::path::Path;
use std::ptr;
use std::sync::Arc;

use async_trait::async_trait;

use crate::engine::{GenOpts, Token, TokenStream, Zen5Engine, Zen5Error};

/// Options for opening the C engine. Mirrors `ds4_engine_options` with safe
/// defaults appropriate for hanzod.
#[derive(Debug, Clone)]
pub struct EngineOptions {
    pub backend: Backend,
    pub n_threads: i32,
    pub mtp_draft_tokens: i32,
    pub mtp_margin: f32,
    pub warm_weights: bool,
    pub quality: bool,
}

impl Default for EngineOptions {
    fn default() -> Self {
        Self {
            backend: Backend::Auto,
            n_threads: 0, // 0 = let the runtime pick
            mtp_draft_tokens: 4,
            mtp_margin: 0.7,
            warm_weights: true,
            quality: false,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum Backend {
    Auto,
    Metal,
    Cuda,
    Cpu,
}

impl Backend {
    fn resolve(self) -> sys::ds4_backend {
        match self {
            Backend::Metal => sys::ds4_backend_DS4_BACKEND_METAL,
            Backend::Cuda => sys::ds4_backend_DS4_BACKEND_CUDA,
            Backend::Cpu => sys::ds4_backend_DS4_BACKEND_CPU,
            Backend::Auto => {
                if cfg!(target_os = "macos") {
                    sys::ds4_backend_DS4_BACKEND_METAL
                } else if cfg!(target_os = "linux") {
                    sys::ds4_backend_DS4_BACKEND_CUDA
                } else {
                    sys::ds4_backend_DS4_BACKEND_CPU
                }
            }
        }
    }

    fn name(b: sys::ds4_backend) -> &'static str {
        // ds4_backend is now a u32 typedef (bindgen + matching stub form).
        match b {
            sys::ds4_backend_DS4_BACKEND_METAL => "ffi/metal",
            sys::ds4_backend_DS4_BACKEND_CUDA => "ffi/cuda",
            sys::ds4_backend_DS4_BACKEND_CPU => "ffi/cpu",
            _ => "ffi/unknown",
        }
    }
}

/// RAII wrapper around `ds4_engine`.
pub struct Engine {
    inner: Arc<EngineInner>,
    backend_name: &'static str,
}

struct EngineInner {
    handle: *mut sys::ds4_engine,
}

// Safety: the C runtime guards engine-level state with an internal mutex.
unsafe impl Send for EngineInner {}
unsafe impl Sync for EngineInner {}

impl Drop for EngineInner {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe { sys::ds4_engine_close(self.handle) };
            self.handle = ptr::null_mut();
        }
    }
}

impl std::fmt::Debug for Engine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ffi::Engine")
            .field("backend", &self.backend_name)
            .finish()
    }
}

impl Engine {
    /// Load a GGUF model from disk.
    pub fn load(path: &Path, opts: EngineOptions) -> Result<Self, Zen5Error> {
        let path_c = CString::new(path.to_string_lossy().as_bytes())
            .map_err(|e| Zen5Error::Load(format!("invalid model path: {e}")))?;
        let backend = opts.backend.resolve();
        let backend_name = Backend::name(backend);

        let c_opts = sys::ds4_engine_options {
            model_path: path_c.as_ptr(),
            mtp_path: ptr::null(),
            backend,
            n_threads: opts.n_threads,
            mtp_draft_tokens: opts.mtp_draft_tokens,
            mtp_margin: opts.mtp_margin,
            directional_steering_file: ptr::null(),
            directional_steering_attn: 0.0,
            directional_steering_ffn: 0.0,
            warm_weights: opts.warm_weights,
            quality: opts.quality,
        };

        let mut handle: *mut sys::ds4_engine = ptr::null_mut();
        let rc = unsafe { sys::ds4_engine_open(&mut handle, &c_opts) };
        if rc != 0 || handle.is_null() {
            return Err(Zen5Error::Load(format!(
                "ds4_engine_open returned {rc} for {}",
                path.display()
            )));
        }

        Ok(Self {
            inner: Arc::new(EngineInner { handle }),
            backend_name,
        })
    }
}

#[async_trait]
impl Zen5Engine for Engine {
    fn backend(&self) -> &'static str {
        self.backend_name
    }

    async fn complete(&self, prompt: &str, opts: GenOpts) -> Result<TokenStream, Zen5Error> {
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel::<Result<Token, Zen5Error>>();
        let inner = Arc::clone(&self.inner);
        let prompt = prompt.to_owned();

        // Sessions don't cross threads in the C runtime; run on a blocking
        // thread and feed tokens back over the channel.
        tokio::task::spawn_blocking(move || {
            let mut tokens = match tokenize(&inner, &prompt) {
                Ok(t) => t,
                Err(e) => {
                    let _ = tx.send(Err(e));
                    return;
                }
            };

            // TODO(ffi): swap argmax for ds4_session_create + ds4_session_sample
            // so opts.{temperature, top_k, top_p, min_p, seed} are honored.
            // Argmax demonstrates the end-to-end token-emit path today.
            let _ = (opts.temperature, opts.top_k, opts.top_p, opts.min_p, opts.seed, opts.think);

            // bindgen typedefs ds4_token_emit_fn/ds4_generation_done_fn as
            // `Option<unsafe extern "C" fn(...)>` — declare these as
            // `unsafe extern "C" fn` so Some(emit)/Some(done) typecheck.
            unsafe extern "C" fn emit(ud: *mut std::ffi::c_void, tok: std::os::raw::c_int) {
                // Safety: ud is the raw pointer of a Box we leaked below.
                let tx = unsafe {
                    &*(ud as *const tokio::sync::mpsc::UnboundedSender<Result<Token, Zen5Error>>)
                };
                let _ = tx.send(Ok(Token {
                    id: tok as i32,
                    text: String::new(), // TODO: decode via ds4_token_text
                    logprob: 0.0,
                }));
            }
            unsafe extern "C" fn done(_ud: *mut std::ffi::c_void) {}

            let tx_box = Box::new(tx.clone());
            let tx_ptr = Box::into_raw(tx_box) as *mut std::ffi::c_void;
            let rc = unsafe {
                sys::ds4_engine_generate_argmax(
                    inner.handle,
                    &tokens,
                    opts.max_tokens as i32,
                    0, // ctx_size = 0 → engine default
                    Some(emit),
                    Some(done),
                    tx_ptr,
                    None,            // progress: ds4_session_progress_fn (Option<fn>)
                    ptr::null_mut(), // progress_ud: *mut c_void
                )
            };
            // Reclaim the sender we leaked across the FFI boundary.
            unsafe {
                drop(Box::from_raw(
                    tx_ptr as *mut tokio::sync::mpsc::UnboundedSender<Result<Token, Zen5Error>>,
                ));
            }
            unsafe { sys::ds4_tokens_free(&mut tokens) };
            if rc != 0 {
                let _ = tx.send(Err(Zen5Error::Inference(format!(
                    "ds4_engine_generate_argmax returned {rc}"
                ))));
            }
        });

        // Wrap the channel receiver in a stream without pulling in tokio-stream
        // as a dependency — futures::stream::unfold is enough.
        let stream = futures::stream::unfold(rx, |mut rx| async move {
            rx.recv().await.map(|item| (item, rx))
        });
        Ok(Box::pin(stream))
    }

    async fn embed(&self, _text: &str) -> Result<Vec<f32>, Zen5Error> {
        // TODO(ffi): zen5-engine doesn't expose hidden states yet. Proposal:
        // add `ds4_session_hidden_state(s, layer)` upstream.
        Err(Zen5Error::Backend(
            "embed() not yet supported by zen5-engine; use a dedicated embedding model".into(),
        ))
    }
}

fn tokenize(inner: &EngineInner, text: &str) -> Result<sys::ds4_tokens, Zen5Error> {
    let c = CString::new(text)
        .map_err(|e| Zen5Error::Inference(format!("prompt contains NUL: {e}")))?;
    let mut out = sys::ds4_tokens {
        v: ptr::null_mut(),
        len: 0,
        cap: 0,
    };
    unsafe { sys::ds4_tokenize_text(inner.handle, c.as_ptr(), &mut out) };
    Ok(out)
}
