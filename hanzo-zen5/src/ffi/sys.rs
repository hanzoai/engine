//! Raw FFI bindings for `ds4.h`.
//!
//! When the vendored `zen5-engine-src/ds4.h` is present at build time,
//! `build.rs` runs bindgen and writes `$OUT_DIR/bindings.rs`. We `include!`
//! it here. When the submodule is missing the build script prints a warning
//! and skips bindgen; in that case this file falls back to a minimal
//! hand-written prototype set so the wrapper still compiles. Calls into the
//! stubs become link errors at final link time, which is the desired
//! behavior — opting into the FFI feature without the submodule is a config
//! error worth surfacing loudly.
//!
//! All identifiers match the C header exactly; we mark them `non_*_case`
//! to silence Rust's naming lints.

#![allow(
    non_camel_case_types,
    non_snake_case,
    non_upper_case_globals,
    dead_code,
    improper_ctypes,
    unexpected_cfgs,
    missing_debug_implementations
)]

// Pulled in from bindgen output when the build script runs successfully.
// The `cfg(zen5_bindings)` flag is set by build.rs after a successful run.
// Wrapped in an inner module so bindgen's `#![allow(...)]` inner attrs apply
// to the module rather than the enclosing file (which would be a hard error).
#[cfg(zen5_bindings)]
#[allow(non_upper_case_globals, non_camel_case_types, non_snake_case, dead_code,
        improper_ctypes, unexpected_cfgs, missing_debug_implementations)]
mod gen {
    include!(concat!(env!("OUT_DIR"), "/bindings.rs"));
}
#[cfg(zen5_bindings)]
pub use gen::*;

// Stub fallback used when the submodule is not vendored. Keeps the wrapper
// compiling under `cargo check`; final link will fail iff someone actually
// calls into these stubs without the C runtime present.
#[cfg(not(zen5_bindings))]
mod stub {
    use std::os::raw::{c_char, c_int, c_void};

    #[repr(C)]
    #[derive(Debug)]
    pub struct ds4_engine {
        _private: [u8; 0],
    }
    #[repr(C)]
    #[derive(Debug)]
    pub struct ds4_session {
        _private: [u8; 0],
    }

    // Match bindgen's representation: typedef'd as u32 with top-level consts.
    // Keeping the same surface lets call sites use `sys::ds4_backend_DS4_BACKEND_X`
    // and `sys::ds4_backend` regardless of whether the real C lib is linked or
    // we're falling back to stubs.
    pub type ds4_backend = ::std::os::raw::c_uint;
    pub const ds4_backend_DS4_BACKEND_METAL: ds4_backend = 0;
    pub const ds4_backend_DS4_BACKEND_CUDA: ds4_backend = 1;
    pub const ds4_backend_DS4_BACKEND_CPU: ds4_backend = 2;

    #[repr(C)]
    #[derive(Debug)]
    pub struct ds4_tokens {
        pub v: *mut c_int,
        pub len: c_int,
        pub cap: c_int,
    }

    #[repr(C)]
    #[derive(Debug)]
    pub struct ds4_engine_options {
        pub model_path: *const c_char,
        pub mtp_path: *const c_char,
        pub backend: ds4_backend,
        pub n_threads: c_int,
        pub mtp_draft_tokens: c_int,
        pub mtp_margin: f32,
        pub directional_steering_file: *const c_char,
        pub directional_steering_attn: f32,
        pub directional_steering_ffn: f32,
        pub warm_weights: bool,
        pub quality: bool,
    }

    pub type ds4_token_emit_fn = Option<unsafe extern "C" fn(ud: *mut c_void, token: c_int)>;
    pub type ds4_generation_done_fn = Option<unsafe extern "C" fn(ud: *mut c_void)>;

    extern "C" {
        pub fn ds4_engine_open(out: *mut *mut ds4_engine, opt: *const ds4_engine_options)
            -> c_int;
        pub fn ds4_engine_close(e: *mut ds4_engine);
        pub fn ds4_engine_generate_argmax(
            e: *mut ds4_engine,
            prompt: *const ds4_tokens,
            n_predict: c_int,
            ctx_size: c_int,
            emit: ds4_token_emit_fn,
            done: ds4_generation_done_fn,
            emit_ud: *mut c_void,
            progress: *mut c_void,
            progress_ud: *mut c_void,
        ) -> c_int;
        pub fn ds4_tokenize_text(
            e: *mut ds4_engine,
            text: *const c_char,
            out: *mut ds4_tokens,
        );
        pub fn ds4_tokens_free(tv: *mut ds4_tokens);
        pub fn ds4_token_text(
            e: *mut ds4_engine,
            token: c_int,
            len: *mut usize,
        ) -> *mut c_char;
        pub fn ds4_token_eos(e: *mut ds4_engine) -> c_int;
    }
}

#[cfg(not(zen5_bindings))]
pub use stub::*;
