/* C ABI over the Hanzo native inference engine — text generation + embeddings,
 * routed by model name through the engine's native multi-model support.
 * See ../src/lib.rs. */
#ifndef HANZO_ENGINE_FFI_H
#define HANZO_ENGINE_FFI_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Load all configured models (HANZO_FFI_MODELS) if needed. 1 = ready, 0 = fail. */
int32_t hanzo_ffi_ready(void);

/* Text generation on model `model` (NULL/empty = default). On success (0) writes
 * a heap UTF-8 buffer to *out / *out_len; release with hanzo_ffi_free.
 * Errors: -1 bad args, -2 engine unavailable, -3 inference failed. */
int32_t hanzo_ffi_infer(const uint8_t *model, size_t model_len,
                        const uint8_t *prompt, size_t prompt_len,
                        uint8_t **out, size_t *out_len);

/* Embedding on model `model` (NULL/empty = default). On success (0) writes a
 * heap float buffer of the model's native dimension to *out / *out_count;
 * release with hanzo_ffi_free_f32. Errors as above. */
int32_t hanzo_ffi_embed(const uint8_t *model, size_t model_len,
                        const uint8_t *text, size_t text_len,
                        float **out, size_t *out_count);

/* Load one model into the LIVE engine at runtime, routable immediately by `name`.
 * `kind` in {"gguf","plain","embedding"}; `source` is the same value the startup
 * HANZO_FFI_MODELS spec uses (abs .gguf path, or HF repo / local dir). Engine
 * must already be up (hanzo_ffi_ready). Returns 0 on success.
 * Errors: -1 bad args, -2 engine unavailable, -3 load failed (bad spec, missing
 * weights, or name/alias conflict). */
int32_t hanzo_ffi_load(const uint8_t *name, size_t name_len,
                       const uint8_t *kind, size_t kind_len,
                       const uint8_t *source, size_t source_len);

/* Unload model `name` from the LIVE engine. Returns 0 on success.
 * Errors: -1 bad args, -2 engine unavailable, -3 unload failed (not found, or it
 * is the last remaining model — the engine refuses to go empty). */
int32_t hanzo_ffi_unload(const uint8_t *name, size_t name_len);

/* List the LIVE engine's routable model ids, newline-joined, into a fresh heap
 * UTF-8 buffer written to *out / *out_len (release with hanzo_ffi_free). An empty
 * engine yields *out_len = 0 (and *out may be NULL). Returns 0 on success.
 * Errors: -1 bad args, -2 engine unavailable, -3 list failed. */
int32_t hanzo_ffi_list(uint8_t **out, size_t *out_len);

/* Release a buffer from hanzo_ffi_infer or hanzo_ffi_list. */
void hanzo_ffi_free(uint8_t *ptr, size_t len);

/* Release a buffer from hanzo_ffi_embed. */
void hanzo_ffi_free_f32(float *ptr, size_t count);

#ifdef __cplusplus
}
#endif

#endif /* HANZO_ENGINE_FFI_H */
