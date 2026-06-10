# MuseTalk committed reference fixtures

PyTorch reference tensors for the real-weight MuseTalk stage of the `dub_e2e` integration test
(`hanzo-engine/tests/dub_e2e.rs`: `musetalk_render_real`).

They were dumped ONCE, OFFLINE, by running the real PyTorch MuseTalk 1.5 pipeline
(`unet.pth` + `sd-vae-ft-mse`) on a fixed reference face crop + fixed post-PositionalEncoding
whisper audio feature, then committed here. **No python runs at `cargo test` time** -- the test
loads these committed bytes, runs the engine's `MuseTalk` (built over the converted real
`unet.safetensors` + `vae.safetensors`) on the SAME per-stage inputs, and compares per-stage
cosine. This is the same numerical contract the `musetalk-bench realverify` validates at
cosine ~1.0 on CUDA f16.

Layout (NumPy `.npy`, f32):

| file              | shape            | role   | stage                                        |
|-------------------|------------------|--------|----------------------------------------------|
| `face_crop.npy`   | (1, 3, 256, 256) | input  | normalized (mean=std=0.5) reference face     |
| `masked_img.npy`  | (1, 3, 256, 256) | input  | normalized lower-half-masked face            |
| `audio_feat.npy`  | (1, 50, 384)     | input  | post-PositionalEncoding whisper feature      |
| `enc_mode.npy`    | (1, 4, 32, 32)   | target | VAE-encode mode of `face_crop`               |
| `unet_in.npy`     | (1, 8, 32, 32)   | target | cat([VAE-encode(masked), VAE-encode(face)])  |
| `unet_pred.npy`   | (1, 4, 32, 32)   | target | UNet single-step pred on `unet_in`+`audio`   |
| `vae_dec.npy`     | (1, 3, 256, 256) | target | raw VAE-decode of `unet_pred` (pre-denorm)   |

## CPU vs CUDA

CUDA is the gated target: `cargo test --features cuda dub_e2e` asserts every stage cosine
>= 0.999 vs these references. On CPU the test still loads the real weights and runs the full
graph (asserting finite / correct-shape), but does NOT assert the cosines: hanzo-ml's CPU
conv/groupnorm path diverges from the PyTorch f32 reference even for the validated bench code
(measured VAE-encode cosine ~0.896 on CPU), and the CPU VAE-decode `up_blocks` path has an
additional known caveat. These are CPU-backend matters, not MuseTalk-wiring issues.

## Re-blessing (offline only)

If a deliberate, understood change to the MuseTalk math lands, regenerate these and re-copy in
the same commit. The dump scripts live in the ml repo (`zen-dub-run/dump_ref.py` and friends)
and are NEVER invoked by `cargo test`. The ~3.7 GB real weights are NOT committed; the test is
env-gated on `MUSETALK_WDIR` (default: the spark layout) and falls back to a random-init
graph/shape check when the weights are absent.
