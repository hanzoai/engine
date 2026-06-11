# zen-3-tts committed reference fixtures

These are PyTorch reference tensors for the engine's `codec_validation` cargo tests
(`speech_models/qwen3_tts/mod.rs`: `codec_matches_reference`, `talker_matches_reference`,
`prefill_matches_reference`, `full_generation_greedy_matches_reference`).

They were generated ONCE, OFFLINE, by running the real `QwenLM/Qwen3-TTS` PyTorch model on a
fixed prompt + fixed seed with a fully greedy/deterministic decode, then committed here. **No
python runs at `cargo test` time** -- the tests load these committed bytes and compare (cosine
> 0.99 / 0.999). This is the golden-fixture replacement for live PyTorch dumps.

Layout (raw little-endian; dims in `meta.txt`):

| file                 | dtype | shape                | stage                                   |
|----------------------|-------|----------------------|-----------------------------------------|
| `input_ids.i64`      | i64   | (T_ids,)             | tokenized prompt ids                    |
| `tk_prefill.f32`     | f32   | (1, 19, 1024)        | talker prefill inputs_embeds            |
| `tk_hidden.f32`      | f32   | (1, 19, 1024)        | talker last_hidden_state (post-norm)    |
| `tk_logits.f32`      | f32   | (1, 19, 3072)        | talker codec_head logits                |
| `tk_frame0_codes.i64`| i64   | (16,)                | greedy frame-0 codes (all 16 groups)    |
| `greedy_TQ.i64`      | i64   | (48, 16)             | greedy code grid, codebook-0 == col 0   |
| `codes_QT.i64`       | i64   | (16, 48)             | same codes, transposed for the codec    |
| `ref_quant.f32`      | f32   | (1, 512, 48)         | SplitRVQ.decode output                  |
| `ref_pretrans.f32`   | f32   | (1, 48, 1024)        | pre_transformer output                  |
| `ref_upsample.f32`   | f32   | (1, 1024, 192)       | post-upsample                           |
| `ref_wav.f32`        | f32   | (1, 1, 92160)        | final clamped 24 kHz waveform           |

## Re-blessing (offline only -- separated from the test path)

If a deliberate, understood change to the TTS math lands, regenerate these and re-copy them in
the same commit. The re-gen scripts live in the ml repo and are **never invoked by `cargo test`**:

    # ml repo: native-dub/reference/
    ZEN3_REF_OUT=/tmp/tts-ref gen_tts_ref.sh        # provisions transformers==4.57.3, runs dump_tts_ref.py
    cp /tmp/tts-ref/{*.f32,*.i64,meta.txt} <engine>/hanzo-engine/tests/fixtures/zen3_tts/

The `ZEN3_REF_*` env vars still override the fixture paths if you want the tests to read a fresh
dump directly (used while re-blessing). With no env set, the committed fixtures here are used.

The 1.8 GB model weights are NOT committed; the tests are env-gated on `ZEN3_TTS_DIR` (default:
the spark layout) and cleanly skip when the weights are absent.
