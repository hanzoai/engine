# hanzo-film

Long-form film / episode orchestration over the Hanzo Engine `/v1` endpoints. A
one-paragraph brief becomes a validated Film Bible, thousands of independent
per-shot render jobs, dialogue + score, and a final assembled mp4 — with all
orchestration state as plain JSON files in a project directory (no database).

The engine is the model backend (LLM, image, TTS, embeddings, and the async
`/v1/videos` job). This crate is a pure client + ffmpeg assembler; it holds no
model weights and builds independently of the inference crates.

## Pipeline

```
plan     brief -> bible.json                 (LLM, structured output)
assets   reference image per character/loc   (image endpoint / procedural)
render   bible -> per-shot clips             (parallel, idempotent, continuity)
audio    dialogue TTS + score + per-shot mix (TTS endpoint / placeholder tone)
assemble concat + mux -> film.mp4 + timeline.json (EDL manifest)
verify   identity-coherence scoring hook
```

Each stage is a resumable CLI subcommand; `run` chains them. Every stage reads
the bible and writes into the project dir, so a re-run picks up where it left off.

## Backends (one interface, two impls)

Each modality is an enum: an engine-backed variant (the real target) and a
dependency-free variant that proves the whole pipeline end-to-end on a box with
no image/video/TTS model loaded. If an engine call fails, the pipeline logs a
warning and falls back per item.

| modality | engine variant            | fallback variant                  |
|----------|---------------------------|-----------------------------------|
| image    | `/v1/images/generations`  | procedural color card             |
| video    | `/v1/videos` (WAN, async) | placeholder: keyframe + Ken-Burns |
| speech   | `/v1/audio/speech`        | quiet tone sized to the line      |
| music    | `/v1/audio/music`*        | silence                           |
| coherence| `/v1/embeddings` (vision) | downsampled-luma cosine (proxy)   |

\* WAN video and ACE-Step music endpoints land from sibling work; the interface
is wired and used the moment they answer.

## The Bible schema (the contract)

`bible.json` is versioned (`BIBLE_VERSION`) and validated before any render job
runs (`Bible::validate`): every id unique, every ref (`location_ref`,
`character_ref`, `scene_ref`) resolvable, durations positive.

```jsonc
{
  "version": 1,
  "title": "...", "logline": "...",
  "style": { "prompt": "cinematic, warm light",
             "lora": null, "grade": "eq=contrast=1.05" },   // grade = ffmpeg filter
  "characters": [ { "id": "c1", "name": "Lila", "description": "...",
                    "reference_image": "assets/char_c1.png", "voice_id": "v1" } ],
  "locations":  [ { "id": "l1", "description": "...", "reference_image": "assets/loc_l1.png" } ],
  "scenes": [ {
    "id": "sc1", "location_ref": "l1", "characters": ["c1","c2"], "synopsis": "...",
    "shots": [ {
      "id": "sc1_sh1", "scene_ref": "sc1",
      "duration_s": 4.0, "shot_type": "wide",
      "action_prompt": "one vivid visual sentence",
      "dialogue": [ { "character_ref": "c1", "line": "..." } ],
      "continuity": "cut"          // "cut" | "continue"
    } ]
  } ]
}
```

Characters are global and referenced by every scene/shot, which is what makes a
character *recur* coherently.

### Continuity & parallelism

Shots are independent jobs. A `continue` shot conditions on the prior shot's
tail frame (same camera run); a `cut` starts fresh from its scene/location
anchor. The renderer partitions shots into **continuity runs** (a `cut` plus the
`continue` shots after it): runs are mutually independent and dispatched
concurrently up to `--concurrency`, while shots inside a run stay sequential.

### Idempotent re-runs

Each shot writes a spec-hash sidecar (`shots/<id>.json`) covering everything that
determines its pixels (prompt, geometry, style, conditioning, renderer + version).
A re-run skips any shot whose clip exists and whose hash still matches — change a
prompt and only that shot (and its dependents) re-render.

## Project directory

```
<proj>/
  project.json      brief + config
  bible.json        the validated contract
  assets/           char_<id>.png, loc_<id>.png
  shots/            <shot>.mp4, <shot>.json (spec-hash), <shot>.tail.png, <shot>.av.mp4
  audio/            line_<shot>_<n>.wav, music_<scene>.wav, mix_<shot>.wav
  timeline.json     EDL manifest (durable assembly artifact)
  coherence.json    identity-consistency scores
  film.mp4          final render
```

## Usage

```sh
# 1. serve a planner model (any chat model; structured output via response_format)
hanzo serve -p 1234 text -m Qwen/Qwen3-4B --format gguf -f model.gguf

# 2. orchestrate
hanzo-film new  ./myfilm --brief "A lighthouse keeper finds a message from her future self." \
                --engine http://127.0.0.1:1234 --width 768 --height 432 --fps 24 --concurrency 8
hanzo-film run  ./myfilm --scenes 2 --shots-per-scene 3     # plan..assemble..verify
# or step-by-step, each resumable:
hanzo-film plan ./myfilm && hanzo-film assets ./myfilm && hanzo-film render ./myfilm \
  && hanzo-film audio ./myfilm && hanzo-film assemble ./myfilm && hanzo-film verify ./myfilm

# real WAN video (once the endpoint lands) — same project, just the video backend:
hanzo-film new ./myfilm --brief "..." --video wan
```

## Tests

```sh
cargo test -p hanzo-film            # 32 unit (schema, JSON repair, hashing, runs, timeline)
                                    # + 1 offline integration: full pipeline -> mp4, ffprobed
```

The offline integration test renders a complete mp4 (video + audio) through the
procedural/placeholder backends, so `cargo test` proves the pipeline with no
engine and no GPU.
