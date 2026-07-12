# Fast-local replica fabric (Qwen3-30B-A3B) — runbook

3 independent full replicas of `Qwen3-30B-A3B-Instruct-2507-Q4_K_M` (18.5GB, 3B active MoE),
one per box, load-balanced behind `hanzo-router`. Goal: N parallel agents, each at full GPU speed.

## Front endpoint (what the user points a client at)
    export ANTHROPIC_BASE_URL=http://10.0.0.144:1234   # evo, hanzo-router
    export ANTHROPIC_API_KEY=local                      # engine ignores auth (dev)
    hanzo code
OpenAI clients: `http://10.0.0.144:1234/v1`. Both `/v1/messages` (Anthropic) and
`/v1/chat/completions` (OpenAI) proxy through, SSE streamed byte-for-byte.

## Model-id gotcha (why the router rewrites)
Each engine serves ONE model and strict-matches its served id or `default`; ANY other
model id -> HTTP 500. Claude Code sends `claude-*` ids, so the router pins
`--upstream-model default` on every forwarded request (routing/affinity still key on the
client's original model). Code: hanzo-router 0.1.1, branch feat/router-upstream-model.
Build: `cd ~/work/hanzo/engine && PATH=/usr/bin:$HOME/.cargo/bin:$PATH CC=/usr/bin/gcc \
cargo build --release -p hanzo-router --features proxy` (the `~/.local/bin/cc` Claude shim
shadows gcc and breaks the link with `unknown option -m64` — force /usr/bin first).

## Per-box serve commands (native `hanzo serve`, all bind :8080, model id = `default`)
GGUF paths differ per box; config+tokenizer live in a local `qwen3-30b-meta/` dir
(config.json, generation_config.json, tokenizer.json, tokenizer_config.json + a symlink to
the gguf) so `-m <dir>` loads locally with NO HuggingFace fetch (HF 404s offline).

spark (CUDA GB10), on spark:
    hanzo serve -m Qwen/Qwen3-30B-A3B-Instruct-2507 --format gguf \
      -f ~/models/Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf -p 8080 --host 0.0.0.0 -n 0:48
    # (spark has the HF repo cached, so it uses the repo id directly)

evo (ROCm Strix Halo gfx1151), on evo — ONE ROCm proc only (concurrent kfd init wedges):
    LD_LIBRARY_PATH=/opt/rocm/lib:/opt/rocm/core-7.13/lib \
    ~/work/hanzo/engine-ring/target/release/hanzo serve \
      -m ~/models/qwen3-30b-meta --format gguf -f Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf \
      -p 8080 --host 127.0.0.1 -n 0:48 --pa-context-len 131072
    # -n 0:48 forces all 48 layers onto rocm[0] (auto-mapper misreads APU unified mem -> CPU offload = 4 tok/s trap)
    # --pa-context-len 131072 caps KV to ~24GB (default grabs 81GB -> only 7GB free)
    # engine-ring = hanzo 1.7.28 (ROCm-linked); ldd shows libamdhip. engine-rocm is 1.7.21.

    # KNOWN evo-ONLY bug (ROCm/gfx1151, NOT config): the last prefill position's logits
    # are slightly off, so the FIRST generated token's argmax flips on *borderline* prompts
    # (e.g. "reply with only: pong" -> "system\npong"). Realistic agent/code prompts are
    # clean (measured 7/8; only the contrived edge case leaked). Present in 1.7.21 AND 1.7.28,
    # with Q4K_FALLBACK=1 and with --paged-attn off -> it's the ROCm forward pass, not a knob.
    # Signature == the CUDA MoE topk "last-row collapse" bug fixed in ml 0.11.17; the HIP topk
    # path needs the same fix. FIX PATH: port that fix to the ROCm MoE routing kernel + rebuild
    # engine-rocm. Until then evo is a fast (60 tok/s) replica that occasionally glitches token 1.
    # To exclude evo: drop its --replica from the router (spark+dbc alone = 2 clean replicas).

dbc (Metal M4 Max), on dbc:
    ~/work/hanzo/engine-metal/target/release/hanzo serve \
      -m /Users/a/models/qwen3-30b-meta --format gguf -f Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf \
      -p 8080 --host 0.0.0.0 -n 0:48 --paged-attn on --pa-context-len 131072
    # paged-attn is auto-OFF on Metal; force `on` so KV is bounded + block-shared

## Router (on evo)
    ~/work/hanzo/engine/target/release/hanzo-router --host 0.0.0.0 --port 1234 \
      --model qwen3-30b-a3b --upstream-model default \
      --replica http://192.168.77.2:8080   `# spark, direct 2.5GbE link` \
      --replica http://127.0.0.1:8080      `# evo local` \
      --replica http://10.0.0.132:8080     `# dbc over LAN`
Admin: `GET /v1/replicas` (health+inflight), `POST /v1/replicas` (add), `GET /health`.
Prefix-affinity (conversation sticks to one replica) + least-loaded spill + /health eviction/restore.

## Measured (Qwen3-30B-A3B Q4_K_M, 256-tok decode)
single-box: spark(CUDA) 48 tok/s, evo(ROCm) 60, dbc(Metal) 72.
Through router single-stream: 58-63 tok/s. 3 concurrent agents (1 per box): ~186 tok/s
aggregate, each 62-68 tok/s = the parallel win (3x). Beyond 3, agents double up per box and
per-agent decode drops (MoE: distinct experts per seq -> more weight reads/step; evo/APU worst
because it also hosts the router/console and shares memory bandwidth). Spread across boxes is
the design; the router does exactly this. Sweet spot = <=3 concurrent (one full-speed agent/GPU).

## Net topology (from evo, where the router runs)
evo 10.0.0.144 / 192.168.77.1(->spark) ; spark 10.0.0.242 / 192.168.77.2 ; dbc 10.0.0.132.

## Live PIDs (this session): spark serve 3895887 | evo serve 2920376 | dbc serve 21351 | router 2883009
All nohup+disown (survive session end). memguard.sh is a dead one-shot (already tripped);
current config is memory-bounded, no guard needed. To kill a serve: `kill -TERM <pid>` by exact
PID -- NEVER `pkill -f "release/hanzo serve"` (the pattern self-matches your own shell -> kills it).
