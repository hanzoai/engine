# Router heads — the learned routing policy

`router.heads.json` is the serve bundle the engine mounts via
`ROUTER_HEADS=<path>` (loaded by `hanzo_router::Policy::load_heads` at startup;
unset = the rule-based `prefer()` policy). One `Policy` type either way.

Fit from eval events (per-prompt, per-arm outcomes with real prompt text):

    cargo run --release -p hanzo-router-retrain --bin router-fit -- \
      --events <events.jsonl> --out heads.safetensors --gamma 0.1

which writes the servable `heads.heads.json` next to `--out`. Current bundle:
2981 events (gpqa_diamond + livecodebench, 373 prompts x 8 arms), gamma 0.1
picked by held-out decision quality (mean realized correctness of the chosen
arm, prompt-level splits, 8 seeds): head 0.9215 +- 0.011 vs rule-based
0.9162 +- 0.007 vs best-single-arm 0.9215 (gpt-5.5) vs oracle 0.9402.
