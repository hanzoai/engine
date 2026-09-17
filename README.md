# Benchmark ledger

Every run of the engine against llama.cpp that we stand behind, as measured. This branch holds
data only. It is append-only: a run is added, never edited, and a number that turns out wrong is
answered by a newer run beside it.

## A run

`bench-runs/<host>-<backend>-<model>-<YYYYMMDD-HHMMSS>/`, written by `scripts/dossier.sh` on `main`:

| file | what it is |
|---|---|
| `manifest.json` | the engine commit and whether its tree was clean, the llama.cpp commit, both binaries' SHA-256, the model's path, size and SHA-256, the host, GPU, shapes and repetitions |
| `quiet-gate.txt` | what else was on the GPU when the run started. The harness refuses to start beside another GPU job |
| `hanzo_<tag>.json` | our raw per-repetition `[wall_s, tokens]` |
| `llama_<tag>.json` | `llama-bench -o json`, unmodified |
| `*.log`, `*.err` | both engines' full output |
| `board.json`, `board.md` | the scores, from `scripts/bench_stats.py` |

Both engines load the same GGUF, run the same shapes, and are timed by wall clock.

## The scores

Every published figure is a function of the raw samples in the run directory and nothing else:
our first timed repetition is discarded when three or more were taken, because llama-bench warms
each shape before it times and ours pays pipeline compilation on the first; the interval is
Student-t, two-sided, 95%; the
ratio is ours over theirs, its interval the two relative intervals added in quadrature. `WIN`
means the whole interval lies above 1.00, `LOSS` that it lies below, `PARITY` that it spans it.
A coefficient of variation above 5% in either engine marks the cell `noisy`: the box was not
quiet, and the verdict is not settled.

## Reading all of it

`scripts/bench_ledger.py --runs bench-runs` on `main` emits every row of every run as one
document. hanzo.ai/benchmarks/inference renders that document and cites the commit of this
branch it was generated from.
