| model | backend | phase | n | conc | hanzo t/s | llama t/s | ratio | verdict |
|---|---|---|---|---|---|---|---|---|
| models | Metal | prefill | 2048 | 1 | 3301.5±4.3 | 3696.9±111.4 | 0.893±0.027 | LOSS |
| models | Metal | prefill | 4096 | 1 | 2767.3±113.6 | 2863.0±69.4 | 0.967±0.046 | PARITY |
| models | Metal | prefill | 500 | 1 | 3345.3±9.4 | 3988.9±14.8 | 0.839±0.004 | LOSS |
| models | Metal | prefill | 512 | 1 | 3386.0±11.6 | 4097.5±39.3 | 0.826±0.008 | LOSS |
| models | Metal | decode | 128 | 1 | 233.4±1.5 | 253.8±1.0 | 0.920±0.007 | LOSS |
