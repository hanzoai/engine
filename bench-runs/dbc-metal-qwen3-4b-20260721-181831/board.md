| model | backend | phase | n | conc | hanzo t/s | llama t/s | ratio | verdict |
|---|---|---|---|---|---|---|---|---|
| models | Metal | prefill | 2048 | 1 | 1221.4±48.9 | 1223.8±23.6 | 0.998±0.044 | PARITY |
| models | Metal | prefill | 512 | 1 | 1411.0±7.5 | 1628.5±22.2 | 0.866±0.013 | LOSS |
| models | Metal | decode | 128 | 1 | 110.7±0.3 | 136.7±0.8 | 0.809±0.005 | LOSS |
