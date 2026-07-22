| model | backend | phase | n | conc | hanzo t/s | llama t/s | ratio | verdict |
|---|---|---|---|---|---|---|---|---|
| qwen3-1.7b | ROCm | prefill | 2048 | 1 | 3506.1±47.0 | 4551.3±20.1 | 0.770±0.011 | LOSS |
| qwen3-1.7b | ROCm | prefill | 4096 | 1 | 2840.8±93.0 | 4103.3±16.0 | 0.692±0.023 | LOSS |
| qwen3-1.7b | ROCm | prefill | 500 | 1 | 3603.8±19.4 | 4678.0±186.3 | 0.770±0.031 | LOSS |
| qwen3-1.7b | ROCm | prefill | 512 | 1 | 3693.6±14.0 | 4742.5±214.0 | 0.779±0.035 | LOSS |
| qwen3-1.7b | ROCm | decode | 128 | 1 | 113.7±1.1 | 133.8±0.7 | 0.850±0.009 | LOSS |
