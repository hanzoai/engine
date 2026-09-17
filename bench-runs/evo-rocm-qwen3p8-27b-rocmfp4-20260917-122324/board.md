| model | backend | phase | n | conc | hanzo t/s | llama t/s | ratio | verdict | best hanzo | best llama | best ratio |
|---|---|---|---|---|---|---|---|---|---|---|---|
| qwen3.8-27b | ROCm | prefill | 2048 | 1 | 49.4±0.5 | 349.7±4.8 | 0.141±0.002 | LOSS | 49.44 | 351.02 | 0.141 |
| qwen3.8-27b | ROCm | prefill | 4096 | 1 | 46.9±4.4 | 338.0±2.4 | 0.139±0.013 | LOSS | 47.28 | 339.00 | 0.139 |
| qwen3.8-27b | ROCm | prefill | 500 | 1 | 50.9±0.7 | 375.7±20.6 | 0.135±0.008 | LOSS | 51.65 | 380.68 | 0.136 |
| qwen3.8-27b | ROCm | prefill | 512 | 1 | 50.4±2.8 | 343.8±20.7 | 0.147±0.012 | LOSS | 50.89 | 352.28 | 0.144 |
| qwen3.8-27b | ROCm | decode | 128 | 1 | 8.6±1.8 | 13.0±0.2 | 0.665±0.139 | LOSS | 8.77 | 13.03 | 0.673 |
