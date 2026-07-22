| model | backend | phase | n | conc | hanzo t/s | llama t/s | ratio | verdict |
|---|---|---|---|---|---|---|---|---|
| qwen3-1.7b | Vulkan | prefill | 2048 | 1 | 1475.0±37.6 | 4256.8±28.9 | 0.347±0.009 | LOSS |
| qwen3-1.7b | Vulkan | prefill | 4096 | 1 | 922.8±33.3 | 3850.3±84.0 | 0.240±0.010 | LOSS |
| qwen3-1.7b | Vulkan | prefill | 500 | 1 | 1098.9±241.4! | 4449.6±33.6 | 0.247±0.054 | LOSS |
| qwen3-1.7b | Vulkan | prefill | 512 | 1 | 3042.3±10.9 | 4547.7±49.4 | 0.669±0.008 | LOSS |
| qwen3-1.7b | Vulkan | decode | 128 | 1 | 80.3±2.4 | 154.5±0.9 | 0.519±0.016 | LOSS |
