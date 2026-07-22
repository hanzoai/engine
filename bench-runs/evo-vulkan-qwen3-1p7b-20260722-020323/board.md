| model | backend | phase | n | conc | hanzo t/s | llama t/s | ratio | verdict |
|---|---|---|---|---|---|---|---|---|
| qwen3-1.7b | Vulkan | prefill | 2048 | 1 | 1474.7±12.4 | 4401.1±26.1 | 0.335±0.003 | LOSS |
| qwen3-1.7b | Vulkan | prefill | 4096 | 1 | 970.1±6.5 | 3932.3±36.4 | 0.247±0.003 | LOSS |
| qwen3-1.7b | Vulkan | prefill | 500 | 1 | 2386.3±31.8 | 4675.7±30.6 | 0.510±0.008 | LOSS |
| qwen3-1.7b | Vulkan | prefill | 512 | 1 | 3117.0±19.1 | 4724.6±40.7 | 0.660±0.007 | LOSS |
| qwen3-1.7b | Vulkan | decode | 128 | 1 | 109.9±1.4 | 155.3±0.2 | 0.707±0.009 | LOSS |
