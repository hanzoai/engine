| model | backend | phase | n | conc | hanzo t/s | llama t/s | ratio | verdict |
|---|---|---|---|---|---|---|---|---|
| models | CUDA | prefill | 2048 | 1 | 3774.4±89.0 | 3592.5±79.2 | 1.051±0.034 | WIN |
| models | CUDA | prefill | 512 | 1 | 3857.6±336.4! | 3648.2±232.9! | 1.057±0.114 | PARITY |
| models | CUDA | decode | 128 | 1 | 47.3±3.9! | 53.7±0.9 | 0.880±0.073 | LOSS |
