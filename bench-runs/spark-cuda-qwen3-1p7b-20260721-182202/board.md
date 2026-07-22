| model | backend | phase | n | conc | hanzo t/s | llama t/s | ratio | verdict |
|---|---|---|---|---|---|---|---|---|
| models | CUDA | prefill | 2048 | 1 | 7285.6±745.2! | 7326.8±394.0! | 0.994±0.115 | PARITY |
| models | CUDA | prefill | 4096 | 1 | 7474.8±217.1 | 6871.3±125.0 | 1.088±0.037 | WIN |
| models | CUDA | prefill | 500 | 1 | 7961.3±228.7 | 7250.2±208.5 | 1.098±0.045 | WIN |
| models | CUDA | prefill | 512 | 1 | 7873.6±608.2! | 6190.0±1081.7! | 1.272±0.243 | WIN |
| models | CUDA | decode | 128 | 1 | 86.2±1.0 | 102.8±1.1 | 0.839±0.013 | LOSS |
