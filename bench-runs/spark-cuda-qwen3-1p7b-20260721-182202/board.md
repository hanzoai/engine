| model | backend | phase | n | conc | hanzo t/s | llama t/s | ratio | verdict |
|---|---|---|---|---|---|---|---|---|
| models | CUDA | prefill | 2048 | 1 | 7001.4±918.0! | 7326.8±394.0! | 0.956±0.135 | PARITY |
| models | CUDA | prefill | 4096 | 1 | 7181.3±739.2! | 6871.3±125.0 | 1.045±0.109 | PARITY |
| models | CUDA | prefill | 500 | 1 | 7605.6±889.7! | 7250.2±208.5 | 1.049±0.126 | PARITY |
| models | CUDA | prefill | 512 | 1 | 7588.4±852.4! | 6190.0±1081.7! | 1.226±0.255 | PARITY |
| models | CUDA | decode | 128 | 1 | 86.5±1.1 | 102.8±1.1 | 0.842±0.014 | LOSS |
