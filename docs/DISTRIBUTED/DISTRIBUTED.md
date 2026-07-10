# Distributed inference in Hanzo Engine

Hanzo Engine supports distributed inference with a few strategies
- [NCCL](NCCL.md) (recommended for CUDA)
- [Ring backend](RING.md) (supported on all devices)

**What backend is best?**
- **For CUDA-only system**: NCCL
- **Anything else**: Ring backend

The Ring backend is also **heterogenous**! This means that you can use the Ring backend on any set of multiple devices connected over TCP.
For example, you can connect 2 Metal systems, or 2 Metal and 1 CPU system with the Ring backend!

For quantized **GGUF** models the Ring backend runs [pipeline parallelism](RING.md#pipeline-parallelism-gguf--quantized-models): the model is split by layer range across the ring so a model larger than any single node fits across the cluster. This works across mixed vendors (e.g. a ROCm head with a CUDA worker).