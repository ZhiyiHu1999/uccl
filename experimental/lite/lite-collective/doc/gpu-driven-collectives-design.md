# Experimental device-callable collectives

UCCL-lite provides experimental single-node AllGather, AllReduce, and
ReduceScatter operations that can be called directly from a user CUDA kernel.
Unlike the corresponding NCCL APIs, an operation does not return to the CPU and
does not launch another kernel.

## Supported Senarios

- 1n * 2g, 1n * 4g
- One participating CUDA block per rank.
- All threads in the participating block call `liteAllGatherBlock`.
- All ranks call it in the same program order with the same byte count.
- Payloads up to the `maxBytesPerRank` selected during initialization.
- AllReduce and ReduceScatter support templated arithmetic types with sum, min,
  and max reductions.

The current data path uses GPU SM loads/stores to a shared pinned-host slab.
It is intended for small control payloads and small messages on systems without
NVLink or GPUDirect RDMA.  CudaIPC, multi-node CPU-proxy transport, chunked
DMA, CUDA graph capture, warp scope, and concurrent collectives are not yet
covered.

## Usage

Include `gpu_collectives.cuh` in an NVCC-compiled source.  After creating the
NCCL-compatible communicator, every rank collectively initializes a handle:

```cpp
mscclppDeviceAllGatherHandle_t deviceHandle{};
NCCLCHECK(mscclppGetDeviceAllGatherHandle(
    comm, maxBytesPerRank, &deviceHandle));
```

The handle is a POD value and can be passed directly as a kernel argument:

```cpp
__global__ void fusedKernel(mscclppDeviceAllGatherHandle_t handle,
                            float const* input, float* gathered,
                            size_t count) {
  // Local computation performed by this block.
  // ...
  __syncthreads();

  int result = liteAllGatherBlock(
      handle, input, gathered, count * sizeof(float));
  if (result != mscclppDeviceCollectiveSuccess) return;

  // Continue computing with gathered data.
  // ...
}
```

AllReduce uses an element count:

```cpp
int result = liteAllReduceBlock(
    handle, input, reduced, count, liteReduceSum);
```

ReduceScatter takes `nranks * recvCount` input elements on each rank and
produces `recvCount` elements. Rank `r` receives shard `r`:

```cpp
int result = liteReduceScatterBlock(
    handle, input, shard, recvCount, liteReduceSum);
```

For ReduceScatter, `maxBytesPerRank` must cover
`nranks * recvCount * sizeof(T)`. For AllReduce it must cover
`count * sizeof(T)`.

The context is owned by `comm` and is released by `ncclCommDestroy`.  The
caller must ensure that kernels using the handle have completed before
destroying the communicator.

## Progress and launch constraints

The routine spin-waits for every peer rank.  All participating rank kernels
must therefore be able to run concurrently.  Launching unrelated blocking
work ahead of the collective, conditionally skipping it on one rank, or
calling it from multiple blocks per rank can deadlock.  A future collective
launch API should enforce occupancy constraints in the same spirit as
NVSHMEM's collective launch.

## Kernel latency benchmark

The MPI benchmark uses one process per GPU. CUDA events surround each
single-block collective kernel, and MPI reports the maximum elapsed time among
all ranks for every iteration. The output contains average, p50, p95, and
minimum kernel latency in microseconds; it does not calculate speedup.

```bash
NP=4 CUDA_VISIBLE_DEVICES=0,1,2,3 \
  scripts/benchmark-device-collectives.sh
```

Optional positional arguments select per-rank byte sizes:

```bash
scripts/benchmark-device-collectives.sh 128 1K 4K 16K 64K
```
