# Validation and benchmarking

## Development-host checks

## The development host is macOS without nvcc, CUDA headers, GPUs or IB.

* Portable selector checks passed for host mapped/cooperative/DMA boundaries, chunk sizes, the P=2 packing gap, small-message priority, NUMA eligibility, P=2 NUMA exclusion, fallback and capture rejection.
* A temporary C++ simulation compiled the actual `gpu_collectives.cuh`, `host_context.hpp` and `service.hpp` against CUDA/bootstrap/network substitutes. It passed 2n×1g, 2n×2g, 2n×4g single-slab, dual-rail and NUMA-group exchanges, including mapped and DMA-only operation, odd sizes, in-place/unaligned output, slot/FIFO wrap and mixed AllGather/reduction epochs. Runs were bounded to 15 seconds. It models one CPU thread per CTA and immediate mock copies/writes; it does not model hardware visibility, asynchronous DMA timing or transport failures.
* Clang generated sm_80 PTX for a translation unit instantiating all three device routines with declaration shims. Host context and benchmark syntax also passed shim-based checks. These are not nvcc builds against real CUDA/MPI headers.
* Shell syntax and diff whitespace checks passed.
* `make collective` and `make -C nccl` stopped at missing `cuda_fp16.h` and a pre-existing incomplete `std::string` declaration in `core/errors.hpp` under the local libc++ toolchain. The benchmark build stopped because `nvcc` is absent.

Temporary validation harnesses were kept outside the source tree. Per AGENTS.md, standalone correctness executables and test-only source files were removed; correctness preflight now resides in the benchmark.

## Target-machine build

From `experimental/lite/lite-collective`:

```sh
make collective
make -C nccl
make -C nccl device-collectives-bench
```

Set `NCCL_BASELINE_LIB` to an actual native NCCL library, not the UCCL compatibility library. The benchmark sets NCCL's minimum/maximum CTAs and channels to one, and disables GDR before either communicator is initialized. The GPU routine launches exactly one CTA. Confirm actual native NCCL kernel grid dimensions with a profiler on the selected NCCL version before claiming a measured one-SM comparison.

## Bounded benchmark runs

`benchmark.sh` includes a 15-second MPI execution timeout. Use targeted sizes and small iteration counts for initial checks; split long matrices across invocations. A timeout is a failure, not a result. Set `MPI_HOME`, `NCCL_BASELINE_LIB`, NIC/bootstrap variables and host names for the testbed. Forwarded host tuning variables are listed in the script.

```sh
export NCCL_BASELINE_LIB=/path/to/native/libnccl.so
export WARMUP_ITERS=1 ITERS=3

# One node: explicit host SHM, including DMA-only mapping policy.
NP=2 CUDA_VISIBLE_DEVICES=0,1 collective/lite/gpu_driven/benchmark.sh 1048577
NP=4 CUDA_VISIBLE_DEVICES=0,1,2,3 collective/lite/gpu_driven/benchmark.sh 524289
NP=4 CUDA_VISIBLE_DEVICES=0,1,2,3 MSCCLPP_NCCL_HOST_ALLGATHER_MAP_SLAB=0 \
  collective/lite/gpu_driven/benchmark.sh 4096

# CUDA IPC needs T >= 8 MiB, host SHM disabled, and peer accessibility.
NP=4 CUDA_VISIBLE_DEVICES=0,1,2,3 UCCL_GPU_DRIVEN_BACKEND=cuda_ipc \
  MSCCLPP_NCCL_HOST_ALLGATHER=0 collective/lite/gpu_driven/benchmark.sh 2097153

# Two nodes: substitute real node names. Rank placement must be contiguous.
NP=2 HOSTS=nodeA,nodeB CUDA_VISIBLE_DEVICES=0 \
  collective/lite/gpu_driven/benchmark.sh 1048577
NP=4 HOSTS=nodeA,nodeB CUDA_VISIBLE_DEVICES=0,1 \
  collective/lite/gpu_driven/benchmark.sh 524289
NP=8 HOSTS=nodeA,nodeB CUDA_VISIBLE_DEVICES=0,1,2,3 \
  collective/lite/gpu_driven/benchmark.sh 2097153
```

AllGather accepts arbitrary byte sizes, including odd tails. Float reductions are skipped for sizes not divisible by four. Before timing, the benchmark checks changing input across repeated calls inside a single kernel, aligned/unaligned and in-place/out-of-place AllGather, plus int/float Sum/Min/Max reductions. Timed GPU output and accumulated return status are checked before native NCCL overwrites output. Untimed checks and MPI barriers are excluded from latency samples.

Required hardware matrix: all five target topologies; both sides and equality of every documented threshold; repeated mixed sizes crossing small/NUMA dispatch; asymmetric versus symmetric NUMA layouts; more than 16 MiB per-rank NUMA input; single/dual NIC generic blocks around 2 MiB; host tuning overrides; cooperative capability disabled; in-place and odd tails; mapped slabs disabled; repeated FIFO/slot wrap; injected service/transport failures; and explicit capture rejection. Test IPC only where CUDA P2P access is available. The same benchmark process iterates requested sizes in order, so passing a small/large/small sequence exercises context switching.

Generated measurements default to `.tmp/gpu-driven-benchmarks/`. No CUDA/IB measurements from this change are available yet; do not infer performance from the portable simulation.
