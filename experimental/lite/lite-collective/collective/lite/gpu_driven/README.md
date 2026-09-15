# GPU-driven lite collectives

This directory is the self-contained GPU-driven lite-collective subproject.
It provides device-callable AllGather, AllReduce, and ReduceScatter operations,
host-memory, CUDA IPC, and two-node host-staged RDMA backends, a compile test,
and an MPI benchmark.

The implementation is intentionally separate from the CPU-driven primitives in
`../cpu_switch`. The NCCL compatibility layer only includes the public device
API and the thin host-context adapter from this directory.

Build the compile test or benchmark from `lite-collective` with:

```bash
make -C nccl device-allgather-compile-test
make -C nccl device-collectives-bench
```

Run the host-memory benchmark with
`./collective/lite/gpu_driven/benchmark.sh`. Select direct peer GPU memory with:

```bash
UCCL_GPU_DRIVEN_BACKEND=cuda_ipc \
  ./collective/lite/gpu_driven/benchmark.sh
```

Balanced two-node communicators automatically select `host_rdma`. Supported
layouts are 2n x 1g, 2n x 2g, and 2n x 4g. CUDA IPC remains an intra-node-only
backend; inter-node payloads are moved by a CPU leader proxy over IB while the
collective itself is initiated and completed inside the user kernel.

See `../../../doc/gpu-driven-collectives-design.md` for API usage and runtime
constraints.
