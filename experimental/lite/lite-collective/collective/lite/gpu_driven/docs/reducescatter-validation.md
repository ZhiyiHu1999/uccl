# ReduceScatter validation

## Local checks (2026-09-29)

* Host C++17 selector checks passed: P2P/IPC-ring boundaries, explicit no-IPC,
  host subpaths, 512 KiB TwoRankSmall, strict 128/512 KiB full-input thresholds,
  hierarchy disable, chunk override rounding, unsupported topology, generic
  type fallback, capacity and multiplication overflow.
* A temporary threaded host simulation of the actual service/schedule headers
  passed 40 topology/backend/policy combinations and 720 mixed-size invocations.
  Layouts: 1n×2g, 1n×4g, 2n×1g, 2n×2g, 2n×4g. Coverage includes in-place,
  scalar tails, stage/pair barriers, ring ownership, four/five-slot wrap, and
  alternating small/large policy regimes. AddressSanitizer and UndefinedBehaviorSanitizer
  also passed this simulation. CUDA copies/events and RDMA writes
  were simulated; this does **not** verify CUDA/NIC memory visibility or timing.
* Host service/initialization syntax was checked with temporary CUDA/transport
  stubs. These temporary validation programs are not repository test artifacts.
* `make collective` and `make -C nccl` were attempted. Both stopped before
  compiling the changed CUDA code: this macOS host lacks `cuda_fp16.h`/CUDA;
  the existing `core/errors.hpp` also reports an incomplete `std::string` with
  this libc++ toolchain. No CUDA build or GPU benchmark pass is claimed.

## Policy/schedule alignment with the CPU reference (2026-09-29)

Selector changes (lead classes, split-final, host-read final add, async final, eager
post, direct partner, two-rank mapped send, lead 0 for host paths) were checked with a
temporary host C++17 program over `reducescatter_plan.hpp`. Schedule changes (pull
HostRing, per-target parallel LocalRows copies, prepare/post/finish pipeline, own-shard
add from input, pack+push partner copy) were reviewed statically only: the earlier host
simulation was not re-run and nothing was compiled with CUDA. Treat them as untested
until the target-machine matrix below passes, in particular HostRing (`n-1` rows,
3-row ring stride), eager vs non-eager pipelines at 2n×2g/2n×4g, and 2n×4g with
`DIRECT_PARTNER_COPY=0`. The benchmark line now also prints async/split/host_final/
async_final/eager.

## Per-path device entries (2026-09-29)

`liteReduceScatterBlock` now switches on `plan.path` to one `__device__` entry per path
(`reducescatter_{ipc,host,two_rank,hierarchical}.cuh`). The eight single-node paths
(LocalRows, TwoLocal, P2pRing, IpcRing, HostSmall, HostRing, HostRead, HostBulk) are
composed on the device from FIFO primitives (`RsCopy`, `RsHostSum`, `RsBarrier`) and the
CTA `liteRsSum`; the service executes the primitives, the FIFO has new task kinds and
the handle carries a `LiteRsDeviceView`. The two-node paths keep the whole-invocation
task with a per-path CPU chunk preparation. None of this has been compiled with CUDA or
run: in particular check the new FIFO primitive kinds, the device-mapped row addressing
(`LiteRsLayout`), per-target `RsCopy` streams, and barrier/epoch handling across
repeated calls with different sizes.

## HostRing throughput changes (2026-10-03)

`liteRsSum` now uses 16-byte uncached loads with four independent loads per thread, and
HostRing replaces its FIFO barriers with neighbour flags in the mapped control page. The
benchmark takes `UCCL_GPU_DRIVEN_BENCH_THREADS` (multiple of 32, default 256, at most 1024)
for the thread count of the single CTA; the SM budget stays one block. Not compiled or run:
compare HostRing at 1 MiB..1 GiB with 256 and 1024 threads. A previous measurement with
stuck processes holding the GPUs (NCCL itself ten times slower) must not be used as baseline.

## Target-machine benchmark matrix

Use the existing benchmark; it performs untimed correctness preflight before
NCCL comparison and retains one block/one SM. B is the output shard size,
while the handle is initialized for R*B full-input capacity.

```bash
# From experimental/lite/lite-collective
make collective
make -C nccl
NP=4 UCCL_GPU_DRIVEN_BACKEND=cuda_ipc \
  collective/lite/gpu_driven/benchmark.sh -c reducescatter \
  -b 4 -e 64M -f 2 -g 1 -w 20 -n 100

# Mixed regimes, boundary neighbors and partial final chunks, preserving order.
NP=4 collective/lite/gpu_driven/benchmark.sh -c reducescatter \
  4 32764 32768 65536 65540 262144 1048572 1048576 2097152 2097156 \
  4194308 4 2097156 65540

# Explicit host paths, repeat with HOST_READ=0 / DIRECT_RING=0 for bulk DMA.
NP=4 MSCCLPP_NCCL_RS_NO_CUDAIPC=1 \
  collective/lite/gpu_driven/benchmark.sh -c reducescatter \
  -b 4 -e 64M -g 1 -w 20 -n 100
```

Repeat with NP=2 for 1n×2g. For two nodes set `HOSTS` to the actual two hostnames,
`CUDA_VISIBLE_DEVICES` to 1, 2 or 4 devices per node, and NP to 2, 4 or 8.
The script forwards exported `MSCCLPP_NCCL_RS_*` policy variables to MPI ranks;
all ranks must agree. The executable and Markdown report record selected path,
chunk bytes, slot count and lead.

Run both mapping capabilities where the target supports them; force host bulk
with `NO_CUDAIPC_HOST_READ=0` and `NO_CUDAIPC_DIRECT_RING=0`. Vary
`LOCAL_PARALLEL_COPY`, `EAGER_RDMA_POST`, `LOCAL_LEAD_CHUNKS`, `DIRECT_PARTNER_COPY`, its `_2D` variant, `MAPPED_SEND_FINAL_REDUCE`,
`HOST_READ_FINAL_ADD`, `SPLIT_FINAL_REDUCE`, `ASYNC_FINAL_ADD`, and chunk sizes.
Include more than five chunks and more than eight calls to exercise payload and
FIFO reuse, plus mixed float/sum, int and min/max on mapped handles.
DMA-only RS preflight explicitly reports that generic int/min/max coverage is
unavailable; optimized float/sum still receives correctness checks and timing.

Remaining required hardware checks: real CUDA compilation, DMA progress with a
resident caller CTA, system-scope visibility of mapped/IPC payload, multi-node
RDMA readiness/ACK, failure/timeout cleanup, and performance against native NCCL.
No fixed benchmark timeout or shortened sweep is part of this validation plan.
