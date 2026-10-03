# AllReduce validation

## Local checks (2026-09-29)

* A temporary host C++17 program over `allreduce_plan.hpp` checked the selector: one rank,
  1n×2g / 1n×4g RS+AG, irregular counts, non-float/sum, the 64 KiB and 128 KiB thresholds
  (both sides, and 2n×2g never selecting the two-leader path), unmapped slabs falling back to
  RS+AG, the 64 MiB ring threshold, `2RANK_RING_ALLREDUCE` forcing, `AR_RS_AG_MIN_BYTES`, and
  the ring accepting irregular counts. It is not a repository test.
* Nothing has been compiled with CUDA or run: the host has no CUDA toolchain. The FIFO task
  kind, the second connection of the 2n×1g ring, the shared control-word protocol
  (`previousAck` / `previous`), the AllReduce schedules and the device entries are untested.

## Target-machine benchmark matrix

`B` for AllReduce is the complete tensor per rank; the benchmark prints
`allreduce bytes=... path=... chunk_bytes=...` and the report has an "AllReduce paths"
section. Use the existing benchmark (correctness preflight, one block, NCCL comparison).

```bash
# 1n*4g: RS+AG through the AllGather / ReduceScatter entries (IPC backend).
NP=4 UCCL_GPU_DRIVEN_BACKEND=cuda_ipc \
  collective/lite/gpu_driven/benchmark.sh -c allreduce \
  -b 4B -e 1G -f 2 -g 1 -w 20 -n 100

# Boundaries: 64 KiB / 128 KiB small paths, non-divisible counts (Generic),
# repeated mixed sizes to exercise slot and epoch reuse.
NP=8 HOSTS=host1,host2 CUDA_VISIBLE_DEVICES=0,1,2,3 \
  collective/lite/gpu_driven/benchmark.sh -c allreduce \
  4 65532 65536 65540 65544 131068 131072 131076 262144 4 65536 131072 1048576
```

Cover all five layouts (`NP=2` 1n×2g; `NP=4` 1n×4g; two nodes with `HOSTS`, 1/2/4 devices per
node and `NP` 2/4/8). Vary `MSCCLPP_NCCL_2RANK_RING_ALLREDUCE=1` with
`MSCCLPP_NCCL_2RANK_RING_CHUNK_BYTES` on 2n×1g (irregular counts, single/multiple loops,
partial tails), `MSCCLPP_NCCL_AR_RS_AG_MIN_BYTES` to force Generic, and the
`MSCCLPP_NCCL_RS_*` variables that change the RS stage. Repeat with
`MSCCLPP_NCCL_RS_NO_CUDAIPC=1` and `UCCL_GPU_DRIVEN_BACKEND=host`.

Remaining required hardware checks: real CUDA compilation, in-place RS+AG with every AG
path (in particular the IPC ring with a registered output), the small paths' final-row
visibility and slot reuse across many calls, two-channel ring correctness (channel
independence, tail parts, count smaller than a part), RDMA readiness/ACK across mixed RS and
AllReduce calls on the same handle, failure/timeout cleanup and performance against native
NCCL. No fixed benchmark timeout or shortened sweep is part of this plan.
