# GPU-driven AllReduce

The entry point is `liteAllReduceBlock<T>(handle, src, dst, count, op)`. One CTA on every
rank calls it in the same order inside the user kernel; the CPU initializes resources once
and starts the service. AllReduce follows the collective and primitive architecture of
AllGather and ReduceScatter: a pure plan selects one optimization path, and the entry
dispatches to one independent `__device__` function per path. No path calls a CPU
collective, NCCL send/recv or launches a child kernel.

## Semantics and notation

* `R = N * P` ranks (`N` nodes, `P` ranks per node), `C = count`, `B = C * sizeof(T)`: the
  complete tensor bytes of one rank. Input and output have the same size, and rank `r`
  receives `reduce(src[k][i], k = 0..R-1)` for every `i`. In place means `src == dst`.
* `maxBytesPerRank` bounds the complete tensor `B`.
* For the equal-shard composition, `S = B / R` is the reduced shard size. ReduceScatter sees
  `B_RS = S`, `T_RS = B`; AllGather sees `B_AG = S`, `T_AG = B`. Never feed `B` to a
  per-shard threshold of those collectives.
* Optimized paths apply only to float/sum. Every other type or operation, irregular counts
  and ineligible configurations use the arithmetic-template full-tensor `Generic` path
  (`liteAllReduceGenericBlock`), which keeps its mapped-slab requirement.
* Argument checks (null pointers, zero count, alignment, capacity) complete before any FIFO
  work is published.

## Source map

| File | Content |
|---|---|
| [allreduce_plan.hpp](../allreduce_plan.hpp) | `LiteAllReducePolicy`, `LiteAllReducePath`, pure `litePlanAllReduce` (host/device/benchmark share it) |
| [allreduce.cuh](../allreduce.cuh) | validation, float/sum dispatch (`switch (plan.path)`), Copy, Generic, `liteArInvokePath` |
| [allreduce_rs_ag.cuh](../allreduce_rs_ag.cuh) | `liteAllReduceRsAgBlock` |
| [allreduce_small.cuh](../allreduce_small.cuh) | `liteAllReduceSmallMappedBlock`, `liteAllReduceSmallTwoLeaderBlock` |
| [allreduce_two_rank_ring.cuh](../allreduce_two_rank_ring.cuh) | `liteAllReduceTwoRankRingBlock` |
| [allreduce_service.hpp](../allreduce_service.hpp) | policy read/compare, CPU schedules of the network paths, `executeLiteAllReduce` |
| [gpu_collectives.cuh](../gpu_collectives.cuh) | `liteAllReduceGenericBlock` (the full-tensor staging path) |

The network paths reuse the ReduceScatter context (host slab rows, GPU scratch, control
words, the connection of each local rank to the same-index rank of the other node) and the
whole-invocation `AllReduce` FIFO task whose CPU schedule requests CTA copy/sum phases
through `liteRsInvokeTask`; see [reducescatter.md](reducescatter.md).

## CPU path correspondence

The reference is `runSendRecvAllReduce()` in `nccl/native_collectives.cu`. Thresholds are
defaults; KiB/MiB are binary. Rows are matched top to bottom within a topology.

| Topology / condition | GPU path | CPU reference | Device entry |
|---|---|---|---|
| 1 rank | Copy | (trivial copy) | (CTA copy in `liteAllReduceGenericBlock`) |
| 2n×1g, float/sum, B ≥ 64 MiB or `2RANK_RING_ALLREDUCE` | TwoRankRing | `runTwoRankRingSimpleAllReduce2Node` | `liteAllReduceTwoRankRingBlock` |
| 2n×2g / 2n×4g, float/sum, 0 < B ≤ 64 KiB, mapped host slab | SmallMapped | `runSmallMappedAllReduce2Node` | `liteAllReduceSmallMappedBlock` |
| 2n×4g, float/sum, 64 KiB < B ≤ 128 KiB, even C | SmallTwoLeader | `runSmallTwoLeaderAllReduce2Node` | `liteAllReduceSmallTwoLeaderBlock` |
| float/sum, C % R == 0, ReduceScatter path eligible for S, B ≥ `AR_RS_AG_MIN_BYTES` | ReduceScatterAllGather | RS + AG tail of `runSendRecvAllReduce` | `liteAllReduceRsAgBlock` |
| everything else (other type/op, C % R != 0, missing mappings, unsupported layouts) | Generic | `runHierarchicalAllReduce2Node` (two nodes) / generic staging | `liteAllReduceGenericBlock` |

Single-node layouts (1n×2g, 1n×4g) take RS+AG or Generic. The CPU reference does not use its
`runSendRecvAllReduce` there; it uses the registered MSCCLPP selector algorithms, which are
not device-callable. No size crossover is invented: `AR_RS_AG_MIN_BYTES` defaults to zero
(RS+AG whenever eligible) until GPU-driven measurements establish one.

## What each path does

* **ReduceScatterAllGather**: `ReduceScatter(src → dst + rank*S)` leaves each rank's reduced
  shard at its global-rank output offset; `AllGather(dst + rank*S → dst)` restores the tensor
  in global-rank order. Each collective picks its own path from its own plan, so this path
  inherits their selection, chunking and policy variables. Phase and epoch identities stay
  separate because the two collectives run through their own entries.
* **SmallMapped** (two nodes, small B): every rank copies its whole input into its row of the
  node's mapped host slab with the CTA and all local ranks barrier. The node leader (local
  rank 0) CPU-sums the local rows into a node partial, writes it to the remote leader over
  RDMA, waits for the remote partial, CPU-adds the two into a final row and publishes a final
  flag. Every rank waits for the flag and copies the final row into its output with the CTA;
  a done barrier lets the slot be reused.
* **SmallTwoLeader** (2n×4g, 64–128 KiB): the same idea but the tensor is split in halves.
  Input is staged by D2H DMA, local ranks 0 and 2 each reduce one half across the four local
  ranks, exchange it with the same-index remote rank, add the two node partials and publish
  their final half; every rank H2D-copies both final halves into its output.
* **TwoRankRing** (2n×1g, large B): the NCCL RING/SIMPLE data order over two channels with
  independent connections. The tensor is split in two channel ranges; each loop of a channel
  splits `2 × part` elements into two parts, rank `r` owning part `r`: D2H the peer part → RDMA
  → H2D the remote contribution of the own part and add it into the output (CTA) → D2H the
  reduced own part → RDMA → H2D the remote final peer part into the output. Part size is
  `MSCCLPP_NCCL_2RANK_RING_CHUNK_BYTES` (default 4 MiB) bounded by two rows of the RS
  scratch layout.
* **Generic**: the original path: each rank stages the complete tensor chunk by chunk through
  the payload slab and reduces it element-wise; it serves any supported type/operation and
  irregular counts (including `C < R`). It never truncates the tensor to make `C` divisible.
  A tensor larger than one staging epoch (two-node slab rows capped by
  `UCCL_GPU_DRIVEN_STAGING_MAX_BYTES`) is reduced in contiguous slices of one epoch each.

## Policy variables

All variables are snapshotted at host initialization and compared field by field across ranks;
the RS variables (`MSCCLPP_NCCL_RS_*`) also apply because RS+AG and the network paths share
the ReduceScatter context.

* `MSCCLPP_NCCL_2RANK_RING_ALLREDUCE`: any value other than `0`/`false`/`FALSE` forces
  TwoRankRing below 64 MiB (float/sum, 2n×1g).
* `MSCCLPP_NCCL_2RANK_RING_CHUNK_BYTES`: ring part bytes, truncated to whole floats.
* `MSCCLPP_NCCL_AR_RS_AG_MIN_BYTES`: smallest `B` that composes RS+AG (default 0).

Fixed thresholds: SmallMapped `B ≤ 64 KiB`, SmallTwoLeader `64 KiB < B ≤ 128 KiB` (the source
constant is 128 KiB), ring auto-enable `B ≥ 64 MiB`.

## Network schedules

Rows of one slot (row capacity = the RS `chunkCapacity`, so small paths need `B ≤ chunkCapacity`):
row 0 = input of local rank `r` at `hostRow(slot, r, 0)`, row 1 = local partial (RDMA source;
the remote partial arrives at the same offset of the receive slab), row 2 = final. Control rows:
0 input-ready / remote data-ready, 1 remote ACK, 2 done, 3 final flag.

* **Node exchange** (leaders only): CPU sum of the local rows, RDMA write of the partial and
  ready signal, wait for the remote partial, CPU add, publish the final flag, send the ACK. The
  ACK lets the remote reuse our receive row, so before a slot is rewritten a leader waits for
  the previous remote ACK (`previousAck`) and every rank for the previous local done barrier
  (`previous`). Ranks that do not exchange (non-leaders) never wait for or send ACKs.
* **Epochs** come from the ReduceScatter context's counter, which advances once per collective
  call (RS or AllReduce), so ring, RS and AllReduce words never regress each other. Inside a
  call, the stages of the ring are numbered by a sequence and every control word carries the
  stamp `(epoch << 32) | sequence`; small paths use sequence 1. Slots rotate with the epoch.
* **TwoRankRing** uses slot `k` and connection `k` for channel `k`, control rows 0 (peer part
  arrived) and 1 (final part arrived). Stage 0 sits at row 0 and stage 1 at row 2 (each up to
  two rows); a stage is consumed before the next one is produced, so no ACK is needed. One
  service thread interleaves the two channels step by step; the CPU reference uses one worker
  thread per channel.

## Initialization, capacity and ownership

`maxBytesPerRank` bounds `B`. The second connection of the ring is created at initialization
together with the first, and all ranks must agree on the AllReduce policy. IPC backend with
RS+AG: the AllGather stage may run the IPC ring, which pushes into the next rank's output, so
the output must be registered exactly as for AllGather. Resource lifetime follows the
ReduceScatter context: streams/events, mappings, registrations and both connections are
released after the service is stopped and joined.

## RS+AG in-place safety

In-place RS+AG must not let AllGather overwrite an input region another rank still needs.
No ReduceScatter path reads another rank's original input (peers read staged copies, mailbox
rows or partials, see [reducescatter.md](reducescatter.md)), and the data a rank receives in
AllGather for shard `q` only exists after rank `q` finished its ReduceScatter and therefore
after every contribution to `q` was consumed. The IPC-ring AllGather additionally starts with
an all-rank ready barrier. If a future ReduceScatter path reads a peer's original input, add
an explicit all-rank boundary between the two phases first.

## Known deviations from the CPU reference

* Single node: RS+AG or Generic instead of the registered MSCCLPP selector algorithms.
* Two nodes, large B: standalone GPU-driven ReduceScatter (its own selection) followed by
  AllGather, instead of the CPU's `runSendRecvReduceScatter` variants; irregular counts use the
  Generic path instead of `runHierarchicalAllReduce2Node`.
* The ring's two channels are interleaved by one service thread, and the part size is bounded
  by the scratch row capacity.
* Small paths require `B ≤ chunkCapacity` of the RS layout; a handle created for a small
  `maxBytesPerRank` falls back to RS+AG or Generic.

## Validation boundary

This document describes implementation and protocol. It has not been compiled with CUDA or
validated on GPUs/IB hardware, and no performance parity with the CPU-driven version or NCCL
is claimed. See [docs/allreduce-validation.md](../docs/allreduce-validation.md).
