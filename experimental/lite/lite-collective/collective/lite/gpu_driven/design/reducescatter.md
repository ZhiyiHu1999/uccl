# GPU-driven ReduceScatter

The entry point is `liteReduceScatterBlock<T>(handle, src, dst, recvCount, op)`.
One CTA on every rank calls it in the same order inside the user kernel. The CPU
initializes resources once and starts the service. Data movement uses the existing
FIFO/service, CUDA DMA, mapped SHM and host-memory RDMA architecture. It does not
call CPU collectives or NCCL send/recv, and it does not launch reduction sub-kernels.

## Semantics and layering

Let `R = N * P`, `C = recvCount`, `B = C * sizeof(T)`, and `T_bytes = R * B`.
Rank r receives the reduction of `[r*C, (r+1)*C)` across all source inputs.
`maxBytesPerRank` bounds the **full input** `T_bytes`, not just the output shard.
The in-place layout is `dst = src + rank*C`. Null pointers, zero counts,
multiplication overflow, misalignment, invalid rank/op and over-capacity requests
are rejected, and these checks complete before any FIFO work is published.

* `reducescatter_plan.hpp`: pure host/device path selection (no allocation, memory access or global state, so the device caller, the service and the benchmark always agree). Each threshold compares either the per-rank output bytes B or the full input bytes T_bytes, exactly as the CPU reference does.
* `reducescatter.cuh`: argument checks, float/sum dispatch, and `switch (plan.path)` to the per-path device entries. Also holds `liteRsInvokePath`, the single-CTA phase execution loop used by the two-node paths.
* `reducescatter_primitives.cuh`: the device-side primitives (see "Primitives" below) that the single-node paths compose.
* Per-path `__device__` entries, one function per plan path (like AllGather's `liteAllGather<Path>Block`), each checking its own prerequisites and naming its CPU reference:
  * `reducescatter_ipc.cuh`: `LocalRows`, `TwoLocal`, `P2pRing`, `IpcRing`. Composed on the device from primitives.
  * `reducescatter_host.cuh`: `HostSmall`, `HostRing`, `HostRead`, `HostBulk`. Composed on the device from primitives.
  * `reducescatter_two_rank.cuh`: `TwoRankSmall`, `TwoRankPipeline`. Whole-invocation task; CPU schedule.
  * `reducescatter_hierarchical.cuh`: `SmallHost`, `HostStaged`, `HierarchicalTwo`, `HierarchicalFour`. Whole-invocation task; CPU schedule.
* `reducescatter_generic.cuh`: the original full-input staging algorithm; keeps arithmetic-template sum/min/max.
* `reducescatter_protocol.hpp`: FIFO primitive payload (`LiteRsOp`), the scratch/host-slab geometry (`LiteRsLayout`) shared by service and device, the device view stored in the handle (`LiteRsDeviceView`), and the whole-invocation phase words.
* `reducescatter_service.hpp`: initialization, scratch/SHM/IPC/QP lifecycle; executes the FIFO primitives (`enqueueLiteRsCopy`, `executeLiteRsHostSum`, `executeLiteRsBarrier`) and the two-node schedule (`executeLiteReduceScatter`).
* `reducescatter_local_schedule.hpp`: the CPU sum and row staging helpers still needed by the CPU-reduced paths.
* `reducescatter_network_schedule.hpp`: CPU schedule of the two-node paths, `liteRsNetwork` (prepare / post / finish per chunk). The per-path chunk preparation is `liteRsPrepareCpuChunk` (SmallHost, HostStaged, TwoRankSmall without mapping), `liteRsPrepareTwoRankChunk` (TwoRankSmall, TwoRankPipeline) and `liteRsPrepareHierarchicalChunk` (HierarchicalTwo, HierarchicalFour).

### Primitives

Each operation is executed by whichever agent it needs. Operations that need the CPU, the copy engines or the NIC are FIFO primitives executed by the service; operations that need the SMs are CTA primitives executed by the calling CTA. A single-node path is a `__device__` function that sequences them; every ticket is awaited before the path returns.

| Primitive | Executed by | Meaning |
|---|---|---|
| `liteRsCopyAsync` / `liteRsAwait` (`RsCopy`) | service stream (DMA) | 1D or 2D copy between any two UVA addresses: D2D over IPC, D2H, H2D. Streams 0–3 allow parallel copies |
| `liteRsHostSum` (`RsHostSum`) | service CPU | float sum of up to four pinned-host rows into a host row (AVX-512 or scalar) |
| `liteRsBarrier` (`RsBarrier`) | service | publish this rank's epoch in a control row of the node's host slab and wait for all local ranks |
| `liteRsSum` | CTA threads | elementwise float sum of up to four rows in peer scratch / mapped host / input; volatile loads, system fence, CTA barrier |
| `liteRsEpochLoad` / `liteRsEpochStore` | CTA thread 0 | persistent barrier epoch in the FIFO control page |

Row addresses come from `LiteRsDeviceView` (peer GPU scratch pointers, CPU and device-mapped addresses of the host slab) using the same `LiteRsLayout` offset functions as the service. The two-node paths are not decomposed yet: they submit one whole-invocation `ReduceScatter` task and the CPU schedule requests the CTA phases (like AllGather's `NetworkAllGather`).

Each path has one device entry (`liteReduceScatter<Path>Block`) and one CPU-reference function (see the path table). Single-node entries contain the algorithm; two-node entries submit the invocation and the like-named CPU schedule owns DMA, RDMA, barriers, ACKs and slot reuse.

Optimized paths apply only to float/sum. Other types/ops use the generic path,
whose mapped-slab requirement is unchanged.

A single rank does a direct CTA copy (skipped when in place). One node, or two nodes with a uniform layout, up to eight ranks are supported. Dedicated optimizations cover 1n×2g, 1n×4g, 2n×1g, 2n×2g and 2n×4g; other supported layouts use the generic path.

Initialization rejects three or more nodes: the multi-node schedule pairs each rank only with its same-local-index peer on the other node, so extra nodes would be silently skipped. AllReduce composes this ReduceScatter with AllGather only in its own `ReduceScatterAllGather` path (see [allreduce.md](allreduce.md)); ReduceScatter itself never depends on AllReduce.

## CPU path correspondence

The reference is `runLiteInterReduceScatter()` in `nccl/ReduceScatter/multi-node.cu`, together with the `single-node.cu` it includes. All thresholds below are defaults; KiB/MiB are binary. Within a group, rows are matched top to bottom. The benchmark prints the selected path, chunk, slots and lead.

### Vocabulary

* **Shard**: the `B`-byte slice of the input that belongs to one destination rank. Rank `r`'s output is the sum of every rank's shard `r`.
* **Row**: one shard-sized (or chunk-sized) buffer inside a scratch area, indexed by source rank or target rank.
* **Partial**: a sum over some, but not all, contributing ranks. Node partial = the sum over the ranks of one node.
* **Staging**: copying data into an intermediate buffer (GPU scratch, or pinned host memory) so a peer, the CPU or the NIC can read it.
* **IPC**: GPUs of one node can read/write each other's scratch (`cudaIpc`).
* **Mapped**: pinned host memory that GPU threads can access directly.

### Path table

| Topology / condition                                                               | GPU path         | CPU reference function                                      | Device entry | Schedule |
| ---------------------------------------------------------------------------------- | ---------------- | ----------------------------------------------------------- | --- | --- |
| 1 rank                                                                             | Copy             | (trivial copy)                                              | (CTA copy in liteReduceScatterGenericBlock) | `-` |
| 1n×4g, IPC, P2P forced or 256 KiB ≤ B ≤ 2 MiB                                   | P2pRing          | `runP2pRingReduceScatter`                                 | `liteReduceScatterP2pRingBlock` | device-composed |
| 1n×4g, IPC, B ≥ 1 MiB and local ring on (default B ≥ 2 MiB)                     | IpcRing          | `runLocalFourRankRingReduceScatter`                       | `liteReduceScatterIpcRingBlock` | device-composed |
| 1n×4g, IPC, otherwise                                                             | LocalRows        | `runLocalFourRankReduceScatter`                           | `liteReduceScatterLocalRowsBlock` | device-composed |
| 1n×2g, IPC                                                                        | TwoLocal         | none (GPU extension of LocalRows)                           | `liteReduceScatterTwoLocalBlock` | device-composed |
| Single node, no IPC, B ≤ 64 KiB                                                   | HostSmall        | `runNoCudaIpcHostReduceScatter`                           | `liteReduceScatterHostSmallBlock` | device-composed |
| Single node, no IPC, mapped, direct ring on, B ≥ 1 MiB                            | HostRing         | `runNoCudaIpcDirectRingSingleNodeReduceScatter`           | `liteReduceScatterHostRingBlock` | device-composed |
| Single node, no IPC, mapped, host-read on                                          | HostRead         | `runNoCudaIpcHostReadSingleNodeReduceScatter`             | `liteReduceScatterHostReadBlock` | device-composed |
| Single node, no IPC, otherwise                                                     | HostBulk         | `runNoCudaIpcBulkSingleNodeReduceScatter`                 | `liteReduceScatterHostBulkBlock` | device-composed |
| 2n×1g, B ≤ 512 KiB and slot capacity suffices                                    | TwoRankSmall     | `runTwoRankSmallHostReduceScatter`                        | `liteReduceScatterTwoRankSmallBlock` | `liteRsNetwork / liteRsPrepareTwoRankChunk` |
| 2n×1g, otherwise                                                                  | TwoRankPipeline  | `runTwoRankReduceScatter` / `runTwoRankPipelinedChunks` | `liteReduceScatterTwoRankPipelineBlock` | `liteRsNetwork / liteRsPrepareTwoRankChunk` |
| 2n×2g with T_bytes < 128 KiB, or 2n×4g with T_bytes < 512 KiB (slots sufficient) | SmallHost        | `runSmallHostReduceScatter`                               | `liteReduceScatterSmallHostBlock` | `liteRsNetwork / liteRsPrepareCpuChunk` |
| 2n×2g / 2n×4g, no IPC                                                            | HostStaged       | `runNoCudaIpcHostReduceScatter` (multi-node)              | `liteReduceScatterHostStagedBlock` | `liteRsNetwork / liteRsPrepareCpuChunk` |
| 2n×2g, hierarchy on                                                               | HierarchicalTwo  | `runTwoNodeTwoGpuHierReduceScatter`                       | `liteReduceScatterHierarchicalTwoBlock` | `liteRsNetwork / liteRsPrepareHierarchicalChunk` |
| 2n×2g, hierarchy off                                                              | Generic          | (CPU: host NCCL send/recv fallback; not callable here)      | `liteReduceScatterGenericBlock` | (none) |
| 2n×4g, otherwise                                                                  | HierarchicalFour | `runChunk` / `runPipelinedChunks`                       | `liteReduceScatterHierarchicalFourBlock` | `liteRsNetwork / liteRsPrepareHierarchicalChunk` |

With explicit `NO_CUDAIPC=1`, multi-node no-IPC layouts always take HostStaged, even for small T_bytes (the CPU never runs `runSmallHostReduceScatter` there); otherwise SmallHost is tried first.

### What each path does

**Single node, IPC**

* **P2pRing / IpcRing** (ring): the ranks form a ring. In step 1 every rank sends the shard that the *previous* rank owns to its next neighbour; each later step adds the received partial to the local copy of the next shard and forwards it. After `P-1` steps the partial that arrives at rank `r` is complete for shard `r` and is written to the output. Data moves GPU to GPU by DMA into the neighbour's scratch. Good for larger messages because every link is busy at once. The two paths are the same algorithm; the CPU reference implements them with different transports and therefore they win in different size ranges. **P2pRing** (256 KiB ≤ B ≤ 2 MiB, or `P2P_RING=1`) uses NCCL `send/recv`, which NCCL orders on the stream without CPU flags, and moves the whole message at once. **IpcRing** (B ≥ 2 MiB, or B ≥ 1 MiB with `LOCAL_RING=1`) uses `cudaMemcpyAsync` into the neighbour's IPC-mapped scratch with a CPU-published flag after each step, in chunks of up to 16 MiB with several slots in flight. P2pRing is tested first, so for 1–2 MiB it wins even when the local ring is enabled. In the GPU implementation each has its own device entry (`liteReduceScatterP2pRingBlock`, `liteReduceScatterIpcRingBlock`) but both run **the same CPU schedule** (`liteRsPushRing`): NCCL send/recv cannot be called here, so P2pRing also pushes through IPC DMA into the neighbour's mailbox, chunked at 16 MiB (identical to the CPU for B ≤ 2 MiB). Only the selection thresholds, the device entry and the reported path name differ. The split is kept so that each GPU path maps to one CPU path and thresholds can be tuned separately once hardware data exists; do not expect a performance difference, and do not merge them without merging their selection rules.
* **LocalRows** (all-to-one per target): every rank copies shard `t` into row "my rank" of target `t`'s scratch. When all `P` rows have arrived, each rank sums its `P` rows into the output. Small messages (≤ 32 KiB) are scattered by the calling CTA instead of DMA; messages ≥ 64 KiB copy to all targets in parallel streams. Best for small messages (B < 256 KiB by default) because it takes only two rounds (scatter, then sum), whereas the ring pays a synchronization per step.
* **TwoLocal** (1n×2g): LocalRows with two ranks: send only the peer's shard, read your own shard directly from the input, add.

**Single node, shared host memory, no IPC**

* **HostSmall**: every rank copies its whole input to pinned host memory; the CPU sums the `P` rows of the target rank; the result is copied back to the GPU.
* **HostBulk**: every rank copies only the peer-owned shards to host memory; each rank then copies the `P-1` peer rows back into GPU scratch and the CTA sums them with its own shard.
* **HostRead**: like HostBulk but without the copy back: the CTA sums the peer rows by reading them straight out of mapped host memory.
* **HostRing**: the ring above, but the intermediate partials live in mapped host memory. Each step one CTA pass reads the previous rank's row, adds the local shard, and writes this rank's row; there is no push copy.

**Two nodes, one GPU per node (2n×1g)**

* **TwoRankSmall**: send the peer's shard in one RDMA message, receive the peer's contribution to your shard, add it to your own shard. With mapped memory the CTA writes the outgoing shard directly into the send buffer and does the final add itself; otherwise the CPU does the copy/add.
* **TwoRankPipeline**: the same exchange but split into chunks (up to five in flight) so the copy to host, the network transfer and the copy back overlap.

**Two nodes, several GPUs per node**

Every rank talks over the network only to the same-index rank on the other node, and only *partial sums* cross the network.

* **SmallHost / HostStaged** (CPU-driven): each rank copies its input to host memory; the CPU sums, for its own target and for the peer's matching target, the inputs of all local ranks; the peer's partial is sent over RDMA; the receiver adds it to its own partial and copies the result to the GPU. HostStaged does this per 256 KiB chunk, one chunk at a time.
* **HierarchicalTwo** (2n×2g): each rank first gives its local partner the shards the partner needs (same-node exchange). Every rank then computes two node partials: one for its own shard and one for the shard of its matching rank on the other node. The second is sent over RDMA; the first is kept. The received partial is added to the kept one to give the output.
* **HierarchicalFour** (2n×4g): local ranks form pairs `(0,1)` and `(2,3)`. (1) *Pair step*: pair members exchange the rows the other one needs and sum them, giving pair partials for two targets per rank (its own index and the index `l^2` in the other pair) on both nodes. (2) *Cross-pair step*: the pair partial for target `l^2` is read by rank `l^2`, which adds it to its own to obtain the complete node partial. (3) *Network step*: the node partial for the remote peer is sent over RDMA; the received partial is added to the local node partial. Chunks flow through four slots.

Explicit `MSCCLPP_NCCL_RS_NO_CUDAIPC=1` forbids the RS context from exporting or
opening peer GPU scratch. When IPC is unavailable, initialization collectively
selects a host path. Resources for other collectives on a shared handle remain
governed by their own initialization policy. RS does not reuse AllGather's
registered outputs, and users do not register ReduceScatter outputs. Single-node RS
establishes no IB connections.

## Single-node path details

The algorithms are summarized in "What each path does"; this section lists the rules that
are easy to get wrong. They are implemented in `reducescatter_ipc.cuh` and
`reducescatter_host.cuh`.

* **LocalRows scatter**: messages up to `LOCAL_DEVICE_FLAG_MAX_BYTES` (32 KiB) with the
  mapping prerequisites are scattered by the calling CTA (`liteRsSum` with one source)
  directly into the peers' scratch; otherwise `RsCopy` DMA is posted and awaited before
  the barrier. With B ≥ 64 KiB (not device-flag, not skip-self; `LOCAL_PARALLEL_COPY`,
  default on) each target's copy uses its own service stream (`RsCopy` stream = target).
* **Self row**: for non-device-flag messages with B ≤ 32 KiB, and always for TwoLocal
  and the host paths, the self row is read directly from the input; larger LocalRows
  messages include a self copy. Scratch is reused only after all consumers finished.
* **IPC ring step** `step`: send shard `(rank-step-1+P)%P`; after receiving, add local shard
  `(rank-step-2+P)%P`. The last step writes the output. The previous partial must finish
  sending before its mailbox is overwritten. Each chunk uses two mailboxes with
  independent ready/consume epochs, and tail chunks are preserved.
* **HostRing phases**: each rank owns `P-1` mapped-host rows per chunk. Phase 0 copies shard
  `me-1` into row 0; phase `k` (1 ≤ k ≤ P-2) writes `row[k] = shard(me-1-k) + prev.row[k-1]`;
  the final phase writes `shard(me) + prev.row[P-2]` to the output. Each phase is followed
  by a local barrier, and a final barrier lets the next chunk rewrite the rows. The ring
  area of the scratch stride holds three rows.
* **Chunking and slots**: IPC/host ring chunks default to 16 MiB; local rows/bulk default to
  2 MiB. Local scratch uses one chunk slot. The CPU reference's multi-slot / IPC-event
  mechanism is replaced by independent stream completion plus an epoch barrier; no
  cross-process IPC events are created.

## Two-node details

Every registered memory object is CPU pinned host memory; there is no GDR. Each local
rank has its own connection to the same-local-rank peer on the other node.

* **SmallHost / HostStaged**: the CPU service of rank `l` reduces only the inputs needed
  for target `l` on this node and target `l` on the peer, so all `l` together produce every
  node partial. The local partial uses a separate row so it does not overwrite source
  rows that other CPU readers have not read yet. CPU sum reuses CpuSwitch's runtime
  AVX-512 detection; `DISABLE_AVX512=1` forces scalar code.
* **Scratch visibility**: all remote loads in the pair / cross-pair steps target scratch that
  was mapped at initialization and whose staging has completed; peer original inputs are
  never accessed.
* **Direct partner copy**: by default T_bytes ≥ 1 MiB (always for 2n×2g) DMAs the odd/even
  target rows assigned to the partner straight into the partner's scratch. When disabled,
  they are first packed into the rank's own scratch and then pushed to the partner, as in
  the CPU pack + D2D path. 2D mode copies the odd/even rows in one batch with a `2*B`
  source pitch and `2*capacity` scratch pitch; the other parity is not sent.
* **Final-add variants**: writing the remote partial directly into a mapped-send slab,
  mapped-host final add, and split-final (produce the remote partial and start its D2H
  first, then let the same CTA compute the local partial). With async-final, ready remote
  contributions may be H2D'd ahead of time into separate scratch, overlapping with the
  CTA phases of later chunks; the final reduction never overwrites unread local in-place
  input. For a synchronous single chunk (not `recordAsync`), `CPU_FINAL_ADD` can D2H the
  local partial and do the final add on the CPU. Defaults are in the pipeline policy
  table below.

## Chunk, pipeline and tuning

`CHUNK_BYTES` defaults to a 2 MiB capacity cap. The effective chunk is chosen by
applying, in order:

1. An explicit `LAYOUT_CHUNK_BYTES` override, truncated to whole floats and bounded by capacity.
2. A 1 MiB cap for 2n×4g with B ≥ 2 MiB and for 2n×2g with T_bytes ≥ 8 MiB.
3. Eligible `MAPPED_HOST_SINGLE_CHUNK` may use the whole B when B ≤ 2 MiB.
4. Otherwise a 512 KiB cap for B ≤ 2 MiB, and a 1 MiB cap for larger B.

SmallHost also checks full-input/partial slot capacity (GPU scratch layouts differ from
the CPU ones, so capacity checks are the GPU's own bounds: a row must fit `chunkCapacity`). TwoRankSmall defaults to
512 KiB, not the 1 MiB threshold from older documents. Threshold comparisons strictly
distinguish `<` from `<=`. The default chunk for no-IPC multi-node paths is 256 KiB.
Actual tails keep float elements intact.

All variables below carry the prefix `MSCCLPP_NCCL_RS_`, are snapshotted at host
initialization, and are compared field by field collectively:

* `NO_CUDAIPC`, `P2P_RING`, `LOCAL_RING`, `LOCAL_RING_CHUNK_BYTES`, `LOCAL_DEVICE_FLAG_MAX_BYTES`.
* `NO_CUDAIPC_HOST_SMALL_BYTES`, `NO_CUDAIPC_HOST_READ`, `NO_CUDAIPC_BULK_CHUNK_BYTES`,
  `NO_CUDAIPC_DIRECT_RING`, `NO_CUDAIPC_DIRECT_RING_MIN_BYTES`, `NO_CUDAIPC_DIRECT_RING_CHUNK_BYTES`,
  `NO_CUDAIPC_CHUNK_BYTES`.
* `TWO_RANK_SMALL_HOST_BYTES`, `TWO_RANK_MAPPED_HOST`, `SMALL_HOST_FULL_BYTES`, `2N2G_HIER`.
* `CHUNK_BYTES`, `LAYOUT_CHUNK_BYTES`, `LOCAL_LEAD_CHUNKS`, `SHORT_LOCAL_LEAD_CHUNKS`, `LONG_LOCAL_LEAD_CHUNKS`.
* `LOCAL_PARALLEL_COPY`, `EAGER_RDMA_POST`.
* `DIRECT_PARTNER_COPY`, `DIRECT_PARTNER_COPY_2D`, `MAPPED_SEND_FINAL_REDUCE`,
  `MAPPED_HOST_SINGLE_CHUNK`, `HOST_READ_FINAL_ADD`, `SPLIT_FINAL_REDUCE`,
  `ASYNC_FINAL_ADD`, `CPU_FINAL_ADD`, `DISABLE_AVX512`.

The two-rank network uses five slots; the other hierarchy/host pipelines use four.

### Pipeline policy (mirrors the CPU per-layout defaults)

`recordAsync` is the CPU `recordAsyncD2h`: true for pipelined chunks and for the
2n×2g single chunk with B ≤ 512 KiB; false for a single 2n×4g chunk. Unless the
corresponding environment variable is set:

| Policy                                 | 2n×2g                                       | 2n×4g                                                                        |
| -------------------------------------- | -------------------------------------------- | ----------------------------------------------------------------------------- |
| Local lead (chunks ≤ 2 / 3–7 / ≥ 8) | 1 / 3 / 3                                    | 1 / 1 / 1                                                                     |
| Split-final                            | `recordAsync` and mapped send (B ≤ 1 MiB) | T_bytes ≥ 16 MiB, or `recordAsync` and B ≤ 1 MiB; never when B ≤ 512 KiB |
| Host-read final add                    | mapped and split-final                       | mapped, and (single chunk with chunk ≤ 512 KiB, or split-final)              |
| Async final add                        | never                                        | pipelined only (all sizes)                                                    |
| Eager RDMA post                        | pipelined and T_bytes ≥ 4 MiB               | pipelined and T_bytes ≥ 8 MiB                                                |
| Direct partner copy                    | always (2D rows)                             | T_bytes ≥ 1 MiB                                                              |
| CPU final add                          | sync single chunk only,`CPU_FINAL_ADD`     | same                                                                          |

Two-rank leads use the 1 / 3 / 3 classes (`SHORT`/`LOCAL`/`LONG_LOCAL_LEAD_CHUNKS`);
an explicit class variable overrides every layout, `0` meaning no lead. SmallHost and
HostStaged run with lead 0 (the CPU host path handles one chunk at a time). The lead is
bounded by `slots - 1`. With direct partner copy off (2n×4g, T_bytes < 1 MiB) the rank
first packs the partner's parity rows into its own scratch and then pushes them, like
the CPU pack kernel plus D2D push; packed rows (partner parity) and received rows (own
parity) never share a row index. Two-rank chunks never use the mapped send slab or
host-read final add outside TwoRankSmall, and their own shard is added directly from
the input.

Chunk issue order follows the CPU loops. Each chunk is *prepared* (local phases, D2H
staging recorded on an event), *posted* (event awaited, RDMA payload + ready signal) and
*finished* (wait remote ready, consume, ACK). Non-eager pipelines finish chunk `i-lead`
after preparing chunk `i` (post, wait, consume, ACK); eager pipelines only post chunk
`i-lead` and finish a chunk when its slot is recycled or at the end. A slot is reused
only after its previous chunk finished, so remote ACKs are ordered independently of
send completion.

The GPU has only one CTA; the multiple reduction kernels / parallel streams in the CPU
reference do not imply extra CTAs (the split-final D2H overlaps with the second CTA
phase instead of a second stream). The CPU's IPC-event / stream-memory-wait choices are
uniformly converted here into DMA completion plus system-scope phase/epoch
publication, without waiting for the stream that runs the user kernel to finish.

### Known deviations from the CPU reference

* 2n×2g with hierarchy disabled runs the GPU Generic path; the CPU falls back to
  `runSendRecvReduceScatter` (host NCCL send/recv), which cannot be called here.
* 1n×2g (TwoLocal) has no CPU reference; it extends the 1n×4g rows algorithm.
* P2pRing replaces NCCL send/recv with IPC DMA and is chunked at the ring chunk size.
* LocalRows small-message device-flag scatter is a CTA scatter followed by a service
  `RsBarrier`, not the CPU's single scatter kernel that also waits on device flags: the
  peer wait is a CPU-executed barrier task.
* Every single-node barrier is a FIFO round trip to the service (all-rank barrier),
  where the CPU paths use event / flag waits between GPU kernels.
* The CPU's default multi-HCA round-robin for 2n×2g messages ≤ 2 MiB depends on the
  first call's size, while GPU connections are created at initialization; only the
  locality-based NIC choice is used.
* Parallel split-final streams, IPC event/device-flag sync modes and
  `PAIR_CONNECTION_WRITE` are not modeled; local barriers are all-rank rather than
  peer-only, and remote reads of pair partials replace CPU cross-pair pushes.

## Primitive protocol, visibility and reclamation

Single-node paths post a sequence of `RsCopy` / `RsHostSum` / `RsBarrier` tasks and await
each ticket; their barrier epochs come from a persistent counter in
`LiteReduceScatterControl` (CTA thread 0 only). Two-node paths submit one
`ReduceScatter` FIFO task per call. The service publishes the source
pointers, destination, count and source count in `LiteReduceScatterControl`, then
releases `requested=phase`. After the CTA acquire observes the phase, the CTA copies/sums
together; all writing threads system-fence and hit a CTA barrier, and finally one thread
releases `completed=phase`. The final FIFO completion signals that the whole invocation
is done. These numbers are independent: FIFO ticket, CTA phase, RS chunk epoch, and the
existing AG/generic epochs.

DMA progresses on nonblocking service streams, and ready is published only after a CUDA
event confirms real completion. The NIC payload write and the ready signal use the same
connection; the signal's source address remains stable until flush. The receiver H2Ds /
CTA-consumes only after reading ready, then sends an ACK. Slot reuse requires both
every local reader's done and the remote ACK; a send completion cannot substitute for
the ACK. The fixed capacity-strided row layout keeps old slots' credits across
consecutive calls with different B, chunk and path.

Initialization allocates SHM, CUDA scratch, streams/events, IPC mappings, registrations
and QPs; nothing is allocated during a call. Service waits check stop/FIFO errors;
errors propagate to the node's control, and a CTA timeout poisons the FIFO. A failed
handle cannot be used further. Before destruction the user synchronizes the calling
stream, then the service is stopped and joined, DMA is drained, QPs/registrations and
IPC mappings are closed, and finally scratch/SHM are freed.

## Validation boundary

This document describes the implementation and protocol. It does not claim that every
GPU/IB environment has been validated or that CPU-version performance has been reached.
This machine has no CUDA toolchain, so real CUDA compilation, GPU visibility, RDMA ACK
behavior and performance still need to be validated on target machines. The local checks
completed and the suggested benchmark matrix are in
[docs/reducescatter-validation.md](../docs/reducescatter-validation.md).
