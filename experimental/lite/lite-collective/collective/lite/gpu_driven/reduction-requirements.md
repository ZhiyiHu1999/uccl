# GPU-driven ReduceScatter and AllReduce requirements

This file supplements the AllGather requirements in [AGENTS.md](AGENTS.md).
These are implementation targets, not a claim that all optimized paths already
exist. Follow the same primitive layer, host initialization, single-CTA execution,
and intra-/inter-node visibility and resource-reuse contracts.

Use the current source for dispatch conditions when older design documents
disagree. All paths below run inside the caller's one participating CTA per rank.
Adapt GPU kernels into block routines and submit CPU DMA/RDMA/reduction work
through the task FIFO. Do not invoke host collectives, NCCL send/recv, or child
kernels from a device invocation. CPU workers must not launch additional reduction
kernels to bypass the one-CTA requirement.

## Shared conventions

* `R = nranks = N * P`, where `N` is the node count and `P` is the equal number
  of ranks per node; `nodeId = rank / P`, `localRank = rank % P`.
* KiB/MiB/GiB are binary units. Sizes below describe one rank's input or output,
  not the aggregate input across all ranks.
* Optimized host references generally implement `ncclFloat32` / `ncclSum`.
  Select these optimizations only for that combination. Preserve the existing
  device API's supported template types and `liteReduceSum/Min/Max` through a
  compatible generic path; do not silently apply float addition to another type
  or operation. New optimized type/op combinations need explicit validation.
* Keep initialization and selection collective-consistent. Resolve environment
  settings, topology, IPC/mapping capabilities, scratch capacity, and network
  resources before publishing work. Unsupported configurations return an explicit
  error or use an agreed native generic path; never silently invoke real NCCL.
* Preserve the current device API's argument-validation behavior, including
  zero-count rejection unless that contract is deliberately extended. Check
  multiplication/offset overflow, element alignment, and capacity before use.
* `maxBytesPerRank` currently bounds the complete source bytes of a reduction:
  the full input for both ReduceScatter and AllReduce. A small output shard does
  not make an oversized ReduceScatter input valid. Chunking does not silently
  expand the handle contract.

### ReduceScatter

Host references are [single-node.cu](../../../nccl/ReduceScatter/single-node.cu)
and [multi-node.cu](../../../nccl/ReduceScatter/multi-node.cu), starting at
`runLiteInterReduceScatter()`. The separate `runSendRecvReduceScatter()` in
[native_collectives.cu](../../../nccl/native_collectives.cu) is also used by
host AllReduce; do not confuse its dispatch with the standalone ReduceScatter
entry point.

Symbols:

* `C = recvCount`: output elements per rank; `s = sizeof(element)`.
* `B = C * s`: output shard bytes per rank (`bytesPerRank` in the host reference).
* `T = R * B`: complete input bytes per rank (`fullBytes`/`messageBytes`).
* Rank `r` receives `reduce(src[k][r*C + i], k = 0..R-1)` for `i = 0..C-1`.
* Standard in-place layout is `dst = src + rank * C` in element units.

#### Case 1: one rank only

* Copy the input to output using the caller's CTA; skip copying when in-place.
* No reduction, peer synchronization, host staging, or network exchange is needed.

#### Case 2: one node only, multiple ranks

* For `1n * 4g`, preserve the actual selection order in
  `runLiteInterReduceScatter()`:
  * Explicit no-CUDA-IPC mode takes priority; its subpaths are described below.
  * `useP2pRingForLocalFour(B)` selects the send/recv ring when explicitly forced
    by `MSCCLPP_NCCL_RS_P2P_RING`, or when `256 KiB <= B <= 2 MiB`.
    Adapt the data order of `runP2pRingReduceScatter()` to device primitives;
    do not retain its host `ncclSend/ncclRecv` calls.
  * Otherwise follow `runLocalFourRankReduceScatter()`. Its CUDA IPC ring is
    eligible for `B >= 1 MiB`, with `MSCCLPP_NCCL_RS_LOCAL_RING` controlling the
    mode and default selection at `B >= 2 MiB`. The earlier P2P-ring branch
    still takes priority at equality and throughout its interval.
  * Remaining calls use the local scratch-row copy/reduction path. Preserve
    `MSCCLPP_NCCL_RS_LOCAL_DEVICE_FLAG_MAX_BYTES` (default 32 KiB), its mapping
    prerequisites, and the reference's self-row and copy/reduction ordering.
* CUDA IPC ring: follow `runLocalFourRankRingReduceScatter()`; circulate shards
  through the next/previous local ranks, reduce the corresponding local input
  at each step, and leave each final shard on its owning global rank.
  Use three steps for four ranks and bounded scratch slots. Default chunk size
  is 16 MiB per output shard, controlled by
  `MSCCLPP_NCCL_RS_LOCAL_RING_CHUNK_BYTES`; preserve partial final chunks.
* For `1n * 2g`, add a two-rank specialization of the same ownership rule:
  exchange the peer-owned shard, then reduce the incoming contribution with
  the local owned shard. Use CUDA IPC scratch when available and host staging
  otherwise. This is a required extension: the standalone host dispatcher does
  not accept `1n * 2g`; do not pretend the four-rank helpers already support it.
* Local-only paths must not require IB. Do not select NVLink/NVLS algorithms.

##### Case 2 SHM path details: no-CUDA-IPC mode

* Preserve explicit `MSCCLPP_NCCL_RS_NO_CUDAIPC=1` behavior: no peer GPU mappings
  or IPC events. For absent IPC capability, select a supported host path
  consistently during setup rather than attempting invalid peer accesses.
* For `1n * 4g`, follow the priority in `runLiteInterReduceScatter()`:
  * `B <= 64 KiB` by default: `runNoCudaIpcHostReduceScatter()` stages source
    rows to pinned SHM, CPU-reduces the target shard, and copies the result back.
  * Otherwise, with direct ring enabled and `B >= 1 MiB` by default:
    `runNoCudaIpcDirectRingSingleNodeReduceScatter()` passes partial shards
    through mapped host mailboxes using GPU copy/add and readiness/consumption
    flags. Default ring chunks are 16 MiB.
  * Otherwise use `runNoCudaIpcHostReadSingleNodeReduceScatter()` when enabled
    and eligible: stage the peer-owned rows to source-local SHM, read peer
    contributions from mapped host memory, and read self directly from source.
    Default bulk chunks are 2 MiB.
  * If host-read is unavailable, adapt
    `runNoCudaIpcBulkSingleNodeReduceScatter()` using DMA scratch, then the
    supported CPU host path if necessary.
* Preserve the `MSCCLPP_NCCL_RS_NO_CUDAIPC_{HOST_SMALL_BYTES,HOST_READ,
  BULK_CHUNK_BYTES,DIRECT_RING,DIRECT_RING_MIN_BYTES,DIRECT_RING_CHUNK_BYTES}`
  tuning semantics. Adapt stream-memory waits into the device/service protocol;
  never queue a dependency on completion of the waiting caller kernel.
* Generalize shard ownership and peer count explicitly for `1n * 2g`; do not
  reuse four-rank indexing unchanged.

#### Case 3: two nodes, one rank per node (`2n * 1g`)

* Follow `runTwoRankReduceScatter()`. Exchange only the peer-owned input shard;
  the output is the incoming contribution plus the source's locally owned shard.
* For `B <= 512 KiB` by default and sufficient host-slot capacity, follow
  `runTwoRankSmallHostReduceScatter()`:
  * With mapped receive/control resources, GPU-copy the peer-owned shard to
    host staging, RDMA-exchange it, then GPU-read and add the incoming shard.
  * Preserve the DMA/CPU-final alternative when mapped access is unavailable.
  * `MSCCLPP_NCCL_RS_TWO_RANK_SMALL_HOST_BYTES` controls this limit. The current
    source default is 512 KiB, despite the older design document saying 1 MiB.
* Otherwise use the chunked D2H -> RDMA -> H2D -> GPU-add path, with five
  pipeline slots and deferred ACK until the incoming slot's consumer finishes.
  Follow `runTwoRankPipelinedChunks()` when more than one chunk is needed.
* The default chunk-capacity ceiling is 2 MiB, but the actual chunk size comes
  from `effectiveChunkBytesForLayout()`; do not treat the capacity as an
  unconditional 2 MiB transfer size. Preserve the shared chunk policy below.
* Never H2D the remote contribution over an unread local in-place source shard.
  Use scratch or finish consuming/preserving that local contribution first.

#### Case 4: two nodes, two ranks per node (`2n * 2g`)

* For `T < 128 KiB` by default, first use `runSmallHostReduceScatter()` if the
  complete input and partial slots fit. Stage every local input, CPU-reduce
  local-node contributions to the local-owned and remote-owned shards, exchange
  remote-owned partials, combine, and deliver only each rank's output shard.
* At `T >= 128 KiB`, or if the small path is ineligible, follow
  `runTwoNodeTwoGpuHierReduceScatter()` with the default
  `MSCCLPP_NCCL_RS_2N2G_HIER=1`:
  * Exchange needed rows with the local peer through IPC scratch.
  * Produce one local-owned and one matching remote-owned node partial per rank.
  * Stage and exchange only the remote-owned partial with the matching remote
    local rank; reduce the incoming partial with the local-owned partial.
* Use four pipeline slots. Preserve the single-chunk path and
  `runTwoNodeTwoGpuPipelinedChunks()` for multi-chunk work. Default local lead
  is one unless the corresponding lead tuning variable is set.
* For `T >= 8 MiB`, the hierarchy uses chunks capped at 1 MiB unless
  `MSCCLPP_NCCL_RS_LAYOUT_CHUNK_BYTES` overrides it. Smaller calls follow the
  shared effective-chunk policy.
* If hierarchy is disabled, adapt the native generic algorithm behind
  `runSendRecvReduceScatter()` to the same device primitives; its host API
  calls are not an allowed device fallback.

#### Case 5: two nodes, four ranks per node (`2n * 4g`)

* For `T < 512 KiB` by default, first use the same small host-slot algorithm
  as Case 4, with four local contributors and sufficient slot capacity.
  Keep final CPU output separate from source rows still needed by other readers.
* Otherwise follow `runChunk()` / `runPipelinedChunks()`:
  * Form local pairs `(0,1)` and `(2,3)`, exchange the partner's required rows,
    reduce within each pair, exchange cross-pair partials, and produce the
    local-owned and matching remote-owned node partials.
  * Preserve direct partner-row copying at `T >= 1 MiB` by default, with
    `MSCCLPP_NCCL_RS_DIRECT_PARTNER_COPY` and its 2D variant controlling it.
    Do not replace staged scratch exchange with unvalidated direct peer-input
    loads; pair topology and IPC accessibility must be established at setup.
  * Send the remote-owned partial through pinned registered host memory to the
    matching remote rank. Combine its incoming partial with the local partial
    and publish only the owned output shard.
* Use four compact pipeline slots. Preserve the single-chunk alternative,
  tiered chunk sizing, and bounded local lead in the reference. Distinguish
  partner-copy, cross-pair, remote-send, and final-add completion per slot.
* Preserve eligible mapped-send, host-read final-add, and split-final-reduce
  policies. Adapt split GPU phases to the same CTA so CPU/NIC progress can
  overlap local reduction; do not launch remote/local/final-add child kernels.
* `MSCCLPP_NCCL_RS_IPC_EVENT_SYNC` controls host reference synchronization;
  default event sync is disabled for `1 MiB <= T <= 4 MiB`. Retain ordering
  through epochs and actual copy completion in the GPU-driven implementation.
  Large-message async final-add begins at `B >= 32 MiB` in the reference;
  reproduce useful DMA overlap within the one-CTA limit.

#### Shared ReduceScatter chunking and transport requirements

* Preserve `MSCCLPP_NCCL_RS_SMALL_HOST_FULL_BYTES`; without an override the
  strict small threshold is 128 KiB for `2n * 2g` and 512 KiB for `2n * 4g`.
* Follow `effectiveChunkBytesForLayout()` and `effectiveChunkBytes()`:
  * An explicit `MSCCLPP_NCCL_RS_LAYOUT_CHUNK_BYTES` override is capped by the
    allocated chunk capacity and rounded to complete float elements.
  * `2n * 4g` with `B >= 2 MiB` uses a 1 MiB cap before the generic policy.
  * Otherwise eligible mapped-host single-chunk mode can use `B` for
    `B <= 2 MiB`, bounded by capacity and its mapped-send/host-read predicates.
  * Remaining `B <= 2 MiB` calls use a 512 KiB cap; larger calls use a 1 MiB
    cap. Apply Case 4's additional large-message rule.
  * `MSCCLPP_NCCL_RS_CHUNK_BYTES` controls the capacity ceiling; always clamp
    effective sizes to actual scratch and host allocation.
* No-CUDA-IPC multi-node mode uses host-staged CPU reduction, with default
  256 KiB chunks controlled by `MSCCLPP_NCCL_RS_NO_CUDAIPC_CHUNK_BYTES`.
  Preserve runtime AVX-512 availability checking and scalar fallback, including
  `MSCCLPP_NCCL_RS_DISABLE_AVX512`; do not require AVX-512 globally.
* RDMA payload and ready publication must have ordered visibility. Local send
  completion, remote arrival, final reduction, and GPU consumption are separate
  events. Deferred ACK must be flushed safely when switching message regimes.
* One/two-node topologies beyond these specializations use the compatible
  generic device path when supported. Three or more nodes require an explicit
  extension to setup, reduction routing, rank arrays, connections, and validation;
  the two-node matching-peer algorithm must not silently omit other nodes.

### AllReduce

Host references are in [native_collectives.cu](../../../nccl/native_collectives.cu),
starting at `runSendRecvAllReduce()`. Single-node host execution also uses the
registered collective selector; those host launches are algorithm references,
not device-callable implementations.

Symbols:

* `C = count`, `s = sizeof(element)`, `B = C * s`: complete input and output
  bytes per rank. Rank `r` receives `reduce(src[k][i], k = 0..R-1)` for every `i`.
* For equal-shard RS+AG, require `C % R == 0`; `S = B / R` is the reduced shard
  size. ReduceScatter uses `B_RS = S`, `T_RS = B`; AllGather uses
  `B_AG = S`, `T_AG = B`. Never feed `B` to a per-shard threshold.
* In-place AllReduce means `src == dst` for the complete tensor.

#### Case 1: one rank only

* Copy directly with the caller CTA; no work beyond required block completion
  when in-place. No host or network work is required.

#### Case 2: one node only, multiple ranks (`1n * 2g`, `1n * 4g`)

* Provide a native single-CTA local reduction path using IPC scratch or mapped
  SHM and the shared staging primitives. Preserve a compatible generic path
  for small tensors, irregular counts, and supported non-float/sum operations.
* For divisible counts, compose the ReduceScatter requirements above with the
  AllGather requirements in `AGENTS.md`. Apply each stage's own eligibility,
  including explicit host-AllGather enabling and IPC policy. If an optimized
  AllGather stage is unavailable, select a valid generic device path before
  starting; do not assume every local size is covered by the optimized paths.
* Do not invent a size crossover from the host selector or claim RS+AG is always
  faster. This composition is an implementation target; preserve a correct
  baseline and establish GPU-driven selection thresholds with measurements.
* Avoid NVLink/NVLS dependencies and preserve input needed by later RS steps
  before writing in-place reduced output.

#### Case 3: two nodes, one rank per node (`2n * 1g`)

* First adapt `runTwoRankRingSimpleAllReduce2Node()` for float/sum with
  `B >= 64 MiB` by default. Preserve the existing explicit-force semantics of
  `MSCCLPP_NCCL_2RANK_RING_ALLREDUCE` for smaller messages.
* Use two logical channels with independent connections, staging/control slots,
  and progress. Default channel chunks are 4 MiB, controlled by
  `MSCCLPP_NCCL_2RANK_RING_CHUNK_BYTES`.
* Follow the reference's channel partition and partial-tail handling. For each
  channel's portion, send peer-owned input, reduce incoming own data with the
  local contribution, then exchange reduced pieces to complete both outputs.
  Stage all network traffic through registered host memory.
* Keep ring epochs separate from ReduceScatter pair epochs. Allocate streams,
  events, and workers once at initialization, instead of retaining per-call
  host-reference allocations or thread creation.
* Both channels share the caller's single CTA; two channels do not authorize
  two GPU blocks. CPU DMA/network progress may overlap the CTA's reduction.
* For smaller/ineligible calls use compatible device reduction or equal-shard
  RS+AG. Irregular counts need the generic full-tensor path below; never truncate
  the tensor to make `C` divisible by `R`.

#### Case 4: two nodes, two ranks per node (`2n * 2g`)

* First follow `runSmallMappedAllReduce2Node()` for float/sum with
  `0 < B <= 64 KiB` and valid host/control mappings:
  * Each rank GPU-stages its complete input and publishes readiness.
  * A CPU node leader reduces all local contributions, exchanges one node
    partial through RDMA, and CPU-reduces the two node partials.
  * Every local rank waits for final readiness and copies the complete result
    from mapped host memory. Reuse waits for all local readers.
* Otherwise use RS+AG when `C % R == 0` and both stages are eligible, with
  `S = B / 4`. This topology does not select the four-local-rank two-leader path.
* Use the native generic reduction for irregular counts, missing mappings, or
  unsupported optimized type/op combinations.

#### Case 5: two nodes, four ranks per node (`2n * 4g`)

* First use the same mapped small AllReduce for `0 < B <= 64 KiB`.
* Next follow `runSmallTwoLeaderAllReduce2Node()` for float/sum with
  `64 KiB < B <= 128 KiB` and even `C`:
  * Stage each local rank's input. Local ranks 0 and 2 act as CPU part leaders,
    each reducing one tensor half across all four local ranks.
  * Exchange each half's node partial with its corresponding remote leader,
    reduce both node contributions, and publish both final halves.
  * Every local rank consumes both halves to assemble its complete output.
  * The source constant `kNativeSmallAllReduceMaxBytes` is 128 KiB. Do not
    extend this branch to 256 KiB based on the older design document.
* Otherwise use RS+AG for divisible `C`, with `S = B / 8`, then a compatible
  generic path if the optimized composition is unavailable.
* The host AllReduce reference uses `runSendRecvReduceScatter()`, which tries
  `runGpuLocalReduceScatter2Node()`, `runChunkedGpuLocalReduceScatter2Node()`
  (including `runPipelinedNumaPairReduceScatter2Node()`), then host staging.
  It does not call `runLiteInterReduceScatter()`. Reusing the standalone
  ReduceScatter requirements above is the shared GPU-driven target; document
  that routing choice rather than claiming identical host dispatch thresholds.

#### Shared AllReduce composition and generic-path requirements

* ReduceScatter must finish producing a shard before AllGather publishes it.
  Store the shard at its global-rank output offset or in dedicated scratch;
  AllGather must restore the complete tensor in global-rank order.
* For in-place RS+AG, protect every original input region still needed by any
  rank before AllGather overwrites it. A local shard becoming ready alone is
  insufficient. Initially use an explicit all-rank RS-phase completion boundary;
  chunk overlap requires equivalent per-region reader-lifetime tracking.
* Keep RS and AG phase/epoch identities distinct even when sharing FIFOs or
  buffers. AG completion includes all output copies; release a shared slot only
  after all RS/AG readers, network operations, and remote credits complete.
* Provide a full-tensor chunked reduction for `C % R != 0`, including `C < R`.
  For two nodes, adapt the data dependencies of
  `runHierarchicalAllReduce2Node()`: reduce all local contributions to a chunk,
  exchange node partials, then combine them into every rank's output. Replace
  host send/recv and synchronization with device/service primitives. The
  existing generic device staging/reduction is also a valid compatibility path.
* Do not independently switch algorithms after peers have published work.
  Ordinary transport/DMA/reduction failures propagate to all waiting participants;
  they are not eligibility failures to be hidden by a retry on another protocol.
* More than two nodes remain an extension requiring complete participation of
  every node; do not reuse a two-node pair protocol as a general AllReduce.

## GPU-driven adaptation and acceptance

* Prepare IPC mappings, mapped/registered host slabs, queues, topology-specific
  scratch, NIC connections, completion state, and service streams during host
  initialization. The current handle supports one rank, or 2-8 total ranks on
  one/two balanced nodes; broaden this only with matching setup and validation.
* GPU publishes work only after its payload is system-visible. CPU publishes
  copy/reduction/network readiness only after actual completion. A block barrier,
  volatile flag, or FIFO head advance does not replace these completion rules.
* CPU progress must continue sends, receives, and ACKs while the caller CTA waits.
  Service streams cannot wait for the caller kernel to exit. Signal failures to
  GPU waiters; stop/join workers before freeing resources.
* Validate both sides and equality of every threshold, single/multiple chunks,
  partial tails, element-aligned non-vector-aligned buffers, in-place/out-of-place
  operation, mixed-size repeated invocations, slot wraparound, and RS/AG phase
  transitions. Confirm every contributor is reduced exactly once and every
  output shard belongs to the correct global rank.
* Validate supported types/ops and floating-point results with suitable numerical
  tolerances; a changed reduction order need not match bit-for-bit. Exercise IPC
  enabled/disabled, mapped/DMA paths, and insufficient-capacity rejection.
* Put correctness preflight in `device_collectives_bench.cu`; do not add separate
  correctness executables. Compare against native NCCL under the same no-GDR,
  local-P2P policy and one-SM constraint on `1n * 2g`, `1n * 4g`, `2n * 1g`,
  `2n * 2g`, and `2n * 4g`. Verify the constraint rather than merely setting an
  environment variable. Keep commands, implementation status, and performance
  evidence in `docs/`; do not claim host-path parity before measurement.
