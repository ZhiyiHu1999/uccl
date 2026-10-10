# GPU-driven AllGather

This document defines AllGather requirements and optimization paths. Update this document when AllGather selection or algorithms change. Benchmark instructions and results remain in `docs/`.

## Implementation contract and source navigation

| Component                       | Source / entry point                                                                                                                                            | Implementation contract                                                                                                                                                                                                                                                                                                            |
| ------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Selection and dispatch          | [allgather_plan.hpp](../allgather_plan.hpp); `liteAllGatherBlock()` in [gpu_collectives.cuh](../gpu_collectives.cuh)                                                | Preserve the device-callable API. One participating CTA per rank, except HostCooperative's optional cooperative grid; benchmark with one block.                                                                                                                                                                                    |
| HostMapped                      | [allgather_host_mapped.cuh](../allgather_host_mapped.cuh); `liteAllGatherHostMappedBlock()`                                                                      | One epoch/slot per invocation; pack current-size rank rows and copy the complete output including self; publish done after copying.                                                                                                                                                                                                |
| HostCooperative                 | [allgather_host_cooperative.cuh](../allgather_host_cooperative.cuh); `liteAllGatherHostCooperativeBlock()`                                                       | Same whole-call epoch and packed layout; synchronize the participating block or cooperative grid before ready/done publication.                                                                                                                                                                                                    |
| HostDma                         | [allgather_host_dma.cuh](../allgather_host_dma.cuh); `liteAllGatherHostDmaBlock()`; [service.hpp](../service.hpp)                                                   | One epoch/slot and one whole-call `HostAllGather` task; per-chunk stream-ordered readiness, batched left/right H2D, done after all copies.                                                                                                                                                                                       |
| IpcRing                         | [allgather_ipc.cuh](../allgather_ipc.cuh); `liteAllGatherIpcRingBlock()`                                                                                         | One epoch,`nranks - 1` full-block DMA pushes into next rank's registered output; first source is input, later sources are received rows. No generic chunk/SM-scratch loop.                                                                                                                                                       |
| IPC registration and completion | [ipc_output_registration.hpp](../ipc_output_registration.hpp); `mscclppRegisterDeviceCollectiveIpcOutput()`; [service.hpp](../service.hpp)                          | Register cudaMalloc output collectively before use, with equal sizes/relative offsets; outside benchmark timing. Finish users before replacement/unregistration; unregister before freeing. Nonblocking `IpcCopySelf`/`IpcPush` completes after CUDA events; admission/final barriers protect reuse.                           |
| OrderedSmall                    | [allgather_ordered_small.cuh](../allgather_ordered_small.cuh); `executeOrderedSmallSchedule()` in [network_service.cu](../network_service.cu)                       | `LiteNetworkControl` coordinates CTA packing/receive phases; preserve their distinct thread counts and early self-copy in parallel/compact one-rank cases. Unmapped payload uses ordered DMA before small fallback. Payload-slot flags require exact epoch equality; monotonic counters use `>=`; retain error/timeout checks. |
| SmallFallback                   | [allgather_small_fallback.cuh](../allgather_small_fallback.cuh); `executeSmallFallbackSchedule()`                                                                | Submit a whole invocation to the GPU-private network schedule, preserving packing, receive and slot-reuse rules.                                                                                                                                                                                                                   |
| OneRankPipeline                 | [allgather_one_rank_pipeline.cuh](../allgather_one_rank_pipeline.cuh); `executeSingleSlabSchedule()`                                                             | Submit a whole invocation; preserve the whole-message epoch and specialized chunk pipeline, rather than generic per-chunk Stage/Gather tasks.                                                                                                                                                                                      |
| SingleSlab                      | [allgather_single_slab.cuh](../allgather_single_slab.cuh); `executeSingleSlabSchedule()`                                                                         | Preserve DMA self-copy, receive batches, ordered QP submissions and ACK/slot reuse.                                                                                                                                                                                                                                                |
| NumaSplit                       | [allgather_numa_split.cuh](../allgather_numa_split.cuh); `executeNumaSchedule()`                                                                                 | Preserve independent per-group epochs, layouts and slot lifetimes.                                                                                                                                                                                                                                                                 |
| Network service boundary        | [network_service.hpp](../network_service.hpp), [network_service.cu](../network_service.cu), [service.hpp](../service.hpp)                                                | GPU-private schedules may perform bootstrap reuse barriers; never call native/public host collectives, launch child kernels or wait for the caller kernel. FIFO completion covers transport output and CTA SM phases; polling observes aborts.                                                                                     |
| Initialization and ownership    | [host_context.hpp](../host_context.hpp); [host_staging_buffer.hpp](../host_staging_buffer.hpp)                                                                        | Prepare contexts, mappings, QPs, streams and events during setup. Keep epochs/layouts independent of native host calls and the legacy reduction proxy; FIFO tickets are not algorithm epochs. Preserve dynamic slot counts and explicit move/resource ownership.                                                                   |
| CPU reference isolation         | [allgather_intranode.cu](../../allgather_intranode.cu), [allgather_multinode.cu](../../allgather_multinode.cu), [cpu_staging_channel.hpp](../../cpu_staging_channel.hpp) | References, not GPU-only edit targets. Review shared CpuSwitch/NodeExchangeBuffer changes across both execution paths. Design targets do not by themselves establish implementation completeness or validation.                                                                                                                    |
| Notation                        | `B`, `N`, `P`, `T`                                                                                                                                      | `B`: input bytes per rank; `N`: nodes; `P`: ranks per node; `N = nranks / P`; `T = B × nranks`: output bytes per rank. KiB/MiB/GiB are binary units; network block size is distinct from `B` and `T`.                                                                                                               |

## Backend preference contract

* `UCCL_GPU_DRIVEN_BACKEND=host` uses host memory on one node and host RDMA on two nodes.
* Unset/empty `UCCL_GPU_DRIVEN_BACKEND` or `cuda_ipc` prefers IPC for eligible single-node AllGather calls; otherwise use host memory. Two-node initialization always resolves to host RDMA.
* `MSCCLPP_NCCL_HOST_ALLGATHER` does not gate GPU-driven AllGather or override IPC preference. Native CPU-driven semantics are unchanged.
* Peer-access eligibility is agreed by all ranks during setup. Invalid arguments, unsupported topology/capture, resource allocation errors, and in-flight transport failures remain errors, not independent protocol switches.

## Intranode

### Copy — one rank

* D2D copy directly.
* No other operation when in-place.

### IpcRing — CUDA IPC ring

* Similar algorithm as `runIntraNodeCudaIpcAllGather()` in `allgather_intranode.cu`.
* This path is choosen when all below conditions are met:
  * CUDA IPC is preferred (unset/empty backend or `cuda_ipc`), and every GPU pair supports peer access;
  * ~~`hasIB=true`;~~  (Do not implement this rule, it is a bug in host-driven case)
  * IPC event sync is enabled；
  * `T ≥ 8 MiB`;
  * No CUDA Graph capture.

### Host memory selection

* Explicit `host` backend and fallback for ineligible IPC calls. Initialize host resources before returning an IPC-preferred handle; select the protocol before publishing work, never after an operation fails.
* Similar algorithm as `runIntraNodeShmAllGather()` in `allgather_intranode.cu`.
  * Implement internel paths [`hostAllGatherMappedKernel()`](../allgather_intranode.cu), [`hostAllGatherCoopKernel()`](../allgather_intranode.cu), and [`runIntraNodeShmAllGather()` ](../allgather_intranode.cu)as in host-driven case (selection conditions are all kept, `hostAllGatherCoopKernel()` should also be kept even if we use only one CTA for GPU-driven collectives).
  * Chunk size also follow the logics in host-driven collectives.
* This path is chosen when all below conditions are met:
  * `host` was explicitly requested, or IPC capability/event-sync/size eligibility is unmet.
  * Bytes fit the initialized handle capacity. GPU-driven fallback ignores the native host enable/minimum/1-GiB admission gates; host subpath tuning remains supported.
  * No CUDA Graph capture

### HostMapped, HostCooperative and HostDma

* Preserve selection priority in `runIntraNodeShmAllGather()`:

#### HostMapped

* `hostAllGatherMappedKernel()` logic: `T <= 4 KiB`, one chunk, `B` divisible by 8, 8-byte-aligned input/output, and valid device mappings for slab/control.

#### HostCooperative

* `hostAllGatherCoopKernel()` logic: the first branch did not match, `T <= 512 KiB`, the same alignment/mapping/single-chunk requirements, and cooperative-launch capability. Distribute both copies over a cooperative grid with grid-wide barriers. A single-block caller uses the same phases with block barriers; benchmark with one block to keep the one-SM budget.

#### HostDma

* DMA pipeline: remaining eligible calls, including unaligned or unmapped small inputs. Stage each input chunk D2H, copy self directly, and consume peer chunks through H2D after readiness publication. Preserve the separate left/right peer-range service streams where useful.
* Default per-rank chunk size: `B` when `B <= 1 MiB`; 1 MiB when `1 MiB < B <= 32 MiB`; 4 MiB when `B > 32 MiB`.
* Preserve the tuning semantics of `MSCCLPP_NCCL_HOST_ALLGATHER_{KERNEL_MAX_BYTES, COOP_MAX_BYTES, CHUNK_BYTES,MAP_SLAB,SELF_KERNEL}`. `MAP_SLAB=0` disables mapped-copy branches. The host SHM implementation supports 2-8 ranks.

## Internode

### OrderedSmall

* Device entry: `liteAllGatherOrderedSmallBlock()` in `allgather_ordered_small.cuh`.
* Service schedule: `executeOrderedSmallSchedule()` in `network_service.cu`.
* CPU reference: `runSmallOrdered()` in `../allgather_multinode.cu`.
* Select before other internode paths within the small-message threshold. The
  topology sections below specify the distinct packing, receiving, DMA and
  thread-count requirements for each ranks-per-node configuration. Exact-epoch
  slot polling and the CTA/service handshake are specified in the implementation
  contract above.

### OneRankPipeline

* Device entry: `liteAllGatherOneRankPipelineBlock()` in `allgather_one_rank_pipeline.cuh`.
* CPU reference: `runSingleSlab()` -> `runOneRankChunkPipeline()`.
* Applies to two nodes with one rank per node and `1 MiB <= B <= 1 GiB`.
  Use 512 KiB chunks, a send window of one, overlapping D2H/RDMA/H2D, and
  direct self-copy (skipped in-place), as detailed below.

### Topology: two nodes, one rank per node (`2n * 1g`)

* Small ordered-slot AllGather: follow `runSmallOrdered()` in `../allgather_multinode.cu` when `T < 2 MiB` (`B < 1 MiB`).
  * Stage in final global-rank order, except for the compact-segment variant below. The CPU proxy posts payload and readiness writes with the required QP ordering; the GPU consumes only published data.
  * When `T < 64 KiB` and both slab/control are device-mapped, preserve the register-copy specializations:
    * `T == 128 bytes`: `oneRankTinyRegisterCopyKernel()` logic, with one active copying thread.
    * `T == 256 bytes`: `oneRankRegisterCopyKernel()` logic.
    * Other sizes: `oneRankCompactRegisterCopyKernel()` logic, with a compact data-plus-flag segment per source rank. The reference uses 256 threads for `T == 16 KiB` or `32 KiB`, otherwise 128; adapt active-thread participation to the caller's block.
  * If the mapped-register branch does not match, use DMA staging. For in-place output or `T >= 128 KiB`, except `T == 32 KiB`, follow `copyOneRankPerNodeChunkToOutput()`: direct local copy (skip in-place) and remote H2D only. Otherwise copy the complete ordered host slot to output.
* Large-message pipeline: follow `runSingleSlab()` -> `runOneRankChunkPipeline()` when `1 MiB <= B <= 1 GiB`.
  * Use 512 KiB chunks and a send window of one. Queue D2H work; send each chunk after its DMA completion and queue remote H2D as incoming readiness advances. Retain D2H/RDMA/H2D overlap.
  * Copy self directly or skip in-place. Distinguish chunk readiness from whole-slot completion and remote ACK.
* Otherwise use generic single-slab exchange, including `B > 1 GiB` if handle capacity supports it, or eligible small calls whose optimized paths are unavailable.
* Do not select `runOneRankGpuDirect()` for this no-GDR target. The host reference disables it with `kEnableOneRankGpuDirect=false`; its dormant dispatch interval is `1 MiB <= T < 2 MiB`.

### Topology: two nodes, two ranks per node (`2n * 2g`)

* Small ordered-slot AllGather: follow `runSmallOrdered()` when `T < 128 KiB` (`B < 32 KiB`).
  * `T <= 256 bytes` with slab/control mappings: use `multiRankTinyPackKernel()` logic.
  * `512 bytes <= T <= 4 KiB` with slab/control mappings: use `multiRankRegisterPackKernel()` logic.
  * Otherwise use DMA staging; do not fill the `(256, 512)` gap with register packing.
  * Independently select `multiRankHostRecvKernel()` logic when `T <= 4 KiB` and the send slab is device-mapped; otherwise use complete-slot H2D. DMA packing can therefore coexist with SM-based receiving. Validate all control pointers accessed by the adapted routine.
  * Each local rank publishes staged input. One node leader/proxy exchanges the contiguous node block; both local ranks consume the complete rank-ordered result.
* For `T >= 128 KiB`, or after eligible small-path fallback, use `runSingleSlab()` -> `exchangeGroupChunk()` with 512 KiB per-rank chunks.
* Preserve the host dispatch exclusion of NUMA split for `P == 2`, even when multiple NIC groups are detected.

### Topology: two nodes, more than two ranks per node (especially `2n * 4g`)

* When `T < 128 KiB`, first use `runSmallOrdered()`, then the small fallback below if unavailable. This takes priority over NUMA split.
  * Aggregate local inputs into the ordered host slot, exchange the node block through its leader/proxy, then copy the complete output from host memory.
  * Do not apply one-rank or two-rank-per-node topology-specific register-copy/pack branches.
* Otherwise select `runNumaSplit()` when `getNicGroupLayout()` returns multiple symmetric groups; if unavailable, use `runSingleSlab()`.
* Single-slab uses 2 MiB per-rank chunks. NUMA split uses up to group chunk capacity (16 MiB per rank in the host reference), with individual RDMA writes split at 2 MiB.

### Extension: three or more balanced nodes (extension beyond current GPU-driven support)

* Follow `runLiteAllGather()` selection: do not use the two-node small ordered/fallback paths, regardless of message size.
* Select NUMA split only for multiple symmetric groups and `P != 2`; otherwise select single-slab. `P == 1` does not enable the two-node one-rank pipeline.
* Node/group leaders send their local block directly to every peer node's corresponding leader. This is a hierarchical all-to-all exchange among leaders, not an inter-node ring or recursive doubling.
* Single-slab per-rank chunks are 512 KiB for `P == 2`, otherwise 2 MiB. NUMA split follows the more-than-two-ranks-per-node chunking rules.
* The host reference supports up to 16 balanced nodes and 8 ranks per node. These are reference limits, not existing device-handle guarantees. Extend initialization, topology/control arrays, addressing, proxy connections, and validation together before enabling this case.

### Shared internode invariants

* All inter-node references in this section are in `../allgather_multinode.cu`. Require usable IB transport and pinned host staging; no GDR. Rank numbering uses `nodeId = rank / P`, `localRank = rank % P`, with equal `P` on all nodes.

### SingleSlab

* Follow `runSingleSlab()`, `exchangeGroupChunk()`, and `copyGroupChunkToOutput()`.
  * For GPU-driven service calls, before reading local send-slab rows, every consumer must observe the group's
    D2H readiness in `NodeExchangeBuffer`. The leader's pre-send wait covers the
    leader only; remote `rdmaReady` does not order this node's D2H operations.
    Gate this added wait on `activeDeviceCall`; do not alter native CPU-driven
    execution when fixing the device-service path.
  * Each rank stages a chunk; the leader/proxy waits for all local publications, sends the contiguous node block to every remote node, then publishes network readiness after payload visibility.
  * Assemble output in global-rank order. Preserve direct self-source copies: the host path pre-copies self for `P > 1` and `B >= 512 KiB`, skipping self-copy in-place. Input must still be staged for peers.
  * Require local reader completion and remote consumption/ACK before slot reuse. FIFO dequeue is not DMA/RDMA completion.

### NumaSplit

* Follow `getNicGroupLayout()`, `getNumaContext()`, and `runNumaSplit()`.
  * Scan GPUs in local-rank order, forming contiguous groups when GPU NUMA identity changes; group count is bounded by available IB transports. Nodes must have matching group boundaries and sizes; otherwise use one group/single-slab.
  * Each group owns staging buffers and a leader, with transport selected for its GPU locality. Leaders exchange with the matching group on all peer nodes; every local rank consumes every group's data.
  * Multiple NICs alone do not imply multiple groups. CPU service parallelism does not relax the single-CTA GPU constraint.

### Dual-rail within SingleSlab

* Follow `writeDataStripedToRemoteRecv()` inside generic single-slab exchange.
  * Require two nodes, a non-NUMA-split context, an available second IB transport, and current `blockBytes >= 2 MiB`. Here `blockBytes = groupSize * currentPerRankChunkBytes`, not `T`.
  * Split the block between rails and ensure both payload transfers precede readiness; otherwise use `writeDataDirectToRemoteRecv()` or ordinary connection writes.
  * Default `2n * 2g` chunks form at most a 1 MiB block and do not qualify. Do not automatically apply striping to small ordered slots or the specialized one-rank pipeline.

### SmallFallback

* Follow `runSmallFallback()` and `copySmallFallbackOutput()` only for two nodes within the small threshold (`T < 2 MiB` for `P == 1`, otherwise `T < 128 KiB`).
  * Stage inputs, exchange node blocks, CPU-repack into global-rank order, perform one complete-output H2D per rank, and protect reuse with completion/ACK.
  * Preserve priority: ordered small -> small fallback -> eligible NUMA split -> single-slab. The host reference tries another path only on `ncclInvalidUsage`/`ncclInvalidArgument`; other errors propagate.
  * For device calls, validate eligibility consistently before publishing work. Never switch protocols independently after peers have entered a collective or invoke external NCCL as an implicit device-side fallback.

## GPU-driven adaptation boundaries and validation

* The three-or-more-node extension is beyond current GPU-driven support. Copy describes the single-rank behavior; multi-rank targets use 2-8 total ranks on one or two nodes. Size thresholds never bypass `maxBytesPerRank` or allocated slot/control capacity.
* Keep Intranode's requested IPC-event and graph-capture eligibility policy, but implement invocation progress through device flags and the CPU service protocol. A DMA service stream must not wait for the user kernel to finish while that kernel is waiting for its DMA.
* Prepare registration, IPC mappings, NIC groups, streams, and queues during collective host initialization. GPU routines post work and observe completion; CPU workers execute CUDA runtime calls and IB operations. No per-call host collective invocation or child-kernel launch.
* Validate both sides and equality of branch thresholds, aligned/unaligned tails, in-place/out-of-place inputs, multiple chunks, slot wraparound, and ordered output for each supported topology. Do not claim host-path parity without correctness and performance evidence; keep testing commands and results in `docs/`.
