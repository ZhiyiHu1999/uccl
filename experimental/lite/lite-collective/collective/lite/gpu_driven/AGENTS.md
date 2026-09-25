# Development guide for GPU-driven collectives

* Use this file to implement features and fixes in this directory to achieve the below mentioned functions following the below architecture.
* Keep it focused on code navigation, development decisions, invariants, and validation.
* Put implementation details, benchmarking/testing cookbook(steps + commands), and benchmark results in separate files in `docs/`.

## Functions

* Implement device-callable collectives (AllGather, ReduceScatter, AllReduce) that can run inside a user CUDA kernel after one host-side collective initialization.
* The implemented collectives are especially optimized for the following topologies: 1n * 2g, 1n * 4g, 2n * 1g, 2n * 2g, 2n * 4g (where *a* n * *b* g means *a* nodes, with *b* GPUs per node in the topology). The hardware does not support NVLink or GPU Direct RDMA (GDR).

## Architecture

### Intranode Data path

For exhanging data/performing data reduction within the same node, we generally have two paths:

* The first path only involves GPU. It leverages GPU SM to copy data from source into the intermediate buffer, and GPU SM perform reduction operation on data residing in the intermediate buffer. then copy reduced/aggregated data from the intermediate buffer to the destination. The intermediate buffer can have two types: reside in the GPU memory, accessed by peers through Pcie CUDA IPC; or reside in host memory.
* The second path offloads part of the work to CPU. It leverages CPU service threads (currently have some implementation in 'gpu_staging_channel.hpp'): GPU cooperates with CPU through a task FIFO.
  1. GPU acts as the provider by copying data from source to the intermediate buffer, this can be done through: (a) For small data chunk, GPU SM load from source buffer, then store to pinned host memory. After the copy completion, GPU posts a working element (the element contains details about the copy completion to help CPU fetch data) into a ring FIFO (can contain several work elements). (b) For large data chunk, GPU post working element directly into a ring FIFO, so CPU can fetch the task and launches data movement through 'cudamemcpy()' or 'cudamemcpyasync()'. Two mechanisms could be adopted by GPU to notify the CPU of the arrival of new task: by setting a flag in the FIFO slot that contains the task or by adding one to a 'tail' variable polled by CPU; The CPU notifies the GPU of task completion (so GPU could reuse the FIFO slot and the buffer) through also two similar mechanisms: by resetting the flag in the slot of the FIFO or adding one to a 'head' variable polled by the GPU.
  2. The CPU consumes the work elements in the FIFO, and accordingly aggregates data, perform reduction (could use the SIMD optimization similar to the ones in the CPU switch abstraction), and notify GPU of the data readiness.
  3. Data movement from intermediate buffer to the destination, this can either be done by GPU SM load/store or by DMA engine (CPU launches 'cudamemcpy' or 'cudamemcpyasync'). Here, CPU can be the data provider and GPU got notified as the consumer. The GPU needs to know work completion with the ring FIFO involvement.

### Internode Data Path

With no GDR, all inter-node payloads pass through pinned host memory registered with the NIC. The GPU initiates each operation through device-visible queues/flags; CPU service threads post IB work and drive network progress. This is GPU-driven invocation with CPU-assisted transport, not direct GPU-to-NIC communication.

#### Data flow

`source GPU -> local pinned send buffer -> local NIC -> remote NIC -> remote pinned receive buffer -> destination GPU`

* **Prepare resources once on the host.**

  * Collectively discover the node/rank and GPU/NIC NUMA layout, allocate shared pinned staging/control buffers, register network payload buffers, exchange remote addresses/keys, establish connections, and start CPU service threads.
  * Map the network buffer into GPU address space.
  * Allocate task FIFOs, completion counters, streams, and events before issuing the device handle passed to the kernel; do not register memory or establish connections for each device-driven collective invocation.
* **Stage outgoing data.**

  * For small chunks, GPU threads copy input to mapped pinned memory, make the payload system-visible, then publish a ready descriptor.
  * For large chunks, the GPU publishes a DMA request and the CPU service thread submits D2H on an independent stream. The CPU must observe actual DMA completion before allowing the NIC to read that chunk. A descriptor identifies the operation/epoch, chunk, buffer slot, offsets, byte count, and destination or group, so multiple outstanding tasks cannot alias accidentally.
* **Aggregate locally and send.**

  * A node leader, or one leader per NUMA/NIC group, waits for the required local contributions and posts RDMA writes from registered host memory.
  * AllGather aggregates rank contributions without reduction; ReduceScatter/AllReduce may reduce contributions on the GPU or CPU according to their algorithm before sending.
  * CPU reduction may use SIMD, but its result must be ready before NIC access.
  * Group leaders use their local NICs and corresponding remote groups; single-rank nodes need no local aggregation.
* **Publish remote readiness.**

  * Transfer payload before publishing its epoch/chunk-ready value. Use a transport ordering/completion protocol that guarantees payload visibility before the receiver observes readiness; the host ordered-slot path uses data and flag writes on the same QP.
  * With multiple QPs or rails, readiness must cover every contributing transfer. A local send completion alone does not mean the remote GPU has consumed the payload.
* **Consume incoming data.**

  * For small chunks, the GPU waits on mapped readiness and loads data from pinned memory.
  * For large chunks, the receiving CPU service thread waits for network readiness, submits H2D, then publishes GPU-visible completion only after that copy finishes.
  * For reductions, publish final output completion only after all required reduction/copy work is complete. Local contributions can be read directly from the source when valid, but must still be published for other consumers.
* **Retire work and reuse resources.**

  * Track FIFO descriptor ownership, local staging readiness, network completion, and GPU consumption separately.
  * A FIFO head advance may release the descriptor but must not implicitly release an in-flight payload buffer.
  * Reuse a send slot only after NIC reads and local readers finish;
  * Reuse a remote receive slot only after its GPU/DMA consumers finish and the required ACK/credit reaches the sender. Use epoch/sequence values to distinguish successive uses of a slot and avoid accepting stale flags.Optimizations for better performance:

#### Considerations for better perormance:

* Pipeline chunks through buffer staging, RDMA, and receive consumption using independent buffers/slots and explicit dependencies. Per-chunk GPU flags alone do not create network overlap: the CPU proxy must send and receive incrementally instead of waiting for the whole input. Preserve the algorithm-specific chunk size, window, and slot-reuse protocol described below.
* Keep progress independent of the calling user kernel in case of deadlocks. A service stream must be separated from the collective kernel invocation stream, and never wait for that kernel to finish while the kernel waits for its DMA or RDMA completion. CPU workers must continue both send and receive progress, including ACK handling, rather than block indefinitely on one direction.
* Separate GPU-to-CPU publication from CPU/NIC-to-GPU visibility. Apply the appropriate system-scope ordering and supported polling mechanism at each boundary; a block barrier or a plain volatile flag alone is not a complete cross-device visibility protocol.

### Primitive Layer

* Develop a primitive layer to abstract each kind of opeartion in both intra-/inter-node data/control plane. The primitives will be leveraged to compose collective algorithms described below.
* Reuse the staging/queue concepts in `../gpu_staging_channel.hpp` and buffer-management concepts in `../node_exchange_buffer.hpp`. Extend allocation, device-handle fields, proxy processing, completion/error reporting, and cleanup together when introducing a new task or control field. Propagate failures to waiting participants, and stop/join workers before freeing buffers they can access.

### AllGather

These cases describe implementation targets, not a claim that all paths already exist. Preserve the device-callable API and one participating CTA per rank. Adapt reference kernels into routines executed by that CTA; submit CPU DMA/RDMA work through the task FIFO/service thread rather than invoking a host collective or launching a child kernel.

Host references below are in `../allgather_intranode.cu` or `../allgather_multinode.cu`. KiB/MiB/GiB are binary units. Distinguish per-rank input `B`, complete output `T`, and the current network block size.

Symbols：

* `B`: Input number of bytes per rank.
* `N = number of nodes; P = number of ranks per node`，`N =  nranks / P`.
* `T = B × nranks`: output number of bytes per rank.

#### Case 1: one rank only

* D2D copy directly.
* No other operation when in-place.

#### Case 2: one node only, multiple ranks

* CUDA IPC ring AllGather
  * Similar algorithm as `runIntraNodeCudaIpcAllGather()` in `allgather_intranode.cu`.
  * This path is choosen when all below conditions are met:
    * `HOST_ALLGATHER` (host SHM) is disabled (this option is disabled by default, and `MSCCLPP_NCCL_CUDAIPC_EVENT_SYNC` is enabled by default);
    * ~~`hasIB=true`;~~  (Do not implement this rule, it is a bug in host-driven case)
    * IPC event sync is enabled；
    * `T ≥ 8 MiB`;
    * No CUDA Graph capture.
* Through shared host memory
  * Not a fallback of 'CUDA IPC ring AllGather', requires explicit enabling.
  * Similar algorithm as `runIntraNodeShmAllGather()` in `allgather_intranode.cu`.
    * Implement internel paths [`hostAllGatherMappedKernel()`](/Users/zhiyihu/Desktop/pr_to_uccl/uccl/experimental/lite/lite-collective/collective/lite/allgather_intranode.cu:151), [`hostAllGatherCoopKernel()`](/Users/zhiyihu/Desktop/pr_to_uccl/uccl/experimental/lite/lite-collective/collective/lite/allgather_intranode.cu:190), and [`runIntraNodeShmAllGather()` ](/Users/zhiyihu/Desktop/pr_to_uccl/uccl/experimental/lite/lite-collective/collective/lite/allgather_intranode.cu:507)as in host-driven case (selection conditions are all kept, `hostAllGatherCoopKernel()` should also be kept even if we use only one CTA for GPU-driven collectives).
    * Chunk size also follow the logics in host-driven collectives.
  * This path is chosen when all below conditions are met:
    * `MSCCLPP_NCCL_HOST_ALLGATHER` is enabled
    * `T ≤ 1 GiB`
    * satisfy the lower bound of `HOST_ALLGATHER_MIN_BYTES`
    * No CUDA Graph capture

##### Case 2 SHM path details: shared-host subpaths and chunking

* Preserve selection priority in `runIntraNodeShmAllGather()`:
  * `hostAllGatherMappedKernel()` logic: `T <= 4 KiB`, one chunk, `B` divisible by 8, 8-byte-aligned input/output, and valid device mappings for slab/control.
  * `hostAllGatherCoopKernel()` logic: the first branch did not match, `T <= 512 KiB`, the same alignment/mapping/single-chunk requirements, and cooperative-launch capability. Retain this branch as requested, but adapt its phases to the caller's single CTA; do not introduce a multi-CTA launch.
  * DMA pipeline: remaining eligible calls, including unaligned or unmapped small inputs. Stage each input chunk D2H, copy self directly, and consume peer chunks through H2D after readiness publication. Preserve the separate left/right peer-range service streams where useful.
* Default per-rank chunk size: `B` when `B <= 1 MiB`; 1 MiB when `1 MiB < B <= 32 MiB`; 4 MiB when `B > 32 MiB`.
* Preserve the tuning semantics of `MSCCLPP_NCCL_HOST_ALLGATHER_{KERNEL_MAX_BYTES, COOP_MAX_BYTES, CHUNK_BYTES,MAP_SLAB}`. `MAP_SLAB=0` disables mapped-copy branches. The host SHM implementation supports 2-8 ranks.

#### Case 3: two nodes, one rank per node (`2n * 1g`)

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

#### Case 4: two nodes, two ranks per node (`2n * 2g`)

* Small ordered-slot AllGather: follow `runSmallOrdered()` when `T < 128 KiB` (`B < 32 KiB`).
  * `T <= 256 bytes` with slab/control mappings: use `multiRankTinyPackKernel()` logic.
  * `512 bytes <= T <= 4 KiB` with slab/control mappings: use `multiRankRegisterPackKernel()` logic.
  * Otherwise use DMA staging; do not fill the `(256, 512)` gap with register packing.
  * Independently select `multiRankHostRecvKernel()` logic when `T <= 4 KiB` and the send slab is device-mapped; otherwise use complete-slot H2D. DMA packing can therefore coexist with SM-based receiving. Validate all control pointers accessed by the adapted routine.
  * Each local rank publishes staged input. One node leader/proxy exchanges the contiguous node block; both local ranks consume the complete rank-ordered result.
* For `T >= 128 KiB`, or after eligible small-path fallback, use `runSingleSlab()` -> `exchangeGroupChunk()` with 512 KiB per-rank chunks.
* Preserve the host dispatch exclusion of NUMA split for `P == 2`, even when multiple NIC groups are detected.

#### Case 5: two nodes, more than two ranks per node (especially `2n * 4g`)

* When `T < 128 KiB`, first use `runSmallOrdered()`, then the small fallback below if unavailable. This takes priority over NUMA split.
  * Aggregate local inputs into the ordered host slot, exchange the node block through its leader/proxy, then copy the complete output from host memory.
  * Do not apply Case 3 or Case 4's topology-specific register-copy/pack branches.
* Otherwise select `runNumaSplit()` when `getNicGroupLayout()` returns multiple symmetric groups; if unavailable, use `runSingleSlab()`.
* Single-slab uses 2 MiB per-rank chunks. NUMA split uses up to group chunk capacity (16 MiB per rank in the host reference), with individual RDMA writes split at 2 MiB.

#### Case 6: three or more balanced nodes (extension beyond current GPU-driven support)

* Follow `runLiteAllGather()` selection: do not use the two-node small ordered/fallback paths, regardless of message size.
* Select NUMA split only for multiple symmetric groups and `P != 2`; otherwise select single-slab. `P == 1` does not enable the two-node one-rank pipeline.
* Node/group leaders send their local block directly to every peer node's corresponding leader. This is a hierarchical all-to-all exchange among leaders, not an inter-node ring or recursive doubling.
* Single-slab per-rank chunks are 512 KiB for `P == 2`, otherwise 2 MiB. NUMA split follows Case 5's chunking.
* The host reference supports up to 16 balanced nodes and 8 ranks per node. These are reference limits, not existing device-handle guarantees. Extend initialization, topology/control arrays, addressing, proxy connections, and validation together before enabling this case.

#### Shared inter-node algorithm details and invariants

* All inter-node references in this section are in `../allgather_multinode.cu`. Require usable IB transport and pinned host staging; no GDR. Rank numbering uses `nodeId = rank / P`, `localRank = rank % P`, with equal `P` on all nodes.
* Generic single-slab: follow `runSingleSlab()`, `exchangeGroupChunk()`, and `copyGroupChunkToOutput()`.
  * Each rank stages a chunk; the leader/proxy waits for all local publications, sends the contiguous node block to every remote node, then publishes network readiness after payload visibility.
  * Assemble output in global-rank order. Preserve direct self-source copies: the host path pre-copies self for `P > 1` and `B >= 512 KiB`, skipping self-copy in-place. Input must still be staged for peers.
  * Require local reader completion and remote consumption/ACK before slot reuse. FIFO dequeue is not DMA/RDMA completion.
* NUMA split: follow `getNicGroupLayout()`, `getNumaContext()`, and `runNumaSplit()`.
  * Scan GPUs in local-rank order, forming contiguous groups when GPU NUMA identity changes; group count is bounded by available IB transports. Nodes must have matching group boundaries and sizes; otherwise use one group/single-slab.
  * Each group owns staging buffers and a leader, with transport selected for its GPU locality. Leaders exchange with the matching group on all peer nodes; every local rank consumes every group's data.
  * Multiple NICs alone do not imply multiple groups. CPU service parallelism does not relax the single-CTA GPU constraint.
* Dual-rail: follow `writeDataStripedToRemoteRecv()` inside generic single-slab exchange.
  * Require two nodes, a non-NUMA-split context, an available second IB transport, and current `blockBytes >= 2 MiB`. Here `blockBytes = groupSize * currentPerRankChunkBytes`, not `T`.
  * Split the block between rails and ensure both payload transfers precede readiness; otherwise use `writeDataDirectToRemoteRecv()` or ordinary connection writes.
  * Default `2n * 2g` chunks form at most a 1 MiB block and do not qualify. Do not automatically apply striping to small ordered slots or the specialized one-rank pipeline.
* Small fallback: follow `runSmallFallback()` and `copySmallFallbackOutput()` only for two nodes within the small threshold (`T < 2 MiB` for `P == 1`, otherwise `T < 128 KiB`).
  * Stage inputs, exchange node blocks, CPU-repack into global-rank order, perform one complete-output H2D per rank, and protect reuse with completion/ACK.
  * Preserve priority: ordered small -> small fallback -> eligible NUMA split -> single-slab. The host reference tries another path only on `ncclInvalidUsage`/`ncclInvalidArgument`; other errors propagate.
  * For device calls, validate eligibility consistently before publishing work. Never switch protocols independently after peers have entered a collective or invoke external NCCL as an implicit device-side fallback.

#### GPU-driven adaptation boundaries

* Case 1 and Case 6 extend the current initialization contract (2-8 total ranks on one or two nodes). The contracts below describe current support until those extensions are implemented. Size thresholds never bypass `maxBytesPerRank` or allocated slot/control capacity.
* Keep Case 2's requested IPC-event and graph-capture eligibility policy, but implement invocation progress through device flags and the CPU service protocol. A DMA service stream must not wait for the user kernel to finish while that kernel is waiting for its DMA.
* Prepare registration, IPC mappings, NIC groups, streams, and queues during collective host initialization. GPU routines post work and observe completion; CPU workers execute CUDA runtime calls and IB operations. No per-call host collective invocation or child-kernel launch.
* Validate both sides and equality of branch thresholds, aligned/unaligned tails, in-place/out-of-place inputs, multiple chunks, slot wraparound, and ordered output for each supported topology. Do not claim host-path parity without correctness and performance evidence; keep testing commands and results in `docs/`.

### ReduceScatter

TBD

### AllReduce

TBD

## Scope and working approach

- Implement device-callable collectives (AllGather, ReduceScatter, AllReduce) that can run inside a user CUDA kernel after one collective host initialization.
- Uncommited codes could be abandoned if the guideline conflicts with the code.
- Only 1 GPU block/SM should be involved in all GPU-driven collectives.
- The source code of GPU-driven collectives should reside in a standalone directory `gpu_driven/`
- Development for files with testing purpose should be limited to benchmarking, other testing/validating files should be removed.
- Benchmarking should compare the performance of gpu-driven collectives with the performance of native NCCL host-driven collectives. The comparison should be fair by controlling the number of SMs in both cases to 1.
- Keep changes scoped to the requested feature. Preserve existing public APIs and `mscclpp` / `MSCCLPP_*` names unless the task requires an interface change.
- For a feature request, identify the affected collective, backend, message-size range, and execution model from the request and current code. State reasonable assumptions and proceed; ask only when an unresolved choice changes required behavior or compatibility.
- Treat current limits as implementation boundaries, not permanent prohibitions. If a requested feature expands them, update device code, host setup, validation, and documentation together.
- If this `AGENTS.md` got updated, and the code base has been generated based on the previous version of `AGENTS.md`, all temporary compromising restricted to previous AGENTS.md should be abandoned for next run, anddevote .

## Benchmark command-line requirements

- The script and executable must support nccl-tests-style sweeps with
  `-b BEGIN -e END -f FACTOR`, binary size suffixes `B/K/M/G` (case-insensitive),
  `-w WARMUPS`, `-n ITERS`, and `-g 1` (one GPU per MPI process).
- Require both range bounds. Multiply by an integer factor >= 2 (default 2)
  while within the inclusive end bound; prevent arithmetic overflow.
- Preserve positional size lists and their order, including repeated sizes.
  Reject mixing ranges with positional sizes, invalid values and unsupported GPU counts.
- CLI iteration counts override environment defaults. Reports must record the
  effective counts. Help must work without GPU initialization.
- Keep the 15-second execution limit explicit. Never silently reduce requested
  iterations or treat skipped paths/timeouts as successful measurements.

## Finish the task

- Summarize the behavior implemented, relevant files, validation performed, and remaining limitations.
- Update this guide only when code navigation, contracts, or development procedures change. Put benchmark tables and implementation history under `docs/`.
- Keep generated build outputs and temporary benchmark artifacts out of source changes. Avoid unrelated refactors and edits to vendored code.
