# GPU-driven collectives

## API and execution contract

Call `mscclppGetDeviceCollectiveHandle` collectively once, then pass its handle by value into a user CUDA kernel. Every thread of exactly one participating CTA per rank calls `liteAllGatherBlock`, `liteAllReduceBlock<T>`, or `liteReduceScatterBlock<T>` in the same order, with matching sizes and operations. HostCooperative AllGather additionally supports all threads of a cooperatively launched grid; multiple CTAs cooperate on one invocation, not independent calls. A handle cannot be used by concurrent independent invocations or streams. Resources remain owned by the communicator; synchronize the caller stream before destroying it.

Supported topologies are one rank, or 2–8 total ranks on one or two balanced nodes, including 1n×2g, 1n×4g, 2n×1g, 2n×2g and 2n×4g. Rank numbering must be node-contiguous. Device compilation requires sm_70 or newer for system-scope acquire/release. All inter-node payloads reside in pinned host memory registered with IB; there is no GDR path.

AllReduce and ReduceScatter implement Sum, Min and Max for arithmetic template types (benchmark coverage includes int and float). ReduceScatter capacity is the complete input size, not the receiving shard. In-place ReduceScatter uses `dst = src + rank * recvCount`. Reductions require all payload slabs to be device mapped; AllGather additionally supports DMA-only slabs. One-rank calls copy directly, skipping identical pointers.

`liteAllGatherBlock` accepts an optional `graphCaptured` argument. It must describe the enclosing kernel launch; multi-rank AllGather rejects capture/replay before publishing work. Ordinary four-argument callers remain supported. The device cannot discover host stream capture state.

## File responsibilities

| File | Responsibility |
| --- | --- |
| `allgather_plan.hpp` | Side-effect-free topology, size, alignment and policy selection |
| `task_fifo.hpp` | DMA descriptors, FIFO ownership and system-scope publication |
| `gpu_collectives.cuh` | Device handle, single-CTA collectives, staging, reduction and retirement |
| `host_context.hpp` | Collective setup, NUMA discovery, registrations, connections, network proxies and cleanup |
| `service.hpp` | DMA task service and invocation bridge to GPU-private network schedules |
| `network_service.hpp`, `network_protocol.hpp` | Shared-schedule host API and ordered SM-phase handshake |
| `device_collectives_bench.cu` | Native NCCL comparison with untimed correctness preflight |
| `benchmark.sh` | Build, MPI launch, bounded execution and benchmark report |

## AllGather selection

Host policy is snapshotted during collective initialization. Shared-host AllGather requires explicit `MSCCLPP_NCCL_HOST_ALLGATHER=1`, total output within the configured minimum and 1 GiB maximum, and no capture. The benchmark explicitly opts into host mode unless configured otherwise. Mapped, cooperative-phase and DMA branches retain their priority, alignment and capability conditions. Cooperative phases support a full cooperative grid; the benchmark retains one block (one SM). Per-rank chunks default to B up to 1 MiB, 1 MiB through 32 MiB, then 4 MiB, retaining the host tuning variables.

CUDA IPC AllGather requires host AllGather disabled, IPC event synchronization
enabled, total output at least 8 MiB, and no capture. Its independent
`allgather_ipc.cuh` routine sends complete rank blocks to next's registered output
using CPU-submitted D2D DMA, with one epoch and `nranks - 1` steps per invocation.
GPU threads only publish tasks and synchronize; the IPC path does not SM-copy
payloads or use the generic chunk loop. The CPU service publishes descriptor
completion after a CUDA event; the GPU then publishes hop completion to peers.
All-rank admission and final completion protect repeated in-place calls.

After handle initialization and before launching a kernel, collectively call
`mscclppRegisterDeviceCollectiveIpcOutput(comm, output, allocationBytes)` on the
cudaMalloc allocation base. Regions must have equal sizes and each call must use
the same relative destination offset on every rank. Subregions (including
unaligned destinations) are supported within this allocation. The source may be
separate or the rank's in-place output row. Only one region is registered per
communicator's IPC context; no concurrent users are allowed. Synchronize users,
then collectively unregister with `(comm, nullptr, 0)` before freeing output, or
re-register a replacement. No memory export/open occurs inside device invocations.
The benchmark registers once per allocated test output outside preflight/timing
and unregisters before freeing it. Real CUDA compilation, DMA visibility and
performance validation of this replacement are pending.

## GPU-private multi-node transport

The five device entry points post `NetworkAllGather` descriptors to the service.
The service executes private scheduling functions in `network_service.cu`,
with dedicated contexts prepared once. These were copied from the previously
shared implementation; native CPU entry points now use their restored baseline
implementation in `../allgather_multinode.cu`.
It does not call `ncclAllGather`, `runLiteAllGather`, or launch kernels. Payload
and control memory, QPs, streams, and slot/chunk events are prepared collectively
before returning a handle. A service-owned stream replaces the CPU caller's
input/output stream; it never waits for the calling GPU kernel to finish.

- OrderedSmall preserves the reference's slot sizing (up to 1024), layout changes and ring-wrap
  barriers, direct-QP ordered/compact writes and their signaled polling cadence.
  The service publishes the chosen slot/epoch and mapped addresses. The caller
  CTA performs SM packing and receiving with the reference participation counts:
  one-rank tiny/parallel/compact specializations and separate P=2 pack/receive
  counts. Mapping availability is agreed at setup; unmapped payloads use the
  ordered DMA path. Only a pre-publication eligibility failure invokes fallback.
- SmallFallback preserves D2H, node exchange, CPU repacking, full-output H2D and
  final ACK scheduling.
- OneRankPipeline preserves one full-message epoch/slot, all-chunk D2H submission,
  512 KiB chunks with stable ready words, receive-paced send window of one,
  full-message D2D self-copy and event/ACK-controlled slot reuse.
- SingleSlab preserves per-chunk epochs, capacity-dependent slot counts, node/group
  H2D batches, self-copy conditions, direct/striped QP operations and ACK rules.
- NumaSplit preserves independent group epochs/slots and own-group staging, then
  group/node ordered H2D on the owning group's stream. NUMA discovery uses the
  CPU reference implementation, including exclusion of P=2 splitting.

The FIFO sequence is distinct from every algorithm epoch. GPU output publication
is fenced before posting an invocation. FIFO completion follows DMA completion;
OrderedSmall additionally waits for the calling CTA's SM completion. The CTA
waits for this ticket before returning. This is the device-callable equivalent
of output dependencies placed on the CPU caller's stream, not an assertion of
identical invocation overhead or performance.

The original RDMA proxy/NUMA device handles remain for reductions. Multi-node
AllGather uses the primary FIFO and its dedicated private schedule owner instead
of their fixed two-slot protocol. No independent invocations may overlap on a
handle. Cleanup stops the worker and drains streams/QPs before freeing context
resources. Device timeout/abort is observed by GPU-private CPU-service polling loops; a
failed handle must be destroyed after its user kernel completes.

## Limits and validation status

Three or more nodes (the guide's Case 6 extension) remain unsupported. Two-node reductions still stage and exchange their complete input per invocation, rather than using a specialized reduction pipeline. Allocations scale with the requested maximum capacity; NUMA contexts additionally allocate group resources at initialization, capped at 16 MiB per rank per slot.

The development machine has no CUDA/IB toolchain or hardware. Portable protocol simulation and Clang device-code generation passed, but these do not establish CUDA cache visibility, real copy-engine overlap, IB behavior, performance parity, or speedup. No hardware benchmark result is claimed. See [validation.md](validation.md) for exact checks and target-machine commands.

## Single-node host executors

The three independent executors follow the CPU reference in
`../allgather_intranode.cu`, using one epoch/slot per invocation:

- HostMapped: packed current-size rank rows, SM stage, all-peer ready wait,
  contiguous full-output SM copy including self, then done.
- HostCooperative: the same phases distributed across a cooperative grid, with
  grid barriers before ready and done. The existing one-block call uses a
  thread-block group; multi-block callers require cooperative launch and all
  grid threads must enter HostCooperative with identical arguments. Setup owns
  the grid's shared epoch/status scratch. Other paths are still single-CTA. Opt in with the sixth argument of
  `liteAllGatherBlock(h, src, dst, bytes, false, true)`; non-Cooperative
  selections are rejected before reserving an epoch. The default remains one
  participating CTA, including callers inside a larger user grid.
- HostDma: one FIFO descriptor submits all chunks. The CPU service uses the
  reference staging channel's `put` and `wait`, with stream-ordered per-chunk
  readiness. Left/right rank ranges are batched into contiguous or 2D H2D copies
  on independent streams. Host allocation pitch and output message pitch can
  differ. Self uses D2D DMA on a fourth independent stream by default; setting
  `MSCCLPP_NCCL_HOST_ALLGATHER_SELF_KERNEL=1` selects caller-SM self-copy for
  aligned inputs, preserving the one-block benchmark constraint. In-place self
  copies are skipped. All stream events and caller-SM work finish before done.

The benchmark continues to launch exactly one block; no multi-SM performance
claim is made. Path thresholds and inter-node/IPC algorithms are unchanged.
CUDA compilation and hardware validation of these changes remain required.

## Multi-node entry files

| Device path | Entry file | Shared CPU transport schedule |
| --- | --- | --- |
| OrderedSmall | `allgather_ordered_small.cuh` | `executeOrderedSmallSchedule` |
| SmallFallback | `allgather_small_fallback.cuh` | `executeSmallFallbackSchedule` |
| OneRankPipeline | `allgather_one_rank_pipeline.cuh` | `executeSingleSlabSchedule` → `runOneRankChunkPipeline` |
| SingleSlab | `allgather_single_slab.cuh` | `executeSingleSlabSchedule` → `exchangeGroupChunk` |
| NumaSplit | `allgather_numa_split.cuh` | `executeNumaSchedule` |

See `validation.md` for the limits of local validation. CUDA/IB hardware results
for this rewrite are still pending; sharing code is not a performance claim.

## CPU/GPU source isolation

`allgather_intranode.cu`, `allgather_multinode.cu` and `cpu_staging_channel.hpp`
are restored byte-for-byte to commit `0a1edbea`. GPU network service has its own
translation unit; CPU code no longer includes the GPU network bridge or calls
its schedules. The GPU host context uses its own NUMA-option helper and
`DeviceHostStagingBuffer`, preserving move ownership fixes without altering
CpuStagingChannel. Low-level CpuSwitch/NodeExchangeBuffer remain shared.
The snapshot duplicates code deliberately to isolate future GPU fixes; reference
changes must be reviewed and ported explicitly, not assumed to propagate.
