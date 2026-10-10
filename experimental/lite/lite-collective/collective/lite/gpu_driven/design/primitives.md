# GPU-driven primitives

This document describes the basic operations that exist in the current GPU-driven
implementation and how they are composed. For the common architecture see
[AGENTS.md](../AGENTS.md); for AllGather selection and algorithms see
[allgather.md](allgather.md); for ReduceScatter see [reducescatter.md](reducescatter.md).
A "primitive" here is a basic operation in the implementation; it does not mean a
unified primitive class or a separately compiled module exists. When changing these
operations, also check the device callers, host initialization, service, error
propagation and resource release.

## Execution model and source map

A GPU user kernel initiates the collective; the CPU prepares resources at
initialization and starts the service. Device functions never create new CTAs and
never launch child kernels. Normally one CTA cooperates: one thread submits the task
or updates the control word, and the other threads take part in copying and
synchronization as the algorithm requires. HostCooperative may use the full
cooperative grid, while the benchmark still launches a single block. The same handle
must not run different collectives concurrently.

| File | Primitives implemented |
|---|---|
| [gpu_collectives.cuh](../gpu_collectives.cuh) | Address computation, CTA copy, reduction operators, epoch/slot management, waiting, task submission and completion checks |
| [task_fifo.hpp](../task_fifo.hpp) | `LiteTask`, 8-slot FIFO, system-scope acquire/release |
| [allgather_host_cooperative.cuh](../allgather_host_cooperative.cuh) | Copy distributed over a cooperative group |
| [service.hpp](../service.hpp) | CPU consumes tasks, submits DMA, checks CUDA events, publishes completion |
| [host_staging_buffer.hpp](../host_staging_buffer.hpp) | GPU-private host staging buffer, stream-ordered put/wait, resource ownership |
| [allgather_network.cuh](../allgather_network.cuh) | Device-side submit, wait and completion wrappers for network AllGather |
| [network_protocol.hpp](../network_protocol.hpp) | SM-phase handshake between the CPU schedule and the calling CTA |
| [network_service.cu](../network_service.cu) | GPU-private network schedule, RDMA data/control operations, ACK and slot reuse |
| [host_context.hpp](../host_context.hpp) | Initialization, mapping, service lifecycle and cleanup |
| [reducescatter_protocol.hpp](../reducescatter_protocol.hpp) / [reducescatter.cuh](../reducescatter.cuh) | CTA copy/sum phase requests and completion inside a FIFO invocation (`liteRsInvokePath`); per-path device entries in `reducescatter_{ipc,host,two_rank,hierarchical}.cuh` |
| [reducescatter_primitives.cuh](../reducescatter_primitives.cuh) | Single-node RS primitives: `RsCopy` (1D/2D DMA), `RsHostSum` (CPU sum), `RsBarrier` (node barrier) FIFO tasks and the CTA `liteRsSum` |
| [reducescatter_service.hpp](../reducescatter_service.hpp) | RS scratch/IPC/SHM/RDMA initialization; executes the RS FIFO primitives and the two-node schedule |
| [ipc_output_registration.hpp](../ipc_output_registration.hpp) | IPC output-region registration, exchange and mapping |

The native CPU-driven collectives are not an entry point for GPU-driven primitives.
GPU-only changes must not modify `../../allgather_intranode.cu`,
`../../allgather_multinode.cu` or `../../cpu_staging_channel.hpp`; the shared
NodeExchangeBuffer/CpuSwitch must still be checked across paths.

## Common control primitives

### Publish and observe

`liteStoreRelease()` publishes a control word with `st.release.sys.global.u64`, and
`liteLoadAcquire()` reads one with `ld.acquire.sys.global.u64`. The device target
requires sm_70+. Control locations are naturally aligned 64-bit words, and writers are
assigned by the protocol. The CPU reads and writes FIFO/control words with
acquire/release atomics.

A CTA's `__syncthreads()` or a cooperative group's `group.sync()` synchronizes the
participating threads; it cannot replace the cross-CPU/GPU publication protocol. After
multiple threads move data, the caller must complete visibility handling in the
barrier/system-fence order used by the implementation before publishing ready. The
copy helpers themselves do not publish ready.

### Wait, failure and retirement

* `liteCollectiveWait()` waits for `observed >= target` by default, suitable for
  monotonic epochs/tickets.
* `liteCollectiveWait<true>()` waits for exact equality. OrderedSmall's reused
  payload-slot flag uses this mode so that old payload bytes are not mistaken for a
  newer epoch after a size change.
* Waiting also checks the FIFO error, the `~0ULL` error marker and `timeoutCycles`. On
  failure, `liteCollectivePoison()` publishes the error state and the wait returns
  failure; the protocol must not be switched and retried after communication has begun.
* `liteCollectiveWaitReusable()` checks the required network/group reusable conditions
  and each rank's done. NIC completion does not imply that local GPU/DMA readers have
  finished.
* On failure, the network wait wrappers also set `LiteNetworkControl::abort` so the CPU
  schedule stops waiting. A failed handle should be destroyed after the calling stream
  finishes and must not be reused.

### Addressing, epochs and slots

`mscclppDeviceCollectiveSlab/Ready/Done()` compute payload/control addresses per
backend. Host memory, IPC and host RDMA have different strides and indexing; they must
not be treated as one contiguous common layout. Dedicated AllGather executors may also
use packed rows for the current message size, or offsets given by the network schedule.

`liteCollectiveBeginBlock()` allocates a collective epoch, selects a whole-message slot
and waits for the previous user to leave; chunk readiness is managed separately from
whole-message slots. The current generic protocol uses two whole-message slots. This
does not mean every optimized path uses the same double-buffering loop: the network
AllGather's epoch/slot is managed by its own schedule and does not call BeginBlock in
place of its protocol.

The FIFO ticket, the collective epoch and the network chunk-ready value are different
numbers and cannot be substituted for one another.

## Intranode Data Path

### Data flow and responsibilities

Intranode data movement lets the calling path choose SM or DMA. The two can be
combined, but not every stage goes through the FIFO.

* **SM path**: threads of the calling kernel write the input into an accessible
  intermediate payload and publish ready; consumer threads wait for ready, then read
  the data, finish the copy or reduction and finally publish done. HostMapped /
  HostCooperative AllGather uses a mapped host slab; generic reduction may use mapped
  host or IPC payload. Inter-thread synchronization and system-scope publication are
  organized by the executor.
* **Host DMA path**: the GPU submits a task and the CPU service schedules
  `source GPU -> pinned host row -> destination GPU` on separate streams; the self row
  may be a direct D2D. D2H/H2D dependencies are expressed with stream-ordered
  readiness, waits and CUDA events.
* **IPC AllGather path**: the GPU submits a copy task and the CPU service issues DMA
  into the peer's output region through the IPC mapping. Data does not pass through a
  host payload slab, but control still needs GPU/CPU-service cooperation.

The CPU service consumes descriptors and drives DMA; the RS-specific float/sum host
paths also reuse CpuSwitch's CPU reduction (runtime AVX-512 or scalar). When the SM
finishes preparing data it publishes ready according to the specific path; it cannot
be described uniformly as "copy, then submit a FIFO task". The actual FIFO uses
increasing submitted/completed/retired counters, detailed below, rather than an
abstract head/tail or either/or flag scheme.

### SM copy and reduction

`liteCollectiveCopyBlock(dst, src, bytes, activeThreads)` distributes the copy across
the threads of the current CTA with a thread stride: when both ends are 8-byte aligned
it copies 64-bit words and then handles the byte tail; otherwise it copies byte by
byte. `activeThreads` may limit how many threads take part; it does not add CTAs or
schedule more SMs.

`liteAllGatherGroupCopy(group, ...)` distributes the copy using `group.thread_rank()`
and `group.size()`. Given a thread block it works only within that block; given a
cooperative grid it works across the participating blocks. Its 64-bit accesses rely on
the calling path having already met the alignment requirement, and the caller performs
the synchronization.

`liteApplyReduction<T>()` provides generic sum/min/max element operations. Staging and
result computation for generic reduction are done by device threads together with the
mapped host/IPC payload, with chunks managed by `liteCollectiveChunkBytes()`. The
optimized RS float/sum uses the separate `litePlanReduceScatter()` and, per path,
chooses CPU sum or a CTA phase; CPU reduction is not the unified data path of all
collectives.

### GPU-to-CPU task FIFO

Data flow: `GPU descriptor -> mapped host FIFO -> CPU service -> DMA -> completion word -> GPU`.

* `LiteTaskFifo` has 8 descriptor slots and follows a single-producer / single-CPU-consumer
  protocol. Multiple GPU threads must not call `litePostTask()` at the same time
  without coordination.
* `litePostTask()` first checks for a free slot using `submitted` and `retired`, then
  writes the descriptor, and finally release-publishes `submitted + 1`, returning that
  value as the ticket. It waits when there is no space and returns 0 on failure.
* The CPU acquire-reads submitted, takes out the descriptor and submits the
  corresponding work. For asynchronous DMA, it publishes slot.completed only after
  `cudaEventQuery()` confirms that all required events have completed.
* `liteWaitTask()` waits until the corresponding slot's completed reaches the ticket.
  The CPU advances only the contiguous completed ticket prefix into retired, allowing
  descriptor slots to be reused.
* submitted means the task was published; completed means it finished; retired means
  the descriptor can be reused. **None of these values alone can replace a payload's
  ready/done, a network ACK, or a slot-reuse protocol.**

The current descriptor kinds are listed below. Not every AllGather path uses all of them:

| Task kind | CPU service responsibility |
|---|---|
| `Stage` | Submit a copy into the staging buffer; publish staging readiness on completion |
| `Gather` / `GatherPacked` | Submit gather copies from the staging layout to the output |
| `IpcCopySelf` | D2D copy of the self row from local input to output |
| `IpcPush` | Submit a D2D copy into the next rank's mapped output region through the registered output offset |
| `HostAllGather` | Perform DMA submission and completion tracking for one full host-memory AllGather |
| `NetworkAllGather` | Call the GPU-private network schedule to complete one network AllGather |
| `ReduceScatter` | Run one RS schedule; the CPU handles DMA/RDMA or host float sum, and requests a phase from the calling CTA when GPU arithmetic is needed |

### Host-memory DMA

Data flow: `source GPU -> pinned host rank row -> peer output GPU`; self may be a
direct D2D. `enqueueDeviceHostAllGather()` uses buffer.put to submit per-chunk D2H and
publishes ready in stream order. Receiving streams wait via buffer.wait for the source
rank/chunk to become ready and then submit H2D.

The left and right peer ranges each organize batched copies. When the capacity stride
differs from the current output row stride, `cudaMemcpy2DAsync()` is used; the host
slab must not be copied directly as though it were tightly packed for the current size.
Four service streams share staging, peer receive and the self copy, and the completion
check covers all four streams. If self SM copy is enabled, the device executor is
responsible for that copy and its synchronization.

### CUDA IPC DMA

The current IPC AllGather is not an implementation where the SM first copies into an
IPC staging buffer and a peer SM then reads it. The output region is collectively
registered and mapped before the call via `mscclppRegisterDeviceCollectiveIpcOutput()`.
The destination in an `IpcPush` descriptor is the address of the locally registered
output; after validating the range, the service adds the relative offset to the next
rank's mapped base and submits the DMA.

The GPU publishes the task and waits for task completion corresponding to the CUDA
event; only then may it publish hop ready. IPC ring admission, hop readiness and the
final done are maintained by the executor. Output registration is not needed on every
device collective; before replacing, unregistering or freeing it, the user must have
finished.

## Internode Data Path

### Data flow and responsibilities

`source GPU -> local pinned send buffer -> local NIC -> remote NIC -> remote pinned receive buffer -> destination GPU`

The GPU initiates the call and takes part in the required SM stages; the CPU
service/schedule submits IB work and drives the transfer. All cross-node payload goes
through NIC-registered host memory; GDR is not used.

1. **Prepare resources**: host initialization discovers nodes/ranks and GPU/NIC groups,
   allocates staging/control, registers NIC memory, exchanges remote addresses and
   keys, establishes connections, and prepares the FIFO, streams and events. The
   control area must satisfy device-access requirements; whether the payload is mapped
   determines whether the corresponding SM stages are available, and one must not
   assume every path requires a mapped payload.
2. **Prepare send data**: depending on the path, the CTA writes into the mapped host
   payload, or the CPU submits D2H. Before readiness is published, the corresponding
   SM/DMA work must be complete. Network AllGather submits the whole call; chunks,
   slots and offsets are organized by the schedule, and not every chunk is submitted as
   its own FIFO descriptor.
3. **Gather locally and send**: the node/group leader waits for local contributions and
   then sends the corresponding host data block. AllGather gathers each rank's data and
   performs no reduction; the arithmetic of generic reduction remains on the GPU.
   Optimized RS first reduces into a node partial, and the network sends only the
   remote-owned partial; small messages / no-IPC may use CPU sum.
4. **Publish remote readiness**: publish the epoch or chunk-ready in the path's
   data/control transfer order. The receiver may consume the payload only after
   observing the control value; with multiple connections/rails, all related transfers
   must be covered.
5. **Consume and retire**: the CTA's SM, or a CPU-submitted H2D, moves the data into
   the output; after consumption actually completes, done, ACK and slot reuse advance
   per the path. Send completion does not mean the receiving GPU has consumed.

The specific control interfaces used by these stages follow.

### Invocation and optional CTA phases

`liteNetworkAllGatherPost()` has one thread of the CTA submit a whole-invocation
`NetworkAllGather` task. `networkPath` names the path selected by the executor, and the
CPU runs its schedule using the GPU-private network context. This is not a loop that
turns every network chunk into a generic Stage/Gather task.

For OrderedSmall, the CPU provides the slab/control addresses, offsets, epoch and
fields such as stageWithSm/receiveWithSm through `LiteNetworkControl`, and finally
release-publishes prepared=ticket. The calling CTA, on observing prepared, performs the
required SM packing/receive stages and then publishes deviceDone=ticket. The CPU waits
for those stages and the transport output to finish and then publishes task completion;
`liteNetworkAllGatherFinish()` waits for that completion and synchronizes the result to
the CTA. prepared/deviceDone use the FIFO ticket, while the epoch inside the descriptor
uses the algorithm's own numbering.

### Staging, transfer and remote readiness

* D2H readiness must be published only after the DMA has actually completed. Both the
  sender NIC and the local receive consumers must observe the readiness each needs; the
  remote rdmaReady does not mean other ranks on this node have finished their D2H.
* The CPU leader/group leader sends host data that is already prepared. The single-slab,
  NUMA-grouped and pipeline schedules each use their own chunks, connections and control
  layouts, and cannot be forced into one buffer-slot protocol.
* `signalRdmaReadyAtomic()` is the remote-ready publication operation of the current
  generic/NUMA schedule; it uses an RDMA atomic to advance the control word by the
  difference from the last published epoch. It is not a counter contended between CPU
  threads. The order of data transfer and control publication, and the connection
  completion handling, must be maintained as one protocol.
* Ordered-slot and one-rank pipeline use their own data/flag write protocols; not all
  remote-ready publication uses an atomic. The pipeline keeps a stable flag source for
  each chunk so that, while the NIC reads asynchronously, the source cannot be
  overwritten by the next chunk's ready value.
* Multi-rail readiness must cover the data transfer of every rail; waiting on only one
  connection must not let the receiver start copying.

### Receive, ACK and buffer reuse

The receiver waits for the corresponding remote control value and then consumes the
receive slab using the SM or H2D. Only after the required GPU/DMA consumers finish may
done/ACK be published or the slot released, per the path protocol. The sender finishing
its write, the receiver observing ready, and the receiver finishing consumption are
three different events.

The generic, NUMA and pipeline schedules each manage epochs, chunk readiness, ACKs and
slot reuse. They ultimately return whole-call completion to the user CTA through FIFO
completion. FIFO dequeue, a single CUDA event or one RDMA flush cannot stand in for the
whole-call completion condition.

### Progress and overlap

The pipeline overlaps D2H, RDMA and H2D through independent buffers/slots and explicit
dependencies. Merely adding GPU chunk flags does not automatically produce network
overlap; the CPU schedule must advance chunks per path and keep that path's window
size, ACK and slot-reuse conditions. For concrete chunk parameters see the AllGather
design.

Service-stream progress is independent of the calling kernel and must not wait for the
user kernel that is waiting on it to finish. Network scheduling must handle send,
receive and ACK together and must not let unbounded waiting in one direction block the
other direction's needed progress. GPU-to-CPU publication and NIC-to-receiver readiness
are different boundaries, each following its own protocol; a CTA barrier or a plain
volatile flag cannot replace these visibility conditions.

## Resource lifecycle and implementation boundaries

Initialization completes, on the host, the FIFO mapping, payload/control resource
preparation, IPC/NIC registration, connection exchange, and creation of nonblocking
service streams and events. The host fallback of IPC-preferred AllGather is also
prepared in advance, and the path is selected before communication is published.
Capacity, topology or runtime errors must not let ranks independently switch protocol
midway.

The CPU service does not wait for the user kernel that initiated the collective to
finish, and does not launch child kernels for device calls. Before releasing resources,
signal stop/abort, join workers, drain submitted copies, and then release events,
registrations and buffers. Detailed ownership is defined by
`DeviceCollectiveContext::releaseResources()`.

This document describes source behavior; it is not a conclusion about hardware
correctness or performance validation. After changing a primitive, verify the mapping
switches, alignment/tails, in-place, FIFO wrap, mixed message sizes, epoch/slot reuse
and error propagation across participants, and run the collective correctness preflight
in the corresponding single-node/multi-node GPU environment.

## AllReduce

AllReduce introduces no primitive of its own. `ReduceScatterAllGather` calls the
ReduceScatter and AllGather entries back to back; the two-node small and ring paths submit a
whole-invocation `AllReduce` FIFO task (`LiteTaskKind::AllReduce`) and execute the CTA copy/sum
phases their CPU schedule requests through the same `liteRsInvokeTask` loop as the
ReduceScatter network paths. See [allreduce.md](allreduce.md).

## ReduceScatter CTA phase primitive

`LiteTaskKind::ReduceScatter`, like network AllGather, is submitted to the FIFO per
invocation. The RS control area is a separate `LiteReduceScatterControl`; it does not
borrow AllGather's `prepared/deviceDone`. The CPU fills in up to four float sources, the
destination and the count, and release-publishes an incrementing `requested`. After the
calling CTA acquires it, it performs copy/sum; all threads system-fence and barrier, and
then `completed` is published. Only after the CPU observes completed may it read the
result, start dependent DMA, or reuse that phase descriptor. Before the first FIFO task
is published, all CTA lanes fence/barrier, which supports inputs just produced inside
the user kernel.

This primitive launches no extra kernel. A source can be local CUDA scratch, staged IPC
scratch, or a mapped host payload; the operator is float sum only, and other types/ops
continue to use the existing generic implementation. RS's CPU sum reuses
`cpu_switch/cpu_reduction.hpp`, and the RS policy may force scalar.

RS maintains local ready/pair/done and remote ready/ACK within its own fixed-capacity
layout; the two-rank network uses five slots and the hierarchy four. CUDA event
completion, NIC write completion and CTA consumption completion are tracked separately.
Per chunk, D2H staging completion (a per-slot event) gates the RDMA post, which is
decoupled from chunk preparation; ACK follows CTA/H2D consumption only.
For the complete scratch address layout, topology schedules and selection rules see
[reducescatter.md](reducescatter.md).
