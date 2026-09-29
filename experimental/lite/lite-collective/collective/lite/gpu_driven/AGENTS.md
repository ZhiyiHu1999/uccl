# GPU-driven collectives

## Project identity and target

- Implement device-callable collectives (AllGather, ReduceScatter, AllReduce) that can run inside a user CUDA kernel after one host-side collective initialization.
- Optimize especially for these topologies: 1n × 2g, 1n × 4g, 2n × 1g, 2n × 2g, and 2n × 4g, where *a*n × *b*g means *a* nodes with *b* GPUs per node.
- The target hardware does not support NVLink or GPUDirect RDMA (GDR).

## Architecture

### Primitive Layer

The primitive layer provides data movement and synchronization for device-callable
collectives. Within a node, GPU threads copy/reduce mapped payloads, or submit
DMA work to a CPU service through a task FIFO. Between nodes, CPU services move
GPU data through pinned host buffers and RDMA, coordinating readiness,
completion and buffer reuse with the calling GPU.

[design/primitives.md](design/primitives.md) describes the implemented Intranode
and Internode data paths, copy/reduction helpers, acquire/release operations,
FIFO protocol, DMA/RDMA progress, and resource lifecycle. Read it before changing
these operations, and update it together with initialization, error propagation,
payload retirement or cleanup changes. 

### Collective Layer

Collective algorithms are composed of primitive operations. Collective layer is in charge of optimization path selection and primitives scheduling.

#### AllGather

* Read [allgather.md](design/allgather.md) before modifying AllGather selection,
  initialization, executors, or service schedules. It defines the backend
  preference contract and each Intranode/Internode optimization path.
* Keep AllGather design requirements in that document and update it together
  with algorithm or selection changes. Common Architecture remains here;
  benchmark instructions and results remain in `docs/`.

#### ReduceScatter

* Read [design/reducescatter.md](design/reducescatter.md) before modifying
  selection, scratch layouts, CTA phases, transport schedules, or setup.
  The CPU references are `nccl/ReduceScatter/{single-node,multi-node}.cu`,
  especially `runLiteInterReduceScatter`; do not substitute AllReduce's
  different `runSendRecvReduceScatter` dispatch.
* `reducescatter_plan.hpp` owns float/sum policy and topology/size selection.
  `reducescatter.cuh` switches on `plan.path` to one `__device__` entry per path
  (`reducescatter_{ipc,host,two_rank,hierarchical}.cuh`, named
  `liteReduceScatter<Path>Block`). Primitives are split by executor: FIFO primitives
  (`RsCopy`, `RsHostSum`, `RsBarrier`, executed by the service) for anything that needs
  the CPU, copy engines or NIC, and CTA primitives (`liteRsSum`) for SM work, declared
  in `reducescatter_primitives.cuh`. Single-node paths are composed from them on the
  device; the two-node paths still submit one whole-invocation task through
  `liteRsInvokePath` with a CPU schedule. Adding a path means adding its plan branch,
  its device entry (and, for two-node paths, its CPU schedule case), and its row in the
  design document.
  `reducescatter_generic.cuh` retains the
  arithmetic-template sum/min/max fallback.
* `reducescatter_service.hpp` owns GPU-private resources and phase primitives;
  `reducescatter_local_schedule.hpp` and `reducescatter_network_schedule.hpp`
  implement local and two-node schedules. Do not call CPU collective entry
  points, native NCCL, or launch kernels from these schedules.
* Capacity bounds the **full input**, while selection thresholds use output
  shard bytes B or full input T as documented. Preserve in-place ownership.
  Generic reductions still require mapped slabs; optimized float/sum supports
  DMA-only host payloads through CPU reduction/H2D or GPU scratch.
* Compare RS policy and capability collectively before starting workers.
  Keep FIFO tickets, CTA phase sequences, RS chunk epochs, and generic/AG
  epochs separate. Only consumer completion permits ACK and scratch reuse.
* Policy defaults in `reducescatter_plan.hpp` must track the CPU reference per layout
  (2n×2g and 2n×4g differ in lead, split-final, host-read final add, async final and
  eager post). When changing a CPU default, mirror it in the plan and in the policy
  table of the design document; deliberate deviations belong in its "Known
  deviations" list.
* Use the existing benchmark for correctness/performance coverage. Include
  tails, repeated/mixed sizes, in-place, all five target layouts, IPC/no-IPC,
  and mapped/DMA paths. Record missing CUDA/IB hardware validation explicitly.

#### AllReduce

TBD

### Benchmark command-line requirements

- Benchmarking should compare the performance of gpu-driven collectives with the performance of native NCCL host-driven collectives. The comparison should be fair by controlling the number of SMs in both cases to 1.
- Support `-c/--collective allgather|allreduce|reducescatter|all` (default `all`).
  Execute correctness preflight, warmup, timing, and NCCL comparison only for the
  selected collective; reports must omit unselected sections.
- The script and executable must support nccl-tests-style sweeps with
  `-b BEGIN -e END -f FACTOR`, binary size suffixes `B/K/M/G` (case-insensitive),
  `-w WARMUPS`, `-n ITERS`, and `-g 1` (one GPU per MPI process).
- Require both range bounds. Multiply by an integer factor >= 2 (default 2)
  while within the inclusive end bound; prevent arithmetic overflow.
- Preserve positional size lists and their order, including repeated sizes.
  Reject mixing ranges with positional sizes, invalid values and unsupported GPU counts.
- CLI iteration counts override environment defaults. Reports must record the
  effective counts. Help must work without GPU initialization.
- Do not impose a fixed wall-clock timeout on benchmark runs. Honor requested
  ranges and iteration counts; never silently reduce iterations or treat skipped
  paths/interrupted runs as successful measurements.

## Other specifications

- Implement the target functions described in the guideline.
- The source code of GPU-driven collectives should reside in the standalone directory `gpu_driven/`
- Keep architecture and algorithm designs in `design/`; put benchmarking instructions and results in `docs/`.
- Uncommited codes could be abandoned if the given guideline conflicts with the code.
- HostCooperative AllGather may use multiple blocks with cooperative launch;
  all other device entry paths retain the single participating CTA contract.
  Benchmark all paths with exactly one block to retain the one-SM budget.
  Multi-block callers must select HostCooperative on every rank, participate
  with the complete grid, pass `cooperativeGrid=true` to `liteAllGatherBlock`,
  and respect cooperative launch occupancy limits.
  Never concurrently invoke separate operations on the same handle.
- Development for files with testing purpose should be limited to benchmarking, other testing/validating files should be removed.
- Keep changes scoped to the requested feature. Preserve existing public APIs and `mscclpp` / `MSCCLPP_*` names unless the task requires an interface change.
- For a feature request, identify the affected collective, backend, message-size range, and execution model from the request and current code. State reasonable assumptions and proceed; ask only when an unresolved choice changes required behavior or compatibility.
- Treat current limits as implementation boundaries, not permanent prohibitions. If a requested feature expands them, update device code, host setup, validation, and documentation together.
- If this `AGENTS.md` got updated, and the code base has been generated based on the previous version of `AGENTS.md`, all temporary compromising restricted to previous AGENTS.md should be abandoned for next run.
- Follow this guide when implementing features and fixes in this directory, using the architecture described above.
- Keep this guide focused on code navigation, development decisions, invariants, and validation.
- Keep generated build outputs and temporary benchmark artifacts out of source changes. Avoid unrelated refactors and edits to thrid-party's code.

## Finish the task

- Summarize the behavior implemented, relevant files, validation performed, and remaining limitations.
