# Validation and benchmarking

## Development-host checks

## The development host is macOS without nvcc, CUDA headers, GPUs or IB.

* Portable selector checks passed for host mapped/cooperative/DMA boundaries, chunk sizes, the P=2 packing gap, small-message priority, NUMA eligibility, P=2 NUMA exclusion, fallback and capture rejection.
* A temporary C++ simulation compiled the actual `gpu_collectives.cuh`, `host_context.hpp` and `service.hpp` against CUDA/bootstrap/network substitutes. It passed 2n×1g, 2n×2g, 2n×4g single-slab, dual-rail and NUMA-group exchanges, including mapped and DMA-only operation, odd sizes, in-place/unaligned output, slot/FIFO wrap and mixed AllGather/reduction epochs. Runs were bounded to 15 seconds. It models one CPU thread per CTA and immediate mock copies/writes; it does not model hardware visibility, asynchronous DMA timing or transport failures.
* Clang generated sm_80 PTX for a translation unit instantiating all three device routines with declaration shims. Host context and benchmark syntax also passed shim-based checks. These are not nvcc builds against real CUDA/MPI headers.
* Shell syntax and diff whitespace checks passed.
* `make collective` and `make -C nccl` stopped at missing `cuda_fp16.h` and a pre-existing incomplete `std::string` declaration in `core/errors.hpp` under the local libc++ toolchain. The benchmark build stopped because `nvcc` is absent.

Temporary validation harnesses were kept outside the source tree. Per AGENTS.md, standalone correctness executables and test-only source files were removed; correctness preflight now resides in the benchmark.

## Target-machine build

From `experimental/lite/lite-collective`:

```sh
make all
make -C nccl all
```

`make -C nccl all` builds the compatibility libraries and the GPU-driven
benchmark at `nccl/build/device_collectives_bench`; it requires MPI as well as
CUDA. Set `MPI_HOME` (or `MPI_CXX`) for a non-default MPI installation.
The standalone `device-collectives-bench` target remains available and is also
invoked automatically by `benchmark.sh`.

Set `NCCL_BASELINE_LIB` to an actual native NCCL library, not the UCCL compatibility library. The benchmark sets NCCL's minimum/maximum CTAs and channels to one, and disables GDR before either communicator is initialized. The GPU routine launches exactly one CTA. Confirm actual native NCCL kernel grid dimensions with a profiler on the selected NCCL version before claiming a measured one-SM comparison.

## Size sweeps and command-line options

Use `-c allgather` (or `--collective allgather`) to run only AllGather, including
its correctness preflight and native NCCL comparison. Other choices are
`allreduce`, `reducescatter`, and `all` (the compatibility default). Unselected
collectives are not run and are omitted from the report.

```sh
NP=2 CUDA_VISIBLE_DEVICES=0,1 UCCL_GPU_DRIVEN_BACKEND=cuda_ipc \
  MSCCLPP_NCCL_HOST_ALLGATHER=0 MSCCLPP_NCCL_CUDAIPC_EVENT_SYNC=1 \
  bash collective/lite/gpu_driven/benchmark.sh \
  -c allgather -g 1 -b 4M -e 1G -f 2 -w 1 -n 3
```

Two-rank IPC AllGather requires at least 4 MiB input per rank. Starting at 128B
uses host memory below that threshold and IPC above it when peer access is available.
`MSCCLPP_NCCL_HOST_ALLGATHER` no longer affects GPU-driven selection. There is no fixed
wall-clock timeout on the benchmark run.

Both `benchmark.sh` and `device_collectives_bench` accept
`-b BEGIN -e END -f FACTOR -g 1 -w WARMUPS -n ITERS`.
Both bounds are required. The default factor is 2 (integer >= 2); multiply until
exceeding END, including END only if the progression reaches it. Sizes accept
case-insensitive binary `B/K/M/G` suffixes. Positional sizes remain supported in
order, but cannot be mixed with ranges. Only one GPU per MPI process (`-g 1`)
is supported; `NP` sets the process count. CLI `-w/-n` override environment
settings. `-h`/`--help` displays usage; script help requires no build.

```sh
# Host mode: powers of two from 128 B through 1 GiB.
NP=2 CUDA_VISIBLE_DEVICES=0,1 bash collective/lite/gpu_driven/benchmark.sh \
  -g 1 -b 128B -e 1G -f 2 -w 1 -n 3

# CUDA IPC: two-rank AllGather skips per-rank input below 4 MiB.
NP=2 CUDA_VISIBLE_DEVICES=0,1 UCCL_GPU_DRIVEN_BACKEND=cuda_ipc \
  MSCCLPP_NCCL_HOST_ALLGATHER=0 MSCCLPP_NCCL_CUDAIPC_EVENT_SYNC=1 \
  bash collective/lite/gpu_driven/benchmark.sh \
  -g 1 -b 128B -e 1G -f 2 -w 1 -n 3

# Small-message latency and a single large-message throughput point.
NP=2 bash collective/lite/gpu_driven/benchmark.sh -b 128 -e 64K -f 2 -w 100 -n 1000
NP=2 bash collective/lite/gpu_driven/benchmark.sh -b 256M -e 256M -w 20 -n 50
```

The script runs the requested sweep without a fixed wall-clock timeout. Use
Ctrl+C to interrupt a run; an interrupted run is not a completed sweep.
Set distinct `RESULT_FILE` values to retain separate reports. Bytes denote AllGather input, AllReduce full input/output, and
ReduceScatter output shard per rank; RS input is that size times rank count.
Staging capacity is allocated from the maximum requested size (times rank count
when ReduceScatter is selected),
so large sweeps require additional memory beyond input/output allocations.

## Benchmark runs

`benchmark.sh` launches MPI without a timeout wrapper. Use targeted sizes and small iteration counts for initial checks, then run the desired full sweep. Internal protocol stall/error checks remain in place; these are separate from total benchmark duration. Set `MPI_HOME`, `NCCL_BASELINE_LIB`, NIC/bootstrap variables and host names for the testbed. Forwarded host tuning variables are listed in the script.

```sh
export NCCL_BASELINE_LIB=/path/to/native/libnccl.so
export WARMUP_ITERS=1 ITERS=3

# One node: explicit host SHM, including DMA-only mapping policy.
NP=2 CUDA_VISIBLE_DEVICES=0,1 collective/lite/gpu_driven/benchmark.sh 1048577
NP=4 CUDA_VISIBLE_DEVICES=0,1,2,3 collective/lite/gpu_driven/benchmark.sh 524289
NP=4 CUDA_VISIBLE_DEVICES=0,1,2,3 MSCCLPP_NCCL_HOST_ALLGATHER_MAP_SLAB=0 \
  collective/lite/gpu_driven/benchmark.sh 4096

# CUDA IPC needs T >= 8 MiB, host SHM disabled, and peer accessibility.
NP=4 CUDA_VISIBLE_DEVICES=0,1,2,3 UCCL_GPU_DRIVEN_BACKEND=cuda_ipc \
  MSCCLPP_NCCL_HOST_ALLGATHER=0 collective/lite/gpu_driven/benchmark.sh 2097153

# Two nodes: substitute real node names. Rank placement must be contiguous.
NP=2 HOSTS=nodeA,nodeB CUDA_VISIBLE_DEVICES=0 \
  collective/lite/gpu_driven/benchmark.sh 1048577
NP=4 HOSTS=nodeA,nodeB CUDA_VISIBLE_DEVICES=0,1 \
  collective/lite/gpu_driven/benchmark.sh 524289
NP=8 HOSTS=nodeA,nodeB CUDA_VISIBLE_DEVICES=0,1,2,3 \
  collective/lite/gpu_driven/benchmark.sh 2097153
```

AllGather accepts arbitrary byte sizes, including odd tails. Float reductions are skipped for sizes not divisible by four. Before timing, the benchmark checks changing input across repeated calls inside a single kernel, aligned/unaligned and in-place/out-of-place AllGather, plus int/float Sum/Min/Max reductions. Timed GPU output and accumulated return status are checked before native NCCL overwrites output. Untimed checks and MPI barriers are excluded from latency samples.

Required hardware matrix: all five target topologies; both sides and equality of every documented threshold; repeated mixed sizes crossing small/NUMA dispatch; asymmetric versus symmetric NUMA layouts; more than 16 MiB per-rank NUMA input; single/dual NIC generic blocks around 2 MiB; host tuning overrides; cooperative capability disabled; in-place and odd tails; mapped slabs disabled; repeated FIFO/slot wrap; injected service/transport failures; and explicit capture rejection. Test IPC only where CUDA P2P access is available. The same benchmark process iterates requested sizes in order, so passing a small/large/small sequence exercises context switching.

Generated measurements default to `.tmp/gpu-driven-benchmarks/`. No CUDA/IB measurements from this change are available yet; do not infer performance from the portable simulation.

## IPC DMA ring replacement checks

The IPC ring now uses full-block D2D DMA pushes through the CPU service and a
collectively registered output region. Registration/unregistration is outside
benchmark timing. Run 2- and 4-rank single-node IPC sweeps, including the 8 MiB
total eligibility boundary and large blocks. Existing preflight covers repeated
in-kernel calls, in-place/out-of-place operation and unaligned output offsets;
size sweeps also replace and unregister output mappings between allocations.
Inspect a CUDA timeline for D2D copies on the independent service stream, no
payload copy kernels, and no dependency on the waiting caller kernel. Compare
against the old IPC measurements without claiming a speedup before rerunning.

A temporary host simulation of the actual isolated ring routine checks 2/4/8
ranks, unaligned destinations, in-place/out-of-place inputs and 20 repeated epochs.
This models copies with host memcpy and acquire/release flags, not CUDA memory
visibility or asynchronous copy-engine behavior. The local build is blocked by
missing nvcc; hardware validation remains required.

## Host AllGather CPU-reference alignment

The host executors now reserve one epoch/slot for the complete invocation.
A temporary C++ protocol simulation exercised the actual executor headers and
the service enqueue helper with asynchronous mock DMA streams: 2/4/8 ranks,
18 epochs, one/four cooperative CTAs, in/out-of-place, non-aligned DMA tails,
allocation pitch larger than output pitch, and packed/DMA path switching passed.
This models one thread per CTA and does not validate CUDA warp execution,
cache visibility, cooperative-launch residency, or real DMA performance.
No standalone test files were added to the repository.

The normal benchmark remains one block (one-SM budget), including Cooperative.
Its existing preflight tests aligned/unaligned and in/out-of-place calls repeated
inside a kernel. On CUDA hardware, rerun host AllGather for 2 and 4 ranks with
the usual size sweep; also compare `MSCCLPP_NCCL_HOST_ALLGATHER_MAP_SLAB=0`
and `MSCCLPP_NCCL_HOST_ALLGATHER_SELF_KERNEL=0/1`. Multi-block cooperative
correctness requires a cooperatively launched caller with all grid threads
participating and occupancy-safe grid dimensions; it is not part of the
one-SM performance measurement.

Local `make -C nccl device-collectives-bench` could not run because `nvcc` is
absent. Real CUDA compilation and hardware correctness/performance are pending.

## Multi-node executor extraction

A temporary C++ trace comparison compiled the five actual extracted executor
headers and the original shared executor against instrumented primitives.
4,704 cases compared return values, copy ranges, ready/done calls, FIFO task
fields, compact/generic flags, and barriers across 2n×1g/2n×2g/2n×4g, all ranks,
in/out-of-place, mapped/unmapped, NUMA selection, odd tails, multiple chunks,
and injected wait/task-post failures. Traces matched. This verifies structural
equivalence in a one-thread-per-CTA model, not CUDA or RDMA correctness.
The test is temporary and is not shipped in the repository.

`make -C nccl device-collectives-bench` stopped because `nvcc` is unavailable
on the development machine. Real CUDA compilation and multi-node hardware
validation of the extraction remain pending.

## Shared CPU/device network schedules

The subsequent rewrite replaces the extracted generic GPU schedules with shared
CPU-reference transport schedules. The earlier 4704-case extraction test is
historical and does not validate this rewrite.

Local checks for the rewrite:

- Source-body comparison confirms unchanged SingleSlab, SmallFallback, NUMA
  scheduling and pipeline/copy primitives after factoring. Group-input polling
  adds a device-only cancellation hook; native CPU calls retain their old wait.
- An actual shared OrderedSmall schedule plus actual device entry-point mock
  simulation passed 84 cases: 2n×1g/2g/4g, mapped/ DMA-only, in/out-of-place,
  repeated epochs and dynamic-slot wrap. Service-side kernel stubs fail if
  invoked; none were invoked. One CPU thread models each CTA.
- An actual shared OneRankPipeline simulation checks whole-message epochs, all
  D2H submissions before RDMA sending, odd tails, in-place output, single/multiple
  slots and ACK-controlled reuse. CUDA events/copies and RDMA are mocked.
- Clang CUDA sm_80 syntax checking of the actual device header and all three
  collectives passed with declaration-only CUDA/cooperative-group shims. This
  is not an nvcc build or verification against the real CUDA SDK.
- `make -C nccl all` stopped on unavailable CUDA headers and an existing libc++
  incomplete-string error in `core/errors.hpp`. Hardware compilation, linker
  verification and multi-node CUDA/IB runs remain required.

Benchmark launch syntax is unchanged. Run allgather on 2n×1g/2g/4g with the usual
size sweeps. Keep 100 warmups and 1000 samples for comparable measurements; the
existing preflight runs repeated in/out-of-place and unaligned calls outside
timed samples. No standalone test programs are added to the repository.

## OrderedSmall size-transition flag regression

The device ordered/compact slot flag now uses exact epoch equality. A new slot
layout can place the flag over old payload; accepting an arbitrary larger value
can let the GPU consume data before the current RDMA transfer. FIFO completion,
prepared descriptors and monotonic local-ready counters still use >=.

A temporary host harness executing the actual wait helpers verifies old-payload
rejection followed by exact-epoch acceptance, unchanged monotonic waits, timeout,
service errors and poison handling. Clang CUDA syntax checking with declaration
shims passes. The nvcc benchmark build cannot run on this development machine
(`nvcc` unavailable); real CUDA/IB validation remains pending.

On two nodes, with the usual HOSTS/NIC environment, repeat the reproducer:

```bash
NP=2 CUDA_VISIBLE_DEVICES=0 bash collective/lite/gpu_driven/benchmark.sh \
  -c allgather -g 1 -b 128B -e 256B -f 2 -w 100 -n 1000 \
  2>&1 | tee gpu-driven-size-transition.log
```

Both sizes must pass preflight and finish. Also run the full size sweep to cover
further layout transitions. This targeted fix does not reset epochs or change
slot layout, network ordering, or the CPU reference implementation.

## SingleSlab local D2H consumer ordering

`exchangeGroupChunk` now waits for all local group D2H publications on non-leader
consumers of GPU-driven service calls before issuing output copies. The leader already does this before its
RDMA send. Incoming remote readiness alone does not make local send-slab rows
safe to read. The additional wait is gated by `activeDeviceCall`, which is set
only by the device-service invocation scope. Native CPU-driven execution skips
this added wait. The existing device cancellation hook is retained.

A temporary concurrent host harness executing the actual consumer-gate source
fragment passed with remote readiness already published and local D2H ranks
released individually. Output consumption remained blocked until every local
rank was ready, for 2/4-rank groups and zero/nonzero group bases. This does not
model CUDA/IB memory visibility. `make -C nccl all` remains blocked on missing
CUDA headers and the existing libc++ incomplete-string error on macOS.

Hardware regression: run 2n×4g directly at `-b 1M -e 1M -w 20 -n 50`, then
`-b 1M -e 1G -f 2 -w 20 -n 50`; also repeat the 2n×2g sweep. The benchmark
preflight must pass both in-place modes and aligned/unaligned buffers. Confirm
the actual selected network path if failures persist; this correction targets
SingleSlab, not a claim that every possible preflight failure is resolved.

## CPU/GPU implementation isolation

The three native files (`allgather_intranode.cu`, `allgather_multinode.cu`,
`cpu_staging_channel.hpp`) are restored byte-for-byte to `0a1edbea`.
The GPU-private network implementation is now `gpu_driven/network_service.cu`;
GPU host staging uses `gpu_driven/host_staging_buffer.hpp`. Earlier references
in this log to shared schedules describe the pre-isolation implementation.

Checks performed:
- Byte comparisons against the baseline pass for all three restored files.
- Whitespace-normalized body comparisons pass for the migrated SingleSlab,
  SmallFallback, NUMA, pipeline, exchange/copy helpers and device service bridge.
- The private staging class matches the preceding ownership-fixed version after
  normalizing its private type names and relative include path.
- The 84-case OrderedSmall host simulation passes against the isolated schedule.
  Native kernel-launch branches and native entry wrappers were removed from the
  private translation unit; the caller CTA still performs its SM phases.
- Make dry-run resolves the new CUDA compilation rule. Actual compilation stops
  because nvcc is unavailable. Real CUDA/MPI linking and hardware regression are
  still required; this refactor does not claim to fix the outstanding 2n×4g issue.


Backend preference regression (run on CUDA hosts):

```sh
# Small -> IPC -> small exercises independent host/IPC epochs in one process.
NP=4 CUDA_VISIBLE_DEVICES=0,1,2,3 UCCL_GPU_DRIVEN_BACKEND=cuda_ipc \
  bash collective/lite/gpu_driven/benchmark.sh -c allgather 128B 2M 128B 2M
# Event-sync disabled must use host for every size.
NP=4 CUDA_VISIBLE_DEVICES=0,1,2,3 UCCL_GPU_DRIVEN_BACKEND=cuda_ipc \
  MSCCLPP_NCCL_CUDAIPC_EVENT_SYNC=0 \
  bash collective/lite/gpu_driven/benchmark.sh -c allgather 128B 2M
# With HOSTS set to two nodes, this must select host_rdma.
NP=2 CUDA_VISIBLE_DEVICES=0 UCCL_GPU_DRIVEN_BACKEND=cuda_ipc \
  bash collective/lite/gpu_driven/benchmark.sh -c allgather 128B 4M
```

Also repeat with the preference unset, host enable set to both 0 and 1,
and peer access unavailable. Check per-size selected_backend log lines.
