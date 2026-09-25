# GPU-driven collectives

## API and execution contract

Call `mscclppGetDeviceCollectiveHandle` collectively once, then pass its handle by value into a user CUDA kernel. Every thread of exactly one participating CTA per rank calls `liteAllGatherBlock`, `liteAllReduceBlock<T>`, or `liteReduceScatterBlock<T>` in the same order, with matching sizes and operations. A handle cannot be used concurrently by multiple CTAs or streams. Resources remain owned by the communicator; synchronize the caller stream before destroying it.

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
| `service.hpp` | Per-rank D2H/H2D service, event polling and descriptor completion |
| `device_collectives_bench.cu` | Native NCCL comparison with untimed correctness preflight |
| `benchmark.sh` | Build, MPI launch, bounded execution and benchmark report |

## AllGather selection

Host policy is snapshotted during collective initialization. Shared-host AllGather requires explicit `MSCCLPP_NCCL_HOST_ALLGATHER=1`, total output within the configured minimum and 1 GiB maximum, and no capture. The benchmark explicitly opts into host mode unless configured otherwise. Mapped, cooperative-phase and DMA branches retain their priority, alignment and capability conditions. Cooperative phases execute inside the single caller CTA. Per-rank chunks default to B up to 1 MiB, 1 MiB through 32 MiB, then 4 MiB, retaining the host tuning variables.

CUDA IPC AllGather requires host AllGather disabled, IPC event synchronization enabled, total output at least 8 MiB, and no capture. It forwards chunks around a ring using per-hop readiness and consumption credits. Progress uses device flags; no event or stream waits depend on completion of the calling kernel.

Two-node small ordered exchange takes priority over NUMA split. Thresholds are total < 2 MiB for P=1 and total < 128 KiB otherwise. P=1 preserves the 128-byte single-thread and 256-byte register-copy cases; other mapped totals below 64 KiB use an aligned data-plus-flag segment per source. The proxy writes each segment's payload before its flag on the same QP. P=2 preserves the `(256,512)`-byte packing gap and independently selects mapped receiving up to 4 KiB. P>2 small messages use DMA.

Normal small slots hold complete output in global-rank order. DMA receive issues one full-output H2D when required, otherwise copies self directly and transfers remote data. Unmapped small slots use the CPU-repacking fallback and one full-output H2D. Capability decisions are agreed during initialization; ranks do not independently switch protocols after publication.

The P=1 pipeline uses 512 KiB chunks for 1 MiB <= B <= 1 GiB, a network send window of one, and independent D2H/H2D streams. Generic single-slab uses 512 KiB for P=2 and 2 MiB otherwise. Each chunk owns an epoch and one of two slots; the CTA submits the next D2H before consuming the current chunk. For P>1 and B >= 512 KiB, self output is copied once before the pipeline (skipped in-place), while the input is still staged for peers.

Generic AllGather staging packs a contiguous node block at the current chunk stride. The proxy sends that block incrementally, splitting writes at 2 MiB. Dual rail applies only to non-NUMA generic exchange when the current node block is at least 2 MiB. Both rails finish before primary-QP readiness. Small ordered exchange and the P=1 specialized pipeline do not stripe.

## NUMA groups

Initialization gathers each rank's actual GPU NUMA identity and available IB count on its owning host. Contiguous changes in GPU NUMA identity form groups, bounded by available transports. Nodes must have identical group boundaries; asymmetric layouts use the single-slab path. P=2 is excluded.

Each group owns a NUMA-placed shared host buffer, registration, leader connection and proxy. A group's leader selects its GPU-local IB transport and exchanges with the matching remote leader. All local ranks map and consume every group's output. Group chunks are bounded at 16 MiB per rank and individual writes at 2 MiB. Small ordered calls continue to use the baseline context.

The NUMA context owns separate epochs, task FIFO and service resources, prepared during host initialization. The primary device handle references the initialized NUMA handle. Switching between small messages, NUMA AllGather and reductions cannot reuse the other context's epochs. Every local rank publishes consumption to every group, and slot reuse waits for all group NIC completions plus local reader completion. NUMA splitting does not launch additional GPU CTAs.

## Ownership, errors and cleanup

FIFO completion releases descriptors only. D2H readiness follows actual CUDA event completion; H2D completion is published only after both peer-range streams finish. GPU writers fence every participating thread before publishing readiness. NIC payload and readiness use ordered writes, with both rails covered before publication.

Send slots wait for NIC completion and local readers. Remote receive slots wait for all local consumers and a returned ACK before the sender can reuse them. Epochs distinguish successive uses. Compact flags occupy their own aligned words, including for odd byte counts.

Services catch errors and publish fatal sentinels; device waits have a finite deadline and poison failed handles. Network proxies attempt to notify the remote node on their established connection. A failed handle must be destroyed after its user stream completes. Cleanup stops and joins workers, drains issued copies, then destroys events, registrations and buffers.

## Limits and validation status

Three or more nodes (the guide's Case 6 extension) remain unsupported. Two-node reductions still stage and exchange their complete input per invocation, rather than using a specialized reduction pipeline. Allocations scale with the requested maximum capacity; NUMA contexts additionally allocate group resources at initialization, capped at 16 MiB per rank per slot.

The development machine has no CUDA/IB toolchain or hardware. Portable protocol simulation and Clang device-code generation passed, but these do not establish CUDA cache visibility, real copy-engine overlap, IB behavior, performance parity, or speedup. No hardware benchmark result is claimed. See [validation.md](validation.md) for exact checks and target-machine commands.
