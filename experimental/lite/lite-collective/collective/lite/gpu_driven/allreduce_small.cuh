#pragma once

// Two-node small-message AllReduce paths. Each is a __device__ entry that submits
// one whole-invocation `AllReduce` task; the like-named CPU schedule in
// allreduce_schedule.hpp drives DMA/RDMA and requests CTA phases.
//
// SmallMapped (2n*2g / 2n*4g, 0 < B <= 64 KiB, mapped host slab): every rank stages
// its complete input into the node's mapped host slab with the CTA and publishes
// readiness; the node leader (local rank 0) CPU-reduces the local rows, exchanges
// one node partial over RDMA and CPU-reduces the two partials; every rank waits for
// the final row and copies the complete result out with the CTA.
// CPU: runSmallMappedAllReduce2Node.
static __device__ __forceinline__ int liteAllReduceSmallMappedBlock(
    mscclppDeviceCollectiveHandle_t const& h, float const* src, float* dst,
    size_t bytes, LiteAllReducePlan const& plan) {
  if (!h.reduceScatterMapped || !h.reduceScatterView.hostSendDevice)
    return mscclppDeviceCollectiveInvalidUsage;
  return liteArInvokePath(h, src, dst, bytes, plan);
}

// SmallTwoLeader (2n*4g, 64 KiB < B <= 128 KiB, even count): input is staged by DMA;
// local ranks 0 and 2 each reduce one half of the tensor across the four local
// ranks, exchange it with the corresponding remote leader and reduce the two node
// partials; every rank assembles the complete output from both final halves.
// CPU: runSmallTwoLeaderAllReduce2Node.
static __device__ __forceinline__ int liteAllReduceSmallTwoLeaderBlock(
    mscclppDeviceCollectiveHandle_t const& h, float const* src, float* dst,
    size_t bytes, LiteAllReducePlan const& plan) {
  return liteArInvokePath(h, src, dst, bytes, plan);
}
