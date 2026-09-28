#pragma once

// executeSingleSlabSchedule/exchangeGroupChunk: CPU-reference slots,
// D2D self copy, group-batched receive and native dual-rail/QP ordering.
// The service runs the same transport schedule as the CPU entry point.
// GPU publication replaces the host invocation/input-stream dependency.
static __device__ __forceinline__ int liteAllGatherSingleSlabBlock(
    mscclppDeviceCollectiveHandle_t const& h, char const* src, char* dst,
    size_t bytesPerRank, LiteAllGatherPlan const&) {
  unsigned long long ticket = liteNetworkAllGatherPost(
      h, src, dst, bytesPerRank, LiteDeviceAllGatherPath::SingleSlab);
  return liteNetworkAllGatherFinish(h, ticket);
}
