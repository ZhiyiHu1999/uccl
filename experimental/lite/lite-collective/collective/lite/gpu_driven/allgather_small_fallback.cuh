#pragma once

// executeSmallFallbackSchedule: D2H, CPU repack, full H2D and ACK.
// The service runs the same transport schedule as the CPU entry point.
// GPU publication replaces the host invocation/input-stream dependency.
static __device__ __forceinline__ int liteAllGatherSmallFallbackBlock(
    mscclppDeviceCollectiveHandle_t const& h, char const* src, char* dst,
    size_t bytesPerRank, LiteAllGatherPlan const&) {
  unsigned long long ticket = liteNetworkAllGatherPost(
      h, src, dst, bytesPerRank, LiteDeviceAllGatherPath::SmallFallback);
  return liteNetworkAllGatherFinish(h, ticket);
}
