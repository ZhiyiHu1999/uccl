#pragma once

// executeNumaSchedule: independent group epochs/slots, own-group D2H,
// group/node-ordered H2D and CPU-reference group retirement.
// The service runs the same transport schedule as the CPU entry point.
// GPU publication replaces the host invocation/input-stream dependency.
static __device__ __forceinline__ int liteAllGatherNumaSplitBlock(
    mscclppDeviceCollectiveHandle_t const& h, char const* src, char* dst,
    size_t bytesPerRank, LiteAllGatherPlan const&) {
  unsigned long long ticket = liteNetworkAllGatherPost(
      h, src, dst, bytesPerRank, LiteDeviceAllGatherPath::NumaSplit);
  return liteNetworkAllGatherFinish(h, ticket);
}
