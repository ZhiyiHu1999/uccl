#pragma once

// runOneRankChunkPipeline: one epoch per full message, all D2H queued,
// 512 KiB chunks, receive-paced send window of one, whole-slot retirement.
// The service runs the same transport schedule as the CPU entry point.
// GPU publication replaces the host invocation/input-stream dependency.
static __device__ __forceinline__ int liteAllGatherOneRankPipelineBlock(
    mscclppDeviceCollectiveHandle_t const& h, char const* src, char* dst,
    size_t bytesPerRank, LiteAllGatherPlan const&) {
  unsigned long long ticket = liteNetworkAllGatherPost(
      h, src, dst, bytesPerRank, LiteDeviceAllGatherPath::OneRankPipeline);
  return liteNetworkAllGatherFinish(h, ticket);
}
