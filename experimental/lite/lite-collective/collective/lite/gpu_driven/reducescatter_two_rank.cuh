#pragma once

// Included from reducescatter.cuh after liteRsInvokePath. Each function below is
// the device entry of exactly one plan path. It checks that path's own
// prerequisites (the handle-wide IPC / mapped capabilities agreed at setup) and
// submits the invocation; the like-named CPU schedule in
// reducescatter_*_schedule.hpp drives the path's DMA/RDMA and requests the CTA
// arithmetic phases. No entry calls a CPU collective or launches a kernel.

// TwoRankSmall (2n*1g, B <= 512 KiB): one RDMA message each way; mapped CTA stage/final or DMA + CPU final. CPU: runTwoRankSmallHostReduceScatter.
static __device__ __forceinline__ int liteReduceScatterTwoRankSmallBlock(
    mscclppDeviceCollectiveHandle_t const& h, float const* src, float* dst,
    size_t bytes, LiteReduceScatterPlan const& plan) {
  return liteRsInvokePath(h, src, dst, bytes, plan);
}

// TwoRankPipeline (2n*1g): chunked exchange of the peer-owned shard, five slots. CPU: runTwoRankReduceScatter / runTwoRankPipelinedChunks.
static __device__ __forceinline__ int liteReduceScatterTwoRankPipelineBlock(
    mscclppDeviceCollectiveHandle_t const& h, float const* src, float* dst,
    size_t bytes, LiteReduceScatterPlan const& plan) {
  return liteRsInvokePath(h, src, dst, bytes, plan);
}

