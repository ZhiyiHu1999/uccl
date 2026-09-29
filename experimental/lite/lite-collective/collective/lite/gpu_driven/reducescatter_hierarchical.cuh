#pragma once

// Included from reducescatter.cuh after liteRsInvokePath. Each function below is
// the device entry of exactly one plan path. It checks that path's own
// prerequisites (the handle-wide IPC / mapped capabilities agreed at setup) and
// submits the invocation; the like-named CPU schedule in
// reducescatter_*_schedule.hpp drives the path's DMA/RDMA and requests the CTA
// arithmetic phases. No entry calls a CPU collective or launches a kernel.

// SmallHost (2n*2g / 2n*4g, small T): CPU sums node partials. CPU: runSmallHostReduceScatter.
static __device__ __forceinline__ int liteReduceScatterSmallHostBlock(
    mscclppDeviceCollectiveHandle_t const& h, float const* src, float* dst,
    size_t bytes, LiteReduceScatterPlan const& plan) {
  return liteRsInvokePath(h, src, dst, bytes, plan);
}

// HostStaged (multi-node, no IPC): CPU host staging, sequential 256 KiB chunks. CPU: runNoCudaIpcHostReduceScatter (multi-node).
static __device__ __forceinline__ int liteReduceScatterHostStagedBlock(
    mscclppDeviceCollectiveHandle_t const& h, float const* src, float* dst,
    size_t bytes, LiteReduceScatterPlan const& plan) {
  return liteRsInvokePath(h, src, dst, bytes, plan);
}

// HierarchicalTwo (2n*2g, IPC): partner rows, two node partials, matching remote rank. CPU: runTwoNodeTwoGpuHierReduceScatter.
static __device__ __forceinline__ int liteReduceScatterHierarchicalTwoBlock(
    mscclppDeviceCollectiveHandle_t const& h, float const* src, float* dst,
    size_t bytes, LiteReduceScatterPlan const& plan) {
  if (!h.reduceScatterIpc) return mscclppDeviceCollectiveInvalidUsage;
  return liteRsInvokePath(h, src, dst, bytes, plan);
}

// HierarchicalFour (2n*4g, IPC): pair, cross-pair, matching remote partial. CPU: runChunk / runPipelinedChunks.
static __device__ __forceinline__ int liteReduceScatterHierarchicalFourBlock(
    mscclppDeviceCollectiveHandle_t const& h, float const* src, float* dst,
    size_t bytes, LiteReduceScatterPlan const& plan) {
  if (!h.reduceScatterIpc) return mscclppDeviceCollectiveInvalidUsage;
  return liteRsInvokePath(h, src, dst, bytes, plan);
}

