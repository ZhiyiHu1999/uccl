#pragma once
#include <type_traits>

// Shared primitive of every optimization path: one whole invocation occupies one
// FIFO descriptor tagged with plan.path. While the path's CPU schedule (the
// like-named function in reducescatter_*_schedule.hpp) progresses DMA/RDMA, each
// arithmetic phase it requests executes in this caller's CTA.
static __device__ __forceinline__ int liteRsInvokePath(
    mscclppDeviceCollectiveHandle_t const& h, float const* src, float* dst,
    size_t bytes, LiteReduceScatterPlan const& plan) {
  __shared__ unsigned long long ticket, phase;
  __shared__ int status, sources;
  __shared__ size_t count;
  __shared__ float const* inputs[4];
  __shared__ float* output;
  unsigned tid = mscclppDeviceCollectiveThreadId();
  unsigned threads = mscclppDeviceCollectiveThreadCount();
  auto* control = &h.tasks->reduceScatter;
  // Input may have been produced by this CTA inside the same user kernel.
  // Publish every lane's writes before the service can start D2H/IPC DMA.
  __threadfence_system();
  __syncthreads();
  if (!tid) {
    status = mscclppDeviceCollectiveSuccess;
    phase = liteLoadAcquire(
        reinterpret_cast<unsigned long long*>(&control->completed));
    LiteTask task{};
    task.kind = LiteTaskKind::ReduceScatter;
    task.source = reinterpret_cast<uint64_t>(src);
    task.destination = reinterpret_cast<uint64_t>(dst);
    task.bytes = bytes;
    task.reduceScatterPath = plan.path;
    ticket = litePostTask(h, task);
    if (!ticket) status = mscclppDeviceCollectiveTransportError;
  }
  __syncthreads();
  if (status) return status;
  for (;;) {
    if (!tid) {
      unsigned long long start = clock64();
      for (;;) {
        if (liteLoadAcquire(
                reinterpret_cast<unsigned long long*>(&h.tasks->error))) {
          status = mscclppDeviceCollectiveTransportError;
          break;
        }
        auto completed = liteLoadAcquire(reinterpret_cast<unsigned long long*>(
            &h.tasks->slots[(ticket - 1) % kLiteTaskSlots].completed));
        if (completed >= ticket) {
          sources = 0;
          break;
        }
        auto requested = liteLoadAcquire(
            reinterpret_cast<unsigned long long*>(&control->requested));
        if (requested > phase) {
          phase = requested;
          count = control->count;
          sources = control->sources;
          output = reinterpret_cast<float*>(control->destination);
          for (int r = 0; r < sources; ++r)
            inputs[r] = reinterpret_cast<float const*>(control->source[r]);
          break;
        }
        if (h.timeoutCycles && clock64() - start > h.timeoutCycles) {
          liteCollectivePoison(h);
          status = mscclppDeviceCollectiveTransportError;
          break;
        }
      }
    }
    __syncthreads();
    if (status) return status;
    if (!sources) return mscclppDeviceCollectiveSuccess;
    for (size_t i = tid; i < count; i += threads) {
      // Payload may have been updated by DMA/another GPU while this kernel
      // remains resident. Volatile loads prevent reusing a prior phase's value.
      float value = reinterpret_cast<float const volatile*>(inputs[0])[i];
      for (int r = 1; r < sources; ++r)
        value += reinterpret_cast<float const volatile*>(inputs[r])[i];
      output[i] = value;
    }
    __threadfence_system();
    __syncthreads();
    if (!tid)
      liteStoreRelease(
          reinterpret_cast<unsigned long long*>(&control->completed), phase);
    __syncthreads();
  }
}

// Per-path device entries live in their own files, one function per plan path.
#include "reducescatter_primitives.cuh"
#include "reducescatter_ipc.cuh"
#include "reducescatter_host.cuh"
#include "reducescatter_two_rank.cuh"
#include "reducescatter_hierarchical.cuh"

// Select a topology/size plan and dispatch to its independent executor.
// Non-float, non-sum operations and the Generic path use the arithmetic-template
// staging implementation.
static __device__ __forceinline__ int liteReduceScatterFloatSumBlock(
    mscclppDeviceCollectiveHandle_t const& h, float const* src, float* dst,
    size_t recvCount) {
  size_t bytes = recvCount * sizeof(float);
  auto plan = litePlanReduceScatter(h.reduceScatterPolicy, h.nranks,
                                    h.ranksPerNode, h.maxBytesPerRank, bytes,
                                    true, h.reduceScatterIpc,
                                    h.reduceScatterMapped);
  switch (plan.path) {
    case LiteReduceScatterPath::LocalRows:
      return liteReduceScatterLocalRowsBlock(h, src, dst, bytes, plan);
    case LiteReduceScatterPath::TwoLocal:
      return liteReduceScatterTwoLocalBlock(h, src, dst, bytes, plan);
    case LiteReduceScatterPath::P2pRing:
      return liteReduceScatterP2pRingBlock(h, src, dst, bytes, plan);
    case LiteReduceScatterPath::IpcRing:
      return liteReduceScatterIpcRingBlock(h, src, dst, bytes, plan);
    case LiteReduceScatterPath::HostSmall:
      return liteReduceScatterHostSmallBlock(h, src, dst, bytes, plan);
    case LiteReduceScatterPath::HostRing:
      return liteReduceScatterHostRingBlock(h, src, dst, bytes, plan);
    case LiteReduceScatterPath::HostRead:
      return liteReduceScatterHostReadBlock(h, src, dst, bytes, plan);
    case LiteReduceScatterPath::HostBulk:
      return liteReduceScatterHostBulkBlock(h, src, dst, bytes, plan);
    case LiteReduceScatterPath::TwoRankSmall:
      return liteReduceScatterTwoRankSmallBlock(h, src, dst, bytes, plan);
    case LiteReduceScatterPath::TwoRankPipeline:
      return liteReduceScatterTwoRankPipelineBlock(h, src, dst, bytes, plan);
    case LiteReduceScatterPath::SmallHost:
      return liteReduceScatterSmallHostBlock(h, src, dst, bytes, plan);
    case LiteReduceScatterPath::HostStaged:
      return liteReduceScatterHostStagedBlock(h, src, dst, bytes, plan);
    case LiteReduceScatterPath::HierarchicalTwo:
      return liteReduceScatterHierarchicalTwoBlock(h, src, dst, bytes, plan);
    case LiteReduceScatterPath::HierarchicalFour:
      return liteReduceScatterHierarchicalFourBlock(h, src, dst, bytes, plan);
    case LiteReduceScatterPath::Generic:
      return liteReduceScatterGenericBlock(h, src, dst, recvCount,
                                           liteReduceSum);
    default:
      return mscclppDeviceCollectiveInvalidUsage;
  }
}

template <typename T>
static __device__ __forceinline__ int liteReduceScatterBlock(
    mscclppDeviceCollectiveHandle_t const& h, T const* src, T* dst,
    size_t recvCount, liteReduceOp op = liteReduceSum) {
  if (!mscclppDeviceCollectiveHandleValid(h) || !src || !dst || !recvCount ||
      h.nranks < 1 || h.nranks > MSCCLPP_DEVICE_COLLECTIVE_MAX_RANKS ||
      h.rank < 0 || h.rank >= h.nranks || recvCount > SIZE_MAX / sizeof(T) ||
      recvCount * sizeof(T) >
          h.maxBytesPerRank / static_cast<size_t>(h.nranks) ||
      ((reinterpret_cast<uintptr_t>(src) | reinterpret_cast<uintptr_t>(dst)) &
       (alignof(T) - 1)) ||
      h.maxBytesPerRank % alignof(T) || op < liteReduceSum ||
      op > liteReduceMax)
    return mscclppDeviceCollectiveInvalidArgument;
  if (h.tasks &&
      liteLoadAcquire(reinterpret_cast<unsigned long long*>(&h.tasks->error)))
    return mscclppDeviceCollectiveTransportError;
  if constexpr (std::is_same<T, float>::value) {
    if (op == liteReduceSum && h.nranks > 1 && h.reduceScatterPrepared &&
        h.tasks)
      return liteReduceScatterFloatSumBlock(h, src, dst, recvCount);
  }
  return liteReduceScatterGenericBlock(h, src, dst, recvCount, op);
}
