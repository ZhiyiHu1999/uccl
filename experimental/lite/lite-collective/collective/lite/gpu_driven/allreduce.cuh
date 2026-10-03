#pragma once
#include <type_traits>

// Device-callable AllReduce. Like AllGather and ReduceScatter, the collective
// layer selects a path from a pure plan (allreduce_plan.hpp) and dispatches to
// one independent __device__ entry per path:
//   allreduce_rs_ag.cuh          ReduceScatterAllGather (composes the RS and AG entries)
//   allreduce_small.cuh          SmallMapped, SmallTwoLeader (two nodes, small B)
//   allreduce_two_rank_ring.cuh  TwoRankRing (2n*1g, large B)
// Copy and Generic are handled here. Every path runs in the single participating
// CTA; the network paths submit one whole-invocation `AllReduce` FIFO task whose
// CPU schedule requests CTA phases through liteRsInvokeTask.

// Whole-invocation AllReduce task: the CPU schedule of `path`
// (allreduce_schedule.hpp) drives DMA/RDMA and requests CTA copy/sum phases.
static __device__ __forceinline__ int liteArInvokePath(
    mscclppDeviceCollectiveHandle_t const& h, float const* src, float* dst,
    size_t bytes, LiteAllReducePlan const& plan) {
  LiteTask task{};
  task.kind = LiteTaskKind::AllReduce;
  task.source = reinterpret_cast<uint64_t>(src);
  task.destination = reinterpret_cast<uint64_t>(dst);
  task.bytes = bytes;
  task.allReducePath = plan.path;
  return liteRsInvokeTask(h, task);
}

#include "allreduce_rs_ag.cuh"
#include "allreduce_small.cuh"
#include "allreduce_two_rank_ring.cuh"

// Block-scoped AllReduce. Each rank contributes `count` elements and receives the
// element-wise reduction of all ranks (in place when src == dst). Float/sum uses
// the optimized paths; every other type/operation, irregular counts and
// ineligible configurations use the arithmetic-template full-tensor path.
// All ranks must pass identical counts and operations.
template <typename T>
static __device__ __forceinline__ int liteAllReduceBlock(
    mscclppDeviceCollectiveHandle_t const& h, T const* src, T* dst,
    size_t count, liteReduceOp op = liteReduceSum) {
  if constexpr (std::is_same<T, float>::value) {
    if (op == liteReduceSum && mscclppDeviceCollectiveHandleValid(h) && src &&
        dst && count && h.nranks > 1 &&
        h.nranks <= MSCCLPP_DEVICE_COLLECTIVE_MAX_RANKS &&
        h.reduceScatterPrepared && h.tasks &&
        count <= SIZE_MAX / sizeof(float) &&
        !((reinterpret_cast<uintptr_t>(src) | reinterpret_cast<uintptr_t>(dst)) &
          (alignof(float) - 1))) {
      size_t bytes = count * sizeof(float);
      auto plan = litePlanAllReduce(h.allReducePolicy, h.reduceScatterPolicy,
                                    h.nranks, h.ranksPerNode,
                                    h.maxBytesPerRank, bytes, true,
                                    h.reduceScatterIpc, h.reduceScatterMapped);
      switch (plan.path) {
        case LiteAllReducePath::ReduceScatterAllGather:
          return liteAllReduceRsAgBlock(h, src, dst, count);
        case LiteAllReducePath::SmallMapped:
          return liteAllReduceSmallMappedBlock(h, src, dst, bytes, plan);
        case LiteAllReducePath::SmallTwoLeader:
          return liteAllReduceSmallTwoLeaderBlock(h, src, dst, bytes, plan);
        case LiteAllReducePath::TwoRankRing:
          return liteAllReduceTwoRankRingBlock(h, src, dst, bytes, plan);
        default:
          break;  // Generic
      }
    }
  }
  return liteAllReduceGenericBlock(h, src, dst, count, op);
}
