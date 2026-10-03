#pragma once
#include <stddef.h>
#include <stdint.h>
#include "reducescatter_plan.hpp"

#if defined(__CUDACC__)
#define LITE_AR_HD __host__ __device__
#else
#define LITE_AR_HD
#endif

// Snapshotted and compared across ranks before allocating transport resources.
// Sizes and the 2n*1g ring switch follow the CPU reference (native_collectives.cu).
struct LiteAllReducePolicy {
  // MSCCLPP_NCCL_2RANK_RING_CHUNK_BYTES: bytes of one ring part.
  size_t ringChunk = 4 * 1024 * 1024;
  // MSCCLPP_NCCL_AR_RS_AG_MIN_BYTES: smallest B that composes RS+AG. The host
  // selector has no such crossover; zero means "whenever eligible" until
  // GPU-driven measurements establish one.
  size_t rsAgMin = 0;
  // MSCCLPP_NCCL_2RANK_RING_ALLREDUCE: force the ring below its 64 MiB default.
  int ringForce = 0;
};

enum class LiteAllReducePath {
  Unsupported,
  Copy,
  Generic,
  ReduceScatterAllGather,
  SmallMapped,
  SmallTwoLeader,
  TwoRankRing
};

struct LiteAllReducePlan {
  LiteAllReducePath path = LiteAllReducePath::Unsupported;
  // TwoRankRing: bytes of one ring part (stage buffer), a multiple of 4 bytes.
  size_t chunkBytes = 0;
};

// Thresholds of runSendRecvAllReduce / runSmallMappedAllReduce2Node /
// runSmallTwoLeaderAllReduce2Node / runTwoRankRingSimpleAllReduce2Node.
constexpr size_t kLiteArSmallMappedMaxBytes = 64 * 1024;
constexpr size_t kLiteArTwoLeaderMaxBytes = 128 * 1024;
constexpr size_t kLiteArRingAutoMinBytes = 64 * 1024 * 1024;

// B is the complete tensor bytes per rank (input == output size). RS+AG splits
// it into R equal shards of S = B / R bytes: ReduceScatter sees B_RS = S,
// T_RS = B and AllGather B_AG = S, T_AG = B.
LITE_AR_HD inline LiteAllReducePlan litePlanAllReduce(
    LiteAllReducePolicy const& q, LiteReduceScatterPolicy const& rq, int ranks,
    int local, size_t capacity, size_t bytes, bool floatSum, bool ipc,
    bool mapped) {
  LiteAllReducePlan p;
  if (ranks < 1 || local < 1 || ranks % local || !bytes) return p;
  if (ranks == 1) {
    p.path = LiteAllReducePath::Copy;
    return p;
  }
  // Every other case has the arithmetic-template full-tensor staging path.
  p.path = LiteAllReducePath::Generic;
  if (!floatSum || ranks > 8 || bytes % sizeof(float) || bytes > capacity)
    return p;
  if (ranks != local && ranks != 2 * local) return p;
  if (local != 1 && local != 2 && local != 4) return p;
  size_t cap = liteRsChunkCapacity(rq, capacity, ranks);
  if (ranks == 2 * local) {
    if (local == 1) {
      // 2n*1g: two-channel ring for B >= 64 MiB (or forced).
      size_t part = q.ringChunk < 2 * cap ? q.ringChunk : 2 * cap;
      part -= part % sizeof(float);
      if (part && (q.ringForce || bytes >= kLiteArRingAutoMinBytes)) {
        p.path = LiteAllReducePath::TwoRankRing;
        p.chunkBytes = part;
        return p;
      }
    } else if (bytes <= kLiteArSmallMappedMaxBytes && mapped && bytes <= cap) {
      p.path = LiteAllReducePath::SmallMapped;
      return p;
    } else if (local == 4 && bytes > kLiteArSmallMappedMaxBytes &&
               bytes <= kLiteArTwoLeaderMaxBytes && bytes % 8 == 0 &&
               bytes <= cap) {
      p.path = LiteAllReducePath::SmallTwoLeader;
      return p;
    }
  }
  // Equal-shard ReduceScatter + AllGather. Needs C % R == 0 and an eligible RS
  // path for the shard; never truncates the tensor.
  size_t r = static_cast<size_t>(ranks);
  if (bytes % (r * sizeof(float)) == 0 && bytes >= q.rsAgMin) {
    auto rs = litePlanReduceScatter(rq, ranks, local, capacity, bytes / r, true,
                                    ipc, mapped);
    if (rs.path != LiteReduceScatterPath::Unsupported)
      p.path = LiteAllReducePath::ReduceScatterAllGather;
  }
  return p;
}
#undef LITE_AR_HD

inline char const* liteAllReducePathName(LiteAllReducePath path) {
  switch (path) {
    case LiteAllReducePath::Unsupported:
      return "Unsupported";
    case LiteAllReducePath::Copy:
      return "Copy";
    case LiteAllReducePath::Generic:
      return "Generic";
    case LiteAllReducePath::ReduceScatterAllGather:
      return "ReduceScatterAllGather";
    case LiteAllReducePath::SmallMapped:
      return "SmallMapped";
    case LiteAllReducePath::SmallTwoLeader:
      return "SmallTwoLeader";
    case LiteAllReducePath::TwoRankRing:
      return "TwoRankRing";
  }
  return "Unknown";
}
