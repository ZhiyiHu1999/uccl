#pragma once

#include <stddef.h>
#include <stdint.h>

#if defined(__CUDACC__)
#define LITE_PLAN_HD __host__ __device__
#else
#define LITE_PLAN_HD
#endif

// Policy is snapshotted collectively at initialization. No getenv, allocation,
// CUDA runtime call, or protocol fallback is allowed inside a device call.
struct LiteAllGatherPolicy {
  size_t minBytes = 0;
  size_t kernelMaxBytes = 4096;
  size_t coopMaxBytes = 512 * 1024;
  size_t chunkBytes = 0;
  int hostEnabled = 0;
  int ipcEventSync = 1;
  int mapSlab = 1;
  int cooperative = 0;
};

enum class LiteAllGatherPath {
  Unsupported,
  Copy,
  IpcRing,
  HostMapped,
  HostCooperative,
  HostDma,
  OrderedSmall,
  SmallFallback,
  OneRankPipeline,
  SingleSlab,
  NumaSplit
};

struct LiteAllGatherPlan {
  LiteAllGatherPath path = LiteAllGatherPath::Unsupported;
  size_t chunkBytes = 0;
  bool stageWithSm = false;
  bool receiveWithSm = false;
  unsigned activeThreads = 0;  // zero means the entire caller CTA
};

LITE_PLAN_HD inline LiteAllGatherPlan litePlanAllGather(
    LiteAllGatherPolicy const& policy, int nranks, int ranksPerNode, bool ipc,
    bool mapped, size_t capacity, size_t bytes, uintptr_t alignment,
    bool graphCaptured = false, int nicGroups = 1) {
  LiteAllGatherPlan p;
  if (nranks < 1 || nranks > 8 || ranksPerNode < 1 || nranks % ranksPerNode ||
      bytes == 0 || bytes > capacity ||
      bytes > SIZE_MAX / static_cast<size_t>(nranks))
    return p;
  size_t total = bytes * static_cast<size_t>(nranks);
  if (nranks == 1) {
    p.path = LiteAllGatherPath::Copy;
    p.chunkBytes = bytes;
    return p;
  }
  if (graphCaptured) return p;
  if (nranks == ranksPerNode) {
    if (ipc) {
      if (!policy.hostEnabled && policy.ipcEventSync &&
          total >= 8 * 1024 * 1024) {
        p.path = LiteAllGatherPath::IpcRing;
        p.chunkBytes = bytes < 1024 * 1024 ? bytes : 1024 * 1024;
        p.stageWithSm = p.receiveWithSm = true;
      }
      return p;
    }
    if (!policy.hostEnabled || total < policy.minBytes ||
        total > (size_t{1} << 30))
      return p;
    size_t chunk = bytes <= 1024 * 1024        ? bytes
                   : bytes <= 32 * 1024 * 1024 ? 1024 * 1024
                                               : 4 * 1024 * 1024;
    if (policy.chunkBytes) {
      chunk = policy.chunkBytes < 256 * 1024 ? bytes
              : policy.chunkBytes < bytes    ? policy.chunkBytes
                                             : bytes;
    }
    p.chunkBytes = chunk;
    p.path = LiteAllGatherPath::HostDma;
    bool vector = !(alignment & 7) && !(bytes & 7) && chunk == bytes &&
                  mapped && policy.mapSlab;
    if (vector && total <= policy.kernelMaxBytes)
      p.path = LiteAllGatherPath::HostMapped;
    else if (vector && policy.cooperative && total <= policy.coopMaxBytes)
      p.path = LiteAllGatherPath::HostCooperative;
    p.stageWithSm = p.receiveWithSm = p.path != LiteAllGatherPath::HostDma;
    return p;
  }
  if (nranks != 2 * ranksPerNode || ipc) return p;
  size_t smallLimit = ranksPerNode == 1 ? 2 * 1024 * 1024 : 128 * 1024;
  if (total < smallLimit) {
    p.path = mapped ? LiteAllGatherPath::OrderedSmall
                    : LiteAllGatherPath::SmallFallback;
    p.chunkBytes = bytes;
    if (ranksPerNode == 1 && total < 64 * 1024 && mapped) {
      p.stageWithSm = p.receiveWithSm = true;
      p.activeThreads = total == 128                                 ? 1
                        : (total == 16 * 1024 || total == 32 * 1024) ? 256
                                                                     : 128;
    } else if (ranksPerNode == 2 && mapped) {
      p.stageWithSm = total <= 256 || (total >= 512 && total <= 4096);
      p.receiveWithSm = total <= 4096;
      if (total <= 256) p.activeThreads = 1;
    }
    return p;
  }
  if (ranksPerNode > 2 && nicGroups > 1) {
    p.path = LiteAllGatherPath::NumaSplit;
    p.chunkBytes = bytes < 16 * 1024 * 1024 ? bytes : 16 * 1024 * 1024;
    return p;
  }
  p.path = LiteAllGatherPath::SingleSlab;
  p.chunkBytes = ranksPerNode == 2 ? 512 * 1024 : 2 * 1024 * 1024;
  if (ranksPerNode == 1 && bytes >= 1024 * 1024 && bytes <= (size_t{1} << 30)) {
    p.path = LiteAllGatherPath::OneRankPipeline;
    p.chunkBytes = 512 * 1024;
  }
  return p;
}

#undef LITE_PLAN_HD
