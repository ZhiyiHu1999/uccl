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

enum class LiteDeviceAllGatherPath {
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
  LiteDeviceAllGatherPath path = LiteDeviceAllGatherPath::Unsupported;
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
    p.path = LiteDeviceAllGatherPath::Copy;
    p.chunkBytes = bytes;
    return p;
  }
  if (graphCaptured) return p;
  if (nranks == ranksPerNode) {
    if (ipc) {
      if (!policy.hostEnabled && policy.ipcEventSync &&
          total >= 8 * 1024 * 1024) {
        p.path = LiteDeviceAllGatherPath::IpcRing;
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
    p.path = LiteDeviceAllGatherPath::HostDma;
    bool vector = !(alignment & 7) && !(bytes & 7) && chunk == bytes &&
                  mapped && policy.mapSlab;
    if (vector && total <= policy.kernelMaxBytes)
      p.path = LiteDeviceAllGatherPath::HostMapped;
    else if (vector && policy.cooperative && total <= policy.coopMaxBytes)
      p.path = LiteDeviceAllGatherPath::HostCooperative;
    p.stageWithSm = p.receiveWithSm = p.path != LiteDeviceAllGatherPath::HostDma;
    return p;
  }
  if (nranks != 2 * ranksPerNode || ipc) return p;
  size_t smallLimit = ranksPerNode == 1 ? 2 * 1024 * 1024 : 128 * 1024;
  if (total < smallLimit) {
    p.path = mapped ? LiteDeviceAllGatherPath::OrderedSmall
                    : LiteDeviceAllGatherPath::SmallFallback;
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
    p.path = LiteDeviceAllGatherPath::NumaSplit;
    p.chunkBytes = bytes < 16 * 1024 * 1024 ? bytes : 16 * 1024 * 1024;
    return p;
  }
  p.path = LiteDeviceAllGatherPath::SingleSlab;
  p.chunkBytes = ranksPerNode == 2 ? 512 * 1024 : 2 * 1024 * 1024;
  if (ranksPerNode == 1 && bytes >= 1024 * 1024 && bytes <= (size_t{1} << 30)) {
    p.path = LiteDeviceAllGatherPath::OneRankPipeline;
    p.chunkBytes = 512 * 1024;
  }
  return p;
}

#undef LITE_PLAN_HD
