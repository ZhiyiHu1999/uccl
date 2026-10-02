#pragma once
#include <stddef.h>
#include <stdint.h>

#if defined(__CUDACC__)
#define LITE_RS_HD __host__ __device__
#else
#define LITE_RS_HD
#endif

// Snapshotted and compared across ranks before allocating transport resources.
struct LiteReduceScatterPolicy {
  size_t chunkCapacity = 2 * 1024 * 1024;
  size_t layoutChunk = 0;
  size_t ringChunk = 16 * 1024 * 1024;
  size_t hostRingChunk = 16 * 1024 * 1024;
  size_t hostRingMin = 1024 * 1024;
  size_t hostSmall = 64 * 1024;
  size_t hostBulkChunk = 2 * 1024 * 1024;
  size_t hostChunk = 256 * 1024;
  size_t smallFull = 0;  // zero selects the topology default
  size_t twoRankSmall = 512 * 1024;
  size_t deviceFlagMax = 32 * 1024;
  // kLiteRsUnset selects the CPU reference default for the layout.
  size_t lead = ~size_t{0}, shortLead = ~size_t{0}, longLead = ~size_t{0};
  int noIpc = 0, p2pRing = 0, localRing = -1;
  int localParallel = 1, eagerPost = 0;
  int hostRead = 1, hostRing = 1, hierarchy = 1;
  int mappedTwoRank = 1, cpuFinal = 0, mappedSingle = 0;
  int mappedSend = -1, hostFinal = -1, splitFinal = -1;
  int directPartner = -1, partner2d = -1, asyncFinal = -1;
  int disableAvx512 = 0;
};

enum class LiteReduceScatterPath {
  Unsupported,
  Copy,
  Generic,
  LocalRows,
  P2pRing,
  IpcRing,
  TwoLocal,
  HostSmall,
  HostRing,
  HostRead,
  HostBulk,
  TwoRankSmall,
  TwoRankPipeline,
  SmallHost,
  HierarchicalTwo,
  HierarchicalFour,
  HostStaged
};
struct LiteReduceScatterPlan {
  LiteReduceScatterPath path = LiteReduceScatterPath::Unsupported;
  size_t chunkBytes = 0;
  unsigned slots = 1, lead = 0;
  bool deviceFlags = false, directPartner = false, partner2d = false;
  bool mappedSend = false, hostFinal = false, splitFinal = false;
  bool asyncFinal = false;
  // Mirrors CPU recordAsyncD2h: pipelined chunks, or the 2n*2g async single
  // chunk (B <= 512 KiB). Synchronous single chunks may use CPU_FINAL_ADD.
  bool recordAsync = false, cpuFinal = false, eagerPost = false;
};
LITE_RS_HD inline size_t liteRsLead(size_t configured, size_t fallback) {
  return configured == ~size_t{0} ? fallback : configured;
}
LITE_RS_HD inline size_t liteRsMin(size_t a, size_t b) { return a < b ? a : b; }
// Row capacity of the RS scratch / host-slab layout. Shared by the service
// (allocation) and the AllReduce planner so both derive the same bound.
LITE_RS_HD inline size_t liteRsChunkCapacity(LiteReduceScatterPolicy const& q,
                                            size_t capacity, int ranks) {
  size_t cap = capacity / static_cast<size_t>(ranks);
  size_t limit = q.chunkCapacity > q.hostBulkChunk ? q.chunkCapacity
                                                   : q.hostBulkChunk;
  if (cap > limit) cap = limit;
  if (cap < 4) cap = 4;
  return cap & ~size_t{3};
}
LITE_RS_HD inline LiteReduceScatterPlan litePlanReduceScatter(
    LiteReduceScatterPolicy const& q, int ranks, int local, size_t capacity,
    size_t bytes, bool floatSum, bool ipc, bool mapped) {
  LiteReduceScatterPlan p;
  if (ranks < 1 || ranks > 8 || local < 1 || ranks % local || !bytes ||
      bytes > capacity / static_cast<size_t>(ranks))
    return p;
  if (ranks == 1) {
    p.path = LiteReduceScatterPath::Copy;
    p.chunkBytes = bytes;
    return p;
  }
  if (ranks != local && ranks != 2 * local) return p;
  p.path = LiteReduceScatterPath::Generic;
  if (!floatSum || (local != 1 && local != 2 && local != 4)) return p;
  size_t total = bytes * ranks;
  ipc = ipc && !q.noIpc;
  p.chunkBytes = liteRsMin(bytes, q.chunkCapacity);
  if (ranks == local) {
    if (!ipc) {
      if (bytes <= q.hostSmall) {
        p.path = LiteReduceScatterPath::HostSmall;
      } else if (mapped && q.hostRing && bytes >= q.hostRingMin) {
        p.path = LiteReduceScatterPath::HostRing;
        p.chunkBytes = liteRsMin(bytes, q.hostRingChunk);
      } else {
        p.path = mapped && q.hostRead ? LiteReduceScatterPath::HostRead
                                      : LiteReduceScatterPath::HostBulk;
        p.chunkBytes = liteRsMin(bytes, q.hostBulkChunk);
      }
    } else if (local == 2) {
      p.path = LiteReduceScatterPath::TwoLocal;
    } else if (q.p2pRing || (bytes >= 256 * 1024 && bytes <= 2 * 1024 * 1024)) {
      p.path = LiteReduceScatterPath::P2pRing;
      p.chunkBytes = liteRsMin(bytes, q.ringChunk);
    } else if (bytes >= 1024 * 1024 &&
               (q.localRing == 1 ||
                (q.localRing < 0 && bytes >= 2 * 1024 * 1024))) {
      p.path = LiteReduceScatterPath::IpcRing;
      p.chunkBytes = liteRsMin(bytes, q.ringChunk);
    } else {
      p.path = LiteReduceScatterPath::LocalRows;
      p.deviceFlags = mapped && bytes <= q.deviceFlagMax;
    }
  } else {
    p.slots = local == 1 ? 5 : 4;
    bool mappedSendMode =
        q.mappedSend >= 0 ? q.mappedSend != 0 : bytes <= 1024 * 1024;
    size_t chunk = bytes <= 2 * 1024 * 1024 ? 512 * 1024 : 1024 * 1024;
    if (q.mappedSingle && mappedSendMode && q.hostFinal != 0 &&
        bytes <= 2 * 1024 * 1024)
      chunk = bytes;
    if ((local == 4 && bytes >= 2 * 1024 * 1024) ||
        (local == 2 && total >= 8 * 1024 * 1024))
      chunk = 1024 * 1024;
    if (q.layoutChunk >= sizeof(float)) chunk = q.layoutChunk;
    p.chunkBytes = liteRsMin(bytes, liteRsMin(q.chunkCapacity, chunk));
    size_t small =
        q.smallFull ? q.smallFull : (local == 2 ? 128 * 1024 : 512 * 1024);
    size_t chunks = p.chunkBytes ? (bytes - 1) / p.chunkBytes + 1 : 1;
    // configuredShort/Long/LocalLeadChunks: 1 / 3 / 3 by chunk count.
    size_t leadClass =
        chunks <= 2 ? q.shortLead : chunks >= 8 ? q.longLead : q.lead;
    size_t leadDefault = chunks <= 2 ? 1 : 3;
    if (local == 1) {
      p.path = bytes <= q.twoRankSmall && bytes <= q.chunkCapacity
                   ? LiteReduceScatterPath::TwoRankSmall
                   : LiteReduceScatterPath::TwoRankPipeline;
      if (p.path == LiteReduceScatterPath::TwoRankSmall) {
        // CPU runTwoRankSmallHostReduceScatter ignores CPU_FINAL_ADD.
        p.chunkBytes = bytes;
        p.mappedSend = p.hostFinal = mapped && q.mappedTwoRank;
      } else {
        // CPU runTwoRankPipelinedChunks is always D2H / H2D DMA + add.
        p.asyncFinal = chunks > 1;
        p.lead = static_cast<unsigned>(
            liteRsMin(liteRsLead(leadClass, leadDefault), p.slots - 1));
      }
    } else if (!q.noIpc && total < small &&
               (total - 1) / p.slots < q.chunkCapacity) {
      p.path = LiteReduceScatterPath::SmallHost;
    } else if (!ipc) {
      // runNoCudaIpcHostReduceScatter: strictly sequential chunks (lead 0).
      p.path = LiteReduceScatterPath::HostStaged;
      p.chunkBytes = liteRsMin(bytes, liteRsMin(q.hostChunk, q.chunkCapacity));
    } else if (local == 2 && !q.hierarchy) {
      p.path = LiteReduceScatterPath::Generic;
    } else {
      p.path = local == 2 ? LiteReduceScatterPath::HierarchicalTwo
                          : LiteReduceScatterPath::HierarchicalFour;
      bool pipelined = chunks > 1;
      p.recordAsync = pipelined || (local == 2 && bytes <= 512 * 1024);
      p.mappedSend = mapped && mappedSendMode;
      bool splitMode = q.splitFinal >= 0
                           ? q.splitFinal != 0
                           : (total >= 16 * 1024 * 1024 ||
                              (p.recordAsync && bytes <= 1024 * 1024));
      // 2n*4g: never for 8 ranks with B <= 512 KiB. 2n*2g: only with the
      // mapped send slab (remotePartialInSendSlab) on an async chunk.
      p.splitFinal = local == 2 ? (p.recordAsync && p.mappedSend && splitMode)
                                : (!(bytes <= 512 * 1024) && splitMode);
      // completeChunkRemote host-read final add.
      bool hostMode = q.hostFinal >= 0
                          ? q.hostFinal != 0
                          : ((!p.recordAsync && p.chunkBytes <= 512 * 1024) ||
                             p.splitFinal);
      p.hostFinal = mapped && hostMode && (!p.recordAsync || p.splitFinal);
      p.cpuFinal = q.cpuFinal && !p.recordAsync;
      // 2n*2g always pushes 2D-copied partner rows; 2n*4g follows total size.
      p.directPartner = local == 2 ||
                        (q.directPartner >= 0 ? q.directPartner != 0
                                              : total >= 1024 * 1024);
      p.partner2d = local == 2 || (q.partner2d >= 0 ? q.partner2d != 0
                                                    : p.directPartner);
      p.asyncFinal = q.asyncFinal >= 0 ? q.asyncFinal != 0
                                       : (pipelined ? local == 4
                                                    : bytes >= 32 * 1024 * 1024);
      p.eagerPost = pipelined &&
                    (q.eagerPost || (local == 2 && total >= 4 * 1024 * 1024) ||
                     (local == 4 && total >= 8 * 1024 * 1024));
      // 2n*2g follows the 1/3/3 lead classes; 2n*4g (runPipelinedChunks)
      // defaults to one unless the class variable is set.
      size_t lead = liteRsLead(leadClass, local == 4 ? 1 : leadDefault);
      p.lead = static_cast<unsigned>(liteRsMin(lead, p.slots - 1));
    }
  }
  p.chunkBytes -= p.chunkBytes % sizeof(float);
  if (!p.chunkBytes) p.path = LiteReduceScatterPath::Unsupported;
  return p;
}
#undef LITE_RS_HD

inline char const* liteReduceScatterPathName(LiteReduceScatterPath path) {
  switch (path) {
    case LiteReduceScatterPath::Unsupported:
      return "Unsupported";
    case LiteReduceScatterPath::Copy:
      return "Copy";
    case LiteReduceScatterPath::Generic:
      return "Generic";
    case LiteReduceScatterPath::LocalRows:
      return "LocalRows";
    case LiteReduceScatterPath::P2pRing:
      return "P2pRing";
    case LiteReduceScatterPath::IpcRing:
      return "IpcRing";
    case LiteReduceScatterPath::TwoLocal:
      return "TwoLocal";
    case LiteReduceScatterPath::HostSmall:
      return "HostSmall";
    case LiteReduceScatterPath::HostRing:
      return "HostRing";
    case LiteReduceScatterPath::HostRead:
      return "HostRead";
    case LiteReduceScatterPath::HostBulk:
      return "HostBulk";
    case LiteReduceScatterPath::TwoRankSmall:
      return "TwoRankSmall";
    case LiteReduceScatterPath::TwoRankPipeline:
      return "TwoRankPipeline";
    case LiteReduceScatterPath::SmallHost:
      return "SmallHost";
    case LiteReduceScatterPath::HierarchicalTwo:
      return "HierarchicalTwo";
    case LiteReduceScatterPath::HierarchicalFour:
      return "HierarchicalFour";
    case LiteReduceScatterPath::HostStaged:
      return "HostStaged";
  }
  return "Unknown";
}
