/*
Device-driven collectives for UCCL-lite.

A handle is initialized collectively on the host once, then copied by value into
a user kernel. The block routine is called by every thread in exactly one block
per rank, in the same order on every rank.
*/

#pragma once

#include "allgather_plan.hpp"
#include "nccl.h"
#include "task_fifo.hpp"
#include <stddef.h>
#include <stdint.h>

static constexpr int MSCCLPP_DEVICE_COLLECTIVE_MAX_RANKS = 8;
static constexpr int MSCCLPP_DEVICE_COLLECTIVE_SLOTS = 2;
static constexpr int MSCCLPP_DEVICE_COLLECTIVE_MAX_CHUNKS = 1024;
static constexpr size_t MSCCLPP_DEVICE_COLLECTIVE_SMALL_BYTES = 4 * 1024;
static constexpr size_t MSCCLPP_DEVICE_COLLECTIVE_CHUNK_BYTES = 256 * 1024;

typedef enum mscclppDeviceCollectiveBackend {
  mscclppDeviceCollectiveHostMemory = 0,
  mscclppDeviceCollectiveCudaIpc = 1,
  mscclppDeviceCollectiveHostRdma = 2,
} mscclppDeviceCollectiveBackend_t;

typedef struct mscclppDeviceCollectiveHandle {
  char* slab;
  char* control;
  char* peerSlabs[MSCCLPP_DEVICE_COLLECTIVE_MAX_RANKS];
  unsigned long long* peerReady[MSCCLPP_DEVICE_COLLECTIVE_MAX_RANKS];
  unsigned long long* peerDone[MSCCLPP_DEVICE_COLLECTIVE_MAX_RANKS];
  size_t peerReadySlotStride[MSCCLPP_DEVICE_COLLECTIVE_MAX_RANKS];
  size_t peerDoneSlotStride[MSCCLPP_DEVICE_COLLECTIVE_MAX_RANKS];
  // Network buffers contain node/group blocks, packed at the current chunk
  // stride. Reductions retain capacity-strided rows in the same allocation.
  size_t peerSlabSlotStride[MSCCLPP_DEVICE_COLLECTIVE_MAX_RANKS];
  int peerSlabIndex[MSCCLPP_DEVICE_COLLECTIVE_MAX_RANKS];
  unsigned long long* groupReusable[MSCCLPP_DEVICE_COLLECTIVE_MAX_RANKS];
  unsigned long long* groupDone[MSCCLPP_DEVICE_COLLECTIVE_MAX_RANKS];
  int groupCount;
  mscclppDeviceCollectiveHandle const* numaHandle;
  unsigned long long* slotReusable;
  unsigned long long* publishedBytes;
  size_t publishedBytesSlotStride;
  unsigned long long* localEpoch;
  size_t maxBytesPerRank;
  size_t slotStride;
  size_t counterStride;
  int rank;
  int nranks;
  int backend;
  int ranksPerNode;
  int reductionsMapped;
  LiteAllGatherPolicy allGatherPolicy;
  LiteTaskFifo* tasks;
  unsigned long long timeoutCycles;
} mscclppDeviceCollectiveHandle_t;

#ifdef __cplusplus
extern "C" {
#endif

/*
Collective host-side initialization.
Every rank in comm must call with the same backend and maxBytesPerRank.
CUDA IPC requires peer access between every participating GPU pair.
Balanced two-node communicators automatically use host-staged RDMA.
*/
ncclResult_t mscclppGetDeviceCollectiveHandle(
    ncclComm_t comm, size_t maxBytesPerRank,
    mscclppDeviceCollectiveBackend_t backend,
    mscclppDeviceCollectiveHandle_t* handle);

#ifdef __cplusplus
}  // extern "C"
#endif

#if defined(__CUDACC__)

// Return values for device-side argument errors.  A zero return means success.
enum mscclppDeviceCollectiveResult {
  mscclppDeviceCollectiveSuccess = 0,
  mscclppDeviceCollectiveInvalidArgument = 1,
  mscclppDeviceCollectiveInvalidUsage = 2,
  mscclppDeviceCollectiveTransportError = 3,
};

enum liteReduceOp {
  liteReduceSum = 0,
  liteReduceMin = 1,
  liteReduceMax = 2,
};

static __device__ __forceinline__ volatile unsigned long long*
mscclppDeviceCollectiveReady(mscclppDeviceCollectiveHandle_t const& h, int slot,
                             int chunk, int rank) {
  if (h.backend == mscclppDeviceCollectiveHostRdma) {
    return h.peerReady[rank] +
           static_cast<size_t>(slot) * h.peerReadySlotStride[rank];
  }
  if (h.backend == mscclppDeviceCollectiveCudaIpc) {
    return h.peerReady[rank] +
           static_cast<size_t>(slot) * h.peerReadySlotStride[rank] + chunk;
  }
  size_t index =
      static_cast<size_t>(slot) * (MSCCLPP_DEVICE_COLLECTIVE_MAX_CHUNKS *
                                   MSCCLPP_DEVICE_COLLECTIVE_MAX_RANKS) +
      static_cast<size_t>(chunk) * MSCCLPP_DEVICE_COLLECTIVE_MAX_RANKS +
      static_cast<size_t>(rank);
  return reinterpret_cast<unsigned long long volatile*>(
      h.control + index * h.counterStride);
}

static __device__ __forceinline__ volatile unsigned long long*
mscclppDeviceCollectiveDone(mscclppDeviceCollectiveHandle_t const& h, int slot,
                            int rank) {
  if (h.backend == mscclppDeviceCollectiveCudaIpc ||
      h.backend == mscclppDeviceCollectiveHostRdma) {
    return h.peerDone[rank] +
           static_cast<size_t>(slot) * h.peerDoneSlotStride[rank];
  }
  size_t index =
      MSCCLPP_DEVICE_COLLECTIVE_SLOTS * MSCCLPP_DEVICE_COLLECTIVE_MAX_CHUNKS *
          MSCCLPP_DEVICE_COLLECTIVE_MAX_RANKS +
      static_cast<size_t>(slot) * MSCCLPP_DEVICE_COLLECTIVE_MAX_RANKS +
      static_cast<size_t>(rank);
  return reinterpret_cast<unsigned long long volatile*>(
      h.control + index * h.counterStride);
}

static __device__ __forceinline__ char* mscclppDeviceCollectiveSlab(
    mscclppDeviceCollectiveHandle_t const& h, int slot, int rank,
    size_t networkRowBytes = 0) {
  if (h.backend == mscclppDeviceCollectiveHostRdma) {
    size_t row = networkRowBytes ? networkRowBytes : h.maxBytesPerRank;
    return h.peerSlabs[rank] + slot * h.peerSlabSlotStride[rank] +
           h.peerSlabIndex[rank] * row;
  }
  if (h.backend == mscclppDeviceCollectiveCudaIpc)
    return h.peerSlabs[rank] + static_cast<size_t>(slot) * h.maxBytesPerRank;
  return h.slab + static_cast<size_t>(slot) * h.slotStride +
         static_cast<size_t>(rank) * h.maxBytesPerRank;
}

static __device__ __forceinline__ bool mscclppDeviceCollectiveHandleValid(
    mscclppDeviceCollectiveHandle_t const& h) {
  if (h.localEpoch == nullptr || h.nranks < 1 ||
      h.nranks > MSCCLPP_DEVICE_COLLECTIVE_MAX_RANKS || h.rank < 0 ||
      h.rank >= h.nranks) {
    return false;
  }
  if (h.nranks == 1) return true;
  if (h.backend == mscclppDeviceCollectiveHostMemory) {
    return (h.slab != nullptr || h.tasks != nullptr) && h.control != nullptr;
  }
  if (h.backend != mscclppDeviceCollectiveCudaIpc &&
      h.backend != mscclppDeviceCollectiveHostRdma) {
    return false;
  }
  for (int r = 0; r < h.nranks; ++r) {
    if ((h.peerSlabs[r] == nullptr && h.tasks == nullptr) ||
        h.peerReady[r] == nullptr || h.peerDone[r] == nullptr) {
      return false;
    }
  }
  if (h.backend == mscclppDeviceCollectiveHostRdma &&
      (h.slotReusable == nullptr || h.publishedBytes == nullptr ||
       h.publishedBytesSlotStride == 0)) {
    return false;
  }
  return true;
}

static __device__ __forceinline__ void liteCollectivePoison(
    mscclppDeviceCollectiveHandle_t const& h) {
  if (h.tasks)
    liteStoreRelease(reinterpret_cast<unsigned long long*>(&h.tasks->error), 1);
  for (int slot = 0; slot < MSCCLPP_DEVICE_COLLECTIVE_SLOTS; ++slot) {
    int chunks = h.backend == mscclppDeviceCollectiveHostRdma
                     ? 1
                     : MSCCLPP_DEVICE_COLLECTIVE_MAX_CHUNKS;
    for (int chunk = 0; chunk < chunks; ++chunk)
      liteStoreRelease(mscclppDeviceCollectiveReady(h, slot, chunk, h.rank),
                       ~0ULL);
    liteStoreRelease(mscclppDeviceCollectiveDone(h, slot, h.rank), ~0ULL);
    for (int g = 0; g < h.groupCount; ++g)
      liteStoreRelease(
          h.groupDone[g] + slot * MSCCLPP_DEVICE_COLLECTIVE_MAX_RANKS, ~0ULL);
  }
}

// Fatal service errors and stalled peers terminate all device waits. A failed
// handle is poisoned and must be destroyed after its calling stream completes.
static __device__ __forceinline__ bool liteCollectiveWait(
    mscclppDeviceCollectiveHandle_t const& h,
    unsigned long long const volatile* value, unsigned long long target) {
  unsigned long long start = clock64();
  for (;;) {
    if (h.tasks && liteLoadAcquire(reinterpret_cast<unsigned long long*>(
                       &h.tasks->error))) {
      liteCollectivePoison(h);
      return false;
    }
    unsigned long long observed = liteLoadAcquire(value);
    if (observed == ~0ULL) {
      liteCollectivePoison(h);
      return false;
    }
    if (observed >= target) return true;
    if (h.timeoutCycles && clock64() - start > h.timeoutCycles) {
      liteCollectivePoison(h);
      return false;
    }
  }
}

static __device__ __forceinline__ bool liteCollectiveWaitReusable(
    mscclppDeviceCollectiveHandle_t const& h, int slot,
    unsigned long long previous) {
  if (h.backend == mscclppDeviceCollectiveHostRdma &&
      !liteCollectiveWait(h, h.slotReusable + slot, previous))
    return false;
  for (int g = 0; g < h.groupCount; ++g)
    if (!liteCollectiveWait(h, h.groupReusable[g] + slot, previous))
      return false;
  // NIC completion does not retire local GPU/DMA readers.
  for (int r = 0; r < h.nranks; ++r) {
    if (!liteCollectiveWait(h, mscclppDeviceCollectiveDone(h, slot, r),
                            previous))
      return false;
  }
  return true;
}

static __device__ __forceinline__ unsigned int
mscclppDeviceCollectiveThreadId() {
  return threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z);
}

static __device__ __forceinline__ unsigned int
mscclppDeviceCollectiveThreadCount() {
  return blockDim.x * blockDim.y * blockDim.z;
}

static __device__ __forceinline__ void liteCollectiveCopyBlock(
    char* dst, char const* src, size_t bytes, unsigned activeThreads = 0) {
  unsigned int tid = mscclppDeviceCollectiveThreadId();
  unsigned int threadCount = mscclppDeviceCollectiveThreadCount();
  if (activeThreads && activeThreads < threadCount) threadCount = activeThreads;
  if (tid >= threadCount) return;
  uintptr_t alignment =
      reinterpret_cast<uintptr_t>(dst) | reinterpret_cast<uintptr_t>(src);
  if ((alignment & (alignof(unsigned long long) - 1)) == 0) {
    size_t words = bytes / sizeof(unsigned long long);
    auto* wordDst = reinterpret_cast<unsigned long long*>(dst);
    auto const* wordSrc = reinterpret_cast<unsigned long long const*>(src);
    for (size_t i = tid; i < words; i += threadCount) {
      wordDst[i] = wordSrc[i];
    }
    size_t wordBytes = words * sizeof(unsigned long long);
    for (size_t i = wordBytes + tid; i < bytes; i += threadCount) {
      dst[i] = src[i];
    }
    return;
  }
  for (size_t i = tid; i < bytes; i += threadCount) dst[i] = src[i];
}

template <typename T>
static __device__ __forceinline__ T liteApplyReduction(T lhs, T rhs,
                                                       liteReduceOp op) {
  if (op == liteReduceMin) return rhs < lhs ? rhs : lhs;
  if (op == liteReduceMax) return rhs > lhs ? rhs : lhs;
  return lhs + rhs;
}

static __device__ __forceinline__ size_t liteCollectiveChunkBytes(
    mscclppDeviceCollectiveHandle_t const& h, size_t sourceBytes) {
  // Reduction staging uses SM copies with per-chunk readiness on mapped
  // host/IPC memory. Its two-node proxy exchanges the full input per epoch;
  // AllGather independently selects chunk sizes and posts DMA FIFO tasks.
  if (h.backend == mscclppDeviceCollectiveHostRdma ||
      sourceBytes <= MSCCLPP_DEVICE_COLLECTIVE_SMALL_BYTES) {
    return sourceBytes;
  }
  size_t required = sourceBytes / MSCCLPP_DEVICE_COLLECTIVE_MAX_CHUNKS +
                    (sourceBytes % MSCCLPP_DEVICE_COLLECTIVE_MAX_CHUNKS != 0);
  size_t chunkBytes = MSCCLPP_DEVICE_COLLECTIVE_CHUNK_BYTES;
  if (required > chunkBytes) {
    chunkBytes = ((required + MSCCLPP_DEVICE_COLLECTIVE_CHUNK_BYTES - 1) /
                  MSCCLPP_DEVICE_COLLECTIVE_CHUNK_BYTES) *
                 MSCCLPP_DEVICE_COLLECTIVE_CHUNK_BYTES;
  }
  return sourceBytes < chunkBytes ? sourceBytes : chunkBytes;
}

// Allocate a collective epoch and reserve one of the two whole-message slots.
// Chunk readiness is handled separately so a large invocation can progress
// through the slot without waiting for the entire input to be published.
static __device__ __forceinline__ int liteCollectiveBeginBlock(
    mscclppDeviceCollectiveHandle_t const& h, void const* srcVoid,
    size_t sourceBytes, unsigned long long* epochOut, int* slotOut,
    size_t* chunkBytesOut, int* chunkCountOut) {
  __shared__ unsigned long long epoch;
  __shared__ int status;
  __shared__ size_t chunkBytes;
  __shared__ int chunkCount;
  unsigned int tid = mscclppDeviceCollectiveThreadId();

  if (tid == 0) {
    status = mscclppDeviceCollectiveSuccess;
    if (h.tasks &&
        liteLoadAcquire(reinterpret_cast<unsigned long long*>(&h.tasks->error)))
      status = mscclppDeviceCollectiveTransportError;
    if (!mscclppDeviceCollectiveHandleValid(h) || srcVoid == nullptr ||
        sourceBytes == 0 || sourceBytes > h.maxBytesPerRank || h.nranks < 1 ||
        h.nranks > MSCCLPP_DEVICE_COLLECTIVE_MAX_RANKS || h.rank < 0 ||
        h.rank >= h.nranks) {
      status = mscclppDeviceCollectiveInvalidArgument;
    } else {
      chunkBytes = liteCollectiveChunkBytes(h, sourceBytes);
      size_t chunks =
          sourceBytes / chunkBytes + (sourceBytes % chunkBytes != 0);
      if (chunks > MSCCLPP_DEVICE_COLLECTIVE_MAX_CHUNKS) {
        status = mscclppDeviceCollectiveInvalidUsage;
      }
    }
    if (status == mscclppDeviceCollectiveSuccess) {
      chunkCount = static_cast<int>(sourceBytes / chunkBytes +
                                    (sourceBytes % chunkBytes != 0));
      epoch = atomicAdd(h.localEpoch, 1ULL) + 1ULL;
    }
  }
  __syncthreads();
  if (status != mscclppDeviceCollectiveSuccess) return status;

  int slot = static_cast<int>((epoch - 1ULL) % MSCCLPP_DEVICE_COLLECTIVE_SLOTS);
  if (tid == 0 && epoch > static_cast<unsigned long long>(
                              MSCCLPP_DEVICE_COLLECTIVE_SLOTS)) {
    unsigned long long previous = epoch - static_cast<unsigned long long>(
                                              MSCCLPP_DEVICE_COLLECTIVE_SLOTS);
    if (!liteCollectiveWaitReusable(h, slot, previous))
      status = mscclppDeviceCollectiveTransportError;
  }
  __syncthreads();

  if (status != mscclppDeviceCollectiveSuccess) return status;
  if (tid == 0) {
    *epochOut = epoch;
    *slotOut = slot;
    *chunkBytesOut = chunkBytes;
    *chunkCountOut = chunkCount;
  }
  __syncthreads();
  return mscclppDeviceCollectiveSuccess;
}

// Publish one portion of this rank's input and wait until the corresponding
// portion from every rank is readable.  All block threads participate.
static __device__ __forceinline__ int liteCollectiveStageChunkBlock(
    mscclppDeviceCollectiveHandle_t const& h, void const* srcVoid,
    size_t offset, size_t bytes, int chunk, unsigned long long epoch,
    int slot) {
  char const* src = static_cast<char const*>(srcVoid);
  char* selfSlab = mscclppDeviceCollectiveSlab(h, slot, h.rank);
  __shared__ int stageStatus;
  liteCollectiveCopyBlock(selfSlab + offset, src + offset, bytes);
  __threadfence_system();
  __syncthreads();

  if (mscclppDeviceCollectiveThreadId() == 0) {
    __threadfence_system();
    if (h.backend == mscclppDeviceCollectiveHostRdma) {
      h.publishedBytes[static_cast<size_t>(slot) * h.publishedBytesSlotStride +
                       h.rank % (h.nranks / 2)] = bytes;
      __threadfence_system();
    }
    stageStatus = mscclppDeviceCollectiveSuccess;
    liteStoreRelease(mscclppDeviceCollectiveReady(h, slot, chunk, h.rank),
                     epoch);
    __threadfence_system();
    for (int r = 0; r < h.nranks; ++r) {
      unsigned long long volatile* ready =
          mscclppDeviceCollectiveReady(h, slot, chunk, r);
      if (!liteCollectiveWait(h, ready, epoch)) {
        stageStatus = mscclppDeviceCollectiveTransportError;
        break;
      }
    }
    __threadfence_system();
  }
  __syncthreads();
  __threadfence_system();
  return stageStatus;
}

static __device__ __forceinline__ void liteCollectiveDoneBlock(
    mscclppDeviceCollectiveHandle_t const& h, int slot,
    unsigned long long epoch) {
  __threadfence_system();
  __syncthreads();
  if (mscclppDeviceCollectiveThreadId() == 0) {
    liteStoreRelease(mscclppDeviceCollectiveDone(h, slot, h.rank), epoch);
    for (int g = 0; g < h.groupCount; ++g)
      liteStoreRelease(
          h.groupDone[g] + slot * MSCCLPP_DEVICE_COLLECTIVE_MAX_RANKS, epoch);
    __threadfence_system();
  }
  __syncthreads();
}

// Post one DMA task from thread zero. FIFO wrap cannot overwrite descriptors
// whose events have not completed, even if later tasks complete first.
static __device__ __forceinline__ unsigned long long litePostTask(
    mscclppDeviceCollectiveHandle_t const& h, LiteTask const& task) {
  if (!h.tasks) return 0;
  auto* submitted = reinterpret_cast<unsigned long long*>(&h.tasks->submitted);
  unsigned long long head = liteLoadAcquire(submitted);
  if (head >= kLiteTaskSlots &&
      !liteCollectiveWait(
          h, reinterpret_cast<unsigned long long*>(&h.tasks->retired),
          head - kLiteTaskSlots + 1))
    return 0;
  h.tasks->slots[head % kLiteTaskSlots].task = task;
  liteStoreRelease(submitted, head + 1);
  return head + 1;
}

static __device__ __forceinline__ bool liteWaitTask(
    mscclppDeviceCollectiveHandle_t const& h, unsigned long long ticket) {
  return ticket &&
         liteCollectiveWait(
             h,
             reinterpret_cast<unsigned long long*>(
                 &h.tasks->slots[(ticket - 1) % kLiteTaskSlots].completed),
             ticket);
}

// AllGather chunks own individual epochs/slots. Prefetching the next chunk
// overlaps D2H with current RDMA/H2D without releasing any in-flight payload.
// graphCaptured must describe the enclosing kernel launch; a device routine
// cannot query cudaStreamIsCapturing. All ranks must pass identical sizes and
// launch mode, and all threads of exactly one CTA must participate.
static __device__ __forceinline__ int liteAllGatherBlock(
    mscclppDeviceCollectiveHandle_t const& original, void const* srcVoid,
    void* dstVoid, size_t bytesPerRank, bool graphCaptured = false) {
  // Small ordered exchange takes priority over NUMA splitting. The independent
  // initialized handle owns its own epochs, FIFOs and per-group proxies.
  bool useNuma =
      original.numaHandle && original.nranks > 0 &&
      bytesPerRank <= SIZE_MAX / static_cast<size_t>(original.nranks) &&
      bytesPerRank * original.nranks >= 128 * 1024;
  auto const& h = useNuma ? *original.numaHandle : original;
  if (!mscclppDeviceCollectiveHandleValid(h) || !srcVoid || !dstVoid)
    return mscclppDeviceCollectiveInvalidArgument;
  bool mapped =
      h.backend == mscclppDeviceCollectiveHostMemory ? h.slab != nullptr : true;
  if (h.backend != mscclppDeviceCollectiveHostMemory)
    for (int r = 0; r < h.nranks; ++r)
      mapped = mapped && h.peerSlabs[r] != nullptr;
  mapped = mapped && h.reductionsMapped;
  auto plan = litePlanAllGather(h.allGatherPolicy, h.nranks, h.ranksPerNode,
                                h.backend == mscclppDeviceCollectiveCudaIpc,
                                mapped, h.maxBytesPerRank, bytesPerRank,
                                reinterpret_cast<uintptr_t>(srcVoid) |
                                    reinterpret_cast<uintptr_t>(dstVoid),
                                graphCaptured, h.groupCount);
  if (plan.path == LiteDeviceAllGatherPath::Unsupported)
    return mscclppDeviceCollectiveInvalidUsage;
  auto* dst = static_cast<char*>(dstVoid);
  auto* src = static_cast<char const*>(srcVoid);
  if (plan.path == LiteDeviceAllGatherPath::Copy) {
    if (src != dst) liteCollectiveCopyBlock(dst, src, bytesPerRank);
    __syncthreads();
    return mscclppDeviceCollectiveSuccess;
  }
  size_t totalBytes = bytesPerRank * static_cast<size_t>(h.nranks);
  bool network = h.backend == mscclppDeviceCollectiveHostRdma;
  bool compact = network && plan.stageWithSm && h.ranksPerNode == 1 &&
                 totalBytes != 128 && totalBytes != 256;
  uint64_t networkFlags = network ? kLitePackedNetworkBit : 0;
  if (compact) networkFlags |= kLiteCompactNetworkBit;
  if (plan.path == LiteDeviceAllGatherPath::SingleSlab && !useNuma)
    networkFlags |= kLiteGenericNetworkBit;
  bool small = plan.path == LiteDeviceAllGatherPath::OrderedSmall ||
               plan.path == LiteDeviceAllGatherPath::SmallFallback;
  bool inPlace = src == dst + static_cast<size_t>(h.rank) * bytesPerRank;
  bool packedReceive =
      !plan.receiveWithSm && small &&
      (plan.path == LiteDeviceAllGatherPath::SmallFallback || h.ranksPerNode != 1 ||
       (!(inPlace || totalBytes >= 128 * 1024) || totalBytes == 32 * 1024));
  __shared__ unsigned long long epochs[2];
  __shared__ int slots[2];
  __shared__ size_t unusedBytes;
  __shared__ int unusedCount;
  __shared__ int status;
  unsigned tid = mscclppDeviceCollectiveThreadId();
  size_t chunks =
      bytesPerRank / plan.chunkBytes + (bytesPerRank % plan.chunkBytes != 0);
  bool preCopySelf =
      network && h.ranksPerNode > 1 && bytesPerRank >= 512 * 1024;
  if (preCopySelf && !inPlace)
    liteCollectiveCopyBlock(dst + h.rank * bytesPerRank, src, bytesPerRank);
  __syncthreads();
  for (size_t step = 0; step <= chunks; ++step) {
    // Stage one chunk ahead on an independent service stream. The two slots
    // are enough for a one-block send window and bidirectional copy overlap.
    if (step < chunks) {
      size_t offset = step * plan.chunkBytes;
      size_t bytes = bytesPerRank - offset < plan.chunkBytes
                         ? bytesPerRank - offset
                         : plan.chunkBytes;
      int lane = static_cast<int>(step % 2);
      int rc =
          liteCollectiveBeginBlock(h, src + offset, bytes, &epochs[lane],
                                   &slots[lane], &unusedBytes, &unusedCount);
      if (rc) return rc;
      // Inputs may have just been produced by every thread of the user CTA.
      // Publish all those stores before a CPU-submitted copy engine reads them.
      __threadfence_system();
      __syncthreads();
      if (plan.stageWithSm) {
        size_t row = compact ? ((bytes + 7) & ~size_t{7}) + 8 : bytes;
        char* target = mscclppDeviceCollectiveSlab(h, slots[lane], h.rank,
                                                   network ? row : 0);
        liteCollectiveCopyBlock(target, src + offset, bytes,
                                plan.activeThreads);
        __threadfence_system();
        __syncthreads();
        if (tid == 0) {
          if (h.publishedBytes)
            h.publishedBytes[slots[lane] * h.publishedBytesSlotStride +
                             h.rank % h.ranksPerNode] = bytes | networkFlags;
          liteStoreRelease(
              mscclppDeviceCollectiveReady(h, slots[lane], 0, h.rank),
              epochs[lane]);
        }
      } else if (tid == 0) {
        LiteTask task{};
        task.kind = LiteTaskKind::Stage;
        task.networkFlags = networkFlags;
        task.source = reinterpret_cast<uint64_t>(src + offset);
        task.bytes = bytes;
        task.slot = slots[lane];
        task.epoch = epochs[lane];
        status =
            litePostTask(h, task) ? 0 : mscclppDeviceCollectiveTransportError;
      }
      __syncthreads();
      if (!plan.stageWithSm && status) return status;
    }
    if (!step) continue;
    size_t chunk = step - 1;
    int lane = static_cast<int>(chunk % 2);
    size_t offset = chunk * plan.chunkBytes;
    size_t bytes = bytesPerRank - offset < plan.chunkBytes
                       ? bytesPerRank - offset
                       : plan.chunkBytes;
    if (tid == 0) {
      status = 0;
      for (int r = 0; r < h.nranks; ++r) {
        if (!liteCollectiveWait(
                h, mscclppDeviceCollectiveReady(h, slots[lane], 0, r),
                epochs[lane])) {
          status = mscclppDeviceCollectiveTransportError;
          break;
        }
      }
    }
    __syncthreads();
    if (status) return status;
    __threadfence_system();
    size_t row = compact ? ((bytes + 7) & ~size_t{7}) + 8 : bytes;
    if (compact) {
      if (tid == 0 &&
          !liteCollectiveWait(
              h,
              reinterpret_cast<unsigned long long*>(
                  mscclppDeviceCollectiveSlab(h, slots[lane], 1 - h.rank, row) +
                  row - 8),
              epochs[lane]))
        status = mscclppDeviceCollectiveTransportError;
      __syncthreads();
      if (status) return status;
    }
    char* self = dst + static_cast<size_t>(h.rank) * bytesPerRank + offset;
    if (!packedReceive && !preCopySelf && self != src + offset)
      liteCollectiveCopyBlock(self, src + offset, bytes, plan.activeThreads);
    if (plan.path == LiteDeviceAllGatherPath::IpcRing) {
      int previous = (h.rank + h.nranks - 1) % h.nranks;
      int next = (h.rank + 1) % h.nranks;
      for (int hop = 1; hop < h.nranks; ++hop) {
        if (tid == 0 &&
            !liteCollectiveWait(
                h,
                mscclppDeviceCollectiveReady(h, slots[lane], hop - 1, previous),
                epochs[lane]))
          status = mscclppDeviceCollectiveTransportError;
        __syncthreads();
        if (status) return status;
        __threadfence_system();
        int sourceRank = (h.rank + h.nranks - hop) % h.nranks;
        char* received =
            dst + static_cast<size_t>(sourceRank) * bytesPerRank + offset;
        liteCollectiveCopyBlock(
            received, mscclppDeviceCollectiveSlab(h, slots[lane], previous),
            bytes);
        __threadfence_system();
        __syncthreads();
        if (tid == 0) {
          // A separate per-hop credit protects the predecessor's forwarding
          // slab. Readiness alone cannot authorize overwriting its payload.
          liteStoreRelease(mscclppDeviceCollectiveReady(h, slots[lane],
                                                        h.nranks + hop, h.rank),
                           epochs[lane]);
          if (!liteCollectiveWait(h,
                                  mscclppDeviceCollectiveReady(
                                      h, slots[lane], h.nranks + hop, next),
                                  epochs[lane]))
            status = mscclppDeviceCollectiveTransportError;
        }
        __syncthreads();
        if (status) return status;
        if (hop + 1 < h.nranks) {
          liteCollectiveCopyBlock(
              mscclppDeviceCollectiveSlab(h, slots[lane], h.rank), received,
              bytes);
          __threadfence_system();
          __syncthreads();
          if (tid == 0)
            liteStoreRelease(
                mscclppDeviceCollectiveReady(h, slots[lane], hop, h.rank),
                epochs[lane]);
        }
        __syncthreads();
      }
    } else if (plan.receiveWithSm) {
      for (int hop = 1; hop < h.nranks; ++hop) {
        int r = (h.rank + h.nranks - hop) % h.nranks;
        liteCollectiveCopyBlock(
            dst + static_cast<size_t>(r) * bytesPerRank + offset,
            mscclppDeviceCollectiveSlab(h, slots[lane], r, network ? row : 0),
            bytes, plan.activeThreads);
      }
    } else if (tid == 0) {
      LiteTask task{};
      task.kind =
          packedReceive ? LiteTaskKind::GatherPacked : LiteTaskKind::Gather;
      task.networkFlags = networkFlags;
      task.destination = reinterpret_cast<uint64_t>(dst);
      task.bytes = bytes;
      task.outputStride = bytesPerRank;
      task.outputOffset = offset;
      task.slot = slots[lane];
      task.epoch = epochs[lane];
      if (!liteWaitTask(h, litePostTask(h, task)))
        status = mscclppDeviceCollectiveTransportError;
    }
    __syncthreads();
    if (status) return status;
    liteCollectiveDoneBlock(h, slots[lane], epochs[lane]);
  }
  return mscclppDeviceCollectiveSuccess;
}

// Block-scoped AllReduce.  Each rank contributes count elements and receives
// the element-wise reduction of all ranks.  T must support +, <, and >.
template <typename T>
static __device__ __forceinline__ int liteAllReduceBlock(
    mscclppDeviceCollectiveHandle_t const& h, T const* src, T* dst,
    size_t count, liteReduceOp op = liteReduceSum) {
  __shared__ unsigned long long epoch;
  __shared__ int slot;
  __shared__ size_t chunkBytes;
  __shared__ int chunkCount;
  if (h.nranks > 1 && h.backend == mscclppDeviceCollectiveHostMemory && !h.slab)
    return mscclppDeviceCollectiveInvalidUsage;
  if (h.backend == mscclppDeviceCollectiveHostRdma)
    for (int r = 0; r < h.nranks && r < MSCCLPP_DEVICE_COLLECTIVE_MAX_RANKS;
         ++r)
      if (!h.peerSlabs[r]) return mscclppDeviceCollectiveInvalidUsage;
  if (h.nranks > 1 && !h.reductionsMapped)
    return mscclppDeviceCollectiveInvalidUsage;
  size_t sourceBytes = count * sizeof(T);
  if (src == nullptr || dst == nullptr || count == 0 ||
      sourceBytes / sizeof(T) != count ||
      ((reinterpret_cast<uintptr_t>(src) | reinterpret_cast<uintptr_t>(dst)) &
       (alignof(T) - 1)) != 0 ||
      (h.maxBytesPerRank % alignof(T)) != 0 || op < liteReduceSum ||
      op > liteReduceMax) {
    return mscclppDeviceCollectiveInvalidArgument;
  }
  if (mscclppDeviceCollectiveHandleValid(h) && h.nranks == 1 &&
      sourceBytes <= h.maxBytesPerRank) {
    if (src != dst)
      liteCollectiveCopyBlock(reinterpret_cast<char*>(dst),
                              reinterpret_cast<char const*>(src), sourceBytes);
    __syncthreads();
    return mscclppDeviceCollectiveSuccess;
  }
  int rc = liteCollectiveBeginBlock(h, src, sourceBytes, &epoch, &slot,
                                    &chunkBytes, &chunkCount);
  if (rc != mscclppDeviceCollectiveSuccess) return rc;

  unsigned int tid = mscclppDeviceCollectiveThreadId();
  unsigned int threadCount = mscclppDeviceCollectiveThreadCount();
  for (int chunk = 0; chunk < chunkCount; ++chunk) {
    size_t byteOffset = static_cast<size_t>(chunk) * chunkBytes;
    size_t bytes = sourceBytes - byteOffset < chunkBytes
                       ? sourceBytes - byteOffset
                       : chunkBytes;
    rc = liteCollectiveStageChunkBlock(h, src, byteOffset, bytes, chunk, epoch,
                                       slot);
    if (rc != mscclppDeviceCollectiveSuccess) return rc;
    size_t first = byteOffset / sizeof(T);
    size_t elements = bytes / sizeof(T);
    T const* rankZero =
        reinterpret_cast<T const*>(mscclppDeviceCollectiveSlab(h, slot, 0));
    for (size_t local = tid; local < elements; local += threadCount) {
      size_t i = first + local;
      T value = rankZero[i];
      for (int r = 1; r < h.nranks; ++r) {
        T const* peer =
            reinterpret_cast<T const*>(mscclppDeviceCollectiveSlab(h, slot, r));
        value = liteApplyReduction(value, peer[i], op);
      }
      dst[i] = value;
    }
    __syncthreads();
  }
  liteCollectiveDoneBlock(h, slot, epoch);
  return mscclppDeviceCollectiveSuccess;
}

// Block-scoped ReduceScatter.  Each rank contributes nranks * recvCount
// elements.  Rank r receives the reduction of shard r, containing recvCount
// elements.  T must support +, <, and >.
template <typename T>
static __device__ __forceinline__ int liteReduceScatterBlock(
    mscclppDeviceCollectiveHandle_t const& h, T const* src, T* dst,
    size_t recvCount, liteReduceOp op = liteReduceSum) {
  if (src == nullptr || dst == nullptr || h.nranks < 1 ||
      h.nranks > MSCCLPP_DEVICE_COLLECTIVE_MAX_RANKS || recvCount == 0 ||
      recvCount > static_cast<size_t>(-1) / static_cast<size_t>(h.nranks)) {
    return mscclppDeviceCollectiveInvalidArgument;
  }
  if (h.nranks > 1 && h.backend == mscclppDeviceCollectiveHostMemory && !h.slab)
    return mscclppDeviceCollectiveInvalidUsage;
  if (h.backend == mscclppDeviceCollectiveHostRdma)
    for (int r = 0; r < h.nranks && r < MSCCLPP_DEVICE_COLLECTIVE_MAX_RANKS;
         ++r)
      if (!h.peerSlabs[r]) return mscclppDeviceCollectiveInvalidUsage;
  if (h.nranks > 1 && !h.reductionsMapped)
    return mscclppDeviceCollectiveInvalidUsage;
  size_t inputCount = recvCount * static_cast<size_t>(h.nranks);
  if (inputCount > static_cast<size_t>(-1) / sizeof(T) ||
      ((reinterpret_cast<uintptr_t>(src) | reinterpret_cast<uintptr_t>(dst)) &
       (alignof(T) - 1)) != 0 ||
      (h.maxBytesPerRank % alignof(T)) != 0 || op < liteReduceSum ||
      op > liteReduceMax) {
    return mscclppDeviceCollectiveInvalidArgument;
  }

  __shared__ unsigned long long epoch;
  __shared__ int slot;
  __shared__ size_t chunkBytes;
  __shared__ int chunkCount;
  size_t sourceBytes = inputCount * sizeof(T);
  if (mscclppDeviceCollectiveHandleValid(h) && h.nranks == 1 &&
      sourceBytes <= h.maxBytesPerRank) {
    if (src != dst)
      liteCollectiveCopyBlock(reinterpret_cast<char*>(dst),
                              reinterpret_cast<char const*>(src), sourceBytes);
    __syncthreads();
    return mscclppDeviceCollectiveSuccess;
  }
  int rc = liteCollectiveBeginBlock(h, src, sourceBytes, &epoch, &slot,
                                    &chunkBytes, &chunkCount);
  if (rc != mscclppDeviceCollectiveSuccess) return rc;

  unsigned int tid = mscclppDeviceCollectiveThreadId();
  unsigned int threadCount = mscclppDeviceCollectiveThreadCount();
  size_t shardBegin = static_cast<size_t>(h.rank) * recvCount;
  size_t shardEnd = shardBegin + recvCount;
  for (int chunk = 0; chunk < chunkCount; ++chunk) {
    size_t byteOffset = static_cast<size_t>(chunk) * chunkBytes;
    size_t bytes = sourceBytes - byteOffset < chunkBytes
                       ? sourceBytes - byteOffset
                       : chunkBytes;
    rc = liteCollectiveStageChunkBlock(h, src, byteOffset, bytes, chunk, epoch,
                                       slot);
    if (rc != mscclppDeviceCollectiveSuccess) return rc;
    size_t chunkBegin = byteOffset / sizeof(T);
    size_t chunkEnd = chunkBegin + bytes / sizeof(T);
    size_t reduceBegin = chunkBegin > shardBegin ? chunkBegin : shardBegin;
    size_t reduceEnd = chunkEnd < shardEnd ? chunkEnd : shardEnd;
    T const* rankZero =
        reinterpret_cast<T const*>(mscclppDeviceCollectiveSlab(h, slot, 0));
    for (size_t sourceIndex = reduceBegin + tid; sourceIndex < reduceEnd;
         sourceIndex += threadCount) {
      T value = rankZero[sourceIndex];
      for (int r = 1; r < h.nranks; ++r) {
        T const* peer =
            reinterpret_cast<T const*>(mscclppDeviceCollectiveSlab(h, slot, r));
        value = liteApplyReduction(value, peer[sourceIndex], op);
      }
      dst[sourceIndex - shardBegin] = value;
    }
    __syncthreads();
  }
  liteCollectiveDoneBlock(h, slot, epoch);
  return mscclppDeviceCollectiveSuccess;
}

#endif  // defined(__CUDACC__)
