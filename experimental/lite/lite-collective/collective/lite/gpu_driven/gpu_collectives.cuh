/*
Device-driven collectives for UCCL-lite.

A handle is initialized collectively on the host once, then copied by value into a user kernel.  
The block routine is called by every thread in exactly one block per rank, 
in the same order on every rank.
*/

#pragma once

#include "nccl.h"

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
    ncclComm_t comm, 
    size_t maxBytesPerRank,
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
};

enum liteReduceOp {
  liteReduceSum = 0,
  liteReduceMin = 1,
  liteReduceMax = 2,
};

static __device__ __forceinline__ volatile unsigned long long*
mscclppDeviceCollectiveReady(const mscclppDeviceCollectiveHandle_t& h,
                            int slot, int chunk, int rank) {
  if (h.backend == mscclppDeviceCollectiveHostRdma) {
    return h.peerReady[rank] +
           static_cast<size_t>(slot) * h.peerReadySlotStride[rank];
  }
  if (h.backend == mscclppDeviceCollectiveCudaIpc) {
    return h.peerReady[rank] +
           static_cast<size_t>(slot) * h.peerReadySlotStride[rank] + chunk;
  }
  size_t index = static_cast<size_t>(slot) *
                     (MSCCLPP_DEVICE_COLLECTIVE_MAX_CHUNKS *
                      MSCCLPP_DEVICE_COLLECTIVE_MAX_RANKS) +
                 static_cast<size_t>(chunk) *
                     MSCCLPP_DEVICE_COLLECTIVE_MAX_RANKS +
                 static_cast<size_t>(rank);
  return reinterpret_cast<volatile unsigned long long*>(
      h.control + index * h.counterStride);
}

static __device__ __forceinline__ volatile unsigned long long*
mscclppDeviceCollectiveDone(const mscclppDeviceCollectiveHandle_t& h,
                           int slot, int rank) {
  if (h.backend == mscclppDeviceCollectiveCudaIpc ||
      h.backend == mscclppDeviceCollectiveHostRdma) {
    return h.peerDone[rank] +
           static_cast<size_t>(slot) * h.peerDoneSlotStride[rank];
  }
  size_t index =
      MSCCLPP_DEVICE_COLLECTIVE_SLOTS *
          MSCCLPP_DEVICE_COLLECTIVE_MAX_CHUNKS *
          MSCCLPP_DEVICE_COLLECTIVE_MAX_RANKS +
      static_cast<size_t>(slot) * MSCCLPP_DEVICE_COLLECTIVE_MAX_RANKS +
      static_cast<size_t>(rank);
  return reinterpret_cast<volatile unsigned long long*>(
      h.control + index * h.counterStride);
}

static __device__ __forceinline__ char* mscclppDeviceCollectiveSlab(
    const mscclppDeviceCollectiveHandle_t& h, int slot, int rank) {
  if (h.backend == mscclppDeviceCollectiveCudaIpc ||
      h.backend == mscclppDeviceCollectiveHostRdma) {
    size_t stride = h.backend == mscclppDeviceCollectiveHostRdma
                        ? h.slotStride
                        : h.maxBytesPerRank;
    return h.peerSlabs[rank] + static_cast<size_t>(slot) * stride;
  }
  return h.slab + static_cast<size_t>(slot) * h.slotStride +
         static_cast<size_t>(rank) * h.maxBytesPerRank;
}

static __device__ __forceinline__ bool mscclppDeviceCollectiveHandleValid(
    const mscclppDeviceCollectiveHandle_t& h) {
  if (h.localEpoch == nullptr || h.nranks < 2 ||
      h.nranks > MSCCLPP_DEVICE_COLLECTIVE_MAX_RANKS || h.rank < 0 ||
      h.rank >= h.nranks) {
    return false;
  }
  if (h.backend == mscclppDeviceCollectiveHostMemory) {
    return h.slab != nullptr && h.control != nullptr;
  }
  if (h.backend != mscclppDeviceCollectiveCudaIpc &&
      h.backend != mscclppDeviceCollectiveHostRdma) {
    return false;
  }
  for (int r = 0; r < h.nranks; ++r) {
    if (h.peerSlabs[r] == nullptr || h.peerReady[r] == nullptr ||
        h.peerDone[r] == nullptr) {
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

static __device__ __forceinline__ void liteCollectiveWaitReusable(
    const mscclppDeviceCollectiveHandle_t& h, int slot,
    unsigned long long previous) {
  if (h.backend == mscclppDeviceCollectiveHostRdma) {
    volatile unsigned long long* reusable = h.slotReusable + slot;
    while (*reusable < previous) {
    }
    return;
  }
  for (int r = 0; r < h.nranks; ++r) {
    volatile unsigned long long* done =
        mscclppDeviceCollectiveDone(h, slot, r);
    while (*done < previous) {
    }
  }
}

static __device__ __forceinline__ unsigned int
mscclppDeviceCollectiveThreadId() {
  return threadIdx.x + blockDim.x *
      (threadIdx.y + blockDim.y * threadIdx.z);
}

static __device__ __forceinline__ unsigned int
mscclppDeviceCollectiveThreadCount() {
  return blockDim.x * blockDim.y * blockDim.z;
}

static __device__ __forceinline__ void liteCollectiveCopyBlock(
    char* dst, const char* src, size_t bytes) {
  unsigned int tid = mscclppDeviceCollectiveThreadId();
  unsigned int threadCount = mscclppDeviceCollectiveThreadCount();
  uintptr_t alignment = reinterpret_cast<uintptr_t>(dst) |
                        reinterpret_cast<uintptr_t>(src);
  if ((alignment & (alignof(unsigned long long) - 1)) == 0) {
    size_t words = bytes / sizeof(unsigned long long);
    auto* wordDst = reinterpret_cast<unsigned long long*>(dst);
    const auto* wordSrc = reinterpret_cast<const unsigned long long*>(src);
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
    const mscclppDeviceCollectiveHandle_t& h, size_t sourceBytes) {
  // The host-driven DMA path uses 256 KiB chunks.  A device routine cannot
  // launch that CPU/copy-engine pipeline, so mapped-host and CUDA-IPC backends
  // use the same partitioning with SM copies and per-chunk ready flags.  The
  // two-node proxy currently transfers one complete message per epoch.
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
    const mscclppDeviceCollectiveHandle_t& h, const void* srcVoid,
    size_t sourceBytes, unsigned long long* epochOut, int* slotOut,
    size_t* chunkBytesOut, int* chunkCountOut) {
  __shared__ unsigned long long epoch;
  __shared__ int status;
  __shared__ size_t chunkBytes;
  __shared__ int chunkCount;
  unsigned int tid = mscclppDeviceCollectiveThreadId();

  if (tid == 0) {
    status = mscclppDeviceCollectiveSuccess;
    if (!mscclppDeviceCollectiveHandleValid(h) || srcVoid == nullptr ||
        sourceBytes == 0 || sourceBytes > h.maxBytesPerRank || h.nranks < 2 ||
        h.nranks > MSCCLPP_DEVICE_COLLECTIVE_MAX_RANKS || h.rank < 0 ||
        h.rank >= h.nranks) {
      status = mscclppDeviceCollectiveInvalidArgument;
    } else {
      chunkBytes = liteCollectiveChunkBytes(h, sourceBytes);
      size_t chunks = (sourceBytes + chunkBytes - 1) / chunkBytes;
      if (chunks > MSCCLPP_DEVICE_COLLECTIVE_MAX_CHUNKS) {
        status = mscclppDeviceCollectiveInvalidUsage;
      }
    }
    if (status == mscclppDeviceCollectiveSuccess) {
      chunkCount = static_cast<int>((sourceBytes + chunkBytes - 1) /
                                    chunkBytes);
      epoch = atomicAdd(h.localEpoch, 1ULL) + 1ULL;
    }
  }
  __syncthreads();
  if (status != mscclppDeviceCollectiveSuccess) return status;

  int slot = static_cast<int>((epoch - 1ULL) %
                              MSCCLPP_DEVICE_COLLECTIVE_SLOTS);
  if (tid == 0 &&
      epoch > static_cast<unsigned long long>(MSCCLPP_DEVICE_COLLECTIVE_SLOTS)) {
    unsigned long long previous =
        epoch - static_cast<unsigned long long>(MSCCLPP_DEVICE_COLLECTIVE_SLOTS);
    liteCollectiveWaitReusable(h, slot, previous);
  }
  __syncthreads();

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
static __device__ __forceinline__ void liteCollectiveStageChunkBlock(
    const mscclppDeviceCollectiveHandle_t& h, const void* srcVoid,
    size_t offset, size_t bytes, int chunk, unsigned long long epoch,
    int slot) {
  const char* src = static_cast<const char*>(srcVoid);
  char* selfSlab = mscclppDeviceCollectiveSlab(h, slot, h.rank);
  liteCollectiveCopyBlock(selfSlab + offset, src + offset, bytes);
  __syncthreads();

  if (mscclppDeviceCollectiveThreadId() == 0) {
    __threadfence_system();
    if (h.backend == mscclppDeviceCollectiveHostRdma) {
      h.publishedBytes[static_cast<size_t>(slot) *
                       h.publishedBytesSlotStride +
                       h.rank % (h.nranks / 2)] = bytes;
      __threadfence_system();
    }
    *mscclppDeviceCollectiveReady(h, slot, chunk, h.rank) = epoch;
    __threadfence_system();
    for (int r = 0; r < h.nranks; ++r) {
      volatile unsigned long long* ready =
          mscclppDeviceCollectiveReady(h, slot, chunk, r);
      while (*ready < epoch) {
      }
    }
    __threadfence_system();
  }
  __syncthreads();
}

static __device__ __forceinline__ void liteCollectiveDoneBlock(
    const mscclppDeviceCollectiveHandle_t& h, int slot,
    unsigned long long epoch) {
  if (mscclppDeviceCollectiveThreadId() == 0) {
    __threadfence_system();
    *mscclppDeviceCollectiveDone(h, slot, h.rank) = epoch;
    __threadfence_system();
  }
  __syncthreads();
}

// Device-callable, block-scoped AllGather.
//
// Supported MVP path:
//   * one node or two balanced nodes, and at most 8 ranks;
//   * exactly one participating block per rank;
//   * all threads in that block call this routine with identical arguments;
//   * src/dst are device-accessible and bytesPerRank is identical on all ranks;
//   * calls occur in identical program order on all ranks.
//
// dst contains nranks consecutive blocks of bytesPerRank bytes in rank order.
// Depending on the initialized backend, SM loads/stores target either mapped
// pinned host memory or peer GPU memory opened through CUDA IPC. The operation
// does not launch a child kernel or require a host call per invocation.
static __device__ __forceinline__ int liteAllGatherBlock(
    const mscclppDeviceCollectiveHandle_t& h, 
    const void* srcVoid,
    void* dstVoid, 
    size_t bytesPerRank) {
  if (dstVoid == nullptr) {
    return mscclppDeviceCollectiveInvalidArgument;
  }

  __shared__ unsigned long long epoch;
  __shared__ int slot;
  __shared__ size_t chunkBytes;
  __shared__ int chunkCount;
  int rc = liteCollectiveBeginBlock(h, srcVoid, bytesPerRank, &epoch, &slot,
                                    &chunkBytes, &chunkCount);
  if (rc != mscclppDeviceCollectiveSuccess) return rc;

  char* dst = static_cast<char*>(dstVoid);
  for (int chunk = 0; chunk < chunkCount; ++chunk) {
    size_t offset = static_cast<size_t>(chunk) * chunkBytes;
    size_t bytes = bytesPerRank - offset < chunkBytes
                       ? bytesPerRank - offset
                       : chunkBytes;
    liteCollectiveStageChunkBlock(h, srcVoid, offset, bytes, chunk, epoch,
                                  slot);
    const char* src = static_cast<const char*>(srcVoid);
    for (int sourceRank = 0; sourceRank < h.nranks; ++sourceRank) {
      char* rankDst = dst + static_cast<size_t>(sourceRank) * bytesPerRank;
      const char* rankSrc = sourceRank == h.rank
                                ? src
                                : mscclppDeviceCollectiveSlab(
                                      h, slot, sourceRank);
      liteCollectiveCopyBlock(rankDst + offset, rankSrc + offset, bytes);
    }
    __syncthreads();
  }

  liteCollectiveDoneBlock(h, slot, epoch);
  return mscclppDeviceCollectiveSuccess;
}

// Block-scoped AllReduce.  Each rank contributes count elements and receives
// the element-wise reduction of all ranks.  T must support +, <, and >.
template <typename T>
static __device__ __forceinline__ int liteAllReduceBlock(
    const mscclppDeviceCollectiveHandle_t& h, const T* src, T* dst,
    size_t count, liteReduceOp op = liteReduceSum) {
  __shared__ unsigned long long epoch;
  __shared__ int slot;
  __shared__ size_t chunkBytes;
  __shared__ int chunkCount;
  size_t sourceBytes = count * sizeof(T);
  if (src == nullptr || dst == nullptr || count == 0 ||
      sourceBytes / sizeof(T) != count ||
      ((reinterpret_cast<uintptr_t>(src) |
        reinterpret_cast<uintptr_t>(dst)) & (alignof(T) - 1)) != 0 ||
      (h.maxBytesPerRank % alignof(T)) != 0 || op < liteReduceSum ||
      op > liteReduceMax) {
    return mscclppDeviceCollectiveInvalidArgument;
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
    liteCollectiveStageChunkBlock(h, src, byteOffset, bytes, chunk, epoch,
                                  slot);
    size_t first = byteOffset / sizeof(T);
    size_t elements = bytes / sizeof(T);
    const T* rankZero = reinterpret_cast<const T*>(
        mscclppDeviceCollectiveSlab(h, slot, 0));
    for (size_t local = tid; local < elements; local += threadCount) {
      size_t i = first + local;
      T value = rankZero[i];
      for (int r = 1; r < h.nranks; ++r) {
        const T* peer = reinterpret_cast<const T*>(
            mscclppDeviceCollectiveSlab(h, slot, r));
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
    const mscclppDeviceCollectiveHandle_t& h, const T* src, T* dst,
    size_t recvCount, liteReduceOp op = liteReduceSum) {
  if (src == nullptr || dst == nullptr || h.nranks < 2 ||
      h.nranks > MSCCLPP_DEVICE_COLLECTIVE_MAX_RANKS || recvCount == 0 ||
      recvCount > static_cast<size_t>(-1) / static_cast<size_t>(h.nranks)) {
    return mscclppDeviceCollectiveInvalidArgument;
  }
  size_t inputCount = recvCount * static_cast<size_t>(h.nranks);
  if (inputCount > static_cast<size_t>(-1) / sizeof(T) ||
      ((reinterpret_cast<uintptr_t>(src) |
        reinterpret_cast<uintptr_t>(dst)) & (alignof(T) - 1)) != 0 ||
      (h.maxBytesPerRank % alignof(T)) != 0 || op < liteReduceSum ||
      op > liteReduceMax) {
    return mscclppDeviceCollectiveInvalidArgument;
  }

  __shared__ unsigned long long epoch;
  __shared__ int slot;
  __shared__ size_t chunkBytes;
  __shared__ int chunkCount;
  size_t sourceBytes = inputCount * sizeof(T);
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
    liteCollectiveStageChunkBlock(h, src, byteOffset, bytes, chunk, epoch,
                                  slot);
    size_t chunkBegin = byteOffset / sizeof(T);
    size_t chunkEnd = chunkBegin + bytes / sizeof(T);
    size_t reduceBegin = chunkBegin > shardBegin ? chunkBegin : shardBegin;
    size_t reduceEnd = chunkEnd < shardEnd ? chunkEnd : shardEnd;
    const T* rankZero = reinterpret_cast<const T*>(
        mscclppDeviceCollectiveSlab(h, slot, 0));
    for (size_t sourceIndex = reduceBegin + tid; sourceIndex < reduceEnd;
         sourceIndex += threadCount) {
      T value = rankZero[sourceIndex];
      for (int r = 1; r < h.nranks; ++r) {
        const T* peer = reinterpret_cast<const T*>(
            mscclppDeviceCollectiveSlab(h, slot, r));
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
