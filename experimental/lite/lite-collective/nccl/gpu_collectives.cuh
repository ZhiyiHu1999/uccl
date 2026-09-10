// Experimental device-callable collectives for UCCL-lite.
//
// This API is intentionally separate from NCCL's host API.  A handle is
// initialized collectively on the host once, then copied by value into a user
// kernel.  The block routine is called by every thread in exactly one block per
// rank, in the same order on every rank.
#pragma once

#include "nccl.h"

#include <stddef.h>
#include <stdint.h>

static constexpr int MSCCLPP_DEVICE_ALLGATHER_MAX_RANKS = 8;
static constexpr int MSCCLPP_DEVICE_ALLGATHER_SLOTS = 2;
static constexpr int MSCCLPP_DEVICE_ALLGATHER_MAX_CHUNKS = 1024;

typedef struct mscclppDeviceAllGatherHandle {
  char* slab;
  char* control;
  unsigned long long* localEpoch;
  size_t maxBytesPerRank;
  size_t slotStride;
  size_t counterStride;
  int rank;
  int nranks;
} mscclppDeviceAllGatherHandle_t;

#ifdef __cplusplus
extern "C" {
#endif

// Collective host-side initialization.  Every rank in comm must call with the
// same maxBytesPerRank.  Only single-node communicators are supported for now.
ncclResult_t mscclppGetDeviceAllGatherHandle(
    ncclComm_t comm, size_t maxBytesPerRank,
    mscclppDeviceAllGatherHandle_t* handle);

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
mscclppDeviceAllGatherReady(const mscclppDeviceAllGatherHandle_t& h,
                            int slot, int rank) {
  size_t index = static_cast<size_t>(slot) *
                     (MSCCLPP_DEVICE_ALLGATHER_MAX_CHUNKS *
                      MSCCLPP_DEVICE_ALLGATHER_MAX_RANKS) +
                 static_cast<size_t>(rank);
  return reinterpret_cast<volatile unsigned long long*>(
      h.control + index * h.counterStride);
}

static __device__ __forceinline__ volatile unsigned long long*
mscclppDeviceAllGatherDone(const mscclppDeviceAllGatherHandle_t& h,
                           int slot, int rank) {
  size_t index =
      MSCCLPP_DEVICE_ALLGATHER_SLOTS *
          MSCCLPP_DEVICE_ALLGATHER_MAX_CHUNKS *
          MSCCLPP_DEVICE_ALLGATHER_MAX_RANKS +
      static_cast<size_t>(slot) * MSCCLPP_DEVICE_ALLGATHER_MAX_RANKS +
      static_cast<size_t>(rank);
  return reinterpret_cast<volatile unsigned long long*>(
      h.control + index * h.counterStride);
}

static __device__ __forceinline__ unsigned int
mscclppDeviceAllGatherThreadId() {
  return threadIdx.x + blockDim.x *
      (threadIdx.y + blockDim.y * threadIdx.z);
}

static __device__ __forceinline__ unsigned int
mscclppDeviceAllGatherThreadCount() {
  return blockDim.x * blockDim.y * blockDim.z;
}

template <typename T>
static __device__ __forceinline__ T liteApplyReduction(T lhs, T rhs,
                                                       liteReduceOp op) {
  if (op == liteReduceMin) return rhs < lhs ? rhs : lhs;
  if (op == liteReduceMax) return rhs > lhs ? rhs : lhs;
  return lhs + rhs;
}

// Allocate the next collective epoch, wait for its slot to be reusable, copy
// the local input into that slot, and wait until every rank has published its
// input.  All block threads must call this helper.
static __device__ __forceinline__ int liteCollectiveStageBlock(
    const mscclppDeviceAllGatherHandle_t& h, const void* srcVoid,
    size_t sourceBytes, unsigned long long* epochOut, int* slotOut) {
  __shared__ unsigned long long epoch;
  __shared__ int status;
  unsigned int tid = mscclppDeviceAllGatherThreadId();
  unsigned int threadCount = mscclppDeviceAllGatherThreadCount();

  if (tid == 0) {
    status = mscclppDeviceCollectiveSuccess;
    if (h.slab == nullptr || h.control == nullptr || h.localEpoch == nullptr ||
        srcVoid == nullptr || sourceBytes == 0 ||
        sourceBytes > h.maxBytesPerRank || h.nranks < 2 ||
        h.nranks > MSCCLPP_DEVICE_ALLGATHER_MAX_RANKS || h.rank < 0 ||
        h.rank >= h.nranks) {
      status = mscclppDeviceCollectiveInvalidArgument;
    } else {
      epoch = atomicAdd(h.localEpoch, 1ULL) + 1ULL;
    }
  }
  __syncthreads();
  if (status != mscclppDeviceCollectiveSuccess) return status;

  int slot = static_cast<int>((epoch - 1ULL) %
                              MSCCLPP_DEVICE_ALLGATHER_SLOTS);
  if (tid == 0 &&
      epoch > static_cast<unsigned long long>(MSCCLPP_DEVICE_ALLGATHER_SLOTS)) {
    unsigned long long previous =
        epoch - static_cast<unsigned long long>(MSCCLPP_DEVICE_ALLGATHER_SLOTS);
    for (int r = 0; r < h.nranks; ++r) {
      volatile unsigned long long* done =
          mscclppDeviceAllGatherDone(h, slot, r);
      while (*done < previous) {
      }
    }
  }
  __syncthreads();

  const char* src = static_cast<const char*>(srcVoid);
  char* selfSlab = h.slab + static_cast<size_t>(slot) * h.slotStride +
                   static_cast<size_t>(h.rank) * h.maxBytesPerRank;
  for (size_t i = tid; i < sourceBytes; i += threadCount) selfSlab[i] = src[i];
  __syncthreads();

  if (tid == 0) {
    __threadfence_system();
    *mscclppDeviceAllGatherReady(h, slot, h.rank) = epoch;
    __threadfence_system();
    for (int r = 0; r < h.nranks; ++r) {
      volatile unsigned long long* ready =
          mscclppDeviceAllGatherReady(h, slot, r);
      while (*ready < epoch) {
      }
    }
    __threadfence_system();
    *epochOut = epoch;
    *slotOut = slot;
  }
  __syncthreads();
  return mscclppDeviceCollectiveSuccess;
}

static __device__ __forceinline__ void liteCollectiveDoneBlock(
    const mscclppDeviceAllGatherHandle_t& h, int slot,
    unsigned long long epoch) {
  if (mscclppDeviceAllGatherThreadId() == 0) {
    __threadfence_system();
    *mscclppDeviceAllGatherDone(h, slot, h.rank) = epoch;
    __threadfence_system();
  }
  __syncthreads();
}

// Device-callable, block-scoped AllGather.
//
// Supported MVP path:
//   * a single node and at most 8 ranks;
//   * exactly one participating block per rank;
//   * all threads in that block call this routine with identical arguments;
//   * src/dst are device-accessible and bytesPerRank is identical on all ranks;
//   * calls occur in identical program order on all ranks.
//
// dst contains nranks consecutive blocks of bytesPerRank bytes in rank order.
// The implementation uses SM loads/stores to a device-mapped shared pinned-host
// slab.  It does not launch a child kernel or require a host call per operation.
static __device__ __forceinline__ int liteAllGatherBlock(
    const mscclppDeviceAllGatherHandle_t& h, const void* srcVoid,
    void* dstVoid, size_t bytesPerRank) {
  __shared__ unsigned long long epoch;
  __shared__ int status;
  unsigned int tid = mscclppDeviceAllGatherThreadId();
  unsigned int threadCount = mscclppDeviceAllGatherThreadCount();

  if (tid == 0) {
    status = mscclppDeviceCollectiveSuccess;
    if (h.slab == nullptr || h.control == nullptr || h.localEpoch == nullptr ||
        srcVoid == nullptr || dstVoid == nullptr || bytesPerRank == 0 ||
        bytesPerRank > h.maxBytesPerRank || h.nranks < 2 ||
        h.nranks > MSCCLPP_DEVICE_ALLGATHER_MAX_RANKS || h.rank < 0 ||
        h.rank >= h.nranks) {
      status = mscclppDeviceCollectiveInvalidArgument;
    } else {
      epoch = atomicAdd(h.localEpoch, 1ULL) + 1ULL;
    }
  }
  __syncthreads();
  if (status != mscclppDeviceCollectiveSuccess) return status;

  int slot = static_cast<int>((epoch - 1ULL) %
                              MSCCLPP_DEVICE_ALLGATHER_SLOTS);

  // Do not overwrite a double-buffered slot until every rank finished its
  // previous use.  Only one thread polls; the whole block joins below.
  if (tid == 0 &&
      epoch > static_cast<unsigned long long>(MSCCLPP_DEVICE_ALLGATHER_SLOTS)) {
    unsigned long long previous =
        epoch - static_cast<unsigned long long>(MSCCLPP_DEVICE_ALLGATHER_SLOTS);
    for (int r = 0; r < h.nranks; ++r) {
      volatile unsigned long long* done =
          mscclppDeviceAllGatherDone(h, slot, r);
      while (*done < previous) {
      }
    }
  }
  __syncthreads();

  const char* src = static_cast<const char*>(srcVoid);
  char* dst = static_cast<char*>(dstVoid);
  char* selfSlab = h.slab + static_cast<size_t>(slot) * h.slotStride +
                   static_cast<size_t>(h.rank) * h.maxBytesPerRank;

  for (size_t i = tid; i < bytesPerRank; i += threadCount) {
    selfSlab[i] = src[i];
  }
  __syncthreads();

  if (tid == 0) {
    __threadfence_system();
    *mscclppDeviceAllGatherReady(h, slot, h.rank) = epoch;
    __threadfence_system();
    for (int r = 0; r < h.nranks; ++r) {
      volatile unsigned long long* ready =
          mscclppDeviceAllGatherReady(h, slot, r);
      while (*ready < epoch) {
      }
    }
    __threadfence_system();
  }
  __syncthreads();

  size_t totalBytes = bytesPerRank * static_cast<size_t>(h.nranks);
  for (size_t i = tid; i < totalBytes; i += threadCount) {
    int sourceRank = static_cast<int>(i / bytesPerRank);
    size_t sourceOffset = i - static_cast<size_t>(sourceRank) * bytesPerRank;
    dst[i] = h.slab[static_cast<size_t>(slot) * h.slotStride +
                    static_cast<size_t>(sourceRank) * h.maxBytesPerRank +
                    sourceOffset];
  }
  __syncthreads();

  if (tid == 0) {
    __threadfence_system();
    *mscclppDeviceAllGatherDone(h, slot, h.rank) = epoch;
    __threadfence_system();
  }
  __syncthreads();
  return mscclppDeviceCollectiveSuccess;
}

// Block-scoped AllReduce.  Each rank contributes count elements and receives
// the element-wise reduction of all ranks.  T must support +, <, and >.
template <typename T>
static __device__ __forceinline__ int liteAllReduceBlock(
    const mscclppDeviceAllGatherHandle_t& h, const T* src, T* dst,
    size_t count, liteReduceOp op = liteReduceSum) {
  __shared__ unsigned long long epoch;
  __shared__ int slot;
  size_t sourceBytes = count * sizeof(T);
  if (count == 0 || sourceBytes / sizeof(T) != count ||
      (h.maxBytesPerRank % alignof(T)) != 0 || op < liteReduceSum ||
      op > liteReduceMax) {
    return mscclppDeviceCollectiveInvalidArgument;
  }
  int rc = liteCollectiveStageBlock(h, src, sourceBytes, &epoch, &slot);
  if (rc != mscclppDeviceCollectiveSuccess) return rc;

  unsigned int tid = mscclppDeviceAllGatherThreadId();
  unsigned int threadCount = mscclppDeviceAllGatherThreadCount();
  char* slotBase = h.slab + static_cast<size_t>(slot) * h.slotStride;
  for (size_t i = tid; i < count; i += threadCount) {
    T value = reinterpret_cast<const T*>(slotBase)[i];
    for (int r = 1; r < h.nranks; ++r) {
      const T* peer = reinterpret_cast<const T*>(
          slotBase + static_cast<size_t>(r) * h.maxBytesPerRank);
      value = liteApplyReduction(value, peer[i], op);
    }
    dst[i] = value;
  }
  __syncthreads();
  liteCollectiveDoneBlock(h, slot, epoch);
  return mscclppDeviceCollectiveSuccess;
}

// Block-scoped ReduceScatter.  Each rank contributes nranks * recvCount
// elements.  Rank r receives the reduction of shard r, containing recvCount
// elements.  T must support +, <, and >.
template <typename T>
static __device__ __forceinline__ int liteReduceScatterBlock(
    const mscclppDeviceAllGatherHandle_t& h, const T* src, T* dst,
    size_t recvCount, liteReduceOp op = liteReduceSum) {
  if (h.nranks < 2 ||
      h.nranks > MSCCLPP_DEVICE_ALLGATHER_MAX_RANKS || recvCount == 0 ||
      recvCount > static_cast<size_t>(-1) / static_cast<size_t>(h.nranks)) {
    return mscclppDeviceCollectiveInvalidArgument;
  }
  size_t inputCount = recvCount * static_cast<size_t>(h.nranks);
  if (inputCount > static_cast<size_t>(-1) / sizeof(T) ||
      (h.maxBytesPerRank % alignof(T)) != 0 || op < liteReduceSum ||
      op > liteReduceMax) {
    return mscclppDeviceCollectiveInvalidArgument;
  }

  __shared__ unsigned long long epoch;
  __shared__ int slot;
  int rc = liteCollectiveStageBlock(
      h, src, inputCount * sizeof(T), &epoch, &slot);
  if (rc != mscclppDeviceCollectiveSuccess) return rc;

  unsigned int tid = mscclppDeviceAllGatherThreadId();
  unsigned int threadCount = mscclppDeviceAllGatherThreadCount();
  size_t shardBase = static_cast<size_t>(h.rank) * recvCount;
  char* slotBase = h.slab + static_cast<size_t>(slot) * h.slotStride;
  for (size_t i = tid; i < recvCount; i += threadCount) {
    size_t sourceIndex = shardBase + i;
    T value = reinterpret_cast<const T*>(slotBase)[sourceIndex];
    for (int r = 1; r < h.nranks; ++r) {
      const T* peer = reinterpret_cast<const T*>(
          slotBase + static_cast<size_t>(r) * h.maxBytesPerRank);
      value = liteApplyReduction(value, peer[sourceIndex], op);
    }
    dst[i] = value;
  }
  __syncthreads();
  liteCollectiveDoneBlock(h, slot, epoch);
  return mscclppDeviceCollectiveSuccess;
}

#endif  // defined(__CUDACC__)
