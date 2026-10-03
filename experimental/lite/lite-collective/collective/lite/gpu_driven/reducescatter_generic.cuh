#pragma once
// Block-scoped ReduceScatter.  Each rank contributes nranks * recvCount
// elements.  Rank r receives the reduction of shard r, containing recvCount
// elements.  T must support +, <, and >.
template <typename T>
static __device__ __forceinline__ int liteReduceScatterGenericBlock(
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

