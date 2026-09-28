#pragma once

// rank rows are packed at the current message stride, not allocation capacity.
static __device__ __forceinline__ int liteAllGatherHostMappedBlock(
    mscclppDeviceCollectiveHandle_t const& h, char const* src, char* dst,
    size_t bytesPerRank) {
  __shared__ unsigned long long epoch;
  __shared__ int slot, status, unusedCount;
  __shared__ size_t unusedBytes;
  int rc = liteCollectiveBeginBlock(h, src, bytesPerRank, &epoch, &slot,
                                    &unusedBytes, &unusedCount);
  if (rc) return rc;
  char* slab = h.slab + static_cast<size_t>(slot) * h.slotStride;
  liteCollectiveCopyBlock(slab + h.rank * bytesPerRank, src, bytesPerRank);
  // Every producer thread publishes its own writes before the leader's flag.
  __threadfence_system();
  __syncthreads();
  if (mscclppDeviceCollectiveThreadId() == 0) {
    status = 0;
    liteStoreRelease(mscclppDeviceCollectiveReady(h, slot, 0, h.rank), epoch);
    for (int r = 0; r < h.nranks; ++r) {
      if (!liteCollectiveWait(h, mscclppDeviceCollectiveReady(h, slot, 0, r),
                              epoch)) {
        status = mscclppDeviceCollectiveTransportError;
        break;
      }
    }
  }
  __syncthreads();
  if (status) return status;
  __threadfence_system();
  // Include our own row, matching the reference's contiguous full-output copy.
  liteCollectiveCopyBlock(dst, slab, bytesPerRank * h.nranks);
  liteCollectiveDoneBlock(h, slot, epoch);
  return mscclppDeviceCollectiveSuccess;
}
