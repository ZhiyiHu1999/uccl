#pragma once

// CPU reference: runIntraNodeShmAllGather DMA pipeline. One descriptor owns
// all chunks of a whole-message slot; the service enqueues independent D2H and
// left/right H2D streams with per-chunk ready dependencies.
static __device__ __forceinline__ int liteAllGatherHostDmaBlock(
    mscclppDeviceCollectiveHandle_t const& h, char const* src, char* dst,
    size_t bytesPerRank, LiteAllGatherPlan const& plan) {
  __shared__ unsigned long long epoch, ticket;
  __shared__ int slot, status, unusedCount;
  __shared__ size_t unusedBytes;
  size_t chunks =
      bytesPerRank / plan.chunkBytes + (bytesPerRank % plan.chunkBytes != 0);
  if (chunks > MSCCLPP_DEVICE_COLLECTIVE_MAX_CHUNKS)
    return mscclppDeviceCollectiveInvalidUsage;
  int rc = liteCollectiveBeginBlock(h, src, bytesPerRank, &epoch, &slot,
                                    &unusedBytes, &unusedCount);
  if (rc) return rc;
  char* self = dst + h.rank * bytesPerRank;
  bool selfWithSm =
      h.allGatherPolicy.selfCopyWithSm && !(bytesPerRank & 7) &&
      !((reinterpret_cast<uintptr_t>(src) | reinterpret_cast<uintptr_t>(dst)) &
        7);
  // The service streams must never depend on the calling kernel completing.
  __threadfence_system();
  __syncthreads();
  if (mscclppDeviceCollectiveThreadId() == 0) {
    LiteTask task{};
    task.kind = LiteTaskKind::HostAllGather;
    task.source = reinterpret_cast<uint64_t>(src);
    task.destination = reinterpret_cast<uint64_t>(dst);
    task.bytes = bytesPerRank;
    task.chunkBytes = plan.chunkBytes;
    task.selfCopyWithSm = selfWithSm;
    task.slot = slot;
    task.epoch = epoch;
    ticket = litePostTask(h, task);
  }
  __syncthreads();
  if (!ticket) return mscclppDeviceCollectiveTransportError;
  if (selfWithSm && src != self)
    liteCollectiveCopyBlock(self, src, bytesPerRank);
  if (mscclppDeviceCollectiveThreadId() == 0)
    status =
        liteWaitTask(h, ticket) ? 0 : mscclppDeviceCollectiveTransportError;
  __syncthreads();
  if (status) return status;
  // Publish once, only after D2H, both H2D directions and self-copy complete.
  liteCollectiveDoneBlock(h, slot, epoch);
  return mscclppDeviceCollectiveSuccess;
}
