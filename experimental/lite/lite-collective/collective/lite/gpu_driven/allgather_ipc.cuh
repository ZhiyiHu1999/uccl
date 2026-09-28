#pragma once

// Included after the shared device primitives in gpu_collectives.cuh.
// Output regions are registered collectively before the user kernel starts.
// One epoch per invocation, one full rank block per ring step. No SM payload
// copying and no intermediate IPC scratch on this path.
static __device__ __forceinline__ int liteAllGatherIpcRingBlock(
    mscclppDeviceCollectiveHandle_t const& h, char const* src, char* dst,
    size_t bytes) {
  __shared__ unsigned long long epoch;
  __shared__ int slot, status, unusedCount;
  __shared__ size_t unusedBytes;
  int rc = liteCollectiveBeginBlock(h, src, bytes, &epoch, &slot,
                                   &unusedBytes, &unusedCount);
  if (rc) return rc;
  unsigned tid = mscclppDeviceCollectiveThreadId();
  // Publish every producer thread's stores before the copy engine reads input.
  __threadfence_system();
  __syncthreads();
  if (tid == 0) {
    status = 0;
    liteStoreRelease(mscclppDeviceCollectiveReady(h, slot, 0, h.rank), epoch);
    for (int r = 0; r < h.nranks; ++r)
      if (!liteCollectiveWait(h, mscclppDeviceCollectiveReady(h, slot, 0, r), epoch))
        status = mscclppDeviceCollectiveTransportError;
    char* self = dst + static_cast<size_t>(h.rank) * bytes;
    if (!status && self != src) {
      LiteTask task{};
      task.kind = LiteTaskKind::IpcCopySelf;
      task.source = reinterpret_cast<uint64_t>(src);
      task.destination = reinterpret_cast<uint64_t>(self);
      task.bytes = bytes;
      task.slot = slot;
      task.epoch = epoch;
      if (!liteWaitTask(h, litePostTask(h, task)))
        status = mscclppDeviceCollectiveTransportError;
    }
  }
  __syncthreads();
  if (status) return status;
  int previous = (h.rank + h.nranks - 1) % h.nranks;
  for (int step = 0; step < h.nranks - 1; ++step) {
    if (tid == 0) {
      int block = (h.rank - step + h.nranks) % h.nranks;
      char* localRow = dst + static_cast<size_t>(block) * bytes;
      LiteTask task{};
      task.kind = LiteTaskKind::IpcPush;
      task.source = reinterpret_cast<uint64_t>(step == 0 ? src : localRow);
      // Service translates this registered local output offset to next's mapping.
      task.destination = reinterpret_cast<uint64_t>(localRow);
      task.bytes = bytes;
      task.slot = slot;
      task.epoch = epoch;
      if (!liteWaitTask(h, litePostTask(h, task))) {
        status = mscclppDeviceCollectiveTransportError;
      } else {
        // Publish only after the copy event completed, not after enqueue.
        liteStoreRelease(mscclppDeviceCollectiveReady(h, slot, step + 1, h.rank), epoch);
        if (!liteCollectiveWait(h,
                mscclppDeviceCollectiveReady(h, slot, step + 1, previous), epoch))
          status = mscclppDeviceCollectiveTransportError;
      }
    }
    __syncthreads();
    if (status) return status;
  }
  __threadfence_system();
  liteCollectiveDoneBlock(h, slot, epoch);
  // No rank may return and overwrite its output while a neighbor forwards it.
  if (tid == 0)
    for (int r = 0; r < h.nranks; ++r)
      if (!liteCollectiveWait(h, mscclppDeviceCollectiveDone(h, slot, r), epoch))
        status = mscclppDeviceCollectiveTransportError;
  __syncthreads();
  return status;
}
