#pragma once

// Transport and slot preparation share executeOrderedSmallSchedule with the
// CPU path. Its SM kernel phases execute here inside the caller's CTA; the
// service never launches a child kernel or waits for the caller stream.
static __device__ __forceinline__ int liteAllGatherOrderedSmallBlock(
    mscclppDeviceCollectiveHandle_t const& h, char const* src, char* dst,
    size_t bytesPerRank, LiteAllGatherPlan const&) {
  unsigned long long ticket = liteNetworkAllGatherPost(
      h, src, dst, bytesPerRank, LiteDeviceAllGatherPath::OrderedSmall);
  if (!ticket) return mscclppDeviceCollectiveTransportError;
  __shared__ int status;
  unsigned tid = mscclppDeviceCollectiveThreadId();
  auto& call = h.tasks->network;
  if (tid == 0)
    status =
        liteNetworkAllGatherWait(
            h, reinterpret_cast<unsigned long long*>(&call.prepared), ticket)
            ? 0
            : mscclppDeviceCollectiveTransportError;
  __syncthreads();
  if (status) return status;
  size_t total = bytesPerRank * h.nranks;
  bool compact = call.oneRankRegister && total != 128 && total != 256;
  bool tiny = call.oneRankRegister && total == 128;
  unsigned stageThreads =
      call.oneRankRegister
          ? (tiny ? 1
                  : ((total == 16 * 1024 || total == 32 * 1024) ? 256 : 128))
          : (total <= 256 ? 1 : 128);
  size_t row = compact ? call.segmentBytes : bytesPerRank;
  if (call.stageWithSm) {
    liteCollectiveCopyBlock(call.slab + call.slotOffset + h.rank * row, src,
                            bytesPerRank, stageThreads);
    __threadfence_system();
    __syncthreads();
    if (tid == 0)
      liteStoreRelease(reinterpret_cast<unsigned long long*>(call.control +
                                                             call.readyOffset),
                       call.epoch);
  }
  char* self = dst + h.rank * bytesPerRank;
  // CPU parallel/compact kernels copy self before waiting for the remote flag.
  if (call.oneRankRegister && !tiny && src != self)
    liteCollectiveCopyBlock(self, src, bytesPerRank, stageThreads);
  if (call.receiveWithSm) {
    size_t flagOffset =
        compact ? call.slotOffset + (1 - h.rank) * row + call.flagInSegment
                : call.flagOffset;
    if (tid == 0) {
      // A size change moves this flag into memory previously used as payload.
      // Only the exact epoch proves readiness (as in the native SM kernels);
      // arbitrary old payload bytes may decode to a value greater than epoch.
      if (!liteNetworkAllGatherWait<true>(
              h, reinterpret_cast<unsigned long long*>(call.slab + flagOffset),
              call.epoch))
        status = mscclppDeviceCollectiveTransportError;
      if (!status && !call.oneRankRegister) {
        for (int local = 0; local < call.localGroupSize; ++local)
          if (!liteNetworkAllGatherWait(
                  h,
                  reinterpret_cast<unsigned long long*>(
                      call.control + call.readyBase + local * call.readyStride),
                  call.epoch)) {
            status = mscclppDeviceCollectiveTransportError;
            break;
          }
      }
    }
    __syncthreads();
    if (status) return status;
    __threadfence_system();
    if (call.oneRankRegister) {
      int remote = 1 - h.rank;
      liteCollectiveCopyBlock(dst + remote * bytesPerRank,
                              call.slab + call.slotOffset + remote * row,
                              bytesPerRank, stageThreads);
      if (tiny && src != self)
        liteCollectiveCopyBlock(self, src, bytesPerRank, 1);
    } else {
      // multiRankHostRecvKernel: 128 receiving threads, including self row.
      // Tiny packing's one-thread limit does not apply to reception.
      liteCollectiveCopyBlock(dst, call.slab + call.slotOffset, total, 128);
    }
  }
  __threadfence_system();
  __syncthreads();
  if (tid == 0)
    liteStoreRelease(reinterpret_cast<unsigned long long*>(&call.deviceDone),
                     ticket);
  return liteNetworkAllGatherFinish(h, ticket);
}
