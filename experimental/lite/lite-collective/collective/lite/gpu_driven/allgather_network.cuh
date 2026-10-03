#pragma once

// Invocation sequence belongs to the FIFO; algorithm epochs/slots belong to
// the shared CPU-reference schedule. Do not call BeginBlock for these paths.
static __device__ __forceinline__ unsigned long long liteNetworkAllGatherPost(
    mscclppDeviceCollectiveHandle_t const& h, char const* src, char* dst,
    size_t bytes, LiteDeviceAllGatherPath path) {
  __shared__ unsigned long long ticket;
  __threadfence_system();
  __syncthreads();
  if (mscclppDeviceCollectiveThreadId() == 0) {
    LiteTask task{};
    task.kind = LiteTaskKind::NetworkAllGather;
    task.networkPath = path;
    task.source = reinterpret_cast<uint64_t>(src);
    task.destination = reinterpret_cast<uint64_t>(dst);
    task.bytes = bytes;
    ticket = h.networkAllGather ? litePostTask(h, task) : 0;
  }
  __syncthreads();
  return ticket;
}

template <bool ExactEpoch = false>
static __device__ __forceinline__ bool liteNetworkAllGatherWait(
    mscclppDeviceCollectiveHandle_t const& h,
    unsigned long long const volatile* flag, unsigned long long value) {
  if (liteCollectiveWait<ExactEpoch>(h, flag, value)) return true;
  liteStoreRelease(
      reinterpret_cast<unsigned long long*>(&h.tasks->network.abort), 1);
  return false;
}

static __device__ __forceinline__ int liteNetworkAllGatherFinish(
    mscclppDeviceCollectiveHandle_t const& h, unsigned long long ticket) {
  __shared__ int status;
  if (mscclppDeviceCollectiveThreadId() == 0) {
    status = ticket && liteWaitTask(h, ticket)
                 ? mscclppDeviceCollectiveSuccess
                 : mscclppDeviceCollectiveTransportError;
    if (status && h.tasks)
      liteStoreRelease(
          reinterpret_cast<unsigned long long*>(&h.tasks->network.abort), 1);
  }
  __syncthreads();
  __threadfence_system();
  return status;
}
