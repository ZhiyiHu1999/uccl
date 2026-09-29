#pragma once

// Primitives of the device-composed (single-node) ReduceScatter paths, split by
// who executes them:
//
//  FIFO primitives, executed by the CPU service (need DMA, CPU or the control
//  words in host memory). Every ticket must be awaited before the path returns.
//    liteRsCopyAsync / liteRsAwait   1D or 2D DMA on a service stream
//    liteRsHostSum                   CPU float sum of host rows
//    liteRsBarrier                   node-local barrier on a control row
//
//  CTA primitives, executed by the calling CTA's threads.
//    liteRsSum                       elementwise float sum of up to four rows
//
// All functions are collective over the CTA (every thread calls them with the
// same arguments) and return the same value on every thread.

static __device__ __forceinline__ unsigned long long liteRsSubmit(
    mscclppDeviceCollectiveHandle_t const& h, LiteTask const& task) {
  __shared__ unsigned long long ticket;
  // Publish this CTA's stores before the copy engine / CPU reads them.
  __threadfence_system();
  __syncthreads();
  if (mscclppDeviceCollectiveThreadId() == 0) ticket = litePostTask(h, task);
  __syncthreads();
  return ticket;
}

static __device__ __forceinline__ int liteRsAwait(
    mscclppDeviceCollectiveHandle_t const& h, unsigned long long ticket) {
  __shared__ int status;
  __syncthreads();
  if (mscclppDeviceCollectiveThreadId() == 0)
    status = liteWaitTask(h, ticket) ? mscclppDeviceCollectiveSuccess
                                     : mscclppDeviceCollectiveTransportError;
  __syncthreads();
  return status;
}

// Returns the ticket (0 when the FIFO refused the task). 2D when rows > 1:
// `bytes` is the row width.
static __device__ __forceinline__ unsigned long long liteRsCopyAsync(
    mscclppDeviceCollectiveHandle_t const& h, void* dst, void const* src,
    size_t bytes, int stream, size_t rows = 1, size_t dstPitch = 0,
    size_t srcPitch = 0) {
  LiteTask task{};
  task.kind = LiteTaskKind::RsCopy;
  task.destination = reinterpret_cast<uint64_t>(dst);
  task.source = reinterpret_cast<uint64_t>(src);
  task.bytes = bytes;
  task.rs.stream = stream;
  task.rs.rows = rows;
  task.rs.dstPitch = dstPitch;
  task.rs.srcPitch = srcPitch;
  return liteRsSubmit(h, task);
}

// CPU sum of `n` (<= 4) host rows into `dst` (all CPU addresses).
static __device__ __forceinline__ int liteRsHostSum(
    mscclppDeviceCollectiveHandle_t const& h, char* dst,
    char const* const* rows, int n, size_t bytes) {
  LiteTask task{};
  task.kind = LiteTaskKind::RsHostSum;
  task.destination = reinterpret_cast<uint64_t>(dst);
  task.bytes = bytes;
  task.rs.count = n;
  for (int r = 0; r < n; ++r)
    task.rs.sources[r] = reinterpret_cast<uint64_t>(rows[r]);
  return liteRsAwait(h, liteRsSubmit(h, task));
}

// Node-local barrier: publishes `epoch` in control row (slot 0, kind) and waits
// for every local rank. Post it only after the awaited DMA of this step.
static __device__ __forceinline__ int liteRsBarrier(
    mscclppDeviceCollectiveHandle_t const& h, int kind,
    unsigned long long epoch) {
  LiteTask task{};
  task.kind = LiteTaskKind::RsBarrier;
  task.slot = 0;
  task.epoch = epoch;
  task.rs.barrierKind = kind;
  return liteRsAwait(h, liteRsSubmit(h, task));
}

// CTA arithmetic. Inputs may live in peer GPU scratch or mapped host memory and
// may be rewritten by other agents between calls, so loads are volatile. The
// output may alias one input (in-place ring accumulation).
static __device__ __forceinline__ void liteRsSum(char* dst,
                                                 char const* const* rows,
                                                 int n, size_t bytes) {
  unsigned tid = mscclppDeviceCollectiveThreadId();
  unsigned threads = mscclppDeviceCollectiveThreadCount();
  size_t count = bytes / sizeof(float);
  auto* out = reinterpret_cast<float*>(dst);
  for (size_t i = tid; i < count; i += threads) {
    float value = reinterpret_cast<float const volatile*>(rows[0])[i];
    for (int r = 1; r < n; ++r)
      value += reinterpret_cast<float const volatile*>(rows[r])[i];
    out[i] = value;
  }
  // Make the result visible to peers / the CPU before any barrier is posted.
  __threadfence_system();
  __syncthreads();
}

// Persistent barrier epoch of the device-composed paths.
static __device__ __forceinline__ unsigned long long liteRsEpochLoad(
    mscclppDeviceCollectiveHandle_t const& h) {
  __shared__ unsigned long long epoch;
  __syncthreads();
  if (mscclppDeviceCollectiveThreadId() == 0)
    epoch = liteLoadAcquire(reinterpret_cast<unsigned long long*>(
        &h.tasks->reduceScatter.epoch));
  __syncthreads();
  return epoch;
}

static __device__ __forceinline__ void liteRsEpochStore(
    mscclppDeviceCollectiveHandle_t const& h, unsigned long long epoch) {
  if (mscclppDeviceCollectiveThreadId() == 0)
    liteStoreRelease(reinterpret_cast<unsigned long long*>(
                         &h.tasks->reduceScatter.epoch),
                     epoch);
  __syncthreads();
}
