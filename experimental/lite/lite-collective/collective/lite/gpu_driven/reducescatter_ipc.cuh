#pragma once

// Single-node CUDA-IPC ReduceScatter paths (1n*2g, 1n*4g). Each path is a
// __device__ function that sequences FIFO primitives (DMA, barrier) and CTA
// primitives (sum) from reducescatter_primitives.cuh. Peer GPU scratch is
// addressed through the handle's LiteRsDeviceView; no CPU schedule is involved.

// LocalRows / TwoLocal core. Every rank scatters shard t into row "me" of
// target t's scratch, all ranks barrier, then each rank reduces its own rows.
// CPU: runLocalFourRankReduceScatter.
//   * B <= 32 KiB: the self row is read straight from the input (skip-self);
//   * device flags (mapped, B <= LOCAL_DEVICE_FLAG_MAX_BYTES): the CTA scatters
//     into the peer rows itself instead of submitting DMA;
//   * B >= 64 KiB: one DMA stream per target (LOCAL_PARALLEL_COPY).
static __device__ __forceinline__ int liteRsIpcRowsBlock(
    mscclppDeviceCollectiveHandle_t const& h, float const* srcF, float* dstF,
    size_t B, LiteReduceScatterPlan const& plan) {
  auto const& v = h.reduceScatterView;
  int local = v.layout.local, me = v.layout.me;
  auto const* src = reinterpret_cast<char const*>(srcF);
  auto* dst = reinterpret_cast<char*>(dstF);
  bool skipSelf = !plan.deviceFlags &&
                  (B <= 32 * 1024 ||
                   plan.path == LiteReduceScatterPath::TwoLocal);
  bool parallel = plan.path == LiteReduceScatterPath::LocalRows &&
                  !plan.deviceFlags && !skipSelf &&
                  h.reduceScatterPolicy.localParallel && B >= 64 * 1024;
  unsigned long long epoch = liteRsEpochLoad(h) + 1, seq = 0;  // one epoch per call
  for (size_t off = 0; off < B;) {
    size_t bytes = plan.chunkBytes < B - off ? plan.chunkBytes : B - off;
    unsigned long long e = liteRsStamp(epoch, ++seq);  // one stamp per chunk
    unsigned long long tickets[4] = {0, 0, 0, 0};
    for (int target = 0; target < local; ++target) {
      if (target == me && skipSelf) continue;  // read from the input instead
      char* row = v.deviceRow(0, target, me);
      char const* from = src + target * B + off;
      if (plan.deviceFlags) {
        char const* one[1] = {from};
        liteRsSum(row, one, 1, bytes);
      } else {
        tickets[target] =
            liteRsCopyAsync(h, row, from, bytes, parallel ? target : 0);
        if (!tickets[target]) return mscclppDeviceCollectiveTransportError;
      }
    }
    for (int target = 0; target < local; ++target)
      if (tickets[target]) {
        int rc = liteRsAwait(h, tickets[target]);
        if (rc) return rc;
      }
    int rc = liteRsBarrier(h, 0, e);
    if (rc) return rc;
    char const* rows[4] = {};
    for (int r = 0; r < local; ++r)
      rows[r] = r == me && skipSelf ? src + me * B + off : v.deviceRow(0, me, r);
    liteRsSum(dst + off, rows, local, bytes);
    // Every consumer finished before any scratch row is rewritten.
    rc = liteRsBarrier(h, 2, e);
    if (rc) return rc;
    off += bytes;
  }
  liteRsEpochStore(h, epoch);
  return mscclppDeviceCollectiveSuccess;
}

// P2pRing / IpcRing core: push ring over IPC DMA with two alternating mailbox
// rows. Step s sends shard (me-s-1) (later steps forward the partial that was
// just accumulated) to the next rank, then adds the local shard (me-s-2) to the
// received partial; the last step writes the output.
// CPU: runP2pRingReduceScatter / runLocalFourRankRingReduceScatter.
static __device__ __forceinline__ int liteRsPushRingBlock(
    mscclppDeviceCollectiveHandle_t const& h, float const* srcF, float* dstF,
    size_t B, LiteReduceScatterPlan const& plan) {
  auto const& v = h.reduceScatterView;
  int n = v.layout.local, me = v.layout.me, next = (me + 1) % n;
  auto const* src = reinterpret_cast<char const*>(srcF);
  auto* dst = reinterpret_cast<char*>(dstF);
  unsigned long long epoch = liteRsEpochLoad(h) + 1, seq = 0;  // one epoch per call
  for (size_t off = 0; off < B;) {
    size_t bytes = plan.chunkBytes < B - off ? plan.chunkBytes : B - off;
    for (int step = 0; step < n - 1; ++step) {
      unsigned long long e = liteRsStamp(epoch, ++seq);  // one stamp per step
      int sendShard = (me - step - 1 + n) % n;
      int recvShard = (me - step - 2 + n) % n;
      char const* outgoing = step == 0 ? src + sendShard * B + off
                                       : v.ringRow(0, me, (step - 1) & 1, false);
      char* incoming = v.ringRow(0, me, step & 1, false);
      char* target = v.ringRow(0, next, step & 1, false);
      unsigned long long ticket = liteRsCopyAsync(h, target, outgoing, bytes, 0);
      if (!ticket) return mscclppDeviceCollectiveTransportError;
      int rc = liteRsAwait(h, ticket);
      if (rc) return rc;
      rc = liteRsBarrier(h, 0, e);  // the previous rank's partial has arrived
      if (rc) return rc;
      char const* rows[2] = {src + recvShard * B + off, incoming};
      liteRsSum(step == n - 2 ? dst + off : incoming, rows, 2, bytes);
      // The next rank consumed our push before either mailbox is rewritten.
      rc = liteRsBarrier(h, 1, e);
      if (rc) return rc;
    }
    off += bytes;
  }
  liteRsEpochStore(h, epoch);
  return mscclppDeviceCollectiveSuccess;
}

static __device__ __forceinline__ int liteRsRequireIpc(
    mscclppDeviceCollectiveHandle_t const& h) {
  return h.reduceScatterIpc ? mscclppDeviceCollectiveSuccess
                            : mscclppDeviceCollectiveInvalidUsage;
}

// LocalRows (1n*4g, CUDA IPC): CPU runLocalFourRankReduceScatter.
static __device__ __forceinline__ int liteReduceScatterLocalRowsBlock(
    mscclppDeviceCollectiveHandle_t const& h, float const* src, float* dst,
    size_t bytes, LiteReduceScatterPlan const& plan) {
  if (liteRsRequireIpc(h)) return mscclppDeviceCollectiveInvalidUsage;
  return liteRsIpcRowsBlock(h, src, dst, bytes, plan);
}

// TwoLocal (1n*2g, CUDA IPC): LocalRows with two ranks; the self shard is read
// from the input. No CPU reference (extension).
static __device__ __forceinline__ int liteReduceScatterTwoLocalBlock(
    mscclppDeviceCollectiveHandle_t const& h, float const* src, float* dst,
    size_t bytes, LiteReduceScatterPlan const& plan) {
  if (liteRsRequireIpc(h)) return mscclppDeviceCollectiveInvalidUsage;
  return liteRsIpcRowsBlock(h, src, dst, bytes, plan);
}

// P2pRing (1n*4g, CUDA IPC, 256 KiB..2 MiB): CPU runP2pRingReduceScatter, with
// NCCL send/recv replaced by service DMA into the next rank's mailbox.
static __device__ __forceinline__ int liteReduceScatterP2pRingBlock(
    mscclppDeviceCollectiveHandle_t const& h, float const* src, float* dst,
    size_t bytes, LiteReduceScatterPlan const& plan) {
  if (liteRsRequireIpc(h)) return mscclppDeviceCollectiveInvalidUsage;
  return liteRsPushRingBlock(h, src, dst, bytes, plan);
}

// IpcRing (1n*4g, CUDA IPC, large B): CPU runLocalFourRankRingReduceScatter.
static __device__ __forceinline__ int liteReduceScatterIpcRingBlock(
    mscclppDeviceCollectiveHandle_t const& h, float const* src, float* dst,
    size_t bytes, LiteReduceScatterPlan const& plan) {
  if (liteRsRequireIpc(h)) return mscclppDeviceCollectiveInvalidUsage;
  return liteRsPushRingBlock(h, src, dst, bytes, plan);
}
