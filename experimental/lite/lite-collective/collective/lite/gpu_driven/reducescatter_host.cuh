#pragma once

// Single-node ReduceScatter paths without CUDA IPC: data crosses the node's
// shared pinned host slab. Each path is a __device__ function composing FIFO
// primitives (D2H/H2D DMA, CPU sum, barrier) and CTA primitives (sum).

// HostSmall / HostRead / HostBulk core.
//   HostSmall: D2H of the whole input (2D), CPU sum of the target rows, H2D of
//              the result. CPU: runNoCudaIpcHostReduceScatter (single node).
//   HostRead:  D2H of peer-owned rows only; the CTA reads the peer rows straight
//              from mapped host memory. CPU: ...HostReadSingleNode...
//   HostBulk:  as HostRead, but the peer rows are first copied back into GPU
//              scratch. CPU: ...BulkSingleNode...
static __device__ __forceinline__ int liteRsHostRowsBlock(
    mscclppDeviceCollectiveHandle_t const& h, float const* srcF, float* dstF,
    size_t B, LiteReduceScatterPlan const& plan) {
  auto const& v = h.reduceScatterView;
  int local = v.layout.local, me = v.layout.me;
  auto const* src = reinterpret_cast<char const*>(srcF);
  auto* dst = reinterpret_cast<char*>(dstF);
  bool small = plan.path == LiteReduceScatterPath::HostSmall;
  bool read = plan.path == LiteReduceScatterPath::HostRead;
  unsigned long long e = liteRsEpochLoad(h);
  for (size_t off = 0; off < B;) {
    size_t bytes = plan.chunkBytes < B - off ? plan.chunkBytes : B - off;
    ++e;
    unsigned long long tickets[4] = {0, 0, 0, 0};
    if (small) {
      tickets[0] = liteRsCopyAsync(h, v.hostRow(0, me), src + off, bytes, 0,
                                   v.layout.ranks, v.layout.chunkCapacity, B);
      if (!tickets[0]) return mscclppDeviceCollectiveTransportError;
    } else {
      for (int target = 0; target < local; ++target) {
        if (target == me) continue;
        tickets[target] = liteRsCopyAsync(h, v.hostRow(0, me, target),
                                          src + target * B + off, bytes, 0);
        if (!tickets[target]) return mscclppDeviceCollectiveTransportError;
      }
    }
    for (int t = 0; t < 4; ++t)
      if (tickets[t]) {
        int rc = liteRsAwait(h, tickets[t]);
        if (rc) return rc;
      }
    int rc = liteRsBarrier(h, 0, e);
    if (rc) return rc;
    if (small) {
      char const* hostRows[4] = {};
      for (int r = 0; r < local; ++r) hostRows[r] = v.hostRow(0, r, me);
      char* result = v.hostRow(0, me, v.layout.ranks);
      rc = liteRsHostSum(h, result, hostRows, local, bytes);
      if (rc) return rc;
      unsigned long long t = liteRsCopyAsync(h, dst + off, result, bytes, 1);
      if (!t) return mscclppDeviceCollectiveTransportError;
      rc = liteRsAwait(h, t);
      if (rc) return rc;
    } else {
      char const* rows[4] = {};
      unsigned long long back[4] = {0, 0, 0, 0};
      for (int r = 0; r < local; ++r) {
        if (r == me) {
          rows[r] = src + me * B + off;
        } else if (read) {
          rows[r] = v.mappedRow(0, r, me);
        } else {
          rows[r] = v.deviceRow(0, me, r);
          back[r] = liteRsCopyAsync(h, v.deviceRow(0, me, r), v.hostRow(0, r, me),
                                    bytes, 1);
          if (!back[r]) return mscclppDeviceCollectiveTransportError;
        }
      }
      for (int r = 0; r < local; ++r)
        if (back[r]) {
          rc = liteRsAwait(h, back[r]);
          if (rc) return rc;
        }
      liteRsSum(dst + off, rows, local, bytes);
    }
    rc = liteRsBarrier(h, 2, e);
    if (rc) return rc;
    off += bytes;
  }
  liteRsEpochStore(h, e);
  return mscclppDeviceCollectiveSuccess;
}

// HostRing core (CPU direct ring): every rank owns n-1 mapped-host rows per
// chunk and the next rank pulls the previous rank's row directly, so a step is
// one CTA pass and needs no push copy.
//   row[0]   = shard(me-1)
//   row[k]   = shard(me-1-k) + prev.row[k-1]      (1 <= k <= n-2)
//   output   = shard(me)     + prev.row[n-2]
// The hand-over uses neighbour flags in the mapped control page, so the whole
// path runs on the device without a FIFO task (like the CPU's stream flags):
//   flag[k][me]  = chunk epoch once row k is complete (prev waits for it);
//   flag[3][me]  = chunk epoch once this rank has read prev's rows (prev may then
//                  rewrite them for the next chunk; waited before row 0).
static __device__ __forceinline__ int liteRsHostRingBlock(
    mscclppDeviceCollectiveHandle_t const& h, float const* srcF, float* dstF,
    size_t B, LiteReduceScatterPlan const& plan) {
  auto const& v = h.reduceScatterView;
  int n = v.layout.local, me = v.layout.me, prev = (me + n - 1) % n,
      next = (me + 1) % n;
  auto const* src = reinterpret_cast<char const*>(srcF);
  auto* dst = reinterpret_cast<char*>(dstF);
  unsigned long long e = liteRsEpochLoad(h);
  for (size_t off = 0; off < B;) {
    size_t bytes = plan.chunkBytes < B - off ? plan.chunkBytes : B - off;
    ++e;
    auto shard = [&](int k) {
      return src + static_cast<size_t>(((me - k) % n + n) % n) * B + off;
    };
    // The reader of my rows (next) finished the previous chunk.
    int rc = liteRsWaitFlag(h, v.flag(0, 3, next), e - 1);
    if (rc) return rc;
    char const* first[1] = {shard(1)};
    liteRsSum(v.ringRow(0, me, 0, true), first, 1, bytes);
    liteRsSignalFlag(v.flag(0, 0, me), e);
    for (int k = 1; k <= n - 2; ++k) {
      rc = liteRsWaitFlag(h, v.flag(0, k - 1, prev), e);
      if (rc) return rc;
      char const* rows[2] = {shard(k + 1), v.ringRow(0, prev, k - 1, true)};
      liteRsSum(v.ringRow(0, me, k, true), rows, 2, bytes);
      liteRsSignalFlag(v.flag(0, k, me), e);
    }
    rc = liteRsWaitFlag(h, v.flag(0, n - 2, prev), e);
    if (rc) return rc;
    char const* last[2] = {shard(n), v.ringRow(0, prev, n - 2, true)};
    liteRsSum(dst + off, last, 2, bytes);
    liteRsSignalFlag(v.flag(0, 3, me), e);
    off += bytes;
  }
  liteRsEpochStore(h, e);
  return mscclppDeviceCollectiveSuccess;
}

// HostSmall (no IPC, B <= 64 KiB): CPU runNoCudaIpcHostReduceScatter.
static __device__ __forceinline__ int liteReduceScatterHostSmallBlock(
    mscclppDeviceCollectiveHandle_t const& h, float const* src, float* dst,
    size_t bytes, LiteReduceScatterPlan const& plan) {
  return liteRsHostRowsBlock(h, src, dst, bytes, plan);
}

// HostRing (no IPC, mapped): CPU runNoCudaIpcDirectRingSingleNodeReduceScatter.
static __device__ __forceinline__ int liteReduceScatterHostRingBlock(
    mscclppDeviceCollectiveHandle_t const& h, float const* src, float* dst,
    size_t bytes, LiteReduceScatterPlan const& plan) {
  if (!h.reduceScatterMapped || !h.reduceScatterView.hostSendDevice)
    return mscclppDeviceCollectiveInvalidUsage;
  return liteRsHostRingBlock(h, src, dst, bytes, plan);
}

// HostRead (no IPC, mapped): CPU runNoCudaIpcHostReadSingleNodeReduceScatter.
static __device__ __forceinline__ int liteReduceScatterHostReadBlock(
    mscclppDeviceCollectiveHandle_t const& h, float const* src, float* dst,
    size_t bytes, LiteReduceScatterPlan const& plan) {
  if (!h.reduceScatterMapped || !h.reduceScatterView.hostSendDevice)
    return mscclppDeviceCollectiveInvalidUsage;
  return liteRsHostRowsBlock(h, src, dst, bytes, plan);
}

// HostBulk (no IPC): CPU runNoCudaIpcBulkSingleNodeReduceScatter.
static __device__ __forceinline__ int liteReduceScatterHostBulkBlock(
    mscclppDeviceCollectiveHandle_t const& h, float const* src, float* dst,
    size_t bytes, LiteReduceScatterPlan const& plan) {
  return liteRsHostRowsBlock(h, src, dst, bytes, plan);
}
