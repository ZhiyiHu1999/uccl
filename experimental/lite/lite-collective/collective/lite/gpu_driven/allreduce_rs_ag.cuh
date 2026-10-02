#pragma once

// ReduceScatterAllGather: equal-shard composition of the ReduceScatter and
// AllGather collectives, S = count / nranks elements per shard. CPU reference:
// the RS+AG tail of runSendRecvAllReduce.
//
//   1. ReduceScatter(src -> dst + rank*S): the shard this rank owns, stored at its
//      global-rank output offset (B_RS = S, T_RS = B; RS path chosen by its plan).
//   2. AllGather(dst + rank*S -> dst): restores the complete tensor in global-rank
//      order (B_AG = S, T_AG = B; AG path chosen by its plan).
//
// Phase and epoch identities stay separate because each collective runs through
// its own entry. In-place safety: no ReduceScatter path reads another rank's
// original input (peers read staged copies or partials), so a rank whose RS has
// finished may overwrite its output regions; the IpcRing AllGather additionally
// starts with an all-rank ready barrier. The IPC AllGather pushes into the next
// rank's output, so with the IPC backend the output must be registered exactly as
// for AllGather.
static __device__ __forceinline__ int liteAllReduceRsAgBlock(
    mscclppDeviceCollectiveHandle_t const& h, float const* src, float* dst,
    size_t count) {
  size_t shard = count / static_cast<size_t>(h.nranks);
  float* own = dst + static_cast<size_t>(h.rank) * shard;
  int rc = liteReduceScatterFloatSumBlock(h, src, own, shard);
  if (rc) return rc;
  return liteAllGatherBlock(h, own, dst, shard * sizeof(float));
}
