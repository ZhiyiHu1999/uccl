#pragma once

// All participating CTAs distribute both copies across the grid
// and synchronize before ready and done publication.
template <class Group>
static __device__ __forceinline__ void liteAllGatherGroupCopy(Group group,
                                                              char* dst,
                                                              char const* src,
                                                              size_t bytes) {
  auto* output = reinterpret_cast<unsigned long long*>(dst);
  auto const* input = reinterpret_cast<unsigned long long const*>(src);
  size_t words = bytes / sizeof(unsigned long long);
  for (size_t i = group.thread_rank(); i < words; i += group.size())
    output[i] = input[i];
  for (size_t i = words * sizeof(unsigned long long) + group.thread_rank();
       i < bytes; i += group.size())
    dst[i] = src[i];
}

template <class Group>
static __device__ __forceinline__ int liteAllGatherHostCooperativeGroup(
    mscclppDeviceCollectiveHandle_t const& h, char const* src, char* dst,
    size_t bytesPerRank, Group group, LiteAllGatherGridState* state) {
  __shared__ unsigned long long epoch;
  __shared__ int slot, unusedCount;
  __shared__ size_t unusedBytes;
  // Only the group's first CTA reserves the whole-message slot.
  if (group.thread_rank() < mscclppDeviceCollectiveThreadCount()) {
    int rc = liteCollectiveBeginBlock(h, src, bytesPerRank, &epoch, &slot,
                                      &unusedBytes, &unusedCount);
    if (group.thread_rank() == 0) {
      state->status = rc;
      if (!rc) {
        state->epoch = epoch;
        state->slot = slot;
      }
    }
  }
  group.sync();
  if (state->status) return state->status;
  unsigned long long callEpoch = state->epoch;
  int callSlot = state->slot;
  char* slab = h.slab + static_cast<size_t>(callSlot) * h.slotStride;
  liteAllGatherGroupCopy(group, slab + h.rank * bytesPerRank, src,
                         bytesPerRank);
  __threadfence_system();
  group.sync();
  if (group.thread_rank() == 0) {
    liteStoreRelease(mscclppDeviceCollectiveReady(h, callSlot, 0, h.rank),
                     callEpoch);
    for (int r = 0; r < h.nranks; ++r) {
      if (!liteCollectiveWait(
              h, mscclppDeviceCollectiveReady(h, callSlot, 0, r), callEpoch)) {
        state->status = mscclppDeviceCollectiveTransportError;
        break;
      }
    }
  }
  group.sync();
  if (state->status) return state->status;
  __threadfence_system();
  liteAllGatherGroupCopy(group, dst, slab, bytesPerRank * h.nranks);
  __threadfence_system();
  group.sync();
  if (group.thread_rank() == 0)
    liteStoreRelease(mscclppDeviceCollectiveDone(h, callSlot, h.rank),
                     callEpoch);
  group.sync();
  return mscclppDeviceCollectiveSuccess;
}

// For a multi-block caller, every grid thread must enter this path through a
// cudaLaunchCooperativeKernel launch, with occupancy-safe grid dimensions.
// A single-block caller retains the existing device-callable block API.
static __device__ __forceinline__ int liteAllGatherHostCooperativeBlock(
    mscclppDeviceCollectiveHandle_t const& h, char const* src, char* dst,
    size_t bytesPerRank, LiteAllGatherPlan const&,
    bool cooperativeGrid = false) {
  __shared__ LiteAllGatherGridState blockState;
  if (!cooperativeGrid)
    return liteAllGatherHostCooperativeGroup(
        h, src, dst, bytesPerRank, cooperative_groups::this_thread_block(),
        &blockState);
  auto grid = cooperative_groups::this_grid();
  if (!grid.is_valid() || !h.allGatherGridState)
    return mscclppDeviceCollectiveInvalidUsage;
  return liteAllGatherHostCooperativeGroup(h, src, dst, bytesPerRank, grid,
                                           h.allGatherGridState);
}
