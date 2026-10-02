#pragma once

// TwoRankRing (2n*1g, float/sum, B >= 64 MiB or MSCCLPP_NCCL_2RANK_RING_ALLREDUCE):
// the NCCL RING/SIMPLE data order over two independent channels, each with its own
// connection and staging slot. Both channels are driven by the caller's single CTA;
// two channels do not authorize two blocks. Per loop and channel:
//   D2H peer part -> RDMA -> remote own part H2D and CTA add into the output ->
//   D2H reduced own part -> RDMA -> remote final peer part H2D into the output.
// CPU: runTwoRankRingSimpleAllReduce2Node. Stage buffers are registered host memory;
// the ring keeps its own epochs (no ACK is needed because each stage is consumed
// before the next one is produced).
static __device__ __forceinline__ int liteAllReduceTwoRankRingBlock(
    mscclppDeviceCollectiveHandle_t const& h, float const* src, float* dst,
    size_t bytes, LiteAllReducePlan const& plan) {
  return liteArInvokePath(h, src, dst, bytes, plan);
}
