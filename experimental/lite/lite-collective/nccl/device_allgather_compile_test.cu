#include "gpu_collectives.cuh"

__global__ void deviceAllGatherCompileTest(
    mscclppDeviceAllGatherHandle_t handle, float* input, float* output,
    size_t count, int* status) {
  for (size_t i = threadIdx.x; i < count; i += blockDim.x) {
    // Stand-in for computation fused before communication.
    input[i] *= 2.0f;
  }
  __syncthreads();

  int rc = liteAllGatherBlock(handle, input, output, count * sizeof(float));
  if (rc == mscclppDeviceCollectiveSuccess) {
    rc = liteAllReduceBlock(handle, input, output, count, liteReduceSum);
  }
  if (rc == mscclppDeviceCollectiveSuccess && handle.nranks > 0 &&
      count % static_cast<size_t>(handle.nranks) == 0) {
    rc = liteReduceScatterBlock(
        handle, input, output, count / static_cast<size_t>(handle.nranks),
        liteReduceSum);
  }
  if (threadIdx.x == 0) *status = rc;
}
