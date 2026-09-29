#pragma once
// CPU helpers shared by the CPU-reduced paths. Single-node schedules live on the
// device (reducescatter_{ipc,host}.cuh); only the CPU sum and row staging remain.
// CPU SIMD dispatch is shared with CpuSwitch; RS's explicit disable switch
// selects the scalar implementation without changing other collectives.
static void liteRsCpuSum(LiteReduceScatterContext& c, char* output,
                         char const* const* rows, int n, size_t bytes) {
  float const* inputs[4]{};
  for (int r = 0; r < n; ++r)
    inputs[r] = reinterpret_cast<float const*>(rows[r]);
  auto* dst = reinterpret_cast<float*>(output);
  if (!c.policy.disableAvx512) {
    mscclpp::lite::detail::reduceFloatSum(inputs, n, dst,
                                          bytes / sizeof(float));
  } else {
    for (size_t i = 0; i < bytes / sizeof(float); ++i) {
      float value = inputs[0][i];
      for (int r = 1; r < n; ++r) value += inputs[r][i];
      dst[i] = value;
    }
  }
}

static void liteRsStageRows(LiteReduceScatterContext& c, char* dst,
                            char const* src, size_t shardBytes, size_t bytes,
                            bool copy2d) {
  if (copy2d) {
    MSCCLPP_CUDATHROW(cudaMemcpy2DAsync(dst, c.chunkCapacity, src, shardBytes,
                                        bytes, c.ranks, cudaMemcpyDefault,
                                        c.streams[0]));
  } else {
    for (int r = 0; r < c.ranks; ++r)
      c.copy(dst + r * c.chunkCapacity, src + r * shardBytes, bytes);
  }
}
