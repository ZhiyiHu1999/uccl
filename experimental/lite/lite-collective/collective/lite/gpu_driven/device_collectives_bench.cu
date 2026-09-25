#include "gpu_collectives.cuh"
#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <vector>
#include <cuda_runtime.h>
#include <dlfcn.h>
#include <mpi.h>

#define CUDA_CHECK(call)                                                 \
  do {                                                                   \
    cudaError_t error__ = (call);                                        \
    if (error__ != cudaSuccess) {                                        \
      std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, \
                   cudaGetErrorString(error__));                         \
      MPI_Abort(MPI_COMM_WORLD, 1);                                      \
    }                                                                    \
  } while (0)

#define NCCL_CHECK(call)                                                      \
  do {                                                                        \
    ncclResult_t error__ = (call);                                            \
    if (error__ != ncclSuccess) {                                             \
      std::fprintf(stderr, "UCCL-lite error %s:%d: %s\n", __FILE__, __LINE__, \
                   ncclGetErrorString(error__));                              \
      MPI_Abort(MPI_COMM_WORLD, 1);                                           \
    }                                                                         \
  } while (0)

enum class BenchCollective { AllGather, AllReduce, ReduceScatter };

struct ExternalNcclApi {
  void* library = nullptr;
  decltype(&ncclGetUniqueId) getUniqueId = nullptr;
  decltype(&ncclCommInitRank) commInitRank = nullptr;
  decltype(&ncclCommDestroy) commDestroy = nullptr;
  decltype(&ncclAllGather) allGather = nullptr;
  decltype(&ncclAllReduce) allReduce = nullptr;
  decltype(&ncclReduceScatter) reduceScatter = nullptr;
  decltype(&ncclGetErrorString) getErrorString = nullptr;

  template <typename T>
  T symbol(char const* name) {
    void* value = dlsym(library, name);
    if (value == nullptr) {
      std::fprintf(stderr, "missing NCCL symbol %s: %s\n", name, dlerror());
      MPI_Abort(MPI_COMM_WORLD, 1);
    }
    return reinterpret_cast<T>(value);
  }

  explicit ExternalNcclApi(char const* path) {
    int flags = RTLD_NOW | RTLD_LOCAL;
#if defined(RTLD_DEEPBIND)
    flags |= RTLD_DEEPBIND;
#endif
    library = dlopen(path, flags);
    if (library == nullptr) {
      std::fprintf(stderr, "failed to load NCCL baseline %s: %s\n", path,
                   dlerror());
      MPI_Abort(MPI_COMM_WORLD, 1);
    }
    getUniqueId = symbol<decltype(getUniqueId)>("ncclGetUniqueId");
    commInitRank = symbol<decltype(commInitRank)>("ncclCommInitRank");
    commDestroy = symbol<decltype(commDestroy)>("ncclCommDestroy");
    allGather = symbol<decltype(allGather)>("ncclAllGather");
    allReduce = symbol<decltype(allReduce)>("ncclAllReduce");
    reduceScatter = symbol<decltype(reduceScatter)>("ncclReduceScatter");
    getErrorString = symbol<decltype(getErrorString)>("ncclGetErrorString");
  }

  ~ExternalNcclApi() {
    if (library != nullptr) dlclose(library);
  }

  void check(ncclResult_t result, char const* operation) const {
    if (result == ncclSuccess) return;
    std::fprintf(stderr, "NCCL baseline %s failed: %s\n", operation,
                 getErrorString(result));
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
};

__device__ unsigned char byteValue(int rank, size_t index, int iteration) {
  return static_cast<unsigned char>(rank * 29 + index * 13 + iteration * 7);
}

// Repeated calls inside ONE kernel exercise the actual device-callable API,
// FIFO wrap, payload-slot retirement and freshly produced input visibility.
__global__ void checkGather(mscclppDeviceCollectiveHandle_t h, char* input,
                            char* output, size_t bytes, bool inPlace,
                            unsigned* errors) {
  for (int iteration = 0; iteration < 9; ++iteration) {
    char* source = inPlace ? output + h.rank * bytes : input;
    for (size_t i = threadIdx.x; i < bytes; i += blockDim.x)
      source[i] = static_cast<char>(byteValue(h.rank, i, iteration));
    __syncthreads();
    int rc = liteAllGatherBlock(h, source, output, bytes);
    if (rc) {
      if (!threadIdx.x) atomicAdd(errors, 1);
      return;
    }
    for (size_t i = threadIdx.x; i < bytes * h.nranks; i += blockDim.x)
      if (static_cast<unsigned char>(output[i]) !=
          byteValue(i / bytes, i % bytes, iteration))
        atomicAdd(errors, 1);
    __syncthreads();
  }
}

__device__ int elementValue(int rank, size_t index, int iteration) {
  return rank * 31 + static_cast<int>(index % 113) - 71 + iteration * 3;
}

template <typename T>
__global__ void checkReduction(mscclppDeviceCollectiveHandle_t h, T* input,
                               T* output, size_t count, bool scatter,
                               bool inPlace, liteReduceOp op,
                               unsigned* errors) {
  size_t inputCount = scatter ? count * h.nranks : count;
  T* result = inPlace ? input + (scatter ? h.rank * count : 0) : output;
  for (int iteration = 0; iteration < 5; ++iteration) {
    for (size_t i = threadIdx.x; i < inputCount; i += blockDim.x)
      input[i] = static_cast<T>(elementValue(h.rank, i, iteration));
    __syncthreads();
    int rc = scatter ? liteReduceScatterBlock(h, input, result, count, op)
                     : liteAllReduceBlock(h, input, result, count, op);
    if (rc) {
      if (!threadIdx.x) atomicAdd(errors, 1);
      return;
    }
    for (size_t i = threadIdx.x; i < count; i += blockDim.x) {
      size_t sourceIndex = i + (scatter ? h.rank * count : 0);
      int expected = elementValue(0, sourceIndex, iteration);
      for (int r = 1; r < h.nranks; ++r) {
        int value = elementValue(r, sourceIndex, iteration);
        expected = op == liteReduceSum ? expected + value
                   : op == liteReduceMin
                       ? (expected < value ? expected : value)
                       : (expected > value ? expected : value);
      }
      if (result[i] != static_cast<T>(expected)) atomicAdd(errors, 1);
    }
    __syncthreads();
  }
}

struct Sample {
  float deviceUs;
  float endToEndUs;
};
static_assert(sizeof(Sample) == 2 * sizeof(float));

__global__ void initializeBytes(unsigned char* data, size_t bytes, int rank) {
  for (size_t i = threadIdx.x; i < bytes; i += blockDim.x)
    data[i] = byteValue(rank, i, 0);
}

__global__ void initializeFloats(float* data, size_t count, int rank) {
  size_t tid = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
  for (size_t i = tid; i < count; i += stride) {
    data[i] = static_cast<float>(rank + 1 + i % 97);
  }
}

__global__ void probeDeviceByte(unsigned char volatile* address) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    unsigned char value = *address;
    *address = value;
  }
}

static void probeHandleAddress(char const* name, void* address, int rank) {
  cudaPointerAttributes attributes{};
  cudaError_t attributeResult = cudaPointerGetAttributes(&attributes, address);
  if (attributeResult != cudaSuccess) {
    std::fprintf(stderr, "rank %d: %s=%p cudaPointerGetAttributes failed: %s\n",
                 rank, name, address, cudaGetErrorString(attributeResult));
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
#if CUDART_VERSION >= 10000
  std::fprintf(stderr,
               "rank %d: probing %s=%p type=%d device=%d devicePointer=%p "
               "hostPointer=%p\n",
               rank, name, address, static_cast<int>(attributes.type),
               attributes.device, attributes.devicePointer,
               attributes.hostPointer);
#else
  std::fprintf(stderr,
               "rank %d: probing %s=%p memoryType=%d device=%d "
               "devicePointer=%p hostPointer=%p\n",
               rank, name, address, static_cast<int>(attributes.memoryType),
               attributes.device, attributes.devicePointer,
               attributes.hostPointer);
#endif
  std::fflush(stderr);
  probeDeviceByte<<<1, 1>>>(static_cast<unsigned char volatile*>(address));
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  std::fprintf(stderr, "rank %d: %s probe passed\n", rank, name);
  std::fflush(stderr);
}

static void debugProbeHandle(mscclppDeviceCollectiveHandle_t const& handle,
                             int rank) {
  char const* enabled = std::getenv("UCCL_GPU_DRIVEN_DEBUG_PROBE");
  if (enabled == nullptr || std::strcmp(enabled, "0") == 0) return;
  probeHandleAddress("localEpoch", handle.localEpoch, rank);
  if (handle.backend == mscclppDeviceCollectiveHostMemory) {
    if (handle.slab) probeHandleAddress("hostSlab", handle.slab, rank);
    probeHandleAddress("hostControl", handle.control, rank);
  }
  MPI_Barrier(MPI_COMM_WORLD);
}

__global__ void allGatherBenchKernel(mscclppDeviceCollectiveHandle_t handle,
                                     float const* input, float* output,
                                     size_t count, int* status) {
  int rc = liteAllGatherBlock(handle, input, output, count);
  if (threadIdx.x == 0 && rc) atomicCAS(status, 0, rc);
}

__global__ void allReduceBenchKernel(mscclppDeviceCollectiveHandle_t handle,
                                     float const* input, float* output,
                                     size_t count, int* status) {
  int rc = liteAllReduceBlock(handle, input, output, count, liteReduceSum);
  if (threadIdx.x == 0 && rc) atomicCAS(status, 0, rc);
}

__global__ void reduceScatterBenchKernel(mscclppDeviceCollectiveHandle_t handle,
                                         float const* input, float* output,
                                         size_t recvCount, int* status) {
  int rc =
      liteReduceScatterBlock(handle, input, output, recvCount, liteReduceSum);
  if (threadIdx.x == 0 && rc) atomicCAS(status, 0, rc);
}

static size_t parseSize(char const* text) {
  if (!text || *text < '0' || *text > '9') return 0;
  char* end = nullptr;
  errno = 0;
  unsigned long long value = std::strtoull(text, &end, 10);
  if (errno || end == text || !value) return 0;
  size_t scale = 1;
  if (*end) {
    if (end[1]) return 0;
    if (*end == 'K' || *end == 'k')
      scale = 1024;
    else if (*end == 'M' || *end == 'm')
      scale = 1024 * 1024;
    else if (*end == 'G' || *end == 'g')
      scale = size_t{1} << 30;
    else
      return 0;
  }
  if (value > std::numeric_limits<size_t>::max() / scale) return 0;
  return static_cast<size_t>(value) * scale;
}

static char const* collectiveName(BenchCollective collective) {
  switch (collective) {
    case BenchCollective::AllGather:
      return "allgather";
    case BenchCollective::AllReduce:
      return "allreduce";
    case BenchCollective::ReduceScatter:
      return "reducescatter";
  }
  return "unknown";
}

static Sample launchGpuDrivenOnce(BenchCollective collective,
                                  mscclppDeviceCollectiveHandle_t handle,
                                  float const* input, float* output,
                                  size_t count, int* status, cudaEvent_t start,
                                  cudaEvent_t stop) {
  auto wallStart = std::chrono::steady_clock::now();
  CUDA_CHECK(cudaEventRecord(start));
  switch (collective) {
    case BenchCollective::AllGather:
      allGatherBenchKernel<<<1, 256>>>(handle, input, output, count, status);
      break;
    case BenchCollective::AllReduce:
      allReduceBenchKernel<<<1, 256>>>(handle, input, output, count, status);
      break;
    case BenchCollective::ReduceScatter:
      reduceScatterBenchKernel<<<1, 256>>>(handle, input, output, count,
                                           status);
      break;
  }
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  float milliseconds = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
  auto wallStop = std::chrono::steady_clock::now();
  float endToEndUs =
      std::chrono::duration<float, std::micro>(wallStop - wallStart).count();
  return {milliseconds * 1000.0f, endToEndUs};
}

static Sample launchNcclOnce(BenchCollective collective,
                             ExternalNcclApi const& api, ncclComm_t comm,
                             float const* input, float* output, size_t count,
                             cudaStream_t stream, cudaEvent_t start,
                             cudaEvent_t stop) {
  auto wallStart = std::chrono::steady_clock::now();
  CUDA_CHECK(cudaEventRecord(start, stream));
  ncclResult_t result = ncclSuccess;
  switch (collective) {
    case BenchCollective::AllGather:
      result = api.allGather(input, output, count, ncclUint8, comm, stream);
      break;
    case BenchCollective::AllReduce:
      result = api.allReduce(input, output, count, ncclFloat32, ncclSum, comm,
                             stream);
      break;
    case BenchCollective::ReduceScatter:
      result = api.reduceScatter(input, output, count, ncclFloat32, ncclSum,
                                 comm, stream);
      break;
  }
  api.check(result, collectiveName(collective));
  CUDA_CHECK(cudaEventRecord(stop, stream));
  CUDA_CHECK(cudaEventSynchronize(stop));
  float milliseconds = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
  auto wallStop = std::chrono::steady_clock::now();
  float endToEndUs =
      std::chrono::duration<float, std::micro>(wallStop - wallStart).count();
  return {milliseconds * 1000.0f, endToEndUs};
}

static float mean(std::vector<float> const& values) {
  double sum = 0.0;
  for (float value : values) sum += value;
  return static_cast<float>(sum / values.size());
}

static void runComparison(BenchCollective collective, size_t bytes, int warmups,
                          int iterations, int rank, int nranks,
                          mscclppDeviceCollectiveHandle_t handle,
                          ExternalNcclApi const& ncclApi, ncclComm_t ncclComm,
                          cudaStream_t ncclStream) {
  if (collective == BenchCollective::AllGather &&
      litePlanAllGather(handle.allGatherPolicy, nranks, handle.ranksPerNode,
                        handle.backend == mscclppDeviceCollectiveCudaIpc, true,
                        handle.maxBytesPerRank, bytes, 0)
              .path == LiteDeviceAllGatherPath::Unsupported) {
    if (!rank)
      std::printf(
          "allgather bytes_per_rank=%zu skipped: outside selected backend "
          "policy\n",
          bytes);
    return;
  }
  if (collective != BenchCollective::AllGather && !handle.reductionsMapped) {
    if (!rank)
      std::printf(
          "%s bytes_per_rank=%zu skipped: reductions require mapped payloads\n",
          collectiveName(collective), bytes);
    return;
  }
  if (collective != BenchCollective::AllGather && bytes % sizeof(float)) {
    if (!rank)
      std::printf(
          "%s bytes_per_rank=%zu skipped: float size requires a multiple of "
          "4\n",
          collectiveName(collective), bytes);
    return;
  }
  size_t count =
      collective == BenchCollective::AllGather ? bytes : bytes / sizeof(float);
  size_t inputCount = collective == BenchCollective::ReduceScatter
                          ? count * static_cast<size_t>(nranks)
                          : (bytes + sizeof(float) - 1) / sizeof(float);
  size_t outputCount =
      collective == BenchCollective::AllGather
          ? (bytes * nranks + sizeof(float) - 1) / sizeof(float)
          : count;

  float* input = nullptr;
  float* output = nullptr;
  int* status = nullptr;
  CUDA_CHECK(cudaMalloc(&input, inputCount * sizeof(float) + 16));
  CUDA_CHECK(cudaMalloc(&output, outputCount * sizeof(float) + 16));
  CUDA_CHECK(cudaMalloc(&status, sizeof(int)));
  CUDA_CHECK(cudaMemset(status, 0, sizeof(int)));
  initializeFloats<<<std::min<size_t>(128, (inputCount + 255) / 256), 256>>>(
      input, inputCount, rank);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  // Correctness preflight belongs to the benchmark, outside timed samples.
  // Vary data inside one user kernel to cover FIFO/slot wrap and publication.
  unsigned* errors = nullptr;
  CUDA_CHECK(cudaMalloc(&errors, sizeof(unsigned)));
  CUDA_CHECK(cudaMemset(errors, 0, sizeof(unsigned)));
  if (collective == BenchCollective::AllGather) {
    for (bool inPlace : {false, true})
      for (int unaligned : {0, 1}) {
        checkGather<<<1, 256>>>(handle,
                                reinterpret_cast<char*>(input) + unaligned,
                                reinterpret_cast<char*>(output) + unaligned,
                                bytes, inPlace, errors);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
      }
  } else {
    for (bool inPlace : {false, true})
      for (auto op : {liteReduceSum, liteReduceMin, liteReduceMax}) {
        checkReduction<<<1, 256>>>(handle, input, output, count,
                                   collective == BenchCollective::ReduceScatter,
                                   inPlace, op, errors);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        checkReduction<<<1, 256>>>(handle, reinterpret_cast<int*>(input),
                                   reinterpret_cast<int*>(output), count,
                                   collective == BenchCollective::ReduceScatter,
                                   inPlace, op, errors);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
      }
  }
  unsigned localErrors = 0, totalErrors = 0;
  CUDA_CHECK(cudaMemcpy(&localErrors, errors, sizeof(unsigned),
                        cudaMemcpyDeviceToHost));
  MPI_Allreduce(&localErrors, &totalErrors, 1, MPI_UNSIGNED, MPI_SUM,
                MPI_COMM_WORLD);
  if (totalErrors) {
    if (!rank)
      std::fprintf(stderr, "%s benchmark preflight failed: %u errors\n",
                   collectiveName(collective), totalErrors);
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
  CUDA_CHECK(cudaFree(errors));
  if (collective == BenchCollective::AllGather)
    initializeBytes<<<1, 256>>>(reinterpret_cast<unsigned char*>(input), bytes,
                                rank);
  else
    initializeFloats<<<std::min<size_t>(128, (inputCount + 255) / 256), 256>>>(
        input, inputCount, rank);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  MPI_Barrier(MPI_COMM_WORLD);
  for (int i = 0; i < warmups; ++i) {
    (void)launchGpuDrivenOnce(collective, handle, input, output, count, status,
                              start, stop);
  }
  MPI_Barrier(MPI_COMM_WORLD);
  for (int i = 0; i < warmups; ++i) {
    (void)launchNcclOnce(collective, ncclApi, ncclComm, input, output, count,
                         ncclStream, start, stop);
  }

  std::vector<float> gpuDeviceTimes, gpuEndToEndTimes;
  std::vector<float> ncclDeviceTimes, ncclEndToEndTimes;
  if (rank == 0) {
    gpuDeviceTimes.reserve(iterations);
    gpuEndToEndTimes.reserve(iterations);
    ncclDeviceTimes.reserve(iterations);
    ncclEndToEndTimes.reserve(iterations);
  }
  for (int i = 0; i < iterations; ++i) {
    MPI_Barrier(MPI_COMM_WORLD);
    Sample local = launchGpuDrivenOnce(collective, handle, input, output, count,
                                       status, start, stop);
    Sample maximum{};
    MPI_Reduce(&local, &maximum, 2, MPI_FLOAT, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) {
      gpuDeviceTimes.push_back(maximum.deviceUs);
      gpuEndToEndTimes.push_back(maximum.endToEndUs);
    }
  }
  int deviceStatus = 0, maximumStatus = 0;
  CUDA_CHECK(
      cudaMemcpy(&deviceStatus, status, sizeof(int), cudaMemcpyDeviceToHost));
  MPI_Allreduce(&deviceStatus, &maximumStatus, 1, MPI_INT, MPI_MAX,
                MPI_COMM_WORLD);
  if (maximumStatus) {
    if (!rank)
      std::fprintf(stderr, "%s returned device status %d\n",
                   collectiveName(collective), maximumStatus);
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
  // Validate device output before the NCCL baseline overwrites the same buffer.
  std::vector<float> actual(outputCount);
  CUDA_CHECK(cudaMemcpy(actual.data(), output, outputCount * sizeof(float),
                        cudaMemcpyDeviceToHost));
  if (collective == BenchCollective::AllGather) {
    auto* raw = reinterpret_cast<unsigned char*>(actual.data());
    for (size_t i = 0; i < bytes * nranks; ++i)
      if (raw[i] !=
          static_cast<unsigned char>((i / bytes) * 29 + (i % bytes) * 13)) {
        std::fprintf(stderr, "rank %d: AllGather byte mismatch at %zu\n", rank,
                     i);
        MPI_Abort(MPI_COMM_WORLD, 1);
      }
  } else
    for (size_t i = 0; i < outputCount; ++i) {
      size_t sourceIndex = collective == BenchCollective::ReduceScatter
                               ? static_cast<size_t>(rank) * count + i
                               : i;
      float expected = static_cast<float>(nranks * (nranks + 1) / 2 +
                                          nranks * (sourceIndex % 97));
      if (actual[i] != expected) {
        std::fprintf(stderr, "rank %d: %s mismatch at %zu: %g != %g\n", rank,
                     collectiveName(collective), i, actual[i], expected);
        MPI_Abort(MPI_COMM_WORLD, 1);
      }
    }
  for (int i = 0; i < iterations; ++i) {
    MPI_Barrier(MPI_COMM_WORLD);
    Sample local = launchNcclOnce(collective, ncclApi, ncclComm, input, output,
                                  count, ncclStream, start, stop);
    Sample maximum{};
    MPI_Reduce(&local, &maximum, 2, MPI_FLOAT, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) {
      ncclDeviceTimes.push_back(maximum.deviceUs);
      ncclEndToEndTimes.push_back(maximum.endToEndUs);
    }
  }

  if (rank == 0) {
    float gpuE2e = mean(gpuEndToEndTimes);
    float ncclE2e = mean(ncclEndToEndTimes);
    std::printf(
        "%-14s bytes_per_rank=%-8zu "
        "gpu_avg_device_us=%.3f gpu_avg_e2e_us=%.3f "
        "nccl_avg_device_us=%.3f nccl_avg_e2e_us=%.3f "
        "avg_speedup_e2e=%.3fx\n",
        collectiveName(collective), bytes, mean(gpuDeviceTimes), gpuE2e,
        mean(ncclDeviceTimes), ncclE2e, ncclE2e / gpuE2e);
    std::fflush(stdout);
  }

  cudaEventDestroy(stop);
  cudaEventDestroy(start);
  cudaFree(status);
  cudaFree(output);
  cudaFree(input);
}

int main(int argc, char** argv) {
  // Configure before either NCCL library creates a communicator. One channel
  // and one CTA constrain native NCCL to the same one-SM execution budget.
  setenv("NCCL_MIN_CTAS", "1", 1);
  setenv("NCCL_MAX_CTAS", "1", 1);
  setenv("NCCL_MIN_NCHANNELS", "1", 1);
  setenv("NCCL_MAX_NCHANNELS", "1", 1);
  setenv("NCCL_NET_GDR_LEVEL", "0", 1);
  MPI_Init(&argc, &argv);
  int rank = 0;
  int nranks = 0;
  int localRank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nranks);
  MPI_Comm localComm;
  MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL,
                      &localComm);
  MPI_Comm_rank(localComm, &localRank);
  int localSize = 0;
  MPI_Comm_size(localComm, &localSize);
  bool supportedLayout = localSize == nranks || nranks == 2 * localSize;
  if (!supportedLayout || nranks < 1 || nranks > 8) {
    if (rank == 0)
      std::fprintf(stderr,
                   "benchmark requires one or two balanced nodes and 1-8 "
                   "total ranks\n");
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
  CUDA_CHECK(cudaSetDevice(localRank));

  int warmups = 20;
  int iterations = 100;
  if (char const* value = std::getenv("WARMUP_ITERS")) {
    warmups = std::max(0, std::atoi(value));
  }
  if (char const* value = std::getenv("ITERS")) {
    iterations = std::max(1, std::atoi(value));
  }
  std::vector<size_t> sizes{128, 256, 512, 1024, 4096, 16384, 65536};
  if (argc > 1) {
    sizes.clear();
    for (int i = 1; i < argc; ++i) {
      size_t bytes = parseSize(argv[i]);
      if (bytes == 0) {
        if (rank == 0) std::fprintf(stderr, "invalid size: %s\n", argv[i]);
        MPI_Abort(MPI_COMM_WORLD, 1);
      }
      sizes.push_back(bytes);
    }
  }

  size_t largestBytes = *std::max_element(sizes.begin(), sizes.end());
  if (largestBytes >
      (static_cast<size_t>(-1) - 32) / static_cast<size_t>(nranks)) {
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
  size_t maxStagedBytes = largestBytes * static_cast<size_t>(nranks);

  ncclUniqueId id{};
  if (rank == 0) NCCL_CHECK(ncclGetUniqueId(&id));
  MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);
  ncclComm_t comm = nullptr;
  NCCL_CHECK(ncclCommInitRank(&comm, nranks, id, rank));

  char const* backendEnv = std::getenv("UCCL_GPU_DRIVEN_BACKEND");
  mscclppDeviceCollectiveBackend_t backend = mscclppDeviceCollectiveHostMemory;
  if (backendEnv != nullptr && std::strcmp(backendEnv, "cuda_ipc") == 0) {
    backend = mscclppDeviceCollectiveCudaIpc;
  } else if (backendEnv != nullptr && std::strcmp(backendEnv, "host") != 0) {
    if (rank == 0) {
      std::fprintf(stderr,
                   "UCCL_GPU_DRIVEN_BACKEND must be host or cuda_ipc\n");
    }
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  if (backend == mscclppDeviceCollectiveHostMemory &&
      !std::getenv("MSCCLPP_NCCL_HOST_ALLGATHER"))
    setenv("MSCCLPP_NCCL_HOST_ALLGATHER", "1", 0);
  mscclppDeviceCollectiveHandle_t handle{};
  NCCL_CHECK(
      mscclppGetDeviceCollectiveHandle(comm, maxStagedBytes, backend, &handle));

  char const* ncclBaselinePath = std::getenv("NCCL_BASELINE_LIB");
  if (ncclBaselinePath == nullptr || ncclBaselinePath[0] == '\0') {
    if (rank == 0) {
      std::fprintf(stderr,
                   "NCCL_BASELINE_LIB must point to the real libnccl.so\n");
    }
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
  ExternalNcclApi ncclApi(ncclBaselinePath);
  ncclUniqueId ncclId{};
  if (rank == 0) ncclApi.check(ncclApi.getUniqueId(&ncclId), "get unique ID");
  MPI_Bcast(&ncclId, sizeof(ncclId), MPI_BYTE, 0, MPI_COMM_WORLD);
  ncclComm_t ncclComm = nullptr;
  ncclApi.check(ncclApi.commInitRank(&ncclComm, nranks, ncclId, rank),
                "communicator initialization");
  debugProbeHandle(handle, rank);
  cudaStream_t ncclStream = nullptr;
  CUDA_CHECK(cudaStreamCreateWithFlags(&ncclStream, cudaStreamNonBlocking));

  if (rank == 0) {
    std::printf(
        "GPU-driven vs NCCL collective latency, backend=%s, ranks=%d, "
        "warmups=%d, iterations=%d, nccl=%s, SM budget=1, GDR=off\n",
        handle.backend == mscclppDeviceCollectiveCudaIpc
            ? "cuda_ipc"
            : (handle.backend == mscclppDeviceCollectiveHostRdma ? "host_rdma"
                                                                 : "host"),
        nranks, warmups, iterations, ncclBaselinePath);
  }
  for (size_t bytes : sizes) {
    runComparison(BenchCollective::AllGather, bytes, warmups, iterations, rank,
                  nranks, handle, ncclApi, ncclComm, ncclStream);
    runComparison(BenchCollective::AllReduce, bytes, warmups, iterations, rank,
                  nranks, handle, ncclApi, ncclComm, ncclStream);
    runComparison(BenchCollective::ReduceScatter, bytes, warmups, iterations,
                  rank, nranks, handle, ncclApi, ncclComm, ncclStream);
  }

  CUDA_CHECK(cudaDeviceSynchronize());
  ncclApi.check(ncclApi.commDestroy(ncclComm), "communicator destroy");
  CUDA_CHECK(cudaStreamDestroy(ncclStream));
  NCCL_CHECK(ncclCommDestroy(comm));
  MPI_Comm_free(&localComm);
  MPI_Finalize();
  return 0;
}
