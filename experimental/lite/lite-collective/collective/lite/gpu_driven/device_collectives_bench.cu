#include "gpu_collectives.cuh"

#include <mpi.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define CUDA_CHECK(call)                                                        \
  do {                                                                          \
    cudaError_t error__ = (call);                                                \
    if (error__ != cudaSuccess) {                                                \
      std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,       \
                   cudaGetErrorString(error__));                                \
      MPI_Abort(MPI_COMM_WORLD, 1);                                              \
    }                                                                           \
  } while (0)

#define NCCL_CHECK(call)                                                        \
  do {                                                                          \
    ncclResult_t error__ = (call);                                               \
    if (error__ != ncclSuccess) {                                                \
      std::fprintf(stderr, "UCCL-lite error %s:%d: %s\n", __FILE__, __LINE__,  \
                   ncclGetErrorString(error__));                                \
      MPI_Abort(MPI_COMM_WORLD, 1);                                              \
    }                                                                           \
  } while (0)

enum class BenchCollective { AllGather, AllReduce, ReduceScatter };

__global__ void initializeFloats(float* data, size_t count, int rank) {
  size_t tid = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
  for (size_t i = tid; i < count; i += stride) {
    data[i] = static_cast<float>(rank + 1);
  }
}

__global__ void allGatherBenchKernel(mscclppDeviceAllGatherHandle_t handle,
                                     const float* input, float* output,
                                     size_t count, int* status) {
  int rc = liteAllGatherBlock(handle, input, output, count * sizeof(float));
  if (threadIdx.x == 0) *status = rc;
}

__global__ void allReduceBenchKernel(mscclppDeviceAllGatherHandle_t handle,
                                     const float* input, float* output,
                                     size_t count, int* status) {
  int rc = liteAllReduceBlock(handle, input, output, count, liteReduceSum);
  if (threadIdx.x == 0) *status = rc;
}

__global__ void reduceScatterBenchKernel(
    mscclppDeviceAllGatherHandle_t handle, const float* input, float* output,
    size_t recvCount, int* status) {
  int rc = liteReduceScatterBlock(handle, input, output, recvCount,
                                  liteReduceSum);
  if (threadIdx.x == 0) *status = rc;
}

static size_t parseSize(const char* text) {
  char* end = nullptr;
  unsigned long long value = std::strtoull(text, &end, 10);
  if (end == text || value == 0) return 0;
  if (*end == 'K' || *end == 'k') value *= 1024ULL;
  if (*end == 'M' || *end == 'm') value *= 1024ULL * 1024ULL;
  return static_cast<size_t>(value);
}

static const char* collectiveName(BenchCollective collective) {
  switch (collective) {
    case BenchCollective::AllGather: return "allgather";
    case BenchCollective::AllReduce: return "allreduce";
    case BenchCollective::ReduceScatter: return "reducescatter";
  }
  return "unknown";
}

static float percentile(std::vector<float> values, double fraction) {
  std::sort(values.begin(), values.end());
  size_t index = static_cast<size_t>(fraction * (values.size() - 1));
  return values[index];
}

static float launchOnce(BenchCollective collective,
                        mscclppDeviceAllGatherHandle_t handle,
                        const float* input, float* output, size_t count,
                        int* status, cudaEvent_t start, cudaEvent_t stop) {
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
  return milliseconds * 1000.0f;
}

static void runBenchmark(BenchCollective collective, size_t bytes,
                         int warmups, int iterations, int rank, int nranks,
                         mscclppDeviceAllGatherHandle_t handle) {
  if (bytes % sizeof(float) != 0) {
    if (rank == 0) std::fprintf(stderr, "size must be divisible by 4\n");
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
  size_t count = bytes / sizeof(float);
  size_t inputCount = collective == BenchCollective::ReduceScatter
                          ? count * static_cast<size_t>(nranks)
                          : count;
  size_t outputCount = collective == BenchCollective::AllGather
                           ? count * static_cast<size_t>(nranks)
                           : count;

  float* input = nullptr;
  float* output = nullptr;
  int* status = nullptr;
  CUDA_CHECK(cudaMalloc(&input, inputCount * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&output, outputCount * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&status, sizeof(int)));
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
    (void)launchOnce(collective, handle, input, output, count, status, start,
                     stop);
  }

  std::vector<float> maximumRankTimes;
  if (rank == 0) maximumRankTimes.reserve(iterations);
  for (int i = 0; i < iterations; ++i) {
    MPI_Barrier(MPI_COMM_WORLD);
    float localUs = launchOnce(collective, handle, input, output, count, status,
                               start, stop);
    float maximumUs = 0.0f;
    MPI_Reduce(&localUs, &maximumUs, 1, MPI_FLOAT, MPI_MAX, 0,
               MPI_COMM_WORLD);
    if (rank == 0) maximumRankTimes.push_back(maximumUs);
  }

  int deviceStatus = 0;
  CUDA_CHECK(cudaMemcpy(&deviceStatus, status, sizeof(int),
                        cudaMemcpyDeviceToHost));
  int maximumStatus = 0;
  MPI_Allreduce(&deviceStatus, &maximumStatus, 1, MPI_INT, MPI_MAX,
                MPI_COMM_WORLD);
  if (maximumStatus != mscclppDeviceCollectiveSuccess) {
    if (rank == 0)
      std::fprintf(stderr, "%s returned device status %d\n",
                   collectiveName(collective), maximumStatus);
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  if (rank == 0) {
    double sum = 0.0;
    for (float value : maximumRankTimes) sum += value;
    std::printf("%-14s bytes_per_rank=%-8zu kernel_us avg=%8.3f "
                "p50=%8.3f p95=%8.3f min=%8.3f\n",
                collectiveName(collective), bytes,
                sum / maximumRankTimes.size(),
                percentile(maximumRankTimes, 0.50),
                percentile(maximumRankTimes, 0.95),
                *std::min_element(maximumRankTimes.begin(),
                                  maximumRankTimes.end()));
    std::fflush(stdout);
  }

  cudaEventDestroy(stop);
  cudaEventDestroy(start);
  cudaFree(status);
  cudaFree(output);
  cudaFree(input);
}

int main(int argc, char** argv) {
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
  if (!supportedLayout || nranks < 2 || nranks > 8) {
    if (rank == 0)
      std::fprintf(stderr,
                   "benchmark requires one or two balanced nodes and 2-8 "
                   "total ranks\n");
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
  CUDA_CHECK(cudaSetDevice(localRank));

  int warmups = 20;
  int iterations = 100;
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
  if (largestBytes > static_cast<size_t>(-1) / static_cast<size_t>(nranks)) {
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
  size_t maxStagedBytes = largestBytes * static_cast<size_t>(nranks);

  ncclUniqueId id{};
  if (rank == 0) NCCL_CHECK(ncclGetUniqueId(&id));
  MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);
  ncclComm_t comm = nullptr;
  NCCL_CHECK(ncclCommInitRank(&comm, nranks, id, rank));

  const char* backendEnv = std::getenv("UCCL_GPU_DRIVEN_BACKEND");
  mscclppDeviceCollectiveBackend_t backend =
      mscclppDeviceCollectiveHostMemory;
  if (backendEnv != nullptr && std::strcmp(backendEnv, "cuda_ipc") == 0) {
    backend = mscclppDeviceCollectiveCudaIpc;
  } else if (backendEnv != nullptr && std::strcmp(backendEnv, "host") != 0) {
    if (rank == 0) {
      std::fprintf(stderr,
                   "UCCL_GPU_DRIVEN_BACKEND must be host or cuda_ipc\n");
    }
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  mscclppDeviceAllGatherHandle_t handle{};
  NCCL_CHECK(mscclppGetDeviceCollectiveHandle(
      comm, maxStagedBytes, backend, &handle));

  if (rank == 0) {
    std::printf(
        "GPU-driven device collective kernel latency, backend=%s, ranks=%d, "
        "warmups=%d, iterations=%d\n",
        handle.backend == mscclppDeviceCollectiveCudaIpc
            ? "cuda_ipc"
            : (handle.backend == mscclppDeviceCollectiveHostRdma
                   ? "host_rdma"
                   : "host"),
        nranks, warmups, iterations);
  }
  for (size_t bytes : sizes) {
    runBenchmark(BenchCollective::AllGather, bytes, warmups, iterations, rank,
                 nranks, handle);
    runBenchmark(BenchCollective::AllReduce, bytes, warmups, iterations, rank,
                 nranks, handle);
    runBenchmark(BenchCollective::ReduceScatter, bytes, warmups, iterations,
                 rank, nranks, handle);
  }

  CUDA_CHECK(cudaDeviceSynchronize());
  NCCL_CHECK(ncclCommDestroy(comm));
  MPI_Comm_free(&localComm);
  MPI_Finalize();
  return 0;
}
