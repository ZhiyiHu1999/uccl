#pragma once
#include "nccl.h"
#include "task_fifo.hpp"
#include <memory>
#include <cuda_runtime.h>

namespace mscclpp {
class Communicator;
namespace nccl {
// Prepare dedicated CPU-reference transport resources collectively, before
// issuing a device handle. These are not native-NCCL calls or per-call setup.
void* prepareDeviceAllGatherNetwork(ncclComm_t comm,
                                    std::shared_ptr<Communicator> bootstrap,
                                    int rank, int nranks, int ranksPerNode,
                                    int cudaDevice, size_t capacity,
                                    int* groups);
void executeDeviceAllGatherNetwork(void* context, LiteTask const& task,
                                   uint64_t sequence,
                                   LiteNetworkControl* control,
                                   cudaStream_t stream);
void releaseDeviceAllGatherNetwork(void* context);
}  // namespace nccl
}  // namespace mscclpp
