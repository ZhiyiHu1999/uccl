// Host-side ownership and initialization for GPU-driven lite collectives.
//
// Implementation header: include this only from nccl/nccl.cu, after the full
// ncclComm definition is visible. ncclComm_t is intentionally opaque outside
// that implementation, and this file defines non-inline exported functions.
#pragma once

#include "lite/node_exchange_buffer.hpp"

namespace {

struct DeviceCollectiveContext {
  std::mutex mutex;
  std::unique_ptr<HostStagingBuffer> hostBuffer;
  std::unique_ptr<NodeExchangeBuffer> nodeBuffer;
  mscclpp::RegisteredMemory rdmaSendMemory;
  mscclpp::RegisteredMemory rdmaRecvMemory;
  mscclpp::RegisteredMemory rdmaControlMemory;
  mscclpp::RegisteredMemory remoteRdmaRecvMemory;
  mscclpp::RegisteredMemory remoteRdmaControlMemory;
  mscclpp::Connection rdmaConnection;
  std::thread rdmaProxyThread;
  std::atomic<bool> stopRdmaProxy{false};
  char* localIpcSlab = nullptr;
  unsigned long long* localIpcControl = nullptr;
  std::array<char*, MSCCLPP_DEVICE_COLLECTIVE_MAX_RANKS> peerIpcSlabs{};
  std::array<unsigned long long*, MSCCLPP_DEVICE_COLLECTIVE_MAX_RANKS>
      peerIpcControls{};
  unsigned long long* localEpoch = nullptr;
  size_t maxBytesPerRank = 0;
  int rank = -1;
  int nranks = 0;
  int cudaDevice = -1;
  int nRanksPerNode = 0;
  int localRank = -1;
  int localLeader = -1;
  int remoteLeader = -1;
  mscclppDeviceCollectiveBackend_t backend =
      mscclppDeviceCollectiveHostMemory;

  void releaseResources() {
    stopRdmaProxy.store(true, std::memory_order_release);
    if (rdmaProxyThread.joinable()) rdmaProxyThread.join();
    rdmaConnection = mscclpp::Connection{};
    remoteRdmaControlMemory = mscclpp::RegisteredMemory{};
    remoteRdmaRecvMemory = mscclpp::RegisteredMemory{};
    rdmaControlMemory = mscclpp::RegisteredMemory{};
    rdmaRecvMemory = mscclpp::RegisteredMemory{};
    rdmaSendMemory = mscclpp::RegisteredMemory{};
    if (backend == mscclppDeviceCollectiveCudaIpc) {
      for (int r = 0; r < nranks; ++r) {
        if (r == rank) continue;
        if (peerIpcControls[r] != nullptr) {
          cudaIpcCloseMemHandle(peerIpcControls[r]);
        }
        if (peerIpcSlabs[r] != nullptr) {
          cudaIpcCloseMemHandle(peerIpcSlabs[r]);
        }
      }
      if (localIpcControl != nullptr) cudaFree(localIpcControl);
      if (localIpcSlab != nullptr) cudaFree(localIpcSlab);
    }
    if (localEpoch != nullptr) cudaFree(localEpoch);
    hostBuffer.reset();
    nodeBuffer.reset();
    localIpcSlab = nullptr;
    localIpcControl = nullptr;
    localEpoch = nullptr;
    peerIpcSlabs.fill(nullptr);
    peerIpcControls.fill(nullptr);
  }

  ~DeviceCollectiveContext() {
    int previousDevice = -1;
    cudaGetDevice(&previousDevice);
    if (cudaDevice >= 0) cudaSetDevice(cudaDevice);
    releaseResources();
    if (previousDevice >= 0 && previousDevice != cudaDevice) {
      cudaSetDevice(previousDevice);
    }
  }
};

using DeviceCollectiveContexts =
    std::array<std::unique_ptr<DeviceCollectiveContext>, 2>;
static std::mutex gDeviceCollectiveContextMutex;
static std::unordered_map<ncclComm_t, DeviceCollectiveContexts>
    gDeviceCollectiveContexts;

static void cleanupDeviceCollectiveContext(ncclComm_t comm) {
  std::lock_guard<std::mutex> lock(gDeviceCollectiveContextMutex);
  gDeviceCollectiveContexts.erase(comm);
}

struct DeviceCollectiveConfig {
  size_t maxBytesPerRank;
  int backend;
  int cudaDevice;
};

struct DeviceCollectiveIpcInfo {
  cudaIpcMemHandle_t slab;
  cudaIpcMemHandle_t control;
};

static mscclpp::Transport deviceCollectiveIbTransport(int cudaDevice) {
  static constexpr mscclpp::Transport transports[] = {
      mscclpp::Transport::IB0, mscclpp::Transport::IB1,
      mscclpp::Transport::IB2, mscclpp::Transport::IB3,
      mscclpp::Transport::IB4, mscclpp::Transport::IB5,
      mscclpp::Transport::IB6, mscclpp::Transport::IB7};
  int count = 0;
  try {
    count = std::min(mscclpp::getIBDeviceCount(), 8);
  } catch (...) {
    return mscclpp::Transport::Unknown;
  }
  if (count <= 0) return mscclpp::Transport::Unknown;
  return transports[static_cast<unsigned>(cudaDevice) % count];
}

static bool waitDeviceCollectiveEpoch(
    volatile uint64_t const* value, uint64_t epoch,
    std::atomic<bool> const& stop) {
  int spins = 0;
  while (*value < epoch) {
    if (stop.load(std::memory_order_acquire)) return false;
    if (spins++ < 65536) {
#if defined(__x86_64__) || defined(__i386__)
      asm volatile("pause" ::: "memory");
#endif
    } else {
      std::this_thread::yield();
    }
  }
  std::atomic_thread_fence(std::memory_order_acquire);
  return true;
}

static void runDeviceCollectiveRdmaProxy(DeviceCollectiveContext* context) {
  NebCtrl* control = context->nodeBuffer->control();
  size_t slotStride = static_cast<size_t>(context->nRanksPerNode) *
                      context->maxBytesPerRank;
  for (uint64_t epoch = 1;
       !context->stopRdmaProxy.load(std::memory_order_acquire); ++epoch) {
    int slot = static_cast<int>((epoch - 1) %
                                MSCCLPP_DEVICE_COLLECTIVE_SLOTS);
    if (epoch > MSCCLPP_DEVICE_COLLECTIVE_SLOTS &&
        !waitDeviceCollectiveEpoch(
            &control->remoteAck[slot],
            epoch - MSCCLPP_DEVICE_COLLECTIVE_SLOTS,
            context->stopRdmaProxy)) {
      return;
    }
    for (int local = 0; local < context->nRanksPerNode; ++local) {
      if (!waitDeviceCollectiveEpoch(&control->gpuReady[slot][local], epoch,
                                     context->stopRdmaProxy)) {
        return;
      }
    }

    size_t slotOffset = static_cast<size_t>(slot) * slotStride;
    size_t bytes = static_cast<size_t>(control->gpuBytes[slot][0]);
    for (int local = 0; local < context->nRanksPerNode; ++local) {
      size_t rowOffset = slotOffset +
                         static_cast<size_t>(local) *
                             context->maxBytesPerRank;
      context->rdmaConnection.write(context->remoteRdmaRecvMemory, rowOffset,
                                    context->rdmaSendMemory, rowOffset, bytes);
    }
    control->rxSignal[slot] = epoch;
    context->rdmaConnection.write(
        context->remoteRdmaControlMemory,
        offsetof(NebCtrl, rxReady) + static_cast<size_t>(slot) * sizeof(uint64_t),
        context->rdmaControlMemory,
        offsetof(NebCtrl, rxSignal) + static_cast<size_t>(slot) * sizeof(uint64_t),
        sizeof(uint64_t));
    context->rdmaConnection.flush();
    std::atomic_thread_fence(std::memory_order_release);
    control->slotReusable[slot] = epoch;

    for (int local = 0; local < context->nRanksPerNode; ++local) {
      if (!waitDeviceCollectiveEpoch(&control->gpuDone[slot][local], epoch,
                                     context->stopRdmaProxy)) {
        return;
      }
    }
    control->ackSignal[slot] = epoch;
    context->rdmaConnection.write(
        context->remoteRdmaControlMemory,
        offsetof(NebCtrl, remoteAck) + static_cast<size_t>(slot) * sizeof(uint64_t),
        context->rdmaControlMemory,
        offsetof(NebCtrl, ackSignal) + static_cast<size_t>(slot) * sizeof(uint64_t),
        sizeof(uint64_t));
    context->rdmaConnection.flush();
  }
}

static void initializeCudaIpcDeviceCollective(
    DeviceCollectiveContext& context, ncclComm_t comm,
    size_t alignedMaxBytesPerRank) {
  size_t slabBytes = alignedMaxBytesPerRank *
                     MSCCLPP_DEVICE_COLLECTIVE_SLOTS;
  size_t controlWords = MSCCLPP_DEVICE_COLLECTIVE_SLOTS *
                        (MSCCLPP_DEVICE_COLLECTIVE_MAX_CHUNKS + 1);
  size_t controlBytes = controlWords * sizeof(unsigned long long);
  std::vector<DeviceCollectiveIpcInfo> infos(context.nranks);
  std::vector<int> statuses(context.nranks);
  try {
    MSCCLPP_CUDATHROW(cudaMalloc(&context.localIpcSlab, slabBytes));
    MSCCLPP_CUDATHROW(cudaMalloc(&context.localIpcControl, controlBytes));
    MSCCLPP_CUDATHROW(cudaMemset(context.localIpcControl, 0, controlBytes));
    MSCCLPP_CUDATHROW(
        cudaIpcGetMemHandle(&infos[context.rank].slab, context.localIpcSlab));
    MSCCLPP_CUDATHROW(cudaIpcGetMemHandle(&infos[context.rank].control,
                                         context.localIpcControl));
  } catch (...) {
    statuses[context.rank] = 1;
  }
  comm->comm->bootstrap()->allGather(statuses.data(), sizeof(statuses[0]));
  if (std::any_of(statuses.begin(), statuses.end(),
                  [](int status) { return status != 0; })) {
    throw mscclpp::Error("CUDA IPC allocation or handle export failed",
                         mscclpp::ErrorCode::InvalidUsage);
  }
  comm->comm->bootstrap()->allGather(infos.data(), sizeof(infos[0]));

  statuses.assign(context.nranks, 0);
  try {
    for (int r = 0; r < context.nranks; ++r) {
      if (r == context.rank) {
        context.peerIpcSlabs[r] = context.localIpcSlab;
        context.peerIpcControls[r] = context.localIpcControl;
        continue;
      }
      void* slab = nullptr;
      void* control = nullptr;
      MSCCLPP_CUDATHROW(cudaIpcOpenMemHandle(
          &slab, infos[r].slab, cudaIpcMemLazyEnablePeerAccess));
      try {
        MSCCLPP_CUDATHROW(cudaIpcOpenMemHandle(
            &control, infos[r].control, cudaIpcMemLazyEnablePeerAccess));
      } catch (...) {
        cudaIpcCloseMemHandle(slab);
        throw;
      }
      context.peerIpcSlabs[r] = static_cast<char*>(slab);
      context.peerIpcControls[r] = static_cast<unsigned long long*>(control);
    }
  } catch (...) {
    statuses[context.rank] = 1;
  }
  comm->comm->bootstrap()->allGather(statuses.data(), sizeof(statuses[0]));
  if (std::any_of(statuses.begin(), statuses.end(),
                  [](int status) { return status != 0; })) {
    throw mscclpp::Error("opening a CUDA IPC peer mapping failed",
                         mscclpp::ErrorCode::InvalidUsage);
  }
}

static void initializeHostRdmaDeviceCollective(
    DeviceCollectiveContext& context, ncclComm_t comm) {
  context.localRank = context.rank % context.nRanksPerNode;
  int nodeId = context.rank / context.nRanksPerNode;
  context.localLeader = nodeId * context.nRanksPerNode;
  context.remoteLeader = (1 - nodeId) * context.nRanksPerNode;
  bool isLeader = context.rank == context.localLeader;
  size_t slotStride = static_cast<size_t>(context.nRanksPerNode) *
                      context.maxBytesPerRank;
  size_t slabBytes = MSCCLPP_DEVICE_COLLECTIVE_SLOTS * slotStride;
  auto nonce = static_cast<unsigned long long>(
      reinterpret_cast<uintptr_t>(comm));
  char nameTag[80];
  std::snprintf(nameTag, sizeof(nameTag), "gpu_rdma_%llx_%d_%d", nonce,
                getpid(), context.localLeader);
  int numaNode = -1;
  try {
    numaNode = mscclpp::getDeviceNumaNode(context.cudaDevice);
  } catch (...) {
  }
  context.nodeBuffer = std::make_unique<NodeExchangeBuffer>(
      NodeExchangeBuffer::create(
          comm->comm, context.rank, context.nranks, isLeader,
          context.localLeader, slabBytes, numaNode, context.cudaDevice,
          nameTag));
  if (context.nodeBuffer->sendDevicePtr() == nullptr ||
      context.nodeBuffer->recvDevicePtr() == nullptr ||
      context.nodeBuffer->controlDevicePtr() == nullptr) {
    throw mscclpp::Error(
        "GPU-driven inter-node collectives require mapped host slabs",
        mscclpp::ErrorCode::InvalidUsage);
  }

  mscclpp::Transport transport =
      deviceCollectiveIbTransport(context.cudaDevice);
  struct InitStatus {
    int failed = 0;
    char message[160] = {};
  };
  std::vector<InitStatus> statuses(context.nranks);
  try {
    if (transport == mscclpp::Transport::Unknown) {
      throw mscclpp::Error(
          "GPU-driven inter-node collectives require an IB transport",
          mscclpp::ErrorCode::InvalidUsage);
    }
    if (isLeader) {
      mscclpp::TransportFlags flags(transport);
      context.rdmaSendMemory = comm->comm->registerMemory(
          context.nodeBuffer->sendPtr(), slabBytes, flags);
      context.rdmaRecvMemory = comm->comm->registerMemory(
          context.nodeBuffer->recvPtr(), slabBytes, flags);
      context.rdmaControlMemory = comm->comm->registerMemory(
          context.nodeBuffer->control(), sizeof(NebCtrl), flags);

      mscclpp::EndpointConfig::Ib ibConfig;
      ibConfig.maxCqPollNum = 128;
      mscclpp::EndpointConfig endpointConfig(
          transport, mscclpp::Device(mscclpp::DeviceType::CPU),
          /*maxWriteQueueSize=*/-1, ibConfig);
      int pair = std::min(context.localLeader, context.remoteLeader) *
                     context.nranks +
                 std::max(context.localLeader, context.remoteLeader);
      int tagBase = 0x5b0000 + pair * 8;
      auto connectionFuture =
          comm->comm->connect(endpointConfig, context.remoteLeader, tagBase);
      comm->comm->sendMemory(context.rdmaRecvMemory, context.remoteLeader,
                             tagBase + 1);
      auto remoteRecvFuture =
          comm->comm->recvMemory(context.remoteLeader, tagBase + 1);
      comm->comm->sendMemory(context.rdmaControlMemory, context.remoteLeader,
                             tagBase + 2);
      auto remoteControlFuture =
          comm->comm->recvMemory(context.remoteLeader, tagBase + 2);
      context.rdmaConnection = connectionFuture.get();
      context.remoteRdmaRecvMemory = remoteRecvFuture.get();
      context.remoteRdmaControlMemory = remoteControlFuture.get();
    }
  } catch (std::exception const& ex) {
    statuses[context.rank].failed = 1;
    std::snprintf(statuses[context.rank].message,
                  sizeof(statuses[context.rank].message), "%s", ex.what());
  } catch (...) {
    statuses[context.rank].failed = 1;
    std::snprintf(statuses[context.rank].message,
                  sizeof(statuses[context.rank].message),
                  "unknown RDMA setup error");
  }
  comm->comm->bootstrap()->allGather(statuses.data(), sizeof(statuses[0]));
  for (int r = 0; r < context.nranks; ++r) {
    if (statuses[r].failed) {
      throw mscclpp::Error(
          "GPU-driven RDMA setup failed on rank " + std::to_string(r) +
              ": " + statuses[r].message,
          mscclpp::ErrorCode::SystemError);
    }
  }
  if (isLeader) {
    context.stopRdmaProxy.store(false, std::memory_order_release);
    context.rdmaProxyThread =
        std::thread(runDeviceCollectiveRdmaProxy, &context);
  }
}

static void fillDeviceCollectiveHandle(
    const DeviceCollectiveContext& context,
    mscclppDeviceCollectiveHandle_t* handle) {
  *handle = {};
  handle->localEpoch = context.localEpoch;
  handle->maxBytesPerRank = context.maxBytesPerRank;
  handle->rank = context.rank;
  handle->nranks = context.nranks;
  handle->backend = context.backend;

  if (context.backend == mscclppDeviceCollectiveHostMemory) {
    CscDeviceHandle raw = context.hostBuffer->deviceHandle();
    if (raw.slabDev == nullptr || raw.ctrlDev == nullptr) {
      throw mscclpp::Error(
          "device collective requires GPU-mapped shared host memory",
          mscclpp::ErrorCode::InvalidUsage);
    }
    handle->slab = raw.slabDev;
    handle->control = raw.ctrlDev;
    handle->maxBytesPerRank = raw.bytesPerRank;
    handle->slotStride = raw.slotStride;
    handle->counterStride = raw.counterStride;
    return;
  }

  if (context.backend == mscclppDeviceCollectiveHostRdma) {
    int nodeId = context.rank / context.nRanksPerNode;
    int localBase = nodeId * context.nRanksPerNode;
    int remoteBase = (1 - nodeId) * context.nRanksPerNode;
    size_t slotStride = static_cast<size_t>(context.nRanksPerNode) *
                        context.maxBytesPerRank;
    char* send = const_cast<char*>(context.nodeBuffer->sendDevicePtr());
    char* recv = const_cast<char*>(context.nodeBuffer->recvDevicePtr());
    char* control = context.nodeBuffer->controlDevicePtr();
    for (int local = 0; local < context.nRanksPerNode; ++local) {
      int localGlobal = localBase + local;
      int remoteGlobal = remoteBase + local;
      handle->peerSlabs[localGlobal] =
          send + static_cast<size_t>(local) * context.maxBytesPerRank;
      handle->peerSlabs[remoteGlobal] =
          recv + static_cast<size_t>(local) * context.maxBytesPerRank;
      handle->peerReady[localGlobal] =
          reinterpret_cast<unsigned long long*>(
              control + offsetof(NebCtrl, gpuReady) +
              static_cast<size_t>(local) * sizeof(uint64_t));
      handle->peerReady[remoteGlobal] =
          reinterpret_cast<unsigned long long*>(
              control + offsetof(NebCtrl, rxReady));
      handle->peerDone[localGlobal] =
          reinterpret_cast<unsigned long long*>(
              control + offsetof(NebCtrl, gpuDone) +
              static_cast<size_t>(local) * sizeof(uint64_t));
      handle->peerDone[remoteGlobal] = handle->peerDone[localGlobal];
      handle->peerReadySlotStride[localGlobal] = kNebMaxRanks;
      handle->peerReadySlotStride[remoteGlobal] = 1;
      handle->peerDoneSlotStride[localGlobal] = kNebMaxRanks;
      handle->peerDoneSlotStride[remoteGlobal] = kNebMaxRanks;
    }
    handle->slotStride = slotStride;
    handle->slotReusable = reinterpret_cast<unsigned long long*>(
        control + offsetof(NebCtrl, slotReusable));
    handle->publishedBytes = reinterpret_cast<unsigned long long*>(
        control + offsetof(NebCtrl, gpuBytes));
    handle->publishedBytesSlotStride = kNebMaxRanks;
    return;
  }

  for (int r = 0; r < context.nranks; ++r) {
    handle->peerSlabs[r] = context.peerIpcSlabs[r];
    handle->peerReady[r] = context.peerIpcControls[r];
    handle->peerDone[r] = context.peerIpcControls[r] +
                          MSCCLPP_DEVICE_COLLECTIVE_SLOTS *
                              MSCCLPP_DEVICE_COLLECTIVE_MAX_CHUNKS;
    handle->peerReadySlotStride[r] =
        MSCCLPP_DEVICE_COLLECTIVE_MAX_CHUNKS;
    handle->peerDoneSlotStride[r] = 1;
  }
}

}  // namespace

NCCL_API ncclResult_t mscclppGetDeviceCollectiveHandle(
    ncclComm_t comm, size_t maxBytesPerRank,
    mscclppDeviceCollectiveBackend_t backend,
    mscclppDeviceCollectiveHandle_t* handle) {
  if (comm == nullptr || handle == nullptr || maxBytesPerRank == 0 ||
      (backend != mscclppDeviceCollectiveHostMemory &&
       backend != mscclppDeviceCollectiveCudaIpc)) {
    return ncclInvalidArgument;
  }

  int rank = comm->comm->bootstrap()->getRank();
  int nranks = comm->comm->bootstrap()->getNranks();
  int nRanksPerNode = comm->nRanksPerNode;
  bool singleNode = nranks == nRanksPerNode;
  bool twoNodes = nRanksPerNode > 0 && nranks == 2 * nRanksPerNode;
  if (nranks < 2 || nranks > MSCCLPP_DEVICE_COLLECTIVE_MAX_RANKS ||
      (!singleNode && !twoNodes) ||
      (twoNodes && backend == mscclppDeviceCollectiveCudaIpc)) {
    return ncclInvalidUsage;
  }
  if (maxBytesPerRank >
      std::numeric_limits<size_t>::max() /
          (MSCCLPP_DEVICE_COLLECTIVE_SLOTS *
           static_cast<size_t>(std::max(1, nRanksPerNode)))) {
    return ncclInvalidArgument;
  }

  return runNcclGuarded("device collective handle initialization", [&]() {
    mscclpp::CudaDeviceGuard deviceGuard(comm->cudaDevice);
    std::vector<DeviceCollectiveConfig> configs(nranks);
    configs[rank] = {maxBytesPerRank, static_cast<int>(backend),
                     comm->cudaDevice};
    comm->comm->bootstrap()->allGather(configs.data(), sizeof(configs[0]));
    for (const auto& config : configs) {
      if (config.maxBytesPerRank != maxBytesPerRank ||
          config.backend != static_cast<int>(backend)) {
        throw mscclpp::Error(
            "all ranks must initialize the same device collective backend "
            "and maxBytesPerRank",
            mscclpp::ErrorCode::InvalidUsage);
      }
    }
    if (backend == mscclppDeviceCollectiveCudaIpc) {
      int localStatus = 0;
      for (int r = 0; r < nranks; ++r) {
        if (r == rank) continue;
        int canAccessPeer = 0;
        cudaError_t error = cudaDeviceCanAccessPeer(
            &canAccessPeer, comm->cudaDevice, configs[r].cudaDevice);
        if (error != cudaSuccess || canAccessPeer == 0) localStatus = 1;
      }
      std::vector<int> peerStatuses(nranks);
      peerStatuses[rank] = localStatus;
      comm->comm->bootstrap()->allGather(peerStatuses.data(),
                                         sizeof(peerStatuses[0]));
      if (std::any_of(peerStatuses.begin(), peerStatuses.end(),
                      [](int status) { return status != 0; })) {
        throw mscclpp::Error(
            "CUDA IPC backend requires peer access between every GPU pair",
            mscclpp::ErrorCode::InvalidUsage);
      }
    }

    DeviceCollectiveContext* context = nullptr;
    {
      std::lock_guard<std::mutex> mapLock(gDeviceCollectiveContextMutex);
      auto& entry = gDeviceCollectiveContexts[comm][static_cast<int>(backend)];
      if (!entry) entry = std::make_unique<DeviceCollectiveContext>();
      context = entry.get();
    }

    std::lock_guard<std::mutex> contextLock(context->mutex);
    if (context->localEpoch == nullptr) {
      context->maxBytesPerRank =
          (maxBytesPerRank + 15U) & ~static_cast<size_t>(15U);
      context->rank = rank;
      context->nranks = nranks;
      context->cudaDevice = comm->cudaDevice;
      context->nRanksPerNode = nRanksPerNode;
      context->backend = twoNodes ? mscclppDeviceCollectiveHostRdma : backend;
      try {
        MSCCLPP_CUDATHROW(
            cudaMalloc(&context->localEpoch, sizeof(*context->localEpoch)));
        MSCCLPP_CUDATHROW(cudaMemset(context->localEpoch, 0,
                                    sizeof(*context->localEpoch)));

        if (twoNodes) {
          initializeHostRdmaDeviceCollective(*context, comm);
        } else if (backend == mscclppDeviceCollectiveCudaIpc) {
          initializeCudaIpcDeviceCollective(*context, comm,
                                            context->maxBytesPerRank);
        } else {
          auto nc = static_cast<unsigned long long>(
              reinterpret_cast<uintptr_t>(comm));
          char tagBuffer[64];
          std::snprintf(tagBuffer, sizeof(tagBuffer), "device_%d_%llx",
                        getpid(), nc);
          context->hostBuffer = std::make_unique<HostStagingBuffer>(
              HostStagingBuffer::create(
                  context->maxBytesPerRank,
                  MSCCLPP_DEVICE_COLLECTIVE_SLOTS, comm->comm, rank, nranks,
                  comm->cudaDevice, /*mapSlab=*/true,
                  hostStagingNumaPlacementEnabled(), tagBuffer));
        }
        comm->comm->bootstrap()->barrier();
      } catch (...) {
        context->releaseResources();
        throw;
      }
    } else if (context->maxBytesPerRank < maxBytesPerRank) {
      throw mscclpp::Error(
          "device collective handle was initialized with a smaller "
          "maxBytesPerRank",
          mscclpp::ErrorCode::InvalidUsage);
    }

    fillDeviceCollectiveHandle(*context, handle);
  });
}
