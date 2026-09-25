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
  int groupBase = 0;
  int groupSize = 0;
  int groupNuma = -1;
  // A NUMA aggregate owns one buffer/proxy per contiguous GPU locality group.
  std::vector<std::unique_ptr<DeviceCollectiveContext>> groups;
  std::unique_ptr<DeviceCollectiveContext> numaContext;
  mscclppDeviceCollectiveHandle_t* numaDeviceHandle = nullptr;
  LiteAllGatherPolicy allGatherPolicy{};
  bool reductionsMapped = false;
  unsigned long long timeoutCycles = 0;
  char* smallOutput = nullptr;
  LiteTaskFifo* tasks = nullptr;
  LiteTaskFifo* tasksDevice = nullptr;
  bool tasksRegistered = false;
  cudaStream_t serviceStreams[3]{};
  cudaEvent_t serviceEvents[kLiteTaskSlots][2]{};
  std::thread serviceThread;
  std::unique_ptr<HostStagingBuffer> hostBuffer;
  std::unique_ptr<NodeExchangeBuffer> nodeBuffer;
  mscclpp::RegisteredMemory rdmaSendMemory;
  mscclpp::RegisteredMemory rdmaRecvMemory;
  mscclpp::RegisteredMemory rdmaControlMemory;
  mscclpp::RegisteredMemory remoteRdmaRecvMemory;
  mscclpp::RegisteredMemory remoteRdmaControlMemory;
  mscclpp::Connection rdmaConnection;
  bool dualRail = false;
  mscclpp::Connection rail2Connection;
  mscclpp::RegisteredMemory rail2SendMemory, rail2RecvMemory, rail2RemoteMemory;
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
  mscclppDeviceCollectiveBackend_t backend = mscclppDeviceCollectiveHostMemory;

  void releaseResources() {
    stopRdmaProxy.store(true, std::memory_order_release);
    for (auto& group : groups)
      group->stopRdmaProxy.store(true, std::memory_order_release);
    if (serviceThread.joinable()) serviceThread.join();
    if (rdmaProxyThread.joinable()) rdmaProxyThread.join();
    for (auto& group : groups)
      if (group->rdmaProxyThread.joinable()) group->rdmaProxyThread.join();
    numaContext.reset();
    if (numaDeviceHandle) cudaFree(numaDeviceHandle);
    numaDeviceHandle = nullptr;
    // Drain issued copies before destroying events, registrations or slabs.
    for (auto& stream : serviceStreams) {
      if (stream) {
        cudaStreamSynchronize(stream);
        cudaStreamDestroy(stream);
        stream = nullptr;
      }
    }
    for (auto& pair : serviceEvents)
      for (auto& event : pair) {
        if (event) {
          cudaEventDestroy(event);
          event = nullptr;
        }
      }
    if (smallOutput) {
      cudaFreeHost(smallOutput);
      smallOutput = nullptr;
    }
    if (tasksRegistered) cudaHostUnregister(tasks);
    if (tasks) munmap(tasks, sizeof(LiteTaskFifo));
    tasks = tasksDevice = nullptr;
    tasksRegistered = false;
    rail2Connection = mscclpp::Connection{};
    rail2RemoteMemory = mscclpp::RegisteredMemory{};
    rail2RecvMemory = mscclpp::RegisteredMemory{};
    rail2SendMemory = mscclpp::RegisteredMemory{};
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
    groups.clear();
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
  LiteAllGatherPolicy policy;
  int ibCount;
  int numaNode;
};

struct DeviceCollectiveIpcInfo {
  cudaIpcMemHandle_t slab;
  cudaIpcMemHandle_t control;
};

static mscclpp::Transport deviceCollectiveIbTransport(int cudaDevice) {
  try {
    return mscclpp::lite::selectIBTransportForGpu(cudaDevice);
  } catch (...) {
    return mscclpp::Transport::Unknown;
  }
}

static void failDeviceCollectiveContext(DeviceCollectiveContext& c) {
  for (auto& group : c.groups) failDeviceCollectiveContext(*group);
  if (c.tasks) __atomic_store_n(&c.tasks->error, uint64_t{1}, __ATOMIC_RELEASE);
  for (int slot = 0; slot < MSCCLPP_DEVICE_COLLECTIVE_SLOTS; ++slot) {
    if (c.hostBuffer) {
      for (int chunk = 0; chunk < MSCCLPP_DEVICE_COLLECTIVE_MAX_CHUNKS; ++chunk)
        c.hostBuffer->ctrl()->d2hReady[slot][chunk][c.rank].value.store(
            UINT64_MAX, std::memory_order_release);
      c.hostBuffer->ctrl()->slotDone[slot][c.rank].value.store(
          UINT64_MAX, std::memory_order_release);
    } else if (c.nodeBuffer) {
      __atomic_store_n(&c.nodeBuffer->control()->gpuReady[slot][c.localRank],
                       UINT64_MAX, __ATOMIC_RELEASE);
      __atomic_store_n(&c.nodeBuffer->control()->gpuDone[slot][c.localRank],
                       UINT64_MAX, __ATOMIC_RELEASE);
    }
  }
}

static bool waitDeviceCollectiveEpoch(uint64_t const volatile* value,
                                      uint64_t epoch,
                                      std::atomic<bool> const& stop) {
  int spins = 0;
  for (;;) {
    if (stop.load(std::memory_order_acquire)) return false;
    uint64_t observed = __atomic_load_n(value, __ATOMIC_ACQUIRE);
    if (observed == UINT64_MAX)
      throw mscclpp::Error("device collective participant failed",
                           mscclpp::ErrorCode::SystemError);
    if (observed >= epoch) break;
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
  try {
    mscclpp::CudaDeviceGuard guard(context->cudaDevice);
    NebCtrl* control = context->nodeBuffer->control();
    size_t slotStride = static_cast<size_t>(2 * context->groupSize) *
                        (context->maxBytesPerRank + 16);
    for (uint64_t epoch = 1;
         !context->stopRdmaProxy.load(std::memory_order_acquire); ++epoch) {
      int slot =
          static_cast<int>((epoch - 1) % MSCCLPP_DEVICE_COLLECTIVE_SLOTS);
      if (epoch > MSCCLPP_DEVICE_COLLECTIVE_SLOTS &&
          !waitDeviceCollectiveEpoch(&control->remoteAck[slot],
                                     epoch - MSCCLPP_DEVICE_COLLECTIVE_SLOTS,
                                     context->stopRdmaProxy)) {
        return;
      }
      for (int local = context->groupBase;
           local < context->groupBase + context->groupSize; ++local) {
        if (!waitDeviceCollectiveEpoch(&control->gpuReady[slot][local], epoch,
                                       context->stopRdmaProxy)) {
          return;
        }
      }

      size_t slotOffset = static_cast<size_t>(slot) * slotStride;
      uint64_t descriptor = control->gpuBytes[slot][context->groupBase];
      size_t bytes = static_cast<size_t>(descriptor & ~kLiteNetworkMask);
      if (!bytes || bytes > context->maxBytesPerRank)
        throw mscclpp::Error("invalid device collective chunk size",
                             mscclpp::ErrorCode::InvalidUsage);
      for (int local = context->groupBase;
           local < context->groupBase + context->groupSize; ++local)
        if (control->gpuBytes[slot][local] != descriptor)
          throw mscclpp::Error("inconsistent device collective chunk sizes",
                               mscclpp::ErrorCode::InvalidUsage);
      bool packed = descriptor & kLitePackedNetworkBit;
      bool compact = descriptor & kLiteCompactNetworkBit;
      size_t row = packed ? bytes : context->maxBytesPerRank;
      if (compact) row = ((bytes + 7) & ~size_t{7}) + 8;
      size_t nodeOffset =
          static_cast<size_t>(context->rank / context->nRanksPerNode) *
          context->groupSize * row;
      size_t blockBytes = bytes * context->groupSize;
      bool striped = context->dualRail &&
                     (descriptor & kLiteGenericNetworkBit) &&
                     blockBytes >= 2 * 1024 * 1024;
      size_t split = blockBytes / 2;
      unsigned outstanding[2]{};
      // Packed AllGather is one contiguous node/group block. Reduction rows
      // remain capacity-strided, but the same ownership protocol applies.
      size_t transfers = packed ? 1 : context->groupSize;
      for (size_t local = 0; local < transfers; ++local) {
        size_t length = packed ? blockBytes : bytes;
        size_t rowOffset = slotOffset + nodeOffset + local * row;
        for (size_t offset = 0; offset < length;) {
          size_t logicalOffset = local * bytes + offset;
          bool second = striped && logicalOffset >= split;
          size_t segment = std::min(length - offset, size_t{2 * 1024 * 1024});
          if (striped && !second)
            segment = std::min(segment, split - logicalOffset);
          auto& connection =
              second ? context->rail2Connection : context->rdmaConnection;
          connection.write(
              second ? context->rail2RemoteMemory
                     : context->remoteRdmaRecvMemory,
              rowOffset + offset,
              second ? context->rail2SendMemory : context->rdmaSendMemory,
              rowOffset + offset, segment);
          if (++outstanding[second ? 1 : 0] == 8) {
            connection.flush();
            outstanding[second ? 1 : 0] = 0;
          }
          offset += segment;
        }
      }
      // The readiness write covers BOTH rails. A completion on the primary
      // QP alone cannot publish payload still in flight on the secondary QP.
      if (striped) context->rail2Connection.flush();
      context->rdmaConnection.flush();
      control->rxSignal[slot] = epoch;
      if (compact) {
        // Payload and its in-segment flag use the same ordered QP. The flag
        // has its own aligned word even for odd-sized payloads.
        context->rdmaConnection.write(
            context->remoteRdmaRecvMemory, slotOffset + nodeOffset + row - 8,
            context->rdmaControlMemory,
            offsetof(NebCtrl, rxSignal) + slot * sizeof(uint64_t),
            sizeof(uint64_t));
      }
      context->rdmaConnection.write(
          context->remoteRdmaControlMemory,
          offsetof(NebCtrl, rxReady) +
              static_cast<size_t>(slot) * sizeof(uint64_t),
          context->rdmaControlMemory,
          offsetof(NebCtrl, rxSignal) +
              static_cast<size_t>(slot) * sizeof(uint64_t),
          sizeof(uint64_t));
      context->rdmaConnection.flush();
      std::atomic_thread_fence(std::memory_order_release);
      __atomic_store_n(&control->slotReusable[slot], epoch, __ATOMIC_RELEASE);

      for (int local = 0; local < context->nRanksPerNode; ++local) {
        if (!waitDeviceCollectiveEpoch(&control->gpuDone[slot][local], epoch,
                                       context->stopRdmaProxy)) {
          return;
        }
      }
      control->ackSignal[slot] = epoch;
      context->rdmaConnection.write(
          context->remoteRdmaControlMemory,
          offsetof(NebCtrl, remoteAck) +
              static_cast<size_t>(slot) * sizeof(uint64_t),
          context->rdmaControlMemory,
          offsetof(NebCtrl, ackSignal) +
              static_cast<size_t>(slot) * sizeof(uint64_t),
          sizeof(uint64_t));
      context->rdmaConnection.flush();
    }
  } catch (...) {
    failDeviceCollectiveContext(*context);
    // Best-effort error propagation on the established QP. If the transport
    // itself is broken, peers also have finite device wait deadlines.
    try {
      auto* control = context->nodeBuffer->control();
      for (int slot = 0; slot < MSCCLPP_DEVICE_COLLECTIVE_SLOTS; ++slot) {
        control->rxSignal[slot] = UINT64_MAX;
        context->rdmaConnection.write(
            context->remoteRdmaControlMemory,
            offsetof(NebCtrl, rxReady) + slot * sizeof(uint64_t),
            context->rdmaControlMemory,
            offsetof(NebCtrl, rxSignal) + slot * sizeof(uint64_t),
            sizeof(uint64_t));
      }
      context->rdmaConnection.flush();
    } catch (...) {
    }
    context->stopRdmaProxy.store(true, std::memory_order_release);
  }
}

static void initializeCudaIpcDeviceCollective(DeviceCollectiveContext& context,
                                              ncclComm_t comm,
                                              size_t alignedMaxBytesPerRank) {
  size_t slabBytes = alignedMaxBytesPerRank * MSCCLPP_DEVICE_COLLECTIVE_SLOTS;
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
      MSCCLPP_CUDATHROW(cudaIpcOpenMemHandle(&slab, infos[r].slab,
                                             cudaIpcMemLazyEnablePeerAccess));
      try {
        MSCCLPP_CUDATHROW(cudaIpcOpenMemHandle(&control, infos[r].control,
                                               cudaIpcMemLazyEnablePeerAccess));
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

static void initializeHostRdmaDeviceCollective(DeviceCollectiveContext& context,
                                               ncclComm_t comm) {
  context.localRank = context.rank % context.nRanksPerNode;
  int nodeId = context.rank / context.nRanksPerNode;
  if (!context.groupSize) context.groupSize = context.nRanksPerNode;
  context.localLeader = nodeId * context.nRanksPerNode + context.groupBase;
  context.remoteLeader =
      (1 - nodeId) * context.nRanksPerNode + context.groupBase;
  bool isLeader = context.rank == context.localLeader;
  size_t slotStride = static_cast<size_t>(2 * context.groupSize) *
                      (context.maxBytesPerRank + 16);
  size_t slabBytes = MSCCLPP_DEVICE_COLLECTIVE_SLOTS * slotStride;
  auto nonce =
      static_cast<unsigned long long>(reinterpret_cast<uintptr_t>(comm));
  char nameTag[80];
  std::snprintf(nameTag, sizeof(nameTag), "gpu_rdma_%llx_%d_%d_g%d", nonce,
                getpid(), context.localLeader, context.groupSize);
  int numaNode = context.groupNuma;
  context.nodeBuffer = std::make_unique<NodeExchangeBuffer>(
      NodeExchangeBuffer::create(comm->comm, context.rank, context.nranks,
                                 isLeader, context.localLeader, slabBytes,
                                 numaNode, context.cudaDevice, nameTag));
  std::vector<int> controlStatus(context.nranks);
  controlStatus[context.rank] =
      context.nodeBuffer->controlDevicePtr() == nullptr;
  comm->comm->bootstrap()->allGather(controlStatus.data(), sizeof(int));
  if (std::any_of(controlStatus.begin(), controlStatus.end(),
                  [](int s) { return s != 0; })) {
    throw mscclpp::Error(
        "GPU-driven inter-node collectives require mapped control memory",
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
          context.nodeBuffer->sendPtr(), slabBytes, flags);
      context.rdmaControlMemory = comm->comm->registerMemory(
          context.nodeBuffer->control(), sizeof(NebCtrl), flags);

      mscclpp::EndpointConfig::Ib ibConfig;
      ibConfig.maxCqPollNum = 128;
      mscclpp::EndpointConfig endpointConfig(
          transport, mscclpp::Device(mscclpp::DeviceType::CPU),
          /*maxWriteQueueSize=*/-1, ibConfig);
      int pair =
          std::min(context.localLeader, context.remoteLeader) * context.nranks +
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
      if (context.dualRail) {
        auto transports = mscclpp::lite::getAvailableIBTransports();
        auto other = std::find_if(transports.begin(), transports.end(),
                                  [&](auto t) { return t != transport; });
        if (other == transports.end())
          throw mscclpp::Error(
              "second IB rail disappeared during initialization",
              mscclpp::ErrorCode::InvalidUsage);
        mscclpp::TransportFlags secondFlags(*other);
        context.rail2SendMemory = comm->comm->registerMemory(
            context.nodeBuffer->sendPtr(), slabBytes, secondFlags);
        context.rail2RecvMemory = comm->comm->registerMemory(
            context.nodeBuffer->sendPtr(), slabBytes, secondFlags);
        mscclpp::EndpointConfig secondConfig(
            *other, mscclpp::Device(mscclpp::DeviceType::CPU), -1, ibConfig);
        auto secondFuture = comm->comm->connect(
            secondConfig, context.remoteLeader, tagBase + 3);
        comm->comm->sendMemory(context.rail2RecvMemory, context.remoteLeader,
                               tagBase + 4);
        auto memoryFuture =
            comm->comm->recvMemory(context.remoteLeader, tagBase + 4);
        context.rail2Connection = secondFuture.get();
        context.rail2RemoteMemory = memoryFuture.get();
      }
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
      throw mscclpp::Error("GPU-driven RDMA setup failed on rank " +
                               std::to_string(r) + ": " + statuses[r].message,
                           mscclpp::ErrorCode::SystemError);
    }
  }
}

#include "service.hpp"

static void fillDeviceCollectiveHandle(
    const DeviceCollectiveContext& context,
    mscclppDeviceCollectiveHandle_t* handle) {
  *handle = {};
  handle->localEpoch = context.localEpoch;
  handle->maxBytesPerRank = context.maxBytesPerRank;
  handle->rank = context.rank;
  handle->nranks = context.nranks;
  handle->backend = context.backend;
  handle->ranksPerNode = context.nRanksPerNode;
  handle->reductionsMapped = context.reductionsMapped;
  handle->allGatherPolicy = context.allGatherPolicy;
  handle->tasks = context.tasksDevice;
  handle->timeoutCycles = context.timeoutCycles;
  handle->numaHandle = context.numaDeviceHandle;
  if (context.nranks == 1) return;

  if (context.backend == mscclppDeviceCollectiveHostMemory) {
    CscDeviceHandle raw = context.hostBuffer->deviceHandle();
    if (raw.ctrlDev == nullptr) {
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
    int count =
        context.groups.empty() ? 1 : static_cast<int>(context.groups.size());
    handle->groupCount = count;
    for (int g = 0; g < count; ++g) {
      auto const& group = context.groups.empty() ? context : *context.groups[g];
      size_t slotStride = static_cast<size_t>(2 * group.groupSize) *
                          (group.maxBytesPerRank + 16);
      char* send = const_cast<char*>(group.nodeBuffer->sendDevicePtr());
      char* control = group.nodeBuffer->controlDevicePtr();
      handle->groupReusable[g] = reinterpret_cast<unsigned long long*>(
          control + offsetof(NebCtrl, slotReusable));
      handle->groupDone[g] = reinterpret_cast<unsigned long long*>(
                                 control + offsetof(NebCtrl, gpuDone)) +
                             context.rank % context.nRanksPerNode;
      for (int local = group.groupBase;
           local < group.groupBase + group.groupSize; ++local) {
        int localGlobal = nodeId * context.nRanksPerNode + local;
        int remoteGlobal = (1 - nodeId) * context.nRanksPerNode + local;
        for (int r : {localGlobal, remoteGlobal}) {
          handle->peerSlabs[r] = send;
          handle->peerSlabSlotStride[r] = slotStride;
          handle->peerSlabIndex[r] =
              (r / context.nRanksPerNode) * group.groupSize + local -
              group.groupBase;
          handle->peerDone[r] = reinterpret_cast<unsigned long long*>(
                                    control + offsetof(NebCtrl, gpuDone)) +
                                local;
          handle->peerDoneSlotStride[r] = kNebMaxRanks;
        }
        handle->peerReady[localGlobal] =
            reinterpret_cast<unsigned long long*>(control +
                                                  offsetof(NebCtrl, gpuReady)) +
            local;
        handle->peerReady[remoteGlobal] = reinterpret_cast<unsigned long long*>(
            control + offsetof(NebCtrl, rxReady));
        handle->peerReadySlotStride[localGlobal] = kNebMaxRanks;
        handle->peerReadySlotStride[remoteGlobal] = 1;
        if (localGlobal == context.rank) {
          handle->publishedBytes = reinterpret_cast<unsigned long long*>(
              control + offsetof(NebCtrl, gpuBytes));
          handle->publishedBytesSlotStride = kNebMaxRanks;
          handle->slotReusable = handle->groupReusable[g];
          handle->slotStride = slotStride;
        }
      }
    }
    return;
  }

  for (int r = 0; r < context.nranks; ++r) {
    handle->peerSlabs[r] = context.peerIpcSlabs[r];
    handle->peerReady[r] = context.peerIpcControls[r];
    handle->peerDone[r] =
        context.peerIpcControls[r] +
        MSCCLPP_DEVICE_COLLECTIVE_SLOTS * MSCCLPP_DEVICE_COLLECTIVE_MAX_CHUNKS;
    handle->peerReadySlotStride[r] = MSCCLPP_DEVICE_COLLECTIVE_MAX_CHUNKS;
    handle->peerDoneSlotStride[r] = 1;
  }
}

// Discover actual per-rank NUMA identities on their owning hosts. Device
// ordinals from a remote node must never be queried against the local driver.
static void initializeNumaDeviceCollective(
    DeviceCollectiveContext& parent, ncclComm_t comm,
    std::vector<DeviceCollectiveConfig> const& configs) {
  int p = parent.nRanksPerNode;
  std::vector<int> boundaries[2];
  int groupLimit = p;
  for (auto const& config : configs)
    groupLimit = std::min(groupLimit, config.ibCount);
  if (groupLimit < 2) return;
  for (int node = 0; node < 2; ++node) {
    boundaries[node].push_back(0);
    int previous = configs[node * p].numaNode;
    for (int local = 1; local < p; ++local) {
      int numa = configs[node * p + local].numaNode;
      if (numa != previous &&
          static_cast<int>(boundaries[node].size()) < groupLimit) {
        boundaries[node].push_back(local);
        previous = numa;
      }
    }
    boundaries[node].push_back(p);
  }
  if (boundaries[0] != boundaries[1] || boundaries[0].size() <= 2) return;
  parent.numaContext = std::make_unique<DeviceCollectiveContext>();
  auto& c = *parent.numaContext;
  c.rank = parent.rank;
  c.nranks = parent.nranks;
  c.cudaDevice = parent.cudaDevice;
  c.nRanksPerNode = p;
  c.localRank = c.rank % p;
  c.backend = mscclppDeviceCollectiveHostRdma;
  c.maxBytesPerRank = parent.maxBytesPerRank;
  c.allGatherPolicy = parent.allGatherPolicy;
  c.timeoutCycles = parent.timeoutCycles;
  std::vector<int> statuses(c.nranks);
  try {
    MSCCLPP_CUDATHROW(cudaMalloc(&c.localEpoch, sizeof(*c.localEpoch)));
    MSCCLPP_CUDATHROW(cudaMemset(c.localEpoch, 0, sizeof(*c.localEpoch)));
  } catch (...) {
    statuses[c.rank] = 1;
  }
  comm->comm->bootstrap()->allGather(statuses.data(), sizeof(int));
  if (std::any_of(statuses.begin(), statuses.end(),
                  [](int s) { return s != 0; }))
    throw mscclpp::Error("NUMA epoch allocation failed",
                         mscclpp::ErrorCode::SystemError);
  bool mapped = true;
  for (size_t g = 0; g + 1 < boundaries[0].size(); ++g) {
    auto group = std::make_unique<DeviceCollectiveContext>();
    group->rank = c.rank;
    group->nranks = c.nranks;
    group->cudaDevice = c.cudaDevice;
    group->nRanksPerNode = p;
    group->maxBytesPerRank =
        std::min(c.maxBytesPerRank, size_t{16 * 1024 * 1024});
    group->backend = c.backend;
    group->groupBase = boundaries[0][g];
    group->groupSize = boundaries[0][g + 1] - group->groupBase;
    group->groupNuma = configs[(c.rank / p) * p + group->groupBase].numaNode;
    initializeHostRdmaDeviceCollective(*group, comm);
    mapped = mapped && group->nodeBuffer->sendDevicePtr();
    c.groups.push_back(std::move(group));
  }
  statuses.assign(c.nranks, 0);
  statuses[c.rank] = mapped;
  comm->comm->bootstrap()->allGather(statuses.data(), sizeof(int));
  c.reductionsMapped = std::all_of(statuses.begin(), statuses.end(),
                                   [](int s) { return s != 0; });
  initializeDeviceCollectiveService(c, comm);
  mscclppDeviceCollectiveHandle_t handle{};
  fillDeviceCollectiveHandle(c, &handle);
  statuses.assign(c.nranks, 0);
  try {
    MSCCLPP_CUDATHROW(cudaMalloc(&parent.numaDeviceHandle, sizeof(handle)));
    MSCCLPP_CUDATHROW(cudaMemcpy(parent.numaDeviceHandle, &handle,
                                 sizeof(handle), cudaMemcpyHostToDevice));
  } catch (...) {
    statuses[c.rank] = 1;
  }
  comm->comm->bootstrap()->allGather(statuses.data(), sizeof(int));
  if (std::any_of(statuses.begin(), statuses.end(),
                  [](int s) { return s != 0; }))
    throw mscclpp::Error("NUMA device handle allocation failed",
                         mscclpp::ErrorCode::SystemError);
}

}  // namespace

NCCL_API ncclResult_t
mscclppGetDeviceCollectiveHandle(ncclComm_t comm, size_t maxBytesPerRank,
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
  if (nranks < 1 || nranks > MSCCLPP_DEVICE_COLLECTIVE_MAX_RANKS ||
      (!singleNode && !twoNodes) ||
      (twoNodes && backend == mscclppDeviceCollectiveCudaIpc)) {
    return ncclInvalidUsage;
  }
  if (maxBytesPerRank >= kLiteCompactNetworkBit ||
      maxBytesPerRank > std::numeric_limits<size_t>::max() - 31 ||
      (((maxBytesPerRank + 15) & ~size_t{15}) + 16) >
          std::numeric_limits<size_t>::max() /
              (MSCCLPP_DEVICE_COLLECTIVE_SLOTS *
               static_cast<size_t>(std::max(1, nranks)))) {
    return ncclInvalidArgument;
  }

  return runNcclGuarded("device collective handle initialization", [&]() {
    mscclpp::CudaDeviceGuard deviceGuard(comm->cudaDevice);
    std::vector<DeviceCollectiveConfig> configs(nranks);
    LiteAllGatherPolicy policy;
    auto enabled = [](char const* key, bool fallback) {
      char const* value = std::getenv(key);
      return value ? std::strcmp(value, "0") != 0 : fallback;
    };
    policy.hostEnabled = enabled("MSCCLPP_NCCL_HOST_ALLGATHER", false);
    policy.ipcEventSync = enabled("MSCCLPP_NCCL_CUDAIPC_EVENT_SYNC", true);
    policy.mapSlab = hostAllGatherMapSlabEnabled();
    policy.minBytes = hostAllGatherMinTotalBytes();
    policy.kernelMaxBytes = hostAllGatherKernelMaxBytes();
    policy.coopMaxBytes = hostAllGatherCoopMaxBytes();
    if (char const* value =
            std::getenv("MSCCLPP_NCCL_HOST_ALLGATHER_CHUNK_BYTES")) {
      char* end = nullptr;
      policy.chunkBytes = std::strtoull(value, &end, 10);
      // The host reference treats malformed/sub-minimum overrides as B.
      if (end == value || !policy.chunkBytes) policy.chunkBytes = 1;
    }
    cudaDeviceProp properties{};
    MSCCLPP_CUDATHROW(cudaGetDeviceProperties(&properties, comm->cudaDevice));
    policy.cooperative = properties.cooperativeLaunch;
    configs[rank] = {maxBytesPerRank,
                     static_cast<int>(backend),
                     comm->cudaDevice,
                     policy,
                     0,
                     -1};
    try {
      configs[rank].numaNode = mscclpp::getDeviceNumaNode(comm->cudaDevice);
    } catch (...) {
    }
    if (twoNodes) {
      try {
        configs[rank].ibCount =
            static_cast<int>(mscclpp::lite::getAvailableIBTransports().size());
      } catch (...) {
      }
    }
    comm->comm->bootstrap()->allGather(configs.data(), sizeof(configs[0]));
    for (auto const& config : configs) {
      if (config.maxBytesPerRank != maxBytesPerRank ||
          config.backend != static_cast<int>(backend) ||
          config.policy.hostEnabled != policy.hostEnabled ||
          config.policy.ipcEventSync != policy.ipcEventSync ||
          config.policy.mapSlab != policy.mapSlab ||
          config.policy.minBytes != policy.minBytes ||
          config.policy.kernelMaxBytes != policy.kernelMaxBytes ||
          config.policy.coopMaxBytes != policy.coopMaxBytes ||
          config.policy.chunkBytes != policy.chunkBytes) {
        throw mscclpp::Error(
            "all ranks must initialize the same device collective backend "
            "and maxBytesPerRank",
            mscclpp::ErrorCode::InvalidUsage);
      }
    }
    for (auto const& config : configs)
      policy.cooperative &= config.policy.cooperative;
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
      context->allGatherPolicy = policy;
      context->dualRail = twoNodes && configs[0].ibCount > 1 &&
                          configs[nRanksPerNode].ibCount > 1;
      context->timeoutCycles =
          static_cast<unsigned long long>(properties.clockRate) * 1000ULL *
          30ULL;
      context->stopRdmaProxy.store(false, std::memory_order_release);
      context->rank = rank;
      context->nranks = nranks;
      context->cudaDevice = comm->cudaDevice;
      context->nRanksPerNode = nRanksPerNode;
      context->groupNuma =
          configs[(rank / nRanksPerNode) * nRanksPerNode].numaNode;
      context->backend = twoNodes ? mscclppDeviceCollectiveHostRdma : backend;
      try {
        std::vector<int> allocationStatus(nranks);
        try {
          MSCCLPP_CUDATHROW(
              cudaMalloc(&context->localEpoch, sizeof(*context->localEpoch)));
          MSCCLPP_CUDATHROW(
              cudaMemset(context->localEpoch, 0, sizeof(*context->localEpoch)));
        } catch (...) {
          allocationStatus[rank] = 1;
        }
        comm->comm->bootstrap()->allGather(allocationStatus.data(),
                                           sizeof(int));
        if (std::any_of(allocationStatus.begin(), allocationStatus.end(),
                        [](int s) { return s != 0; }))
          throw mscclpp::Error("device epoch allocation failed",
                               mscclpp::ErrorCode::SystemError);

        if (nranks == 1) {
          // Direct copies need only the capacity/device contract, no SHM/IPC.
        } else if (twoNodes) {
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
          context->hostBuffer =
              std::make_unique<HostStagingBuffer>(HostStagingBuffer::create(
                  context->maxBytesPerRank, MSCCLPP_DEVICE_COLLECTIVE_SLOTS,
                  comm->comm, rank, nranks, comm->cudaDevice,
                  /*mapSlab=*/policy.mapSlab != 0,
                  hostStagingNumaPlacementEnabled(), tagBuffer));
        }
        std::vector<int> controlStatus(nranks);
        controlStatus[rank] =
            context->hostBuffer && !context->hostBuffer->deviceHandle().ctrlDev;
        comm->comm->bootstrap()->allGather(controlStatus.data(), sizeof(int));
        if (std::any_of(controlStatus.begin(), controlStatus.end(),
                        [](int s) { return s != 0; }))
          throw mscclpp::Error(
              "device collectives require mapped control on every rank",
              mscclpp::ErrorCode::InvalidUsage);
        std::vector<int> mappedStatus(nranks);
        mappedStatus[rank] =
            nranks == 1 || backend == mscclppDeviceCollectiveCudaIpc ||
            (context->hostBuffer && context->hostBuffer->hasSlabDev()) ||
            (context->nodeBuffer && context->nodeBuffer->sendDevicePtr());
        comm->comm->bootstrap()->allGather(mappedStatus.data(), sizeof(int));
        context->reductionsMapped =
            std::all_of(mappedStatus.begin(), mappedStatus.end(),
                        [](int s) { return s != 0; });
        initializeDeviceCollectiveService(*context, comm);
        if (twoNodes && nRanksPerNode > 2)
          initializeNumaDeviceCollective(*context, comm, configs);
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
