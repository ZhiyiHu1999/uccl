// Included inside host_context.hpp's implementation namespace.
#pragma once

static DeviceCollectiveContext& deviceCollectiveGroup(
    DeviceCollectiveContext& c, int rank) {
  for (auto& group : c.groups)
    if (rank % c.nRanksPerNode >= group->groupBase &&
        rank % c.nRanksPerNode < group->groupBase + group->groupSize)
      return *group;
  return c;
}

static char* deviceCollectiveHostSlab(DeviceCollectiveContext& c, int slot,
                                      int rank, LiteTask const& task) {
  if (c.hostBuffer) return c.hostBuffer->rankSlabHost(slot, rank);
  auto& group = deviceCollectiveGroup(c, rank);
  size_t row = task.networkFlags & kLitePackedNetworkBit ? task.bytes
                                                         : c.maxBytesPerRank;
  if (task.networkFlags & kLiteCompactNetworkBit)
    row = ((row + 7) & ~size_t{7}) + 8;
  size_t slotStride =
      static_cast<size_t>(2 * group.groupSize) * (group.maxBytesPerRank + 16);
  size_t index = (rank / c.nRanksPerNode) * group.groupSize +
                 rank % c.nRanksPerNode - group.groupBase;
  return group.nodeBuffer->sendPtr(slot * slotStride + index * row);
}

static void deviceCollectivePublishStage(DeviceCollectiveContext& c,
                                         LiteTask const& task) {
  if (c.hostBuffer) {
    c.hostBuffer->ctrl()->d2hReady[task.slot][0][c.rank].value.store(
        task.epoch, std::memory_order_release);
  } else {
    auto* control = deviceCollectiveGroup(c, c.rank).nodeBuffer->control();
    control->gpuBytes[task.slot][c.localRank] = task.bytes | task.networkFlags;
    __atomic_store_n(&control->gpuReady[task.slot][c.localRank], task.epoch,
                     __ATOMIC_RELEASE);
  }
}

// Whole-call DMA submission mirrors runIntraNodeShmAllGather. Capacity-strided
// host rows need a distinct destination pitch when benchmarking smaller sizes.
static void enqueueDeviceHostAllGather(DeviceCollectiveContext& c,
                                      LiteTask const& task, unsigned index) {
  if (!c.hostBuffer || !task.source || !task.destination || !task.chunkBytes ||
      task.chunkBytes > task.bytes ||
      task.bytes / task.chunkBytes + (task.bytes % task.chunkBytes != 0) >
          MSCCLPP_DEVICE_COLLECTIVE_MAX_CHUNKS)
    throw mscclpp::Error("invalid host AllGather descriptor",
                         mscclpp::ErrorCode::InvalidUsage);
  auto& buf = *c.hostBuffer;
  size_t chunks = task.bytes / task.chunkBytes + (task.bytes % task.chunkBytes != 0);
  for (size_t ci = 0; ci < chunks; ++ci) {
    size_t offset = ci * task.chunkBytes;
    size_t bytes = std::min(task.chunkBytes, task.bytes - offset);
    // Stream-ordered publication, after the DMA engine writes this chunk.
    buf.put(c.serviceStreams[0], task.slot, static_cast<int>(ci),
            reinterpret_cast<void const*>(task.source), offset, bytes, task.epoch);
  }
  char* output = reinterpret_cast<char*>(task.destination);
  char* self = output + c.rank * task.bytes;
  if (!task.selfCopyWithSm && reinterpret_cast<char const*>(task.source) != self)
    MSCCLPP_CUDATHROW(cudaMemcpyAsync(
        self, reinterpret_cast<void const*>(task.source), task.bytes,
        cudaMemcpyDeviceToDevice, c.serviceStreams[3]));
  for (size_t ci = 0; ci < chunks; ++ci) {
    size_t offset = ci * task.chunkBytes;
    size_t bytes = std::min(task.chunkBytes, task.bytes - offset);
    for (int side = 0; side < 2; ++side) {
      int first = side == 0 ? 0 : c.rank + 1;
      int last = side == 0 ? c.rank - 1 : c.nranks - 1;
      if (first > last) continue;
      cudaStream_t stream = c.serviceStreams[side == 0 || c.rank == 0 ? 1 : 2];
      for (int r = first; r <= last; ++r)
        buf.wait(stream, task.slot, static_cast<int>(ci), r, task.epoch);
      size_t rows = static_cast<size_t>(last - first + 1);
      char* source = buf.rankSlabHost(task.slot, first) + offset;
      char* destination = output + first * task.bytes + offset;
      if (bytes == buf.bytesPerRank() && bytes == task.bytes) {
        MSCCLPP_CUDATHROW(cudaMemcpyAsync(destination, source, rows * bytes,
                                          cudaMemcpyHostToDevice, stream));
      } else {
        MSCCLPP_CUDATHROW(cudaMemcpy2DAsync(
            destination, task.bytes, source, buf.bytesPerRank(), bytes, rows,
            cudaMemcpyHostToDevice, stream));
      }
    }
  }
  // Completion covers all copies, not just enqueue or descriptor consumption.
  for (int stream = 0; stream < 4; ++stream)
    MSCCLPP_CUDATHROW(cudaEventRecord(c.serviceEvents[index][stream],
                                      c.serviceStreams[stream]));
}

static void runDeviceCollectiveService(DeviceCollectiveContext* context) {
  auto& c = *context;
  struct Pending {
    uint64_t ticket = 0;
    LiteTask task{};
    int eventCount = 1;
  };
  Pending pending[kLiteTaskSlots]{};
  uint64_t next = 0;
  uint64_t retired = 0;
  try {
    mscclpp::CudaDeviceGuard guard(c.cudaDevice);
    while (!c.stopRdmaProxy.load(std::memory_order_acquire)) {
      if (__atomic_load_n(&c.tasks->error, __ATOMIC_ACQUIRE)) {
        failDeviceCollectiveContext(c);
        break;
      }
      bool progress = false;
      uint64_t submitted =
          __atomic_load_n(&c.tasks->submitted, __ATOMIC_ACQUIRE);
      while (next < submitted) {
        unsigned index = next % kLiteTaskSlots;
        auto& item = pending[index];
        if (item.ticket) break;
        item.task = c.tasks->slots[index].task;
        auto const& task = item.task;
        if (task.kind == LiteTaskKind::RsCopy) {
          if (!c.reduceScatter || task.rs.stream < 0 || task.rs.stream > 3)
            throw mscclpp::Error("invalid RsCopy",
                                 mscclpp::ErrorCode::InvalidUsage);
          enqueueLiteRsCopy(*c.reduceScatter, c.serviceStreams[task.rs.stream],
                            task);
          MSCCLPP_CUDATHROW(cudaEventRecord(c.serviceEvents[index][0],
                                            c.serviceStreams[task.rs.stream]));
          item.eventCount = 1;
          item.ticket = ++next;
          progress = true;
          continue;
        }
        if (task.kind == LiteTaskKind::RsHostSum ||
            task.kind == LiteTaskKind::RsBarrier) {
          if (!c.reduceScatter)
            throw mscclpp::Error("ReduceScatter was not prepared",
                                 mscclpp::ErrorCode::InvalidUsage);
          if (task.kind == LiteTaskKind::RsHostSum)
            executeLiteRsHostSum(*c.reduceScatter, task);
          else
            executeLiteRsBarrier(*c.reduceScatter, task);
          ++next;
          __atomic_store_n(&c.tasks->slots[index].completed, next,
                           __ATOMIC_RELEASE);
          progress = true;
          continue;
        }
        if (task.kind == LiteTaskKind::AllReduce) {
          if (!c.reduceScatter)
            throw mscclpp::Error("AllReduce was not prepared",
                                 mscclpp::ErrorCode::InvalidUsage);
          executeLiteAllReduce(*c.reduceScatter, task);
          ++next;
          __atomic_store_n(&c.tasks->slots[index].completed, next,
                           __ATOMIC_RELEASE);
          progress = true;
          continue;
        }
        if (task.kind == LiteTaskKind::ReduceScatter) {
          if (!c.reduceScatter)
            throw mscclpp::Error("ReduceScatter was not prepared", mscclpp::ErrorCode::InvalidUsage);
          executeLiteReduceScatter(*c.reduceScatter, task);
          ++next;
          __atomic_store_n(&c.tasks->slots[index].completed, next, __ATOMIC_RELEASE);
          progress = true;
          continue;
        }
        if (task.kind == LiteTaskKind::NetworkAllGather) {
          if (!c.nativeAllGatherNetwork)
            throw mscclpp::Error("network AllGather was not prepared",
                                 mscclpp::ErrorCode::InvalidUsage);
          mscclpp::nccl::executeDeviceAllGatherNetwork(
              c.nativeAllGatherNetwork, task, next + 1, &c.tasks->network,
              c.serviceStreams[3]);
          if (__atomic_load_n(&c.tasks->error, __ATOMIC_ACQUIRE))
            throw mscclpp::Error("device AllGather failed",
                                 mscclpp::ErrorCode::SystemError);
          ++next;
          __atomic_store_n(&c.tasks->slots[index].completed, next, __ATOMIC_RELEASE);
          progress = true;
          continue;
        }
        if (task.slot < 0 || task.slot >= MSCCLPP_DEVICE_COLLECTIVE_SLOTS ||
            !task.bytes || task.bytes > c.maxBytesPerRank || !task.epoch ||
            (!c.groups.empty() &&
             task.bytes > deviceCollectiveGroup(c, c.rank).maxBytesPerRank)) {
          throw mscclpp::Error("invalid device DMA descriptor",
                               mscclpp::ErrorCode::InvalidUsage);
        }
        item.eventCount = 1;
        if (task.kind == LiteTaskKind::HostAllGather) {
          enqueueDeviceHostAllGather(c, task, index);
          item.eventCount = 4;
        } else if (task.kind == LiteTaskKind::IpcCopySelf ||
            task.kind == LiteTaskKind::IpcPush) {
          uintptr_t base = reinterpret_cast<uintptr_t>(c.ipcOutput);
          if (c.backend != mscclppDeviceCollectiveCudaIpc || !base ||
              !c.ipcNextOutput || task.destination < base ||
              task.destination - base > c.ipcOutputBytes ||
              task.bytes > c.ipcOutputBytes - (task.destination - base))
            throw mscclpp::Error("IPC output is not registered or copy is out of bounds",
                                 mscclpp::ErrorCode::InvalidUsage);
          char* destination = task.kind == LiteTaskKind::IpcPush
              ? c.ipcNextOutput + (task.destination - base)
              : reinterpret_cast<char*>(task.destination);
          MSCCLPP_CUDATHROW(cudaMemcpyAsync(
              destination, reinterpret_cast<void const*>(task.source),
              task.bytes, cudaMemcpyDeviceToDevice, c.serviceStreams[0]));
          MSCCLPP_CUDATHROW(
              cudaEventRecord(c.serviceEvents[index][0], c.serviceStreams[0]));
        } else if (task.kind == LiteTaskKind::Stage) {
          MSCCLPP_CUDATHROW(cudaMemcpyAsync(
              deviceCollectiveHostSlab(c, task.slot, c.rank, task),
              reinterpret_cast<void const*>(task.source), task.bytes,
              cudaMemcpyDeviceToHost, c.serviceStreams[0]));
          MSCCLPP_CUDATHROW(
              cudaEventRecord(c.serviceEvents[index][0], c.serviceStreams[0]));
        } else if (task.kind == LiteTaskKind::GatherPacked) {
          if (!c.smallOutput || task.outputOffset ||
              task.bytes != task.outputStride ||
              task.bytes > size_t{2 * 1024 * 1024} / c.nranks)
            throw mscclpp::Error("invalid ordered small receive",
                                 mscclpp::ErrorCode::InvalidUsage);
          // The ordered slot is already in final rank order. The explicit
          // fallback retains CPU repacking before a single full-output H2D.
          char* source = deviceCollectiveHostSlab(c, task.slot, 0, task);
          if (!(task.networkFlags & kLitePackedNetworkBit) ||
              !c.reductionsMapped) {
            for (int r = 0; r < c.nranks; ++r)
              std::memcpy(c.smallOutput + static_cast<size_t>(r) * task.bytes,
                          deviceCollectiveHostSlab(c, task.slot, r, task),
                          task.bytes);
            source = c.smallOutput;
          }
          MSCCLPP_CUDATHROW(
              cudaMemcpyAsync(reinterpret_cast<void*>(task.destination), source,
                              task.bytes * c.nranks, cudaMemcpyHostToDevice,
                              c.serviceStreams[1]));
          MSCCLPP_CUDATHROW(
              cudaEventRecord(c.serviceEvents[index][0], c.serviceStreams[1]));
        } else if (task.kind == LiteTaskKind::Gather) {
          if (task.outputOffset > task.outputStride ||
              task.bytes > task.outputStride - task.outputOffset)
            throw mscclpp::Error("invalid device gather range",
                                 mscclpp::ErrorCode::InvalidUsage);
          for (int r = 0; r < c.nranks; ++r) {
            if (r == c.rank) continue;
            // Separate left/right streams preserve bidirectional peer-range
            // progress. Neither stream ever waits for the caller kernel.
            int stream = r < c.rank ? 1 : 2;
            char* destination = reinterpret_cast<char*>(task.destination) +
                                static_cast<size_t>(r) * task.outputStride +
                                task.outputOffset;
            MSCCLPP_CUDATHROW(cudaMemcpyAsync(
                destination, deviceCollectiveHostSlab(c, task.slot, r, task),
                task.bytes, cudaMemcpyHostToDevice, c.serviceStreams[stream]));
          }
          MSCCLPP_CUDATHROW(
              cudaEventRecord(c.serviceEvents[index][0], c.serviceStreams[1]));
          MSCCLPP_CUDATHROW(
              cudaEventRecord(c.serviceEvents[index][1], c.serviceStreams[2]));
          item.eventCount = 2;
        } else {
          throw mscclpp::Error("unknown device DMA descriptor",
                               mscclpp::ErrorCode::InvalidUsage);
        }
        item.ticket = ++next;
        progress = true;
      }
      // Poll every outstanding direction rather than synchronizing on D2H
      // while receive-side copies (or the reverse direction) need progress.
      for (unsigned index = 0; index < kLiteTaskSlots; ++index) {
        auto& item = pending[index];
        if (!item.ticket) continue;
        bool complete = true;
        for (int event = 0; event < item.eventCount; ++event) {
          cudaError_t result = cudaEventQuery(c.serviceEvents[index][event]);
          if (result != cudaSuccess && result != cudaErrorNotReady)
            MSCCLPP_CUDATHROW(result);
          complete &= result == cudaSuccess;
        }
        if (!complete) continue;
        if (item.task.kind == LiteTaskKind::HostAllGather) {
          // A failed peer unblocks CUDA stream waits with UINT64_MAX. Never
          // convert that transport error into a successful FIFO completion.
          size_t chunks = item.task.bytes / item.task.chunkBytes +
                          (item.task.bytes % item.task.chunkBytes != 0);
          for (size_t ci = 0; ci < chunks; ++ci)
            for (int r = 0; r < c.nranks; ++r)
              if (c.hostBuffer->ctrl()->d2hReady[item.task.slot][ci][r].value.load(
                      std::memory_order_acquire) == UINT64_MAX)
                throw mscclpp::Error("host AllGather participant failed",
                                     mscclpp::ErrorCode::SystemError);
        }
        if (item.task.kind == LiteTaskKind::Stage)
          deviceCollectivePublishStage(c, item.task);
        if (__atomic_load_n(&c.tasks->error, __ATOMIC_ACQUIRE)) {
          failDeviceCollectiveContext(c);
          return;
        }
        __atomic_store_n(&c.tasks->slots[index].completed, item.ticket,
                         __ATOMIC_RELEASE);
        item.ticket = 0;
        progress = true;
      }
      while (
          retired < next &&
          __atomic_load_n(&c.tasks->slots[retired % kLiteTaskSlots].completed,
                          __ATOMIC_ACQUIRE) >= retired + 1)
        ++retired;
      __atomic_store_n(&c.tasks->retired, retired, __ATOMIC_RELEASE);
      if (!progress) std::this_thread::yield();
    }
  } catch (...) {
    failDeviceCollectiveContext(c);
  }
}

static void initializeDeviceCollectiveService(DeviceCollectiveContext& c,
                                              ncclComm_t comm) {
  std::vector<int> statuses(c.nranks);
  try {
    void* mapping = mmap(nullptr, sizeof(LiteTaskFifo), PROT_READ | PROT_WRITE,
                         MAP_SHARED | MAP_ANONYMOUS, -1, 0);
    if (mapping == MAP_FAILED)
      throw mscclpp::Error("device FIFO mmap failed",
                           mscclpp::ErrorCode::SystemError);
    c.tasks = new (mapping) LiteTaskFifo{};
    MSCCLPP_CUDATHROW(
        cudaHostRegister(mapping, sizeof(LiteTaskFifo),
                         cudaHostRegisterPortable | cudaHostRegisterMapped));
    c.tasksRegistered = true;
    void* device = nullptr;
    MSCCLPP_CUDATHROW(cudaHostGetDevicePointer(&device, mapping, 0));
    c.tasksDevice = static_cast<LiteTaskFifo*>(device);
    if (c.nranks > 1) {
      if (c.backend == mscclppDeviceCollectiveHostRdma)
        MSCCLPP_CUDATHROW(
            cudaHostAlloc(reinterpret_cast<void**>(&c.smallOutput),
                          2 * 1024 * 1024, cudaHostAllocPortable));
      for (auto& stream : c.serviceStreams)
        MSCCLPP_CUDATHROW(
            cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
      for (auto& pair : c.serviceEvents)
        for (auto& event : pair)
          MSCCLPP_CUDATHROW(
              cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
    }
  } catch (...) {
    statuses[c.rank] = 1;
  }
  comm->comm->bootstrap()->allGather(statuses.data(), sizeof(int));
  if (std::any_of(statuses.begin(), statuses.end(),
                  [](int s) { return s != 0; }))
    throw mscclpp::Error("device FIFO/stream initialization failed",
                         mscclpp::ErrorCode::SystemError);
  try {
    if (c.reduceScatter) {
      c.reduceScatter->fifo = c.tasks;
      c.reduceScatter->stop = &c.stopRdmaProxy;
    }
    if (c.nranks > 1)
      c.serviceThread = std::thread(runDeviceCollectiveService, &c);
    if (c.backend == mscclppDeviceCollectiveHostRdma) {
      if (c.groups.empty() && c.rank == c.localLeader)
        c.rdmaProxyThread = std::thread(runDeviceCollectiveRdmaProxy, &c);
      for (auto& group : c.groups)
        if (c.rank == group->localLeader)
          group->rdmaProxyThread =
              std::thread(runDeviceCollectiveRdmaProxy, group.get());
    }
  } catch (...) {
    statuses[c.rank] = 1;
  }
  comm->comm->bootstrap()->allGather(statuses.data(), sizeof(int));
  if (std::any_of(statuses.begin(), statuses.end(),
                  [](int s) { return s != 0; }))
    throw mscclpp::Error("device worker startup failed",
                         mscclpp::ErrorCode::SystemError);
}
