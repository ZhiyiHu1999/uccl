#pragma once

// Collective setup, never called by the service thread or from a device call.
NCCL_API ncclResult_t mscclppRegisterDeviceCollectiveIpcOutput(
    ncclComm_t comm, void* output, size_t bytes) {
  if (!comm) return ncclInvalidArgument;
  return runNcclGuarded("register device IPC output", [&]() {
    mscclpp::CudaDeviceGuard guard(comm->cudaDevice);
    int rank = comm->comm->bootstrap()->getRank();
    int ranks = comm->comm->bootstrap()->getNranks();
    DeviceCollectiveContext* context = nullptr;
    {
      std::lock_guard<std::mutex> lock(gDeviceCollectiveContextMutex);
      auto it = gDeviceCollectiveContexts.find(comm);
      if (it != gDeviceCollectiveContexts.end())
        context = it->second[mscclppDeviceCollectiveCudaIpc].get();
    }
    struct Info { cudaIpcMemHandle_t handle; size_t bytes; int error; };
    std::vector<Info> infos(ranks);
    auto& self = infos[rank];
    self.bytes = bytes;
    try {
      if (!context || (output == nullptr) != (bytes == 0))
        throw mscclpp::Error("invalid IPC output registration",
                             mscclpp::ErrorCode::InvalidUsage);
      // Caller must have finished all kernels using the previous registration.
      MSCCLPP_CUDATHROW(cudaDeviceSynchronize());
      if (output) {
        CUdeviceptr base = 0;
        size_t allocationBytes = 0;
        CUresult result = cuMemGetAddressRange(
            &base, &allocationBytes, reinterpret_cast<CUdeviceptr>(output));
        if (result != CUDA_SUCCESS || base != reinterpret_cast<CUdeviceptr>(output) ||
            bytes > allocationBytes)
          throw mscclpp::Error("register a cudaMalloc allocation base and valid size",
                               mscclpp::ErrorCode::InvalidUsage);
        MSCCLPP_CUDATHROW(cudaIpcGetMemHandle(&self.handle, output));
      }
    } catch (...) { self.error = 1; }
    comm->comm->bootstrap()->allGather(infos.data(), sizeof(Info));
    for (auto const& info : infos)
      if (info.error || info.bytes != bytes)
        throw mscclpp::Error("IPC output registration must agree on all ranks",
                             mscclpp::ErrorCode::InvalidUsage);
    auto& c = *context;
    std::lock_guard<std::mutex> lock(c.mutex);
    // All previous users have drained on all ranks before mappings are changed.
    std::vector<int> errors(ranks);
    try {
      if (c.ipcNextOutputMapping) {
        MSCCLPP_CUDATHROW(cudaIpcCloseMemHandle(c.ipcNextOutputMapping));
        c.ipcNextOutputMapping = nullptr;
      }
      c.ipcNextOutput = c.ipcOutput = nullptr;
      c.ipcOutputBytes = 0;
      if (output && ranks > 1)
        MSCCLPP_CUDATHROW(cudaIpcOpenMemHandle(
            &c.ipcNextOutputMapping, infos[(rank + 1) % ranks].handle,
            cudaIpcMemLazyEnablePeerAccess));
    } catch (...) { errors[rank] = 1; }
    comm->comm->bootstrap()->allGather(errors.data(), sizeof(int));
    if (std::any_of(errors.begin(), errors.end(), [](int e) { return e != 0; })) {
      if (c.ipcNextOutputMapping) cudaIpcCloseMemHandle(c.ipcNextOutputMapping);
      c.ipcNextOutputMapping = nullptr;
      c.ipcNextOutput = c.ipcOutput = nullptr;
      c.ipcOutputBytes = 0;
      throw mscclpp::Error("opening IPC output mapping failed",
                           mscclpp::ErrorCode::InvalidUsage);
    }
    c.ipcOutput = static_cast<char*>(output);
    c.ipcOutputBytes = bytes;
    c.ipcNextOutput = ranks == 1 ? c.ipcOutput
                               : static_cast<char*>(c.ipcNextOutputMapping);
    comm->comm->bootstrap()->barrier();
  });
}
