// Included in host_context.hpp, after ncclComm is defined. GPU-private state;
// no CPU collective entry point, stream callback, or child kernel is used.
#pragma once

struct LiteRsSharedControl {
  // Local stage/pair/retirement and remote ready/ACK use separate words.
  alignas(64) uint64_t value[5][8][4];
};
static_assert(sizeof(LiteRsSharedControl) <= 4096, "RS control page");

struct LiteReduceScatterContext {
  LiteReduceScatterPolicy policy{};
  LiteAllReducePolicy arPolicy{};
  int rank = 0, ranks = 0, local = 0, me = 0, node = 0;
  bool ipc = false, mapped = false;
  size_t capacity = 0, chunkCapacity = 0, ringCapacity = 0, stride = 0;
  unsigned slots = 1;
  uint64_t epoch = 0, phase = 0;
  // previous: last epoch whose local consumers (barrier row 2) retired;
  // previousAck: last epoch whose remote ACK is expected (RS chunks and AllReduce
  // node leaders; other AllReduce ranks never exchange ACKs).
  uint64_t previous[5]{}, previousAck[5]{};
  std::unique_ptr<NodeExchangeBuffer> host;
  char* scratch = nullptr;
  char* peers[4]{};
  cudaStream_t streams[3]{};
  cudaEvent_t events[5]{};
  // One event per slot marking that a chunk's D2H staging finished before its
  // RDMA post (posting is decoupled from preparation, as in the CPU pipelines).
  cudaEvent_t postEvents[5]{};
  LiteRsLayout layout;
  mscclpp::Connection connection;
  // Second independent channel of the two-rank AllReduce ring (2n*1g only).
  mscclpp::Connection connection2;
  // Raw QP posting of the two channels, as the CPU reference does
  // (postPairDataAndSignal / postSmallSignal): the payload and its ready word go
  // out in one doorbell and only every kWireSignalEvery-th post is signaled, so a
  // post never waits for the NIC. Reuse of the send row and of the signal source
  // words is protected by the remote ACK (slot credit), not by flushing.
  struct Wire {
    std::shared_ptr<mscclpp::IbQp> qp;
    mscclpp::IbMr const* mr = nullptr;
    mscclpp::IbMrInfo remote{};
    uint64_t writes = 0;
  };
  Wire wire[2];
  mscclpp::Transport ibTransport = mscclpp::Transport::Unknown;
  static constexpr uint64_t kWireSignalEvery = 128;
  // TEMPORARY diagnostics (UCCL_GPU_DRIVEN_RS_TRACE=1): per-call average time of
  // each stage of the two-node schedule, printed by rank 0 every 50 calls.
  struct Trace {
    enum Stage { Credit, Pair, PairBarrier, AddRemote, AddLocal, D2hIssue, Post,
                 WaitRemote, H2d, Cta, Ack, Total, kStages };
    bool on = false;
    uint64_t calls = 0;
    double us[kStages] = {};
  } trace;
  struct Timed {
    Trace& t;
    int stage;
    std::chrono::steady_clock::time_point begin;
    Timed(Trace& tr, int s) : t(tr), stage(s) {
      if (t.on) begin = std::chrono::steady_clock::now();
    }
    ~Timed() {
      if (t.on)
        t.us[stage] += std::chrono::duration<double, std::micro>(
                           std::chrono::steady_clock::now() - begin).count();
    }
  };
  mscclpp::RegisteredMemory sendMemory, recvMemory, remoteMemory;
  LiteTaskFifo* fifo = nullptr;
  std::atomic<bool>* stop = nullptr;

  ~LiteReduceScatterContext() {
    for (auto s : streams)
      if (s) cudaStreamSynchronize(s);
    connection = {};
    connection2 = {};
    for (auto& w : wire) w = Wire{};
    remoteMemory = {};
    recvMemory = {};
    sendMemory = {};
    for (int i = 0; i < local; ++i)
      if (i != me && peers[i]) cudaIpcCloseMemHandle(peers[i]);
    for (auto e : events)
      if (e) cudaEventDestroy(e);
    for (auto e : postEvents)
      if (e) cudaEventDestroy(e);
    for (auto s : streams)
      if (s) cudaStreamDestroy(s);
    if (scratch) cudaFree(scratch);
  }
  void check() const {
    if (stop->load(std::memory_order_acquire) ||
        __atomic_load_n(&fifo->error, __ATOMIC_ACQUIRE))
      throw mscclpp::Error("device ReduceScatter aborted",
                           mscclpp::ErrorCode::SystemError);
  }
  void wait(uint64_t const* word, uint64_t target) const {
    for (;;) {
      check();
      uint64_t v = __atomic_load_n(word, __ATOMIC_ACQUIRE);
      if (v == UINT64_MAX)
        throw mscclpp::Error("device ReduceScatter peer failed",
                             mscclpp::ErrorCode::SystemError);
      if (v >= target) return;
      std::this_thread::yield();
    }
  }
  void event(cudaEvent_t e) const {
    for (;;) {
      check();
      cudaError_t status = cudaEventQuery(e);
      if (status == cudaSuccess) return;
      if (status != cudaErrorNotReady) MSCCLPP_CUDATHROW(status);
      std::this_thread::yield();
    }
  }
  void drain(int stream, unsigned slot) {
    MSCCLPP_CUDATHROW(cudaEventRecord(events[slot], streams[stream]));
    event(events[slot]);
  }
  LiteRsSharedControl* ctrl(bool receive = false) const {
    return reinterpret_cast<LiteRsSharedControl*>(receive ? host->recvPtr()
                                                          : host->sendPtr());
  }
  size_t offset(unsigned slot, int r, unsigned row = 0) const {
    return layout.hostOffset(slot, r, row);
  }
  // Device-visible description of this context for the single-node device
  // paths (same geometry functions as the service uses).
  LiteRsDeviceView view() const {
    LiteRsDeviceView v;
    v.layout = layout;
    for (int r = 0; r < local && r < 4; ++r) v.peers[r] = peers[r];
    v.hostSend = host ? host->sendPtr() : nullptr;
    v.hostSendDevice =
        host ? const_cast<char*>(host->sendDevicePtr()) : nullptr;
    return v;
  }
  char* hostRow(unsigned slot, int r, unsigned row = 0,
                bool receive = false) const {
    return (receive ? host->recvPtr() : host->sendPtr()) + offset(slot, r, row);
  }
  char* mappedRow(unsigned slot, int r, unsigned row = 0,
                  bool receive = false) const {
    auto* base = receive ? host->recvDevicePtr() : host->sendDevicePtr();
    return const_cast<char*>(base) + offset(slot, r, row);
  }
  char* deviceRow(unsigned slot, int r, unsigned row = 0) const {
    return peers[r] + layout.scratchOffset(slot, row);
  }
  void barrier(unsigned slot, unsigned kind, uint64_t e) {
    __atomic_store_n(&ctrl()->value[slot][kind][me], e, __ATOMIC_RELEASE);
    for (int r = 0; r < local; ++r) wait(&ctrl()->value[slot][kind][r], e);
  }
  void poison() {
    if (!host) return;
    for (unsigned s = 0; s < slots; ++s)
      for (unsigned k = 0; k < 6; ++k)
        __atomic_store_n(&ctrl()->value[s][k][me], UINT64_MAX,
                         __ATOMIC_RELEASE);
  }
  void gpu(void* dst, size_t bytes, void const* a, void const* b = nullptr,
           void const* c = nullptr, void const* d = nullptr) {
    auto& control = fifo->reduceScatter;
    control.source[0] = reinterpret_cast<uint64_t>(a);
    control.source[1] = reinterpret_cast<uint64_t>(b);
    control.source[2] = reinterpret_cast<uint64_t>(c);
    control.source[3] = reinterpret_cast<uint64_t>(d);
    control.destination = reinterpret_cast<uint64_t>(dst);
    control.count = bytes / sizeof(float);
    control.sources = d ? 4 : c ? 3 : b ? 2 : 1;
    __atomic_store_n(&control.requested, ++phase, __ATOMIC_RELEASE);
    wait(&control.completed, phase);
  }
  void copy(void* dst, void const* src, size_t bytes, int stream = 0) {
    MSCCLPP_CUDATHROW(
        cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDefault, streams[stream]));
  }
  // Signalling memory is stable until flush. Payload and ready are ordered on
  // the same CPU-memory QP. ACK is sent only after H2D/CTA consumption.
  void wireOpen(int channel, mscclpp::Connection& conn) {
    auto& w = wire[channel];
    w.qp = conn.getIbQp();
    if (!w.qp) return;
    sendMemory.getIbMrInfo(ibTransport, &w.mr, nullptr);
    remoteMemory.getIbMrInfo(ibTransport, nullptr, &w.remote);
    if (!w.mr) w.qp.reset();
  }
  void wireDrain(Wire& w) {
    while (w.qp->getNumSendCqItems() > 0) {
      int completed = w.qp->pollSendCq();
      if (completed < 0)
        throw mscclpp::Error("ReduceScatter pollSendCq failed",
                             mscclpp::ErrorCode::SystemError);
      for (int i = 0; i < completed; ++i)
        if (w.qp->getSendWcStatus(i) != 0)
          throw mscclpp::Error("ReduceScatter RDMA write failed: " +
                                   w.qp->getSendWcStatusString(i),
                               mscclpp::ErrorCode::SystemError);
      check();
    }
  }
  // Writes `bytes` at `dataOffset` of the remote receive slab (same offset as the
  // local send row) and then publishes control word (slot, kind) = stamp.
  void postDataAndSignal(unsigned slot, unsigned kind, uint64_t stamp,
                         size_t dataOffset, size_t bytes, int channel = 0) {
    auto& w = wire[channel];
    if (!w.qp) {  // not an IB queue pair: synchronous fallback
      auto& conn = channel ? connection2 : connection;
      if (bytes) conn.write(remoteMemory, dataOffset, sendMemory, dataOffset, bytes);
      signal(slot, kind, stamp, channel);
      return;
    }
    auto* word = &ctrl()->value[slot][kind + 6][me];
    *word = stamp;
    size_t src = reinterpret_cast<char*>(word) - host->sendPtr();
    size_t dst = reinterpret_cast<char*>(&ctrl(true)->value[slot][kind][me]) -
                 host->recvPtr();
    bool signaled = (++w.writes % kWireSignalEvery) == 0;
    if (bytes)
      w.qp->stageSendWrite(w.mr, w.remote, static_cast<uint32_t>(bytes), 0,
                           dataOffset, dataOffset, false);
    w.qp->stageSendWrite(w.mr, w.remote, sizeof(uint64_t), 0, src, dst,
                         signaled);
    w.qp->postSend();
    if (signaled) wireDrain(w);
  }
  void postSignal(unsigned slot, unsigned kind, uint64_t stamp,
                  int channel = 0) {
    postDataAndSignal(slot, kind, stamp, 0, 0, channel);
  }
  void signal(unsigned slot, unsigned kind, uint64_t e, int channel = 0) {
    unsigned sourceKind = kind + 6;
    auto* word = &ctrl()->value[slot][sourceKind][me];
    *word = e;
    size_t src = reinterpret_cast<char*>(word) - host->sendPtr();
    size_t dst = reinterpret_cast<char*>(&ctrl(true)->value[slot][kind][me]) -
                 host->recvPtr();
    auto& conn = channel ? connection2 : connection;
    conn.write(remoteMemory, dst, sendMemory, src, sizeof(uint64_t));
    conn.flush();
  }
};

static LiteReduceScatterPolicy readLiteReduceScatterPolicy() {
  LiteReduceScatterPolicy p;
  auto bytes = [](char const* key, size_t fallback, bool zero = false) {
    char const* value = std::getenv(key);
    if (!value || !*value || *value == '-') return fallback;
    char* end = nullptr;
    unsigned long long n = std::strtoull(value, &end, 0);
    if (end == value) return fallback;
    size_t scale = 1;
    if (*end) {
      if (end[1]) return fallback;
      if (*end == 'k' || *end == 'K')
        scale = 1024;
      else if (*end == 'm' || *end == 'M')
        scale = 1024 * 1024;
      else if (*end == 'g' || *end == 'G')
        scale = 1024 * 1024 * 1024;
      else
        return fallback;
    }
    if (n > SIZE_MAX / scale || (!n && !zero)) return fallback;
    return static_cast<size_t>(n) * scale;
  };
  auto mode = [](char const* key, int fallback) {
    char const* v = std::getenv(key);
    return v && *v ? (*v == '0' ? 0 : 1) : fallback;
  };
#define RS_BYTES(field, key) p.field = bytes("MSCCLPP_NCCL_RS_" key, p.field)
#define RS_MODE(field, key) p.field = mode("MSCCLPP_NCCL_RS_" key, p.field)
  RS_BYTES(chunkCapacity, "CHUNK_BYTES");
  RS_BYTES(layoutChunk, "LAYOUT_CHUNK_BYTES");
  RS_BYTES(ringChunk, "LOCAL_RING_CHUNK_BYTES");
  RS_BYTES(hostRingChunk, "NO_CUDAIPC_DIRECT_RING_CHUNK_BYTES");
  p.hostRingMin = bytes("MSCCLPP_NCCL_RS_NO_CUDAIPC_DIRECT_RING_MIN_BYTES",
                        p.hostRingMin, true);
  p.hostSmall =
      bytes("MSCCLPP_NCCL_RS_NO_CUDAIPC_HOST_SMALL_BYTES", p.hostSmall, true);
  RS_BYTES(hostBulkChunk, "NO_CUDAIPC_BULK_CHUNK_BYTES");
  RS_BYTES(hostChunk, "NO_CUDAIPC_CHUNK_BYTES");
  RS_BYTES(smallFull, "SMALL_HOST_FULL_BYTES");
  p.twoRankSmall =
      bytes("MSCCLPP_NCCL_RS_TWO_RANK_SMALL_HOST_BYTES", p.twoRankSmall, true);
  p.deviceFlagMax = bytes("MSCCLPP_NCCL_RS_LOCAL_DEVICE_FLAG_MAX_BYTES",
                          p.deviceFlagMax, true);
  // Zero is a valid lead; unset keeps the CPU per-layout default.
  p.lead = bytes("MSCCLPP_NCCL_RS_LOCAL_LEAD_CHUNKS", p.lead, true);
  p.shortLead = bytes("MSCCLPP_NCCL_RS_SHORT_LOCAL_LEAD_CHUNKS", p.shortLead, true);
  p.longLead = bytes("MSCCLPP_NCCL_RS_LONG_LOCAL_LEAD_CHUNKS", p.longLead, true);
  RS_MODE(noIpc, "NO_CUDAIPC");
  RS_MODE(p2pRing, "P2P_RING");
  RS_MODE(localParallel, "LOCAL_PARALLEL_COPY");
  RS_MODE(eagerPost, "EAGER_RDMA_POST");
  RS_MODE(localRing, "LOCAL_RING");
  RS_MODE(hostRead, "NO_CUDAIPC_HOST_READ");
  RS_MODE(hostRing, "NO_CUDAIPC_DIRECT_RING");
  RS_MODE(hierarchy, "2N2G_HIER");
  RS_MODE(mappedTwoRank, "TWO_RANK_MAPPED_HOST");
  RS_MODE(cpuFinal, "CPU_FINAL_ADD");
  RS_MODE(mappedSingle, "MAPPED_HOST_SINGLE_CHUNK");
  RS_MODE(mappedSend, "MAPPED_SEND_FINAL_REDUCE");
  RS_MODE(hostFinal, "HOST_READ_FINAL_ADD");
  RS_MODE(splitFinal, "SPLIT_FINAL_REDUCE");
  RS_MODE(directPartner, "DIRECT_PARTNER_COPY");
  RS_MODE(partner2d, "DIRECT_PARTNER_COPY_2D");
  RS_MODE(asyncFinal, "ASYNC_FINAL_ADD");
  RS_MODE(disableAvx512, "DISABLE_AVX512");
#undef RS_BYTES
#undef RS_MODE
  p.layoutChunk &= ~size_t{3};
  p.hostSmall &= ~size_t{3};
  p.hostRingMin &= ~size_t{3};
  auto alignedChunk = [](size_t n, size_t fallback) {
    n &= ~size_t{3};
    return n ? n : fallback;
  };
  p.ringChunk = alignedChunk(p.ringChunk, 16 * 1024 * 1024);
  p.hostRingChunk = alignedChunk(p.hostRingChunk, 16 * 1024 * 1024);
  p.hostBulkChunk = alignedChunk(p.hostBulkChunk, 2 * 1024 * 1024);
  p.hostChunk = alignedChunk(p.hostChunk, 256 * 1024);
  return p;
}

static bool sameLiteReduceScatterPolicy(LiteReduceScatterPolicy const& a,
                                        LiteReduceScatterPolicy const& b) {
  return a.chunkCapacity == b.chunkCapacity && a.layoutChunk == b.layoutChunk &&
         a.ringChunk == b.ringChunk && a.hostRingChunk == b.hostRingChunk &&
         a.hostRingMin == b.hostRingMin && a.hostSmall == b.hostSmall &&
         a.hostBulkChunk == b.hostBulkChunk && a.hostChunk == b.hostChunk &&
         a.smallFull == b.smallFull && a.twoRankSmall == b.twoRankSmall &&
         a.deviceFlagMax == b.deviceFlagMax && a.lead == b.lead &&
         a.shortLead == b.shortLead && a.longLead == b.longLead &&
         a.noIpc == b.noIpc && a.p2pRing == b.p2pRing &&
         a.localParallel == b.localParallel && a.eagerPost == b.eagerPost &&
         a.localRing == b.localRing && a.hostRead == b.hostRead &&
         a.hostRing == b.hostRing && a.hierarchy == b.hierarchy &&
         a.mappedTwoRank == b.mappedTwoRank && a.cpuFinal == b.cpuFinal &&
         a.mappedSingle == b.mappedSingle && a.mappedSend == b.mappedSend &&
         a.hostFinal == b.hostFinal && a.splitFinal == b.splitFinal &&
         a.directPartner == b.directPartner && a.partner2d == b.partner2d &&
         a.asyncFinal == b.asyncFinal && a.disableAvx512 == b.disableAvx512;
}

static void prepareLiteReduceScatter(LiteReduceScatterContext& c,
                                     ncclComm_t comm, size_t capacity,
                                     bool preferIpc) {
  auto bootstrap = comm->comm->bootstrap();
  c.rank = bootstrap->getRank();
  c.ranks = bootstrap->getNranks();
  c.local = comm->nRanksPerNode;
  c.me = c.rank % c.local;
  c.node = c.rank / c.local;
  c.capacity = capacity;
  c.slots = c.ranks == c.local ? 1 : c.local == 1 ? 5 : 4;
  c.chunkCapacity = liteRsChunkCapacity(c.policy, capacity, c.ranks);
  c.ringCapacity =
      c.ranks == c.local
          ? std::max(size_t{4}, std::min(capacity / c.ranks,
                                         std::max(c.policy.ringChunk,
                                                  c.policy.hostRingChunk))) &
                ~size_t{3}
          : 0;
  if (c.ringCapacity > SIZE_MAX / 3 ||
      c.chunkCapacity > (SIZE_MAX - 3 * c.ringCapacity) / (c.ranks + 6))
    throw mscclpp::Error("ReduceScatter scratch size overflow",
                         mscclpp::ErrorCode::InvalidUsage);
  // Three ring rows: IPC ring alternates two mailboxes; the host direct ring
  // keeps one row per step so the next rank can pull without a copy.
  c.stride = (c.ranks + 6) * c.chunkCapacity + 3 * c.ringCapacity;
  if (c.stride > (SIZE_MAX - 4096) / (c.slots * c.local))
    throw mscclpp::Error("ReduceScatter scratch size overflow",
                         mscclpp::ErrorCode::InvalidUsage);
  c.layout.ranks = c.ranks;
  c.layout.local = c.local;
  c.layout.me = c.me;
  c.layout.node = c.node;
  c.layout.slots = c.slots;
  c.layout.chunkCapacity = c.chunkCapacity;
  c.layout.ringCapacity = c.ringCapacity;
  c.layout.stride = c.stride;
  size_t slabBytes = 4096 + c.slots * c.local * c.stride;
  c.host = std::make_unique<NodeExchangeBuffer>(NodeExchangeBuffer::create(
      comm->comm, c.rank, c.ranks, c.me == 0, c.rank - c.me, slabBytes, -1,
      comm->cudaDevice,
      "device_rs_" + std::to_string(getpid()) + "_" +
          std::to_string(reinterpret_cast<uintptr_t>(comm)) + "_" +
          std::to_string(preferIpc)));
  if (c.me == 0) {
    std::memset(c.host->sendPtr(), 0, 4096);
    std::memset(c.host->recvPtr(), 0, 4096);
  }
  bootstrap->barrier();
  struct Info {
    int device;
    int ok;
    int mapped;
    cudaIpcMemHandle_t memory;
  };
  std::vector<Info> info(c.ranks);
  std::vector<int> status(c.ranks);
  try {
    MSCCLPP_CUDATHROW(cudaMalloc(&c.scratch, c.slots * c.stride));
    for (auto& stream : c.streams)
      MSCCLPP_CUDATHROW(
          cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    for (auto& event : c.events)
      MSCCLPP_CUDATHROW(
          cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
    for (auto& event : c.postEvents)
      MSCCLPP_CUDATHROW(
          cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
    info[c.rank].device = comm->cudaDevice;
    info[c.rank].mapped = c.host->sendDevicePtr() && c.host->recvDevicePtr();
    info[c.rank].ok = preferIpc && !c.policy.noIpc && c.local > 1;
    if (info[c.rank].ok)
      MSCCLPP_CUDATHROW(cudaIpcGetMemHandle(&info[c.rank].memory, c.scratch));
  } catch (...) {
    status[c.rank] = 1;
  }
  bootstrap->allGather(status.data(), sizeof(int));
  for (int s : status)
    if (s)
      throw mscclpp::Error("ReduceScatter allocation failed",
                           mscclpp::ErrorCode::SystemError);
  bootstrap->allGather(info.data(), sizeof(Info));
  c.ipc = true;
  c.mapped = true;
  for (auto const& i : info) {
    c.ipc &= i.ok != 0;
    c.mapped &= i.mapped != 0;
  }
  c.peers[c.me] = c.scratch;
  if (c.ipc) {
    int base = c.rank - c.me;
    for (int r = 0; r < c.local; ++r) {
      if (r == c.me) continue;
      int access = 0;
      cudaError_t rc = cudaDeviceCanAccessPeer(&access, comm->cudaDevice,
                                               info[base + r].device);
      if (rc != cudaSuccess || !access) status[c.rank] = 1;
    }
    bootstrap->allGather(status.data(), sizeof(int));
    for (int s : status)
      if (s) c.ipc = false;
  }
  if (c.ipc) {
    for (int r = 0; r < c.local; ++r) {
      if (r == c.me) continue;
      void* ptr = nullptr;
      cudaError_t rc = cudaIpcOpenMemHandle(
          &ptr, info[c.rank - c.me + r].memory, cudaIpcMemLazyEnablePeerAccess);
      if (rc == cudaSuccess)
        c.peers[r] = static_cast<char*>(ptr);
      else
        status[c.rank] = 1;
    }
    bootstrap->allGather(status.data(), sizeof(int));
    for (int s : status)
      if (s) c.ipc = false;
  }
  if (!c.ipc) {
    for (int r = 0; r < c.local; ++r)
      if (r != c.me && c.peers[r]) {
        cudaIpcCloseMemHandle(c.peers[r]);
        c.peers[r] = nullptr;
      }
  }
  if (c.ranks != c.local) {
    auto transport = mscclpp::Transport::Unknown;
    // Register and validate on every rank before exchanging connection futures.
    status.assign(c.ranks, 0);
    try {
      transport = mscclpp::lite::selectIBTransportForGpu(comm->cudaDevice);
      mscclpp::TransportFlags flags(transport);
      c.sendMemory =
          comm->comm->registerMemory(c.host->sendPtr(), slabBytes, flags);
      c.recvMemory =
          comm->comm->registerMemory(c.host->recvPtr(), slabBytes, flags);
    } catch (...) {
      status[c.rank] = 1;
    }
    bootstrap->allGather(status.data(), sizeof(int));
    for (int s : status)
      if (s)
        throw mscclpp::Error("ReduceScatter RDMA registration failed",
                             mscclpp::ErrorCode::SystemError);
    int peer = (1 - c.node) * c.local + c.me;
    int tag = 0x5d0000 + c.me * 4 + (preferIpc ? 64 : 0);
    mscclpp::EndpointConfig config(transport,
                                   mscclpp::Device(mscclpp::DeviceType::CPU));
    auto connection = comm->comm->connect(config, peer, tag);
    // The 2n*1g AllReduce ring drives two independent channels.
    auto second = c.local == 1 ? comm->comm->connect(config, peer, tag + 2)
                               : connection;
    comm->comm->sendMemory(c.recvMemory, peer, tag + 1);
    auto memory = comm->comm->recvMemory(peer, tag + 1);
    c.connection = connection.get();
    c.ibTransport = transport;
    if (char const* v = std::getenv("UCCL_GPU_DRIVEN_RS_TRACE"))
      c.trace.on = v[0] && v[0] != '0';
    if (c.local == 1) c.connection2 = second.get();
    c.remoteMemory = memory.get();
    // Needs the remote memory registered above.
    c.wireOpen(0, c.connection);
    if (c.local == 1) c.wireOpen(1, c.connection2);
  }
}

#include "reducescatter_local_schedule.hpp"
#include "reducescatter_network_schedule.hpp"

// Whole-invocation two-node schedules. Single-node paths are not scheduled
// here: their device entries compose the RsCopy / RsHostSum / RsBarrier
// primitives below with CTA arithmetic.
static void executeLiteReduceScatter(LiteReduceScatterContext& c,
                                     LiteTask const& task) {
  auto p = litePlanReduceScatter(c.policy, c.ranks, c.local, c.capacity,
                                 task.bytes, true, c.ipc, c.mapped);
  if (c.ranks == c.local || !task.source || !task.destination ||
      task.bytes % sizeof(float) || p.path != task.reduceScatterPath ||
      p.path == LiteReduceScatterPath::Unsupported ||
      p.path == LiteReduceScatterPath::Generic ||
      p.path == LiteReduceScatterPath::Copy)
    throw mscclpp::Error("invalid ReduceScatter descriptor",
                         mscclpp::ErrorCode::InvalidUsage);
  try {
    liteRsNetwork(c, task, p);
  } catch (...) {
    c.poison();
    throw;
  }
}

// RsCopy: asynchronous 1D/2D DMA on a service stream. The caller records the
// completion event on serviceEvents[index][0].
static void enqueueLiteRsCopy(LiteReduceScatterContext& c,
                              cudaStream_t stream, LiteTask const& task) {
  auto* dst = reinterpret_cast<void*>(task.destination);
  auto const* src = reinterpret_cast<void const*>(task.source);
  if (!dst || !src || !task.bytes)
    throw mscclpp::Error("invalid RsCopy descriptor",
                         mscclpp::ErrorCode::InvalidUsage);
  if (task.rs.rows > 1)
    MSCCLPP_CUDATHROW(cudaMemcpy2DAsync(dst, task.rs.dstPitch, src,
                                        task.rs.srcPitch, task.bytes,
                                        task.rs.rows, cudaMemcpyDefault,
                                        stream));
  else
    MSCCLPP_CUDATHROW(cudaMemcpyAsync(dst, src, task.bytes, cudaMemcpyDefault,
                                      stream));
}

// RsHostSum: CPU float sum of `count` host rows into a host row.
static void executeLiteRsHostSum(LiteReduceScatterContext& c,
                                 LiteTask const& task) {
  char const* rows[4]{};
  int n = task.rs.count;
  if (n < 1 || n > 4 || !task.destination || !task.bytes ||
      task.bytes % sizeof(float))
    throw mscclpp::Error("invalid RsHostSum descriptor",
                         mscclpp::ErrorCode::InvalidUsage);
  for (int r = 0; r < n; ++r)
    rows[r] = reinterpret_cast<char const*>(task.rs.sources[r]);
  liteRsCpuSum(c, reinterpret_cast<char*>(task.destination), rows, n,
               task.bytes);
}

// RsBarrier: publish this rank's epoch in control row (slot, kind) and wait for
// every local rank. The device posts it only after its own DMA completed.
static void executeLiteRsBarrier(LiteReduceScatterContext& c,
                                 LiteTask const& task) {
  if (task.slot < 0 || task.slot >= 5 || task.rs.barrierKind < 0 ||
      task.rs.barrierKind >= 8 || !task.epoch)
    throw mscclpp::Error("invalid RsBarrier descriptor",
                         mscclpp::ErrorCode::InvalidUsage);
  try {
    c.barrier(task.slot, task.rs.barrierKind, task.epoch);
  } catch (...) {
    c.poison();
    throw;
  }
}
