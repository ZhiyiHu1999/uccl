// Included in host_context.hpp after reducescatter_service.hpp. AllReduce reuses
// the ReduceScatter context: its host slab rows, GPU scratch, control words and
// the per-local-rank RDMA connection to the same-index rank of the other node.
// GPU-private; no CPU collective entry point, stream callback or child kernel.
#pragma once

static LiteAllReducePolicy readLiteAllReducePolicy() {
  LiteAllReducePolicy p;
  auto bytes = [](char const* key, size_t fallback) {
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
    if (n > SIZE_MAX / scale) return fallback;
    return static_cast<size_t>(n) * scale;
  };
  // Same parsing as the CPU reference (isTwoRankRingAllReduceEnabled).
  char const* force = std::getenv("MSCCLPP_NCCL_2RANK_RING_ALLREDUCE");
  p.ringForce = force && std::strcmp(force, "0") != 0 &&
                std::strcmp(force, "false") != 0 &&
                std::strcmp(force, "FALSE") != 0;
  char const* chunk = std::getenv("MSCCLPP_NCCL_2RANK_RING_CHUNK_BYTES");
  if (chunk && *chunk) {
    char* end = nullptr;
    unsigned long long value = std::strtoull(chunk, &end, 10);
    if (end != chunk && value >= sizeof(float))
      p.ringChunk = static_cast<size_t>(value) / sizeof(float) * sizeof(float);
  }
  p.rsAgMin = bytes("MSCCLPP_NCCL_AR_RS_AG_MIN_BYTES", p.rsAgMin);
  return p;
}

static bool sameLiteAllReducePolicy(LiteAllReducePolicy const& a,
                                    LiteAllReducePolicy const& b) {
  return a.ringChunk == b.ringChunk && a.rsAgMin == b.rsAgMin &&
         a.ringForce == b.ringForce;
}

// Slot credit before rows of slot `s` are rewritten: the remote ACK of the
// previous use (only ranks that exchange over the network) and the local done
// barrier of every rank.
static void liteArCredit(LiteReduceScatterContext& c, unsigned s,
                         bool exchanges) {
  if (exchanges && c.previousAck[s])
    c.wait(&c.ctrl(true)->value[s][1][c.me], c.previousAck[s]);
  if (c.previous[s])
    for (int r = 0; r < c.local; ++r)
      c.wait(&c.ctrl()->value[s][2][r], c.previous[s]);
}

// Node leader work of the small paths. Rows: input of local rank r at
// hostRow(s, r, 0); local partial hostRow(s, me, 1); remote partial arrives at
// the same offset of the receive slab; final hostRow(s, me, 2). Publishes the
// final flag (control row 3) and ACKs the remote so it may reuse our receive row.
static void liteArNodeExchange(LiteReduceScatterContext& c, unsigned s,
                               uint64_t e, size_t byteOffset, size_t bytes) {
  char const* rows[4]{};
  for (int r = 0; r < c.local; ++r) rows[r] = c.hostRow(s, r, 0) + byteOffset;
  char* partial = c.hostRow(s, c.me, 1);
  liteRsCpuSum(c, partial, rows, c.local, bytes);
  size_t off = c.offset(s, c.me, 1);
  c.postDataAndSignal(s, 0, e, off, bytes);
  c.wait(&c.ctrl(true)->value[s][0][c.me], e);
  char const* pair[2] = {partial, c.hostRow(s, c.me, 1, true)};
  liteRsCpuSum(c, c.hostRow(s, c.me, 2), pair, 2, bytes);
  __atomic_store_n(&c.ctrl()->value[s][3][c.me], e, __ATOMIC_RELEASE);
  c.postSignal(s, 1, e);
}

// SmallMapped: CTA stage -> local barrier -> leader exchange -> CTA copy-out.
static void liteArSmallMapped(LiteReduceScatterContext& c,
                              LiteTask const& task, uint64_t epoch) {
  size_t bytes = task.bytes;
  uint64_t e = liteRsStamp(epoch, 1);
  unsigned s = (epoch - 1) % c.slots;
  bool leader = c.me == 0;
  liteArCredit(c, s, leader);
  auto const* src = reinterpret_cast<char const*>(task.source);
  auto* dst = reinterpret_cast<char*>(task.destination);
  c.gpu(c.mappedRow(s, c.me, 0), bytes, src);
  c.barrier(s, 0, e);  // every input row is complete
  if (leader) liteArNodeExchange(c, s, e, 0, bytes);
  c.wait(&c.ctrl()->value[s][3][0], e);  // leader's final row
  c.gpu(dst, bytes, c.mappedRow(s, 0, 2));
  c.barrier(s, 2, e);  // all readers done before the slot is reused
  c.previous[s] = e;
  if (leader) c.previousAck[s] = e;
}

// SmallTwoLeader: D2H stage -> barrier -> part leaders (local 0 and 2) exchange
// one half each -> H2D of both final halves.
static void liteArSmallTwoLeader(LiteReduceScatterContext& c,
                                 LiteTask const& task, uint64_t epoch) {
  size_t bytes = task.bytes, half = bytes / 2;
  uint64_t e = liteRsStamp(epoch, 1);
  unsigned s = (epoch - 1) % c.slots;
  bool part = c.me == 0 || c.me == 2;
  liteArCredit(c, s, part);
  auto const* src = reinterpret_cast<char const*>(task.source);
  auto* dst = reinterpret_cast<char*>(task.destination);
  c.copy(c.hostRow(s, c.me, 0), src, bytes);
  c.drain(0, s);
  c.barrier(s, 0, e);
  if (part) liteArNodeExchange(c, s, e, c.me == 0 ? 0 : half, half);
  c.wait(&c.ctrl()->value[s][3][0], e);
  c.wait(&c.ctrl()->value[s][3][2], e);
  c.copy(dst, c.hostRow(s, 0, 2), half, 1);
  c.copy(dst + half, c.hostRow(s, 2, 2), half, 1);
  c.drain(1, s);
  c.barrier(s, 2, e);
  c.previous[s] = e;
  if (part) c.previousAck[s] = e;
}

// TwoRankRing (allreduce_two_rank_ring.cuh describes the data order).
// Channel k uses slot k, connection k and control rows 0 (peer part ready) and 1
// (final part ready). Stage buffers: stage 0 at row 0, stage 1 at row 2 (each up
// to two rows). A stage is consumed before the next one is produced, so no ACK
// exists; epochs are drawn from the shared monotonic counter.
static void liteArTwoRankRing(LiteReduceScatterContext& c, LiteTask const& task,
                              LiteAllReducePlan const& p, uint64_t epoch) {
  uint64_t seq = 0;  // numbers the stages of this call
  auto const* src = reinterpret_cast<char const*>(task.source);
  auto* dst = reinterpret_cast<char*>(task.destination);
  size_t count = task.bytes / sizeof(float);
  size_t chunkCount = p.chunkBytes / sizeof(float);
  struct Channel {
    size_t start, count, elem;
  };
  size_t first = (count + 1) / 2;
  Channel ch[2] = {{0, first, 0}, {first, count - first, 0}};
  size_t widest = std::max(ch[0].count, ch[1].count);
  size_t loops = (widest + 2 * chunkCount - 1) / (2 * chunkCount);
  struct Part {
    bool active = false;
    size_t ownOff = 0, ownBytes = 0, peerOff = 0, peerBytes = 0;
    uint64_t sendEpoch = 0, finalEpoch = 0;
  };
  for (size_t loop = 0; loop < loops; ++loop) {
    Part part[2];
    for (int k = 0; k < 2; ++k) {
      auto& g = ch[k];
      if (g.elem >= g.count) continue;
      size_t loopCount = std::min(2 * chunkCount, g.count - g.elem);
      size_t a = std::min(chunkCount, (loopCount + 1) / 2);
      size_t counts[2] = {a, loopCount - a};
      size_t starts[2] = {g.start + g.elem, g.start + g.elem + a};
      int own = c.rank, peer = c.rank ^ 1;
      part[k] = {true,
                 starts[own] * sizeof(float), counts[own] * sizeof(float),
                 starts[peer] * sizeof(float), counts[peer] * sizeof(float),
                 0, 0};
      g.elem += 2 * chunkCount;
    }
    // Peer part D2H, then post it.
    for (int k = 0; k < 2; ++k)
      if (part[k].active && part[k].peerBytes)
        c.copy(c.hostRow(k, c.me, 0), src + part[k].peerOff, part[k].peerBytes,
               k);
    for (int k = 0; k < 2; ++k)
      if (part[k].active && part[k].peerBytes) c.drain(k, k);
    for (int k = 0; k < 2; ++k) {
      if (!part[k].active) continue;
      part[k].sendEpoch = liteRsStamp(epoch, ++seq);
      c.postDataAndSignal(k, 0, part[k].sendEpoch, c.offset(k, c.me, 0),
                          part[k].peerBytes, k);
    }
    // Own part: remote contribution H2D, CTA add into the output, D2H of the
    // reduced part, then post it.
    for (int k = 0; k < 2; ++k) {
      if (!part[k].active) continue;
      c.wait(&c.ctrl(true)->value[k][0][c.me], part[k].sendEpoch);
      if (!part[k].ownBytes) continue;
      c.copy(c.deviceRow(k, c.me, 0), c.hostRow(k, c.me, 0, true),
             part[k].ownBytes, k);
      c.drain(k, k);
      c.gpu(dst + part[k].ownOff, part[k].ownBytes, src + part[k].ownOff,
            c.deviceRow(k, c.me, 0));
      c.copy(c.hostRow(k, c.me, 2), dst + part[k].ownOff, part[k].ownBytes, k);
      c.drain(k, k);
    }
    for (int k = 0; k < 2; ++k) {
      if (!part[k].active) continue;
      part[k].finalEpoch = liteRsStamp(epoch, ++seq);
      c.postDataAndSignal(k, 1, part[k].finalEpoch, c.offset(k, c.me, 2),
                          part[k].ownBytes, k);
    }
    // Remote final part H2D into the output.
    for (int k = 0; k < 2; ++k) {
      if (!part[k].active) continue;
      c.wait(&c.ctrl(true)->value[k][1][c.me], part[k].finalEpoch);
      if (!part[k].peerBytes) continue;
      c.copy(dst + part[k].peerOff, c.hostRow(k, c.me, 2, true),
             part[k].peerBytes, k);
      c.drain(k, k);
    }
  }
}

static void executeLiteAllReduce(LiteReduceScatterContext& c,
                                 LiteTask const& task) {
  auto p = litePlanAllReduce(c.arPolicy, c.policy, c.ranks, c.local, c.capacity,
                             task.bytes, true, c.ipc, c.mapped);
  if (c.ranks != 2 * c.local || !task.source || !task.destination ||
      task.bytes % sizeof(float) || p.path != task.allReducePath)
    throw mscclpp::Error("invalid AllReduce descriptor",
                         mscclpp::ErrorCode::InvalidUsage);
  try {
    uint64_t epoch = ++c.epoch;  // one epoch per collective call
    switch (p.path) {
      case LiteAllReducePath::SmallMapped:
        liteArSmallMapped(c, task, epoch);
        break;
      case LiteAllReducePath::SmallTwoLeader:
        liteArSmallTwoLeader(c, task, epoch);
        break;
      case LiteAllReducePath::TwoRankRing:
        liteArTwoRankRing(c, task, p, epoch);
        break;
      default:
        throw mscclpp::Error("unsupported AllReduce path",
                             mscclpp::ErrorCode::InvalidUsage);
    }
  } catch (...) {
    c.poison();
    throw;
  }
}
