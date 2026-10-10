#pragma once
struct LiteRsChunk {
  // stamp: liteRsStamp(call epoch, chunk number); ordered across chunks and calls.
  uint64_t stamp = 0;
  size_t offset = 0, bytes = 0;
  unsigned slot = 0;
  bool cpu = false, prefetched = false, posted = false;
  // pairIssued: the partner-row copy was already issued (hierarchical paths).
  bool pairIssued = false;
};

// Common per-chunk state shared by the per-path chunk preparations.
struct LiteRsChunkView {
  char const* src;
  unsigned s, remoteRow, ownRow;
  int owned, remote;
  char* send;
};

// SmallHost / HostStaged / TwoRankSmall without a mapped slab: the CPU reduces.
// runSmallHostReduceScatter, runNoCudaIpcHostReduceScatter (multi-node) and the
// DMA/CPU-final branch of runTwoRankSmallHostReduceScatter.
static void liteRsPrepareCpuChunk(LiteReduceScatterContext& c,
                                  LiteTask const& task,
                                  LiteReduceScatterPlan const& p,
                                  LiteRsChunk const& w,
                                  LiteRsChunkView const& v) {
  auto *src = v.src; unsigned s = v.s, remoteRow = v.remoteRow, ownRow = v.ownRow;
  int owned = v.owned, remote = v.remote; char* send = v.send;
  liteRsStageRows(c, c.hostRow(s, c.me), src, task.bytes, w.bytes, true);
  c.drain(0, s);
  c.barrier(s, 0, w.stamp);
  char const* inputs[4]{};
  for (int r = 0; r < c.local; ++r) inputs[r] = c.hostRow(s, r, remote);
  liteRsCpuSum(c, send, inputs, c.local, w.bytes);
  for (int r = 0; r < c.local; ++r) inputs[r] = c.hostRow(s, r, owned);
  liteRsCpuSum(c, c.hostRow(s, c.me, ownRow), inputs, c.local, w.bytes);
}

// TwoRankSmall (mapped) / TwoRankPipeline: only the peer-owned shard leaves the
// node. runTwoRankSmallHostReduceScatter's mapped branch and scheduleTwoRankChunk.
static void liteRsPrepareTwoRankChunk(LiteReduceScatterContext& c,
                                      LiteTask const& task,
                                      LiteReduceScatterPlan const& p,
                                      LiteRsChunk const& w,
                                      LiteRsChunkView const& v) {
  auto *src = v.src; unsigned s = v.s, remoteRow = v.remoteRow;
  int remote = v.remote; char* send = v.send;
  if (p.mappedSend)
    c.gpu(c.mappedRow(s, c.me, remoteRow), w.bytes,
          src + remote * task.bytes);
  else
    c.copy(send, src + remote * task.bytes, w.bytes);
  // The own shard stays in the input; the final add reads it directly
  // (launchTwoRankFinalizeMapped / launchAdd(sendbuff shard, incoming)).
}

// HierarchicalTwo / HierarchicalFour: local partner exchange (and cross-pair for
// four GPUs), node partials, remote-owned partial staged for RDMA.
// scheduleTwoNodeTwoGpuChunk / scheduleChunkLocal.
// Issues the partner-row copy asynchronously; liteRsPrepareHierarchicalChunk
// waits for it, so the next chunk's copy can overlap this chunk's CTA phases.
static void liteRsIssueHierarchicalPair(LiteReduceScatterContext& c,
                                        LiteTask const& task,
                                        LiteReduceScatterPlan const& p,
                                        LiteRsChunk const& w,
                                        LiteRsChunkView const& v) {
  auto *src = v.src; unsigned s = v.s;
  int partner = c.me ^ 1;
  // Only the partner's assigned parity of global shards is needed. Keep
  // original row indices in scratch so both 2g and pair/cross-pair use the
  // same ownership rule; the 2D variant batches these strided rows. Packed
  // rows (parity of the partner) and received rows (parity of this rank)
  // never share an index.
  int parity = partner & 1;
  char* stage = c.deviceRow(s, p.directPartner ? partner : c.me);
  LiteReduceScatterContext::Timed timedPair(c.trace, LiteReduceScatterContext::Trace::Pair);
  if (p.partner2d) {
    MSCCLPP_CUDATHROW(cudaMemcpy2DAsync(
        stage + parity * c.chunkCapacity, 2 * c.chunkCapacity,
        src + parity * task.bytes, 2 * task.bytes, w.bytes, c.ranks / 2,
        cudaMemcpyDefault, c.streams[0]));
  } else {
    for (int row = parity; row < c.ranks; row += 2)
      c.copy(stage + row * c.chunkCapacity, src + row * task.bytes, w.bytes);
  }
  if (!p.directPartner) {
    // CPU pack kernel then D2D push of the packed rows to the partner.
    c.drain(0, s);
    char* pushed = c.deviceRow(s, partner);
    if (p.partner2d) {
      MSCCLPP_CUDATHROW(cudaMemcpy2DAsync(
          pushed + parity * c.chunkCapacity, 2 * c.chunkCapacity,
          stage + parity * c.chunkCapacity, 2 * c.chunkCapacity, w.bytes,
          c.ranks / 2, cudaMemcpyDefault, c.streams[0]));
    } else {
      for (int row = parity; row < c.ranks; row += 2)
        c.copy(pushed + row * c.chunkCapacity, stage + row * c.chunkCapacity,
               w.bytes);
    }
  }
  MSCCLPP_CUDATHROW(cudaEventRecord(c.pairEvents[s], c.streams[0]));
}

static void liteRsPrepareHierarchicalChunk(LiteReduceScatterContext& c,
                                           LiteTask const& task,
                                           LiteReduceScatterPlan const& p,
                                           LiteRsChunk const& w,
                                           LiteRsChunkView const& v) {
  auto *src = v.src; unsigned s = v.s, remoteRow = v.remoteRow, ownRow = v.ownRow;
  int owned = v.owned, remote = v.remote; char* send = v.send;
  {
    LiteReduceScatterContext::Timed timedPair(c.trace, LiteReduceScatterContext::Trace::Pair);
    c.event(c.pairEvents[s]);
  }
  { LiteReduceScatterContext::Timed t(c.trace, LiteReduceScatterContext::Trace::PairBarrier);
    c.barrier(s, 0, w.stamp); }
  char* partnerRows = c.deviceRow(s, c.me);
  if (c.local == 4) {
    // Each pair member computes its own and cross-pair assigned ranks.
    // Pair rows: own/local, own/remote, cross/local, cross/remote.
    for (int part = 0; part < 4; ++part) {
      int targetLocal = c.me ^ (part >= 2 ? 2 : 0);
      int targetNode = part & 1 ? 1 - c.node : c.node;
      int target = targetNode * c.local + targetLocal;
      c.gpu(c.deviceRow(s, c.me, c.ranks + part), w.bytes,
            src + target * task.bytes,
            partnerRows + target * c.chunkCapacity);
    }
    c.barrier(s, 1, w.stamp);
  }
  auto reduceNode = [&](bool remotePart) {
    LiteReduceScatterContext::Timed timedAdd(
        c.trace, remotePart ? LiteReduceScatterContext::Trace::AddRemote : LiteReduceScatterContext::Trace::AddLocal);
    unsigned resultRow = remotePart ? remoteRow : ownRow;
    char* result = remotePart && p.mappedSend
                       ? c.mappedRow(s, c.me, resultRow)
                       : c.deviceRow(s, c.me, resultRow);
    if (c.local == 2) {
      int target = remotePart ? remote : owned;
      c.gpu(result, w.bytes, src + target * task.bytes,
            partnerRows + target * c.chunkCapacity);
    } else {
      unsigned part = remotePart ? 1 : 0;
      c.gpu(result, w.bytes, c.deviceRow(s, c.me, c.ranks + part),
            c.deviceRow(s, c.me ^ 2, c.ranks + 2 + part));
    }
  };
  if (!p.splitFinal) reduceNode(false);
  reduceNode(true);
  if (!p.mappedSend) {
    LiteReduceScatterContext::Timed t(c.trace, LiteReduceScatterContext::Trace::D2hIssue);
    // Own stream: the partner-row copy of the next chunk must not queue behind
    // this D2H (the CTA add above already completed).
    c.copy(send, c.deviceRow(s, c.me, remoteRow), w.bytes, 2);
  }
  // Remote D2H progresses while this same CTA computes the local partial.
  if (p.splitFinal) reduceNode(false);
}

static bool liteRsHierarchical(LiteReduceScatterPlan const& p) {
  return p.path == LiteReduceScatterPath::HierarchicalTwo ||
         p.path == LiteReduceScatterPath::HierarchicalFour;
}

static LiteRsChunkView liteRsChunkView(LiteReduceScatterContext& c,
                                       LiteTask const& task,
                                       LiteRsChunk const& w) {
  LiteRsChunkView v;
  v.src = reinterpret_cast<char const*>(task.source) + w.offset;
  v.s = w.slot;
  v.remoteRow = c.ranks + 4;
  v.ownRow = c.ranks + 5;
  v.owned = c.rank;
  v.remote = (1 - c.node) * c.local + c.me;
  v.send = c.hostRow(w.slot, c.me, v.remoteRow);
  return v;
}

// Takes the slot (credits) and, on the hierarchical paths, starts the
// asynchronous partner-row copy. Everything that waits for the partner or runs
// CTA phases is in liteRsPrepareNetworkChunk.
static LiteRsChunk liteRsBeginNetworkChunk(LiteReduceScatterContext& c,
                                           LiteTask const& task,
                                           LiteReduceScatterPlan const& p,
                                           size_t offset, uint64_t epoch,
                                           size_t index) {
  LiteRsChunk w;
  w.stamp = liteRsStamp(epoch, index + 1);
  w.offset = offset;
  w.bytes = std::min(p.chunkBytes, task.bytes - offset);
  // Slots rotate with the call as well as the chunk, so back-to-back one-chunk
  // calls do not all wait for the ACK of slot 0.
  w.slot = (epoch - 1 + index) % c.slots;
  w.cpu = p.path == LiteReduceScatterPath::SmallHost ||
          p.path == LiteReduceScatterPath::HostStaged ||
          (p.path == LiteReduceScatterPath::TwoRankSmall && !p.mappedSend);
  unsigned s = w.slot;
  // Local consumers retired before this slot was released; remote ACK is
  // independent of send completion, and persists across message regimes.
  {
    LiteReduceScatterContext::Timed timed(c.trace, LiteReduceScatterContext::Trace::Credit);
    if (c.previousAck[s])
      c.wait(&c.ctrl(true)->value[s][1][c.me], c.previousAck[s]);
    if (c.previous[s])
      for (int r = 0; r < c.local; ++r)
        c.wait(&c.ctrl()->value[s][2][r], c.previous[s]);
  }
  if (liteRsHierarchical(p)) {
    liteRsIssueHierarchicalPair(c, task, p, w, liteRsChunkView(c, task, w));
    w.pairIssued = true;
  }
  return w;
}

static void liteRsPrepareNetworkChunk(LiteReduceScatterContext& c,
                                      LiteTask const& task,
                                      LiteReduceScatterPlan const& p,
                                      LiteRsChunk const& w) {
  unsigned s = w.slot;
  LiteRsChunkView v = liteRsChunkView(c, task, w);
  switch (p.path) {
    case LiteReduceScatterPath::SmallHost:
    case LiteReduceScatterPath::HostStaged:
      liteRsPrepareCpuChunk(c, task, p, w, v);
      break;
    case LiteReduceScatterPath::TwoRankSmall:
      if (w.cpu)
        liteRsPrepareCpuChunk(c, task, p, w, v);
      else
        liteRsPrepareTwoRankChunk(c, task, p, w, v);
      break;
    case LiteReduceScatterPath::TwoRankPipeline:
      liteRsPrepareTwoRankChunk(c, task, p, w, v);
      break;
    default:
      liteRsPrepareHierarchicalChunk(c, task, p, w, v);
      break;
  }
  // D2H staging is only awaited when the chunk is posted, so it overlaps
  // with the CTA phases of the chunks prepared after it.
  MSCCLPP_CUDATHROW(cudaEventRecord(c.postEvents[s],
                                    c.streams[liteRsHierarchical(p) ? 2 : 0]));
}

// Posts the prepared chunk's remote-owned partial. Non-eager pipelines post
// just before completion; eager pipelines post `lead` chunks behind preparation.
static void liteRsPostNetworkChunk(LiteReduceScatterContext& c,
                                   LiteRsChunk& w) {
  if (w.posted) return;
  LiteReduceScatterContext::Timed timed(c.trace, LiteReduceScatterContext::Trace::Post);
  c.event(c.postEvents[w.slot]);
  size_t off = c.offset(w.slot, c.me, c.ranks + 4);
  c.postDataAndSignal(w.slot, 0, w.stamp, off, w.bytes);
  w.posted = true;
}

// Start incoming DMA before the next chunk's CTA phases. The receive event
// remains owned by this slot until its final add finishes.
static void liteRsPrefetch(LiteReduceScatterContext& c,
                           LiteReduceScatterPlan const& p, LiteRsChunk& w) {
  if (w.cpu || w.prefetched || p.hostFinal || !p.asyncFinal) return;
  uint64_t ready =
      __atomic_load_n(&c.ctrl(true)->value[w.slot][0][c.me], __ATOMIC_ACQUIRE);
  if (ready == UINT64_MAX) {
    c.check();
    return;
  }
  if (ready < w.stamp) return;
  c.copy(c.deviceRow(w.slot, c.me, c.ranks + 4),
         c.hostRow(w.slot, c.me, c.ranks + 4, true), w.bytes, 1);
  MSCCLPP_CUDATHROW(cudaEventRecord(c.events[w.slot], c.streams[1]));
  w.prefetched = true;
}

static void liteRsFinishNetworkChunk(LiteReduceScatterContext& c,
                                     LiteTask const& task,
                                     LiteReduceScatterPlan const& p,
                                     LiteRsChunk const& w) {
  unsigned s = w.slot, remoteRow = c.ranks + 4, ownRow = c.ranks + 5;
  using Trace = LiteReduceScatterContext::Trace;
  using Timed = LiteReduceScatterContext::Timed;
  {
    Timed timed(c.trace, Trace::WaitRemote);
    c.wait(&c.ctrl(true)->value[s][0][c.me], w.stamp);
  }
  char* output = reinterpret_cast<char*>(task.destination) + w.offset;
  if (w.cpu || p.cpuFinal) {
    if (!w.cpu) {
      c.copy(c.hostRow(s, c.me, ownRow), c.deviceRow(s, c.me, ownRow), w.bytes);
      c.drain(0, s);
    }
    char* own = c.hostRow(s, c.me, ownRow);
    char const* rows[2]{own, c.hostRow(s, c.me, remoteRow, true)};
    liteRsCpuSum(c, own, rows, 2, w.bytes);
    c.copy(output, own, w.bytes, 1);
    c.drain(1, s);
  } else {
    char* incoming;
    if (p.hostFinal) {
      incoming = c.mappedRow(s, c.me, remoteRow, true);
    } else {
      incoming = c.deviceRow(s, c.me, remoteRow);
      Timed timed(c.trace, Trace::H2d);
      if (w.prefetched)
        c.event(c.events[s]);
      else {
        c.copy(incoming, c.hostRow(s, c.me, remoteRow, true), w.bytes, 1);
        c.drain(1, s);
      }
    }
    // Two-rank chunks add the own shard straight from the input; hierarchies
    // add the saved local partial.
    char const* own =
        c.local == 1 ? reinterpret_cast<char const*>(task.source) + w.offset +
                           static_cast<size_t>(c.rank) * task.bytes
                     : c.deviceRow(s, c.me, ownRow);
    Timed timedCta(c.trace, Trace::Cta);
    c.gpu(output, w.bytes, own, incoming);
  }
  Timed timedAck(c.trace, Trace::Ack);
  c.barrier(s, 2, w.stamp);
  c.postSignal(s, 1, w.stamp);
  c.previous[s] = c.previousAck[s] = w.stamp;
}

// Same issue order as the CPU pipelines (runPipelinedChunks and the two-rank /
// 2n*2g variants): prepare chunk i, then either post `lead` chunks behind
// (eager) or complete `lead` chunks behind (post + wait + consume + ACK). A slot
// is recycled only after its previous chunk finished. Lead 0 (SmallHost,
// HostStaged) is the strictly sequential CPU host schedule.
static void liteRsNetwork(LiteReduceScatterContext& c, LiteTask const& task,
                          LiteReduceScatterPlan const& p) {
  auto totalBegin = std::chrono::steady_clock::now();
  LiteRsChunk pending[5]{};
  size_t prepared = 0, posted = 0, finished = 0, begun = 0;
  size_t chunks = (task.bytes - 1) / p.chunkBytes + 1;
  uint64_t epoch = ++c.epoch;  // one epoch per collective call
  auto postUpTo = [&](size_t n) {
    for (; posted < n; ++posted)
      liteRsPostNetworkChunk(c, pending[posted % c.slots]);
  };
  auto finishUpTo = [&](size_t n) {
    for (; finished < n; ++finished) {
      postUpTo(finished + 1);
      liteRsFinishNetworkChunk(c, task, p, pending[finished % c.slots]);
    }
  };
  // Fixed capacity-strided slots make changing B/chunk policy safe. No ACK or
  // CUDA event from the prior call is discarded when changing small/large
  // paths.
  for (size_t i = 0; i < chunks; ++i) {
    if (i >= c.slots) finishUpTo(i - c.slots + 1);
    if (begun <= i) {
      pending[i % c.slots] =
          liteRsBeginNetworkChunk(c, task, p, i * p.chunkBytes, epoch, i);
      begun = i + 1;
    }
    // Start the next chunk's partner-row copy before this chunk's CTA phases;
    // that needs the next slot free, so the oldest chunk is finished first
    // (one chunk less lead than the CPU pipeline on a full ring).
    // Not with mapped_send: the CTA then writes the remote partial over PCIe
    // while the copy runs, which made the 1 MiB case slower.
    if (liteRsHierarchical(p) && !p.mappedSend && i + 1 < chunks) {
      if (i + 2 > c.slots) finishUpTo(i + 2 - c.slots);
      pending[(i + 1) % c.slots] = liteRsBeginNetworkChunk(
          c, task, p, (i + 1) * p.chunkBytes, epoch, i + 1);
      begun = i + 2;
    }
    for (size_t j = finished; j < prepared; ++j)
      liteRsPrefetch(c, p, pending[j % c.slots]);
    liteRsPrepareNetworkChunk(c, task, p, pending[i % c.slots]);
    ++prepared;
    size_t target = prepared > p.lead ? prepared - p.lead : 0;
    if (p.eagerPost)
      postUpTo(target);
    else
      finishUpTo(target);
  }
  postUpTo(chunks);
  finishUpTo(chunks);
  if (c.trace.on) {
    auto& t = c.trace;
    t.us[LiteReduceScatterContext::Trace::Total] +=
        std::chrono::duration<double, std::micro>(
            std::chrono::steady_clock::now() - totalBegin).count();
    if (++t.calls % 50 == 0 && c.rank == 0) {
      double n = static_cast<double>(t.calls);
      std::fprintf(stderr,
                   "[rs-trace] chunks=%zu avg_us/call: total=%.0f credit=%.0f "
                   "pair_copy=%.0f pair_barrier=%.0f add_remote=%.0f "
                   "add_local=%.0f d2h_issue=%.0f post=%.0f wait_remote=%.0f "
                   "h2d=%.0f cta=%.0f ack=%.0f\n",
                   chunks, t.us[11] / n, t.us[0] / n, t.us[1] / n, t.us[2] / n,
                   t.us[3] / n, t.us[4] / n, t.us[5] / n, t.us[6] / n,
                   t.us[7] / n, t.us[8] / n, t.us[9] / n, t.us[10] / n);
      t.calls = 0;
      for (auto& v : t.us) v = 0;
    }
  }
}
