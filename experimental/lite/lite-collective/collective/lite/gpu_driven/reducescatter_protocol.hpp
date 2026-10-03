#pragma once
#include "reducescatter_plan.hpp"

#if defined(__CUDACC__)
#define LITE_RS_LAYOUT_HD __host__ __device__
#else
#define LITE_RS_LAYOUT_HD
#endif

// Two kinds of RS work exist:
//  * Single-node paths are composed on the device from FIFO primitives
//    (RsCopy, RsHostSum, RsBarrier: CPU/DMA work) and CTA primitives (sum).
//  * Two-node paths submit one whole-invocation `ReduceScatter` task; their CPU
//    schedule requests CTA phases through requested/completed below.
// Epochs, FIFO tickets and phase numbers are independent counters.
struct LiteReduceScatterControl {
  alignas(64) uint64_t requested;
  alignas(64) uint64_t completed;
  uint64_t source[4];
  uint64_t destination;
  size_t count;
  int sources;
  // Single-node device-composed paths: persistent barrier epoch (CTA thread 0
  // only; one collective per handle at a time).
  alignas(64) uint64_t epoch;
};

// Payload of the RS FIFO primitives. Addresses are plain UVA pointers; the
// device derives them from LiteRsLayout so host and device never disagree.
struct LiteRsOp {
  uint64_t sources[4];       // RsHostSum inputs (CPU addresses)
  int count;                 // number of RsHostSum inputs
  int stream;                // service stream for RsCopy
  int barrierKind;           // control row for RsBarrier
  size_t rows, srcPitch, dstPitch;  // RsCopy 2D when rows > 1
};

// Scratch / host-slab geometry shared by the service and the device paths.
struct LiteRsLayout {
  int ranks = 0, local = 0, me = 0, node = 0;
  unsigned slots = 1;
  size_t chunkCapacity = 0, ringCapacity = 0, stride = 0;
  // Offset inside the node's shared host slab; row r of slot s of local rank q.
  LITE_RS_LAYOUT_HD size_t hostOffset(unsigned slot, int q,
                                      unsigned row = 0) const {
    return 4096 + (static_cast<size_t>(slot) * local + q) * stride +
           static_cast<size_t>(row) * chunkCapacity;
  }
  // Offset inside a rank's GPU scratch.
  LITE_RS_LAYOUT_HD size_t scratchOffset(unsigned slot, unsigned row = 0) const {
    return static_cast<size_t>(slot) * stride +
           static_cast<size_t>(row) * chunkCapacity;
  }
  // Ring rows follow the ranks+6 chunk rows; IPC rings alternate two, the host
  // direct ring keeps one row per step (up to three).
  LITE_RS_LAYOUT_HD size_t ringTail(unsigned parity) const {
    return static_cast<size_t>(ranks + 6) * chunkCapacity +
           static_cast<size_t>(parity) * ringCapacity;
  }
};

// Device-visible view of the private RS resources (stored in the handle).
struct LiteRsDeviceView {
  LiteRsLayout layout;
  char* peers[4] = {};          // GPU scratch of each local rank (IPC pointers)
  char* hostSend = nullptr;     // CPU address of this node's shared host slab
  char* hostSendDevice = nullptr;  // device-mapped address, null if unmapped
  LITE_RS_LAYOUT_HD char* hostRow(unsigned slot, int q, unsigned row = 0) const {
    return hostSend + layout.hostOffset(slot, q, row);
  }
  LITE_RS_LAYOUT_HD char* mappedRow(unsigned slot, int q,
                                    unsigned row = 0) const {
    return hostSendDevice + layout.hostOffset(slot, q, row);
  }
  LITE_RS_LAYOUT_HD char* deviceRow(unsigned slot, int q,
                                    unsigned row = 0) const {
    return peers[q] + layout.scratchOffset(slot, row);
  }
  // Control word value[slot][kind][rank] of the node's shared control page (the
  // first 4096 bytes of the host slab, uint64_t value[5][8][4]); needs the
  // device mapping. Single-writer: only rank `rank` stores to it.
  LITE_RS_LAYOUT_HD unsigned long long* flag(unsigned slot, unsigned kind,
                                             int rank) const {
    return reinterpret_cast<unsigned long long*>(hostSendDevice) +
           (static_cast<size_t>(slot) * 8 + kind) * 4 + rank;
  }
  LITE_RS_LAYOUT_HD char* ringRow(unsigned slot, int q, unsigned parity,
                                  bool onHost) const {
    return onHost ? mappedRow(slot, q) + layout.ringTail(parity)
                  : deviceRow(slot, q) + layout.ringTail(parity);
  }
};
#undef LITE_RS_LAYOUT_HD
