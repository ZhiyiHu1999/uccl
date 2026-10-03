#pragma once
#include <stddef.h>
#include <stdint.h>

// One outstanding AllGather per handle. The service publishes a CPU-reference
// ordered-slot descriptor; the calling CTA executes its SM phases in place.
// prepared/deviceDone are FIFO sequence numbers, not algorithm epochs.
struct LiteNetworkControl {
  alignas(64) uint64_t prepared = 0;
  alignas(64) uint64_t deviceDone = 0;
  alignas(64) uint64_t abort = 0;
  uint64_t epoch = 0;
  char* slab = nullptr;
  char* control = nullptr;
  size_t slotOffset = 0;
  size_t flagOffset = 0;
  size_t segmentBytes = 0;
  size_t flagInSegment = 0;
  size_t readyOffset = 0;
  size_t readyBase = 0;
  size_t readyStride = 0;
  int localGroupSize = 0;
  bool mapped = true;
  bool oneRankRegister = false;
  bool stageWithSm = false;
  bool receiveWithSm = false;
};
