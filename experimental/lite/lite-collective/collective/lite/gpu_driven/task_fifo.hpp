#pragma once

#include <stddef.h>
#include <stdint.h>

// Single CTA producer / CPU consumer. Dequeue, DMA completion, and payload
// retirement are distinct: completion releases only the descriptor. The
// collective ready/done and RDMA ACK protocol owns the payload slots.
constexpr unsigned kLiteTaskSlots = 8;
constexpr uint64_t kLiteGenericNetworkBit = uint64_t{1} << 63;
constexpr uint64_t kLitePackedNetworkBit = uint64_t{1} << 62;
constexpr uint64_t kLiteCompactNetworkBit = uint64_t{1} << 61;
constexpr uint64_t kLiteNetworkMask =
    kLiteGenericNetworkBit | kLitePackedNetworkBit | kLiteCompactNetworkBit;
enum class LiteTaskKind : uint64_t { Stage, Gather, GatherPacked };
struct LiteTask {
  LiteTaskKind kind;
  uint64_t epoch;
  uint64_t networkFlags;
  uint64_t source;
  uint64_t destination;
  size_t bytes;
  size_t outputStride;
  size_t outputOffset;
  int slot;
};
struct alignas(64) LiteTaskSlot {
  LiteTask task;
  alignas(64) uint64_t completed;
};
struct LiteTaskFifo {
  alignas(64) uint64_t submitted;
  alignas(64) uint64_t retired;
  alignas(64) uint64_t error;
  LiteTaskSlot slots[kLiteTaskSlots];
};

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 700
#error "GPU-driven collectives require system-scope acquire/release (sm_70+)"
#endif

#if defined(__CUDACC__)
// All control locations are naturally aligned, single-writer 64-bit words.
// System scope matters here: CTA barriers alone do not publish to the CPU.
static __device__ __forceinline__ unsigned long long liteLoadAcquire(
    unsigned long long const volatile* p) {
  unsigned long long value;
#if __CUDA_ARCH__ >= 700
  asm volatile("ld.acquire.sys.global.u64 %0, [%1];"
               : "=l"(value)
               : "l"(p)
               : "memory");
#else
  value = *p;
  __threadfence_system();
#endif
  return value;
}
static __device__ __forceinline__ void liteStoreRelease(
    unsigned long long volatile* p, unsigned long long value) {
#if __CUDA_ARCH__ >= 700
  asm volatile("st.release.sys.global.u64 [%0], %1;" ::"l"(p), "l"(value)
               : "memory");
#else
  __threadfence_system();
  *p = value;
#endif
}
#endif
