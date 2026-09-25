#pragma once

#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

static constexpr char liteBenchmarkUsage[] =
    "Usage: device_collectives_bench [sizes...]\n"
    "       device_collectives_bench -b BEGIN -e END [-f FACTOR] [-g 1] "
    "[-w WARMUPS] [-n ITERS]\n"
    "Sizes: positive bytes, optionally suffixed B/K/M/G (binary units).\n"
    "Range: multiply by FACTOR (integer >= 2, default 2) while <= END.\n"
    "Both -b and -e are required; do not mix ranges with positional sizes.\n"
    "-c, --collective: allgather/allreduce/reducescatter/all (default all).\n"
    "-g: GPUs per MPI process; only 1 is supported.\n"
    "-w: warmups >= 0; -n: measured iterations >= 1. CLI overrides env.\n"
    "Defaults: WARMUP_ITERS=20, ITERS=100; sizes 128 256 512 1K 4K 16K 64K.\n"
    "-h, --help: show this help without initializing MPI or CUDA.\n"
    "Bytes mean AG input, AR full input/output, RS output shard per rank.\n";

struct LiteBenchmarkOptions {
  std::string collective = "all";
  int warmups = 20;
  int iterations = 100;
  bool help = false;
  bool printConfig = false;
  std::vector<size_t> sizes;
};

inline bool liteBenchmarkNumber(char const* text, size_t& result,
                                bool allowSuffix) {
  if (!text || *text < '0' || *text > '9') return false;
  char* end = nullptr;
  errno = 0;
  unsigned long long value = std::strtoull(text, &end, 10);
  if (errno || end == text) return false;
  size_t scale = 1;
  if (*end) {
    if (!allowSuffix || end[1]) return false;
    switch (*end) {
      case 'B': case 'b': break;
      case 'K': case 'k': scale = 1024; break;
      case 'M': case 'm': scale = 1024 * 1024; break;
      case 'G': case 'g': scale = size_t{1} << 30; break;
      default: return false;
    }
  }
  if (value > std::numeric_limits<size_t>::max() / scale) return false;
  result = static_cast<size_t>(value) * scale;
  return true;
}

inline bool liteParseBenchmarkOptions(int argc, char** argv,
                                     LiteBenchmarkOptions& out,
                                     std::string& error) {
  out = LiteBenchmarkOptions{};
  for (int i = 1; i < argc; ++i) {
    if (!std::strcmp(argv[i], "-h") || !std::strcmp(argv[i], "--help")) {
      out.help = true;
      return true;
    }
  }
  char const* warmupText = std::getenv("WARMUP_ITERS");
  char const* iterationText = std::getenv("ITERS");
  size_t begin = 0, end = 0, factor = 2;
  bool hasBegin = false, hasEnd = false, hasFactor = false;
  auto fail = [&](std::string const& message) {
    error = message;
    return false;
  };
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--print-config") {
      out.printConfig = true;
      continue;
    }
    if (arg == "-c" || arg == "--collective") {
      if (++i == argc) return fail("Missing value for " + arg);
      out.collective = argv[i];
      if (out.collective != "all" && out.collective != "allgather" &&
          out.collective != "allreduce" && out.collective != "reducescatter")
        return fail("Invalid collective: " + out.collective);
      continue;
    }
    if (arg == "-b" || arg == "-e" || arg == "-f" || arg == "-g" ||
        arg == "-w" || arg == "-n") {
      if (++i == argc) return fail("Missing value for " + arg);
      if (arg == "-w") { warmupText = argv[i]; continue; }
      if (arg == "-n") { iterationText = argv[i]; continue; }
      size_t value = 0;
      if (!liteBenchmarkNumber(argv[i], value, arg == "-b" || arg == "-e") ||
          value == 0) return fail("Invalid value for " + arg + ": " + argv[i]);
      if (arg == "-b") { begin = value; hasBegin = true; }
      if (arg == "-e") { end = value; hasEnd = true; }
      if (arg == "-f") { factor = value; hasFactor = true; }
      if (arg == "-g" && value != 1)
        return fail("Only -g 1 is supported: use MPI ranks for multiple GPUs");
    } else {
      size_t value = 0;
      if (!liteBenchmarkNumber(argv[i], value, true) || !value)
        return fail("Invalid option or size: " + arg);
      out.sizes.push_back(value);
    }
  }
  auto iterations = [&](char const* text, int& value, bool zeroAllowed) {
    if (!text) return true;
    size_t parsed = 0;
    if (!liteBenchmarkNumber(text, parsed, false) ||
        parsed > static_cast<size_t>(std::numeric_limits<int>::max()) ||
        (!zeroAllowed && parsed == 0)) return false;
    value = static_cast<int>(parsed);
    return true;
  };
  if (!iterations(warmupText, out.warmups, true))
    return fail("Invalid warmup count (-w / WARMUP_ITERS)");
  if (!iterations(iterationText, out.iterations, false))
    return fail("Invalid iteration count (-n / ITERS)");
  if (hasBegin || hasEnd || hasFactor) {
    if (!hasBegin || !hasEnd || !out.sizes.empty() || begin > end || factor < 2)
      return fail("Require -b <= -e, -f >= 2, and no positional sizes");
    for (size_t bytes = begin;;) {
      out.sizes.push_back(bytes);
      if (bytes > end / factor) break;  // Also prevents multiplication overflow.
      bytes *= factor;
    }
  } else if (out.sizes.empty()) {
    out.sizes = {128, 256, 512, 1024, 4096, 16384, 65536};
  }
  return true;
}
