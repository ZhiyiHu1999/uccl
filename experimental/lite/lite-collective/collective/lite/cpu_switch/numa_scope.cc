// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "numa_scope.hpp"

#if defined(__linux__)
#include <numa.h>
#include <sched.h>
#endif

namespace mscclpp::lite {

class NumaScope::Impl {
 public:
#if defined(__linux__)
  cpu_set_t original{};
  bool restore = false;
#endif
};

NumaScope::NumaScope(int numaNode) : impl_(std::make_unique<Impl>()) {
#if defined(__linux__)
  if (numaNode < 0 || numa_available() < 0 ||
      sched_getaffinity(0, sizeof(impl_->original), &impl_->original) != 0) {
    return;
  }
  bitmask* nodeCpus = numa_allocate_cpumask();
  if (nodeCpus == nullptr) return;
  if (numa_node_to_cpus(numaNode, nodeCpus) == 0) {
    cpu_set_t selected;
    CPU_ZERO(&selected);
    for (unsigned cpu = 0;
         cpu < nodeCpus->size && cpu < static_cast<unsigned>(CPU_SETSIZE);
         ++cpu) {
      if (numa_bitmask_isbitset(nodeCpus, cpu) &&
          CPU_ISSET(cpu, &impl_->original)) {
        CPU_SET(cpu, &selected);
      }
    }
    impl_->restore = sched_setaffinity(0, sizeof(selected), &selected) == 0;
  }
  numa_free_cpumask(nodeCpus);
#else
  (void)numaNode;
#endif
}

NumaScope::~NumaScope() {
#if defined(__linux__)
  if (impl_->restore) {
    sched_setaffinity(0, sizeof(impl_->original), &impl_->original);
  }
#endif
}

}  // namespace mscclpp::lite
