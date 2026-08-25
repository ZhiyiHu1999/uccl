// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#pragma once

#include <memory>

namespace mscclpp::lite {

/* Temporarily binds the calling thread to CPUs belonging to one NUMA node. */
class NumaScope {
 public:
  explicit NumaScope(int numaNode);
  ~NumaScope();

  NumaScope(NumaScope const&) = delete;
  NumaScope& operator=(NumaScope const&) = delete;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace mscclpp::lite
