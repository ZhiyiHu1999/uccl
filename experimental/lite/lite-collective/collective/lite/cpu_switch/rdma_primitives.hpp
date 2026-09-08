// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#pragma once

#include <cstddef>
#include <cstdint>

namespace mscclpp::lite {

class RdmaPrimitives {
 public:
  // Offset-only ports own their local/remote registrations.
  template <typename Port>
  void rdmaWrite(Port& port, size_t localOffset, size_t remoteOffset,
                 size_t bytes) const {
    port.write(localOffset, remoteOffset, bytes);
  }

  template <typename Port>
  void rdmaWriteAndFlush(Port& port, size_t localOffset, size_t remoteOffset,
                         size_t bytes) const {
    port.writeAndFlush(localOffset, remoteOffset, bytes);
  }

  // Core Connection objects take the registrations explicitly.  Keep this
  // overload templated so CpuSwitch does not introduce a dependency from the
  // collective primitives layer back to a concrete transport implementation.
  template <typename Connection, typename RemoteMemory, typename LocalMemory>
  void rdmaWrite(Connection& connection, RemoteMemory const& remoteMemory,
                 size_t remoteOffset, LocalMemory const& localMemory,
                 size_t localOffset, size_t bytes) const {
    connection.write(remoteMemory, remoteOffset, localMemory, localOffset,
                     bytes);
  }

  template <typename Connection, typename RemoteMemory, typename LocalMemory>
  void rdmaWriteAndFlush(Connection& connection,
                         RemoteMemory const& remoteMemory, size_t remoteOffset,
                         LocalMemory const& localMemory, size_t localOffset,
                         size_t bytes) const {
    rdmaWrite(connection, remoteMemory, remoteOffset, localMemory, localOffset,
              bytes);
    connection.flush();
  }

  template <typename Connection>
  void rdmaFlush(Connection& connection) const {
    connection.flush();
  }

  // Ordered remote epoch publication. updateAndSync is an RDMA atomic on IB
  // and the transport-specific equivalent on the other Connection backends.
  template <typename Connection, typename RemoteMemory>
  void rdmaSignal(Connection& connection, RemoteMemory const& remoteMemory,
                  size_t remoteOffset, uint64_t* localValue,
                  uint64_t epoch) const {
    connection.updateAndSync(remoteMemory, remoteOffset, localValue, epoch);
  }

  template <typename Port>
  void signal(Port& port, uint64_t epoch) const {
    port.signal(epoch);
  }

  template <typename Port>
  void wait(Port const& port, uint64_t epoch) const {
    port.wait(epoch);
  }
};

}  // namespace mscclpp::lite
