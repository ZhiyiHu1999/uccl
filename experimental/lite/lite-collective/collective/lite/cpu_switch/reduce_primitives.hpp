#pragma once

#include "cpu_reduction.hpp"
#include "types.hpp"

#include <vector>

namespace mscclpp::lite {

template <typename T>
struct Sum {};

template <typename T>
struct Product {};

template <typename T>
struct Min {};

template <typename T>
struct Max {};

// Primary template reserved for future reduction implementations.
template <typename T, typename RedOp>
class ReducePrimitives {};

// Float-sum specialization backed by the scalar/AVX-512 implementations in
// cpu_reduction.
template <>
class ReducePrimitives<float, Sum<float>> {
 public:
  // Element-wise sum of all contiguous input spans into output.
  void reduce(std::vector<Span<float const>> const& inputs,
              Span<float> output) const {
    checkReduction(inputs, output);
    std::vector<float const*> pointers(inputs.size());
    for (size_t i = 0; i < inputs.size(); ++i) {
      pointers[i] = inputs[i].data;
    }
    detail::reduceFloatSum(pointers.data(), 
                           pointers.size(), 
                           output.data, 
                           output.count);
  }

  // Sum the target row from every source matrix into one output span.
  void reduceRows(std::vector<Rows<float const>> const& inputs, 
                  size_t targetRow, 
                  Span<float> output) const {
    std::vector<Span<float const>> rows(inputs.size());
    for (size_t source = 0; source < inputs.size(); ++source) {
      checkRows(inputs[source]);
      rows[source] = inputs[source].row(targetRow);
    }
    reduce(rows, output);
  }

  // Sum two target rows from every source matrix into two output spans.
  void reduceTwoRows(std::vector<Rows<float const>> const& inputs,
                     size_t firstTarget, 
                     Span<float> firstOutput, 
                     size_t secondTarget, 
                     Span<float> secondOutput) const {
    if (inputs.empty()) {
      throw std::invalid_argument("CpuSwitch reduction needs an input");
    }
    if ((firstOutput.data == nullptr && firstOutput.count != 0) ||
        (secondOutput.data == nullptr && secondOutput.count != 0)) {
      throw std::invalid_argument(
          "CpuSwitch two-row reduction output is null");
    }
    for (size_t source = 0; source < inputs.size(); ++source) {
      auto const& input = inputs[source];
      checkRows(input);
      if (firstTarget >= input.rowCount || secondTarget >= input.rowCount ||
          input.stride != inputs[0].stride ||
          input.columnCount != firstOutput.count ||
          input.columnCount != secondOutput.count) {
        throw std::invalid_argument(
            "CpuSwitch two-row reduction shape mismatch");
      }
    }

    std::vector<float const*> pointers(inputs.size());
    for (size_t source = 0; source < inputs.size(); ++source) {
      pointers[source] = inputs[source].data;
    }
    detail::reduceTwoFloatSum(pointers.data(), 
                              pointers.size(), 
                              inputs[0].stride, 
                              firstTarget,
                              firstOutput.data, 
                              secondTarget, 
                              secondOutput.data, 
                              firstOutput.count);
  }

  // Sum input into accumulator and overwrite accumulator with the result.
  void reduceInPlace(Span<float> accumulator, Span<float const> input) const {
    if (accumulator.count != input.count ||
        (accumulator.data == nullptr && accumulator.count != 0) ||
        (input.data == nullptr && input.count != 0)) {
      throw std::invalid_argument("CpuSwitch in-place reduction shape mismatch");
    }
    std::vector<Span<float const>> inputs{{accumulator.data, 
                                           accumulator.count, 
                                           accumulator.numaNode,
                                           accumulator.device}, 
                                          input};
    reduce(inputs, accumulator);
  }

 private:
  template <typename RowT>
  static void checkRows(Rows<RowT> rows) {
    // Each row must fit within its stride, and non-empty rows need valid data.
    if (rows.stride < rows.columnCount ||
        (rows.data == nullptr && rows.rowCount * rows.columnCount != 0)) {
      throw std::invalid_argument("CpuSwitch rows are invalid");
    }
  }

  static void checkReduction(
      std::vector<Span<float const>> const& inputs, Span<float> output) {
    if (inputs.empty()) {
      throw std::invalid_argument("CpuSwitch reduction needs an input");
    }
    if (output.data == nullptr && output.count != 0) {
      throw std::invalid_argument("CpuSwitch reduction output is null");
    }
    for (auto const& input : inputs) {
      if (input.count != output.count ||
          (input.data == nullptr && input.count != 0)) {
        throw std::invalid_argument("CpuSwitch reduction shapes do not match");
      }
    }
  }
};

}  // namespace mscclpp::lite
