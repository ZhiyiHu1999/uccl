#pragma once

#include <cstddef>

namespace mscclpp::lite::detail {

/*
Element-wise sum of inputCount float rows into output.
Dispatches to the AVX-512 implementation when supported, otherwise scalar.
*/
void reduceFloatSum(float const* const* inputs, 
                    size_t inputCount,
                    float* output, 
                    size_t count);

/*
Element-wise sum of two strided regions from inputCount float rows, 
writing the two reductions to firstOutput and secondOutput in a single pass.
*/
void reduceTwoFloatSum(float const* const* inputs, 
                       size_t inputCount,
                       size_t inputStride, 
                       size_t firstOffset,
                       float* firstOutput, 
                       size_t secondOffset,
                       float* secondOutput, 
                       size_t count);

}  // namespace mscclpp::lite::detail
