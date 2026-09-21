// Copyright 2026-present the xvec project
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "distance_matrix_fp16.h"

float squared_euclidean_distance_fp16_avx512(const uint16_t *lhs,
                                              const uint16_t *rhs,
                                              int64_t size) {
  __m512 sum = _mm512_setzero_ps();
  int64_t index = 0;
  for (; index + 16 <= size; index += 16) {
    __m512 left = _mm512_cvtph_ps(
        _mm256_loadu_si256((const __m256i *)(lhs + index)));
    __m512 right = _mm512_cvtph_ps(
        _mm256_loadu_si256((const __m256i *)(rhs + index)));
    __m512 difference = _mm512_sub_ps(left, right);
    sum = _mm512_add_ps(sum, _mm512_mul_ps(difference, difference));
  }
  float result = horizontal_add_fp32_v512(sum);
  for (; index < size; ++index) {
    float difference = _cvtsh_ss(lhs[index]) - _cvtsh_ss(rhs[index]);
    result += difference * difference;
  }
  return result;
}

float inner_product_fp16_avx512(const uint16_t *lhs, const uint16_t *rhs,
                                int64_t size) {
  __m512 sum = _mm512_setzero_ps();
  int64_t index = 0;
  for (; index + 16 <= size; index += 16) {
    __m512 left = _mm512_cvtph_ps(
        _mm256_loadu_si256((const __m256i *)(lhs + index)));
    __m512 right = _mm512_cvtph_ps(
        _mm256_loadu_si256((const __m256i *)(rhs + index)));
    sum = _mm512_add_ps(sum, _mm512_mul_ps(left, right));
  }
  float result = horizontal_add_fp32_v512(sum);
  for (; index < size; ++index) {
    result += _cvtsh_ss(lhs[index]) * _cvtsh_ss(rhs[index]);
  }
  return result;
}

float inner_product_and_squared_norm_fp16_avx512(
    const uint16_t *lhs, const uint16_t *rhs, int64_t size, float *lhs_norm,
    float *rhs_norm) {
  __m512 dot = _mm512_setzero_ps();
  __m512 left_sum = _mm512_setzero_ps();
  __m512 right_sum = _mm512_setzero_ps();
  int64_t index = 0;
  for (; index + 16 <= size; index += 16) {
    __m512 left = _mm512_cvtph_ps(
        _mm256_loadu_si256((const __m256i *)(lhs + index)));
    __m512 right = _mm512_cvtph_ps(
        _mm256_loadu_si256((const __m256i *)(rhs + index)));
    dot = _mm512_add_ps(dot, _mm512_mul_ps(left, right));
    left_sum = _mm512_add_ps(left_sum, _mm512_mul_ps(left, left));
    right_sum = _mm512_add_ps(right_sum, _mm512_mul_ps(right, right));
  }
  float dot_result = horizontal_add_fp32_v512(dot);
  float left_result = horizontal_add_fp32_v512(left_sum);
  float right_result = horizontal_add_fp32_v512(right_sum);
  for (; index < size; ++index) {
    float left = _cvtsh_ss(lhs[index]);
    float right = _cvtsh_ss(rhs[index]);
    dot_result += left * right;
    left_result += left * left;
    right_result += right * right;
  }
  *lhs_norm = left_result;
  *rhs_norm = right_result;
  return dot_result;
}
