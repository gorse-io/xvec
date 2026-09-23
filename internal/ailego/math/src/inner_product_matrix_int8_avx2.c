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

#include <immintrin.h>
#include <stdint.h>

// Like zvec's AVX2 INT8 batch kernel, widen signed bytes before madd_epi16.
// This also handles -128 without the saturation/sign corner cases of maddubs.
int64_t inner_product_int8_avx2(const int8_t *lhs, const int8_t *rhs,
                               int64_t size) {
  int64_t result = 0;
  while (size >= 32) {
    // Even an all -128 dot product fits in int32 for each 65536-byte chunk.
    // Reduce chunks to int64 before accumulating larger dimensions.
    int64_t count = size < 65536 ? (size & ~31LL) : 65536;
    __m256i sum_lo = _mm256_setzero_si256();
    __m256i sum_hi = _mm256_setzero_si256();
    for (int64_t i = 0; i < count; i += 32) {
      __m256i left = _mm256_loadu_si256((const __m256i *)(lhs + i));
      __m256i right = _mm256_loadu_si256((const __m256i *)(rhs + i));
      __m256i left_lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(left));
      __m256i left_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(left, 1));
      __m256i right_lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(right));
      __m256i right_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(right, 1));
      sum_lo = _mm256_add_epi32(sum_lo, _mm256_madd_epi16(left_lo, right_lo));
      sum_hi = _mm256_add_epi32(sum_hi, _mm256_madd_epi16(left_hi, right_hi));
    }
    __m256i sum = _mm256_add_epi32(sum_lo, sum_hi);
    __m128i half = _mm_add_epi32(_mm256_castsi256_si128(sum),
                                 _mm256_extracti128_si256(sum, 1));
    half = _mm_hadd_epi32(half, half);
    half = _mm_hadd_epi32(half, half);
    result += _mm_cvtsi128_si32(half);
    lhs += count;
    rhs += count;
    size -= count;
  }
  for (int64_t i = 0; i < size; ++i) {
    result += (int64_t)lhs[i] * rhs[i];
  }
  return result;
}
