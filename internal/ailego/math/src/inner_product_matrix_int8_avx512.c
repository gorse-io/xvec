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

int64_t inner_product_int8_avx512(const int8_t *lhs, const int8_t *rhs,
                                 int64_t size) {
  int64_t result = 0;
  while (size >= 64) {
    // Even an all -128 dot product fits in int32 for each 65536-byte chunk.
    // Reduce chunks to int64 before accumulating larger dimensions.
    int64_t count = size < 65536 ? (size & ~63LL) : 65536;
    __m512i sum_lo = _mm512_setzero_si512();
    __m512i sum_hi = _mm512_setzero_si512();
    for (int64_t i = 0; i < count; i += 64) {
      __m512i left = _mm512_loadu_si512((const void *)(lhs + i));
      __m512i right = _mm512_loadu_si512((const void *)(rhs + i));
      __m512i left_lo =
          _mm512_cvtepi8_epi16(_mm512_castsi512_si256(left));
      __m512i left_hi =
          _mm512_cvtepi8_epi16(_mm512_extracti64x4_epi64(left, 1));
      __m512i right_lo =
          _mm512_cvtepi8_epi16(_mm512_castsi512_si256(right));
      __m512i right_hi =
          _mm512_cvtepi8_epi16(_mm512_extracti64x4_epi64(right, 1));
      sum_lo = _mm512_add_epi32(sum_lo, _mm512_madd_epi16(left_lo, right_lo));
      sum_hi = _mm512_add_epi32(sum_hi, _mm512_madd_epi16(left_hi, right_hi));
    }
    __m512i sum = _mm512_add_epi32(sum_lo, sum_hi);
    __m128i quarter = _mm_add_epi32(
        _mm_add_epi32(_mm512_castsi512_si128(sum),
                      _mm512_extracti32x4_epi32(sum, 1)),
        _mm_add_epi32(_mm512_extracti32x4_epi32(sum, 2),
                      _mm512_extracti32x4_epi32(sum, 3)));
    quarter = _mm_hadd_epi32(quarter, quarter);
    quarter = _mm_hadd_epi32(quarter, quarter);
    result += _mm_cvtsi128_si32(quarter);
    lhs += count;
    rhs += count;
    size -= count;
  }
  for (int64_t i = 0; i < size; ++i) {
    result += (int64_t)lhs[i] * rhs[i];
  }
  return result;
}
