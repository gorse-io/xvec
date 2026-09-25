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

static inline __m512i batch_dot_half_avx512(__m256i left, __m256i right) {
  return _mm512_madd_epi16(_mm512_cvtepi8_epi16(left),
                           _mm512_cvtepi8_epi16(right));
}

// vectors contains query followed by four candidates. The aggregate pointer
// keeps this function within the six-register SysV argument ABI GoAT supports.
void xvec_avx512_batch_inner_products_int8_4(const int8_t *const *vectors,
                                              int64_t size,
                                              int64_t *output) {
  const int8_t *query = vectors[0];
  const int8_t *candidates[4] = {vectors[1], vectors[2], vectors[3], vectors[4]};
  int64_t totals[4] = {0, 0, 0, 0};
  while (size != 0) {
    int64_t count = size < 65536 ? size : 65536;
    __m512i sums[4];
    for (int j = 0; j < 4; ++j) sums[j] = _mm512_setzero_si512();
    for (int64_t i = 0; i < count; i += 64) {
      __m512i q = _mm512_loadu_si512((const void *)(query + i));
      __m256i q_low = _mm512_castsi512_si256(q);
      __m256i q_high = _mm512_extracti64x4_epi64(q, 1);
      for (int j = 0; j < 4; ++j) {
        __m512i value = _mm512_loadu_si512((const void *)(candidates[j] + i));
        sums[j] = _mm512_add_epi32(
            sums[j], batch_dot_half_avx512(
                         q_low, _mm512_castsi512_si256(value)));
        sums[j] = _mm512_add_epi32(
            sums[j], batch_dot_half_avx512(
                         q_high, _mm512_extracti64x4_epi64(value, 1)));
      }
    }
    for (int j = 0; j < 4; ++j) {
      totals[j] += (int64_t)_mm512_reduce_add_epi32(sums[j]);
      candidates[j] += count;
    }
    query += count;
    size -= count;
  }
  for (int j = 0; j < 4; ++j) output[j] = totals[j];
}
