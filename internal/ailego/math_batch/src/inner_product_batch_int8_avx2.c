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

// Share each widened query block across four signed INT8 candidates, as in
// zvec's one-to-many scorer. Chunk reduction preserves exact int64 totals.
void xvec_avx2_batch_inner_products_int8_4(
    const int8_t *query, const int8_t *first, const int8_t *second,
    const int8_t *third, const int8_t *fourth, int64_t size, int64_t *output) {
  const int8_t *candidates[4] = {first, second, third, fourth};
  int64_t totals[4] = {0, 0, 0, 0};
  while (size >= 32) {
    int64_t count = size < 65536 ? (size & ~31LL) : 65536;
    __m256i low[4], high[4];
    for (int j = 0; j < 4; ++j) {
      low[j] = _mm256_setzero_si256();
      high[j] = _mm256_setzero_si256();
    }
    for (int64_t i = 0; i < count; i += 32) {
      __m256i q = _mm256_loadu_si256((const __m256i *)(query + i));
      __m256i q_low = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(q));
      __m256i q_high = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(q, 1));
      for (int j = 0; j < 4; ++j) {
        __m256i v = _mm256_loadu_si256((const __m256i *)(candidates[j] + i));
        __m256i v_low = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(v));
        __m256i v_high = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(v, 1));
        low[j] = _mm256_add_epi32(low[j], _mm256_madd_epi16(q_low, v_low));
        high[j] = _mm256_add_epi32(high[j], _mm256_madd_epi16(q_high, v_high));
      }
    }
    for (int j = 0; j < 4; ++j) {
      __m256i sum = _mm256_add_epi32(low[j], high[j]);
      __m128i half = _mm_add_epi32(_mm256_castsi256_si128(sum), _mm256_extracti128_si256(sum, 1));
      half = _mm_hadd_epi32(half, half);
      half = _mm_hadd_epi32(half, half);
      totals[j] += _mm_cvtsi128_si32(half);
      candidates[j] += count;
    }
    query += count;
    size -= count;
  }
  for (int64_t i = 0; i < size; ++i) {
    for (int j = 0; j < 4; ++j) {
      totals[j] += (int64_t)query[i] * candidates[j][i];
    }
  }
  for (int j = 0; j < 4; ++j) output[j] = totals[j];
}
