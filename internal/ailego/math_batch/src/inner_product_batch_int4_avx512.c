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

static inline __m256i decode_low_avx512(__m256i packed, __m256i mask,
                                        __m256i sign) {
  __m256i nibble = _mm256_and_si256(packed, mask);
  return _mm256_sub_epi8(_mm256_xor_si256(nibble, sign), sign);
}

static inline __m256i decode_high_avx512(__m256i packed, __m256i mask,
                                         __m256i sign) {
  __m256i nibble = _mm256_and_si256(_mm256_srli_epi16(packed, 4), mask);
  return _mm256_sub_epi8(_mm256_xor_si256(nibble, sign), sign);
}

static inline __m512i dot_bytes_avx512(__m256i left, __m256i right) {
  return _mm512_madd_epi16(_mm512_cvtepi8_epi16(left),
                           _mm512_cvtepi8_epi16(right));
}

void xvec_avx512_batch_inner_products_int4_4(
    const uint8_t *query, const uint8_t *first, const uint8_t *second,
    const uint8_t *third, const uint8_t *fourth, int64_t size,
    int64_t *output) {
  const uint8_t *candidates[4] = {first, second, third, fourth};
  const __m256i mask = _mm256_set1_epi8(0x0f);
  const __m256i sign = _mm256_set1_epi8(0x08);
  int64_t totals[4] = {0, 0, 0, 0};
  while (size != 0) {
    int64_t count = size < 1048576 ? size : 1048576;
    __m512i sums[4];
    for (int j = 0; j < 4; ++j) {
      sums[j] = _mm512_setzero_si512();
    }
    for (int64_t i = 0; i < count; i += 32) {
      __m256i packed_query =
          _mm256_loadu_si256((const __m256i *)(query + i));
      __m256i query_low = decode_low_avx512(packed_query, mask, sign);
      __m256i query_high = decode_high_avx512(packed_query, mask, sign);
      for (int j = 0; j < 4; ++j) {
        __m256i packed =
            _mm256_loadu_si256((const __m256i *)(candidates[j] + i));
        sums[j] = _mm512_add_epi32(
            sums[j], dot_bytes_avx512(
                         query_low, decode_low_avx512(packed, mask, sign)));
        sums[j] = _mm512_add_epi32(
            sums[j], dot_bytes_avx512(
                         query_high, decode_high_avx512(packed, mask, sign)));
      }
    }
    for (int j = 0; j < 4; ++j) {
      totals[j] += (int64_t)_mm512_reduce_add_epi32(sums[j]);
      candidates[j] += count;
    }
    query += count;
    size -= count;
  }
  for (int j = 0; j < 4; ++j) {
    output[j] = totals[j];
  }
}
