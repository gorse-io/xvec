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

static inline __m256i decode_low_avx2(__m256i packed, __m256i mask,
                                      __m256i sign) {
  __m256i nibble = _mm256_and_si256(packed, mask);
  return _mm256_sub_epi8(_mm256_xor_si256(nibble, sign), sign);
}

static inline __m256i decode_high_avx2(__m256i packed, __m256i mask,
                                       __m256i sign) {
  __m256i nibble = _mm256_and_si256(_mm256_srli_epi16(packed, 4), mask);
  return _mm256_sub_epi8(_mm256_xor_si256(nibble, sign), sign);
}

static inline __m256i dot_bytes_avx2(__m256i candidate, __m256i query,
                                     __m256i query_abs) {
  // Signed INT4 values are in [-8, 7]: neither negation nor maddubs
  // saturation can overflow. Reuse the decoded query across four candidates.
  return _mm256_madd_epi16(
      _mm256_maddubs_epi16(query_abs, _mm256_sign_epi8(candidate, query)),
      _mm256_set1_epi16(1));
}

static inline int64_t horizontal_sum_int32(__m256i values) {
  __m128i sum = _mm_add_epi32(_mm256_castsi256_si128(values),
                            _mm256_extracti128_si256(values, 1));
  sum = _mm_hadd_epi32(sum, sum);
  sum = _mm_hadd_epi32(sum, sum);
  return (int64_t)_mm_cvtsi128_si32(sum);
}

void xvec_avx2_batch_inner_products_int4_4(
    const uint8_t *query, const uint8_t *first, const uint8_t *second,
    const uint8_t *third, const uint8_t *fourth, int64_t size, int64_t *output) {
  const uint8_t *candidates[4] = {first, second, third, fourth};
  const __m256i mask = _mm256_set1_epi8(0x0f);
  const __m256i sign = _mm256_set1_epi8(0x08);
  int64_t totals[4] = {0, 0, 0, 0};
  while (size != 0) {
    // At most 2 * 8 * 8 * 2^20 per chunk, including the horizontal sum.
    int64_t count = size < 1048576 ? size : 1048576;
    __m256i sums[4];
    for (int j = 0; j < 4; ++j) {
      sums[j] = _mm256_setzero_si256();
    }
    for (int64_t i = 0; i < count; i += 32) {
      __m256i packed_query =
          _mm256_loadu_si256((const __m256i *)(query + i));
      __m256i query_low = decode_low_avx2(packed_query, mask, sign);
      __m256i query_high = decode_high_avx2(packed_query, mask, sign);
      __m256i query_low_abs = _mm256_abs_epi8(query_low);
      __m256i query_high_abs = _mm256_abs_epi8(query_high);
      for (int j = 0; j < 4; ++j) {
        __m256i packed =
            _mm256_loadu_si256((const __m256i *)(candidates[j] + i));
        sums[j] = _mm256_add_epi32(
            sums[j], dot_bytes_avx2(decode_low_avx2(packed, mask, sign),
                                    query_low, query_low_abs));
        sums[j] = _mm256_add_epi32(
            sums[j], dot_bytes_avx2(decode_high_avx2(packed, mask, sign),
                                    query_high, query_high_abs));
      }
    }
    for (int j = 0; j < 4; ++j) {
      totals[j] += horizontal_sum_int32(sums[j]);
      candidates[j] += count;
    }
    query += count;
    size -= count;
  }
  for (int j = 0; j < 4; ++j) {
    output[j] = totals[j];
  }
}
