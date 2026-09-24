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

static inline __m256i decode_int4_low(__m256i packed, __m256i mask,
                                      __m256i sign) {
  __m256i nibble = _mm256_and_si256(packed, mask);
  return _mm256_sub_epi8(_mm256_xor_si256(nibble, sign), sign);
}

static inline __m256i decode_int4_high(__m256i packed, __m256i mask,
                                       __m256i sign) {
  __m256i nibble = _mm256_and_si256(_mm256_srli_epi16(packed, 4), mask);
  return _mm256_sub_epi8(_mm256_xor_si256(nibble, sign), sign);
}

static inline __m256i multiply_add_int8(__m256i left, __m256i right) {
  __m256i left_lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(left));
  __m256i left_hi =
      _mm256_cvtepi8_epi16(_mm256_extracti128_si256(left, 1));
  __m256i right_lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(right));
  __m256i right_hi =
      _mm256_cvtepi8_epi16(_mm256_extracti128_si256(right, 1));
  return _mm256_add_epi32(_mm256_madd_epi16(left_lo, right_lo),
                          _mm256_madd_epi16(left_hi, right_hi));
}

static inline int64_t horizontal_sum_int32(__m256i values) {
  __m128i sum = _mm_add_epi32(_mm256_castsi256_si128(values),
                              _mm256_extracti128_si256(values, 1));
  sum = _mm_hadd_epi32(sum, sum);
  sum = _mm_hadd_epi32(sum, sum);
  return (int64_t)_mm_cvtsi128_si32(sum);
}

int64_t inner_product_int4_avx2(const uint8_t *left, const uint8_t *right,
                                int64_t size, int64_t mask_bits,
                                int64_t sign_bits) {
  const __m256i mask = _mm256_set1_epi64x(mask_bits);
  const __m256i sign = _mm256_set1_epi64x(sign_bits);
  int64_t result = 0;
  while (size != 0) {
    int64_t count = size < 1048576 ? size : 1048576;
    __m256i sum = _mm256_setzero_si256();
    for (int64_t i = 0; i < count; i += 32) {
      __m256i packed_left =
          _mm256_loadu_si256((const __m256i *)(left + i));
      __m256i packed_right =
          _mm256_loadu_si256((const __m256i *)(right + i));
      sum = _mm256_add_epi32(
          sum, multiply_add_int8(decode_int4_low(packed_left, mask, sign),
                                 decode_int4_low(packed_right, mask, sign)));
      sum = _mm256_add_epi32(
          sum, multiply_add_int8(decode_int4_high(packed_left, mask, sign),
                                 decode_int4_high(packed_right, mask, sign)));
    }
    result += horizontal_sum_int32(sum);
    left += count;
    right += count;
    size -= count;
  }
  return result;
}

int64_t squared_euclidean_int4_avx2(const uint8_t *left,
                                    const uint8_t *right, int64_t size,
                                    int64_t mask_bits, int64_t sign_bits) {
  const __m256i mask = _mm256_set1_epi64x(mask_bits);
  const __m256i sign = _mm256_set1_epi64x(sign_bits);
  int64_t result = 0;
  while (size != 0) {
    int64_t count = size < 1048576 ? size : 1048576;
    __m256i sum = _mm256_setzero_si256();
    for (int64_t i = 0; i < count; i += 32) {
      __m256i packed_left =
          _mm256_loadu_si256((const __m256i *)(left + i));
      __m256i packed_right =
          _mm256_loadu_si256((const __m256i *)(right + i));
      __m256i low_difference = _mm256_sub_epi8(
          decode_int4_low(packed_left, mask, sign),
          decode_int4_low(packed_right, mask, sign));
      __m256i high_difference = _mm256_sub_epi8(
          decode_int4_high(packed_left, mask, sign),
          decode_int4_high(packed_right, mask, sign));
      sum = _mm256_add_epi32(
          sum, multiply_add_int8(low_difference, low_difference));
      sum = _mm256_add_epi32(
          sum, multiply_add_int8(high_difference, high_difference));
    }
    result += horizontal_sum_int32(sum);
    left += count;
    right += count;
    size -= count;
  }
  return result;
}

void dot_norms_int4_avx2(const uint8_t *left, const uint8_t *right,
                         int64_t size, int64_t mask_bits, int64_t sign_bits,
                         int64_t *out) {
  const __m256i mask = _mm256_set1_epi64x(mask_bits);
  const __m256i sign = _mm256_set1_epi64x(sign_bits);
  int64_t dot = 0;
  int64_t left_norm = 0;
  int64_t right_norm = 0;
  while (size != 0) {
    int64_t count = size < 1048576 ? size : 1048576;
    __m256i dot_sum = _mm256_setzero_si256();
    __m256i left_sum = _mm256_setzero_si256();
    __m256i right_sum = _mm256_setzero_si256();
    for (int64_t i = 0; i < count; i += 32) {
      __m256i packed_left =
          _mm256_loadu_si256((const __m256i *)(left + i));
      __m256i packed_right =
          _mm256_loadu_si256((const __m256i *)(right + i));
      __m256i left_low = decode_int4_low(packed_left, mask, sign);
      __m256i left_high = decode_int4_high(packed_left, mask, sign);
      __m256i right_low = decode_int4_low(packed_right, mask, sign);
      __m256i right_high = decode_int4_high(packed_right, mask, sign);
      dot_sum = _mm256_add_epi32(dot_sum,
                                 multiply_add_int8(left_low, right_low));
      dot_sum = _mm256_add_epi32(dot_sum,
                                 multiply_add_int8(left_high, right_high));
      left_sum = _mm256_add_epi32(left_sum,
                                  multiply_add_int8(left_low, left_low));
      left_sum = _mm256_add_epi32(left_sum,
                                  multiply_add_int8(left_high, left_high));
      right_sum = _mm256_add_epi32(right_sum,
                                   multiply_add_int8(right_low, right_low));
      right_sum = _mm256_add_epi32(
          right_sum, multiply_add_int8(right_high, right_high));
    }
    dot += horizontal_sum_int32(dot_sum);
    left_norm += horizontal_sum_int32(left_sum);
    right_norm += horizontal_sum_int32(right_sum);
    left += count;
    right += count;
    size -= count;
  }
  out[0] = dot;
  out[1] = left_norm;
  out[2] = right_norm;
}
