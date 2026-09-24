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

static inline int64_t horizontal_sum_int32_avx2(__m256i value) {
  __m128i half = _mm_add_epi32(_mm256_castsi256_si128(value),
                               _mm256_extracti128_si256(value, 1));
  half = _mm_hadd_epi32(half, half);
  half = _mm_hadd_epi32(half, half);
  return (int64_t)_mm_cvtsi128_si32(half);
}

static inline __m256i dot_int8_vectors_avx2(__m256i left, __m256i right) {
  __m256i left_lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(left));
  __m256i left_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(left, 1));
  __m256i right_lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(right));
  __m256i right_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(right, 1));
  return _mm256_add_epi32(_mm256_madd_epi16(left_lo, right_lo),
                          _mm256_madd_epi16(left_hi, right_hi));
}

static inline __m256i sign_extend_nibbles_avx2(__m256i value,
                                                __m256i sign) {
  return _mm256_sub_epi8(_mm256_xor_si256(value, sign), sign);
}

static inline __m256i dot_int4_vectors_avx2(__m256i left, __m256i right,
                                             __m256i mask, __m256i sign) {
  __m256i left_low = sign_extend_nibbles_avx2(_mm256_and_si256(left, mask), sign);
  __m256i left_high = sign_extend_nibbles_avx2(
      _mm256_and_si256(_mm256_srli_epi16(left, 4), mask), sign);
  __m256i right_low = sign_extend_nibbles_avx2(_mm256_and_si256(right, mask), sign);
  __m256i right_high = sign_extend_nibbles_avx2(
      _mm256_and_si256(_mm256_srli_epi16(right, 4), mask), sign);
  return _mm256_add_epi32(dot_int8_vectors_avx2(left_low, right_low),
                          dot_int8_vectors_avx2(left_high, right_high));
}

static inline __m256i squared_l2_int4_vectors_avx2(
    __m256i left, __m256i right, __m256i mask, __m256i sign) {
  __m256i left_low = sign_extend_nibbles_avx2(_mm256_and_si256(left, mask), sign);
  __m256i left_high = sign_extend_nibbles_avx2(
      _mm256_and_si256(_mm256_srli_epi16(left, 4), mask), sign);
  __m256i right_low = sign_extend_nibbles_avx2(_mm256_and_si256(right, mask), sign);
  __m256i right_high = sign_extend_nibbles_avx2(
      _mm256_and_si256(_mm256_srli_epi16(right, 4), mask), sign);
  __m256i low = _mm256_sub_epi16(
      _mm256_cvtepi8_epi16(_mm256_castsi256_si128(left_low)),
      _mm256_cvtepi8_epi16(_mm256_castsi256_si128(right_low)));
  __m256i high = _mm256_sub_epi16(
      _mm256_cvtepi8_epi16(_mm256_castsi256_si128(left_high)),
      _mm256_cvtepi8_epi16(_mm256_castsi256_si128(right_high)));
  __m256i sum = _mm256_add_epi32(_mm256_madd_epi16(low, low),
                                 _mm256_madd_epi16(high, high));
  low = _mm256_sub_epi16(
      _mm256_cvtepi8_epi16(_mm256_extracti128_si256(left_low, 1)),
      _mm256_cvtepi8_epi16(_mm256_extracti128_si256(right_low, 1)));
  high = _mm256_sub_epi16(
      _mm256_cvtepi8_epi16(_mm256_extracti128_si256(left_high, 1)),
      _mm256_cvtepi8_epi16(_mm256_extracti128_si256(right_high, 1)));
  return _mm256_add_epi32(
      sum, _mm256_add_epi32(_mm256_madd_epi16(low, low),
                            _mm256_madd_epi16(high, high)));
}

static inline void accumulate_int8_avx2(__m256i left, __m256i right,
                                        __m256i *dot, __m256i *left_norm,
                                        __m256i *right_norm) {
  __m256i left_lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(left));
  __m256i left_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(left, 1));
  __m256i right_lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(right));
  __m256i right_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(right, 1));
  *dot = _mm256_add_epi32(*dot, _mm256_madd_epi16(left_lo, right_lo));
  *dot = _mm256_add_epi32(*dot, _mm256_madd_epi16(left_hi, right_hi));
  *left_norm = _mm256_add_epi32(*left_norm, _mm256_madd_epi16(left_lo, left_lo));
  *left_norm = _mm256_add_epi32(*left_norm, _mm256_madd_epi16(left_hi, left_hi));
  *right_norm = _mm256_add_epi32(*right_norm, _mm256_madd_epi16(right_lo, right_lo));
  *right_norm = _mm256_add_epi32(*right_norm, _mm256_madd_epi16(right_hi, right_hi));
}

static inline void accumulate_int4_avx2(__m256i left, __m256i right,
                                        __m256i *dot, __m256i *left_norm,
                                        __m256i *right_norm, __m256i mask,
                                        __m256i sign) {
  *dot = _mm256_add_epi32(*dot,
                          dot_int4_vectors_avx2(left, right, mask, sign));
  *left_norm = _mm256_add_epi32(
      *left_norm, dot_int4_vectors_avx2(left, left, mask, sign));
  *right_norm = _mm256_add_epi32(
      *right_norm, dot_int4_vectors_avx2(right, right, mask, sign));
}

int64_t squared_euclidean_int8_avx2(const int8_t *lhs, const int8_t *rhs,
                                     int64_t size) {
  int64_t result = 0;
  while (size >= 32) {
    int64_t count = size < 32768 ? (size & ~31LL) : 32768;
    __m256i sum = _mm256_setzero_si256();
    for (int64_t i = 0; i < count; i += 32) {
      __m256i left = _mm256_loadu_si256((const __m256i *)(lhs + i));
      __m256i right = _mm256_loadu_si256((const __m256i *)(rhs + i));
      __m256i difference = _mm256_sub_epi8(_mm256_max_epi8(left, right),
                                            _mm256_min_epi8(left, right));
      __m256i low =
          _mm256_cvtepu8_epi16(_mm256_castsi256_si128(difference));
      __m256i high =
          _mm256_cvtepu8_epi16(_mm256_extracti128_si256(difference, 1));
      sum = _mm256_add_epi32(sum, _mm256_madd_epi16(low, low));
      sum = _mm256_add_epi32(sum, _mm256_madd_epi16(high, high));
    }
    result += horizontal_sum_int32_avx2(sum);
    lhs += count;
    rhs += count;
    size -= count;
  }
  return result;
}

int64_t inner_product_int4_avx2(const uint8_t *lhs, const uint8_t *rhs,
                                 int64_t size, int64_t mask_word,
                                 int64_t sign_word) {
  int64_t result = 0;
  const __m256i mask = _mm256_set1_epi64x((int64_t)mask_word);
  const __m256i sign = _mm256_set1_epi64x((int64_t)sign_word);
  while (size >= 32) {
    int64_t count = size < 1048576 ? (size & ~31LL) : 1048576;
    __m256i sum = _mm256_setzero_si256();
    for (int64_t i = 0; i < count; i += 32) {
      __m256i left = _mm256_loadu_si256((const __m256i *)(lhs + i));
      __m256i right = _mm256_loadu_si256((const __m256i *)(rhs + i));
      sum = _mm256_add_epi32(
          sum, dot_int4_vectors_avx2(left, right, mask, sign));
    }
    result += horizontal_sum_int32_avx2(sum);
    lhs += count;
    rhs += count;
    size -= count;
  }
  return result;
}

int64_t squared_euclidean_int4_avx2(const uint8_t *lhs, const uint8_t *rhs,
                                     int64_t size, int64_t mask_word,
                                     int64_t sign_word) {
  int64_t result = 0;
  const __m256i mask = _mm256_set1_epi64x((int64_t)mask_word);
  const __m256i sign = _mm256_set1_epi64x((int64_t)sign_word);
  while (size >= 32) {
    int64_t count = size < 1048576 ? (size & ~31LL) : 1048576;
    __m256i sum = _mm256_setzero_si256();
    for (int64_t i = 0; i < count; i += 32) {
      __m256i left = _mm256_loadu_si256((const __m256i *)(lhs + i));
      __m256i right = _mm256_loadu_si256((const __m256i *)(rhs + i));
      sum = _mm256_add_epi32(
          sum, squared_l2_int4_vectors_avx2(left, right, mask, sign));
    }
    result += horizontal_sum_int32_avx2(sum);
    lhs += count;
    rhs += count;
    size -= count;
  }
  return result;
}

void dot_norms_int8_avx2(const int8_t *lhs, const int8_t *rhs, int64_t size,
                         int64_t *dot_out, int64_t *lhs_norm_out,
                         int64_t *rhs_norm_out) {
  int64_t dot_result = 0;
  int64_t lhs_norm_result = 0;
  int64_t rhs_norm_result = 0;
  while (size >= 32) {
    int64_t count = size < 65536 ? (size & ~31LL) : 65536;
    __m256i dot = _mm256_setzero_si256();
    __m256i lhs_norm = _mm256_setzero_si256();
    __m256i rhs_norm = _mm256_setzero_si256();
    for (int64_t i = 0; i < count; i += 32) {
      __m256i left = _mm256_loadu_si256((const __m256i *)(lhs + i));
      __m256i right = _mm256_loadu_si256((const __m256i *)(rhs + i));
      accumulate_int8_avx2(left, right, &dot, &lhs_norm, &rhs_norm);
    }
    dot_result += horizontal_sum_int32_avx2(dot);
    lhs_norm_result += horizontal_sum_int32_avx2(lhs_norm);
    rhs_norm_result += horizontal_sum_int32_avx2(rhs_norm);
    lhs += count;
    rhs += count;
    size -= count;
  }
  *dot_out = dot_result;
  *lhs_norm_out = lhs_norm_result;
  *rhs_norm_out = rhs_norm_result;
}

void dot_norms_int4_avx2(const uint8_t *lhs, const uint8_t *rhs, int64_t size,
                         int64_t *out, int64_t mask_word,
                         int64_t sign_word) {
  int64_t dot_result = 0;
  int64_t lhs_norm_result = 0;
  int64_t rhs_norm_result = 0;
  const __m256i mask = _mm256_set1_epi64x((int64_t)mask_word);
  const __m256i sign = _mm256_set1_epi64x((int64_t)sign_word);
  while (size >= 32) {
    int64_t count = size < 1048576 ? (size & ~31LL) : 1048576;
    __m256i dot = _mm256_setzero_si256();
    __m256i lhs_norm = _mm256_setzero_si256();
    __m256i rhs_norm = _mm256_setzero_si256();
    for (int64_t i = 0; i < count; i += 32) {
      __m256i left = _mm256_loadu_si256((const __m256i *)(lhs + i));
      __m256i right = _mm256_loadu_si256((const __m256i *)(rhs + i));
      accumulate_int4_avx2(left, right, &dot, &lhs_norm, &rhs_norm, mask,
                           sign);
    }
    dot_result += horizontal_sum_int32_avx2(dot);
    lhs_norm_result += horizontal_sum_int32_avx2(lhs_norm);
    rhs_norm_result += horizontal_sum_int32_avx2(rhs_norm);
    lhs += count;
    rhs += count;
    size -= count;
  }
  out[0] = dot_result;
  out[1] = lhs_norm_result;
  out[2] = rhs_norm_result;
}
