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

static inline int64_t horizontal_sum_int32_sse41(__m128i value) {
  value = _mm_hadd_epi32(value, value);
  value = _mm_hadd_epi32(value, value);
  return (int64_t)_mm_cvtsi128_si32(value);
}

static inline __m128i dot_int8_vectors_sse41(__m128i left, __m128i right) {
  __m128i left_lo = _mm_cvtepi8_epi16(left);
  __m128i left_hi = _mm_cvtepi8_epi16(_mm_srli_si128(left, 8));
  __m128i right_lo = _mm_cvtepi8_epi16(right);
  __m128i right_hi = _mm_cvtepi8_epi16(_mm_srli_si128(right, 8));
  return _mm_add_epi32(_mm_madd_epi16(left_lo, right_lo),
                       _mm_madd_epi16(left_hi, right_hi));
}

static inline __m128i sign_extend_nibbles_sse41(__m128i value,
                                                 __m128i sign) {
  return _mm_sub_epi8(_mm_xor_si128(value, sign), sign);
}

static inline __m128i dot_int4_vectors_sse41(__m128i left, __m128i right,
                                              __m128i mask, __m128i sign) {
  __m128i left_low = sign_extend_nibbles_sse41(_mm_and_si128(left, mask), sign);
  __m128i left_high = sign_extend_nibbles_sse41(
      _mm_and_si128(_mm_srli_epi16(left, 4), mask), sign);
  __m128i right_low = sign_extend_nibbles_sse41(_mm_and_si128(right, mask), sign);
  __m128i right_high = sign_extend_nibbles_sse41(
      _mm_and_si128(_mm_srli_epi16(right, 4), mask), sign);
  return _mm_add_epi32(dot_int8_vectors_sse41(left_low, right_low),
                       dot_int8_vectors_sse41(left_high, right_high));
}

static inline __m128i squared_l2_int4_vectors_sse41(
    __m128i left, __m128i right, __m128i mask, __m128i sign) {
  __m128i left_low = sign_extend_nibbles_sse41(_mm_and_si128(left, mask), sign);
  __m128i left_high = sign_extend_nibbles_sse41(
      _mm_and_si128(_mm_srli_epi16(left, 4), mask), sign);
  __m128i right_low = sign_extend_nibbles_sse41(_mm_and_si128(right, mask), sign);
  __m128i right_high = sign_extend_nibbles_sse41(
      _mm_and_si128(_mm_srli_epi16(right, 4), mask), sign);
  __m128i low = _mm_sub_epi16(_mm_cvtepi8_epi16(left_low),
                              _mm_cvtepi8_epi16(right_low));
  __m128i high = _mm_sub_epi16(_mm_cvtepi8_epi16(left_high),
                               _mm_cvtepi8_epi16(right_high));
  __m128i sum = _mm_add_epi32(_mm_madd_epi16(low, low),
                              _mm_madd_epi16(high, high));
  low = _mm_sub_epi16(_mm_cvtepi8_epi16(_mm_srli_si128(left_low, 8)),
                      _mm_cvtepi8_epi16(_mm_srli_si128(right_low, 8)));
  high = _mm_sub_epi16(_mm_cvtepi8_epi16(_mm_srli_si128(left_high, 8)),
                       _mm_cvtepi8_epi16(_mm_srli_si128(right_high, 8)));
  return _mm_add_epi32(sum, _mm_add_epi32(_mm_madd_epi16(low, low),
                                           _mm_madd_epi16(high, high)));
}

static inline void accumulate_int8_sse41(__m128i left, __m128i right,
                                         __m128i *dot, __m128i *left_norm,
                                         __m128i *right_norm) {
  __m128i left_lo = _mm_cvtepi8_epi16(left);
  __m128i left_hi = _mm_cvtepi8_epi16(_mm_srli_si128(left, 8));
  __m128i right_lo = _mm_cvtepi8_epi16(right);
  __m128i right_hi = _mm_cvtepi8_epi16(_mm_srli_si128(right, 8));
  *dot = _mm_add_epi32(*dot, _mm_madd_epi16(left_lo, right_lo));
  *dot = _mm_add_epi32(*dot, _mm_madd_epi16(left_hi, right_hi));
  *left_norm = _mm_add_epi32(*left_norm, _mm_madd_epi16(left_lo, left_lo));
  *left_norm = _mm_add_epi32(*left_norm, _mm_madd_epi16(left_hi, left_hi));
  *right_norm = _mm_add_epi32(*right_norm, _mm_madd_epi16(right_lo, right_lo));
  *right_norm = _mm_add_epi32(*right_norm, _mm_madd_epi16(right_hi, right_hi));
}

static inline void accumulate_int4_sse41(__m128i left, __m128i right,
                                         __m128i *dot, __m128i *left_norm,
                                         __m128i *right_norm, __m128i mask,
                                         __m128i sign) {
  *dot = _mm_add_epi32(*dot, dot_int4_vectors_sse41(left, right, mask, sign));
  *left_norm = _mm_add_epi32(
      *left_norm, dot_int4_vectors_sse41(left, left, mask, sign));
  *right_norm = _mm_add_epi32(
      *right_norm, dot_int4_vectors_sse41(right, right, mask, sign));
}

int64_t inner_product_int8_sse41(const int8_t *lhs, const int8_t *rhs,
                                  int64_t size) {
  int64_t result = 0;
  while (size >= 16) {
    int64_t count = size < 32768 ? (size & ~15LL) : 32768;
    __m128i sum = _mm_setzero_si128();
    for (int64_t i = 0; i < count; i += 16) {
      __m128i left = _mm_loadu_si128((const __m128i *)(lhs + i));
      __m128i right = _mm_loadu_si128((const __m128i *)(rhs + i));
      sum = _mm_add_epi32(sum, dot_int8_vectors_sse41(left, right));
    }
    result += horizontal_sum_int32_sse41(sum);
    lhs += count;
    rhs += count;
    size -= count;
  }
  return result;
}

int64_t squared_euclidean_int8_sse41(const int8_t *lhs, const int8_t *rhs,
                                      int64_t size) {
  int64_t result = 0;
  while (size >= 16) {
    int64_t count = size < 32768 ? (size & ~15LL) : 32768;
    __m128i sum = _mm_setzero_si128();
    for (int64_t i = 0; i < count; i += 16) {
      __m128i left = _mm_loadu_si128((const __m128i *)(lhs + i));
      __m128i right = _mm_loadu_si128((const __m128i *)(rhs + i));
      __m128i difference =
          _mm_sub_epi8(_mm_max_epi8(left, right), _mm_min_epi8(left, right));
      __m128i low = _mm_cvtepu8_epi16(difference);
      __m128i high = _mm_cvtepu8_epi16(_mm_srli_si128(difference, 8));
      sum = _mm_add_epi32(sum, _mm_madd_epi16(low, low));
      sum = _mm_add_epi32(sum, _mm_madd_epi16(high, high));
    }
    result += horizontal_sum_int32_sse41(sum);
    lhs += count;
    rhs += count;
    size -= count;
  }
  return result;
}

int64_t inner_product_int4_sse41(const uint8_t *lhs, const uint8_t *rhs,
                                  int64_t size, int64_t mask_word,
                                  int64_t sign_word) {
  int64_t result = 0;
  const __m128i mask = _mm_set1_epi64x((int64_t)mask_word);
  const __m128i sign = _mm_set1_epi64x((int64_t)sign_word);
  while (size >= 16) {
    int64_t count = size < 1048576 ? (size & ~15LL) : 1048576;
    __m128i sum = _mm_setzero_si128();
    for (int64_t i = 0; i < count; i += 16) {
      __m128i left = _mm_loadu_si128((const __m128i *)(lhs + i));
      __m128i right = _mm_loadu_si128((const __m128i *)(rhs + i));
      sum = _mm_add_epi32(sum,
                          dot_int4_vectors_sse41(left, right, mask, sign));
    }
    result += horizontal_sum_int32_sse41(sum);
    lhs += count;
    rhs += count;
    size -= count;
  }
  return result;
}

int64_t squared_euclidean_int4_sse41(const uint8_t *lhs, const uint8_t *rhs,
                                      int64_t size, int64_t mask_word,
                                      int64_t sign_word) {
  int64_t result = 0;
  const __m128i mask = _mm_set1_epi64x((int64_t)mask_word);
  const __m128i sign = _mm_set1_epi64x((int64_t)sign_word);
  while (size >= 16) {
    int64_t count = size < 1048576 ? (size & ~15LL) : 1048576;
    __m128i sum = _mm_setzero_si128();
    for (int64_t i = 0; i < count; i += 16) {
      __m128i left = _mm_loadu_si128((const __m128i *)(lhs + i));
      __m128i right = _mm_loadu_si128((const __m128i *)(rhs + i));
      sum = _mm_add_epi32(
          sum, squared_l2_int4_vectors_sse41(left, right, mask, sign));
    }
    result += horizontal_sum_int32_sse41(sum);
    lhs += count;
    rhs += count;
    size -= count;
  }
  return result;
}

void dot_norms_int8_sse41(const int8_t *lhs, const int8_t *rhs, int64_t size,
                          int64_t *dot_out, int64_t *lhs_norm_out,
                          int64_t *rhs_norm_out) {
  int64_t dot_result = 0;
  int64_t lhs_norm_result = 0;
  int64_t rhs_norm_result = 0;
  while (size >= 16) {
    int64_t count = size < 32768 ? (size & ~15LL) : 32768;
    __m128i dot = _mm_setzero_si128();
    __m128i lhs_norm = _mm_setzero_si128();
    __m128i rhs_norm = _mm_setzero_si128();
    for (int64_t i = 0; i < count; i += 16) {
      __m128i left = _mm_loadu_si128((const __m128i *)(lhs + i));
      __m128i right = _mm_loadu_si128((const __m128i *)(rhs + i));
      accumulate_int8_sse41(left, right, &dot, &lhs_norm, &rhs_norm);
    }
    dot_result += horizontal_sum_int32_sse41(dot);
    lhs_norm_result += horizontal_sum_int32_sse41(lhs_norm);
    rhs_norm_result += horizontal_sum_int32_sse41(rhs_norm);
    lhs += count;
    rhs += count;
    size -= count;
  }
  *dot_out = dot_result;
  *lhs_norm_out = lhs_norm_result;
  *rhs_norm_out = rhs_norm_result;
}

void dot_norms_int4_sse41(const uint8_t *lhs, const uint8_t *rhs, int64_t size,
                          int64_t *out, int64_t mask_word,
                          int64_t sign_word) {
  int64_t dot_result = 0;
  int64_t lhs_norm_result = 0;
  int64_t rhs_norm_result = 0;
  const __m128i mask = _mm_set1_epi64x((int64_t)mask_word);
  const __m128i sign = _mm_set1_epi64x((int64_t)sign_word);
  while (size >= 16) {
    int64_t count = size < 1048576 ? (size & ~15LL) : 1048576;
    __m128i dot = _mm_setzero_si128();
    __m128i lhs_norm = _mm_setzero_si128();
    __m128i rhs_norm = _mm_setzero_si128();
    for (int64_t i = 0; i < count; i += 16) {
      __m128i left = _mm_loadu_si128((const __m128i *)(lhs + i));
      __m128i right = _mm_loadu_si128((const __m128i *)(rhs + i));
      accumulate_int4_sse41(left, right, &dot, &lhs_norm, &rhs_norm, mask,
                            sign);
    }
    dot_result += horizontal_sum_int32_sse41(dot);
    lhs_norm_result += horizontal_sum_int32_sse41(lhs_norm);
    rhs_norm_result += horizontal_sum_int32_sse41(rhs_norm);
    lhs += count;
    rhs += count;
    size -= count;
  }
  out[0] = dot_result;
  out[1] = lhs_norm_result;
  out[2] = rhs_norm_result;
}
