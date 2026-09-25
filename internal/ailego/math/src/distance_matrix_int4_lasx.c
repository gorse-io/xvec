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

#include <lasxintrin.h>

static inline __m256i decode_low_lasx(__m256i packed) {
  const __m256i sign = __lasx_xvrepli_b(8);
  __m256i nibble = __lasx_xvand_v(packed, __lasx_xvrepli_b(15));
  return __lasx_xvsub_b(__lasx_xvxor_v(nibble, sign), sign);
}

static inline __m256i decode_high_lasx(__m256i packed) {
  const __m256i sign = __lasx_xvrepli_b(8);
  __m256i nibble = __lasx_xvsrli_b(packed, 4);
  return __lasx_xvsub_b(__lasx_xvxor_v(nibble, sign), sign);
}

static inline __m256i multiply_add_int8_lasx(__m256i left,
                                             __m256i right) {
  const __m256i zero = __lasx_xvrepli_w(0);
  __m256i even = __lasx_xvmulwev_h_b(left, right);
  __m256i odd = __lasx_xvmulwod_h_b(left, right);
  __m256i even32 = __lasx_xvadd_w(__lasx_xvaddwev_w_h(even, zero),
                                  __lasx_xvaddwod_w_h(even, zero));
  __m256i odd32 = __lasx_xvadd_w(__lasx_xvaddwev_w_h(odd, zero),
                                 __lasx_xvaddwod_w_h(odd, zero));
  return __lasx_xvadd_w(even32, odd32);
}

static inline long horizontal_sum_int32_lasx(__m256i values) {
  int lanes[8];
  __lasx_xvst(values, lanes, 0);
  long sum = 0;
  for (int lane = 0; lane < 8; ++lane) {
    sum += lanes[lane];
  }
  return sum;
}

long inner_product_int4_lasx(const unsigned char *left, const unsigned char *right,
                                long size) {
  long result = 0;
  while (size != 0) {
    long count = size < 1048576 ? size : 1048576;
    __m256i sum = __lasx_xvrepli_w(0);
    for (long i = 0; i < count; i += 32) {
      __m256i packed_left = __lasx_xvld(left + i, 0);
      __m256i packed_right = __lasx_xvld(right + i, 0);
      sum = __lasx_xvadd_w(
          sum, multiply_add_int8_lasx(decode_low_lasx(packed_left),
                                      decode_low_lasx(packed_right)));
      sum = __lasx_xvadd_w(
          sum, multiply_add_int8_lasx(decode_high_lasx(packed_left),
                                      decode_high_lasx(packed_right)));
    }
    result += horizontal_sum_int32_lasx(sum);
    left += count;
    right += count;
    size -= count;
  }
  return result;
}

long squared_euclidean_int4_lasx(const unsigned char *left,
                                    const unsigned char *right, long size) {
  long result = 0;
  while (size != 0) {
    long count = size < 1048576 ? size : 1048576;
    __m256i sum = __lasx_xvrepli_w(0);
    for (long i = 0; i < count; i += 32) {
      __m256i packed_left = __lasx_xvld(left + i, 0);
      __m256i packed_right = __lasx_xvld(right + i, 0);
      __m256i low_difference = __lasx_xvsub_b(
          decode_low_lasx(packed_left), decode_low_lasx(packed_right));
      __m256i high_difference = __lasx_xvsub_b(
          decode_high_lasx(packed_left), decode_high_lasx(packed_right));
      sum = __lasx_xvadd_w(
          sum, multiply_add_int8_lasx(low_difference, low_difference));
      sum = __lasx_xvadd_w(
          sum, multiply_add_int8_lasx(high_difference, high_difference));
    }
    result += horizontal_sum_int32_lasx(sum);
    left += count;
    right += count;
    size -= count;
  }
  return result;
}

void dot_norms_int4_lasx(const unsigned char *left, const unsigned char *right,
                         long size, long *out) {
  long dot = 0;
  long left_norm = 0;
  long right_norm = 0;
  while (size != 0) {
    long count = size < 1048576 ? size : 1048576;
    __m256i dot_sum = __lasx_xvrepli_w(0);
    __m256i left_sum = __lasx_xvrepli_w(0);
    __m256i right_sum = __lasx_xvrepli_w(0);
    for (long i = 0; i < count; i += 32) {
      __m256i packed_left = __lasx_xvld(left + i, 0);
      __m256i packed_right = __lasx_xvld(right + i, 0);
      __m256i left_low = decode_low_lasx(packed_left);
      __m256i left_high = decode_high_lasx(packed_left);
      __m256i right_low = decode_low_lasx(packed_right);
      __m256i right_high = decode_high_lasx(packed_right);
      dot_sum = __lasx_xvadd_w(
          dot_sum, multiply_add_int8_lasx(left_low, right_low));
      dot_sum = __lasx_xvadd_w(
          dot_sum, multiply_add_int8_lasx(left_high, right_high));
      left_sum = __lasx_xvadd_w(
          left_sum, multiply_add_int8_lasx(left_low, left_low));
      left_sum = __lasx_xvadd_w(
          left_sum, multiply_add_int8_lasx(left_high, left_high));
      right_sum = __lasx_xvadd_w(
          right_sum, multiply_add_int8_lasx(right_low, right_low));
      right_sum = __lasx_xvadd_w(
          right_sum, multiply_add_int8_lasx(right_high, right_high));
    }
    dot += horizontal_sum_int32_lasx(dot_sum);
    left_norm += horizontal_sum_int32_lasx(left_sum);
    right_norm += horizontal_sum_int32_lasx(right_sum);
    left += count;
    right += count;
    size -= count;
  }
  out[0] = dot;
  out[1] = left_norm;
  out[2] = right_norm;
}
