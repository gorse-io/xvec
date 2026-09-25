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

#include <arm_neon.h>
#include <stdint.h>

static inline int8x16_t decode_low_neon(uint8x16_t packed) {
  const uint8x16_t sign = vdupq_n_u8(8);
  uint8x16_t nibble = vandq_u8(packed, vdupq_n_u8(15));
  return vreinterpretq_s8_u8(vsubq_u8(veorq_u8(nibble, sign), sign));
}

static inline int8x16_t decode_high_neon(uint8x16_t packed) {
  const uint8x16_t sign = vdupq_n_u8(8);
  uint8x16_t nibble = vshrq_n_u8(packed, 4);
  return vreinterpretq_s8_u8(vsubq_u8(veorq_u8(nibble, sign), sign));
}

static inline int32x4_t multiply_add_int8_neon(int32x4_t sum,
                                               int8x16_t left,
                                               int8x16_t right) {
  sum = vpadalq_s16(sum,
                    vmull_s8(vget_low_s8(left), vget_low_s8(right)));
  return vpadalq_s16(sum, vmull_high_s8(left, right));
}

int64_t inner_product_int4_neon(const uint8_t *left, const uint8_t *right,
                                int64_t size) {
  int64_t result = 0;
  while (size != 0) {
    int64_t count = size < 1048576 ? size : 1048576;
    int32x4_t sum = vdupq_n_s32(0);
    for (int64_t i = 0; i < count; i += 16) {
      uint8x16_t packed_left = vld1q_u8(left + i);
      uint8x16_t packed_right = vld1q_u8(right + i);
      sum = multiply_add_int8_neon(
          sum, decode_low_neon(packed_left), decode_low_neon(packed_right));
      sum = multiply_add_int8_neon(
          sum, decode_high_neon(packed_left), decode_high_neon(packed_right));
    }
    result += (int64_t)vaddvq_s32(sum);
    left += count;
    right += count;
    size -= count;
  }
  return result;
}

int64_t squared_euclidean_int4_neon(const uint8_t *left,
                                    const uint8_t *right, int64_t size) {
  int64_t result = 0;
  while (size != 0) {
    int64_t count = size < 1048576 ? size : 1048576;
    int32x4_t sum = vdupq_n_s32(0);
    for (int64_t i = 0; i < count; i += 16) {
      uint8x16_t packed_left = vld1q_u8(left + i);
      uint8x16_t packed_right = vld1q_u8(right + i);
      int8x16_t low_difference = vsubq_s8(
          decode_low_neon(packed_left), decode_low_neon(packed_right));
      int8x16_t high_difference = vsubq_s8(
          decode_high_neon(packed_left), decode_high_neon(packed_right));
      sum = multiply_add_int8_neon(sum, low_difference, low_difference);
      sum = multiply_add_int8_neon(sum, high_difference, high_difference);
    }
    result += (int64_t)vaddvq_s32(sum);
    left += count;
    right += count;
    size -= count;
  }
  return result;
}

void dot_norms_int4_neon(const uint8_t *left, const uint8_t *right,
                         int64_t size, int64_t *out) {
  int64_t dot = 0;
  int64_t left_norm = 0;
  int64_t right_norm = 0;
  while (size != 0) {
    int64_t count = size < 1048576 ? size : 1048576;
    int32x4_t dot_sum = vdupq_n_s32(0);
    int32x4_t left_sum = vdupq_n_s32(0);
    int32x4_t right_sum = vdupq_n_s32(0);
    for (int64_t i = 0; i < count; i += 16) {
      uint8x16_t packed_left = vld1q_u8(left + i);
      uint8x16_t packed_right = vld1q_u8(right + i);
      int8x16_t left_low = decode_low_neon(packed_left);
      int8x16_t left_high = decode_high_neon(packed_left);
      int8x16_t right_low = decode_low_neon(packed_right);
      int8x16_t right_high = decode_high_neon(packed_right);
      dot_sum = multiply_add_int8_neon(dot_sum, left_low, right_low);
      dot_sum = multiply_add_int8_neon(dot_sum, left_high, right_high);
      left_sum = multiply_add_int8_neon(left_sum, left_low, left_low);
      left_sum = multiply_add_int8_neon(left_sum, left_high, left_high);
      right_sum = multiply_add_int8_neon(right_sum, right_low, right_low);
      right_sum = multiply_add_int8_neon(right_sum, right_high, right_high);
    }
    dot += (int64_t)vaddvq_s32(dot_sum);
    left_norm += (int64_t)vaddvq_s32(left_sum);
    right_norm += (int64_t)vaddvq_s32(right_sum);
    left += count;
    right += count;
    size -= count;
  }
  out[0] = dot;
  out[1] = left_norm;
  out[2] = right_norm;
}
