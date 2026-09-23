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

int64_t inner_product_int8_neon(const int8_t *lhs, const int8_t *rhs,
                               int64_t size) {
  int64_t result = 0;
  while (size >= 16) {
    // Even an all -128 dot product fits in int32 for each 65536-byte chunk.
    // Reduce chunks to int64 before accumulating larger dimensions.
    int64_t count = size < 65536 ? (size & ~15LL) : 65536;
    int32x4_t sum_lo = vdupq_n_s32(0);
    int32x4_t sum_hi = vdupq_n_s32(0);
    for (int64_t i = 0; i < count; i += 16) {
      int8x16_t left = vld1q_s8(lhs + i);
      int8x16_t right = vld1q_s8(rhs + i);
      int16x8_t left_lo = vmovl_s8(vget_low_s8(left));
      int16x8_t left_hi = vmovl_s8(vget_high_s8(left));
      int16x8_t right_lo = vmovl_s8(vget_low_s8(right));
      int16x8_t right_hi = vmovl_s8(vget_high_s8(right));
      sum_lo = vmlal_s16(sum_lo, vget_low_s16(left_lo),
                         vget_low_s16(right_lo));
      sum_hi = vmlal_s16(sum_hi, vget_high_s16(left_lo),
                         vget_high_s16(right_lo));
      sum_lo = vmlal_s16(sum_lo, vget_low_s16(left_hi),
                         vget_low_s16(right_hi));
      sum_hi = vmlal_s16(sum_hi, vget_high_s16(left_hi),
                         vget_high_s16(right_hi));
    }
    int32x4_t sum = vaddq_s32(sum_lo, sum_hi);
    int64x2_t pair = vpaddlq_s32(sum);
    result += vgetq_lane_s64(pair, 0) + vgetq_lane_s64(pair, 1);
    lhs += count;
    rhs += count;
    size -= count;
  }
  for (int64_t i = 0; i < size; ++i) {
    result += (int64_t)lhs[i] * rhs[i];
  }
  return result;
}
