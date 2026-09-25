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

static inline int8x16_t batch_decode_low_neon(uint8x16_t packed) {
  const uint8x16_t sign = vdupq_n_u8(8);
  uint8x16_t nibble = vandq_u8(packed, vdupq_n_u8(15));
  return vreinterpretq_s8_u8(vsubq_u8(veorq_u8(nibble, sign), sign));
}

static inline int8x16_t batch_decode_high_neon(uint8x16_t packed) {
  const uint8x16_t sign = vdupq_n_u8(8);
  uint8x16_t nibble = vshrq_n_u8(packed, 4);
  return vreinterpretq_s8_u8(vsubq_u8(veorq_u8(nibble, sign), sign));
}

static inline int32x4_t batch_dot_bytes_neon(int32x4_t sum,
                                             int8x16_t left,
                                             int8x16_t right) {
  sum = vpadalq_s16(sum,
                    vmull_s8(vget_low_s8(left), vget_low_s8(right)));
  return vpadalq_s16(sum, vmull_high_s8(left, right));
}

void xvec_neon_batch_inner_products_int4_4(
    const uint8_t *query, const uint8_t *first, const uint8_t *second,
    const uint8_t *third, const uint8_t *fourth, int64_t size,
    int64_t *output) {
  const uint8_t *candidates[4] = {first, second, third, fourth};
  int64_t totals[4] = {0, 0, 0, 0};
  while (size != 0) {
    int64_t count = size < 1048576 ? size : 1048576;
    int32x4_t sums[4];
    for (int j = 0; j < 4; ++j) {
      sums[j] = vdupq_n_s32(0);
    }
    for (int64_t i = 0; i < count; i += 16) {
      uint8x16_t packed_query = vld1q_u8(query + i);
      int8x16_t query_low = batch_decode_low_neon(packed_query);
      int8x16_t query_high = batch_decode_high_neon(packed_query);
      for (int j = 0; j < 4; ++j) {
        uint8x16_t packed = vld1q_u8(candidates[j] + i);
        sums[j] = batch_dot_bytes_neon(
            sums[j], query_low, batch_decode_low_neon(packed));
        sums[j] = batch_dot_bytes_neon(
            sums[j], query_high, batch_decode_high_neon(packed));
      }
    }
    for (int j = 0; j < 4; ++j) {
      totals[j] += (int64_t)vaddvq_s32(sums[j]);
      candidates[j] += count;
    }
    query += count;
    size -= count;
  }
  for (int j = 0; j < 4; ++j) {
    output[j] = totals[j];
  }
}
