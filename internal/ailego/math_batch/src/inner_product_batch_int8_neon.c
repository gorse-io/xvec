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

static inline int32x4_t batch_dot_int8_neon(int32x4_t sum, int8x16_t left,
                                             int8x16_t right) {
  sum = vpadalq_s16(sum, vmull_s8(vget_low_s8(left), vget_low_s8(right)));
  return vpadalq_s16(sum, vmull_high_s8(left, right));
}

void xvec_neon_batch_inner_products_int8_4(const int8_t *const *vectors,
                                            int64_t size, int64_t *output) {
  const int8_t *query = vectors[0];
  const int8_t *candidates[4] = {vectors[1], vectors[2], vectors[3], vectors[4]};
  int64_t totals[4] = {0, 0, 0, 0};
  while (size != 0) {
    int64_t count = size < 65536 ? size : 65536;
    int32x4_t sums[4];
    for (int j = 0; j < 4; ++j) sums[j] = vdupq_n_s32(0);
    for (int64_t i = 0; i < count; i += 16) {
      int8x16_t q = vld1q_s8(query + i);
      for (int j = 0; j < 4; ++j) {
        sums[j] = batch_dot_int8_neon(
            sums[j], q, vld1q_s8(candidates[j] + i));
      }
    }
    for (int j = 0; j < 4; ++j) {
      totals[j] += (int64_t)vaddvq_s32(sums[j]);
      candidates[j] += count;
    }
    query += count;
    size -= count;
  }
  for (int j = 0; j < 4; ++j) output[j] = totals[j];
}
