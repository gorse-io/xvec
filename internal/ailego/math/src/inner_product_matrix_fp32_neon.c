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

float inner_product_fp32_neon(const float *lhs, const float *rhs, int64_t size) {
    const float *last = lhs + size;
    const float *last_aligned = lhs + ((size >> 3) << 3);

    float32x4_t v_sum_0 = vdupq_n_f32(0);
    float32x4_t v_sum_1 = vdupq_n_f32(0);

    for (; lhs != last_aligned; lhs += 8, rhs += 8) {
        v_sum_0 = vfmaq_f32(v_sum_0, vld1q_f32(lhs + 0), vld1q_f32(rhs + 0));
        v_sum_1 = vfmaq_f32(v_sum_1, vld1q_f32(lhs + 4), vld1q_f32(rhs + 4));
    }
    if (last >= last_aligned + 4) {
        v_sum_0 = vfmaq_f32(v_sum_0, vld1q_f32(lhs), vld1q_f32(rhs));
        lhs += 4;
        rhs += 4;
    }

    float result = vaddvq_f32(vaddq_f32(v_sum_0, v_sum_1));
    switch (last - lhs) {
        case 3:
            result += lhs[2] * rhs[2];
            /* FALLTHRU */
        case 2:
            result += lhs[1] * rhs[1];
            /* FALLTHRU */
        case 1:
            result += lhs[0] * rhs[0];
    }
    return result;
}
