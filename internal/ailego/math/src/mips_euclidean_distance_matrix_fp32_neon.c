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

void inner_product_and_squared_norm_fp32_neon(const float *lhs, const float *rhs,
                                               int64_t size, float *dot,
                                               float *lhs_norm, float *rhs_norm) {
    const float *last = lhs + size;
    const float *last_aligned = lhs + ((size >> 3) << 3);

    float32x4_t v_sum_0 = vdupq_n_f32(0);
    float32x4_t v_sum_1 = vdupq_n_f32(0);
    float32x4_t v_sum_norm1 = vdupq_n_f32(0);
    float32x4_t v_sum_norm2 = vdupq_n_f32(0);

    for (; lhs != last_aligned; lhs += 8, rhs += 8) {
        float32x4_t v_lhs_0 = vld1q_f32(lhs + 0);
        float32x4_t v_lhs_1 = vld1q_f32(lhs + 4);
        float32x4_t v_rhs_0 = vld1q_f32(rhs + 0);
        float32x4_t v_rhs_1 = vld1q_f32(rhs + 4);
        v_sum_0 = vfmaq_f32(v_sum_0, v_lhs_0, v_rhs_0);
        v_sum_1 = vfmaq_f32(v_sum_1, v_lhs_1, v_rhs_1);
        v_sum_norm1 = vfmaq_f32(v_sum_norm1, v_lhs_0, v_lhs_0);
        v_sum_norm1 = vfmaq_f32(v_sum_norm1, v_lhs_1, v_lhs_1);
        v_sum_norm2 = vfmaq_f32(v_sum_norm2, v_rhs_0, v_rhs_0);
        v_sum_norm2 = vfmaq_f32(v_sum_norm2, v_rhs_1, v_rhs_1);
    }
    if (last >= last_aligned + 4) {
        float32x4_t v_lhs_0 = vld1q_f32(lhs);
        float32x4_t v_rhs_0 = vld1q_f32(rhs);
        v_sum_0 = vfmaq_f32(v_sum_0, v_lhs_0, v_rhs_0);
        v_sum_norm1 = vfmaq_f32(v_sum_norm1, v_lhs_0, v_lhs_0);
        v_sum_norm2 = vfmaq_f32(v_sum_norm2, v_rhs_0, v_rhs_0);
        lhs += 4;
        rhs += 4;
    }

    float result = vaddvq_f32(vaddq_f32(v_sum_0, v_sum_1));
    float norm1 = vaddvq_f32(v_sum_norm1);
    float norm2 = vaddvq_f32(v_sum_norm2);
    switch (last - lhs) {
        case 3:
            result += lhs[2] * rhs[2];
            norm1 += lhs[2] * lhs[2];
            norm2 += rhs[2] * rhs[2];
            /* FALLTHRU */
        case 2:
            result += lhs[1] * rhs[1];
            norm1 += lhs[1] * lhs[1];
            norm2 += rhs[1] * rhs[1];
            /* FALLTHRU */
        case 1:
            result += lhs[0] * rhs[0];
            norm1 += lhs[0] * lhs[0];
            norm2 += rhs[0] * rhs[0];
    }
    *dot = result;
    *lhs_norm = norm1;
    *rhs_norm = norm2;
}
