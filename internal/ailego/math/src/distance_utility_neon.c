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

static inline float reduce_neon(float32x4_t value) {
    float partial[4];
    vst1q_f32(partial, value);
    return partial[0] + partial[1] + partial[2] + partial[3];
}

void xvec_neon_l2_squared(float *left, float *right, int64_t size, float *output) {
    int64_t vectors = size / 4;
    int64_t remain = size % 4;
    float32x4_t sum = vdupq_n_f32(0);
    for (int64_t index = 0; index < vectors; index++) {
        float32x4_t left_value = vld1q_f32(left);
        float32x4_t right_value = vld1q_f32(right);
        float32x4_t difference = vsubq_f32(left_value, right_value);
        sum = vmlaq_f32(sum, difference, difference);
        left += 4;
        right += 4;
    }
    float result = reduce_neon(sum);
    for (int64_t index = 0; index < remain; index++) {
        float difference = left[index] - right[index];
        result += difference * difference;
    }
    *output = result;
}

void xvec_neon_inner_product(float *left, float *right, int64_t size, float *output) {
    int64_t vectors = size / 4;
    int64_t remain = size % 4;
    float32x4_t sum = vdupq_n_f32(0);
    for (int64_t index = 0; index < vectors; index++) {
        float32x4_t left_value = vld1q_f32(left);
        float32x4_t right_value = vld1q_f32(right);
        sum = vmlaq_f32(sum, left_value, right_value);
        left += 4;
        right += 4;
    }
    float result = reduce_neon(sum);
    for (int64_t index = 0; index < remain; index++) {
        result += left[index] * right[index];
    }
    *output = result;
}

void xvec_neon_dot_norms(float *left, float *right, int64_t size,
                         float *dot, float *left_norm, float *right_norm) {
    int64_t vectors = size / 4;
    int64_t remain = size % 4;
    float32x4_t dot_sum = vdupq_n_f32(0);
    float32x4_t left_sum = vdupq_n_f32(0);
    float32x4_t right_sum = vdupq_n_f32(0);
    for (int64_t index = 0; index < vectors; index++) {
        float32x4_t left_value = vld1q_f32(left);
        float32x4_t right_value = vld1q_f32(right);
        dot_sum = vmlaq_f32(dot_sum, left_value, right_value);
        left_sum = vmlaq_f32(left_sum, left_value, left_value);
        right_sum = vmlaq_f32(right_sum, right_value, right_value);
        left += 4;
        right += 4;
    }
    *dot = reduce_neon(dot_sum);
    *left_norm = reduce_neon(left_sum);
    *right_norm = reduce_neon(right_sum);
    for (int64_t index = 0; index < remain; index++) {
        *dot += left[index] * right[index];
        *left_norm += left[index] * left[index];
        *right_norm += right[index] * right[index];
    }
}
