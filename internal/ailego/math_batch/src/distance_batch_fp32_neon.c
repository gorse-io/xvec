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
    return vaddvq_f32(value);
}

void xvec_neon_batch_inner_products2(const float *query, const float *first,
                                     const float *second, int64_t size,
                                     float *first_output, float *second_output) {
    int64_t vectors = size / 4;
    int64_t remain = size % 4;
    float32x4_t first_sum = vdupq_n_f32(0);
    float32x4_t second_sum = vdupq_n_f32(0);
    for (int64_t index = 0; index < vectors; index++) {
        float32x4_t query_value = vld1q_f32(query);
        first_sum = vfmaq_f32(first_sum, query_value, vld1q_f32(first));
        second_sum = vfmaq_f32(second_sum, query_value, vld1q_f32(second));
        query += 4; first += 4; second += 4;
    }
    *first_output = reduce_neon(first_sum);
    *second_output = reduce_neon(second_sum);
    for (int64_t index = 0; index < remain; index++) {
        *first_output += query[index] * first[index];
        *second_output += query[index] * second[index];
    }
}

void xvec_neon_batch_inner_products4(
        const float *query, const float *first, const float *second,
        const float *third, const float *fourth, int64_t size,
        float *first_output, float *second_output, float *third_output,
        float *fourth_output) {
    int64_t vectors = size / 4;
    int64_t remain = size % 4;
    float32x4_t first_sum = vdupq_n_f32(0);
    float32x4_t second_sum = vdupq_n_f32(0);
    float32x4_t third_sum = vdupq_n_f32(0);
    float32x4_t fourth_sum = vdupq_n_f32(0);
    for (int64_t index = 0; index < vectors; index++) {
        float32x4_t query_value = vld1q_f32(query);
        first_sum = vfmaq_f32(first_sum, query_value, vld1q_f32(first));
        second_sum = vfmaq_f32(second_sum, query_value, vld1q_f32(second));
        third_sum = vfmaq_f32(third_sum, query_value, vld1q_f32(third));
        fourth_sum = vfmaq_f32(fourth_sum, query_value, vld1q_f32(fourth));
        query += 4; first += 4; second += 4; third += 4; fourth += 4;
    }
    *first_output = reduce_neon(first_sum);
    *second_output = reduce_neon(second_sum);
    *third_output = reduce_neon(third_sum);
    *fourth_output = reduce_neon(fourth_sum);
    for (int64_t index = 0; index < remain; index++) {
        *first_output += query[index] * first[index];
        *second_output += query[index] * second[index];
        *third_output += query[index] * third[index];
        *fourth_output += query[index] * fourth[index];
    }
}

void xvec_neon_batch_squared_euclidean_distances2(
        const float *query, const float *first, const float *second,
        int64_t size, float *first_output, float *second_output) {
    int64_t vectors = size / 4;
    int64_t remain = size % 4;
    float32x4_t first_sum = vdupq_n_f32(0);
    float32x4_t second_sum = vdupq_n_f32(0);
    for (int64_t index = 0; index < vectors; index++) {
        float32x4_t query_value = vld1q_f32(query);
        float32x4_t first_difference = vsubq_f32(query_value, vld1q_f32(first));
        float32x4_t second_difference = vsubq_f32(query_value, vld1q_f32(second));
        first_sum = vfmaq_f32(first_sum, first_difference, first_difference);
        second_sum = vfmaq_f32(second_sum, second_difference, second_difference);
        query += 4; first += 4; second += 4;
    }
    *first_output = reduce_neon(first_sum);
    *second_output = reduce_neon(second_sum);
    for (int64_t index = 0; index < remain; index++) {
        float first_difference = query[index] - first[index];
        float second_difference = query[index] - second[index];
        *first_output += first_difference * first_difference;
        *second_output += second_difference * second_difference;
    }
}

void xvec_neon_batch_squared_euclidean_distances4(
        const float *query, const float *first, const float *second,
        const float *third, const float *fourth, int64_t size,
        float *first_output, float *second_output, float *third_output,
        float *fourth_output) {
    int64_t vectors = size / 4;
    int64_t remain = size % 4;
    float32x4_t first_sum = vdupq_n_f32(0);
    float32x4_t second_sum = vdupq_n_f32(0);
    float32x4_t third_sum = vdupq_n_f32(0);
    float32x4_t fourth_sum = vdupq_n_f32(0);
    for (int64_t index = 0; index < vectors; index++) {
        float32x4_t query_value = vld1q_f32(query);
        float32x4_t first_difference = vsubq_f32(query_value, vld1q_f32(first));
        float32x4_t second_difference = vsubq_f32(query_value, vld1q_f32(second));
        float32x4_t third_difference = vsubq_f32(query_value, vld1q_f32(third));
        float32x4_t fourth_difference = vsubq_f32(query_value, vld1q_f32(fourth));
        first_sum = vfmaq_f32(first_sum, first_difference, first_difference);
        second_sum = vfmaq_f32(second_sum, second_difference, second_difference);
        third_sum = vfmaq_f32(third_sum, third_difference, third_difference);
        fourth_sum = vfmaq_f32(fourth_sum, fourth_difference, fourth_difference);
        query += 4; first += 4; second += 4; third += 4; fourth += 4;
    }
    *first_output = reduce_neon(first_sum);
    *second_output = reduce_neon(second_sum);
    *third_output = reduce_neon(third_sum);
    *fourth_output = reduce_neon(fourth_sum);
    for (int64_t index = 0; index < remain; index++) {
        float first_difference = query[index] - first[index];
        float second_difference = query[index] - second[index];
        float third_difference = query[index] - third[index];
        float fourth_difference = query[index] - fourth[index];
        *first_output += first_difference * first_difference;
        *second_output += second_difference * second_difference;
        *third_output += third_difference * third_difference;
        *fourth_output += fourth_difference * fourth_difference;
    }
}
