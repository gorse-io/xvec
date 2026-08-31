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

static inline float reduce256(__m256 value) {
    float partial[8];
    _mm256_storeu_ps(partial, value);
    float sum = 0;
    for (int index = 0; index < 8; index++) {
        sum += partial[index];
    }
    return sum;
}

void xvec_avx_l2_squared(float *left, float *right, int64_t size, float *output) {
    int64_t vectors = size / 8;
    int64_t remain = size % 8;
    __m256 sum = _mm256_setzero_ps();
    for (int64_t index = 0; index < vectors; index++) {
        __m256 left_value = _mm256_loadu_ps(left);
        __m256 right_value = _mm256_loadu_ps(right);
        __m256 difference = _mm256_sub_ps(left_value, right_value);
        sum = _mm256_add_ps(sum, _mm256_mul_ps(difference, difference));
        left += 8;
        right += 8;
    }
    float result = reduce256(sum);
    for (int64_t index = 0; index < remain; index++) {
        float difference = left[index] - right[index];
        result += difference * difference;
    }
    *output = result;
}

void xvec_avx_inner_product(float *left, float *right, int64_t size, float *output) {
    int64_t vectors = size / 8;
    int64_t remain = size % 8;
    __m256 sum = _mm256_setzero_ps();
    for (int64_t index = 0; index < vectors; index++) {
        __m256 left_value = _mm256_loadu_ps(left);
        __m256 right_value = _mm256_loadu_ps(right);
        sum = _mm256_add_ps(sum, _mm256_mul_ps(left_value, right_value));
        left += 8;
        right += 8;
    }
    float result = reduce256(sum);
    for (int64_t index = 0; index < remain; index++) {
        result += left[index] * right[index];
    }
    *output = result;
}

void xvec_avx_dot_norms(float *left, float *right, int64_t size,
                        float *dot, float *left_norm, float *right_norm) {
    int64_t vectors = size / 8;
    int64_t remain = size % 8;
    __m256 dot_sum = _mm256_setzero_ps();
    __m256 left_sum = _mm256_setzero_ps();
    __m256 right_sum = _mm256_setzero_ps();
    for (int64_t index = 0; index < vectors; index++) {
        __m256 left_value = _mm256_loadu_ps(left);
        __m256 right_value = _mm256_loadu_ps(right);
        dot_sum = _mm256_add_ps(dot_sum, _mm256_mul_ps(left_value, right_value));
        left_sum = _mm256_add_ps(left_sum, _mm256_mul_ps(left_value, left_value));
        right_sum = _mm256_add_ps(right_sum, _mm256_mul_ps(right_value, right_value));
        left += 8;
        right += 8;
    }
    *dot = reduce256(dot_sum);
    *left_norm = reduce256(left_sum);
    *right_norm = reduce256(right_sum);
    for (int64_t index = 0; index < remain; index++) {
        *dot += left[index] * right[index];
        *left_norm += left[index] * left[index];
        *right_norm += right[index] * right[index];
    }
}
