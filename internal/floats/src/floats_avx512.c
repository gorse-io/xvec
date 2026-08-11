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

static inline float reduce512(__m512 value) {
    float partial[16];
    _mm512_storeu_ps(partial, value);
    float sum = 0;
    for (int index = 0; index < 16; index++) {
        sum += partial[index];
    }
    return sum;
}

void xvec_avx512_l2_squared(float *left, float *right, int64_t size, float *output) {
    int64_t vectors = size / 16;
    int64_t remain = size % 16;
    __m512 sum = _mm512_setzero_ps();
    for (int64_t index = 0; index < vectors; index++) {
        __m512 left_value = _mm512_loadu_ps(left);
        __m512 right_value = _mm512_loadu_ps(right);
        __m512 difference = _mm512_sub_ps(left_value, right_value);
        sum = _mm512_fmadd_ps(difference, difference, sum);
        left += 16;
        right += 16;
    }
    float result = reduce512(sum);
    for (int64_t index = 0; index < remain; index++) {
        float difference = left[index] - right[index];
        result += difference * difference;
    }
    *output = result;
}

void xvec_avx512_inner_product(float *left, float *right, int64_t size, float *output) {
    int64_t vectors = size / 16;
    int64_t remain = size % 16;
    __m512 sum = _mm512_setzero_ps();
    for (int64_t index = 0; index < vectors; index++) {
        __m512 left_value = _mm512_loadu_ps(left);
        __m512 right_value = _mm512_loadu_ps(right);
        sum = _mm512_fmadd_ps(left_value, right_value, sum);
        left += 16;
        right += 16;
    }
    float result = reduce512(sum);
    for (int64_t index = 0; index < remain; index++) {
        result += left[index] * right[index];
    }
    *output = result;
}

void xvec_avx512_dot_norms(float *left, float *right, int64_t size,
                           float *dot, float *left_norm, float *right_norm) {
    int64_t vectors = size / 16;
    int64_t remain = size % 16;
    __m512 dot_sum = _mm512_setzero_ps();
    __m512 left_sum = _mm512_setzero_ps();
    __m512 right_sum = _mm512_setzero_ps();
    for (int64_t index = 0; index < vectors; index++) {
        __m512 left_value = _mm512_loadu_ps(left);
        __m512 right_value = _mm512_loadu_ps(right);
        dot_sum = _mm512_fmadd_ps(left_value, right_value, dot_sum);
        left_sum = _mm512_fmadd_ps(left_value, left_value, left_sum);
        right_sum = _mm512_fmadd_ps(right_value, right_value, right_sum);
        left += 16;
        right += 16;
    }
    *dot = reduce512(dot_sum);
    *left_norm = reduce512(left_sum);
    *right_norm = reduce512(right_sum);
    for (int64_t index = 0; index < remain; index++) {
        *dot += left[index] * right[index];
        *left_norm += left[index] * left[index];
        *right_norm += right[index] * right[index];
    }
}
