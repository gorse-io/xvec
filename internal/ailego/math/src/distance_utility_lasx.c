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

#include <lasxintrin.h>

static inline float reduce_lasx(__m256 value) {
    float partial[8];
    __lasx_xvst((__m256i)value, partial, 0);
    float sum = 0;
    for (int index = 0; index < 8; index++) {
        sum += partial[index];
    }
    return sum;
}

void xvec_lasx_l2_squared(float *left, float *right, long size, float *output) {
    long vectors = size / 8;
    long remain = size % 8;
    __m256 sum = (__m256)__lasx_xvldi(0);
    for (long index = 0; index < vectors; index++) {
        __m256 left_value = (__m256)__lasx_xvld(left, 0);
        __m256 right_value = (__m256)__lasx_xvld(right, 0);
        __m256 difference = __lasx_xvfsub_s(left_value, right_value);
        sum = __lasx_xvfadd_s(sum, __lasx_xvfmul_s(difference, difference));
        left += 8;
        right += 8;
    }
    float result = reduce_lasx(sum);
    for (long index = 0; index < remain; index++) {
        float difference = left[index] - right[index];
        result += difference * difference;
    }
    *output = result;
}

void xvec_lasx_inner_product(float *left, float *right, long size, float *output) {
    long vectors = size / 8;
    long remain = size % 8;
    __m256 sum = (__m256)__lasx_xvldi(0);
    for (long index = 0; index < vectors; index++) {
        __m256 left_value = (__m256)__lasx_xvld(left, 0);
        __m256 right_value = (__m256)__lasx_xvld(right, 0);
        sum = __lasx_xvfadd_s(sum, __lasx_xvfmul_s(left_value, right_value));
        left += 8;
        right += 8;
    }
    float result = reduce_lasx(sum);
    for (long index = 0; index < remain; index++) {
        result += left[index] * right[index];
    }
    *output = result;
}

void xvec_lasx_dot_norms(float *left, float *right, long size,
                         float *dot, float *left_norm, float *right_norm) {
    long vectors = size / 8;
    long remain = size % 8;
    __m256 dot_sum = (__m256)__lasx_xvldi(0);
    __m256 left_sum = (__m256)__lasx_xvldi(0);
    __m256 right_sum = (__m256)__lasx_xvldi(0);
    for (long index = 0; index < vectors; index++) {
        __m256 left_value = (__m256)__lasx_xvld(left, 0);
        __m256 right_value = (__m256)__lasx_xvld(right, 0);
        dot_sum = __lasx_xvfadd_s(dot_sum, __lasx_xvfmul_s(left_value, right_value));
        left_sum = __lasx_xvfadd_s(left_sum, __lasx_xvfmul_s(left_value, left_value));
        right_sum = __lasx_xvfadd_s(right_sum, __lasx_xvfmul_s(right_value, right_value));
        left += 8;
        right += 8;
    }
    *dot = reduce_lasx(dot_sum);
    *left_norm = reduce_lasx(left_sum);
    *right_norm = reduce_lasx(right_sum);
    for (long index = 0; index < remain; index++) {
        *dot += left[index] * right[index];
        *left_norm += left[index] * left[index];
        *right_norm += right[index] * right[index];
    }
}
