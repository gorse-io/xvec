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

static inline float reduce_fp16_lasx(__m256 value) {
    volatile float partial[8];
    __lasx_xvst((__m256i)value, (void *)partial, 0);
    float sum = 0;
    for (int index = 0; index < 8; index++) {
        sum += partial[index];
    }
    return sum;
}

static inline void load_fp16_lasx(const unsigned short *input, __m256 *low,
                                  __m256 *high) {
    __m256i packed = __lasx_xvld((const void *)input, 0);
    *low = __lasx_xvfcvtl_s_h(packed);
    *high = __lasx_xvfcvth_s_h(packed);
}

float squared_euclidean_distance_fp16_lasx(const unsigned short *left,
                                            const unsigned short *right,
                                            long size) {
    __m256 sum = (__m256)__lasx_xvldi(0);
    for (long index = 0; index < size; index += 16) {
        __m256 left_low, left_high, right_low, right_high;
        load_fp16_lasx(left + index, &left_low, &left_high);
        load_fp16_lasx(right + index, &right_low, &right_high);
        __m256 delta = __lasx_xvfsub_s(left_low, right_low);
        sum = __lasx_xvfadd_s(sum, __lasx_xvfmul_s(delta, delta));
        delta = __lasx_xvfsub_s(left_high, right_high);
        sum = __lasx_xvfadd_s(sum, __lasx_xvfmul_s(delta, delta));
    }
    return reduce_fp16_lasx(sum);
}

float inner_product_fp16_lasx(const unsigned short *left, const unsigned short *right,
                               long size) {
    __m256 sum = (__m256)__lasx_xvldi(0);
    for (long index = 0; index < size; index += 16) {
        __m256 left_low, left_high, right_low, right_high;
        load_fp16_lasx(left + index, &left_low, &left_high);
        load_fp16_lasx(right + index, &right_low, &right_high);
        sum = __lasx_xvfadd_s(sum, __lasx_xvfmul_s(left_low, right_low));
        sum = __lasx_xvfadd_s(sum, __lasx_xvfmul_s(left_high, right_high));
    }
    return reduce_fp16_lasx(sum);
}

float inner_product_and_squared_norm_fp16_lasx(
        const unsigned short *left, const unsigned short *right, long size,
        float *left_norm, float *right_norm) {
    __m256 dot = (__m256)__lasx_xvldi(0);
    __m256 left_sum = (__m256)__lasx_xvldi(0);
    __m256 right_sum = (__m256)__lasx_xvldi(0);
    for (long index = 0; index < size; index += 16) {
        __m256 left_low, left_high, right_low, right_high;
        load_fp16_lasx(left + index, &left_low, &left_high);
        load_fp16_lasx(right + index, &right_low, &right_high);
        dot = __lasx_xvfadd_s(dot, __lasx_xvfmul_s(left_low, right_low));
        dot = __lasx_xvfadd_s(dot, __lasx_xvfmul_s(left_high, right_high));
        left_sum = __lasx_xvfadd_s(left_sum, __lasx_xvfmul_s(left_low, left_low));
        left_sum = __lasx_xvfadd_s(left_sum, __lasx_xvfmul_s(left_high, left_high));
        right_sum = __lasx_xvfadd_s(right_sum, __lasx_xvfmul_s(right_low, right_low));
        right_sum = __lasx_xvfadd_s(right_sum, __lasx_xvfmul_s(right_high, right_high));
    }
    *left_norm = reduce_fp16_lasx(left_sum);
    *right_norm = reduce_fp16_lasx(right_sum);
    return reduce_fp16_lasx(dot);
}
