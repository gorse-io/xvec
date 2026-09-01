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

float inner_product_fp32_lasx(const float *lhs, const float *rhs, long size) {
    long vectors = size / 8;
    long remain = size % 8;
    __m256 sum = (__m256)__lasx_xvldi(0);
    for (long index = 0; index < vectors; index++) {
        __m256 lhs_value = (__m256)__lasx_xvld((void *)lhs, 0);
        __m256 rhs_value = (__m256)__lasx_xvld((void *)rhs, 0);
        sum = __lasx_xvfadd_s(sum, __lasx_xvfmul_s(lhs_value, rhs_value));
        lhs += 8;
        rhs += 8;
    }
    float result = reduce_lasx(sum);
    for (long index = 0; index < remain; index++) {
        result += lhs[index] * rhs[index];
    }
    return result;
}
