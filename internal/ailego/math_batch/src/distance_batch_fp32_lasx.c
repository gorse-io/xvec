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
    for (int index = 0; index < 8; index++) sum += partial[index];
    return sum;
}

void xvec_lasx_batch_inner_products2(const float *query, const float *first,
                                     const float *second, long size,
                                     float *first_output, float *second_output) {
    long vectors = size / 8, remain = size % 8;
    __m256 first_sum = (__m256)__lasx_xvldi(0);
    __m256 second_sum = (__m256)__lasx_xvldi(0);
    for (long index = 0; index < vectors; index++) {
        __m256 query_value = (__m256)__lasx_xvld((void *)query, 0);
        first_sum = __lasx_xvfadd_s(first_sum, __lasx_xvfmul_s(query_value, (__m256)__lasx_xvld((void *)first, 0)));
        second_sum = __lasx_xvfadd_s(second_sum, __lasx_xvfmul_s(query_value, (__m256)__lasx_xvld((void *)second, 0)));
        query += 8; first += 8; second += 8;
    }
    *first_output = reduce_lasx(first_sum);
    *second_output = reduce_lasx(second_sum);
    for (long index = 0; index < remain; index++) {
        *first_output += query[index] * first[index];
        *second_output += query[index] * second[index];
    }
}

void xvec_lasx_batch_inner_products4(
        const float *query, const float *first, const float *second,
        const float *third, const float *fourth, long size,
        float *first_output, float *second_output, float *third_output,
        float *fourth_output) {
    long vectors = size / 8, remain = size % 8;
    __m256 first_sum = (__m256)__lasx_xvldi(0);
    __m256 second_sum = (__m256)__lasx_xvldi(0);
    __m256 third_sum = (__m256)__lasx_xvldi(0);
    __m256 fourth_sum = (__m256)__lasx_xvldi(0);
    for (long index = 0; index < vectors; index++) {
        __m256 query_value = (__m256)__lasx_xvld((void *)query, 0);
        first_sum = __lasx_xvfadd_s(first_sum, __lasx_xvfmul_s(query_value, (__m256)__lasx_xvld((void *)first, 0)));
        second_sum = __lasx_xvfadd_s(second_sum, __lasx_xvfmul_s(query_value, (__m256)__lasx_xvld((void *)second, 0)));
        third_sum = __lasx_xvfadd_s(third_sum, __lasx_xvfmul_s(query_value, (__m256)__lasx_xvld((void *)third, 0)));
        fourth_sum = __lasx_xvfadd_s(fourth_sum, __lasx_xvfmul_s(query_value, (__m256)__lasx_xvld((void *)fourth, 0)));
        query += 8; first += 8; second += 8; third += 8; fourth += 8;
    }
    *first_output = reduce_lasx(first_sum);
    *second_output = reduce_lasx(second_sum);
    *third_output = reduce_lasx(third_sum);
    *fourth_output = reduce_lasx(fourth_sum);
    for (long index = 0; index < remain; index++) {
        *first_output += query[index] * first[index];
        *second_output += query[index] * second[index];
        *third_output += query[index] * third[index];
        *fourth_output += query[index] * fourth[index];
    }
}

void xvec_lasx_batch_squared_euclidean_distances2(
        const float *query, const float *first, const float *second,
        long size, float *first_output, float *second_output) {
    long vectors = size / 8, remain = size % 8;
    __m256 first_sum = (__m256)__lasx_xvldi(0);
    __m256 second_sum = (__m256)__lasx_xvldi(0);
    for (long index = 0; index < vectors; index++) {
        __m256 query_value = (__m256)__lasx_xvld((void *)query, 0);
        __m256 first_difference = __lasx_xvfsub_s(query_value, (__m256)__lasx_xvld((void *)first, 0));
        __m256 second_difference = __lasx_xvfsub_s(query_value, (__m256)__lasx_xvld((void *)second, 0));
        first_sum = __lasx_xvfadd_s(first_sum, __lasx_xvfmul_s(first_difference, first_difference));
        second_sum = __lasx_xvfadd_s(second_sum, __lasx_xvfmul_s(second_difference, second_difference));
        query += 8; first += 8; second += 8;
    }
    *first_output = reduce_lasx(first_sum);
    *second_output = reduce_lasx(second_sum);
    for (long index = 0; index < remain; index++) {
        float first_difference = query[index] - first[index];
        float second_difference = query[index] - second[index];
        *first_output += first_difference * first_difference;
        *second_output += second_difference * second_difference;
    }
}

void xvec_lasx_batch_squared_euclidean_distances4(
        const float *query, const float *first, const float *second,
        const float *third, const float *fourth, long size,
        float *first_output, float *second_output, float *third_output,
        float *fourth_output) {
    long vectors = size / 8, remain = size % 8;
    __m256 first_sum = (__m256)__lasx_xvldi(0);
    __m256 second_sum = (__m256)__lasx_xvldi(0);
    __m256 third_sum = (__m256)__lasx_xvldi(0);
    __m256 fourth_sum = (__m256)__lasx_xvldi(0);
    for (long index = 0; index < vectors; index++) {
        __m256 query_value = (__m256)__lasx_xvld((void *)query, 0);
        __m256 first_difference = __lasx_xvfsub_s(query_value, (__m256)__lasx_xvld((void *)first, 0));
        __m256 second_difference = __lasx_xvfsub_s(query_value, (__m256)__lasx_xvld((void *)second, 0));
        __m256 third_difference = __lasx_xvfsub_s(query_value, (__m256)__lasx_xvld((void *)third, 0));
        __m256 fourth_difference = __lasx_xvfsub_s(query_value, (__m256)__lasx_xvld((void *)fourth, 0));
        first_sum = __lasx_xvfadd_s(first_sum, __lasx_xvfmul_s(first_difference, first_difference));
        second_sum = __lasx_xvfadd_s(second_sum, __lasx_xvfmul_s(second_difference, second_difference));
        third_sum = __lasx_xvfadd_s(third_sum, __lasx_xvfmul_s(third_difference, third_difference));
        fourth_sum = __lasx_xvfadd_s(fourth_sum, __lasx_xvfmul_s(fourth_difference, fourth_difference));
        query += 8; first += 8; second += 8; third += 8; fourth += 8;
    }
    *first_output = reduce_lasx(first_sum);
    *second_output = reduce_lasx(second_sum);
    *third_output = reduce_lasx(third_sum);
    *fourth_output = reduce_lasx(fourth_sum);
    for (long index = 0; index < remain; index++) {
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
