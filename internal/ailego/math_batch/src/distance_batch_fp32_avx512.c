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

void xvec_avx512_batch_inner_products2(
        const float *query, const float *first, const float *second,
        int64_t size, float *first_output, float *second_output) {
    int64_t index = 0;
    __m512 first_sum = _mm512_setzero_ps();
    __m512 second_sum = _mm512_setzero_ps();
    for (; index + 16 <= size; index += 16) {
        __m512 query_value = _mm512_loadu_ps(query + index);
        first_sum = _mm512_add_ps(first_sum, _mm512_mul_ps(query_value, _mm512_loadu_ps(first + index)));
        second_sum = _mm512_add_ps(second_sum, _mm512_mul_ps(query_value, _mm512_loadu_ps(second + index)));
    }
    *first_output = reduce512(first_sum);
    *second_output = reduce512(second_sum);
    for (; index < size; index++) {
        *first_output += query[index] * first[index];
        *second_output += query[index] * second[index];
    }
}

void xvec_avx512_batch_inner_products4(
        const float *query, const float *first, const float *second,
        const float *third, const float *fourth, int64_t size,
        float *first_output, float *second_output, float *third_output,
        float *fourth_output) {
    int64_t index = 0;
    __m512 first_sum = _mm512_setzero_ps();
    __m512 second_sum = _mm512_setzero_ps();
    __m512 third_sum = _mm512_setzero_ps();
    __m512 fourth_sum = _mm512_setzero_ps();
    for (; index + 16 <= size; index += 16) {
        __m512 query_value = _mm512_loadu_ps(query + index);
        first_sum = _mm512_add_ps(first_sum, _mm512_mul_ps(query_value, _mm512_loadu_ps(first + index)));
        second_sum = _mm512_add_ps(second_sum, _mm512_mul_ps(query_value, _mm512_loadu_ps(second + index)));
        third_sum = _mm512_add_ps(third_sum, _mm512_mul_ps(query_value, _mm512_loadu_ps(third + index)));
        fourth_sum = _mm512_add_ps(fourth_sum, _mm512_mul_ps(query_value, _mm512_loadu_ps(fourth + index)));
    }
    *first_output = reduce512(first_sum);
    *second_output = reduce512(second_sum);
    *third_output = reduce512(third_sum);
    *fourth_output = reduce512(fourth_sum);
    for (; index < size; index++) {
        *first_output += query[index] * first[index];
        *second_output += query[index] * second[index];
        *third_output += query[index] * third[index];
        *fourth_output += query[index] * fourth[index];
    }
}

void xvec_avx512_batch_squared_euclidean_distances2(
        const float *query, const float *first, const float *second,
        int64_t size, float *first_output, float *second_output) {
    int64_t index = 0;
    __m512 first_sum = _mm512_setzero_ps();
    __m512 second_sum = _mm512_setzero_ps();
    for (; index + 16 <= size; index += 16) {
        __m512 query_value = _mm512_loadu_ps(query + index);
        __m512 first_difference = _mm512_sub_ps(query_value, _mm512_loadu_ps(first + index));
        __m512 second_difference = _mm512_sub_ps(query_value, _mm512_loadu_ps(second + index));
        first_sum = _mm512_add_ps(first_sum, _mm512_mul_ps(first_difference, first_difference));
        second_sum = _mm512_add_ps(second_sum, _mm512_mul_ps(second_difference, second_difference));
    }
    *first_output = reduce512(first_sum);
    *second_output = reduce512(second_sum);
    for (; index < size; index++) {
        float first_difference = query[index] - first[index];
        float second_difference = query[index] - second[index];
        *first_output += first_difference * first_difference;
        *second_output += second_difference * second_difference;
    }
}

void xvec_avx512_batch_squared_euclidean_distances4(
        const float *query, const float *first, const float *second,
        const float *third, const float *fourth, int64_t size,
        float *first_output, float *second_output, float *third_output,
        float *fourth_output) {
    int64_t index = 0;
    __m512 first_sum = _mm512_setzero_ps();
    __m512 second_sum = _mm512_setzero_ps();
    __m512 third_sum = _mm512_setzero_ps();
    __m512 fourth_sum = _mm512_setzero_ps();
    for (; index + 16 <= size; index += 16) {
        __m512 query_value = _mm512_loadu_ps(query + index);
        __m512 first_difference = _mm512_sub_ps(query_value, _mm512_loadu_ps(first + index));
        __m512 second_difference = _mm512_sub_ps(query_value, _mm512_loadu_ps(second + index));
        __m512 third_difference = _mm512_sub_ps(query_value, _mm512_loadu_ps(third + index));
        __m512 fourth_difference = _mm512_sub_ps(query_value, _mm512_loadu_ps(fourth + index));
        first_sum = _mm512_add_ps(first_sum, _mm512_mul_ps(first_difference, first_difference));
        second_sum = _mm512_add_ps(second_sum, _mm512_mul_ps(second_difference, second_difference));
        third_sum = _mm512_add_ps(third_sum, _mm512_mul_ps(third_difference, third_difference));
        fourth_sum = _mm512_add_ps(fourth_sum, _mm512_mul_ps(fourth_difference, fourth_difference));
    }
    *first_output = reduce512(first_sum);
    *second_output = reduce512(second_sum);
    *third_output = reduce512(third_sum);
    *fourth_output = reduce512(fourth_sum);
    for (; index < size; index++) {
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
