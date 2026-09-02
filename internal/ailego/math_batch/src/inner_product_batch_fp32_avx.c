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

// One-to-many inner product kernel: load every query block once and reuse it
// for two candidates, following zvec's construction scorer layout.
void xvec_avx_batch_inner_products2(float *query, float *first, float *second,
                                    int64_t size, float *first_output,
                                    float *second_output) {
    int64_t vectors = size / 8;
    int64_t remain = size % 8;
    __m256 first_sum = _mm256_setzero_ps();
    __m256 second_sum = _mm256_setzero_ps();
    for (int64_t index = 0; index < vectors; index++) {
        __m256 query_value = _mm256_loadu_ps(query);
        first_sum = _mm256_add_ps(first_sum, _mm256_mul_ps(query_value, _mm256_loadu_ps(first)));
        second_sum = _mm256_add_ps(second_sum, _mm256_mul_ps(query_value, _mm256_loadu_ps(second)));
        query += 8;
        first += 8;
        second += 8;
    }
    *first_output = reduce256(first_sum);
    *second_output = reduce256(second_sum);
    for (int64_t index = 0; index < remain; index++) {
        *first_output += query[index] * first[index];
        *second_output += query[index] * second[index];
    }
}

// One-to-many inner product kernel for centroid assignment. Four accumulators
// fit in AVX registers while amortizing every query load over four candidates.
void xvec_avx_batch_inner_products4(float *query, float *first, float *second,
                                    float *third, float *fourth, int64_t size,
                                    float *first_output, float *second_output,
                                    float *third_output, float *fourth_output) {
    int64_t vectors = size / 8;
    int64_t remain = size % 8;
    __m256 first_sum = _mm256_setzero_ps();
    __m256 second_sum = _mm256_setzero_ps();
    __m256 third_sum = _mm256_setzero_ps();
    __m256 fourth_sum = _mm256_setzero_ps();
    for (int64_t index = 0; index < vectors; index++) {
        __m256 query_value = _mm256_loadu_ps(query);
        first_sum = _mm256_add_ps(first_sum, _mm256_mul_ps(query_value, _mm256_loadu_ps(first)));
        second_sum = _mm256_add_ps(second_sum, _mm256_mul_ps(query_value, _mm256_loadu_ps(second)));
        third_sum = _mm256_add_ps(third_sum, _mm256_mul_ps(query_value, _mm256_loadu_ps(third)));
        fourth_sum = _mm256_add_ps(fourth_sum, _mm256_mul_ps(query_value, _mm256_loadu_ps(fourth)));
        query += 8;
        first += 8;
        second += 8;
        third += 8;
        fourth += 8;
    }
    *first_output = reduce256(first_sum);
    *second_output = reduce256(second_sum);
    *third_output = reduce256(third_sum);
    *fourth_output = reduce256(fourth_sum);
    for (int64_t index = 0; index < remain; index++) {
        *first_output += query[index] * first[index];
        *second_output += query[index] * second[index];
        *third_output += query[index] * third[index];
        *fourth_output += query[index] * fourth[index];
    }
}

void xvec_avx_batch_squared_euclidean_distances2(
        float *query, float *first, float *second, int64_t size,
        float *first_output, float *second_output) {
    int64_t vectors = size / 8;
    int64_t remain = size % 8;
    __m256 first_sum = _mm256_setzero_ps();
    __m256 second_sum = _mm256_setzero_ps();
    for (int64_t index = 0; index < vectors; index++) {
        __m256 query_value = _mm256_loadu_ps(query);
        __m256 first_difference = _mm256_sub_ps(query_value, _mm256_loadu_ps(first));
        __m256 second_difference = _mm256_sub_ps(query_value, _mm256_loadu_ps(second));
        first_sum = _mm256_add_ps(first_sum, _mm256_mul_ps(first_difference, first_difference));
        second_sum = _mm256_add_ps(second_sum, _mm256_mul_ps(second_difference, second_difference));
        query += 8;
        first += 8;
        second += 8;
    }
    *first_output = reduce256(first_sum);
    *second_output = reduce256(second_sum);
    for (int64_t index = 0; index < remain; index++) {
        float first_difference = query[index] - first[index];
        float second_difference = query[index] - second[index];
        *first_output += first_difference * first_difference;
        *second_output += second_difference * second_difference;
    }
}

void xvec_avx_batch_squared_euclidean_distances4(
        float *query, float *first, float *second, float *third, float *fourth,
        int64_t size, float *first_output, float *second_output,
        float *third_output, float *fourth_output) {
    int64_t vectors = size / 8;
    int64_t remain = size % 8;
    __m256 first_sum = _mm256_setzero_ps();
    __m256 second_sum = _mm256_setzero_ps();
    __m256 third_sum = _mm256_setzero_ps();
    __m256 fourth_sum = _mm256_setzero_ps();
    for (int64_t index = 0; index < vectors; index++) {
        __m256 query_value = _mm256_loadu_ps(query);
        __m256 first_difference = _mm256_sub_ps(query_value, _mm256_loadu_ps(first));
        __m256 second_difference = _mm256_sub_ps(query_value, _mm256_loadu_ps(second));
        __m256 third_difference = _mm256_sub_ps(query_value, _mm256_loadu_ps(third));
        __m256 fourth_difference = _mm256_sub_ps(query_value, _mm256_loadu_ps(fourth));
        first_sum = _mm256_add_ps(first_sum, _mm256_mul_ps(first_difference, first_difference));
        second_sum = _mm256_add_ps(second_sum, _mm256_mul_ps(second_difference, second_difference));
        third_sum = _mm256_add_ps(third_sum, _mm256_mul_ps(third_difference, third_difference));
        fourth_sum = _mm256_add_ps(fourth_sum, _mm256_mul_ps(fourth_difference, fourth_difference));
        query += 8;
        first += 8;
        second += 8;
        third += 8;
        fourth += 8;
    }
    *first_output = reduce256(first_sum);
    *second_output = reduce256(second_sum);
    *third_output = reduce256(third_sum);
    *fourth_output = reduce256(fourth_sum);
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
