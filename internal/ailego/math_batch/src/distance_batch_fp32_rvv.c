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

#include <riscv_vector.h>
#include <stdint.h>

static inline float reduce_rvv(vfloat32m1_t value, size_t vl) {
    vfloat32m1_t zero = __riscv_vfmv_v_f_f32m1(0, vl);
    vfloat32m1_t sum = __riscv_vfredosum_vs_f32m1_f32m1(value, zero, vl);
    return __riscv_vfmv_f_s_f32m1_f32(sum);
}

void xvec_rvv_batch_inner_products2(const float *query, const float *first,
                                    const float *second, int64_t size,
                                    float *first_output, float *second_output) {
    size_t vlmax = __riscv_vsetvlmax_e32m1();
    int64_t vectors = size / vlmax;
    int64_t remain = size % vlmax;
    vfloat32m1_t first_sum = __riscv_vfmv_v_f_f32m1(0, vlmax);
    vfloat32m1_t second_sum = __riscv_vfmv_v_f_f32m1(0, vlmax);
    for (int64_t index = 0; index < vectors; index++) {
        vfloat32m1_t query_value = __riscv_vle32_v_f32m1(query, vlmax);
        first_sum = __riscv_vfmacc_vv_f32m1(first_sum, query_value, __riscv_vle32_v_f32m1(first, vlmax), vlmax);
        second_sum = __riscv_vfmacc_vv_f32m1(second_sum, query_value, __riscv_vle32_v_f32m1(second, vlmax), vlmax);
        query += vlmax; first += vlmax; second += vlmax;
    }
    *first_output = reduce_rvv(first_sum, vlmax);
    *second_output = reduce_rvv(second_sum, vlmax);
    if (remain > 0) {
        size_t vl = __riscv_vsetvl_e32m1(remain);
        vfloat32m1_t query_value = __riscv_vle32_v_f32m1(query, vl);
        *first_output += reduce_rvv(__riscv_vfmul_vv_f32m1(query_value, __riscv_vle32_v_f32m1(first, vl), vl), vl);
        *second_output += reduce_rvv(__riscv_vfmul_vv_f32m1(query_value, __riscv_vle32_v_f32m1(second, vl), vl), vl);
    }
}

void xvec_rvv_batch_inner_products4(
        const float *query, const float *first, const float *second,
        const float *third, const float *fourth, int64_t size,
        float *first_output, float *second_output, float *third_output,
        float *fourth_output) {
    size_t vlmax = __riscv_vsetvlmax_e32m1();
    int64_t vectors = size / vlmax;
    int64_t remain = size % vlmax;
    vfloat32m1_t first_sum = __riscv_vfmv_v_f_f32m1(0, vlmax);
    vfloat32m1_t second_sum = __riscv_vfmv_v_f_f32m1(0, vlmax);
    vfloat32m1_t third_sum = __riscv_vfmv_v_f_f32m1(0, vlmax);
    vfloat32m1_t fourth_sum = __riscv_vfmv_v_f_f32m1(0, vlmax);
    for (int64_t index = 0; index < vectors; index++) {
        vfloat32m1_t query_value = __riscv_vle32_v_f32m1(query, vlmax);
        first_sum = __riscv_vfmacc_vv_f32m1(first_sum, query_value, __riscv_vle32_v_f32m1(first, vlmax), vlmax);
        second_sum = __riscv_vfmacc_vv_f32m1(second_sum, query_value, __riscv_vle32_v_f32m1(second, vlmax), vlmax);
        third_sum = __riscv_vfmacc_vv_f32m1(third_sum, query_value, __riscv_vle32_v_f32m1(third, vlmax), vlmax);
        fourth_sum = __riscv_vfmacc_vv_f32m1(fourth_sum, query_value, __riscv_vle32_v_f32m1(fourth, vlmax), vlmax);
        query += vlmax; first += vlmax; second += vlmax; third += vlmax; fourth += vlmax;
    }
    *first_output = reduce_rvv(first_sum, vlmax);
    *second_output = reduce_rvv(second_sum, vlmax);
    *third_output = reduce_rvv(third_sum, vlmax);
    *fourth_output = reduce_rvv(fourth_sum, vlmax);
    if (remain > 0) {
        size_t vl = __riscv_vsetvl_e32m1(remain);
        vfloat32m1_t query_value = __riscv_vle32_v_f32m1(query, vl);
        *first_output += reduce_rvv(__riscv_vfmul_vv_f32m1(query_value, __riscv_vle32_v_f32m1(first, vl), vl), vl);
        *second_output += reduce_rvv(__riscv_vfmul_vv_f32m1(query_value, __riscv_vle32_v_f32m1(second, vl), vl), vl);
        *third_output += reduce_rvv(__riscv_vfmul_vv_f32m1(query_value, __riscv_vle32_v_f32m1(third, vl), vl), vl);
        *fourth_output += reduce_rvv(__riscv_vfmul_vv_f32m1(query_value, __riscv_vle32_v_f32m1(fourth, vl), vl), vl);
    }
}

void xvec_rvv_batch_squared_euclidean_distances2(
        const float *query, const float *first, const float *second,
        int64_t size, float *first_output, float *second_output) {
    size_t vlmax = __riscv_vsetvlmax_e32m1();
    int64_t vectors = size / vlmax;
    int64_t remain = size % vlmax;
    vfloat32m1_t first_sum = __riscv_vfmv_v_f_f32m1(0, vlmax);
    vfloat32m1_t second_sum = __riscv_vfmv_v_f_f32m1(0, vlmax);
    for (int64_t index = 0; index < vectors; index++) {
        vfloat32m1_t query_value = __riscv_vle32_v_f32m1(query, vlmax);
        vfloat32m1_t first_difference = __riscv_vfsub_vv_f32m1(query_value, __riscv_vle32_v_f32m1(first, vlmax), vlmax);
        vfloat32m1_t second_difference = __riscv_vfsub_vv_f32m1(query_value, __riscv_vle32_v_f32m1(second, vlmax), vlmax);
        first_sum = __riscv_vfmacc_vv_f32m1(first_sum, first_difference, first_difference, vlmax);
        second_sum = __riscv_vfmacc_vv_f32m1(second_sum, second_difference, second_difference, vlmax);
        query += vlmax; first += vlmax; second += vlmax;
    }
    *first_output = reduce_rvv(first_sum, vlmax);
    *second_output = reduce_rvv(second_sum, vlmax);
    if (remain > 0) {
        size_t vl = __riscv_vsetvl_e32m1(remain);
        vfloat32m1_t query_value = __riscv_vle32_v_f32m1(query, vl);
        vfloat32m1_t first_difference = __riscv_vfsub_vv_f32m1(query_value, __riscv_vle32_v_f32m1(first, vl), vl);
        vfloat32m1_t second_difference = __riscv_vfsub_vv_f32m1(query_value, __riscv_vle32_v_f32m1(second, vl), vl);
        *first_output += reduce_rvv(__riscv_vfmul_vv_f32m1(first_difference, first_difference, vl), vl);
        *second_output += reduce_rvv(__riscv_vfmul_vv_f32m1(second_difference, second_difference, vl), vl);
    }
}

void xvec_rvv_batch_squared_euclidean_distances4(
        const float *query, const float *first, const float *second,
        const float *third, const float *fourth, int64_t size,
        float *first_output, float *second_output, float *third_output,
        float *fourth_output) {
    size_t vlmax = __riscv_vsetvlmax_e32m1();
    int64_t vectors = size / vlmax;
    int64_t remain = size % vlmax;
    vfloat32m1_t first_sum = __riscv_vfmv_v_f_f32m1(0, vlmax);
    vfloat32m1_t second_sum = __riscv_vfmv_v_f_f32m1(0, vlmax);
    vfloat32m1_t third_sum = __riscv_vfmv_v_f_f32m1(0, vlmax);
    vfloat32m1_t fourth_sum = __riscv_vfmv_v_f_f32m1(0, vlmax);
    for (int64_t index = 0; index < vectors; index++) {
        vfloat32m1_t query_value = __riscv_vle32_v_f32m1(query, vlmax);
        vfloat32m1_t first_difference = __riscv_vfsub_vv_f32m1(query_value, __riscv_vle32_v_f32m1(first, vlmax), vlmax);
        vfloat32m1_t second_difference = __riscv_vfsub_vv_f32m1(query_value, __riscv_vle32_v_f32m1(second, vlmax), vlmax);
        vfloat32m1_t third_difference = __riscv_vfsub_vv_f32m1(query_value, __riscv_vle32_v_f32m1(third, vlmax), vlmax);
        vfloat32m1_t fourth_difference = __riscv_vfsub_vv_f32m1(query_value, __riscv_vle32_v_f32m1(fourth, vlmax), vlmax);
        first_sum = __riscv_vfmacc_vv_f32m1(first_sum, first_difference, first_difference, vlmax);
        second_sum = __riscv_vfmacc_vv_f32m1(second_sum, second_difference, second_difference, vlmax);
        third_sum = __riscv_vfmacc_vv_f32m1(third_sum, third_difference, third_difference, vlmax);
        fourth_sum = __riscv_vfmacc_vv_f32m1(fourth_sum, fourth_difference, fourth_difference, vlmax);
        query += vlmax; first += vlmax; second += vlmax; third += vlmax; fourth += vlmax;
    }
    *first_output = reduce_rvv(first_sum, vlmax);
    *second_output = reduce_rvv(second_sum, vlmax);
    *third_output = reduce_rvv(third_sum, vlmax);
    *fourth_output = reduce_rvv(fourth_sum, vlmax);
    if (remain > 0) {
        size_t vl = __riscv_vsetvl_e32m1(remain);
        vfloat32m1_t query_value = __riscv_vle32_v_f32m1(query, vl);
        vfloat32m1_t first_difference = __riscv_vfsub_vv_f32m1(query_value, __riscv_vle32_v_f32m1(first, vl), vl);
        vfloat32m1_t second_difference = __riscv_vfsub_vv_f32m1(query_value, __riscv_vle32_v_f32m1(second, vl), vl);
        vfloat32m1_t third_difference = __riscv_vfsub_vv_f32m1(query_value, __riscv_vle32_v_f32m1(third, vl), vl);
        vfloat32m1_t fourth_difference = __riscv_vfsub_vv_f32m1(query_value, __riscv_vle32_v_f32m1(fourth, vl), vl);
        *first_output += reduce_rvv(__riscv_vfmul_vv_f32m1(first_difference, first_difference, vl), vl);
        *second_output += reduce_rvv(__riscv_vfmul_vv_f32m1(second_difference, second_difference, vl), vl);
        *third_output += reduce_rvv(__riscv_vfmul_vv_f32m1(third_difference, third_difference, vl), vl);
        *fourth_output += reduce_rvv(__riscv_vfmul_vv_f32m1(fourth_difference, fourth_difference, vl), vl);
    }
}
