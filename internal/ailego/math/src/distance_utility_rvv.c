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

void xvec_rvv_l2_squared(float *left, float *right, int64_t size, float *output) {
    size_t vlmax = __riscv_vsetvlmax_e32m1();
    int64_t vectors = size / vlmax;
    int64_t remain = size % vlmax;
    vfloat32m1_t sum = __riscv_vfmv_v_f_f32m1(0, vlmax);
    for (int64_t index = 0; index < vectors; index++) {
        vfloat32m1_t left_value = __riscv_vle32_v_f32m1(left, vlmax);
        vfloat32m1_t right_value = __riscv_vle32_v_f32m1(right, vlmax);
        vfloat32m1_t difference = __riscv_vfsub_vv_f32m1(left_value, right_value, vlmax);
        sum = __riscv_vfmacc_vv_f32m1(sum, difference, difference, vlmax);
        left += vlmax;
        right += vlmax;
    }
    float result = reduce_rvv(sum, vlmax);
    if (remain > 0) {
        size_t vl = __riscv_vsetvl_e32m1(remain);
        vfloat32m1_t left_value = __riscv_vle32_v_f32m1(left, vl);
        vfloat32m1_t right_value = __riscv_vle32_v_f32m1(right, vl);
        vfloat32m1_t difference = __riscv_vfsub_vv_f32m1(left_value, right_value, vl);
        result += reduce_rvv(__riscv_vfmul_vv_f32m1(difference, difference, vl), vl);
    }
    *output = result;
}

void xvec_rvv_inner_product(float *left, float *right, int64_t size, float *output) {
    size_t vlmax = __riscv_vsetvlmax_e32m1();
    int64_t vectors = size / vlmax;
    int64_t remain = size % vlmax;
    vfloat32m1_t sum = __riscv_vfmv_v_f_f32m1(0, vlmax);
    for (int64_t index = 0; index < vectors; index++) {
        vfloat32m1_t left_value = __riscv_vle32_v_f32m1(left, vlmax);
        vfloat32m1_t right_value = __riscv_vle32_v_f32m1(right, vlmax);
        sum = __riscv_vfmacc_vv_f32m1(sum, left_value, right_value, vlmax);
        left += vlmax;
        right += vlmax;
    }
    float result = reduce_rvv(sum, vlmax);
    if (remain > 0) {
        size_t vl = __riscv_vsetvl_e32m1(remain);
        vfloat32m1_t left_value = __riscv_vle32_v_f32m1(left, vl);
        vfloat32m1_t right_value = __riscv_vle32_v_f32m1(right, vl);
        result += reduce_rvv(__riscv_vfmul_vv_f32m1(left_value, right_value, vl), vl);
    }
    *output = result;
}

void xvec_rvv_dot_norms(float *left, float *right, int64_t size,
                        float *dot, float *left_norm, float *right_norm) {
    size_t vlmax = __riscv_vsetvlmax_e32m1();
    int64_t vectors = size / vlmax;
    int64_t remain = size % vlmax;
    vfloat32m1_t dot_sum = __riscv_vfmv_v_f_f32m1(0, vlmax);
    vfloat32m1_t left_sum = __riscv_vfmv_v_f_f32m1(0, vlmax);
    vfloat32m1_t right_sum = __riscv_vfmv_v_f_f32m1(0, vlmax);
    for (int64_t index = 0; index < vectors; index++) {
        vfloat32m1_t left_value = __riscv_vle32_v_f32m1(left, vlmax);
        vfloat32m1_t right_value = __riscv_vle32_v_f32m1(right, vlmax);
        dot_sum = __riscv_vfmacc_vv_f32m1(dot_sum, left_value, right_value, vlmax);
        left_sum = __riscv_vfmacc_vv_f32m1(left_sum, left_value, left_value, vlmax);
        right_sum = __riscv_vfmacc_vv_f32m1(right_sum, right_value, right_value, vlmax);
        left += vlmax;
        right += vlmax;
    }
    *dot = reduce_rvv(dot_sum, vlmax);
    *left_norm = reduce_rvv(left_sum, vlmax);
    *right_norm = reduce_rvv(right_sum, vlmax);
    if (remain > 0) {
        size_t vl = __riscv_vsetvl_e32m1(remain);
        vfloat32m1_t left_value = __riscv_vle32_v_f32m1(left, vl);
        vfloat32m1_t right_value = __riscv_vle32_v_f32m1(right, vl);
        *dot += reduce_rvv(__riscv_vfmul_vv_f32m1(left_value, right_value, vl), vl);
        *left_norm += reduce_rvv(__riscv_vfmul_vv_f32m1(left_value, left_value, vl), vl);
        *right_norm += reduce_rvv(__riscv_vfmul_vv_f32m1(right_value, right_value, vl), vl);
    }
}
