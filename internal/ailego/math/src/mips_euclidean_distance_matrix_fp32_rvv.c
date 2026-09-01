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

void inner_product_and_squared_norm_fp32_rvv(const float *lhs, const float *rhs,
                                              int64_t size, float *dot,
                                              float *lhs_norm, float *rhs_norm) {
    size_t vlmax = __riscv_vsetvlmax_e32m1();
    int64_t vectors = size / vlmax;
    int64_t remain = size % vlmax;
    vfloat32m1_t dot_sum = __riscv_vfmv_v_f_f32m1(0, vlmax);
    vfloat32m1_t lhs_sum = __riscv_vfmv_v_f_f32m1(0, vlmax);
    vfloat32m1_t rhs_sum = __riscv_vfmv_v_f_f32m1(0, vlmax);
    for (int64_t index = 0; index < vectors; index++) {
        vfloat32m1_t lhs_value = __riscv_vle32_v_f32m1(lhs, vlmax);
        vfloat32m1_t rhs_value = __riscv_vle32_v_f32m1(rhs, vlmax);
        dot_sum = __riscv_vfmacc_vv_f32m1(dot_sum, lhs_value, rhs_value, vlmax);
        lhs_sum = __riscv_vfmacc_vv_f32m1(lhs_sum, lhs_value, lhs_value, vlmax);
        rhs_sum = __riscv_vfmacc_vv_f32m1(rhs_sum, rhs_value, rhs_value, vlmax);
        lhs += vlmax;
        rhs += vlmax;
    }
    *dot = reduce_rvv(dot_sum, vlmax);
    *lhs_norm = reduce_rvv(lhs_sum, vlmax);
    *rhs_norm = reduce_rvv(rhs_sum, vlmax);
    if (remain > 0) {
        size_t vl = __riscv_vsetvl_e32m1(remain);
        vfloat32m1_t lhs_value = __riscv_vle32_v_f32m1(lhs, vl);
        vfloat32m1_t rhs_value = __riscv_vle32_v_f32m1(rhs, vl);
        *dot += reduce_rvv(__riscv_vfmul_vv_f32m1(lhs_value, rhs_value, vl), vl);
        *lhs_norm += reduce_rvv(__riscv_vfmul_vv_f32m1(lhs_value, lhs_value, vl), vl);
        *rhs_norm += reduce_rvv(__riscv_vfmul_vv_f32m1(rhs_value, rhs_value, vl), vl);
    }
}
