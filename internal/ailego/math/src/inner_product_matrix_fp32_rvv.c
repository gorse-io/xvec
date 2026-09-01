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

float inner_product_fp32_rvv(const float *lhs, const float *rhs, int64_t size) {
    size_t vlmax = __riscv_vsetvlmax_e32m1();
    int64_t vectors = size / vlmax;
    int64_t remain = size % vlmax;
    vfloat32m1_t sum = __riscv_vfmv_v_f_f32m1(0, vlmax);
    for (int64_t index = 0; index < vectors; index++) {
        vfloat32m1_t lhs_value = __riscv_vle32_v_f32m1(lhs, vlmax);
        vfloat32m1_t rhs_value = __riscv_vle32_v_f32m1(rhs, vlmax);
        sum = __riscv_vfmacc_vv_f32m1(sum, lhs_value, rhs_value, vlmax);
        lhs += vlmax;
        rhs += vlmax;
    }
    float result = reduce_rvv(sum, vlmax);
    if (remain > 0) {
        size_t vl = __riscv_vsetvl_e32m1(remain);
        vfloat32m1_t lhs_value = __riscv_vle32_v_f32m1(lhs, vl);
        vfloat32m1_t rhs_value = __riscv_vle32_v_f32m1(rhs, vl);
        result += reduce_rvv(__riscv_vfmul_vv_f32m1(lhs_value, rhs_value, vl), vl);
    }
    return result;
}
