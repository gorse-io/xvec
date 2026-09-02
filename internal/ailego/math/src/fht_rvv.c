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

void fht_flip_sign_rvv(uint8_t *signs, float *data, int64_t size) {
    int64_t index = 0;
    while (index < size) {
        size_t vl = __riscv_vsetvl_e32m1(size - index < 8 ? size - index : 8);
        uint32_t masks[8];
        for (size_t lane = 0; lane < vl; lane++) {
            int64_t position = index + lane;
            masks[lane] = (signs[position / 8] & (1u << (position % 8))) ? 0x80000000u : 0;
        }
        vuint32m1_t values = __riscv_vle32_v_u32m1((uint32_t *)(data + index), vl);
        vuint32m1_t sign_masks = __riscv_vle32_v_u32m1(masks, vl);
        __riscv_vse32_v_u32m1((uint32_t *)(data + index), __riscv_vxor_vv_u32m1(values, sign_masks, vl), vl);
        index += vl;
    }
}

void fht_kacs_walk_rvv(float *data, int64_t size) {
    int64_t half = size / 2;
    int64_t base = size % 2;
    int64_t offset = base + half;
    int64_t index = 0;
    while (index < half) {
        size_t vl = __riscv_vsetvl_e32m1(half - index);
        vfloat32m1_t left = __riscv_vle32_v_f32m1(data + index, vl);
        vfloat32m1_t right = __riscv_vle32_v_f32m1(data + index + offset, vl);
        __riscv_vse32_v_f32m1(data + index, __riscv_vfadd_vv_f32m1(left, right, vl), vl);
        __riscv_vse32_v_f32m1(data + index + offset, __riscv_vfsub_vv_f32m1(left, right, vl), vl);
        index += vl;
    }
}

void fht_inv_kacs_walk_rvv(float *data, int64_t size) {
    int64_t half = size / 2;
    int64_t base = size % 2;
    int64_t offset = base + half;
    int64_t index = 0;
    while (index < half) {
        size_t vl = __riscv_vsetvl_e32m1(half - index);
        vfloat32m1_t left = __riscv_vle32_v_f32m1(data + index, vl);
        vfloat32m1_t right = __riscv_vle32_v_f32m1(data + index + offset, vl);
        vfloat32m1_t sum = __riscv_vfadd_vv_f32m1(left, right, vl);
        vfloat32m1_t difference = __riscv_vfsub_vv_f32m1(left, right, vl);
        __riscv_vse32_v_f32m1(data + index, __riscv_vfmul_vf_f32m1(sum, 0.5f, vl), vl);
        __riscv_vse32_v_f32m1(data + index + offset, __riscv_vfmul_vf_f32m1(difference, 0.5f, vl), vl);
        index += vl;
    }
}

void fht_inplace_rvv(float *data, int64_t size) {
    for (int64_t width = 1; width < size; width <<= 1) {
        int64_t step = width << 1;
        for (int64_t block = 0; block < size; block += step) {
            int64_t index = 0;
            while (index < width) {
                size_t vl = __riscv_vsetvl_e32m1(width - index);
                vfloat32m1_t left = __riscv_vle32_v_f32m1(data + block + index, vl);
                vfloat32m1_t right = __riscv_vle32_v_f32m1(data + block + index + width, vl);
                __riscv_vse32_v_f32m1(data + block + index, __riscv_vfadd_vv_f32m1(left, right, vl), vl);
                __riscv_vse32_v_f32m1(data + block + index + width, __riscv_vfsub_vv_f32m1(left, right, vl), vl);
                index += vl;
            }
        }
    }
}
