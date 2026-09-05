// Copyright 2026-present the xvec project
// Licensed under the Apache License, Version 2.0.

#include <riscv_vector.h>
#include <stdint.h>

void flip_sign_rvv(const uint8_t *flip, float *data, int64_t dim) {
    for (int64_t i = 0; i < dim;) {
        size_t vl = __riscv_vsetvl_e32m1((size_t)(dim - i));
        uint32_t masks[vl];
        for (size_t lane = 0; lane < vl; ++lane) {
            int64_t index = i + (int64_t)lane;
            masks[lane] = ((flip[index / 8] >> (index % 8)) & 1U) << 31;
        }
        vuint32m1_t values = __riscv_vle32_v_u32m1((const uint32_t *)(data + i), vl);
        values = __riscv_vxor_vv_u32m1(values, __riscv_vle32_v_u32m1(masks, vl), vl);
        __riscv_vse32_v_u32m1((uint32_t *)(data + i), values, vl);
        i += (int64_t)vl;
    }
}

void kacs_walk_rvv(float *data, int64_t len) {
    const int64_t half = len / 2;
    for (int64_t i = 0; i < half;) {
        size_t vl = __riscv_vsetvl_e32m1((size_t)(half - i));
        vfloat32m1_t x = __riscv_vle32_v_f32m1(data + i, vl);
        vfloat32m1_t y = __riscv_vle32_v_f32m1(data + half + i, vl);
        __riscv_vse32_v_f32m1(data + i, __riscv_vfadd_vv_f32m1(x, y, vl), vl);
        __riscv_vse32_v_f32m1(data + half + i, __riscv_vfsub_vv_f32m1(x, y, vl), vl);
        i += (int64_t)vl;
    }
}
