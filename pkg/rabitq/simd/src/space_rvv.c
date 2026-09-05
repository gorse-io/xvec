// Copyright 2026-present the xvec project
// Licensed under the Apache License, Version 2.0.

#include <riscv_vector.h>
#include <stdint.h>

void scalar_quantize_uint8_rvv(uint8_t *result, const float *data, int64_t dimension, float lo, float delta) {
    const float reciprocal = 1.0F / delta;
    for (int64_t i = 0; i < dimension;) {
        size_t vl = __riscv_vsetvl_e32m1((size_t)(dimension - i));
        vfloat32m1_t values = __riscv_vle32_v_f32m1(data + i, vl);
        values = __riscv_vfmul_vf_f32m1(__riscv_vfsub_vf_f32m1(values, lo, vl), reciprocal, vl);
        float lanes[vl];
        __riscv_vse32_v_f32m1(lanes, values, vl);
        for (size_t lane = 0; lane < vl; ++lane) result[i + (int64_t)lane] = (uint8_t)(lanes[lane] + 0.5F);
        i += (int64_t)vl;
    }
}

void scalar_quantize_uint16_rvv(uint16_t *result, const float *data, int64_t dimension, float lo, float delta) {
    const float reciprocal = 1.0F / delta;
    for (int64_t i = 0; i < dimension;) {
        size_t vl = __riscv_vsetvl_e32m1((size_t)(dimension - i));
        vfloat32m1_t values = __riscv_vle32_v_f32m1(data + i, vl);
        values = __riscv_vfmul_vf_f32m1(__riscv_vfsub_vf_f32m1(values, lo, vl), reciprocal, vl);
        float lanes[vl];
        __riscv_vse32_v_f32m1(lanes, values, vl);
        for (size_t lane = 0; lane < vl; ++lane) result[i + (int64_t)lane] = (uint16_t)(lanes[lane] + 0.5F);
        i += (int64_t)vl;
    }
}

void new_transpose_bin_rvv(const uint16_t *data, uint64_t *result, int64_t padded_dimension, int64_t bits) {
    for (int64_t block = 0; block < padded_dimension / 64; ++block) {
        for (int64_t bit = 0; bit < bits; ++bit) {
            uint64_t plane = 0;
            for (int64_t i = 0; i < 64;) {
                size_t vl = __riscv_vsetvl_e16m1((size_t)(64 - i));
                vuint16m1_t values = __riscv_vle16_v_u16m1(data + block * 64 + i, vl);
                uint16_t lanes[vl];
                __riscv_vse16_v_u16m1(lanes, __riscv_vand_vx_u16m1(__riscv_vsrl_vx_u16m1(values, (size_t)bit, vl), 1, vl), vl);
                for (size_t lane = 0; lane < vl; ++lane) plane |= (uint64_t)lanes[lane] << (63 - i - (int64_t)lane);
                i += (int64_t)vl;
            }
            result[block * bits + bit] = plane;
        }
    }
}

void new_transpose_bin_512_rvv(const uint8_t *data, uint64_t *result, int64_t padded_dimension, int64_t bits) {
    for (int64_t block_start = 0; block_start < padded_dimension; block_start += 512) {
        int64_t block_size = padded_dimension - block_start < 512 ? padded_dimension - block_start : 512;
        int64_t chunks = block_size / 64;
        int64_t output_base = block_start / 64 * bits;
        for (int64_t bit = 0; bit < bits; ++bit) {
            for (int64_t chunk = 0; chunk < chunks; ++chunk) {
                uint64_t plane = 0;
                for (int64_t i = 0; i < 64;) {
                    size_t vl = __riscv_vsetvl_e8m1((size_t)(64 - i));
                    vuint8m1_t values = __riscv_vle8_v_u8m1(data + block_start + chunk * 64 + i, vl);
                    uint8_t lanes[vl];
                    __riscv_vse8_v_u8m1(lanes, __riscv_vand_vx_u8m1(__riscv_vsrl_vx_u8m1(values, (size_t)bit, vl), 1, vl), vl);
                    for (size_t lane = 0; lane < vl; ++lane) plane |= (uint64_t)lanes[lane] << (63 - i - (int64_t)lane);
                    i += (int64_t)vl;
                }
                result[output_base + bit * chunks + chunk] = plane;
            }
        }
    }
}

static inline float reduce_f32_rvv(vfloat32m1_t values, size_t vl) {
    vfloat32m1_t zero = __riscv_vfmv_v_f_f32m1(0, vl);
    return __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredosum_vs_f32m1_f32m1(values, zero, vl));
}

float mask_ip_x0_q_rvv(const float *query, const uint64_t *data, int64_t padded_dimension) {
    float result = 0;
    for (int64_t i = 0; i < padded_dimension;) {
        size_t vl = __riscv_vsetvl_e32m1((size_t)(padded_dimension - i));
        float selected[vl];
        for (size_t lane = 0; lane < vl; ++lane) {
            int64_t index = i + (int64_t)lane;
            selected[lane] = (data[index / 64] & (UINT64_C(1) << (63 - index % 64))) ? query[index] : 0;
        }
        result += reduce_f32_rvv(__riscv_vle32_v_f32m1(selected, vl), vl);
        i += (int64_t)vl;
    }
    return result;
}
