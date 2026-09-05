// Copyright 2026-present the xvec project
// Licensed under the Apache License, Version 2.0.

#include <riscv_vector.h>
#include <stdint.h>

static inline uint8_t excode_value_rvv(const uint8_t *code, int64_t index, int bits) {
    int64_t block, offset, base;
    switch (bits) {
    case 1:
        return (code[index / 8] >> (index % 8)) & 1U;
    case 2:
    case 3: {
        block = index / 64; offset = index % 64; base = block * bits * 8;
        uint8_t value = (code[base + offset % 16] >> (offset / 16 * 2)) & 3U;
        if (bits == 3) value |= ((code[base + 16 + offset / 8] >> (offset % 8)) & 1U) << 2;
        return value;
    }
    case 4:
        block = index / 16; offset = index % 16;
        return (code[block * 8 + offset % 8] >> (offset / 8 * 4)) & 15U;
    case 5: {
        block = index / 64; offset = index % 64; base = block * 40;
        uint8_t value = (code[base + offset / 32 * 16 + offset % 16] >> (offset % 32 / 16 * 4)) & 15U;
        return value | (((code[base + 32 + offset / 8] >> (offset % 8)) & 1U) << 4);
    }
    case 6:
    case 7: {
        block = index / 64; offset = index % 64; base = block * bits * 8;
        uint8_t value;
        if (offset < 48) value = code[base + offset / 16 * 16 + offset % 16] & 63U;
        else {
            int64_t column = offset - 48;
            value = (code[base + column] >> 6) | ((code[base + 16 + column] >> 6) << 2) | ((code[base + 32 + column] >> 6) << 4);
        }
        if (bits == 7) value |= ((code[base + 48 + offset / 8] >> (offset % 8)) & 1U) << 6;
        return value;
    }
    default:
        return code[index];
    }
}

static inline __attribute__((always_inline)) float excode_ip_rvv(const float *query, const uint8_t *code, int64_t dim, int bits) {
    float result = 0;
    for (int64_t i = 0; i < dim;) {
        size_t vl = __riscv_vsetvl_e32m1((size_t)(dim - i));
        float decoded[vl];
        for (size_t lane = 0; lane < vl; ++lane) decoded[lane] = (float)excode_value_rvv(code, i + (int64_t)lane, bits);
        vfloat32m1_t product = __riscv_vfmul_vv_f32m1(__riscv_vle32_v_f32m1(query + i, vl), __riscv_vle32_v_f32m1(decoded, vl), vl);
        vfloat32m1_t zero = __riscv_vfmv_v_f_f32m1(0, vl);
        result += __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredosum_vs_f32m1_f32m1(product, zero, vl));
        i += (int64_t)vl;
    }
    return result;
}

float ip16_fxu1_rvv(const float *query, const uint8_t *code, int64_t dim) { return excode_ip_rvv(query, code, dim, 1); }
float ip64_fxu2_rvv(const float *query, const uint8_t *code, int64_t dim) { return excode_ip_rvv(query, code, dim, 2); }
float ip64_fxu3_rvv(const float *query, const uint8_t *code, int64_t dim) { return excode_ip_rvv(query, code, dim, 3); }
float ip16_fxu4_rvv(const float *query, const uint8_t *code, int64_t dim) { return excode_ip_rvv(query, code, dim, 4); }
float ip64_fxu5_rvv(const float *query, const uint8_t *code, int64_t dim) { return excode_ip_rvv(query, code, dim, 5); }
float ip64_fxu6_rvv(const float *query, const uint8_t *code, int64_t dim) { return excode_ip_rvv(query, code, dim, 6); }
float ip64_fxu7_rvv(const float *query, const uint8_t *code, int64_t dim) { return excode_ip_rvv(query, code, dim, 7); }
float ip16_fxu8_rvv(const float *query, const uint8_t *code, int64_t dim) { return excode_ip_rvv(query, code, dim, 8); }
