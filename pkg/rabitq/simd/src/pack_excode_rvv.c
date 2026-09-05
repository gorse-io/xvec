// Copyright 2026-present the xvec project
// Licensed under the Apache License, Version 2.0.

#include <riscv_vector.h>
#include <stdint.h>

static inline void pack_top_bit_rvv(const uint8_t *raw, uint8_t *compact, int bit) {
    for (int byte = 0; byte < 8; ++byte) {
        uint8_t packed = 0;
        for (int lane = 0; lane < 8; ++lane) packed |= ((raw[byte * 8 + lane] >> bit) & 1U) << lane;
        compact[byte] = packed;
    }
}

void packing_2bit_excode_rvv(const uint8_t *raw, uint8_t *compact, int64_t dimension) {
    for (int64_t offset = 0; offset < dimension; offset += 64) {
        size_t vl = __riscv_vsetvl_e8m1(16);
        vuint8m1_t a = __riscv_vle8_v_u8m1(raw, vl);
        vuint8m1_t b = __riscv_vsll_vx_u8m1(__riscv_vle8_v_u8m1(raw + 16, vl), 2, vl);
        vuint8m1_t c = __riscv_vsll_vx_u8m1(__riscv_vle8_v_u8m1(raw + 32, vl), 4, vl);
        vuint8m1_t d = __riscv_vsll_vx_u8m1(__riscv_vle8_v_u8m1(raw + 48, vl), 6, vl);
        __riscv_vse8_v_u8m1(compact, __riscv_vor_vv_u8m1(__riscv_vor_vv_u8m1(a, b, vl), __riscv_vor_vv_u8m1(c, d, vl), vl), vl);
        raw += 64; compact += 16;
    }
}

void packing_3bit_excode_rvv(const uint8_t *raw, uint8_t *compact, int64_t dimension) {
    for (int64_t offset = 0; offset < dimension; offset += 64) {
        size_t vl = __riscv_vsetvl_e8m1(16);
        vuint8m1_t a = __riscv_vand_vx_u8m1(__riscv_vle8_v_u8m1(raw, vl), 3, vl);
        vuint8m1_t b = __riscv_vsll_vx_u8m1(__riscv_vand_vx_u8m1(__riscv_vle8_v_u8m1(raw + 16, vl), 3, vl), 2, vl);
        vuint8m1_t c = __riscv_vsll_vx_u8m1(__riscv_vand_vx_u8m1(__riscv_vle8_v_u8m1(raw + 32, vl), 3, vl), 4, vl);
        vuint8m1_t d = __riscv_vsll_vx_u8m1(__riscv_vand_vx_u8m1(__riscv_vle8_v_u8m1(raw + 48, vl), 3, vl), 6, vl);
        __riscv_vse8_v_u8m1(compact, __riscv_vor_vv_u8m1(__riscv_vor_vv_u8m1(a, b, vl), __riscv_vor_vv_u8m1(c, d, vl), vl), vl);
        pack_top_bit_rvv(raw, compact + 16, 2);
        raw += 64; compact += 24;
    }
}

void packing_4bit_excode_rvv(const uint8_t *raw, uint8_t *compact, int64_t dimension) {
    for (int64_t offset = 0; offset < dimension; offset += 16) {
        size_t vl = __riscv_vsetvl_e8m1(8);
        vuint8m1_t a = __riscv_vle8_v_u8m1(raw, vl);
        vuint8m1_t b = __riscv_vsll_vx_u8m1(__riscv_vle8_v_u8m1(raw + 8, vl), 4, vl);
        __riscv_vse8_v_u8m1(compact, __riscv_vor_vv_u8m1(a, b, vl), vl);
        raw += 16; compact += 8;
    }
}

void packing_5bit_excode_rvv(const uint8_t *raw, uint8_t *compact, int64_t dimension) {
    for (int64_t offset = 0; offset < dimension; offset += 64) {
        size_t vl = __riscv_vsetvl_e8m1(16);
        vuint8m1_t a = __riscv_vand_vx_u8m1(__riscv_vle8_v_u8m1(raw, vl), 15, vl);
        vuint8m1_t b = __riscv_vsll_vx_u8m1(__riscv_vand_vx_u8m1(__riscv_vle8_v_u8m1(raw + 16, vl), 15, vl), 4, vl);
        vuint8m1_t c = __riscv_vand_vx_u8m1(__riscv_vle8_v_u8m1(raw + 32, vl), 15, vl);
        vuint8m1_t d = __riscv_vsll_vx_u8m1(__riscv_vand_vx_u8m1(__riscv_vle8_v_u8m1(raw + 48, vl), 15, vl), 4, vl);
        __riscv_vse8_v_u8m1(compact, __riscv_vor_vv_u8m1(a, b, vl), vl);
        __riscv_vse8_v_u8m1(compact + 16, __riscv_vor_vv_u8m1(c, d, vl), vl);
        pack_top_bit_rvv(raw, compact + 32, 4);
        raw += 64; compact += 40;
    }
}

static inline void packing_67bit_excode_rvv(const uint8_t *raw, uint8_t *compact, int64_t dimension, int top) {
    const int64_t stride = top ? 56 : 48;
    for (int64_t offset = 0; offset < dimension; offset += 64) {
        size_t vl = __riscv_vsetvl_e8m1(16);
        vuint8m1_t last = __riscv_vle8_v_u8m1(raw + 48, vl);
        for (int group = 0; group < 3; ++group) {
            vuint8m1_t low = __riscv_vand_vx_u8m1(__riscv_vle8_v_u8m1(raw + group * 16, vl), 63, vl);
            vuint8m1_t high = __riscv_vand_vx_u8m1(__riscv_vsll_vx_u8m1(last, 6 - group * 2, vl), 0xc0, vl);
            __riscv_vse8_v_u8m1(compact + group * 16, __riscv_vor_vv_u8m1(low, high, vl), vl);
        }
        if (top) pack_top_bit_rvv(raw, compact + 48, 6);
        raw += 64; compact += stride;
    }
}

void packing_6bit_excode_rvv(const uint8_t *raw, uint8_t *compact, int64_t dimension) { packing_67bit_excode_rvv(raw, compact, dimension, 0); }
void packing_7bit_excode_rvv(const uint8_t *raw, uint8_t *compact, int64_t dimension) { packing_67bit_excode_rvv(raw, compact, dimension, 1); }
