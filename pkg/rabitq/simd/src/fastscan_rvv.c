// Copyright 2026-present the xvec project
// Licensed under the Apache License, Version 2.0.

#include <riscv_vector.h>
#include <stdint.h>

void accumulate_rvv(const uint8_t *codes, const uint8_t *lp_table, uint16_t *result, int64_t dim) {
    size_t vl = __riscv_vsetvl_e16m2(8);
    vuint16m2_t sums0 = __riscv_vmv_v_x_u16m2(0, vl);
    vuint16m2_t sums1 = sums0;
    vuint16m2_t sums2 = sums0;
    vuint16m2_t sums3 = sums0;

    for (int64_t codebook = 0; codebook < dim / 4; ++codebook) {
        vl = __riscv_vsetvl_e8m1(16);
        vuint8m1_t packed = __riscv_vle8_v_u8m1(codes + codebook * 16, vl);
        vuint8m1_t table = __riscv_vle8_v_u8m1(lp_table + codebook * 16, vl);
        vuint8m1_t low = __riscv_vrgather_vv_u8m1(table, __riscv_vand_vx_u8m1(packed, 15, vl), vl);
        vuint8m1_t high = __riscv_vrgather_vv_u8m1(table, __riscv_vsrl_vx_u8m1(packed, 4, vl), vl);

        vl = __riscv_vsetvl_e8m1(8);
        vuint8m1_t even = __riscv_vsll_vx_u8m1(__riscv_vid_v_u8m1(vl), 1, vl);
        vuint8m1_t odd = __riscv_vadd_vx_u8m1(even, 1, vl);
        sums0 = __riscv_vwaddu_wv_u16m2(sums0, __riscv_vrgather_vv_u8m1(low, even, vl), vl);
        sums1 = __riscv_vwaddu_wv_u16m2(sums1, __riscv_vrgather_vv_u8m1(low, odd, vl), vl);
        sums2 = __riscv_vwaddu_wv_u16m2(sums2, __riscv_vrgather_vv_u8m1(high, even, vl), vl);
        sums3 = __riscv_vwaddu_wv_u16m2(sums3, __riscv_vrgather_vv_u8m1(high, odd, vl), vl);
    }

    vl = __riscv_vsetvl_e16m2(8);
    __riscv_vse16_v_u16m2(result, sums0, vl);
    __riscv_vse16_v_u16m2(result + 8, sums1, vl);
    __riscv_vse16_v_u16m2(result + 16, sums2, vl);
    __riscv_vse16_v_u16m2(result + 24, sums3, vl);
}

void transfer_lut_hacc_rvv(const uint16_t *lut, int64_t dim, uint8_t *hc_lut) {
    for (int64_t codebook = 0; codebook < dim / 4; ++codebook) {
        const int64_t group_base = codebook / 4 * 128;
        const int64_t low_base = group_base + codebook % 4 * 16;
        const int64_t high_base = low_base + 64;
        size_t vl = __riscv_vsetvl_e16m2(16);
        vuint16m2_t values = __riscv_vle16_v_u16m2(lut + codebook * 16, vl);
        __riscv_vse8_v_u8m1(hc_lut + low_base, __riscv_vnsrl_wx_u8m1(values, 0, vl), vl);
        __riscv_vse8_v_u8m1(hc_lut + high_base, __riscv_vnsrl_wx_u8m1(values, 8, vl), vl);
    }
}

static inline __attribute__((always_inline)) void accumulate_hacc_half_rvv(
    vuint8m1_t low_table, vuint8m1_t high_table, vuint8m1_t indices,
    vuint32m2_t *even_sum, vuint32m2_t *odd_sum
) {
    size_t vl = __riscv_vsetvl_e8m1(16);
    uint8_t low_values[16], high_values[16];
    __riscv_vse8_v_u8m1(low_values, __riscv_vrgather_vv_u8m1(low_table, indices, vl), vl);
    __riscv_vse8_v_u8m1(high_values, __riscv_vrgather_vv_u8m1(high_table, indices, vl), vl);

    vl = __riscv_vsetvl_e8mf2(8);
    vuint8mf2x2_t low = __riscv_vlseg2e8_v_u8mf2x2(low_values, vl);
    vuint8mf2x2_t high = __riscv_vlseg2e8_v_u8mf2x2(high_values, vl);
    vuint16m1_t even = __riscv_vzext_vf2_u16m1(__riscv_vget_v_u8mf2x2_u8mf2(low, 0), vl);
    vuint16m1_t odd = __riscv_vzext_vf2_u16m1(__riscv_vget_v_u8mf2x2_u8mf2(low, 1), vl);
    even = __riscv_vor_vv_u16m1(even, __riscv_vsll_vx_u16m1(
        __riscv_vzext_vf2_u16m1(__riscv_vget_v_u8mf2x2_u8mf2(high, 0), vl), 8, vl), vl);
    odd = __riscv_vor_vv_u16m1(odd, __riscv_vsll_vx_u16m1(
        __riscv_vzext_vf2_u16m1(__riscv_vget_v_u8mf2x2_u8mf2(high, 1), vl), 8, vl), vl);
    *even_sum = __riscv_vwaddu_wv_u32m2(*even_sum, even, vl);
    *odd_sum = __riscv_vwaddu_wv_u32m2(*odd_sum, odd, vl);
}

void accumulate_hacc_rvv(const uint8_t *codes, const uint8_t *hc_lut, int32_t *result, int64_t dim) {
    size_t vl = __riscv_vsetvl_e32m2(8);
    vuint32m2_t sums0 = __riscv_vmv_v_x_u32m2(0, vl);
    vuint32m2_t sums1 = sums0;
    vuint32m2_t sums2 = sums0;
    vuint32m2_t sums3 = sums0;

    for (int64_t codebook = 0; codebook < dim / 4; ++codebook) {
        const int64_t group_base = codebook / 4 * 128;
        const int64_t low_base = group_base + codebook % 4 * 16;
        const int64_t high_base = low_base + 64;
        vl = __riscv_vsetvl_e8m1(16);
        vuint8m1_t packed = __riscv_vle8_v_u8m1(codes + codebook * 16, vl);
        vuint8m1_t low_table = __riscv_vle8_v_u8m1(hc_lut + low_base, vl);
        vuint8m1_t high_table = __riscv_vle8_v_u8m1(hc_lut + high_base, vl);
        accumulate_hacc_half_rvv(low_table, high_table,
            __riscv_vand_vx_u8m1(packed, 15, vl), &sums0, &sums1);
        accumulate_hacc_half_rvv(low_table, high_table,
            __riscv_vsrl_vx_u8m1(packed, 4, vl), &sums2, &sums3);
    }

    vl = __riscv_vsetvl_e32m2(8);
    __riscv_vse32_v_u32m2((uint32_t *)result, sums0, vl);
    __riscv_vse32_v_u32m2((uint32_t *)result + 8, sums1, vl);
    __riscv_vse32_v_u32m2((uint32_t *)result + 16, sums2, vl);
    __riscv_vse32_v_u32m2((uint32_t *)result + 24, sums3, vl);
}
