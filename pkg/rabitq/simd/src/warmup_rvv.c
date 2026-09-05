// Copyright 2026-present the xvec project
// Licensed under the Apache License, Version 2.0.

#include <riscv_vector.h>
#include <stdint.h>

float warmup_ip_x0_q_512_rvv(const uint64_t *data, const uint64_t *query, float delta, float vl_factor, int64_t padded_dimension, int64_t query_bits) {
    const int64_t words = padded_dimension / 64;
    uint64_t ip = 0, popcount = 0;
    for (int64_t block = 0, query_base = 0; block < words;) {
        int64_t block_words = words - block < 8 ? words - block : 8;
        int64_t consumed = 0;
        while (consumed < block_words) {
            size_t vl = __riscv_vsetvl_e64m1((size_t)(block_words - consumed));
            vuint64m1_t x = __riscv_vle64_v_u64m1(data + block + consumed, vl);
            uint64_t lanes[vl];
            __riscv_vse64_v_u64m1(lanes, x, vl);
            for (size_t lane = 0; lane < vl; ++lane) popcount += (uint64_t)__builtin_popcountll(lanes[lane]);
            for (int64_t bit = 0; bit < query_bits; ++bit) {
                vuint64m1_t y = __riscv_vle64_v_u64m1(query + query_base + bit * block_words + consumed, vl);
                __riscv_vse64_v_u64m1(lanes, __riscv_vand_vv_u64m1(x, y, vl), vl);
                for (size_t lane = 0; lane < vl; ++lane) ip += (uint64_t)__builtin_popcountll(lanes[lane]) << bit;
            }
            consumed += (int64_t)vl;
        }
        block += block_words;
        query_base += block_words * query_bits;
    }
    return vl_factor * (float)popcount + delta * (float)ip;
}
