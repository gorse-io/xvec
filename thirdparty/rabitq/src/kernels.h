// Copyright 2026-present the xvec project
// Licensed under the Apache License, Version 2.0.
//
// This file is derived from VectorDB-NTU/RaBitQ-Library src/simd.
// It was changed to provide C intrinsic templates suitable for GoAT.

#include <immintrin.h>
#include <stdint.h>

#ifndef RABITQ_SUFFIX
#error RABITQ_SUFFIX must be defined
#endif

#define RABITQ_CAT_INNER(a, b) a##b
#define RABITQ_CAT(a, b) RABITQ_CAT_INNER(a, b)
#define RABITQ_FN(name) RABITQ_CAT(name, RABITQ_SUFFIX)

#include "rotator_kernels.h"
#include "space_kernels.h"
#include "pack_kernels.h"
#include "excode_kernels.h"
#include "fastscan_kernels.h"
#include "warmup_kernels.h"

void RABITQ_FN(rabitq_flip_sign)(const uint8_t *signs, float *data, int64_t dim) {
    rabitq_flip_sign(signs, data, dim);
}

void RABITQ_FN(rabitq_kacs_walk)(float *data, int64_t len) {
    rabitq_kacs_walk(data, len);
}

void RABITQ_FN(rabitq_quantize_u8)(uint8_t *out, const float *values, int64_t dim,
                                         float lo, float delta) {
    rabitq_quantize_u8(out, values, dim, lo, delta);
}

void RABITQ_FN(rabitq_quantize_u16)(uint16_t *out, const float *values, int64_t dim,
                                          float lo, float delta) {
    rabitq_quantize_u16(out, values, dim, lo, delta);
}

void RABITQ_FN(rabitq_transpose_bin)(const uint16_t *query, uint64_t *out, int64_t dim,
                                           int64_t width) {
    rabitq_transpose_bin(query, out, dim, width);
}

void RABITQ_FN(rabitq_transpose_bin_512)(const uint8_t *query, uint64_t *out,
                                               int64_t dim, int64_t width) {
    rabitq_transpose_bin_512(query, out, dim, width);
}

float RABITQ_FN(rabitq_mask_ip)(const float *query, const uint8_t *code, int64_t dim) {
    return rabitq_mask_ip(query, code, dim);
}
