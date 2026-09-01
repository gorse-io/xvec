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

#ifndef XVEC_DISTANCE_MATRIX_FP32_H_
#define XVEC_DISTANCE_MATRIX_FP32_H_

#include <immintrin.h>
#include <stdint.h>

#if !defined(__FMA__)
#define _mm_fmadd_ps(a, b, c) _mm_add_ps(_mm_mul_ps((a), (b)), (c))
#define _mm256_fmadd_ps(a, b, c) _mm256_add_ps(_mm256_mul_ps((a), (b)), (c))
#endif

static inline float horizontal_add_fp32_v128(__m128 value) {
#if defined(__SSE3__)
    __m128 x1 = _mm_hadd_ps(value, value);
    __m128 x2 = _mm_hadd_ps(x1, x1);
    return _mm_cvtss_f32(x2);
#else
    __m128 x1 = _mm_movehl_ps(value, value);
    __m128 x2 = _mm_add_ps(value, x1);
    __m128 x3 = _mm_shuffle_ps(x2, x2, 1);
    __m128 x4 = _mm_add_ss(x2, x3);
    return _mm_cvtss_f32(x4);
#endif
}

static inline float horizontal_add_fp32_v256(__m256 value) {
    __m256 x1 = _mm256_hadd_ps(value, value);
    __m256 x2 = _mm256_hadd_ps(x1, x1);
    __m128 x3 = _mm256_extractf128_ps(x2, 1);
    __m128 x4 = _mm_add_ss(_mm256_castps256_ps128(x2), x3);
    return _mm_cvtss_f32(x4);
}

#if defined(__AVX512F__)
static inline float horizontal_add_fp32_v512(__m512 value) {
    __m256 low = _mm512_castps512_ps256(value);
    __m256 high = _mm256_castpd_ps(
        _mm512_extractf64x4_pd(_mm512_castps_pd(value), 1));
    return horizontal_add_fp32_v256(_mm256_add_ps(low, high));
}
#endif

#define FMA_FP32_AVX512(m, q, sum)     sum = _mm512_fmadd_ps((m), (q), (sum));
#define FMA_MASK_FP32_AVX512(m, q, sum, mask)     sum = _mm512_mask3_fmadd_ps((m), (q), (sum), (mask));

#endif
