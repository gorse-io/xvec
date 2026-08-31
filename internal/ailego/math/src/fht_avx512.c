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

#include <immintrin.h>
#include <stdint.h>

void fht_flip_sign_avx512(uint8_t *signs, float *data, int64_t size) {
    int64_t simd_end = size & ~63LL;
    volatile int32_t sign = (int32_t)0x80000000u;
    const __m512 sign_bit = _mm512_castsi512_ps(_mm512_set1_epi32(sign));
    for (int64_t index = 0; index < simd_end; index += 64) {
        uint64_t bits;
        __builtin_memcpy(&bits, signs + index / 8, sizeof(bits));
        for (int64_t block = 0; block < 4; block++) {
            __mmask16 mask = (__mmask16)(bits >> (block * 16));
            __m512 values = _mm512_loadu_ps(data + index + block * 16);
            values = _mm512_mask_xor_ps(values, mask, values, sign_bit);
            _mm512_storeu_ps(data + index + block * 16, values);
        }
    }
    for (int64_t index = simd_end; index < size; index++) {
        if (signs[index / 8] & (1u << (index % 8))) {
            data[index] = -data[index];
        }
    }
}

void fht_kacs_walk_avx512(float *data, int64_t size) {
    int64_t half = size / 2;
    int64_t base = size % 2;
    int64_t offset = base + half;
    int64_t simd_end = half & ~15LL;
    for (int64_t index = 0; index < simd_end; index += 16) {
        __m512 left = _mm512_loadu_ps(data + index);
        __m512 right = _mm512_loadu_ps(data + index + offset);
        _mm512_storeu_ps(data + index, _mm512_add_ps(left, right));
        _mm512_storeu_ps(data + index + offset, _mm512_sub_ps(left, right));
    }
    for (int64_t index = simd_end; index < half; index++) {
        float left = data[index];
        float right = data[index + offset];
        data[index] = left + right;
        data[index + offset] = left - right;
    }

}

void fht_inv_kacs_walk_avx512(float *data, int64_t size) {
    int64_t half = size / 2;
    int64_t base = size % 2;
    int64_t offset = base + half;
    int64_t simd_end = half & ~15LL;
    volatile float scale_scalar = 0.5f;
    const __m512 scale = _mm512_set1_ps(scale_scalar);
    for (int64_t index = 0; index < simd_end; index += 16) {
        __m512 left = _mm512_loadu_ps(data + index);
        __m512 right = _mm512_loadu_ps(data + index + offset);
        _mm512_storeu_ps(data + index, _mm512_mul_ps(_mm512_add_ps(left, right), scale));
        _mm512_storeu_ps(data + index + offset, _mm512_mul_ps(_mm512_sub_ps(left, right), scale));
    }
    for (int64_t index = simd_end; index < half; index++) {
        float left = data[index];
        float right = data[index + offset];
        data[index] = (left + right) * scale_scalar;
        data[index + offset] = (left - right) * scale_scalar;
    }
}

void fht_inplace_avx512(float *data, int64_t size) {
    for (int64_t width = 1; width < size; width <<= 1) {
        int64_t step = width << 1;
        int64_t simd_end = width & ~15LL;
        for (int64_t block = 0; block < size; block += step) {
            for (int64_t index = 0; index < simd_end; index += 16) {
                __m512 left = _mm512_loadu_ps(data + block + index);
                __m512 right = _mm512_loadu_ps(data + block + index + width);
                _mm512_storeu_ps(data + block + index, _mm512_add_ps(left, right));
                _mm512_storeu_ps(data + block + index + width, _mm512_sub_ps(left, right));
            }
            for (int64_t index = simd_end; index < width; index++) {
                float left = data[block + index];
                float right = data[block + index + width];
                data[block + index] = left + right;
                data[block + index + width] = left - right;
            }
        }
    }
}
