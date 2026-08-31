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

void fht_flip_sign_avx2(uint8_t *signs, float *data, int64_t size) {
    int64_t simd_end = size & ~31LL;
    for (int64_t index = 0; index < simd_end; index += 32) {
        uint32_t bits;
        __builtin_memcpy(&bits, signs + index / 8, sizeof(bits));
        for (int64_t block = 0; block < 4; block++) {
            uint64_t byte = (bits >> (block * 8)) & 0xff;
            volatile uint64_t mask0 = ((byte & 0x01) << 31) | ((byte & 0x02) << 62);
            volatile uint64_t mask1 = ((byte & 0x04) << 29) | ((byte & 0x08) << 60);
            volatile uint64_t mask2 = ((byte & 0x10) << 27) | ((byte & 0x20) << 58);
            volatile uint64_t mask3 = ((byte & 0x40) << 25) | ((byte & 0x80) << 56);
            __m256i mask = _mm256_set_epi64x(mask3, mask2, mask1, mask0);
            __m256 values = _mm256_loadu_ps(data + index + block * 8);
            values = _mm256_xor_ps(values, _mm256_castsi256_ps(mask));
            _mm256_storeu_ps(data + index + block * 8, values);
        }
    }
    for (int64_t index = simd_end; index < size; index++) {
        if (signs[index / 8] & (1u << (index % 8))) {
            data[index] = -data[index];
        }
    }
}

void fht_kacs_walk_avx2(float *data, int64_t size) {
    int64_t half = size / 2;
    int64_t base = size % 2;
    int64_t offset = base + half;
    int64_t simd_end = half & ~7LL;
    for (int64_t index = 0; index < simd_end; index += 8) {
        __m256 left = _mm256_loadu_ps(data + index);
        __m256 right = _mm256_loadu_ps(data + index + offset);
        _mm256_storeu_ps(data + index, _mm256_add_ps(left, right));
        _mm256_storeu_ps(data + index + offset, _mm256_sub_ps(left, right));
    }
    for (int64_t index = simd_end; index < half; index++) {
        float left = data[index];
        float right = data[index + offset];
        data[index] = left + right;
        data[index + offset] = left - right;
    }

}

void fht_inv_kacs_walk_avx2(float *data, int64_t size) {
    int64_t half = size / 2;
    int64_t base = size % 2;
    int64_t offset = base + half;
    int64_t simd_end = half & ~7LL;
    volatile float scale_scalar = 0.5f;
    const __m256 scale = _mm256_set1_ps(scale_scalar);
    for (int64_t index = 0; index < simd_end; index += 8) {
        __m256 left = _mm256_loadu_ps(data + index);
        __m256 right = _mm256_loadu_ps(data + index + offset);
        _mm256_storeu_ps(data + index, _mm256_mul_ps(_mm256_add_ps(left, right), scale));
        _mm256_storeu_ps(data + index + offset, _mm256_mul_ps(_mm256_sub_ps(left, right), scale));
    }
    for (int64_t index = simd_end; index < half; index++) {
        float left = data[index];
        float right = data[index + offset];
        data[index] = (left + right) * scale_scalar;
        data[index + offset] = (left - right) * scale_scalar;
    }
}

void fht_inplace_avx2(float *data, int64_t size) {
    for (int64_t width = 1; width < size; width <<= 1) {
        int64_t step = width << 1;
        int64_t simd_end = width & ~7LL;
        for (int64_t block = 0; block < size; block += step) {
            for (int64_t index = 0; index < simd_end; index += 8) {
                __m256 left = _mm256_loadu_ps(data + block + index);
                __m256 right = _mm256_loadu_ps(data + block + index + width);
                _mm256_storeu_ps(data + block + index, _mm256_add_ps(left, right));
                _mm256_storeu_ps(data + block + index + width, _mm256_sub_ps(left, right));
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
