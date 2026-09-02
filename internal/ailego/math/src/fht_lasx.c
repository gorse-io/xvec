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

#include <lasxintrin.h>

void fht_flip_sign_lasx(unsigned char *signs, float *data, long size) {
    long simd_end = size & ~7L;
    for (long index = 0; index < simd_end; index += 8) {
        unsigned int bits = signs[index / 8];
        volatile unsigned int masks[8] = {
            (bits & 0x01u) << 31,
            (bits & 0x02u) << 30,
            (bits & 0x04u) << 29,
            (bits & 0x08u) << 28,
            (bits & 0x10u) << 27,
            (bits & 0x20u) << 26,
            (bits & 0x40u) << 25,
            (bits & 0x80u) << 24,
        };
        __m256i values = __lasx_xvld((void *)(data + index), 0);
        __m256i sign_masks = __lasx_xvld((void *)masks, 0);
        __lasx_xvst(__lasx_xvxor_v(values, sign_masks), data + index, 0);
    }
    for (long index = simd_end; index < size; index++) {
        if (signs[index / 8] & (1u << (index % 8))) {
            data[index] = -data[index];
        }
    }
}

void fht_kacs_walk_lasx(float *data, long size) {
    long half = size / 2;
    long base = size % 2;
    long offset = base + half;
    long simd_end = half & ~7L;
    for (long index = 0; index < simd_end; index += 8) {
        __m256 left = (__m256)__lasx_xvld((void *)(data + index), 0);
        __m256 right = (__m256)__lasx_xvld((void *)(data + index + offset), 0);
        __lasx_xvst((__m256i)__lasx_xvfadd_s(left, right), data + index, 0);
        __lasx_xvst((__m256i)__lasx_xvfsub_s(left, right), data + index + offset, 0);
    }
    for (long index = simd_end; index < half; index++) {
        float left = data[index];
        float right = data[index + offset];
        data[index] = left + right;
        data[index + offset] = left - right;
    }
}

void fht_inv_kacs_walk_lasx(float *data, long size) {
    long half = size / 2;
    long base = size % 2;
    long offset = base + half;
    long simd_end = half & ~7L;
    const __m256 scale = (__m256)__lasx_xvreplgr2vr_w(0x3f000000);
    for (long index = 0; index < simd_end; index += 8) {
        __m256 left = (__m256)__lasx_xvld((void *)(data + index), 0);
        __m256 right = (__m256)__lasx_xvld((void *)(data + index + offset), 0);
        __lasx_xvst((__m256i)__lasx_xvfmul_s(__lasx_xvfadd_s(left, right), scale), data + index, 0);
        __lasx_xvst((__m256i)__lasx_xvfmul_s(__lasx_xvfsub_s(left, right), scale), data + index + offset, 0);
    }
    for (long index = simd_end; index < half; index++) {
        float left = data[index];
        float right = data[index + offset];
        data[index] = (left + right) * 0.5f;
        data[index + offset] = (left - right) * 0.5f;
    }
}

void fht_inplace_lasx(float *data, long size) {
    for (long width = 1; width < size; width <<= 1) {
        long step = width << 1;
        long simd_end = width & ~7L;
        for (long block = 0; block < size; block += step) {
            for (long index = 0; index < simd_end; index += 8) {
                __m256 left = (__m256)__lasx_xvld((void *)(data + block + index), 0);
                __m256 right = (__m256)__lasx_xvld((void *)(data + block + index + width), 0);
                __lasx_xvst((__m256i)__lasx_xvfadd_s(left, right), data + block + index, 0);
                __lasx_xvst((__m256i)__lasx_xvfsub_s(left, right), data + block + index + width, 0);
            }
            for (long index = simd_end; index < width; index++) {
                float left = data[block + index];
                float right = data[block + index + width];
                data[block + index] = left + right;
                data[block + index + width] = left - right;
            }
        }
    }
}
