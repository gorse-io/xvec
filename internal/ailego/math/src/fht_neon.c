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

#include <arm_neon.h>
#include <stdint.h>

void fht_flip_sign_neon(uint8_t *signs, float *data, int64_t size) {
    int64_t simd_end = size & ~3LL;
    for (int64_t index = 0; index < simd_end; index += 4) {
        uint16_t bits = signs[index / 8];
        if (index / 8 + 1 < (size + 7) / 8) {
            bits |= (uint16_t)signs[index / 8 + 1] << 8;
        }
        bits >>= index % 8;
        volatile uint64_t mask0 = ((uint64_t)(bits & 0x01) << 31) |
                                  ((uint64_t)(bits & 0x02) << 62);
        volatile uint64_t mask1 = ((uint64_t)(bits & 0x04) << 29) |
                                  ((uint64_t)(bits & 0x08) << 60);
        uint64x2_t mask64 = vcombine_u64(vcreate_u64(mask0), vcreate_u64(mask1));
        uint32x4_t values = vreinterpretq_u32_f32(vld1q_f32(data + index));
        vst1q_f32(data + index, vreinterpretq_f32_u32(veorq_u32(values, vreinterpretq_u32_u64(mask64))));
    }
    for (int64_t index = simd_end; index < size; index++) {
        if (signs[index / 8] & (1u << (index % 8))) {
            data[index] = -data[index];
        }
    }
}

void fht_kacs_walk_neon(float *data, int64_t size) {
    int64_t half = size / 2;
    int64_t base = size % 2;
    int64_t offset = base + half;
    int64_t simd_end = half & ~3LL;
    for (int64_t index = 0; index < simd_end; index += 4) {
        float32x4_t left = vld1q_f32(data + index);
        float32x4_t right = vld1q_f32(data + index + offset);
        vst1q_f32(data + index, vaddq_f32(left, right));
        vst1q_f32(data + index + offset, vsubq_f32(left, right));
    }
    for (int64_t index = simd_end; index < half; index++) {
        float left = data[index];
        float right = data[index + offset];
        data[index] = left + right;
        data[index + offset] = left - right;
    }

}

void fht_inv_kacs_walk_neon(float *data, int64_t size) {
    int64_t half = size / 2;
    int64_t base = size % 2;
    int64_t offset = base + half;

    int64_t simd_end = half & ~3LL;
    const float32x4_t scale = vdupq_n_f32(0.5f);
    for (int64_t index = 0; index < simd_end; index += 4) {
        float32x4_t left = vld1q_f32(data + index);
        float32x4_t right = vld1q_f32(data + index + offset);
        vst1q_f32(data + index, vmulq_f32(vaddq_f32(left, right), scale));
        vst1q_f32(data + index + offset, vmulq_f32(vsubq_f32(left, right), scale));
    }
    for (int64_t index = simd_end; index < half; index++) {
        float left = data[index];
        float right = data[index + offset];
        data[index] = (left + right) * 0.5f;
        data[index + offset] = (left - right) * 0.5f;
    }
}

void fht_inplace_neon(float *data, int64_t size) {
    for (int64_t width = 1; width < size; width <<= 1) {
        int64_t step = width << 1;
        int64_t simd_end = width & ~3LL;
        for (int64_t block = 0; block < size; block += step) {
            for (int64_t index = 0; index < simd_end; index += 4) {
                float32x4_t left = vld1q_f32(data + block + index);
                float32x4_t right = vld1q_f32(data + block + index + width);
                vst1q_f32(data + block + index, vaddq_f32(left, right));
                vst1q_f32(data + block + index + width, vsubq_f32(left, right));
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
