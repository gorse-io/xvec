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

#include "lasx_common.h"

// LASX port of VectorDB-NTU/RaBitQ-Library rotator kernels at revision
// 540242ea0a68926f1b827bf1f9add844f07a427b.

void flip_sign_lasx(const uint8_t *flip, float *data, int64_t dim) {
    for (int64_t i = 0; i < dim; i += 8) {
        u32x8 mask;
        for (int lane = 0; lane < 8; ++lane) {
            mask[lane] = (flip[(i + lane) / 8] >> ((i + lane) % 8) & 1)
                ? 0x80000000U
                : 0;
        }
        u32x8 values;
        __builtin_memcpy(&values, data + i, sizeof(values));
        values ^= mask;
        __builtin_memcpy(data + i, &values, sizeof(values));
    }
}

void kacs_walk_lasx(float *data, int64_t len) {
    const int64_t half = len / 2;
    for (int64_t i = 0; i < half; i += 8) {
        const f32x8 first = load_f32x8(data + i);
        const f32x8 second = load_f32x8(data + half + i);
        store_f32x8(data + i, first + second);
        store_f32x8(data + half + i, first - second);
    }
}
