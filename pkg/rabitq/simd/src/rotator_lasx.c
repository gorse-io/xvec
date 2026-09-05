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
#include <stdint.h>

// LASX port of VectorDB-NTU/RaBitQ-Library rotator kernels at revision
// 540242ea0a68926f1b827bf1f9add844f07a427b.

void flip_sign_lasx(const uint8_t *flip, float *data, int64_t dim) {
    for (int64_t i = 0; i < dim; i += 8) {
        v8u32 mask;
        for (int lane = 0; lane < 8; ++lane) {
            mask[lane] = (flip[(i + lane) / 8] >> ((i + lane) % 8) & 1)
                ? 0x80000000U
                : 0;
        }
        const __m256i values = __lasx_xvld(data + i, 0);
        __lasx_xvst(__lasx_xvxor_v(values, (__m256i)mask), data + i, 0);
    }
}

void kacs_walk_lasx(float *data, int64_t len) {
    const int64_t half = len / 2;
    for (int64_t i = 0; i < half; i += 8) {
        const __m256 first = (__m256)__lasx_xvld(data + i, 0);
        const __m256 second = (__m256)__lasx_xvld(data + half + i, 0);
        __lasx_xvst((__m256i)__lasx_xvfadd_s(first, second), data + i, 0);
        __lasx_xvst((__m256i)__lasx_xvfsub_s(first, second), data + half + i, 0);
    }
}
