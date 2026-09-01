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

#include "distance_matrix_fp32.h"

float inner_product_and_squared_norm_fp32_avx512(
    const float *lhs, const float *rhs, int64_t size, float *lhs_norm,
    float *rhs_norm) {
  const float *last = lhs + size;
  const float *last_aligned = lhs + ((size >> 5) << 5);

  __m512 zmm_sum_0 = _mm512_setzero_ps();
  __m512 zmm_sum_1 = _mm512_setzero_ps();
  __m512 zmm_sum_norm1 = _mm512_setzero_ps();
  __m512 zmm_sum_norm2 = _mm512_setzero_ps();

  if (((uintptr_t)lhs & 0x3f) == 0 && ((uintptr_t)rhs & 0x3f) == 0) {
    for (; lhs != last_aligned; lhs += 32, rhs += 32) {
      __m512 zmm_lhs_0 = _mm512_load_ps(lhs + 0);
      __m512 zmm_lhs_1 = _mm512_load_ps(lhs + 16);
      __m512 zmm_rhs_0 = _mm512_load_ps(rhs + 0);
      __m512 zmm_rhs_1 = _mm512_load_ps(rhs + 16);
      FMA_FP32_AVX512(zmm_lhs_0, zmm_rhs_0, zmm_sum_0)
      FMA_FP32_AVX512(zmm_lhs_1, zmm_rhs_1, zmm_sum_1)
      FMA_FP32_AVX512(zmm_lhs_0, zmm_lhs_0, zmm_sum_norm1)
      FMA_FP32_AVX512(zmm_lhs_1, zmm_lhs_1, zmm_sum_norm1)
      FMA_FP32_AVX512(zmm_rhs_0, zmm_rhs_0, zmm_sum_norm2)
      FMA_FP32_AVX512(zmm_rhs_1, zmm_rhs_1, zmm_sum_norm2)
    }

    if (last >= last_aligned + 16) {
      __m512 zmm_lhs_0 = _mm512_load_ps(lhs);
      __m512 zmm_rhs_0 = _mm512_load_ps(rhs);
      FMA_FP32_AVX512(zmm_lhs_0, zmm_rhs_0, zmm_sum_0)
      FMA_FP32_AVX512(zmm_lhs_0, zmm_lhs_0, zmm_sum_norm1)
      FMA_FP32_AVX512(zmm_rhs_0, zmm_rhs_0, zmm_sum_norm2)
      lhs += 16;
      rhs += 16;
    }
  } else {
    for (; lhs != last_aligned; lhs += 32, rhs += 32) {
      __m512 zmm_lhs_0 = _mm512_loadu_ps(lhs + 0);
      __m512 zmm_lhs_1 = _mm512_loadu_ps(lhs + 16);
      __m512 zmm_rhs_0 = _mm512_loadu_ps(rhs + 0);
      __m512 zmm_rhs_1 = _mm512_loadu_ps(rhs + 16);
      FMA_FP32_AVX512(zmm_lhs_0, zmm_rhs_0, zmm_sum_0)
      FMA_FP32_AVX512(zmm_lhs_1, zmm_rhs_1, zmm_sum_1)
      FMA_FP32_AVX512(zmm_lhs_0, zmm_lhs_0, zmm_sum_norm1)
      FMA_FP32_AVX512(zmm_lhs_1, zmm_lhs_1, zmm_sum_norm1)
      FMA_FP32_AVX512(zmm_rhs_0, zmm_rhs_0, zmm_sum_norm2)
      FMA_FP32_AVX512(zmm_rhs_1, zmm_rhs_1, zmm_sum_norm2)
    }

    if (last >= last_aligned + 16) {
      __m512 zmm_lhs_0 = _mm512_loadu_ps(lhs);
      __m512 zmm_rhs_0 = _mm512_loadu_ps(rhs);
      FMA_FP32_AVX512(zmm_lhs_0, zmm_rhs_0, zmm_sum_0)
      FMA_FP32_AVX512(zmm_lhs_0, zmm_lhs_0, zmm_sum_norm1)
      FMA_FP32_AVX512(zmm_rhs_0, zmm_rhs_0, zmm_sum_norm2)
      lhs += 16;
      rhs += 16;
    }
  }

  zmm_sum_0 = _mm512_add_ps(zmm_sum_0, zmm_sum_1);
  if (lhs != last) {
    __mmask16 mask = (__mmask16)((1 << (last - lhs)) - 1);
    __m512 zmm_undefined = _mm512_undefined_ps();
    __m512 zmm_lhs_0 = _mm512_mask_loadu_ps(zmm_undefined, mask, lhs);
    __m512 zmm_rhs_0 = _mm512_mask_loadu_ps(zmm_undefined, mask, rhs);
    FMA_MASK_FP32_AVX512(zmm_lhs_0, zmm_rhs_0, zmm_sum_0, mask);
    FMA_MASK_FP32_AVX512(zmm_lhs_0, zmm_lhs_0, zmm_sum_norm1, mask);
    FMA_MASK_FP32_AVX512(zmm_rhs_0, zmm_rhs_0, zmm_sum_norm2, mask);
  }

  *lhs_norm = horizontal_add_fp32_v512(zmm_sum_norm1);
  *rhs_norm = horizontal_add_fp32_v512(zmm_sum_norm2);
  return horizontal_add_fp32_v512(zmm_sum_0);
}
