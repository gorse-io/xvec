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

float inner_product_fp32_avx(const float *lhs, const float *rhs,
                                      int64_t size) {
  const float *last = lhs + size;
  const float *last_aligned = lhs + ((size >> 4) << 4);

  __m256 ymm_sum_0 = _mm256_setzero_ps();
  __m256 ymm_sum_1 = _mm256_setzero_ps();

  if (((uintptr_t)lhs & 0x1f) == 0 && ((uintptr_t)rhs & 0x1f) == 0) {
    for (; lhs != last_aligned; lhs += 16, rhs += 16) {
      __m256 ymm_lhs_0 = _mm256_load_ps(lhs + 0);
      __m256 ymm_lhs_1 = _mm256_load_ps(lhs + 8);
      __m256 ymm_rhs_0 = _mm256_load_ps(rhs + 0);
      __m256 ymm_rhs_1 = _mm256_load_ps(rhs + 8);
      ymm_sum_0 = _mm256_fmadd_ps(ymm_lhs_0, ymm_rhs_0, ymm_sum_0);
      ymm_sum_1 = _mm256_fmadd_ps(ymm_lhs_1, ymm_rhs_1, ymm_sum_1);
    }

    if (last >= last_aligned + 8) {
      ymm_sum_0 =
          _mm256_fmadd_ps(_mm256_load_ps(lhs), _mm256_load_ps(rhs), ymm_sum_0);
      lhs += 8;
      rhs += 8;
    }
  } else {
    for (; lhs != last_aligned; lhs += 16, rhs += 16) {
      __m256 ymm_lhs_0 = _mm256_loadu_ps(lhs + 0);
      __m256 ymm_lhs_1 = _mm256_loadu_ps(lhs + 8);
      __m256 ymm_rhs_0 = _mm256_loadu_ps(rhs + 0);
      __m256 ymm_rhs_1 = _mm256_loadu_ps(rhs + 8);
      ymm_sum_0 = _mm256_fmadd_ps(ymm_lhs_0, ymm_rhs_0, ymm_sum_0);
      ymm_sum_1 = _mm256_fmadd_ps(ymm_lhs_1, ymm_rhs_1, ymm_sum_1);
    }

    if (last >= last_aligned + 8) {
      ymm_sum_0 = _mm256_fmadd_ps(_mm256_loadu_ps(lhs), _mm256_loadu_ps(rhs),
                                  ymm_sum_0);
      lhs += 8;
      rhs += 8;
    }
  }
  float result = horizontal_add_fp32_v256(_mm256_add_ps(ymm_sum_0, ymm_sum_1));

  switch (last - lhs) {
    case 7:
      result += lhs[6] * rhs[6];
      /* FALLTHRU */
    case 6:
      result += lhs[5] * rhs[5];
      /* FALLTHRU */
    case 5:
      result += lhs[4] * rhs[4];
      /* FALLTHRU */
    case 4:
      result += lhs[3] * rhs[3];
      /* FALLTHRU */
    case 3:
      result += lhs[2] * rhs[2];
      /* FALLTHRU */
    case 2:
      result += lhs[1] * rhs[1];
      /* FALLTHRU */
    case 1:
      result += lhs[0] * rhs[0];
  }
  return result;
}
