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

float squared_euclidean_distance_fp32_avx(const float *lhs, const float *rhs,
                                                  int64_t size) {
  const float *last = lhs + size;
  const float *last_aligned = lhs + ((size >> 4) << 4);

  __m256 ymm_sum_0 = _mm256_setzero_ps();
  __m256 ymm_sum_1 = _mm256_setzero_ps();

  if (((uintptr_t)lhs & 0x1f) == 0 && ((uintptr_t)rhs & 0x1f) == 0) {
    for (; lhs != last_aligned; lhs += 16, rhs += 16) {
      __m256 ymm_d_0 =
          _mm256_sub_ps(_mm256_load_ps(lhs + 0), _mm256_load_ps(rhs + 0));
      __m256 ymm_d_1 =
          _mm256_sub_ps(_mm256_load_ps(lhs + 8), _mm256_load_ps(rhs + 8));
      ymm_sum_0 = _mm256_fmadd_ps(ymm_d_0, ymm_d_0, ymm_sum_0);
      ymm_sum_1 = _mm256_fmadd_ps(ymm_d_1, ymm_d_1, ymm_sum_1);
    }

    if (last >= last_aligned + 8) {
      __m256 ymm_d = _mm256_sub_ps(_mm256_load_ps(lhs), _mm256_load_ps(rhs));
      ymm_sum_0 = _mm256_fmadd_ps(ymm_d, ymm_d, ymm_sum_0);
      lhs += 8;
      rhs += 8;
    }
  } else {
    for (; lhs != last_aligned; lhs += 16, rhs += 16) {
      __m256 ymm_d_0 =
          _mm256_sub_ps(_mm256_loadu_ps(lhs + 0), _mm256_loadu_ps(rhs + 0));
      __m256 ymm_d_1 =
          _mm256_sub_ps(_mm256_loadu_ps(lhs + 8), _mm256_loadu_ps(rhs + 8));
      ymm_sum_0 = _mm256_fmadd_ps(ymm_d_0, ymm_d_0, ymm_sum_0);
      ymm_sum_1 = _mm256_fmadd_ps(ymm_d_1, ymm_d_1, ymm_sum_1);
    }

    if (last >= last_aligned + 8) {
      __m256 ymm_d = _mm256_sub_ps(_mm256_loadu_ps(lhs), _mm256_loadu_ps(rhs));
      ymm_sum_0 = _mm256_fmadd_ps(ymm_d, ymm_d, ymm_sum_0);
      lhs += 8;
      rhs += 8;
    }
  }
  float result = horizontal_add_fp32_v256(_mm256_add_ps(ymm_sum_0, ymm_sum_1));

  switch (last - lhs) {
    case 7:
      {
        float difference = lhs[6] - rhs[6];
        result += difference * difference;
      }
      /* FALLTHRU */
    case 6:
      {
        float difference = lhs[5] - rhs[5];
        result += difference * difference;
      }
      /* FALLTHRU */
    case 5:
      {
        float difference = lhs[4] - rhs[4];
        result += difference * difference;
      }
      /* FALLTHRU */
    case 4:
      {
        float difference = lhs[3] - rhs[3];
        result += difference * difference;
      }
      /* FALLTHRU */
    case 3:
      {
        float difference = lhs[2] - rhs[2];
        result += difference * difference;
      }
      /* FALLTHRU */
    case 2:
      {
        float difference = lhs[1] - rhs[1];
        result += difference * difference;
      }
      /* FALLTHRU */
    case 1:
      {
        float difference = lhs[0] - rhs[0];
        result += difference * difference;
      }
  }
  return result;
}
