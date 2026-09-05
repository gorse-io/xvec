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

#ifndef XVEC_RABITQ_SIMD_LASX_COMMON_H
#define XVEC_RABITQ_SIMD_LASX_COMMON_H

// Keep the LASX generators freestanding. Cross-target Clang installations do
// not necessarily have a LoongArch libc sysroot, while these kernels only need
// the compiler's fixed-width integer types.
typedef __UINT8_TYPE__ uint8_t;
typedef __UINT16_TYPE__ uint16_t;
typedef __UINT32_TYPE__ uint32_t;
typedef __UINT64_TYPE__ uint64_t;
typedef __INT32_TYPE__ int32_t;
typedef __INT64_TYPE__ int64_t;
#define UINT32_MAX __UINT32_MAX__

typedef uint8_t u8x8 __attribute__((vector_size(8)));
typedef uint8_t u8x16 __attribute__((vector_size(16)));
typedef uint8_t u8x32 __attribute__((vector_size(32)));
typedef uint16_t u16x16 __attribute__((vector_size(32)));
typedef uint32_t u32x8 __attribute__((vector_size(32)));
typedef int32_t i32x8 __attribute__((vector_size(32)));
typedef uint64_t u64x4 __attribute__((vector_size(32)));
typedef float f32x8 __attribute__((vector_size(32)));

static inline u8x32 splat_u8x32(uint8_t value) {
    return (u8x32){
        value, value, value, value, value, value, value, value,
        value, value, value, value, value, value, value, value,
        value, value, value, value, value, value, value, value,
        value, value, value, value, value, value, value, value,
    };
}

static inline u8x32 load_u8x32(const void *source) {
    u8x32 value;
    __builtin_memcpy(&value, source, sizeof(value));
    return value;
}

static inline u8x16 load_u8x16(const void *source) {
    u8x16 value;
    __builtin_memcpy(&value, source, sizeof(value));
    return value;
}

static inline u16x16 load_u16x16(const void *source) {
    u16x16 value;
    __builtin_memcpy(&value, source, sizeof(value));
    return value;
}

static inline u64x4 load_u64x4(const void *source) {
    u64x4 value;
    __builtin_memcpy(&value, source, sizeof(value));
    return value;
}

static inline f32x8 load_f32x8(const void *source) {
    f32x8 value;
    __builtin_memcpy(&value, source, sizeof(value));
    return value;
}

static inline void store_u8x32(void *destination, u8x32 value) {
    __builtin_memcpy(destination, &value, sizeof(value));
}

static inline void store_u8x16(void *destination, u8x16 value) {
    __builtin_memcpy(destination, &value, sizeof(value));
}

static inline void store_u16x16(void *destination, u16x16 value) {
    __builtin_memcpy(destination, &value, sizeof(value));
}

static inline void store_u32x8(void *destination, u32x8 value) {
    __builtin_memcpy(destination, &value, sizeof(value));
}

static inline void store_f32x8(void *destination, f32x8 value) {
    __builtin_memcpy(destination, &value, sizeof(value));
}

static inline float reduce_f32x8(f32x8 value) {
    float lanes[8];
    store_f32x8(lanes, value);
    return lanes[0] + lanes[1] + lanes[2] + lanes[3] +
           lanes[4] + lanes[5] + lanes[6] + lanes[7];
}

static inline f32x8 convert_u8x8_f32x8(u8x8 value) {
    return __builtin_convertvector(value, f32x8);
}

#endif
