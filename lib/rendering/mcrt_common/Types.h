// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

//
#pragma once

#include <scene_rdl2/common/math/Vec2.h>
#include <scene_rdl2/common/math/Vec3.h>
#include <scene_rdl2/common/math/Xform.h>
#include <scene_rdl2/common/math/simd.h>
#include <scene_rdl2/common/platform/Platform.h>
#include <scene_rdl2/render/util/AlignedAllocator.h>

#include <vector>
#include <utility>

// Hard coded value for capping count of progressive refinement passes.
#define MAX_RENDER_PASSES       1000u

namespace scene_rdl2 {

namespace rdl2 {
class Light;
}
}

namespace moonray {

namespace mcrt_common {

typedef scene_rdl2::math::Xform3f   Mat43;

// This should match up with embree's value for RTC_INVALID_GEOMETRY_ID.
#define RT_INVALID_RAY_ID       (-1)

#if defined (__AVX512F__)
  typedef scene_rdl2::math::Vec2<simd::sseb> sse2b;
  typedef scene_rdl2::math::Vec3<simd::sseb> sse3b;
  typedef scene_rdl2::math::Vec2<simd::ssei> sse2i;
  typedef scene_rdl2::math::Vec3<simd::ssei> sse3i;
  typedef scene_rdl2::math::Vec2<simd::ssef> sse2f;
  typedef scene_rdl2::math::Vec3<simd::ssef> sse3f;
#endif

#if defined (__AVX__)
  typedef scene_rdl2::math::Vec2<simd::avxb> avx2b;
  typedef scene_rdl2::math::Vec3<simd::avxb> avx3b;
  typedef scene_rdl2::math::Vec2<simd::avxi> avx2i;
  typedef scene_rdl2::math::Vec3<simd::avxi> avx3i;
  typedef scene_rdl2::math::Vec2<simd::avxf> avx2f;
  typedef scene_rdl2::math::Vec3<simd::avxf> avx3f;
#endif

#if defined (__AVX512F__)
  typedef scene_rdl2::math::Vec2<simd::mic_m> mic2b;
  typedef scene_rdl2::math::Vec3<simd::mic_m> mic3b;
  typedef scene_rdl2::math::Vec2<simd::mic_i> mic2i;
  typedef scene_rdl2::math::Vec3<simd::mic_i> mic3i;
  typedef scene_rdl2::math::Vec2<simd::mic_f> mic2f;
  typedef scene_rdl2::math::Vec3<simd::mic_f> mic3f;
#endif

#pragma warning push
#pragma warning disable 1684    // warning #1684: conversion from pointer to
                                // same-sized integral type (potential portability problem)

// Optimized AOS <--> SOA transposes cause 64 bit addresses to be split up.
// Here is a class to help deal with this issue.

// Use for non-const addresses. We must use the corresponding non-const class on
// the ISPC side.
struct Address64v
{
    void *get(unsigned idx)                 { return (void *)(((uint64_t(mHigh[idx])) << 32) | mLow[idx]); }
    void set(unsigned idx, void *addr)      { uint64_t p = uint64_t(addr); mHigh[idx] = p >> 32; mLow[idx] = p & 0xffffffff; }

    // Little endian specific layout, modify for big endian architectures.
    uint32_t mLow[VLEN];    // Least significant bits of a 64 bit address.
    uint32_t mHigh[VLEN];   // Most significant bits of a 64 bit address.
};

// Use for const addresses. We must use the corresponding const class on
// the ISPC side.
struct ConstAddress64v
{
    const void *get(unsigned idx) const         { return (void *)(((uint64_t(mHigh[idx])) << 32) | mLow[idx]); }
    void set(unsigned idx, const void *addr)    { uint64_t p = uint64_t(addr); mHigh[idx] = p >> 32; mLow[idx] = p & 0xffffffff; }

    // Little endian specific layout, modify for big endian architectures.
    uint32_t mLow[VLEN];    // Least significant bits of a 64 bit address.
    uint32_t mHigh[VLEN];   // Most significant bits of a 64 bit address.
};

#pragma warning pop

} // mcrt_common
} // namespace moonray

