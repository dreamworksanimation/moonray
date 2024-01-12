// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <scene_rdl2/common/math/BBox.h>
#include <scene_rdl2/common/math/ispc/Xformv.h>
#include <scene_rdl2/common/math/Math.h>
#include <scene_rdl2/common/math/Vec2.h>
#include <scene_rdl2/common/math/Vec3.h>
#include <scene_rdl2/common/math/Vec3fa.h>
#include <scene_rdl2/common/math/Vec4.h>
#include <scene_rdl2/common/math/Xform.h>
#include <scene_rdl2/common/platform/Platform.h>
#include <scene_rdl2/render/util/AlignedAllocator.h>

#include <vector>

namespace moonray {
namespace geom {

#if defined(__AVX512F__)
    typedef simd::mic_f simdf;
    typedef simd::mic_i simdi;
    typedef simd::mic_m simdb;
#elif defined(__AVX__)
    typedef simd::avxf simdf;
    typedef simd::avxi simdi;
    typedef simd::avxb simdb;
#endif

typedef scene_rdl2::math::BBox3f    BBox3f;
typedef scene_rdl2::math::BBox3fa   BBox3fa;
typedef scene_rdl2::math::Xform3f   Mat43;
typedef scene_rdl2::math::Vec2f     Vec2f;
typedef scene_rdl2::math::Vec3f     Vec3f;
typedef scene_rdl2::math::Vec3fa    Vec3fa;
typedef scene_rdl2::math::Vec4f     Vec4f;

} // namespace geom
} // namespace moonray


