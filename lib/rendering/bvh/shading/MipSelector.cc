// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

/// @file MipSelector.h

#include "MipSelector.h"
#include "MipSelector.hh"

#include <scene_rdl2/common/math/MathUtil.h>

namespace moonray {
namespace shading {

using namespace scene_rdl2::math;

namespace
{

#pragma warning(disable:177)   // #177: function was declared but never referenced

inline float
computeAreaHeuristic(Vec2f dx, Vec2f dy)
{
    // Treat dx and dy as edges of a parallelogram and compute the area.
    float len = max(length(dx), 0.00000001f);
    Vec2f normA = dx / len;

    Vec2f perp(-normA.y, normA.x);
    float height = fabsf(dot(perp, dy));
    return len * height;
}

inline float
computeMinHeuristic(Vec2f dx, Vec2f dy)
{
    float dx2 = lengthSqr(dx);
    float dy2 = lengthSqr(dy);
    return min(dx2, dy2);
}

inline float
computeMaxHeuristic(Vec2f dx, Vec2f dy)
{
    float dx2 = lengthSqr(dx);
    float dy2 = lengthSqr(dy);
    return max(dx2, dy2);
}

inline float
computeAvgHeuristic(Vec2f dx, Vec2f dy)
{
    float dx2 = lengthSqr(dx);
    float dy2 = lengthSqr(dy);
    return (dx2 + dy2) * 0.5f;
}

inline float
computeCustomMipSelector(const Vec2f dx, const Vec2f dy)
{
#if DEFAULT_MIP_FILTER == MIP_FILTER_MIN

    float driver = computeMinHeuristic(dx, dy);

#elif DEFAULT_MIP_FILTER == MIP_FILTER_MAX

    float driver = computeMaxHeuristic(dx, dy);

#elif DEFAULT_MIP_FILTER == MIP_FILTER_AVG

    float driver = computeAvgHeuristic(dx, dy);

#elif DEFAULT_MIP_FILTER == MIP_FILTER_AREA

    float driver = computeAreaHeuristic(dx, dy);

#else
    
    float driver = 0.f;

#endif

    float maxTexDimension = rsqrt(max(driver, 0.0000000001f));
    float mipSelector = log2f(max(maxTexDimension, 1.f));
    
    return clamp(mipSelector, 0.f, 15.999999f);
}

inline float
computeOIIOMipSelector(Vec2f dx, Vec2f dy)
{
    float dsdx = dx.x;
    float dtdx = dx.y; 
    float dsdy = dy.x; 
    float dtdy = dy.y; 

    //
    // See adjust_width in texturesys.cpp:
    //

    // deriv calculations from OpenImageIO's texturesys.cpp
    // Clamp degenerate derivatives so they don't cause mathematical problems
    static const float eps = 1.0e-8f, eps2 = eps*eps;
    float dxlen2 = dsdx*dsdx + dtdx*dtdx;
    float dylen2 = dsdy*dsdy + dtdy*dtdy;
    if (dxlen2 < eps2) {   // Tiny dx
        if (dylen2 < eps2) {
            // Tiny dx and dy: Essentially point sampling.  Substitute a
            // tiny but finite filter.
            dsdx = eps; dsdy = 0;
            dtdx = 0;   dtdy = eps;
        } else {
            // Tiny dx, sane dy -- pick a small dx orthogonal to dy, but
            // of length eps.
            float scale = eps / sqrtf(dylen2);
            dsdx = dtdy * scale;
            dtdx = -dsdy * scale;
        }
    } else if (dylen2 < eps2) {
        // Tiny dy, sane dx -- pick a small dy orthogonal to dx, but of
        // length eps.
        float scale = eps / sqrtf(dxlen2);
        dsdy = -dtdx * scale;
        dtdy = dsdx * scale;
    }

    //
    // See TextureSystemImpl::texture_lookup_trilinear_mipmap in texturesys.cpp.
    //

    float sfilt = std::max (fabsf(dsdx), fabsf(dsdy));
    float tfilt = std::max (fabsf(dtdx), fabsf(dtdy));

#if DEFAULT_MIP_FILTER == OIIO_CONSERVATIVE
    float filtwidth = std::max(sfilt, tfilt);
#else
    float filtwidth = std::min(sfilt, tfilt);
#endif

    float maxTexDimension = 1.f / max(filtwidth, 0.0000000001f);
    float mipSelector = log2f(max(maxTexDimension, 1.f));

    return clamp(mipSelector, 0.f, 15.999999f);
}

}   // End of anon namespace.

//----------------------------------------------------------------------------


// A mip selector contains the lowest resolution mip level to look up and the
// interpolation factor between them, like so:
//
// floor(mipSelector)   max dimension of lowest mip
//
//  0                        1
//  1                        2
//  2                        4
//  3                        8
//  4                       16
//  5                       32
//  6                       64
//  7                      128
//  8                      256
//  etc...                 etc...
//
// Lerp factor = frac(mipSelector)
float
computeMipSelector(float dsdx, float dtdx, float dsdy, float dtdy)
{
    Vec2f dx(dsdx, dtdx);
    Vec2f dy(dsdy, dtdy);
#if (DEFAULT_MIP_FILTER < START_OIIO_FILTER)
    return computeCustomMipSelector(dx, dy);
#elif (DEFAULT_MIP_FILTER == OIIO_CONSERVATIVE || DEFAULT_MIP_FILTER == OIIO_NON_CONSERVATIVE)
    return computeOIIOMipSelector(dx, dy);
#else
    #error Invalid value set for DEFAULT_MIP_FILTER.
#endif
}

} // namespace moonray
} // namespace shading


