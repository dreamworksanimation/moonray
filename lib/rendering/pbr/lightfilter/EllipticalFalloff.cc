// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0


#include "EllipticalFalloff.h"

#include <moonray/rendering/pbr/lightfilter/EllipticalFalloff_ispc_stubs.h>


namespace moonray {
namespace pbr {


using namespace scene_rdl2::math;


//----------------------------------------------------------------------------

namespace {

//
// Superellipse soft clipping
//  Inputs:
//    - point Q on the x-y plane
//    - the equations of two superellipses (with major/minor axes given by
//         a,b and A,B for the inner and outer ellipses, respectively)
//  Return value:
//    - 0 if Q was inside the inner ellipse
//    - 1 if Q was outside the outer ellipse
//    - smoothly varying from 0 to 1 in between
//
float
evalSuperEllipse(float x, float y,   // cartesian point we are evaluating at
                 float a, float b,   // inner superellipse major/minor axes
                 float A, float B,   // Outer superellipse major/minor axes
                 float roundness)    // shared roundness for both ellipses
{
    float t;

    x = scene_rdl2::math::abs(x);
    y = scene_rdl2::math::abs(y);

    if (roundness < 0.00001f) {
        /* Simpler case of a square */

        /* 1 - smoothstep(A, a, x) */
        if (x > A) {
            t = 0;
        } else if (x < a) {
            t = 1;
        } else if (A == a) { /* (A == a == x) */
            /* Degenerate case, don't want a divide by zero */
            t = 0;
        } else {
            t = 1 - (x - a) / (A - a);
        }

        /* 1 - smoothstep(B, b, y) */
        if (y > B) {
            t = 0;
        } else if (y >= b) {
            if (B == b) {
                /* Degenerate case, don't want a divide by zero */
                t = 0;
            } else {
                t *= 1 - (y - b) / (B - b);
            }
        }

        return t;

    } else {
        /* Harder, rounded corner case */
        float re = 2.0f / roundness;   /* roundness exponent */
        float q  = a * b * pow(pow(b*x, re) + pow(a*y, re), -1/re);
        float r  = A * B * pow(pow(B*x, re) + pow(A*y, re), -1/re);

        /* smoothstep(r, q, 1) */
        if (1 > r) {
            t = 1;
        } else if (1 < q) {
            t = 0;
        } else if (r == q) {
            /* Degenerate case, don't want a divide by zero */
            /* By setting t to 1, we return 0 */
            t = 1;
        } else {
            t = (1 - q) / (r - q);
        }

        return 1 - t;
    }
}

}   // end of anon namespace

//----------------------------------------------------------------------------

HUD_VALIDATOR(EllipticalFalloff);

EllipticalFalloff::EllipticalFalloff()
{
    init();
}

void
EllipticalFalloff::init(float roundness, float elliptical,
        OldFalloffCurveType curveType, float exp)
{
    mRoundness = saturate(roundness);  // square = 0, circle = 1
    mElliptical = elliptical;
    mOldFalloffCurve.init(curveType, exp);

    mInnerW = 0.0f;
    mInnerH = 0.0f;
    mOuterW = 0.0f;
    mOuterH = 0.0f;
}

void
EllipticalFalloff::setFov(float innerFov, float outerFov)
{
    innerFov = clamp(innerFov, 0.001f, 179.0f);
    outerFov = max(outerFov, innerFov + 0.1f);

    float widthToHeight = 0.0f;

    if (mElliptical >= 0.0f) {
        widthToHeight = 1.0f + mElliptical;
    } else {
        widthToHeight = 1.0f / (1.0f - mElliptical);
    }

    if (isZero(widthToHeight)) {
        mInnerW = 0.0f; mInnerH = 0.0f;
        mOuterW = 0.0f; mOuterH = 0.0f;
    } else if (widthToHeight <= 1.0f) {
        mInnerH = tan(deg2rad(innerFov * 0.5f)) / tan(deg2rad(outerFov * 0.5f));
        mOuterH = 1.0f;
        mInnerW = mInnerH * widthToHeight;
        mOuterW = (mOuterH - mInnerH) * widthToHeight + mInnerW;
    } else {
        mInnerW = tan(deg2rad(innerFov * 0.5f)) / tan(deg2rad(outerFov * 0.5f));
        mOuterW = 1.0f;
        mInnerH = mInnerW / widthToHeight;
        mOuterH = (mOuterW - mInnerW) / widthToHeight + mInnerH;
    }
}

// TODO: store this information to pass it into eval()
bool
EllipticalFalloff::intersect(float u, float v) const
{
    u = u * 2.0f - 1.0f;
    v = v * 2.0f - 1.0f;

    float attenuation = evalSuperEllipse(
        u, v,
        mInnerW, mInnerH,
        mOuterW, mOuterH,
        mRoundness);

    return attenuation > 0.0001f;
}

void
EllipticalFalloff::eval(float u, float v, Color *color) const
{
    u = u * 2.0f - 1.0f;
    v = v * 2.0f - 1.0f;

    float attenuation = evalSuperEllipse(
        u, v,
        mInnerW, mInnerH,
        mOuterW, mOuterH,
        mRoundness);

    attenuation = mOldFalloffCurve.eval(attenuation);

    *color *= attenuation;
}

//----------------------------------------------------------------------------

} // namespace pbr
} // namespace moonray

