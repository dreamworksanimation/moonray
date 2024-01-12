// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0


#include "LightFilter.isph"
#include <scene_rdl2/common/platform/IspcUtil.isph>

//----------------------------------------------------------------------------

ISPC_UTIL_EXPORT_UNIFORM_STRUCT_TO_HEADER(RodLightFilter);

export uniform uint32_t
RodLightFilter_hudValidation(uniform bool verbose)
{
    ROD_LIGHT_FILTER_VALIDATION;
}


//----------------------------------------------------------------------------

varying Vec3f
RodLightFilter_slerpPointRender2Local(const uniform RodLightFilter * uniform rodlif,
                                      const varying Vec3f &p, varying float time)
{
    // inverse of Local2Render is (p + t) * r * s
    // where t = -position, r = render2localRot.l, s = render2localScale
    const Vec3f t = (rodlif->mMb & LIGHTFILTER_MB_TRANSLATION) ?
        -1.f * lerp(rodlif->mPosition[0], rodlif->mPosition[1], time) :
        -1.f * rodlif->mPosition[0];
    const Mat3f r = (rodlif->mMb & LIGHTFILTER_MB_ROTATION)?
        transpose(Mat3f_ctor(slerp(rodlif->mOrientation[0], rodlif->mOrientation[1], time))) : // inverse is transpose
        rodlif->mRender2LocalRot0.l;
    const Mat3f s = (rodlif->mMb & LIGHTFILTER_MB_SCALE) ?
        inverse(lerp(rodlif->mLocal2RenderScale[0], rodlif->mLocal2RenderScale[1], time)) :
        rodlif->mRender2LocalScale0;

    return ((p + t) * r) * s;
}

inline varying Vec3f
RodLightFilter_xformPointRender2Local(const uniform RodLightFilter * uniform rodlif,
                                      const varying Vec3f &p, varying float time)
{
    if (!rodlif->mMb) return transformPoint(rodlif->mRender2Local0, p);

    return RodLightFilter_slerpPointRender2Local(rodlif, p, time);
}

// This function is for quickly determining if a filter can be skipped.
// It returns true if the sphere at point p with radius rad is completely
// outside the region of influence of a filter. Since filters can
// be inverted, this means the sphere is either out of the volume, or
// if the filter is inverted, entirely within the interior of the volume.
// It accounts for the radius and edge zones.
//
// Limitation: Currently this function can only quickly reject if the filter is
// not moving. If the filter is moving (has motion blur, i.e. mMb is true)
// this function does not help. Camera motion blur is fine, however.
varying bool
RodLightFilter_isOutsideInfluence(const uniform RodLightFilter * uniform rodlif,
                                  const varying Vec3f &p, varying float rad)
{
    if (rodlif->mMb) return false;

    Vec3f localP = abs((p - rodlif->mPosition[0]) * rodlif->mRender2LocalRotAndScale);
    float buf = rodlif->mRadiusEdgeSum + rad;

    if (rodlif->mInvert) {
        buf = -buf;
    }

    bool out = false;
    if (localP.x > rodlif->mBoxCorner.x + buf ||
        localP.y > rodlif->mBoxCorner.y + buf ||
        localP.z > rodlif->mBoxCorner.z + buf) {
        out = true;
    }

    if (rodlif->mInvert) {
        out = !out;
    }

    return out;
}

// Signed distance to a round box.
//
// Original implementation (for reference) taken from:
// https://www.iquilezles.org/www/articles/distfunctions/distfunctions.htm
//
//   Provides exact distance to boundary (rather than conservative minimum)
//   float sdRoundBox( vec3 p, vec3 b, float r )
//   {
//     vec3 q = abs(p) - b;
//     return length(max(q,0.0)) + min(max(q.x,max(q.y,q.z)),0.0) - r;
//   }
varying float
RodLightFilter_signedDistanceRoundBox(const uniform RodLightFilter * uniform rodlif,
                                      const varying Vec3f &p )
{
  Vec3f q = abs(p) - rodlif->mBoxCorner;
  return
      length(max(q, Vec3f_ctor(0.f))) +
      min(reduce_max(q), 0.f) - rodlif->mRadius;
}

varying bool
RodLightFilter_canIlluminate(const uniform LightFilter * uniform lif,
        const varying CanIlluminateData &data)
{
    const uniform RodLightFilter * uniform rodlif = (const uniform RodLightFilter * uniform)lif;
    
    // If the filter is less than full effect, it cannot
    // completely block light.
    if (rodlif->mDensity < 1.f) {
        return true;
    }
    
    if (RodLightFilter_isOutsideInfluence(rodlif,
                                          data.shadingPointPosition,
                                          data.shadingPointRadius))
        return true;

    varying Vec3f localPoint =
        RodLightFilter_xformPointRender2Local(
            rodlif, data.shadingPointPosition, data.time);

    varying float r = RodLightFilter_signedDistanceRoundBox(rodlif, localPoint);

    if (rodlif->mInvert) {
        // Everything outside the volume is unlit
        if (isBlack(rodlif->mColor)) {
            if (r > (data.shadingPointRadius + rodlif->mEdge))
            //if (sqrPart > sqr(data.shadingPointRadius + mEdge - regPart))
                return false;
        }
    } else {
        // The only place that is completely unlit
        // is within the volume and if mColor = 0
        if (isBlack(rodlif->mColor)) {
            if (r < -data.shadingPointRadius)
            //if (sqrPart < sqr(-data.shadingPointRadius - regPart))
                return false;
        }
    }

    return true;
}

void
RodLightFilter_eval(const uniform LightFilter * uniform lif,
        const varying EvalData& data,
        varying Color * uniform filterValue)
{
    const uniform RodLightFilter * uniform rodlif = (const uniform RodLightFilter * uniform)lif;

    if (RodLightFilter_isOutsideInfluence(rodlif, data.shadingPointPosition, 0)) {
        *filterValue = Color_ctor(1.f);
        return;
    }

    varying Vec3f localPoint = RodLightFilter_xformPointRender2Local(
        rodlif, data.shadingPointPosition, data.time);

    varying float r = RodLightFilter_signedDistanceRoundBox(rodlif, localPoint);

    float scale = 1.f;
    if (r <= 0.f) {
        scale = 0.f;
    } else if (r < rodlif->mEdge) {
        r *= rodlif->mEdgeInv;
        scale = FloatRampControl_eval1D(r, &rodlif->mRamp);
    }

    if (rodlif->mInvert) {
        scale = 1.f - scale;
    }

    // apply density (lerp)
    scale = 1.f + (scale - 1.f) * rodlif->mDensity;

    *filterValue = rodlif->mColor + (sWhite - rodlif->mColor) * scale;
}

//----------------------------------------------------------------------------

export void
RodLightFilter_init(uniform RodLightFilter * uniform lightfilter)
{
    LightFilter_init((uniform LightFilter * uniform)lightfilter,
            RodLightFilter_canIlluminate, RodLightFilter_eval);
}


//----------------------------------------------------------------------------

