// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0


#include "RodLightFilter.h"
#include <moonray/rendering/pbr/core/Util.h>
#include <moonray/rendering/pbr/lightfilter/RodLightFilter_ispc_stubs.h>
#include <moonray/rendering/shading/ispc/RampControl_ispc_stubs.h>
#include <moonray/common/mcrt_macros/moonray_static_check.h>

namespace moonray {
namespace pbr{

using namespace scene_rdl2;
using namespace scene_rdl2::math;

bool RodLightFilter::sAttributeKeyInitialized;
rdl2::AttributeKey<rdl2::Mat4d> RodLightFilter::sNodeXformKey;
rdl2::AttributeKey<rdl2::Float> RodLightFilter::sWidthKey;
rdl2::AttributeKey<rdl2::Float> RodLightFilter::sDepthKey;
rdl2::AttributeKey<rdl2::Float> RodLightFilter::sHeightKey;
rdl2::AttributeKey<rdl2::Float> RodLightFilter::sRadiusKey;
rdl2::AttributeKey<rdl2::Float> RodLightFilter::sEdgeKey;
rdl2::AttributeKey<rdl2::Rgb> RodLightFilter::sColorKey;
rdl2::AttributeKey<rdl2::Float> RodLightFilter::sIntensityKey;
rdl2::AttributeKey<rdl2::Float> RodLightFilter::sDensityKey;
rdl2::AttributeKey<rdl2::Bool> RodLightFilter::sInvertKey;
rdl2::AttributeKey<rdl2::FloatVector> RodLightFilter::sRampInKey;
rdl2::AttributeKey<rdl2::FloatVector> RodLightFilter::sRampOutKey;
rdl2::AttributeKey<rdl2::IntVector> RodLightFilter::sRampInterpolationTypesKey;

HUD_VALIDATOR(RodLightFilter);

RodLightFilter::RodLightFilter(const rdl2::LightFilter* rdlLightFilter) :
    LightFilter(rdlLightFilter)
{
    if (mRdlLightFilter) {
        initAttributeKeys(mRdlLightFilter->getSceneClass());
    }

    ispc::RodLightFilter_init((ispc::RodLightFilter *)this->asIspc());
}

void
RodLightFilter::initAttributeKeys(const rdl2::SceneClass &sc)
{
    if (sAttributeKeyInitialized) {
        return;
    }

    MOONRAY_START_NON_THREADSAFE_STATIC_WRITE

    sAttributeKeyInitialized = true;

    sNodeXformKey = sc.getAttributeKey<rdl2::Mat4d>("node_xform");
    sWidthKey = sc.getAttributeKey<rdl2::Float>("width");
    sDepthKey = sc.getAttributeKey<rdl2::Float>("depth");
    sHeightKey = sc.getAttributeKey<rdl2::Float>("height");
    sRadiusKey = sc.getAttributeKey<rdl2::Float>("radius");
    sEdgeKey = sc.getAttributeKey<rdl2::Float>("edge");
    sColorKey = sc.getAttributeKey<rdl2::Rgb>("color");
    sIntensityKey = sc.getAttributeKey<rdl2::Float>("intensity");
    sDensityKey = sc.getAttributeKey<rdl2::Float>("density");
    sInvertKey = sc.getAttributeKey<rdl2::Bool>("invert");

    sRampInKey = sc.getAttributeKey<rdl2::FloatVector>("ramp_in_distances");
    sRampOutKey = sc.getAttributeKey<rdl2::FloatVector>("ramp_out_distances");
    sRampInterpolationTypesKey = sc.getAttributeKey<rdl2::IntVector>("ramp_interpolation_types");

    MOONRAY_FINISH_NON_THREADSAFE_STATIC_WRITE
}

bool
RodLightFilter::updateTransforms(const Mat4f &local2Render, int ti)
{
    // update transforms at time index (ti)
    // ti == 0 is at normalized rayTime = 0.f
    // ti == 1 is at normalized rayTime = 1.f;
    MNRY_ASSERT(mRdlLightFilter);


    Quaternion3f rot;
    Mat3f scale;
    Xform3f local2RenderXform = xform<Xform3f>(local2Render);
    decompose(local2RenderXform.l, scale, rot);

    mPosition[ti] = asVec3(local2Render.row3());

    mLocal2RenderScale[ti] = scale;

    // Single shutter-time items (for when there is no MB)
    if (ti == 0) {
        mRender2Local0 = local2RenderXform.inverse();
        mRender2LocalRot0 = Mat3f(rot).transposed();
        mRender2LocalScale0 = scale.inverse();
    }

    Xform3f local2RenderRot = Xform3f(Mat3f(rot), zero);
    mOrientation[ti] = normalize(
        scene_rdl2::math::Quaternion3f(local2RenderRot.row0(),
                           local2RenderRot.row1(),
                           local2RenderRot.row2()));

    // After setting the second quaternion (ti==1), check if the angle from the
    // first quaternion is negative and flip accordingly.
    if (ti==1 && dot(mOrientation[0], mOrientation[1]) < 0) {
        mOrientation[1] *= -1.f;
    }

    return true;
}

bool
RodLightFilter::updateParamAndTransforms(const Mat4f &local2Render0,
                                         const Mat4f &local2Render1)
{
    if (!updateTransforms(local2Render0, /* ti = */ 0)) return false;

    // setup mMb
    mMb = LIGHTFILTER_MB_NONE;
    const rdl2::SceneVariables &vars =
        mRdlLightFilter->getSceneClass().getSceneContext()->getSceneVariables();
    const bool mb = vars.get(rdl2::SceneVariables::sEnableMotionBlur) &&
        !isEqual(local2Render0, local2Render1);

    if (mb) {
        if (!updateTransforms(local2Render1, /* ti = */ 1)) return false;

        // set specific mb bits
        if (!isEqual(mPosition[0], mPosition[1]))
            mMb |= LIGHTFILTER_MB_TRANSLATION;
        if (!isEqual(mOrientation[0], mOrientation[1]))
            mMb |= LIGHTFILTER_MB_ROTATION;
        if (!isEqual(mLocal2RenderScale[0], mLocal2RenderScale[1]))
            mMb |= LIGHTFILTER_MB_SCALE;
    }

    return true;
}

void
RodLightFilter::update(const LightFilterMap& /*lightFilters*/,
                       const Mat4d& world2render)
{
    if (!mRdlLightFilter) {
        return;
    }

    const Mat4d l2w0 = mRdlLightFilter->get<rdl2::Mat4d>(sNodeXformKey, 0.0f);
    const Mat4d l2w1 = mRdlLightFilter->get<rdl2::Mat4d>(sNodeXformKey, 1.0f);

    const Mat4f local2Render0 = toFloat(l2w0 * world2render);
    const Mat4f local2Render1 = toFloat(l2w1 * world2render);

    if (!updateParamAndTransforms(local2Render0, local2Render1)) {
        return;
    }

    mWidth = mRdlLightFilter->get<rdl2::Float>(sWidthKey);
    mDepth = mRdlLightFilter->get<rdl2::Float>(sDepthKey);
    mHeight = mRdlLightFilter->get<rdl2::Float>(sHeightKey);
    mRadius = mRdlLightFilter->get<rdl2::Float>(sRadiusKey);
    mEdge = mRdlLightFilter->get<rdl2::Float>(sEdgeKey);
    mEdgeInv = mEdge ? 1.f/mEdge : 1.f;
    mColor = mRdlLightFilter->get<rdl2::Rgb>(sColorKey);
    mIntensity = mRdlLightFilter->get<rdl2::Float>(sIntensityKey);
    mColor *= mIntensity;
    mDensity = clamp(mRdlLightFilter->get<rdl2::Float>(sDensityKey), 0.f, 1.f);
    mInvert = mRdlLightFilter->get<rdl2::Bool>(sInvertKey);
    mBoxCorner = abs(Vec3f(mWidth, mHeight, mDepth)*0.5f);
    mRender2LocalRotAndScale = mRender2LocalRot0.l * mRender2LocalScale0;
    mRadiusEdgeSum = mRadius+mEdge;

    std::vector<float> inDistancesVec = mRdlLightFilter->get<rdl2::FloatVector>(sRampInKey);
    std::vector<float> outDistancesVec = mRdlLightFilter->get<rdl2::FloatVector>(sRampOutKey);
    std::vector<int> interpolationTypesVec = mRdlLightFilter->get<rdl2::IntVector>(sRampInterpolationTypesKey);

    if (inDistancesVec.size() != outDistancesVec.size() || inDistancesVec.size() != interpolationTypesVec.size()) {
        mRdlLightFilter->error(
            "Rod ramp light filter ramp_in_distances, ramp_out_distances and ramp_interpolation_types are different sizes, using defaults");
        outDistancesVec = {0.f, 1.f};
        inDistancesVec = {0.f, 1.f};
        interpolationTypesVec = {ispc::RAMP_INTERPOLATOR_MODE_MONOTONECUBIC, ispc::RAMP_INTERPOLATOR_MODE_MONOTONECUBIC};
    }

    mRamp.init(
        inDistancesVec.size(),
        inDistancesVec.data(),
        outDistancesVec.data(),
        reinterpret_cast<const ispc::RampInterpolatorMode*>(interpolationTypesVec.data()));
}

Vec3f
RodLightFilter::slerpPointRender2Local(const Vec3f &p, float time) const
{
    // inverse of Local2Render is (p + t) * r * s
    // where t = -position, r = render2localRot.l, s = render2localScale
    const Vec3f t = (mMb & LIGHTFILTER_MB_TRANSLATION) ?
        -1.f * lerp(mPosition[0], mPosition[1], time) :
        -1.f * mPosition[0];
    const Mat3f r = (mMb & LIGHTFILTER_MB_ROTATION)?
        // transpose is inverse
        Mat3f(slerp(mOrientation[0], mOrientation[1], time)).transposed() :
        mRender2LocalRot0.l;
    const Mat3f s = (mMb & LIGHTFILTER_MB_SCALE) ?
        // The inverse() call below, while unfortunate, is necessary.
        // Reversing the order of operations (and lerping precomputed inverses)
        // does not look correct.
        lerp(mLocal2RenderScale[0], mLocal2RenderScale[1], time).inverse() :
        mRender2LocalScale0;

    return ((p + t) * r) * s;
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
//
// TODO: Add support for motion blur
//
// I have not profiled this code. Currently the scalar version incurs a
// significant slow-down with many (43) RLF filters, compared to vector. The
// reason for the difference is unknown. However, each filter incurs an
// overhead for each light sample, as it must do a bounding box check in this
// function (for static filters) and a very-slow full-quality check (using
// slerpPointRender2Local and signedDistanceRoundBox) for moving filters. We
// should be smarter about knowing when a filter applies, using something like
// a BVH.
//
// TODO: Profiling and performance improvements. Possibly use a BVH.
//
bool
RodLightFilter::isOutsideInfluence(const Vec3f &p, float rad) const
{
    if (mMb) return false;

    Vec3f localP = abs((p - mPosition[0]) * mRender2LocalRotAndScale);
    float buf = mRadiusEdgeSum + rad;

    if (mInvert) {
        buf = -buf;
    }

    bool out = false;
    if (localP.x > mBoxCorner.x + buf ||
        localP.y > mBoxCorner.y + buf ||
        localP.z > mBoxCorner.z + buf) {
        out = true;
    }

    if (mInvert) {
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
float RodLightFilter::signedDistanceRoundBox( const Vec3f &p ) const
{
  Vec3f q = abs(p) - mBoxCorner;
  return length(max(q, Vec3f(0.f))) + min(reduce_max(q), 0.f) - mRadius;
}

// TODO: Replace signedDistanceRoundBox with the function below
// that avoids a sqrt(). In my small test with 43 filters
// it reduced render time 1.5%. However, it is less readable.
// Perhaps that could be improved.
//
//
// Optimized version of signedDistanceRoundBox(). Not in use!
//
//   Using this version of the signedDistanceRoundBox function lets us avoid a
//   sqrt in canIlluminate(). It returns a squared part, and a regular part.
//
//   For reference, canIlluminate() does a test like:
//     r = signedDistanceRoundBox();
//     ...
//     if (r < -data.shadingPointRadius)
//         ... ^^^^^^^^^^^^^^^^^^^^^^^^ call this "a"
//
//      r       becomes sqrt(sqrPart)+regPart
//      r < a   becomes sqrt(sqrPart)+regPart <  a
//                      sqrt(sqrPart)     <  a-regPart
//                           sqrPart      < (a-regPart)^2
//
//   To use (untested):
//
//   float sqrPart, regPart;
//   signedDistanceRoundBoxSqrReg(localPoint, sqrPart, regPart);
//
//   if (sqrPart < (a-regPart)*(a-regPart))
//       ...
//
// void RodLightFilter::signedDistanceRoundBoxSqrReg(
//     const Vec3f &p, float &sqrPart, float &regPart ) const
// {
//   Vec3f q = abs(p) - mBoxCorner;
//   sqrPart = lengthSqr(max(q, Vec3f(0.f)));
//   regPart = min(reduce_max(q), 0.f) - mRadius;
// }

bool
RodLightFilter::canIlluminate(const CanIlluminateData& data) const
{
    // Approximate light and shading point as two spheres. We can compute the
    // minimum and maximum distances between any 2 points in those two spheres.

    // If the filter is less than full effect, it cannot
    // completely block light.
    if (mDensity < 1.f) {
        return true;
    }

    if (isOutsideInfluence(data.shadingPointPosition,
                           data.shadingPointRadius)) {
        return true;
    }

    Vec3f localPoint = xformPointRender2Local(data.shadingPointPosition,
                                              data.time);

    float r = signedDistanceRoundBox(localPoint);

    if (mInvert) {
        // Everything outside the volume is unlit
        if (isBlack(mColor)) {
            if (r > (data.shadingPointRadius + mEdge)) {
            //if (sqrPart > sqr(data.shadingPointRadius + mEdge - regPart))
                return false;
            }
        }
    } else {
        // The only place that is completely unlit
        // is within the volume and if mColor = 0
        if (isBlack(mColor)) {
            if (r < -data.shadingPointRadius) {
            //if (sqrPart < sqr(-data.shadingPointRadius - regPart))
                return false;
            }
        }
    }

    return true;
}

Color
RodLightFilter::eval(const EvalData& data) const
{
    if (isOutsideInfluence(data.shadingPointPosition, 0)) {
        return Color(1.f);
    }

    Vec3f localPoint = xformPointRender2Local(data.shadingPointPosition, data.time);

    float r = signedDistanceRoundBox(localPoint);

    float scale = 1.f;
    if (r <= 0.f) {
        scale = 0.f;
    } else if (r < mEdge) {
        r *= mEdgeInv;
        scale = mRamp.eval1D(r);
    }

    if (mInvert) {
        scale = 1.f - scale;
    }

    // apply density (lerp)
    scale = scene_rdl2::math::lerp(1.f, scale, mDensity);

    Color result = scene_rdl2::math::lerp(mColor, Color(1.f), scale);

    return result;
}

} //namespace pbr
} //namespace moonray

