// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0


#include "RectLight.h"
#include <moonray/rendering/pbr/core/Distribution.h>
#include <moonray/rendering/pbr/core/Util.h>

#include <moonray/rendering/pbr/light/RectLight_ispc_stubs.h>

#include <scene_rdl2/scene/rdl2/rdl2.h>
#include <moonray/common/mcrt_macros/moonray_static_check.h>


using namespace scene_rdl2;
using namespace scene_rdl2::math;
using scene_rdl2::logging::Logger;

namespace moonray {
namespace pbr {


bool                             RectLight::sAttributeKeyInitialized;
scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Bool>   RectLight::sNormalizedKey;
scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Bool>   RectLight::sApplySceneScaleKey;
scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Float>  RectLight::sWidthKey;
scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Float>  RectLight::sHeightKey;
scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Float>  RectLight::sSpreadKey;
scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Float>  RectLight::sClearRadiusKey;
scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Float>  RectLight::sClearRadiusFalloffDistanceKey;
scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Int>    RectLight::sClearRadiusInterpolationKey;
scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Int>    RectLight::sSidednessKey;

//----------------------------------------------------------------------------

HUD_VALIDATOR(RectLight);

void
RectLight::computeCorners(Vec3f *corners, float time) const
{
    corners[0] = xformPointLocal2Render(Vec3f(mHalfWidth,  mHalfHeight, 0.f), time);
    corners[1] = xformPointLocal2Render(Vec3f(mHalfWidth, -mHalfHeight, 0.f), time);
    corners[2] = xformPointLocal2Render(Vec3f(-mHalfWidth,  mHalfHeight, 0.f), time);
    corners[3] = xformPointLocal2Render(Vec3f(-mHalfWidth, -mHalfHeight, 0.f), time);
}

float
RectLight::planeDistance(const Vec3f &p, float time) const
{
    Plane pl = isMb() ? Plane(getPosition(time), getDirection(time)) : mRenderPlane;

    float distance = pl.getDistance(p);
    if (mSidedness == LightSidednessType::LIGHT_SIDEDNESS_REVERSE) {
        distance = -distance;
    } else if (mSidedness == LightSidednessType::LIGHT_SIDEDNESS_2_SIDED) {
        distance = scene_rdl2::math::abs(distance);
    }

    return distance;
}

bool
RectLight::getOverlapBounds(const Vec2f& localP, float localLength,
        Vec2f& center, float& width, float& height) const
{
    float minX = math::max(localP.x - localLength, -mHalfWidth);
    float maxX = math::min(localP.x + localLength, mHalfWidth);
    float minY = math::max(localP.y - localLength, -mHalfHeight);
    float maxY = math::min(localP.y + localLength, mHalfHeight);

    if (maxX <= minX || maxY <= minY) {
        // region does not overlap with RectLight
        return false;
    }

    center.x = 0.5f * (maxX + minX);
    center.y = 0.5f * (maxY + minY);
    width = maxX - minX;
    height = maxY - minY;

    return true;
}

RectLight::RectLight(const scene_rdl2::rdl2::Light* rdlLight, bool uniformSampling) :
    LocalParamLight(rdlLight)
{
    initAttributeKeys(rdlLight->getSceneClass());

    ispc::RectLight_init(this->asIspc(), uniformSampling);
}

RectLight::~RectLight() { }

bool
RectLight::updateTransforms(const Mat4f &local2Render, int ti)
{
    // update transforms at time index (ti)
    // ti == 0 is at normalized rayTime = 0.f
    // ti == 1 is at normalized rayTime = 1.f;
    MNRY_ASSERT(getRdlLight());

    Mat4f local2RenderCopy = local2Render;
    if (extractUniformScale(local2Render, &mLocal2RenderScale[ti])) {
        mRender2LocalScale[ti] = 1.0f / math::max(sEpsilon, mLocal2RenderScale[ti]);
        mWidth  = mRdlLight->get<scene_rdl2::rdl2::Float>(sWidthKey);
        mHeight = mRdlLight->get<scene_rdl2::rdl2::Float>(sHeightKey);
    } else {
        mRender2LocalScale[ti] = 1.0f;
        Vec3f nonUniformScale = extractNonUniformScale(local2Render);
        mWidth  = mRdlLight->get<scene_rdl2::rdl2::Float>(sWidthKey) * nonUniformScale.x;
        mHeight = mRdlLight->get<scene_rdl2::rdl2::Float>(sHeightKey) * nonUniformScale.y;
        local2RenderCopy = Mat4f(Vec4f(normalize(asVec3(local2Render.row0()))),
                                 Vec4f(normalize(asVec3(local2Render.row1()))),
                                 Vec4f(normalize(asVec3(local2Render.row2()))),
                                 local2Render.row3());
    }
    mHalfWidth  = mWidth  * 0.5f;
    mHalfHeight = mHeight * 0.5f;

    mLocal2Render[ti] = Xform3f(Mat3f(asVec3(local2RenderCopy.row0()),
                                      asVec3(local2RenderCopy.row1()),
                                      asVec3(local2RenderCopy.row2())),
                                      asVec3(local2RenderCopy.row3()));

    // WARNING: this can be inaccurate, may lead to float precision jitter
    mRender2Local[ti] = mLocal2Render[ti].inverse();

    Mat4f local2RenderRot = local2RenderCopy * Mat4f::scale(Vec4f(mRender2LocalScale[ti]));

    // WARNING: Includes translation, ok because only used with transformVector
    mLocal2RenderRot[ti] = Xform3f(Mat3f(asVec3(local2RenderRot.row0()),
                                         asVec3(local2RenderRot.row1()),
                                         asVec3(local2RenderRot.row2())),
                                         asVec3(local2RenderRot.row3()));

    // WARNING: Includes translation, ok because only used with transformVector
    // and also getXformRender2Local which uses the translation.
    mRender2LocalRot[ti] = mRender2Local[ti] * Xform3f::scale(Vec3f(mLocal2RenderScale[ti]));

    return true;
}

// TODO: motion blur. We might be better off transforming the incoming ray into
// the light frame begin time, rather than attempting to regenerate all the
// cached data for various frame times.
bool
RectLight::update(const Mat4d& world2render)
{
    MNRY_ASSERT(mRdlLight);

    mOn = mRdlLight->get(scene_rdl2::rdl2::Light::sOnKey);
    if (!mOn) {
        return false;
    }

    updateVisibilityFlags();
    updatePresenceShadows();
    updateRayTermination();
    updateTextureFilter();
    updateMaxShadowDistance();

    mWidth  = mRdlLight->get<scene_rdl2::rdl2::Float>(sWidthKey);
    mHeight = mRdlLight->get<scene_rdl2::rdl2::Float>(sHeightKey);
    mHalfWidth  = mWidth  * 0.5f;
    mHalfHeight = mHeight * 0.5f;

    // mSpread: 1 = 90 degree cone angle, 0 = 0 degree cone angle (cross section is square).
    mSpread = math::clamp(mRdlLight->get<scene_rdl2::rdl2::Float>(sSpreadKey), math::sEpsilon, 1.f);
    mTanSpreadTheta = math::tan(mSpread * math::sHalfPi);

    UPDATE_ATTRS_CLEAR_RADIUS

    const Mat4d l2w0 = mRdlLight->get(scene_rdl2::rdl2::Node::sNodeXformKey, /* rayTime = */ 0.f);
    const Mat4d l2w1 = mRdlLight->get(scene_rdl2::rdl2::Node::sNodeXformKey, /* rayTime = */ 1.f);

    const Mat4f local2Render0 = sRotateX180 * toFloat(l2w0 * world2render);
    const Mat4f local2Render1 = sRotateX180 * toFloat(l2w1 * world2render);

    if (!updateParamAndTransforms(local2Render0, local2Render1, mHalfWidth, mHalfHeight)) {
        return false;
    }

    // Compute render space corners of light.
    // Cached at rayTime = 0
    computeCorners(mRenderCorners, 0.f);

    // Compute render space quantities.
    const Vec3f xAxis = mRenderCorners[2] - mRenderCorners[0];
    const Vec3f yAxis = mRenderCorners[1] - mRenderCorners[0];
    mArea = xAxis.length() * yAxis.length();
    mInvArea = 1.0f / mArea;

    // Compute radiance.
    mRadiance = computeLightRadiance(mRdlLight, scene_rdl2::rdl2::Light::sColorKey,
        scene_rdl2::rdl2::Light::sIntensityKey, scene_rdl2::rdl2::Light::sExposureKey, sNormalizedKey,
        sApplySceneScaleKey, mInvArea);
    if (isBlack(mRadiance)) {
        mOn = false;
        return false;
    }

    // Compute render space plane of light.
    // Cached at rayTime = 0.f
    mRenderPlane = Plane(getPosition(0.f), getDirection(0.f));

    if (!updateImageMap(Distribution2D::PLANAR)) {
        return false;
    }

    // Sidedness
    mSidedness = static_cast<LightSidednessType>(mRdlLight->get<scene_rdl2::rdl2::Int>(sSidednessKey));

    return true;
}

//----------------------------------------------------------------------------

bool
RectLight::canIlluminate(const Vec3f p, const Vec3f *n, float time, float radius,
    const LightFilterList* lightFilterList) const
{
    MNRY_ASSERT(mOn);

    // Cull points which are completely on the backside of the light, taking sidedness into account
    const float threshold = sEpsilon - radius;
    if ((mSidedness != LightSidednessType::LIGHT_SIDEDNESS_2_SIDED) && (planeDistance(p, time) < threshold)) {
        return false;
    }

    // Cull lights which are completely on the backside of the point.
    if (n) {
        Plane pl(p, *n);
        float d0, d1, d2, d3;
        if (!isMb()) {
            d0 = pl.getDistance(mRenderCorners[0]);
            d1 = pl.getDistance(mRenderCorners[1]);
            d2 = pl.getDistance(mRenderCorners[2]);
            d3 = pl.getDistance(mRenderCorners[3]);
        } else {
            Vec3f corners[4];
            computeCorners(corners, time);
            d0 = pl.getDistance(corners[0]);
            d1 = pl.getDistance(corners[1]);
            d2 = pl.getDistance(corners[2]);
            d3 = pl.getDistance(corners[3]);
        }
        bool canIllum = d0 > threshold || d1 > threshold ||
                        d2 > threshold || d3 > threshold;
        if (!canIllum) return false;
    }

    if (lightFilterList) {
        return canIlluminateLightFilterList(lightFilterList,
            { getPosition(time),
              xformLocal2RenderScale(math::sqrt(mWidth * mWidth + mHeight * mHeight) / 2, time),
              p, getXformRender2Local(time, lightFilterList->needsLightXform()),
              radius, time
            });
    }

    return true;
}

bool
RectLight::isBounded() const
{
    return true;
}

bool
RectLight::isDistant() const
{
    return false;
}

bool
RectLight::isEnv() const
{
    return false;
}

// Computing a tight bounding box for a rect light
// -----------------------------------------------
//
// Let w and h be the half-width and half-height of the rectangle in local space,
// so that the 4 corners lie at (x,y,z) = (+/-w, +/-h, 0).
// If we denote the local2render matrix by (Mij) with i=0,1,2,3 and j=0,1,2,3,
// then the corners transform to the points (x',y',z') in render space, where
//
//      x' = +/-w * M00 +/-h * M10 + M30
//      y' = +/-w * M01 +/-h * M11 + M31
//      z' = +/-w * M02 +/-h * M12 + M32
//
// Considering just the x' values, the min and max are given respectively by
//
//      xMin' = M30 - (w * |M00| + h * |M10|)
//      xMax' = M30 + (w * |M00| + h * |M10|)
//
// Similar expression arise for the min & max y' and z' values, and the 3
// expressions can be combined into vector form using Vec3f.

BBox3f
RectLight::getBounds() const
{
    const Vec3f pos = getPosition(0.f);
    const Vec3f absXRow  = scene_rdl2::math::abs(mLocal2Render[0].l.vx);
    const Vec3f absYRow  = scene_rdl2::math::abs(mLocal2Render[0].l.vy);
    const Vec3f halfDims = max(mHalfWidth * absXRow + mHalfHeight * absYRow,
                               Vec3f(sEpsilon));     // ensure non-degeneracy
    BBox3f bounds(pos - halfDims, pos + halfDims);

    if (isMb()) {
        const Vec3f pos = getPosition(1.f);
        const Vec3f absXRow  = scene_rdl2::math::abs(mLocal2Render[1].l.vx);
        const Vec3f absYRow  = scene_rdl2::math::abs(mLocal2Render[1].l.vy);
        const Vec3f halfDims = max(mHalfWidth * absXRow + mHalfHeight * absYRow,
                                   Vec3f(sEpsilon));     // ensure non-degeneracy
        bounds.extend(BBox3f(pos - halfDims, pos + halfDims));
    }

    return bounds;
}

bool
RectLight::intersect(const Vec3f &p, const Vec3f *n, const Vec3f &wi, float time,
        float maxDistance, LightIntersection &isect) const
{
    // Get local direction and position
    Vec3f localWi = xformVectorRender2LocalRot(wi, time);
    Vec3f localP  = xformVectorRender2Local(p - getPosition(time), time);

    // Trivial rejection of pos/dir/sidedness cases
    if (rejectPosDirSidedness(localP.z, localWi.z)) return false;

    // Reject if intersection with plane is beyond maxDistance
    float localDistance = -localP.z / localWi.z;
    float renderDistance = xformLocal2RenderScale(localDistance, time);
    if (renderDistance > maxDistance) {
        return false;
    }

    // Reject if intersection is outside rectangle
    Vec3f localHit = localP + localWi * localDistance;
    if (scene_rdl2::math::abs(localHit.x) >= mHalfWidth || scene_rdl2::math::abs(localHit.y) >= mHalfHeight) {
        return false;
    }

    if (mSpread < 1.f) {
        // Reject if intersection point is outside spread
        Vec3f absLocalWi = scene_rdl2::math::abs(localWi);
        if (max(absLocalWi.x, absLocalWi.y) > absLocalWi.z * mTanSpreadTheta) {
            return false;
        }

        // compute pdf
        // get half side length of square spread region
        float spreadLength = localP.z * mTanSpreadTheta;
        Vec2f center;
        float width, height;
        getOverlapBounds(Vec2f(localP.x, localP.y), spreadLength, center, width, height);
        // store pdf if not rejected
        isect.pdf = 1.f / (xformLocal2RenderScale(width, time) * xformLocal2RenderScale(height, time));
    }

    // Fill in isect members.
    // We will only reach this far if p is on the illuminated side of the light's plane (either side, for 2-sided).
    // The light's mDirection is its local z-axis, which is only guaranteed to concide with the normal for lights
    // with regular sidedness. In the general case, the correct normal is obtained by ensuring the ray is travelling
    // against the normal, i.e. the ray's direction has negative local z-coordinate.
    Vec3f normal = getDirection(time);
    isect.N = (localWi.z < 0.0f) ? normal : -normal;
    isect.uv = (mDistribution  ?  asVec2(localHit) * mUvScale + mUvOffset
                                :  math::zero);
    isect.distance = renderDistance;

    return true;
}

bool
RectLight::sample(const Vec3f &p, const Vec3f *n, float time, const Vec3f& r,
        Vec3f &wi, LightIntersection &isect, float rayDirFootprint) const
{
    MNRY_ASSERT(mOn);

    // We don't cast light on points behind the light. Also we can't always
    // rely on canIlluminate() to test for that case, as it's an optional
    // culling function. canIlluminate() can also be passed a non-zero radius,
    // defining a p-region partially in front and partially behind the light.
    // So we have to test for it per sample, here too.
    float distance = planeDistance(p, time);
    if ((mSidedness != LightSidednessType::LIGHT_SIDEDNESS_2_SIDED) && (distance < sEpsilon)) {
        return false;
    }

    Vec3f localHit;

    // There are 3 possible sampling modes:
    // 1) If there is an image distribution, sample that.
    // 2) If there is no image distribution, and the spread parameter is less than 1 and
    //    we are in a good region to sample the spread, then sample that region.
    // 3) Otherwise, sample the shape of the light.
    if (mDistribution) {
        // sample the image distribution if any
        mDistribution->sample(r[0], r[1], 0, &isect.uv, nullptr, mTextureFilter);
        localHit = Vec3f((isect.uv[0] - mUvOffset.x) / mUvScale.x,
                         (isect.uv[1] - mUvOffset.y) / mUvScale.y,
                          0.0f);

        if (mSpread < 1.f) {
            // Reject samples that are outside of the spread region.
            Vec3f localP = xformPointRender2Local(p, time);
            Vec3f localWi = localHit - localP;
            Vec3f absLocalWi = scene_rdl2::math::abs(localWi);
            if (max(absLocalWi.x, absLocalWi.y) > absLocalWi.z * mTanSpreadTheta) {
                return false;
            }
        }

    } else if (mSpread < 1.f) {
        // Uniformly sample the rectangular region of overlap between the spread region and the RectLight.

        // get p in local space
        Vec3f localP = xformPointRender2Local(p, time);

        // get length of square region of influence
        float spreadLength = distance * mTanSpreadTheta;
        float localSpreadLength = xformRender2LocalScale(spreadLength, time);

        Vec2f center;
        float width, height;
        if (!getOverlapBounds(Vec2f(localP.x, localP.y), localSpreadLength, center, width, height)) {
            return false;
        }

        // sample rectangular region of influence
        localHit = Vec3f((0.5f - r[0]) * width + center.x,
                         (0.5f - r[1]) * height + center.y,
                          0.0f);

        isect.uv = zero;
        // store pdf
        isect.pdf = 1.f / (xformLocal2RenderScale(width, time) * xformLocal2RenderScale(height, time));

    } else {
        // Otherwise uniformly distribute samples on rectangle area
        isect.uv = zero;
        localHit = Vec3f((0.5f - r[0]) * mWidth,
                         (0.5f - r[1]) * mHeight,
                          0.0f);
    }

    Vec3f renderHit = xformPointLocal2Render(localHit, time);

    // Compute wi and d
    wi = renderHit - p;
    isect.distance = length(wi);
    if (isect.distance < sEpsilon) {
        return false;
    }
    wi *= rcp(isect.distance);
    if (n  &&  dot(*n, wi) < sEpsilon) {
        return false;
    }

    // We will only reach this far if p is on the illuminated side of the light's plane (either side, for 2-sided).
    // The light's mDirection is its local z-axis, which is only guaranteed to concide with the normal for lights
    // with regular sidedness. The correct normal is obtained by ensuring the ray is travelling against the normal,
    // i.e. the ray's direction has negative dot product with it.
    Vec3f normal = getDirection(time);
    isect.N = (dot(normal, wi) < 0.0f) ? normal : -normal;

    return true;
}

Color
RectLight::eval(mcrt_common::ThreadLocalState* tls, const Vec3f &wi, const Vec3f &p, const LightFilterRandomValues& filterR, float time,
        const LightIntersection &isect, bool fromCamera, const LightFilterList *lightFilterList, float rayDirFootprint,
        float *pdf) const
{
    MNRY_ASSERT(mOn);

    // Point sample the texture
    Color radiance = mRadiance;

    if (mSpread < 1.f) {
        // modify radiance as if it were physically blocked by an eggcrate light filter.
        Vec3f localWi = xformVectorRender2LocalRot(wi, time);
        radiance *= math::max(1.f - math::abs(localWi.x / (localWi.z * mTanSpreadTheta)), 0.f) *
                    math::max(1.f - math::abs(localWi.y / (localWi.z * mTanSpreadTheta)), 0.f);
    }

    if (mDistribution) {
        radiance *= mDistribution->eval(isect.uv[0], isect.uv[1], 0, mTextureFilter);
    }

    if (lightFilterList) {
        evalLightFilterList(lightFilterList, 
                            { tls, &isect, getPosition(time),
                              getDirection(time), p,
                              filterR, time,
                              getXformRender2Local(time, lightFilterList->needsLightXform()),
                              wi
                            },
                            radiance);
    }

    if (pdf) {
        if (mDistribution) {
            // we sampled the image distribution
            *pdf = mInvArea * mDistribution->pdf(isect.uv[0], isect.uv[1], 0, mTextureFilter);
        } else if (mSpread < 1.f) {
            // we sampled the spread region, grab the stored pdf.
            *pdf = isect.pdf;
        } else {
            // we uniformly sampled the rectangle
            *pdf = mInvArea;
        }
        *pdf *= areaToSolidAngleScale(wi, isect.N, isect.distance);
    }

    return radiance;
}

Vec3f
RectLight::getEquiAngularPivot(const Vec3f& r, float time) const
{
    Vec2f uv;
    Vec3f hit;
    // Sample the image distribution if any
    if (mDistribution) {
        mDistribution->sample(r[0], r[1], 0, &uv, nullptr, mTextureFilter);
        hit = Vec3f((uv[0] - mUvOffset.x) / mUvScale.x,
                    (uv[1] - mUvOffset.y) / mUvScale.y,
                     0.0f);
    } else {
        hit = Vec3f((0.5f - r[0]) * mWidth,
                    (0.5f - r[1]) * mHeight,
                     0.0f);
    }
    return xformPointLocal2Render(hit, time);
}

void
RectLight::initAttributeKeys(const scene_rdl2::rdl2::SceneClass &sc)
{
    if (sAttributeKeyInitialized) {
        return;
    }

    MOONRAY_START_NON_THREADSAFE_STATIC_WRITE

    sAttributeKeyInitialized = true;

    sNormalizedKey      = sc.getAttributeKey<scene_rdl2::rdl2::Bool> ("normalized");
    sApplySceneScaleKey = sc.getAttributeKey<scene_rdl2::rdl2::Bool> ("apply_scene_scale");
    sWidthKey           = sc.getAttributeKey<scene_rdl2::rdl2::Float>("width");
    sHeightKey          = sc.getAttributeKey<scene_rdl2::rdl2::Float>("height");
    sSpreadKey          = sc.getAttributeKey<scene_rdl2::rdl2::Float>("spread");
    sSidednessKey       = sc.getAttributeKey<scene_rdl2::rdl2::Int>  ("sidedness");
    INIT_ATTR_KEYS_CLEAR_RADIUS

    MOONRAY_FINISH_NON_THREADSAFE_STATIC_WRITE
}


//----------------------------------------------------------------------------

} // namespace pbr
} // namespace moonray

