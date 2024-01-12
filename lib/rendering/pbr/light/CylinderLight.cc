// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0


#include "CylinderLight.h"
#include <moonray/rendering/pbr/core/Distribution.h>
#include <moonray/rendering/pbr/core/Util.h>

#include <moonray/rendering/pbr/light/CylinderLight_ispc_stubs.h>

#include <scene_rdl2/scene/rdl2/rdl2.h>
#include <moonray/common/mcrt_macros/moonray_static_check.h>

using namespace scene_rdl2;
using namespace scene_rdl2::math;
using scene_rdl2::logging::Logger;

namespace moonray {
namespace pbr {


bool                                                     CylinderLight::sAttributeKeyInitialized;
scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Bool>   CylinderLight::sNormalizedKey;
scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Bool>   CylinderLight::sApplySceneScaleKey;
scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Float>  CylinderLight::sRadiusKey;
scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Float>  CylinderLight::sHeightKey;
scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Int>    CylinderLight::sSidednessKey;
scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Float>  CylinderLight::sClearRadiusKey;
scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Float>  CylinderLight::sClearRadiusFalloffDistanceKey;
scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Int>    CylinderLight::sClearRadiusInterpolationKey;

//----------------------------------------------------------------------------

HUD_VALIDATOR(CylinderLight);

CylinderLight::CylinderLight(const scene_rdl2::rdl2::Light* rdlLight) :
    LocalParamLight(rdlLight)
{
    initAttributeKeys(rdlLight->getSceneClass());

    ispc::CylinderLight_init(this->asIspc());
}

CylinderLight::~CylinderLight() { }

// TODO: motion blur. We might be better off transforming the incoming ray into
// the light frame begin time, rather than attempting to regenerate all the
// cached data for various frame times.
bool
CylinderLight::update(const Mat4d& world2render)
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

    const Mat4d l2w0 = mRdlLight->get(scene_rdl2::rdl2::Node::sNodeXformKey, /* rayTime = */ 0.f);
    const Mat4d l2w1 = mRdlLight->get(scene_rdl2::rdl2::Node::sNodeXformKey, /* rayTime = */ 1.f);
    mLocalRadius     = mRdlLight->get<scene_rdl2::rdl2::Float>(sRadiusKey);
    mLocalHalfHeight = mRdlLight->get<scene_rdl2::rdl2::Float>(sHeightKey) * 0.5f;

    UPDATE_ATTRS_CLEAR_RADIUS

    const Mat4f local2Render0 = toFloat(l2w0 * world2render);
    const Mat4f local2Render1 = toFloat(l2w1 * world2render);
    if (!updateParamAndTransforms(local2Render0, local2Render1, mLocalHalfHeight, 1.0f)) {
        return false;
    }

    // Compute render space quantities
    // TODO: mb scale radiance
    mActualRadius        = xformLocal2RenderScale(mLocalRadius, 0.f);
    mRcpActualRadius     = 1.0f / mActualRadius;
    mActualRadiusSquared = mActualRadius * mActualRadius;
    mActualHalfHeight    = xformLocal2RenderScale(mLocalHalfHeight, 0.f);
    mArea                = sFourPi * mActualRadius * mActualHalfHeight;
    mInvArea             = 1.0f / mArea;

    // Compute radiance.
    mRadiance = computeLightRadiance(mRdlLight, scene_rdl2::rdl2::Light::sColorKey,
        scene_rdl2::rdl2::Light::sIntensityKey, scene_rdl2::rdl2::Light::sExposureKey, sNormalizedKey,
        sApplySceneScaleKey, mInvArea);
    if (isBlack(mRadiance)) {
        mOn = false;
        return false;
    }

    if (!updateImageMap(Distribution2D::PLANAR)) {
        return false;
    }

    // Sidedness
    mSidedness = static_cast<LightSidednessType>(mRdlLight->get<rdl2::Int>(sSidednessKey));
    return true;
}

//----------------------------------------------------------------------------

bool
CylinderLight::canIlluminate(const scene_rdl2::math::Vec3f p, const scene_rdl2::math::Vec3f *n, float time, float radius,
    const LightFilterList* lightFilterList) const
{
    MNRY_ASSERT(mOn);

    if (lightFilterList) {
        float lightRadius = scene_rdl2::math::max(mLocalHalfHeight * 2, mLocalRadius);
        return canIlluminateLightFilterList(lightFilterList,
            { getPosition(time),
              xformLocal2RenderScale(lightRadius, time),
              p, getXformRender2Local(time, lightFilterList->needsLightXform()),
              radius, time
            });
    }

    return true;
}

//----------------------------------------------------------------------------

bool
CylinderLight::isBounded() const
{
    return true;
}

bool
CylinderLight::isDistant() const
{
    return false;
}

bool
CylinderLight::isEnv() const
{
    return false;
}

// Computing a tight bounding box for a cylinder light
// ---------------------------------------------------
//
// See the comments for DiskLight::getBounds() for explanation of tight bounds for a disk.
// Bounding the cross-section of a cylinder works in the same way, with a couple of minor
// modifications - the axis of rotational symmetry is the local y-axis for a cylinder, where
// it's the local z-axis for a disk; and also a non-unit mLocalRadius is supported for a
// cylinder, where a disk light has radius 1 in local space.
//
// The method for a disk is augmented by sweeping the disk along the local y-axis so that
// it covers the interval [-mLocalHalfHeight, mLocalHalfHeight] in the local y-direction.
// In order to bound the resulting volume in render space, to the half-dimesions for disk
// bounds we add the vector h*(M10,M11,M12) where h is the half-height and (Mij) represents
// the local2render matrix.

BBox3f
CylinderLight::getBounds() const
{
    const float r = mLocalRadius;
    const float h = mLocalHalfHeight;
    const Vec3f pos = getPosition(0.f);
    const Vec3f xRow = mLocal2Render[0].l.vx;
    const Vec3f yRow = mLocal2Render[0].l.vy;
    const Vec3f zRow = mLocal2Render[0].l.vz;
    const Vec3f halfDims = r * scene_rdl2::math::sqrt(xRow*xRow + zRow*zRow) + h * scene_rdl2::math::abs(yRow);
    BBox3f bounds(pos - halfDims, pos + halfDims);
    if (isMb()) {
        const Vec3f pos = getPosition(1.f);
        const Vec3f xRow = mLocal2Render[1].l.vx;
        const Vec3f yRow = mLocal2Render[1].l.vy;
        const Vec3f zRow = mLocal2Render[1].l.vz;
        const Vec3f halfDims = r * scene_rdl2::math::sqrt(xRow*xRow + zRow*zRow) + h * scene_rdl2::math::abs(yRow);
        bounds.extend(BBox3f(pos - halfDims, pos + halfDims));
    }
    return bounds;
}


// Ray/cylinder intersection test
// ------------------------------
//
// Define a ray in the usual way:
// p  = origin
// wi = direction
//
// The ray's parametric equation is x = p + t*wi
//
// Define a cylinder as follows:
// x0 = centre (i.e. midpoint along axis between the 2 end planes)
// e  = unit vector along axis (in either direction)
// r  = radius
// h  = half-height, measured along axis from centre to either end plane
//
// The implicit equation of the cylinder's quadric surface is
//
//   |cross(x-x0, e)|^2 = r^2
//
// Substituting the ray's parametric equation, we obtain a quadratic in t:
//
//   |cross(t*wi - dp, e)|^2 = r^2
//
// where dp = x0 - p.
// Now define vectors u = wi x e and v = dp x e. The quadratic in t
// expands as follows:
//
//   (u.u)t^2 - 2(u.v)t + (v.v)-r^2 = 0
//
// Or
// 
//   a*t^2 - 2b*t + c = 0
// 
// with a=u.u, b=u.v, c=v.v-r^2, the discriminant is
//
//   D = b^2-a*c
//
// We can reduce the work required to compute D as follows:
//
//   D = (u.v)^2 - (u.u)((v.v)-r^2)
//     = (u.v)^2 - (u.u)(v.v) + (u.u)r^2
//     = (u.u)r^2 - |u x v|^2
//
// Noting that u.u = a, and also that
//
//   u x v = (wi x e) x (dp x e)
//         = (wi . (dp x e)) e - (e . (dp x e)) wi
//         = - ((wi x e).dp) e
//         = - (u.dp) e
//
// we have
//
//   D = a r^2 - (u.dp)^2
//
// We reject rays for which D is negative. Otherwise we proceed to solve the
// quadratic equation in the usual way:
//
//   t = (b +/- sqrt(D)) / a
//
// where we take the smaller root for regular cylinder lights and the larger for ones with
// reverse sidedness. The computed intersection point can then be tested for inclusion between
// the 2 end planes.
//
// Considering the cylinder as the intersection of an infinite cylinder with a slab defined
// by the end planes, trivial rejection can be done in a couple of easy ways.
//  
// Trivial rejection based on the ray's configuration relative to the slab works by rejecting
// when the origin is outside the slab and the direction is pointing away from it.
// i.e reject if (wi.e>0 && dp.e>h) || (wi.e<0 && dp.e<-h)
//
// Trivial rejection based on considering the infinite cylinder is similar to the rejection
// done for SphereLight. It's based on 2 conditions:
// - whether the ray's origin is inside the cylinder;
// - whether the ray's direction faces away from the cylinder (strictly, from the plane through
// the cylinder's axis perpendicular to the line joining the ray's origin to the cylinder axis).
// These 2 conditions are combined differently depending on the sidedness of the light.
//
// The algebra of the rejection tests is simplified by defining the following:
// A = dp.e, B = wi.e, C = dp.wi
// 
// These are also used to simplify computing the quadratic coefficients a,b,c:
// a = u^2 = (wi x e)^2 = 1 - (wi.e)^2 = 1 - B^2
// b = u.v = (wi x e) . (dp x e) = [wi x e, dp, e] = ((wi x e) x dp) . e = ((wi.dp)e - (e.dp)wi) . e
//   = wi.dp - (e.dp)(wi.e) = C - AB
// c = v^2 - r^2 = dp^2 e^2 - (dp.e)^2 - r^2 = dp^2 - A^2 - r^2
//
// We can use the following conditions to trivially reject certain cases, depending on the light's sidedness:
//   b<0, ray pointing away from cylinder axis (strictly, away from a plane through the axis perpendicular to the
//        plane containing both the axis and the illumination point)
//   c<0, ray origin inside (infinite) cylinder
//   D<0, no real roots, i.e. ray misses the (infinite) cylinder

bool
CylinderLight::intersect(const Vec3f &p, const Vec3f *n, const Vec3f &wi, float time,
        float maxDistance, LightIntersection &isect) const
{
    const Vec3f e  = xformVectorLocal2RenderRot(Vec3f(0.f, 1.f, 0.f), time);
    const Vec3f dp = getPosition(time) - p;

    // Trivial rejection if ray is outside slab and travelling away from it
    const float A = dot(dp, e);
    const float B = dot(wi, e);
    if (A * sign(B) <= -mActualHalfHeight) {
        return false;
    }

    const float C = dot(dp, wi);
    const float b = C - A*B;
    const float c = lengthSqr(dp) - A*A - mActualRadiusSquared;

    // Do trivial rejection for the following cases:
    // regular sidedness: reject if ray is traveling away from axis OR  ray origin is INSIDE cylinder
    // reverse sidedness: reject if ray is traveling away from axis AND ray origin is OUTSIDE cylinder
    // (two-sided is not supported)
    // c > 0  when ray origin is outside cylinder
    // b <= 0 when ray is travelling away from cylinder axis
    if ((c > 0.0f) ? (b <= 0.0f) : (mSidedness == LightSidednessType::LIGHT_SIDEDNESS_REGULAR)) {
        return false;
    }

    // Compute discriminant; reject unless positive
    float a = 1.0f - B*B;
    const float D = b * b - a * c;
    if (D <= 0.0f) {
        return false;
    }

    // Compute intersection t-value along ray
    // Note: a=0 is rejected simply to protect against generating a NaN. It's perfectly fine for 'a' to be very
    // small and overflow the result to +/-inf becuse the t range test will reject it.
    const float s = scene_rdl2::math::sqrt(D);
    const float t = (mSidedness == LightSidednessType::LIGHT_SIDEDNESS_REGULAR) ? (b - s) / a : (b + s) / a;
    if (a == 0.0f || t < 0.0f || t > maxDistance) {
        return false;
    }

    // Compute intersection point relative to cylinder centre; check it's between end planes
    const float y = t * B - A;
    if (scene_rdl2::math::abs(y) > mActualHalfHeight) {
        return false;
    }

    // Fill in isect members
    const Vec3f hitPoint = t * wi - dp;
    // We don't need to worry about 2-sided here because that's not supported for CylinderLight
    isect.N = ((mSidedness == LightSidednessType::LIGHT_SIDEDNESS_REGULAR) ? mRcpActualRadius : -mRcpActualRadius)
                * (hitPoint - y * e);
    isect.uv = mDistribution ? local2uv(xformVectorRender2Local(hitPoint, time)) : scene_rdl2::math::zero;
    isect.distance = t;

    return true;
}


bool
CylinderLight::sample(const Vec3f &p, const Vec3f *n, float time, const Vec3f& r,
        Vec3f &wi, LightIntersection &isect, float rayDirFootprint) const
{
    MNRY_ASSERT(mOn);

    // Sample the image distribution if any, otherwise sample the cylinder
    // uv parameterization directly.
    Vec2f uv;
    if (mDistribution) {
        mDistribution->sample(r[0], r[1], 0, &uv, nullptr, mTextureFilter);
        isect.uv = uv;
    } else {
        uv = Vec2f(r[0], r[1]);
        isect.uv = Vec2f(zero);
    }

    const Vec3f localHit  = uv2local(uv);
    const Vec3f renderHit = xformPointLocal2Render(localHit, time);

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

    const Vec3f localN  = Vec3f(localHit.x, 0.0f, localHit.z);
    const Vec3f renderN = normalize(xformNormalLocal2Render(localN, time));
    // We don't need to worry about 2-sided here because that's not supported for CylinderLight
    isect.N = (mSidedness == LightSidednessType::LIGHT_SIDEDNESS_REGULAR) ? renderN : -renderN;

    // Reject if the generated point is on the back-facing surface of the cylinder
    if (dot(isect.N, wi) > -sEpsilon) {
        return false;
    }

    return true;
}


Color
CylinderLight::eval(mcrt_common::ThreadLocalState* tls, const Vec3f &wi, const Vec3f &p, const LightFilterRandomValues& filterR,
        float time, const LightIntersection &isect, bool fromCamera, const LightFilterList *lightFilterList,
        float rayDirFootprint, float *pdf) const
{
    MNRY_ASSERT(mOn);

    // Point sample the texture
    // TODO: Use proper filtering with ray differentials and mip-mapping.
    Color radiance = mRadiance;

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
        *pdf = mInvArea;
        if (mDistribution) {
            *pdf *= mDistribution->pdf(isect.uv[0], isect.uv[1], 0, mTextureFilter);
        }
        *pdf *= areaToSolidAngleScale(wi, isect.N, isect.distance);
    }

    return radiance;
}

Vec3f
CylinderLight::getEquiAngularPivot(const Vec3f& r, float time) const
{
    // Sample the image distribution if any, otherwise sample the cylinder
    // uv parameterization directly.
    Vec2f uv;
    if (mDistribution) {
        mDistribution->sample(r[0], r[1], 0, &uv, nullptr, mTextureFilter);
    } else {
        uv = Vec2f(r[0], r[1]);
    }
    const Vec3f localHit = uv2local(uv);
    return xformPointLocal2Render(localHit, time);
}

void
CylinderLight::initAttributeKeys(const scene_rdl2::rdl2::SceneClass &sc)
{
    if (sAttributeKeyInitialized) {
        return;
    }

    MOONRAY_START_NON_THREADSAFE_STATIC_WRITE

    sAttributeKeyInitialized = true;

    sNormalizedKey      = sc.getAttributeKey<scene_rdl2::rdl2::Bool> ("normalized");
    sApplySceneScaleKey = sc.getAttributeKey<scene_rdl2::rdl2::Bool> ("apply_scene_scale");
    sRadiusKey          = sc.getAttributeKey<scene_rdl2::rdl2::Float>("radius");
    sHeightKey          = sc.getAttributeKey<scene_rdl2::rdl2::Float>("height");
    sSidednessKey       = sc.getAttributeKey<scene_rdl2::rdl2::Int>  ("sidedness");
    INIT_ATTR_KEYS_CLEAR_RADIUS

    MOONRAY_FINISH_NON_THREADSAFE_STATIC_WRITE
}


//----------------------------------------------------------------------------

} // namespace pbr
} // namespace moonray

