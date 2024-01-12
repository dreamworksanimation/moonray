// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0


#include "SphereLight.h"
#include <moonray/rendering/pbr/core/Distribution.h>
#include <moonray/rendering/pbr/core/Util.h>

#include <moonray/rendering/pbr/light/SphereLight_ispc_stubs.h>

#include <moonray/common/mcrt_macros/moonray_static_check.h>
#include <scene_rdl2/common/math/ReferenceFrame.h>
#include <scene_rdl2/scene/rdl2/rdl2.h>

namespace moonray {
namespace pbr {


using namespace scene_rdl2;
using namespace scene_rdl2::math;
using scene_rdl2::logging::Logger;


bool                             SphereLight::sAttributeKeyInitialized;
scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Bool>   SphereLight::sNormalizedKey;
scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Bool>   SphereLight::sApplySceneScaleKey;
scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Float>  SphereLight::sRadiusKey;
scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Float>  SphereLight::sClearRadiusKey;
scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Float>  SphereLight::sClearRadiusFalloffDistanceKey;
scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Int>    SphereLight::sClearRadiusInterpolationKey;
scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Int>    SphereLight::sSidednessKey;

//----------------------------------------------------------------------------

namespace {

finline Vec2f
local2uv(const Vec3f &local)
{
    MNRY_ASSERT(isNormalized(local));

    Vec2f uv;
    if (scene_rdl2::math::abs(local.y) >= 1.0f) {
        // Handle singularities at poles, to keep function consistent between
        // sample and intersect
        uv.x = 0.0f;
        uv.y = (local.y > 0.0f) ? 0.0f : 1.0f;
    } else {
        // Convert to spherical polar coords with y up.
        // Note that phi is intentionally off by pi, so that adding 0.5 to
        // uv.x brings it into the range [0,1] without using a conditional.
        float theta = scene_rdl2::math::acos(local.y);
        float phi = scene_rdl2::math::atan2(-local.z, -local.x);

        // Generate (u,v)
        uv.x = phi * sOneOverTwoPi + 0.5f;
        uv.y = theta * sOneOverPi;
    }

    return uv;
}

finline Vec3f
vectorFromPolarSinCosVersion(float sinTheta, float cosTheta, float phi)
{
    float sinPhi, cosPhi;
    sincos(phi, &sinPhi, &cosPhi);
    return Vec3f(cosPhi * sinTheta, sinPhi * sinTheta, cosTheta);
}

finline Vec3f
vectorFromPolarCosVersion(float cosTheta, float phi)
{
    float sinTheta = scene_rdl2::math::sqrt(1.0f - cosTheta * cosTheta);
    return vectorFromPolarSinCosVersion(sinTheta, cosTheta, phi);
}

}   // end of anon namespace

//----------------------------------------------------------------------------

HUD_VALIDATOR(SphereLight);

SphereLight::SphereLight(const scene_rdl2::rdl2::Light* rdlLight, bool uniformSampling) :
    LocalParamLight(rdlLight)
{
    initAttributeKeys(rdlLight->getSceneClass());

    ispc::SphereLight_init(this->asIspc(), uniformSampling);
}

SphereLight::~SphereLight() { }

bool
SphereLight::update(const Mat4d& world2render)
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

    // Radius gets baked into the the mLocal2Render and other matrices such
    // that the sphere radius is always 1 in local light space.
    const float radius = mRdlLight->get<scene_rdl2::rdl2::Float>(sRadiusKey);
    
    UPDATE_ATTRS_CLEAR_RADIUS

    const Mat4d l2w0 = mRdlLight->get(scene_rdl2::rdl2::Node::sNodeXformKey, /* rayTime = */ 0.f);
    const Mat4d l2w1 = mRdlLight->get(scene_rdl2::rdl2::Node::sNodeXformKey, /* rayTime = */ 1.f);
    Mat4f sclRot = sRotateX180;
    sclRot[0][0] *= radius;
    sclRot[1][1] *= radius;
    sclRot[2][2] *= radius;
    const Mat4f local2Render0 = sclRot * toFloat(l2w0 * world2render);
    const Mat4f local2Render1 = sclRot * toFloat(l2w1 * world2render);
    if (!updateParamAndTransforms(local2Render0, local2Render1, 1.0f, 1.0f)) {
        return false;
    }

    // Compute render space quantities.
    // TODO: mb scale radiance
    mRadius = getRadius();
    mRadiusSqr = mRadius * mRadius;
    mRcpRadius = 1.0f / mRadius;
    mArea = sFourPi * mRadiusSqr;
    mInvArea = 1.0f / mArea;

    // Compute radiance.
    mRadiance = computeLightRadiance(mRdlLight, scene_rdl2::rdl2::Light::sColorKey,
        scene_rdl2::rdl2::Light::sIntensityKey, scene_rdl2::rdl2::Light::sExposureKey, sNormalizedKey,
        sApplySceneScaleKey, mInvArea);
    if (isBlack(mRadiance)) {
        mOn = false;
        return false;
    }

    if (!updateImageMap(Distribution2D::SPHERICAL)) {
        return false;
    }

    // Sidedness
    mSidedness = static_cast<LightSidednessType>(mRdlLight->get<scene_rdl2::rdl2::Int>(sSidednessKey));   
    return true;
}

//----------------------------------------------------------------------------

bool
SphereLight::canIlluminate(const Vec3f p, const Vec3f *n, float time, float radius,
    const LightFilterList* lightFilterList) const
{
    MNRY_ASSERT(mOn);

    // Reject if sphere is completely on the backside of the point's plane
    if (n) {
        Plane pl(p, *n);
        const float planeDist = pl.getDistance(getPosition(time));
        // mlee: we are losing a small amount of energy here by adding small value.
        const float planeThreshold = sEpsilon - mRadius - radius;
        if (planeDist < planeThreshold) {
            return false;
        }
    }

    // Reject when completely inside an outward-facing sphere light
    if (mSidedness == LightSidednessType::LIGHT_SIDEDNESS_REGULAR) {
        // mlee: we are losing a small amount of energy here by adding small value.
        const float sphereThreshold = mRadius + sEpsilon - radius;
        const Vec3f dp = p - getPosition(time);
        if (dp.lengthSqr() < sqr(sphereThreshold)) {
            return false;
        }
    }

    if (lightFilterList) {
        return canIlluminateLightFilterList(lightFilterList,
            { getPosition(time), mRadius, p,
              getXformRender2Local(time, lightFilterList->needsLightXform()),
              radius, time
            });
    }

    return true;
}

bool
SphereLight::isBounded() const
{
    return true;
}


bool
SphereLight::isDistant() const
{
    return false;
}


bool
SphereLight::isEnv() const
{
    return false;
}


BBox3f
SphereLight::getBounds() const
{
    const Vec3f pos = getPosition(0.f);
    const float rad = mRadius;
    BBox3f bounds(pos - Vec3f(rad), pos + Vec3f(rad));

    if (isMb()) {
        const Vec3f pos1 = getPosition(1.f);
        bounds.extend(BBox3f(pos1 - Vec3f(rad), pos1 + Vec3f(rad)));
    }

    return bounds;
}


// Ray equation:    x = p + t * wi
// Sphere equation: (x-x0)^2 = r^2, with x0 = sphere centre
// Intersection:    t^2 - 2wi.v*t + v^2-r^2 = 0, with v = x0-p and assuming |wi|=1
// This is a quadratic of the form a*t^2 - 2b*t + c = 0, with a=1, b=2wi.v, c=v^2-r^2
// The discriminant D = b^2-c
// For better precision, we use an algebraically equivalent form for the descriminant, D = r^2 - (v - b*wi)^2
// We can use the following conditions to trivially reject certain cases, depending on the light's sidedness:
//   b<0, ray pointing away from sphere center
//   c<0, ray origin inside sphere
//   D<0, no real roots, i.e. ray misses the sphere

bool
SphereLight::intersect(const Vec3f &p, const Vec3f *n,  const Vec3f &wi, float time,
        float maxDistance, LightIntersection &isect) const
{
    const Vec3f v = getPosition(time) - p;
    const float b = dot(wi, v);
    const float v2 = dot(v, v);
    const float c = v2 - mRadiusSqr;

    // Do trivial rejection based on these values, according to sphere's sidedness...
    // regular sidedness: reject if ray is travelling away from sphere center OR  ray origin is INSIDE sphere
    // reverse sidedness: reject if ray is travelling away from sphere center AND ray origin is OUTSIDE sphere
    // c > 0  when ray origin is outside sphere
    // b <= 0 when ray is travelling away from sphere center
    if ((c > 0.0f) ? (b <= 0.0f) : (mSidedness == LightSidednessType::LIGHT_SIDEDNESS_REGULAR)) {
        return false;
    }

    // Reject if discriminant is negative (ray misses sphere)
    const float D = mRadiusSqr - lengthSqr(v - b * wi); // More accurate version of discriminant than b^2-c
    if (D <= 0.0f) {
        return false;
    }

    // Calculate distance to intersection, reject if too big
    const float s = scene_rdl2::math::sqrt(D);
    // Choose near or far intersection depending on sidedness
    const float t = (mSidedness == LightSidednessType::LIGHT_SIDEDNESS_REGULAR) ? b - s : b + s;
    if (t > maxDistance) {
        return false;
    }

    // Fill in isect members
    const Vec3f normal = (t*wi - v) * mRcpRadius;   // Outward-facing normal
    // We don't need to worry about 2-sided here because that's not supported for SphereLight
    isect.N = (mSidedness == LightSidednessType::LIGHT_SIDEDNESS_REGULAR) ? normal : -normal;
    isect.distance = t;

    // Optionally compute uvs for texturing
    isect.uv = mDistribution ? local2uv(xformVectorRender2LocalRot(normal, time)) : zero;

    // Store values which eval() will later use to compute the pdf.
    // See sample() function for details about each case.
    if (c < 0.0f) {
        // Due to earlier rejection, this code only executes for reverse sidedness
        isect.data[1] = -1.0f;  // Signal uniform sampling over the solid angle
        if (n) {
            isect.data[0] = sOneOverTwoPi;
        } else {
            isect.data[0] = sOneOverFourPi;
        }
    } else {
        // This is the nastiest part of all! We need to match the pdf of the sampling scheme used in the sample()
        // function, where the samples were generated on the front-facing visible cap of the sphere and then
        // projected along the ray to the back surface in the case of reverse sidedness. So here, in the reverse
        // sidedness case, we must generate the values of t, the normal vector, and the trig expressions as if
        // the intersection point were on the front surface of the sphere.
        const float sinThetaMaxSq = c / v2;
        float cosTheta;
        if (mSidedness == LightSidednessType::LIGHT_SIDEDNESS_REVERSE) {
            const float t_front = b - s;
            const Vec3f normal_front = (t_front * wi - v) * mRcpRadius;
            cosTheta = -dot(v, normal_front) / scene_rdl2::math::sqrt(v2);
            isect.data[1] = t_front;
        } else {
            cosTheta = -dot(v, normal) / scene_rdl2::math::sqrt(v2);
            isect.data[1] = t;
        }
        isect.data[0] = cosTheta / sinThetaMaxSq;
    }

    return true;
}


bool
SphereLight::sample(const Vec3f &p, const Vec3f *n, float time, const Vec3f& r,
                    Vec3f &wi, LightIntersection &isect, float rayDirFootprint) const
{
    MNRY_ASSERT(mOn);

    const Vec3f v = getPosition(time) - p;
    const float v2 = dot(v, v);
    const float c = v2 - mRadiusSqr;

    float t;
    Vec3f normal;

    // Sampling strategy is different for inside vs outside the sphere
    if (c < 0.0f) {

        // Inside sphere
        if (mSidedness == LightSidednessType::LIGHT_SIDEDNESS_REGULAR) {
            // Trivial rejection when inside an outward-facing sphere light
            return false;
        }

        // Signal uniform sampling over solid angle
        isect.data[1] = -1.0f;

        if (n) {
            // Use surface normal to sample only the visible hemisphere
            Vec3f localWi = vectorFromPolarCosVersion(r[0], sTwoPi * r[1]);
            math::ReferenceFrame frame(*n);
            wi = frame.localToGlobal(localWi);
            isect.data[0] = sOneOverTwoPi;     // pdf = 1 / (solid angle of hemisphere)
        } else {
            // No normal supplied; sample the entire sphere
            wi = vectorFromPolarCosVersion(2.0f * r[0] - 1.0f, sTwoPi * r[1]);
            isect.data[0] = sOneOverFourPi;    // pdf = 1 / (solid angle of full sphere)
        }

        // See comments for intersect()
        const float b = dot(wi, v);
        const float D = b*b - c;    // Precision doesn't need to be improved here because radius is large compared to v
        const float s = scene_rdl2::math::sqrt(D);
        t = b + s;                  // Pick far intersection because we're inside the sphere
        normal = (t*wi - v) * mRcpRadius;  // Outward-facing normal

    } else {

        // See Dutre's "Global Illumination Compendium" section 35, page 20.
        const float sinThetaMaxSq = c / v2;
        const float sinThetaSq = r[0] * sinThetaMaxSq;
        const float sinTheta = scene_rdl2::math::sqrt(sinThetaSq);
        const float cosTheta = scene_rdl2::math::sqrt(1.0f - sinThetaSq);

        math::ReferenceFrame frame(-v/scene_rdl2::math::sqrt(v2));
        const Vec3f localNormal = vectorFromPolarSinCosVersion(sinTheta, cosTheta, sTwoPi*r[1]);
        normal = frame.localToGlobal(localNormal);  // Outward-facing normal

        Vec3f rayDir = v + mRadius * normal;
        const float t2 = lengthSqr(rayDir);
        t = scene_rdl2::math::sqrt(t2);
        if (t < sEpsilon) {
            return false;
        }

        wi = rayDir / t;
        if (n  &&  dot(*n, wi) < sEpsilon) {
            return false;
        }

        // Store values which eval() will later use to compute the pdf.
        isect.data[0] = cosTheta / sinThetaMaxSq;
        isect.data[1] = t;

        // We don't need to worry about 2-sided here because that's not supported for SphereLight
        if (mSidedness == LightSidednessType::LIGHT_SIDEDNESS_REVERSE) {
            // This is nasty. We use the direction of the generated sample, but continue the ray to where
            // it hits the far side of the sphere. It's done this way so the sampling scheme doesn't change,
            // because otherwise it could yield a different image when sample clamping is active. Although
            // no prior production shots used SphereLights with reverse sidedness, we want the result to match
            // that of a SphereLight with regular sidedness when there is no geometry inside the SphereLight.
            // Also, it is important to adjust the values of t and normal AFTER stashing the values in isect.data
            // which will be used by eval() to compute the pdf. This is because the pdf comes from a distribution
            // of points on the front-facing surface of the sphere.
            const float b = dot(wi, v);
            const float D = mRadiusSqr - lengthSqr(v - b * wi); // More accurate version of discriminant than b^2-c
            t = b + scene_rdl2::math::sqrt(max(D, 0.0f));         // Rounding can cause D to go slight negative, hence the max()
            normal = (t*wi - v) * mRcpRadius;   // Outward-facing normal
        }
    }

    isect.N = (mSidedness == LightSidednessType::LIGHT_SIDEDNESS_REGULAR) ? normal : -normal;
    isect.distance = t;

    // Optionally compute uvs for texturing
    isect.uv = mDistribution ? local2uv(xformVectorRender2LocalRot(normal, time)) : zero;

    return true;
}


Color
SphereLight::eval(mcrt_common::ThreadLocalState* tls, const Vec3f &wi, const Vec3f &p, const LightFilterRandomValues& filterR, float time,
        const LightIntersection &isect, bool fromCamera, const LightFilterList *lightFilterList, float rayDirFootprint,
        float *pdf) const
{
    MNRY_ASSERT(mOn);

    Color radiance = mRadiance;

    if (mDistribution) {
        // Note: the distribution doesn't contribute to the pdf, since we ignore it when sampling
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

    // To evaluate the pdf, we need information about which sampling scheme was used (or, in the case of the
    // intersect() function, which one would have been used for the given geometric config) -
    // (a) point outside sphere, cosine-weighted sampling over front-facing surface
    // (b) point inside sphere, uniform sampling over hemisphere above local horizon
    // (c) point inside sphere, uniform sampling over full sphere
    // Cases (b) and (c) only apply to reverse sidedness. In case (a), whether normal or reverse sidedness, 
    // we use isect.data[0] and [1] to pass a pair of intermediate values through to this point for computing
    // the pdf. In cases (b) and (c), we pass the pdf directly in isect.data[0], and use a negative value in
    // isect.data[1] to signal this.
    if (pdf) {
        float t = isect.data[1];
        if (t >= 0.0f) {
            // See Dutre's "Global Illumination Compendium" section 35, page 20.
            float costThetaOverSinThetaSq = isect.data[0];
            MNRY_ASSERT(scene_rdl2::math::isfinite(costThetaOverSinThetaSq));
            *pdf = costThetaOverSinThetaSq / (sPi * mRadiusSqr);
            *pdf *= areaToSolidAngleScale(wi, isect.N, t);
            MNRY_ASSERT(scene_rdl2::math::isfinite(*pdf));
        } else {
            // t<0 is used to signal that isect.data[0] stores the pdf directly
            *pdf = isect.data[0];
        }
    }

    return radiance;
}


void
SphereLight::initAttributeKeys(const scene_rdl2::rdl2::SceneClass &sc)
{
    if (sAttributeKeyInitialized) {
        return;
    }

    MOONRAY_START_NON_THREADSAFE_STATIC_WRITE

    sAttributeKeyInitialized = true;

    sNormalizedKey      = sc.getAttributeKey<scene_rdl2::rdl2::Bool> ("normalized");
    sApplySceneScaleKey = sc.getAttributeKey<scene_rdl2::rdl2::Bool> ("apply_scene_scale");
    sRadiusKey          = sc.getAttributeKey<scene_rdl2::rdl2::Float>("radius");
    sSidednessKey       = sc.getAttributeKey<scene_rdl2::rdl2::Int>  ("sidedness");
    INIT_ATTR_KEYS_CLEAR_RADIUS

    MOONRAY_FINISH_NON_THREADSAFE_STATIC_WRITE
}


//----------------------------------------------------------------------------

} // namespace pbr
} // namespace moonray

