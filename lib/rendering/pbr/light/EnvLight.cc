// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0


#include "EnvLight.h"
#include <moonray/rendering/pbr/core/Distribution.h>

#include <moonray/rendering/pbr/light/EnvLight_ispc_stubs.h>
#include <moonray/rendering/shading/Util.h>

#include <scene_rdl2/scene/rdl2/rdl2.h>
#include <moonray/common/mcrt_macros/moonray_static_check.h>

namespace moonray {
namespace pbr {

using namespace scene_rdl2::math;

bool                             EnvLight::sAttributeKeyInitialized;
scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Bool>   EnvLight::sSampleUpperHemisphereOnlyKey;

/// Special case for EnvLight where we want +Y to be the up direction in world/
/// render space. We still want z to be up in local space.
/// Rotate around X by -90, then around Y by 90 to match orchid's orientation.
const Mat4f EnvLight::sLocalOrientation = Mat4f( 0.0f, 0.0f,-1.0f, 0.0f,
                                                -1.0f, 0.0f, 0.0f, 0.0f,
                                                 0.0f, 1.0f, 0.0f, 0.0f,
                                                 0.0f, 0.0f, 0.0f, 1.0f );


//----------------------------------------------------------------------------

namespace {

// Compute local space dir from uv sphere parameterization. The output vector
// is defined in a z-up local coordinate system.
finline Vec3f
uv2local(const Vec2f &uv)
{
    // We are flipping the u coordinate here since we're on the inside of the
    // sphere. We're flipping v since local z and texture v are in opposite
    // directions.
    float phi = (1.0f - uv.x) * sTwoPi;
    float theta = (1.0f - uv.y) * sPi;

    float sinTheta, cosTheta, sinPhi, cosPhi;
    sincos(theta, &sinTheta, &cosTheta);
    sincos(phi, &sinPhi, &cosPhi);

    return Vec3f(sinTheta * cosPhi, sinTheta * sinPhi, cosTheta);
}


// Compute uv sphere parameterization from local space direction. The input dir
// is assumed to be defined in a z-up local coordinate system.
finline Vec2f
local2uv(const Vec3f &dir)
{
    MNRY_ASSERT(isNormalized(dir));

    float theta = scene_rdl2::math::acos(clamp(dir.z, -1.0f, 1.0f));
    float phi = scene_rdl2::math::atan2(dir.y, dir.x);

    float u = phi * sOneOverTwoPi;
    float v = 1.0f - (theta * sOneOverPi);

    // Handle singularities at poles, to keep function consistant when inverted.
    if (v < 0.0005f || v > 0.9995f) {
        u = 0.0f;
    } else if (u < 0.0f) {
        u += 1.0f;
    }

    // undo the flip of the u coordinate that we did in uv2local
    u = 1.0f - u;

    MNRY_ASSERT(finite(u));
    MNRY_ASSERT(finite(v));

    return Vec2f(u, v);
}

}   // end of anon namespace

//----------------------------------------------------------------------------

HUD_VALIDATOR(EnvLight);

Vec3f
EnvLight::localToGlobal(const Vec3f &v, float time) const
{
    if (!isMb()) return mFrame.localToGlobal(v);

    // construct a new frame
    Mat3f m(slerp(mOrientation[0], mOrientation[1], time));
    return v * m;
}

Vec3f
EnvLight::globalToLocal(const Vec3f &v, float time) const
{
    if (!isMb()) return mFrame.globalToLocal(v);

    // construct a new frame
    Mat3f m = Mat3f(slerp(mOrientation[0], mOrientation[1], time)).transposed();
    return v * m;
}

Xform3f
EnvLight::globalToLocalXform(float time, bool needed) const
{
    if (!needed) {
        return scene_rdl2::math::Xform3f();
    }

    if (!isMb()) {
        // construct a new frame
        return Mat3f(mOrientation[0]).transposed();
    }

    // construct a new frame
    return Mat3f(slerp(mOrientation[0], mOrientation[1], time)).transposed();
}

EnvLight::EnvLight(const scene_rdl2::rdl2::Light* rdlLight) :
    Light(rdlLight),
    mHemispherical(false)
{
    mIsOpaqueInAlpha = false;

    initAttributeKeys(rdlLight->getSceneClass());

    ispc::EnvLight_init(this->asIspc());
}

EnvLight::~EnvLight() { }

bool
EnvLight::update(const Mat4d& world2render)
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

    const Mat4d l2w0 = mRdlLight->get(scene_rdl2::rdl2::Node::sNodeXformKey, /* rayTime = */ 0.0f);
    const Mat4d l2w1 = mRdlLight->get(scene_rdl2::rdl2::Node::sNodeXformKey, /* rayTime = */ 1.0f);
    const Mat4f local2render0 = Mat4f::orthonormalize(sLocalOrientation * toFloat(l2w0 * world2render));
    const Mat4f local2render1 = Mat4f::orthonormalize(sLocalOrientation * toFloat(l2w1 * world2render));
    ReferenceFrame frame0(local2render0);
    ReferenceFrame frame1(local2render1);
    mPosition[0] = mPosition[1] = zero;
    mOrientation[0] = normalize(scene_rdl2::math::Quaternion3f(frame0.getX(), frame0.getY(), frame0.getZ()));
    mOrientation[1] = normalize(scene_rdl2::math::Quaternion3f(frame1.getX(), frame1.getY(), frame1.getZ()));
    if (dot(mOrientation[0], mOrientation[1]) < 0) {
        mOrientation[1] *= -1.f;
    }
    mFrame = frame0;
    mDirection = mFrame.getZ();

    // setup mMb
    mMb = LIGHT_MB_NONE;
    const scene_rdl2::rdl2::SceneVariables &vars =
        getRdlLight()->getSceneClass().getSceneContext()->getSceneVariables();
    if (vars.get(scene_rdl2::rdl2::SceneVariables::sEnableMotionBlur) &&
        getRdlLight()->get(scene_rdl2::rdl2::Light::sMbKey) &&
        (!isEqual(frame0.getX(), frame1.getX()) ||
         !isEqual(frame0.getY(), frame1.getY()) ||
         !isEqual(frame0.getZ(), frame1.getZ())))  {
        mMb = LIGHT_MB_ROTATION;
    }

    // We store solid angle and its inverse in the area members, since this is
    // also useful for the unittest.
    // Note: As with the DiskLight, we sample the whole (theta, phi) domain
    // and let the ImageDistribution be responsible for distributing the
    // samples over the (hemi)sphere as the  Distribution2D::HEMISPHERICAL
    // will zero-out weights in the lower hemisphere and automatically
    // renormalize the pdf accordingly.
    mHemispherical = mRdlLight->get<scene_rdl2::rdl2::Bool>(sSampleUpperHemisphereOnlyKey);
    mArea = (mHemispherical  ?  sTwoPi  :  sFourPi);
    mInvArea = 1.0f / mArea;

    // for EnvLights, color is assumed to always be a radiance value
    mRadiance = computeLightRadiance(mRdlLight, scene_rdl2::rdl2::Light::sColorKey,
        scene_rdl2::rdl2::Light::sIntensityKey, scene_rdl2::rdl2::Light::sExposureKey);
    if (isBlack(mRadiance)) {
        mOn = false;
        return false;
    }

    // Set here in case we early-out
    mLog2TexelAngle = scene_rdl2::math::neg_inf;

    if (!updateImageMap(mHemispherical ? Distribution2D::HEMISPHERICAL :
                                         Distribution2D::SPHERICAL)) {
        return false;
    }

    // Precompute a value for use with ray-footprint-based mip level selection.
    // We base it on the largest extent (lat or long) of the largest texel, storing
    // log2() of the result because it accelerates the mip level selection calculation.
    if (mDistribution) {
        float texelAngleLongitude = sTwoPi / (float)mDistribution->getWidth();
        float texelAngleLatitude  = sPi    / (float)mDistribution->getHeight();
        float texelAngle = max(texelAngleLongitude, texelAngleLatitude);
        mLog2TexelAngle = scene_rdl2::math::log2(texelAngle);
    }

    return true;
}


bool
EnvLight::isBounded() const
{
    return false;
}

bool
EnvLight::isDistant() const
{
    return false;
}

bool
EnvLight::isEnv() const
{
    return true;
}

bool
EnvLight::intersect(const Vec3f &p, const Vec3f *n, const Vec3f &wi, float time,
        float maxDistance, LightIntersection &isect) const
{
    if (mHemispherical && dot(wi, getDirection(time)) < 0.0f) {
        return false;
    }

    if (sEnvLightDistance > maxDistance) {
        return false;
    }

    isect.N = -wi;
    isect.distance = sEnvLightDistance;
    isect.uv = mDistribution ? local2uv(globalToLocal(wi, time)) : zero;

    return true;
}

bool
EnvLight::sample(const Vec3f &p, const Vec3f *n, float time, const Vec3f& r,
                 Vec3f &wi, LightIntersection &isect, float rayDirFootprint) const
{
    MNRY_ASSERT(mOn);

    if (mDistribution) {
        float mipLevel = getMipLevel(rayDirFootprint);
        mDistribution->sample(r[0], r[1], mipLevel, &isect.uv, nullptr, mTextureFilter);

        // Handle singularities at poles so that sample() and intersect()
        // will return identical values. if v == 0, then throw away u value
        // since it's meaningless.
        if (isect.uv.y < 0.0005f || isect.uv.y > 0.9995f) {
            isect.uv.x = 0.0f;
        }

        wi = uv2local(isect.uv);
        wi = localToGlobal(wi, time);
    } else if (mHemispherical) {
        isect.uv = zero;
        wi = shading::sampleLocalHemisphereUniform(r[0], r[1]);
        wi = localToGlobal(wi, time);
    } else {
        isect.uv = zero;
        wi = shading::sampleSphereUniform(r[0], r[1]);
    }

    if (n  &&  dot(*n, wi) < sEpsilon) {
        return false;
    }

    isect.N = -wi;
    isect.distance = sEnvLightDistance;

    return true;
}


Color
EnvLight::eval(mcrt_common::ThreadLocalState* tls, const Vec3f &wi, const Vec3f &p, const LightFilterRandomValues& filterR, float time,
        const LightIntersection &isect, bool fromCamera, const LightFilterList *lightFilterList, float rayDirFootprint,
        float *pdf) const
{
    MNRY_ASSERT(mOn);

    float mipLevel = getMipLevel(rayDirFootprint);

    Color radiance = mRadiance;
    if (mDistribution) {
        radiance *= mDistribution->eval(isect.uv[0], isect.uv[1], mipLevel, mTextureFilter);
    }

    if (lightFilterList) {
        evalLightFilterList(lightFilterList,
                            { tls, &isect, getPosition(time),
                              getDirection(time), p,
                              filterR, time,
                              globalToLocalXform(time, lightFilterList->needsLightXform()),
                              wi
                            },
                            radiance);
    }

    if (pdf) {
        if (mDistribution) {
            // We must account for the mapping transformation so we express the
            // pdf density on the sphere in solid angles (see pbrt section 14.6.5)
            float sinTheta = scene_rdl2::math::sin((isect.uv[1]) * sPi);
            sinTheta = max(sinTheta, sEpsilon);
            *pdf = mDistribution->pdf(isect.uv[0], isect.uv[1], mipLevel, mTextureFilter)
                 / (sTwoPiSqr * sinTheta);
            MNRY_ASSERT(finite(*pdf));
        } else {
            *pdf = mInvArea;
        }
    }

    return radiance;
}

Vec3f
EnvLight::getEquiAngularPivot(const Vec3f& r, float time) const
{
    MNRY_ASSERT_REQUIRE(isBounded(),
        "light with infinite distance should not use equi-angular sampling");
    return getPosition(time);
}

void
EnvLight::initAttributeKeys(const scene_rdl2::rdl2::SceneClass &sc)
{
    if (sAttributeKeyInitialized) {
        return;
    }

    MOONRAY_START_NON_THREADSAFE_STATIC_WRITE

    sAttributeKeyInitialized = true;

    sSampleUpperHemisphereOnlyKey = sc.getAttributeKey<scene_rdl2::rdl2::Bool>("sample_upper_hemisphere_only");

    MOONRAY_FINISH_NON_THREADSAFE_STATIC_WRITE
}


//----------------------------------------------------------------------------

} // namespace pbr
} // namespace moonray

