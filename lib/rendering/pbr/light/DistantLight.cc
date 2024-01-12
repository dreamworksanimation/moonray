// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0


//----------------------------------------------------------------------------
//
// mrday - 9/27/17 - DistantLight overhaul
// ---------------------------------------
// mslee and I tracked down some bugs and accuracy issues with distant lights. We also wanted to
// to rework the behaviour of 'normalised' distant lights, so an overhaul was performed. The
// following points came up during the work.
//
// * The new physical interpretation of a distant light is that it's a spherical cap, where the
// sphere in question is centred on the observer and has infinite radius. This replaces the prior
// interpretation as a disk light at infinity (whose implementation suffered from subtle bugs).
//
// * One accuracy issue was the area computation. The area of the cap is 2pi * (1-cosThetaMax),
// where ThetaMax represents the angle between the centre of the light and the edge (i.e. half
// the angular diameter). When cosThetaMax approaches 1, the expression 1-cosThetaMax suffers from
// catastrophic cancellation, and as a result of this something later in the pipeline was breaking
// down around thetaMax=0.15 degrees (so 0.3 degrees for the light's angular diameter). This was
// easy to fix, though, using a trig identity:
//
//      1-cos(t) = 2(sin(t))^2.
//
// Using sine maintains accuracy all the way down to a zero angle. In fact, an accurate 1-cosThetaMax was
// useful elsewhere, so we store it in the light's mVersineThetaMax. The versine function is 1-cosine.
// With this improvement, the light's angle works all the way down to about 0.00000004 degrees - it could
// perhaps be clamped at this very small value in case it's set to zero (representing a pure directional
// light).
//
// * A somewhat related accuracy issue came up the angle is small. That was in the function
// sampleLocalSphericalCapUniform() which generates the samples over the light's extent.
// The newly added version of this funciton now takes as a parameter the more accurately computed
// versine, rather than the cosine of ThetaMax. It also uses a more numerically robust method to
// recover the sine from the cosine (or versine). The original calculation was
//
//      sin(theta) = sqrt(1 - (cos(theta))^2)
//
// but this suffers from catastrophic cancellation when theta is near zero. So it was replaced with
//
//      sin(theta) = sqrt(versine(theta) * (2.0f - versine(theta))
//
// which is mathematically equivalent but in floating point is far more numerically accurate for small
// angles. This fixed a problem with shadow penumbras from small distant lights, in which catastrophic
// cancellation made it possible to see discrete steps correspoinding to changes of 1 ULP in the cosine
// value. The numerically robust version eliminates the steps and gives the desired smooth gradient.
//
// * There is still one potential accuracy issue. This is the way thresholding is performed in
// DistantLight::intersect(). By comparing the dot product to cosThetaMax, we lose considerable accuracy
// when both values are close to 1. But it doesn't appear to cause any trouble so it can be left as-is
// for now.
//
// * Normalisation of distant lights has been reworked. The new definition of 'normalised' for distant
// lights is that when a distant light of uniform radiance (1,1,1) and any angular extent is placed
// directly overhead above a Lambertian surface of colour (1,1,1), the resulting outgoing radiance will
// be (1,1,1). This requires applying an angle-dependent normalisation factor which was computed by
// integration and derived directly from the rendering equation. The required normalisation factor is
//
//      1/(sin(thetaMax))^2
//
// after clamping thetaMax to be no greater than a right-angle (since if the distant light covers more
// than a hemisphere, only the upper hemisphere contributes light to the surface).
// See DistantLight::update() for the implementation.



#include "DistantLight.h"

#include <moonray/rendering/pbr/light/DistantLight_ispc_stubs.h>
#include <moonray/rendering/shading/Util.h>

#include <scene_rdl2/scene/rdl2/rdl2.h>
#include <moonray/common/mcrt_macros/moonray_static_check.h>

using namespace scene_rdl2;
using namespace scene_rdl2::math;

namespace moonray {
namespace pbr {


bool                             DistantLight::sAttributeKeyInitialized;
scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Bool>   DistantLight::sNormalizedKey;
scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Float>  DistantLight::sAngularExtent;


//----------------------------------------------------------------------------

HUD_VALIDATOR(DistantLight);

Vec3f
DistantLight::localToGlobal(const Vec3f &v, float time) const
{
    if (!isMb()) return mFrame.localToGlobal(v);

    // construct a new frame
    Mat3f m(slerp(mOrientation[0], mOrientation[1], time));
    return v * m;
}

Xform3f
DistantLight::globalToLocalXform(float time, bool needed) const
{
    if (!needed) {
        return math::Xform3f();
    }

    if (!isMb()) {
        // construct a new frame
        return Mat3f(mOrientation[0]).transposed();
    }

    // construct a new frame
    return Mat3f(slerp(mOrientation[0], mOrientation[1], time)).transposed();
}

DistantLight::DistantLight(const scene_rdl2::rdl2::Light* rdlLight, bool uniformSampling) :
    Light(rdlLight)
{
    mIsOpaqueInAlpha = false;

    initAttributeKeys(rdlLight->getSceneClass());

    ispc::DistantLight_init(this->asIspc(), uniformSampling);
}

DistantLight::~DistantLight()
{
}

bool
DistantLight::update(const Mat4d& world2render)
{
    MNRY_ASSERT(mRdlLight);

    mOn = mRdlLight->get(scene_rdl2::rdl2::Light::sOnKey);
    if (!mOn) {
        return false;
    }

    updateVisibilityFlags();
    updatePresenceShadows();
    updateRayTermination();
    updateMaxShadowDistance();

    const Mat4d l2w0 = mRdlLight->get(scene_rdl2::rdl2::Node::sNodeXformKey, /* rayTime = */ 0.0f);
    const Mat4d l2w1 = mRdlLight->get(scene_rdl2::rdl2::Node::sNodeXformKey, /* rayTime = */ 1.0f);
    const Mat4f local2render0 = Mat4f::orthonormalize(sRotateX180 * toFloat(l2w0 * world2render));
    const Mat4f local2render1 = Mat4f::orthonormalize(sRotateX180 * toFloat(l2w1 * world2render));
    ReferenceFrame frame0(local2render0);
    ReferenceFrame frame1(local2render1);
    mPosition[0] = mPosition[1] = zero;
    mOrientation[0] = normalize(math::Quaternion3f(frame0.getX(), frame0.getY(), frame0.getZ()));
    mOrientation[1] = normalize(math::Quaternion3f(frame1.getX(), frame1.getY(), frame1.getZ()));
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

    float angularExtent = mRdlLight->get(sAngularExtent);
    float halfAngle = deg2rad(angularExtent * 0.5f);

    mCullThreshold = scene_rdl2::math::cos(halfAngle + sHalfPi);
    MNRY_ASSERT(mCullThreshold <= 0.0f);

    mCosThetaMax = scene_rdl2::math::cos(halfAngle);
    float sinQuarterAngle = scene_rdl2::math::sin(halfAngle * 0.5f);
    mVersineThetaMax = 2.0f * sinQuarterAngle * sinQuarterAngle;

    // We store solid angle and its inverse in the area members, since this is also useful for the unittest.
    // We use an accurately computed versine here because 1-cosine suffers from catastrophic cancellation
    // when the angle is small.
    mArea = mVersineThetaMax * sTwoPi;
    mInvArea = 1.0f / mArea;

    // Compute radiance.
    mRadiance = computeLightRadiance(mRdlLight, scene_rdl2::rdl2::Light::sColorKey,
        scene_rdl2::rdl2::Light::sIntensityKey, scene_rdl2::rdl2::Light::sExposureKey);

    // Apply normalisation factor if light is normalised. The normalisation factor used here is such that
    // if a distant light of radiance (1,1,1) is directly overhead a Lambertian surface of colour (1,1,1),
    // the  resulting outgoing radiance at the surface will be (1,1,1) regardless of the light's angular
    // extent. This factor can be derived from the rendering equation.
    if (mRdlLight->get<scene_rdl2::rdl2::Bool>(sNormalizedKey)) {
        float s = scene_rdl2::math::sin(min(halfAngle,sHalfPi));
        mRadiance *= 1.0f / (s*s);
    }

    if (isBlack(mRadiance)) {
        mOn = false;
        return false;
    }

    return true;
}

bool
DistantLight::canIlluminate(const Vec3f p, const Vec3f *n, float time, float radius,
    const LightFilterList* lightFilterList) const
{
    MNRY_ASSERT(mOn);

    if (n && dot(-(*n), getDirection(time)) < mCullThreshold) {
        return false;
    }

    if (lightFilterList) {
        return canIlluminateLightFilterList(lightFilterList,
            { getPosition(time),
              math::inf, p, globalToLocalXform(time, lightFilterList->needsLightXform()),
              radius, time 
            });
    }

    return true;
}

bool
DistantLight::isBounded() const
{
    return false;
}

bool
DistantLight::isDistant() const
{
    return true;
}

bool
DistantLight::isEnv() const
{
    return false;
}

bool
DistantLight::intersect(const Vec3f &p, const Vec3f *n, const Vec3f &wi, float time,
        float maxDistance, LightIntersection &isect) const
{
    if (dot(-wi, getDirection(time)) < mCosThetaMax) {
        return false;
    }

    if (sDistantLightDistance > maxDistance) {
        return false;
    }


    isect.N = -wi;
    isect.distance = sDistantLightDistance;
    isect.uv = zero;

    return true;
}

bool
DistantLight::sample(const Vec3f &p, const Vec3f *n, float time, const Vec3f& r,
                     Vec3f &wi, LightIntersection &isect, float rayDirFootprint) const
{
    MNRY_ASSERT(mOn);

    Vec3f sample = shading::sampleLocalSphericalCapUniform2(r[0], r[1], mVersineThetaMax);
    wi = -localToGlobal(sample, time);

    if (n  &&  dot(*n, wi) < sEpsilon) {
        return false;
    }

    isect.N = -wi;
    isect.distance = sDistantLightDistance;
    isect.uv = zero;

    return true;
}

Color
DistantLight::eval(mcrt_common::ThreadLocalState* tls, const Vec3f &wi, const Vec3f &p, const LightFilterRandomValues& filterR,
        float time, const LightIntersection &isect, bool fromCamera, const LightFilterList *lightFilterList,
        float rayDirFootprint, float *pdf) const
{
    MNRY_ASSERT(mOn);

    Color radiance = mRadiance;

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
        *pdf = mInvArea;
    }
    return radiance;
}

Vec3f
DistantLight::getEquiAngularPivot(const Vec3f& r, float time) const
{
    MNRY_ASSERT_REQUIRE(isBounded(),
        "light with infinite distance should not use equi-angular sampling");
    return getPosition(time);
}

void
DistantLight::initAttributeKeys(const scene_rdl2::rdl2::SceneClass &sc)
{
    if (sAttributeKeyInitialized) {
        return;
    }

    MOONRAY_START_NON_THREADSAFE_STATIC_WRITE

    sAttributeKeyInitialized = true;

    sNormalizedKey = sc.getAttributeKey<scene_rdl2::rdl2::Bool>("normalized");

    sAngularExtent = sc.getAttributeKey<scene_rdl2::rdl2::Float>("angular_extent");

    MOONRAY_FINISH_NON_THREADSAFE_STATIC_WRITE
}

//----------------------------------------------------------------------------

} // namespace pbr
} // namespace moonray

