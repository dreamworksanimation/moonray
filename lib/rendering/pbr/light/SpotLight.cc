// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0


#include "SpotLight.h"
#include <moonray/rendering/pbr/core/Util.h>

#include <moonray/rendering/pbr/core/Distribution.h>
#include <moonray/rendering/pbr/light/SpotLight_ispc_stubs.h>

#include <moonray/common/mcrt_macros/moonray_static_check.h>
#include <scene_rdl2/scene/rdl2/rdl2.h>


namespace moonray {
namespace pbr {


using namespace scene_rdl2;
using namespace scene_rdl2::math;
using scene_rdl2::logging::Logger;


bool                             SpotLight::sAttributeKeyInitialized;
scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Bool>   SpotLight::sNormalizedKey;
scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Bool>   SpotLight::sApplySceneScaleKey;
scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Float>  SpotLight::sLensRadiusKey;
scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Float>  SpotLight::sAspectRatioKey;
scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Float>  SpotLight::sFocalPlaneDistanceKey;
scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Float>  SpotLight::sOuterConeAngleKey;
scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Float>  SpotLight::sInnerConeAngleKey;
scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Int>    SpotLight::sAngleFalloffTypeKey;
scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Float>  SpotLight::sBlackLevelKey;
scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Float>  SpotLight::sClearRadiusKey;
scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Float>  SpotLight::sClearRadiusFalloffDistanceKey;
scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Int>    SpotLight::sClearRadiusInterpolationKey;

//----------------------------------------------------------------------------

HUD_VALIDATOR(SpotLight);

void
SpotLight::computeCorners(Vec3f *corners, float time) const
{
    corners[0] = xformPointLocal2Render(Vec3f( mLensRadius,  mLensRadiusY, 0.f), time);
    corners[1] = xformPointLocal2Render(Vec3f( mLensRadius, -mLensRadiusY, 0.f), time);
    corners[2] = xformPointLocal2Render(Vec3f(-mLensRadius,  mLensRadiusY, 0.f), time);
    corners[3] = xformPointLocal2Render(Vec3f(-mLensRadius, -mLensRadiusY, 0.f), time);
}

SpotLight::SpotLight(const scene_rdl2::rdl2::Light* rdlLight) :
    LocalParamLight(rdlLight)
{
    initAttributeKeys(rdlLight->getSceneClass());

    ispc::SpotLight_init(this->asIspc());
}

SpotLight::~SpotLight()
{
}

bool
SpotLight::update(const Mat4d &world2render)
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
    UPDATE_ATTRS_CLEAR_RADIUS

    const Mat4d l2w0 = mRdlLight->get(scene_rdl2::rdl2::Node::sNodeXformKey, /* rayTime = */ 0.f);
    const Mat4d l2w1 = mRdlLight->get(scene_rdl2::rdl2::Node::sNodeXformKey, /* rayTime = */ 1.f);
    Mat4f rot = sRotateX180;
    const Mat4f local2Render0 = rot * toFloat(l2w0 * world2render);
    const Mat4f local2Render1 = rot * toFloat(l2w1 * world2render);

    // Aspect ratio is local y-dimension divided by local x-dimension; mRcpAspectRatio is its reciprocal
    mRcpAspectRatio = 1.0f / mRdlLight->get<scene_rdl2::rdl2::Float>(sAspectRatioKey);

    mLensRadius  = mRdlLight->get<scene_rdl2::rdl2::Float>(sLensRadiusKey);
    mRcpLensRadius  = 1.0f / mLensRadius;
    mLensRadiusY = mLensRadius * mRcpAspectRatio;
    mRcpLensRadiusY = 1.0f / mLensRadiusY;

    // Inner and outer angles are measured from one side to the other
    float outerConeAngle = mRdlLight->get<scene_rdl2::rdl2::Float>(sOuterConeAngleKey);
    float innerConeAngle = mRdlLight->get<scene_rdl2::rdl2::Float>(sInnerConeAngleKey);

    // Outer cone is subject to barely any restriction; inner cone angle must not exceed outer cone angle
    outerConeAngle = clamp(outerConeAngle, -179.999f, 179.999f);
    innerConeAngle = clamp(innerConeAngle, -179.999f, outerConeAngle);

    float tanHalfAngleOuter = scene_rdl2::math::tan(deg2rad(outerConeAngle * 0.5f));
    float tanHalfAngleInner = scene_rdl2::math::tan(deg2rad(innerConeAngle * 0.5f));
    float tanHalfAngleDiff  = tanHalfAngleOuter - tanHalfAngleInner;

    mFocalDistance = mRdlLight->get<scene_rdl2::rdl2::Float>(sFocalPlaneDistanceKey);
    if (mFocalDistance < 0.0001f) {
        mFocalDistance = 0.0001f;
    }

    // If cone tapers inwards, we must clamp the focal plane so it's closer than the apex by some arbitrary amount
    if (tanHalfAngleOuter < 0.0f) {
        mFocalDistance = std::min(mFocalDistance, -0.999f * mLensRadius / tanHalfAngleOuter);
    }

    // Radius of pool of light at focal plane
    mFocalRadius = tanHalfAngleOuter * mFocalDistance + mLensRadius;
    mRcpFocalRadius = 1.0f / mFocalRadius;
    mFocalRadiusY = mFocalRadius * mRcpAspectRatio;
    mRcpFocalRadiusY = 1.0f / mFocalRadiusY;

    // Rate of falloff between inner and outer focal radii
    mFalloffGradient = mFocalRadius / std::max(tanHalfAngleDiff * mFocalDistance, 1.0e-10f);

    // Distance at which sampling scheme switches from sampling over focal plane to sampling over lens
    mCrossOverDistance = mFocalDistance * mLensRadius / (mLensRadius + mFocalRadius);

    if (!updateParamAndTransforms(local2Render0, local2Render1, mLensRadius, mLensRadiusY)) {
        return false;
    }

    // Compute render space corners of light.
    // Cached at rayTime = 0
    computeCorners(mRenderCorners, 0.f);

    // Compute render space quantities.
    // TODO: mb scale radiance
    float halfRenderWidth  = mLensRadius  * mLocal2RenderScale[0];
    float halfRenderHeight = mLensRadiusY * mLocal2RenderScale[0];

    mArea = sPi * halfRenderWidth * halfRenderHeight;
    mInvArea = 1.0f / mArea;

    // Constant used in pdf calc
    float lensRadiusOverFocalRadius = mLensRadius / mFocalRadius;
    mFocalPlanePdfConst = mInvArea * lensRadiusOverFocalRadius * lensRadiusOverFocalRadius;

    // Compute radiance.
    mRadiance = computeLightRadiance(mRdlLight, scene_rdl2::rdl2::Light::sColorKey,
        scene_rdl2::rdl2::Light::sIntensityKey, scene_rdl2::rdl2::Light::sExposureKey, sNormalizedKey,
        sApplySceneScaleKey, mInvArea);
    if (isBlack(mRadiance)) {
        mOn = false;
        return false;
    }

    // Falloff curve
    mFalloffCurve.init(
        FalloffCurveType(mRdlLight->get<scene_rdl2::rdl2::Int>(sAngleFalloffTypeKey)));
    mBlackLevel = mRdlLight->get<scene_rdl2::rdl2::Float>(sBlackLevelKey);

    if (!updateImageMap(Distribution2D::Mapping::CIRCULAR)) {
        return false;
    }

    return true;
}


bool
SpotLight::canIlluminate(const Vec3f p, const Vec3f *n, float time, float radius,
    const LightFilterList* lightFilterList) const
{
    MNRY_ASSERT(mOn);

    // Cull points which are completely on the backside of the light
    Vec3f localP = xformPointRender2Local(p, time);
    if (localP.z <= 0.0f) {
        return false;
    }

    // Cull lights which are completely on the backside of the point
    const float threshold = sEpsilon - radius;
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
              xformLocal2RenderScale(math::max(mLensRadius, mLensRadiusY), time),
              p, getXformRender2Local(time, lightFilterList->needsLightXform()),
              radius, time
            });
    }

    return true;
}

bool
SpotLight::isBounded() const
{
    return true;
}

bool
SpotLight::isDistant() const
{
    return false;
}

bool
SpotLight::isEnv() const
{
    return false;
}

// Computing a tight bounding box for a spotlight
// ----------------------------------------------
//
// Bounds for a spotlight are computed as bounds for the lens. In the spotlight's local coordinates,
// the lens is defined by an ellipse in the (x,y)-plane, centred on the origin, with radius mLensRadius
// in the x-direction and (mLensRadius * mRcpAspectRatio) in the y-direction.
//
// The derivation below is based on the somewhat simpler bounds for a disk light; see DiskLight.cc
//
// In local space, a point (x,y,z) on the perimeter of the disk can be expressed parametrically as
//
//      x = r cos t
//      y = r a sin t
//      z = 0
//
// with t ranging from 0 to 2pi, r = mLensRadius and a = mRcpAspectRatio.
// Suppose the local2render matrix is (Mij) with i=0,1,2,3 and j=0,1,2,3.
// Then the perimeter point (x,y,z) transforms to (x',y',z') in render space, where
//
//      x' = M00 r cos t + M10 r a sin t + M30
//      y' = M01 r cos t + M11 r a sin t + M31
//      z' = M02 r cos t + M12 r a sin t + M32
//
// Consider just the x' coordinate. Its min and max occur at points where dx'/dt=0. Differentiating the above gives
//
//      dx'/dt = -M00 r sin t + M10 r a cos t
//
// If this equals zero then
//
//      tan t = a M10 / M00
//
// Or, using standard trig identities,
//
//      cos^2 t =     M00^2 / (M00^2 + a^2 M10^2)
//      sin^2 t = a^2 M10^2 / (M00^2 + a^2 M10^2)
//
// Taking square roots, and ensuring the chosen combination of signs preserves the sign of tan t,
//
//      (cos t, sin t) = +/- (M00, a M10) / sqrt(M00^2 + a^2 M10^2)
//
// Putting these values into the expression for x' gives
//
//      x' = M30 +/- r (M00^2 + a^2 M10^2) / sqrt(M00^2 + a^2 M10^2)
//         = M30 +/- r sqrt(M00^2 + a^2 M10^2)
//
// The expressions for the min & max y' and z' are similar, and the 3 together can be expressed compactly
// in terms of Vec3f operations.

BBox3f
SpotLight::getBounds() const
{
    const Vec3f pos = getPosition(0.f);
    const Vec3f xRow = mLocal2Render[0].l.vx;
    const Vec3f yRow = mRcpAspectRatio * mLocal2Render[0].l.vy;
    const Vec3f halfDims = max(mLensRadius * sqrt(xRow*xRow + yRow*yRow),
                               Vec3f(sEpsilon));     // ensure non-degeneracy
    BBox3f bounds(pos - halfDims, pos + halfDims);

    if (isMb()) {
        const Vec3f pos = getPosition(1.f);
        const Vec3f xRow = mLocal2Render[1].l.vx;
        const Vec3f yRow = mRcpAspectRatio * mLocal2Render[1].l.vy;
        const Vec3f halfDims = max(mLensRadius * sqrt(xRow*xRow + yRow*yRow),
                                   Vec3f(sEpsilon));     // ensure non-degeneracy
        bounds.extend(BBox3f(pos - halfDims, pos + halfDims));
    }

    return bounds;
}

// Helper functions for converting between the spotlight's coordinate systems:
// Lens coordinates measure a local position on the lens, in world-space distance units.
// Normalized lens coordinates do the same but range from -1 to 1 over each axis of the lens.
// Focal coordinates measure a local position on the focal plane, in world-space distance units.
// Normalized focal coordinates do the same but range from -1 to 1 over each axis of the focal plane.
// uvs range from 0 to 1 in each axis of the relevant plane.
// Note that the lens can have a non 1:1 aspect ratio.

Vec2f
SpotLight::lensToFocal(const Vec2f &lensCoords, const Vec3f &localP) const
{
    float f  = mFocalDistance;
    float z  = localP.z;
    const Vec2f &xy = asVec2(localP);

    // Avoid zero denominator
    if (z == 0.0f) z = 1.0e-20f;

    return ((z-f) * lensCoords + f * xy) / z;
}

Vec2f
SpotLight::focalToLens(const Vec2f &focalCoords, const Vec3f &localP) const
{
    float f  = mFocalDistance;
    float z  = localP.z;
    const Vec2f &xy = asVec2(localP);

    // Avoid zero denominator
    float denom = z - f;
    if (denom == 0.0f) denom = 1.0e-20f;

    return (z * focalCoords - f * xy) / denom;
}

Vec2f
SpotLight::getNormalizedLensCoords(const Vec2f &lensCoords) const
{
    return Vec2f(lensCoords.x * mRcpLensRadius,
                 lensCoords.y * mRcpLensRadiusY);
}

Vec2f
SpotLight::getNormalizedFocalCoords(const Vec2f &focalCoords) const
{
    return Vec2f(focalCoords.x * mRcpFocalRadius,
                 focalCoords.y * mRcpFocalRadiusY);
}

Vec2f
SpotLight::getLensCoords(const Vec2f &normalizedLensCoords) const
{
    return Vec2f(normalizedLensCoords.x * mLensRadius,
                 normalizedLensCoords.y * mLensRadiusY);
}

Vec2f
SpotLight::getFocalCoords(const Vec2f &normalizedFocalCoords) const
{
    return Vec2f(normalizedFocalCoords.x * mFocalRadius,
                 normalizedFocalCoords.y * mFocalRadiusY);
}

Vec2f
SpotLight::getUvsFromNormalized(const Vec2f& normalizedCoords) const
{
    return Vec2f( 0.5f * normalizedCoords.x + 0.5f,
                 -0.5f * normalizedCoords.y + 0.5f);
}

bool
SpotLight::intersect(const Vec3f &p, const Vec3f *n, const Vec3f &wi, float time,
        float maxDistance, LightIntersection &isect) const
{
    // Transform wi into local space.
    Vec3f localWi = xformVectorRender2LocalRot(wi, time);

    // Check that ray isn't traveling away from the light's surface.
    if (localWi.z >= 0.0f) {
        return false;
    }

    // Transform p into local space.
    Vec3f localP = xformVectorRender2Local(p - getPosition(time), time);

    // Check that ray origin is in front of light
    if (localP.z <= 0.0f) {
        return false;
    }

    // Get distance along ray to intersection with lens plane
    float localDistance = -localP.z / localWi.z;
    float renderDistance = xformLocal2RenderScale(localDistance, time);
    if (renderDistance > maxDistance) {
        return false;
    }

    // Compute and test lens intersection
    Vec2f lensCoords = asVec2(localP + localDistance * localWi);
    if (lengthSqr(getNormalizedLensCoords(lensCoords)) > 1.0f) {
        return false;
    }

    isect.N = getDirection(time);
    isect.distance = renderDistance;

    // Map lens pos to focal plane for uvs
    Vec2f focalCoords = lensToFocal(lensCoords, localP);
    Vec2f normalizedFocalCoords = getNormalizedFocalCoords(focalCoords);
    isect.uv = getUvsFromNormalized(normalizedFocalCoords);

    return true;
}

// TODO: sample uniformly over the intersection between the 2 disks (lens, focal plane)
// TODO: put back the texture importance sampling and combine results using MIS
bool
SpotLight::sample(const Vec3f &p, const Vec3f *n, float time, const Vec3f& r,
                  Vec3f &wi, LightIntersection &isect, float rayDirFootprint) const
{
    MNRY_ASSERT(mOn);

    Vec3f localP = xformPointRender2Local(p, time);
    if (localP.z <= 0.0f) {
        return false;
    }

    // Both sampling strategies need a sample point on a disk
    Vec2f diskSample;
    squareSampleToCircle(r[0], r[1], &diskSample.x, &diskSample.y);

    // We'll compute the sample position in the lens' local frame
    Vec2f lensCoords;

    // Choose a sampling strategy based on which disk subtends a larger solid angle:
    // the lens, or the focal plane. Each strategy alone suffers from a region of extreme noise,
    // but when combined, each covers the problem region of the other.
    if (localP.z > mCrossOverDistance) {
        // Sample point on lens
        lensCoords = getLensCoords(diskSample);

        // Map lens pos to focal plane for uvs
        Vec2f focalCoords = lensToFocal(lensCoords, localP);
        Vec2f normalizedFocalCoords = getNormalizedFocalCoords(focalCoords);
        isect.uv = getUvsFromNormalized(normalizedFocalCoords);
    } else {
        // Sample point on focal plane
        Vec2f focalCoords = getFocalCoords(diskSample);

        // Map to lens pos
        lensCoords = focalToLens(focalCoords, localP);

        // Quit if projected point is outside lens
        if (lengthSqr(getNormalizedLensCoords(lensCoords)) > 1.0f) {
            return false;
        }

        isect.uv = getUvsFromNormalized(diskSample);
    }

    Vec3f lensPosRender = xformPointLocal2Render(Vec3f(lensCoords.x, lensCoords.y, 0.0f), time);
    wi = lensPosRender - p;

    if (n  &&  dot(*n, wi) <= 0.0f) {
        return false;
    }

    isect.distance = length(wi);
    isect.N = getDirection(time);

    wi /= isect.distance;

    return true;
}


Color
SpotLight::eval(mcrt_common::ThreadLocalState* tls, const Vec3f &wi, const Vec3f &p, const LightFilterRandomValues& filterR, float time,
        const LightIntersection &isect, bool fromCamera, const LightFilterList *lightFilterList, float rayDirFootprint,
        float *pdf) const
{
    MNRY_ASSERT(mOn);

    // Apply texture if present
    Color radiance = mDistribution ? mDistribution->eval(isect.uv.x, isect.uv.y, 0, mTextureFilter) : sWhite;

    // Apply spotlight falloff function at intersection with focal plane
    Vec2f normalizedFocalCoords = 2.0f * isect.uv - Vec2f(1.0f, 1.0f);
    float r = length(normalizedFocalCoords);
    float falloffParam = (1.0f - r) * mFalloffGradient;
    float falloff = mFalloffCurve.eval(falloffParam);
    if (fromCamera) {
        radiance = lerp(Color(mBlackLevel), radiance, falloff);
    } else {
        radiance *= falloff;
    }

    // Apply light color
    radiance *= mRadiance;

    // Apply light filter if present
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
        float scale = areaToSolidAngleScale(wi, isect.N, isect.distance);
        Vec3f localP  = xformPointRender2Local(p, time);
        if (localP.z > mCrossOverDistance) {
            // Compute pdf for sampling/intersecting the lens
            *pdf = mInvArea * scale;
        } else {
            // Compute pdf for sampling/intersecting the focal plane
            float dz = mFocalDistance - localP.z;
            float zRatio = dz / localP.z;
            *pdf = mFocalPlanePdfConst * scale * zRatio * zRatio;
        }
    }

    return radiance;
}

Vec3f
SpotLight::getEquiAngularPivot(const Vec3f& r, float time) const
{
    // Sampling the image distribution has been removed here,
    // since spotlight textures are no longer applied to the lens.
    // Instead we generate a uniformly distributed point on the lens.
    Vec2f diskSample;
    squareSampleToCircle(r[0], r[1], &diskSample.x, &diskSample.y);
    Vec2f lensCoords = getLensCoords(diskSample);
    return xformPointLocal2Render(Vec3f(lensCoords.x, lensCoords.y, 0.0f), time);
}

void
SpotLight::initAttributeKeys(const scene_rdl2::rdl2::SceneClass &sc)
{
    if (sAttributeKeyInitialized) {
        return;
    }

    MOONRAY_START_NON_THREADSAFE_STATIC_WRITE

    sAttributeKeyInitialized = true;

    sNormalizedKey         = sc.getAttributeKey<scene_rdl2::rdl2::Bool> ("normalized");
    sApplySceneScaleKey    = sc.getAttributeKey<scene_rdl2::rdl2::Bool> ("apply_scene_scale");
    sLensRadiusKey         = sc.getAttributeKey<scene_rdl2::rdl2::Float>("lens_radius");
    sAspectRatioKey        = sc.getAttributeKey<scene_rdl2::rdl2::Float>("aspect_ratio");
    sFocalPlaneDistanceKey = sc.getAttributeKey<scene_rdl2::rdl2::Float>("focal_plane_distance");
    sOuterConeAngleKey     = sc.getAttributeKey<scene_rdl2::rdl2::Float>("outer_cone_angle");
    sInnerConeAngleKey     = sc.getAttributeKey<scene_rdl2::rdl2::Float>("inner_cone_angle");
    sAngleFalloffTypeKey   = sc.getAttributeKey<scene_rdl2::rdl2::Int>  ("angle_falloff_type");
    sBlackLevelKey         = sc.getAttributeKey<scene_rdl2::rdl2::Float>("black_level");
    INIT_ATTR_KEYS_CLEAR_RADIUS

    MOONRAY_FINISH_NON_THREADSAFE_STATIC_WRITE
}


//----------------------------------------------------------------------------

} // namespace pbr
} // namespace moonray

