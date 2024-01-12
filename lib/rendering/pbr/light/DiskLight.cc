// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0


#include "DiskLight.h"
#include <moonray/rendering/pbr/core/Distribution.h>
#include <moonray/rendering/pbr/core/Util.h>

#include <moonray/rendering/pbr/light/DiskLight_ispc_stubs.h>

#include <moonray/common/mcrt_macros/moonray_static_check.h>
#include <scene_rdl2/scene/rdl2/rdl2.h>

using namespace scene_rdl2;
using namespace scene_rdl2::math;
using scene_rdl2::logging::Logger;

namespace moonray {
namespace pbr {

bool                                                     DiskLight::sAttributeKeyInitialized;
scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Bool>   DiskLight::sNormalizedKey;
scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Bool>   DiskLight::sApplySceneScaleKey;
scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Float>  DiskLight::sRadiusKey;
scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Float>  DiskLight::sSpreadKey;
scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Int>    DiskLight::sSidednessKey;
scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Float>  DiskLight::sClearRadiusKey;
scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Float>  DiskLight::sClearRadiusFalloffDistanceKey;
scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Int>    DiskLight::sClearRadiusInterpolationKey;

//----------------------------------------------------------------------------

HUD_VALIDATOR(DiskLight);

float
DiskLight::planeDistance(const Vec3f &p, float time) const
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

void
DiskLight::getSpreadSquare(const Vec3f& renderP, const float renderDistance, const float time,
        Vec3f& center, float& halfSideLength) const
{
    // get center and half length of square region of influence
    center = xformPointRender2Local(renderP, time);
    float renderSpaceSpreadLength = renderDistance * mTanSpreadTheta;
    halfSideLength = xformRender2LocalScale(renderSpaceSpreadLength, time);
}

// Get rectangular bound of region of overlap between DiskLight and square region of influence.
// The square is centered at localP and has a side length 2 * localLength.
// We want to get the center of the rectangular bound and its width and height.
// Returns true if the region overlaps with the light.
bool
getOverlapBounds(const Vec2f& localP, float localLength, Vec2f& center, float& width, float& height)
{
    float x0 = localP.x - localLength;
    float x1 = localP.x + localLength;
    float y0 = localP.y - localLength;
    float y1 = localP.y + localLength;

    float xc = math::clamp(0.f, x0, x1);
    float yc = math::clamp(0.f, y0, y1);

    // square is outside circle. Local radius of DiskLight is 1.
    if (xc * xc + yc * yc >= 1.f) {
        return false;
    }

    float xi = math::sqrt(1 - yc * yc);
    float yi = math::sqrt(1 - xc * xc);
    float minX = math::max(-xi, x0);
    float maxX = math::min( xi, x1);
    float minY = math::max(-yi, y0);
    float maxY = math::min( yi, y1);

    center.x = 0.5f * (maxX + minX);
    center.y = 0.5f * (maxY + minY);
    width = maxX - minX;
    height = maxY - minY;

    return true;
}

bool lightIsInsideSpread(const Vec3f& localP, const float localLength)
{
    // Is DiskLight entirely inside square centered at localP with side length 2 * localLength?
    return localP.x - localLength < -1.f &&
           localP.x + localLength > 1.f &&
           localP.y - localLength < -1.f &&
           localP.y + localLength > 1.f;
}


DiskLight::DiskLight(const scene_rdl2::rdl2::Light* rdlLight) :
    LocalParamLight(rdlLight),
    mDistributionPdfScale(1.0f),
    mRenderPlane()
{
    initAttributeKeys(rdlLight->getSceneClass());

    ispc::DiskLight_init(this->asIspc());
}

DiskLight::~DiskLight() { }

bool
DiskLight::update(const Mat4d& world2render)
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
    // that the disk radius is always 1 in local light space.
    const float radius = mRdlLight->get<scene_rdl2::rdl2::Float>(sRadiusKey);

    // mSpread: 1 = 90 degree cone angle, 0 = 0 degree cone angle (cross section is square).
    mSpread = math::clamp(mRdlLight->get<scene_rdl2::rdl2::Float>(sSpreadKey), math::sEpsilon, 1.f);
    mTanSpreadTheta = math::tan(mSpread * math::sHalfPi);

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
    const float renderRadiusSqr = mLocal2RenderScale[0] * mLocal2RenderScale[0];
    mArea = sPi * renderRadiusSqr;
    mInvArea = 1.0f / mArea;

    // This term is used to adjust the pdf of sampling a point on the light in
    // the case where we are using an ImageDistribution to handle the
    // importance sampling of the image. In that case, the distribution class
    // is setup with a Distribution2D::CIRCULAR mapping (see below), which
    // zeroes-out all weights of the distribution outside of the disc inscribed
    // within the [0, 1)^2 domain. The class also automatically normalizes the
    // weights, as it should. In sample() we actually sample the square, not
    // the disk and let the ImageDistribution class handle the rest. The pdf
    // is therefore = distribution_pdf / square_area
    mDistributionPdfScale = 1.0f / (4.0f * renderRadiusSqr);

    mRadiance = computeLightRadiance(mRdlLight, scene_rdl2::rdl2::Light::sColorKey,
        scene_rdl2::rdl2::Light::sIntensityKey, scene_rdl2::rdl2::Light::sExposureKey, sNormalizedKey,
        sApplySceneScaleKey, mInvArea);
    if (isBlack(mRadiance)) {
        mOn = false;
        return false;
    }

    // cache plane at rayTime == 0
    mRenderPlane = Plane(getPosition(0.f), getDirection(0.f));

    if (!updateImageMap(Distribution2D::CIRCULAR)) {
        return false;
    }

    // Sidedness
    mSidedness = static_cast<LightSidednessType>(mRdlLight->get<scene_rdl2::rdl2::Int>(sSidednessKey));
    return true;
}

//----------------------------------------------------------------------------

bool
DiskLight::canIlluminate(const Vec3f p, const Vec3f *n, float time, float radius,
    const LightFilterList* lightFilterList) const
{
    MNRY_ASSERT(mOn);

    // Cull points which are completely on the backside of the light, taking sidedness into account
    const float threshold = sEpsilon - radius;
    if ((mSidedness != LightSidednessType::LIGHT_SIDEDNESS_2_SIDED) && (planeDistance(p, time) < threshold)) {
        return false;
    }

    // Cull lights which are completely on the backside of the point.
    // See image at lib/rendering/pbr/doc/DiskLightDiagram.jpg for a 2D breakdown
    // of what's going on here.
    // We make used of the fact that the point normal, light normal, and
    // closest point on the disk to the plane of the point all lie on the
    // same plane.
    if (n) {

        // Transform normal from render space into light space.
        Vec3f localN = xformNormalRender2LocalRot(*n, time);

        // Project onto local xy plane in light local space. The disk light
        // itself is a unit circle on the xy plane, so this will collapse
        // the point onto that unit circle.
        localN.z = 0.0f;

        // By normalizing the 2D projected point, we are putting it on the
        // circumference of the disk. If diskP couldn't be normalized
        // it means that the disk light is parallel to the cull plane of point
        // p, in which case any point on the disk is as good as any other for the
        // final plane check, so we just pick (1, 0).
        Vec2f diskP = asVec2(localN);
        float rcpLength = rsqrt(dot(diskP, diskP));
        diskP = (rcpLength < float(one_over_epsilon) ?  diskP * rcpLength  :  Vec2f(1.f, 0.f));

        // diskP will now be the point on the unit circle which is furthest in the
        // direction of n, transform it back into render space to do the final plane
        // check.

        Vec3f renderP = xformPointLocal2Render(Vec3f(diskP.x, diskP.y, 0.f), time);
        bool canIllum = dot(*n, renderP - p) > threshold;
        if (!canIllum) return false;
    }

    if (lightFilterList) {
        return canIlluminateLightFilterList(lightFilterList,
            { getPosition(time),
              xformLocal2RenderScale(1.0f, time),
              p, getXformRender2Local(time, lightFilterList->needsLightXform()),
              radius, time
            });
    }

    return true;
}

bool
DiskLight::isBounded() const
{
    return true;
}

bool
DiskLight::isDistant() const
{
    return false;
}

bool
DiskLight::isEnv() const
{
    return false;
}

// Computing a tight bounding box for a disk light
// -----------------------------------------------
//
// In local space, a point (x,y,z) on the perimeter of the disk can be expressed parametrically as
//
//      x = cos t
//      y = sin t
//      z = 0
//
// with t ranging from 0 to 2pi.
// Note that this is a unit circle, because the disk's radius is baked into the local2render matrix.
// Suppose the local2render matrix is (Mij) with i=0,1,2,3 and j=0,1,2,3.
// Then the perimeter point (x,y,z) transforms to (x',y',z') in render space, where
//
//      x' = M00 cos t + M10 sin t + M30
//      y' = M01 cos t + M11 sin t + M31
//      z' = M02 cos t + M12 sin t + M32
//
// Consider just the x' coordinate. Its min and max occur at points where dx'/dt=0. Differentiating the above gives
//
//      dx'/dt = -M00 sin t + M10 cos t
//
// If this equals zero then
//
//      tan t = M10 / M00
//
// Or, using standard trig identities,
//
//      cos^2 t = M00^2 / (M00^2 + M10^2)
//      sin^2 t = M10^2 / (M00^2 + M10^2)
//
// Taking square roots, and ensuring the chosen combination of signs preserves the sign of tan t,
//
//      (cos t, sin t) = +/- (M00, M10) / sqrt(M00^2 + M10^2)
//
// Putting these values into the expression for x' gives
//
//      x' = M30 +/- (M00^2 + M10^2) / sqrt(M00^2 + M10^2)
//         = M30 +/- sqrt(M00^2 + M10^2)
//
// The expressions for the min & max y' and z' are similar, and the 3 together can be expressed compactly
// in terms of Vec3f operations.

BBox3f
DiskLight::getBounds() const
{
    const Vec3f pos = getPosition(0.f);
    const Vec3f xRow = xformPointLocal2Render(Vec3f(1., 0., 0.), 0.f);
    const Vec3f yRow = xformPointLocal2Render(Vec3f(0., 1., 0.), 0.f);
    const Vec3f halfDims = max(sqrt(xRow*xRow + yRow*yRow), Vec3f(sEpsilon));  // ensure non-degeneracy
    BBox3f bounds(pos - halfDims, pos + halfDims);

    if (isMb()) {
        const Vec3f pos = getPosition(1.f);
        const Vec3f xRow = xformPointLocal2Render(Vec3f(1., 0., 0.), 1.f);
        const Vec3f yRow = xformPointLocal2Render(Vec3f(0., 1., 0.), 1.f);
        const Vec3f halfDims = max(sqrt(xRow*xRow + yRow*yRow), Vec3f(sEpsilon));  // ensure non-degeneracy
        bounds.extend(BBox3f(pos - halfDims, pos + halfDims));
    }

    return bounds;
}

bool
DiskLight::intersect(const Vec3f &p, const Vec3f *n, const Vec3f &wi, float time,
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

    // Reject if intersection is outside disk
    Vec3f localHit = localP + localWi * localDistance;
    if (asVec2(localHit).lengthSqr() > 1.0f) {
        return false;
    }

    if (mSpread < 1.f) {
        // Reject if intersection is outside spread
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

    // Fill in isect members
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
DiskLight::sample(const Vec3f &p, const Vec3f *n, float time, const Vec3f& r,
        Vec3f &wi, LightIntersection &isect, float rayDirFootprint) const
{
    MNRY_ASSERT(mOn);

    float r1 = r[0];
    float r2 = r[1];

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
        // Sample the image distribution if any

        // This type of technique is bad for for vector processors in general
        // since some lanes will go idle as the rejection loop still needs to
        // iterate for other lanes. It should work fine here though because the
        // distribution has been prebuilt with a low probability of sampling
        // outside of the inscribed circle by using ImageDistribution::CIRCULAR.
        // This should ensure we only loop rarely.
        const unsigned maxTries = 8;
        unsigned attempts = 0;
        while(attempts != maxTries) {
            mDistribution->sample(r1, r2, 0, &isect.uv, nullptr, mTextureFilter);
            float x = isect.uv.x * 2.0f - 1.0f;
            float y = isect.uv.y * 2.0f - 1.0f;
            if (x*x + y*y < 1.0f) {
                break;
            }

            // Generate new random numbers based on these ones.
            // TODO: this is hacky code, do a better job of generating
            //       further random numbers
            r1 = scene_rdl2::math::fmod(r1 + 0.789f, 1.0f);
            r2 = scene_rdl2::math::fmod(r2 + 0.331f, 1.0f);

            attempts++;
        }

        if (attempts == maxTries) {
            return false;
        }

        localHit = Vec3f((isect.uv.x - mUvOffset.x) / mUvScale.x,
                         (isect.uv.y - mUvOffset.y) / mUvScale.y,
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
        // Setup parameters of sampling the spread.
        float spreadLength;
        Vec3f localP;
        getSpreadSquare(p, distance, time, localP, spreadLength);

        if (!lightIsInsideSpread(localP, spreadLength)) {
            // Uniformly sample the rectangular overlapping region between the spread region and the DiskLight.
            Vec2f center;
            float width, height;
            if (!getOverlapBounds(Vec2f(localP.x, localP.y), spreadLength, center, width, height)) {
                return false;
            }

            // sample rectangular region of influence
            localHit = Vec3f((0.5f - r1) * width + center.x,
                             (0.5f - r2) * height + center.y,
                             0.0f);

            // reject samples that are outside light
            if (asVec2(localHit).lengthSqr() > 1.f) {
                return false;
            }
            isect.uv = zero;
            // store pdf
            isect.pdf = 1.f / (xformLocal2RenderScale(width, time) * xformLocal2RenderScale(height, time));
        } else {
            // Otherwise uniformly distribute samples on disk area
            isect.uv = zero;
            squareSampleToCircle(r1, r2, &localHit.x, &localHit.y);
            localHit.z = 0.0f;
            isect.pdf = mInvArea;
        }

    } else {
        // Otherwise uniformly distribute samples on disk area
        isect.uv = zero;
        squareSampleToCircle(r1, r2, &localHit.x, &localHit.y);
        localHit.z = 0.0f;
    }

    Vec3f renderHit = xformPointLocal2Render(localHit, time);

    // Compute wi and d
    wi = renderHit - p;
    if (n  &&  dot(*n, wi) < sEpsilon) {
        return false;
    }
    isect.distance = length(wi);
    if (isect.distance < sEpsilon) {
        return false;
    }
    wi *= rcp(isect.distance);

    // We will only reach this far if p is on the illuminated side of the light's plane (either side, for 2-sided).
    // The light's mDirection is its local z-axis, which is only guaranteed to concide with the normal for lights
    // with regular sidedness. The correct normal is obtained by ensuring the ray is travelling against the normal,
    // i.e. the ray's direction has negative dot product with it.
    Vec3f normal = getDirection(time);
    isect.N = (dot(normal, wi) < 0.0f) ? normal : -normal;

    return true;
}

Color
DiskLight::eval(mcrt_common::ThreadLocalState* tls, const Vec3f &wi, const Vec3f &p, const LightFilterRandomValues& filterR, float time,
        const LightIntersection &isect, bool fromCamera, const LightFilterList *lightFilterList, float rayDirFootprint,
        float *pdf) const
{
    MNRY_ASSERT(mOn);

    // Point sample the texture
    // TODO: Use proper filtering with ray differentials and mip-mapping.
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

    if (pdf) {
        if (mDistribution) {
            // we sampled the image distribution
            *pdf = mDistribution->pdf(isect.uv[0], isect.uv[1], 0, mTextureFilter) *
                mDistributionPdfScale;
        } else if (mSpread < 1.f) {
            // grab the stored pdf.
            *pdf = isect.pdf;
        } else {
            // we uniformly sampled the disk
            *pdf = mInvArea;
        }
        *pdf *= areaToSolidAngleScale(wi, isect.N, isect.distance);
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

    return radiance;
}

Vec3f
DiskLight::getEquiAngularPivot(const Vec3f& r, float time) const
{
    Vec3f hit;
    Vec2f uv;
    float r1 = r[0];
    float r2 = r[1];
    // Sample the image distribution if any
    if (mDistribution) {
        // This type of technique is bad for for vector processors in general
        // since some lanes will go idle as the rejection loop still needs to
        // iterate for other lanes. It should work fine here though because the
        // distribution has been prebuilt with a low probability of sampling
        // outside of the inscribed circle by using ImageDistribution::CIRCULAR.
        // This should ensure we only loop rarely.
        const unsigned maxTries = 8;
        unsigned attempts = 0;
        while(attempts != maxTries) {
            mDistribution->sample(r1, r2, 0, &uv, nullptr, mTextureFilter);
            float x = uv.x * 2.0f - 1.0f;
            float y = uv.y * 2.0f - 1.0f;
            if (x*x + y*y < 1.0f) {
                break;
            }
            // Generate new random numbers based on these ones.
            // TODO: this is hacky code, do a better job of generating
            //       further random numbers
            r1 = scene_rdl2::math::fmod(r1 + 0.789f, 1.0f);
            r2 = scene_rdl2::math::fmod(r2 + 0.331f, 1.0f);
            attempts++;
        }
        if (attempts == maxTries) {
            return getPosition(time);
        }
        hit = Vec3f((uv[0] - mUvOffset.x) / mUvScale.x,
                    (uv[1] - mUvOffset.y) / mUvScale.y,
                     0.0f);
    } else {
        squareSampleToCircle(r1, r2, &hit.x, &hit.y);
        hit.z = 0.0f;
    }
    return xformPointLocal2Render(hit, time);
}

void
DiskLight::initAttributeKeys(const scene_rdl2::rdl2::SceneClass &sc)
{
    if (sAttributeKeyInitialized) {
        return;
    }

    MOONRAY_START_NON_THREADSAFE_STATIC_WRITE

    sAttributeKeyInitialized = true;

    sNormalizedKey      = sc.getAttributeKey<scene_rdl2::rdl2::Bool> ("normalized");
    sApplySceneScaleKey = sc.getAttributeKey<scene_rdl2::rdl2::Bool> ("apply_scene_scale");
    sRadiusKey          = sc.getAttributeKey<scene_rdl2::rdl2::Float>("radius");
    sSpreadKey          = sc.getAttributeKey<scene_rdl2::rdl2::Float>("spread");
    sSidednessKey       = sc.getAttributeKey<scene_rdl2::rdl2::Int>  ("sidedness");
    INIT_ATTR_KEYS_CLEAR_RADIUS

    MOONRAY_FINISH_NON_THREADSAFE_STATIC_WRITE
}

//----------------------------------------------------------------------------

} // namespace pbr
} // namespace moonray

