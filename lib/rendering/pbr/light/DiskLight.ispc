// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0


#include "Light.isph"
#include <moonray/rendering/pbr/core/Util.isph>

#include <scene_rdl2/common/platform/IspcUtil.isph>
#include <scene_rdl2/common/math/ispc/asA.isph>
#include <scene_rdl2/common/math/ispc/Math.isph>


//----------------------------------------------------------------------------

ISPC_UTIL_EXPORT_UNIFORM_STRUCT_TO_HEADER(DiskLight);

export uniform uint32_t
DiskLight_hudValidation(uniform bool verbose)
{
    DISK_LIGHT_VALIDATION;
}

//----------------------------------------------------------------------------

static varying float
planeDistance(const uniform DiskLight * uniform light,
              const varying Vec3f &p, varying float time)
{
    const uniform Light * uniform li = (const uniform Light * uniform) light;
    Plane renderPlane;
    if (light->mMb) {
        Plane_init(&renderPlane, Light_getPosition(li, time), Light_getDirection(li, time));
    } else {
        renderPlane = light->mRenderPlane;
    }

    float distance = Plane_getDistance(&renderPlane, p);
    if (light->mSidedness == LIGHT_SIDEDNESS_REVERSE) {
        distance = -distance;
    } else if (light->mSidedness == LIGHT_SIDEDNESS_2_SIDED) {
        distance = abs(distance);
    }

    return distance;
}

// Inputs: render space shading point and distance to light.
// Outputs: center and half side length of valid square region of light influence if spread < 1.
void getSpreadSquare(const uniform Light * uniform li,
        const varying Vec3f& renderP, const varying float renderDistance, const varying float time,
        Vec3f& center, float& halfSideLength)
{
    const uniform LocalParamLight * uniform lpl = (const uniform LocalParamLight * uniform)li;
    const uniform DiskLight * uniform light = (const uniform DiskLight * uniform)li;
    // get center and half length of square region of influence
    center = LocalParamLight_xformPointRender2Local(lpl, renderP, time);
    float renderSpaceSpreadLength = renderDistance * light->mTanSpreadTheta;
    halfSideLength = LocalParamLight_xformRender2LocalScale(lpl, renderSpaceSpreadLength, time);
}

// Get rectangular bound of region of overlap between DiskLight and square region of influence.
// The square is centered at localP and has a side length 2 * localLength. 
// We want to get the center of the rectangular bound and its width and height.
// Returns true if the region overlaps with the light.
varying bool 
getOverlapBounds(const varying Vec2f& localP, varying float localLength, 
        varying Vec2f& center, varying float& width, varying float& height)
{
    float x0 = localP.x - localLength;
    float x1 = localP.x + localLength;
    float y0 = localP.y - localLength;
    float y1 = localP.y + localLength;

    float xc = clamp(0.f, x0, x1);
    float yc = clamp(0.f, y0, y1);

    // square is outside circle. Local radius of DiskLight is 1.
    if (xc * xc + yc * yc >= 1.f) {
        return false;
    }

    float xi = sqrt(1 - yc * yc);
    float yi = sqrt(1 - xc * xc);
    float minX = max(-xi, x0);
    float maxX = min( xi, x1);
    float minY = max(-yi, y0);
    float maxY = min( yi, y1);

    center.x = 0.5f * (maxX + minX);
    center.y = 0.5f * (maxY + minY);
    width = maxX - minX;
    height = maxY - minY;

    return true;
}

varying bool 
lightIsInsideSpread(const varying Vec3f& localP, const varying float localLength) 
{
    // Is DiskLight entirely inside square centered at localP with side length 2 * localLength?
    return localP.x - localLength < -1.f &&
           localP.x + localLength > 1.f &&
           localP.y - localLength < -1.f &&
           localP.y + localLength > 1.f;
}

varying bool
DiskLight_canIlluminate(const uniform Light * uniform li,
        const varying Vec3f &p, const varying Vec3f &cullingNormal, varying float time,
        varying float radius, const uniform LightFilterList * uniform lightFilterList)
{
    const uniform LocalParamLight * uniform lpl = (const uniform LocalParamLight * uniform)li;
    const uniform DiskLight * uniform light = (const uniform DiskLight * uniform)li;

    MNRY_ASSERT(li->mOn);

    // Cull points which are completely on the backside of the light, taking sidedness into account
    const float threshold = sEpsilon - radius;
    if ((light->mSidedness != LIGHT_SIDEDNESS_2_SIDED) && (planeDistance(light, p, time) < threshold)) {
        return false;
    }

    // Cull lights which are completely on the backside of the point.
    // See image at lib/rendering/pbr/doc/DiskLightDiagram.jpg for a 2D breakdown
    // of what's going on here.
    // We make used of the fact that the point normal, light normal, and
    // closest point on the disk to the plane of the point all lie on the
    // same plane.
    if (isValidCullingNormal(cullingNormal)) {

        // Transform normal from render space into light space.
        Vec3f localN = LocalParamLight_xformNormalRender2LocalRot(lpl, cullingNormal, time);

        // Project onto local xy plane in light local space. The disk light
        // itself is a unit circle on the xy plane, so this will collapse
        // the point onto that unit circle.
        localN.z = 0.0f;

        // By normalizing the 2D projected point, we are putting it on the
        // circumference of the disk. If diskP couldn't be normalized
        // it means that the disk light is parallel to the cull plane of point
        // p, in which case any point on the disk is as good as any other for the
        // final plane check, so we just pick (1, 0).
        Vec2f diskP = asAVec2f(localN);
        float rcpLength = rsqrt(lengthSqr(diskP));
        if (rcpLength < sOneOverEpsilon) {
            diskP = diskP * rcpLength;
        } else {
            diskP = Vec2f_ctor(1.0f, 0.0f);
        }

        // diskP will now be the point on the unit circle which is furthest in the
        // direction of n, transform it back into render space to do the final plane
        // check.

        Vec3f renderP = LocalParamLight_xformPointLocal2Render(lpl, Vec3f_ctor(diskP.x, diskP.y, 0.f), time);
        bool canIllum = dot(cullingNormal, renderP - p) > threshold;
        if (!canIllum) return false;
    }

    if (lightFilterList) {
        CanIlluminateData cid;
        cid.lightPosition = Light_getPosition(li, time);
        cid.lightRadius = LocalParamLight_xformLocal2RenderScale(lpl, 1.0f, time);
        cid.shadingPointPosition = p;
        cid.lightRender2LocalXform = LocalParamLight_getXformRender2Local(
            lpl, time, LightFilterList_getNeedsLightXform(lightFilterList));
        cid.shadingPointRadius = radius;
        cid.time = time;
        return LightFilterList_canIlluminate(lightFilterList, cid);
    }

    return true;
}

varying bool
DiskLight_intersect(const uniform Light * uniform li,
        const varying Vec3f &p,  const varying Vec3f &cullingNormal,
        const varying Vec3f &wi, varying float time,
        varying float maxDistance, varying LightIntersection &isect)
{
    const uniform LocalParamLight * uniform lpl = (const uniform LocalParamLight * uniform)li;
    const uniform DiskLight * uniform light = (const uniform DiskLight * uniform)li;

    // Get local direction and position
    Vec3f localWi = LocalParamLight_xformVectorRender2LocalRot(lpl, wi, time);
    Vec3f localP = LocalParamLight_xformVectorRender2Local(lpl, p - Light_getPosition(li, time), time);

    // Trivial rejection of pos/dir/sidedness cases
    if (Light_rejectPosDirSidedness(li, localP.z, localWi.z)) return false;

    // Reject if intersection with plane is beyond maxDistance
    float localDistance = -localP.z / localWi.z;
    float renderDistance = LocalParamLight_xformLocal2RenderScale(lpl, localDistance, time);
    if (renderDistance > maxDistance) {
        return false;
    }

    // Reject if intersection is outside disk
    Vec3f localHit = localP + localWi * localDistance;
    if (lengthSqr(asAVec2f(localHit)) > 1.0f) {
        return false;
    }

    if (light->mSpread < 1.f) {
        // Reject if intersection is outside spread
        Vec3f absLocalWi = abs(localWi);
        if (max(absLocalWi.x, absLocalWi.y) > absLocalWi.z * light->mTanSpreadTheta) {
            return false;
        }

        // compute pdf
        // get half side length of square spread region
        float spreadLength = localP.z * light->mTanSpreadTheta;
        Vec2f center;
        float width, height;
        getOverlapBounds(Vec2f_ctor(localP.x, localP.y), spreadLength, center, width, height);
        // store pdf if not rejected
        isect.pdf = 1.f / (LocalParamLight_xformLocal2RenderScale(lpl, width, time) *
                           LocalParamLight_xformLocal2RenderScale(lpl, height, time));
    }

    // Fill in isect members.
    // We will only reach this far if p is on the illuminated side of the light's plane (either side, for 2-sided).
    // The light's mDirection is its local z-axis, which is only guaranteed to concide with the normal for lights
    // with regular sidedness. The correct normal is obtained by ensuring the ray is travelling against the normal,
    // i.e. the ray's direction has negative local z-coordinate.
    Vec3f normal = Light_getDirection(li, time);
    isect.N = (localWi.z < 0.0f) ? normal : neg(normal);
    isect.uv = light->mDistribution  ?  asAVec2f(localHit) * light->mUvScale + light->mUvOffset
                                      :  Vec2f_ctor(0.0f);
    isect.distance = renderDistance;

    return true;
}

varying bool
DiskLight_sample(const uniform Light * uniform li,
        const varying Vec3f &p, const varying Vec3f &cullingNormal, varying float time,
        const varying Vec3f &r, varying Vec3f &wi, varying LightIntersection &isect, varying float rayDirFootprint)
{
    const uniform LocalParamLight * uniform lpl = (const uniform LocalParamLight * uniform)li;
    const uniform DiskLight * uniform light = (const uniform DiskLight * uniform)li;

    MNRY_ASSERT(li->mOn);
    
    varying float r1 = r.x;
    varying float r2 = r.y;

    // We don't cast light on points behind the light. Also we can't always
    // rely on canIlluminate() to test for that case, as it's an optional
    // culling function. canIlluminate() can also be passed a non-zero radius,
    // defining a p-region partially in front and partially behind the light.
    // So we have to test for it per sample, here too.
    float distance = planeDistance(light, p, time);
    if ((light->mSidedness != LIGHT_SIDEDNESS_2_SIDED) && (distance < sEpsilon)) {
        return false;
    }

    Vec3f localHit;

    // There are 3 possible sampling modes:
    // 1) If there is an image distribution, sample that.
    // 2) If there is no image distribution, and the spread parameter is less than 1 and
    //    we are in a good region to sample the spread, then sample that region.
    // 3) Otherwise, sample the shape of the light.
    if (light->mDistribution) {
        // Sample the image distribution if any

        // This type of technique is bad for for vector processors in general
        // since some lanes will go idle as the rejection loop still needs to
        // iterate for other lanes. It should work fine here though because the
        // distribution has been pre-built with a low probability of sampling
        // outside of the inscribed circle by using ImageDistribution::CIRCULAR.
        // This should ensure we only loop rarely.
        const uniform uint32_t maxTries = 8;
        uniform uint32_t attempts = 0;
        while (attempts != maxTries) {
            ImageDistribution_sample(light->mDistribution, r1, r2, 0, &isect.uv, nullptr, light->mTextureFilter);
            float x = isect.uv.x * 2.0f - 1.0f;
            float y = isect.uv.y * 2.0f - 1.0f;
            if (x*x + y*y < 1.0f) {
                break;
            }

            // Generate new random numbers based on these ones.
            // TODO: this is hacky code, do a better job of generating
            //       further random numbers
            r1 = fmod(r1 + 0.789f, 1.0f);
            r2 = fmod(r2 + 0.331f, 1.0f);
            attempts++;
        }

        if (attempts == maxTries) {
            return false;
        }

        localHit = Vec3f_ctor((isect.uv.x - light->mUvOffset.x) / light->mUvScale.x,
                              (isect.uv.y - light->mUvOffset.y) / light->mUvScale.y,
                              0.0f);

        // Reject samples that are outside of the spread region.
        if (light->mSpread < 1.f) {
            Vec3f localP = LocalParamLight_xformPointRender2Local(lpl, p, time);
            Vec3f localWi = localHit - localP;
            Vec3f absLocalWi = abs(localWi);
            if (max(absLocalWi.x, absLocalWi.y) > absLocalWi.z * light->mTanSpreadTheta) {
                return false;
            }
        }

    } else if (light->mSpread < 1.f) {
        // Setup parameters of sampling the spread.
        float spreadLength;
        Vec3f localP;
        getSpreadSquare(li, p, distance, time, localP, spreadLength);

        if (!lightIsInsideSpread(localP, spreadLength)) {
            // Uniformly sample the rectangular overlapping region between the spread region and the DiskLight.
            Vec2f center;
            float width, height;
            if (!getOverlapBounds(Vec2f_ctor(localP.x, localP.y), spreadLength, center, width, height)) {
                return false;
            }

            // sample rectangular region of influence
            localHit = Vec3f_ctor((0.5f - r1) * width + center.x,
                                  (0.5f - r2) * height + center.y,
                                  0.0f);

            // reject samples that are outside light
            if (lengthSqr(asAVec2f(localHit)) > 1.f) {
                return false;
            }
            isect.uv = Vec2f_ctor(0.0f);
            // store pdf
            isect.pdf = 1.f / (LocalParamLight_xformLocal2RenderScale(lpl, width, time) * 
                               LocalParamLight_xformLocal2RenderScale(lpl, height, time));
        } else {
            // Otherwise uniformly distribute samples on disk area
            isect.uv = Vec2f_ctor(0.0f);
            squareSampleToCircle(r1, r2, &localHit.x, &localHit.y);
            localHit.z = 0.0f;
            isect.pdf = light->mInvArea;
        }

    } else {
        // Otherwise uniformly distribute samples on disk area
        isect.uv = Vec2f_ctor(0.0f);
        squareSampleToCircle(r1, r2, &localHit.x, &localHit.y);
        localHit.z = 0.0f;
    }

    Vec3f renderHit = LocalParamLight_xformPointLocal2Render(lpl, localHit, time);

    // Compute wi and d
    wi = renderHit - p;
    if (isValidCullingNormal(cullingNormal)  &&  dot(cullingNormal, wi) < sEpsilon) {
        return false;
    }
    isect.distance = length(wi);
    if (isect.distance < sEpsilon) {
        return false;
    }
    wi = wi * rcp(isect.distance);

    // We will only reach this far if p is on the illuminated side of the light's plane (either side, for 2-sided).
    // The light's mDirection is its local z-axis, which is only guaranteed to concide with the normal for lights
    // with regular sidedness. The correct normal is obtained by ensuring the ray is travelling against the normal,
    // i.e. the ray's direction has negative dot product with it.
    Vec3f normal = Light_getDirection(li, time);
    isect.N = (dot(normal, wi) < 0.0f) ? normal : neg(normal);

    return true;
}

varying Color
DiskLight_eval(const uniform Light * uniform li, uniform ShadingTLState * uniform tls,
        const varying Vec3f &wi, const varying Vec3f &p, const varying LightFilterRandomValues& filterR, 
        varying float time, const varying LightIntersection &isect, uniform bool fromCamera,
        const uniform LightFilterList * uniform lightFilterList, varying float rayDirFootprint,
        varying float * uniform pdf)
{
    const uniform DiskLight * uniform light = (const uniform DiskLight * uniform)li;
    const uniform LocalParamLight * uniform lpl = (const uniform LocalParamLight * uniform)li;

    MNRY_ASSERT(light->mOn);

    Color radiance = light->mRadiance;
    
    if (light->mSpread < 1.f) {
        // modify radiance as if it were physically blocked by an eggcrate light filter.
        Vec3f localWi = LocalParamLight_xformVectorRender2LocalRot(lpl, wi, time);
        radiance = radiance * max(1.f - abs(localWi.x / (localWi.z * light->mTanSpreadTheta)), 0.f) * 
                              max(1.f - abs(localWi.y / (localWi.z * light->mTanSpreadTheta)), 0.f);
    }

    if (light->mDistribution) {
        // Point sample the texture
        // TODO: Use proper filtering with ray differentials and mip-mapping.
        radiance = radiance * ImageDistribution_eval(light->mDistribution,
                isect.uv.x, isect.uv.y, 0, light->mTextureFilter);
    }

    if (lightFilterList) {
        EvalData evalData;
        evalData.tls = tls;
        evalData.isect = &isect;
        evalData.lightPosition = Light_getPosition(li, time);
        evalData.lightDirection = Light_getDirection(li, time);
        evalData.shadingPointPosition = p;
        evalData.randVar = filterR; 
        evalData.time = time;
        evalData.lightRender2LocalXform = LocalParamLight_getXformRender2Local(
            lpl, time, LightFilterList_getNeedsLightXform(lightFilterList));
        evalData.wi = wi;

        LightFilterList_eval(lightFilterList, evalData, &radiance);
    }

    if (pdf) {
        if (light->mDistribution) {
            // we sampled the image distribution
            *pdf = ImageDistribution_pdf(light->mDistribution, isect.uv.x, isect.uv.y, 0,
                       light->mTextureFilter) * light->mDistributionPdfScale;
        } else if (light->mSpread < 1.f) {
            // grab the stored pdf
            *pdf = isect.pdf;
        } else {
            // we uniformly sampled the disk
            *pdf = light->mInvArea;
        }
        *pdf *= areaToSolidAngleScale(wi, isect.N, isect.distance);
    }

    return radiance;
}

inline uniform float DiskLight_getThetaO(const uniform Light * uniform li)
{
    const uniform DiskLight * uniform light = (const uniform DiskLight * uniform)li;
    return sPi * 0.5f * light->mSpread; 
}
inline uniform float DiskLight_getThetaE(const uniform Light * uniform li) { return sPi * 0.5f; }

//----------------------------------------------------------------------------

export void
DiskLight_init(uniform DiskLight * uniform light)
{
    Light_init((uniform Light * uniform)light, DiskLight_canIlluminate,
                DiskLight_intersect, DiskLight_sample, DiskLight_eval,
                DiskLight_getThetaO, DiskLight_getThetaE);
}

//----------------------------------------------------------------------------

