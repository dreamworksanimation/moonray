// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0


#include "Light.isph"
#include <moonray/rendering/pbr/core/Util.isph>
#include <moonray/rendering/pbr/light/LightUtil.isph>

#include <scene_rdl2/common/platform/IspcUtil.isph>
#include <scene_rdl2/common/math/ispc/asA.isph>


//----------------------------------------------------------------------------

ISPC_UTIL_EXPORT_UNIFORM_STRUCT_TO_HEADER(SpotLight);

export uniform uint32_t
SpotLight_hudValidation(uniform bool verbose)
{
    SPOT_LIGHT_VALIDATION;
}


//----------------------------------------------------------------------------

static void
SpotLight_computeCorners(const uniform SpotLight * uniform light,
                         varying Vec3f * uniform corners, varying float time)
{
    const uniform LocalParamLight * uniform lpl = (const uniform LocalParamLight * uniform) light;

    corners[0] = LocalParamLight_xformPointLocal2Render(lpl, (varying Vec3f) Vec3f_ctor( light->mLensRadius,  light->mLensRadiusY, 0.f), time);
    corners[1] = LocalParamLight_xformPointLocal2Render(lpl, (varying Vec3f) Vec3f_ctor( light->mLensRadius, -light->mLensRadiusY, 0.f), time);
    corners[2] = LocalParamLight_xformPointLocal2Render(lpl, (varying Vec3f) Vec3f_ctor(-light->mLensRadius,  light->mLensRadiusY, 0.f), time);
    corners[3] = LocalParamLight_xformPointLocal2Render(lpl, (varying Vec3f) Vec3f_ctor(-light->mLensRadius, -light->mLensRadiusY, 0.f), time);
}

//----------------------------------------------------------------------------

varying bool
SpotLight_canIlluminate(const uniform Light * uniform li,
        const varying Vec3f &p, const varying Vec3f &cullingNormal, varying float time,
        varying float radius, const uniform LightFilterList * uniform lightFilterList)
{
    const uniform SpotLight * uniform light = (const uniform SpotLight * uniform)li;
    const uniform LocalParamLight * uniform lpl = (const uniform LocalParamLight * uniform) li;

    MNRY_ASSERT(li->mOn);

    // Cull lights which are completely on the back side of the point.
    const float threshold = sEpsilon - radius;
    if (isValidCullingNormal(cullingNormal)) {
        Plane pl;
        Plane_init(&pl, p, cullingNormal);
        varying float d0, d1, d2, d3;
        if (!light->mMb) {
            d0 = Plane_getDistance(&pl, light->mRenderCorners[0]);
            d1 = Plane_getDistance(&pl, light->mRenderCorners[1]);
            d2 = Plane_getDistance(&pl, light->mRenderCorners[2]);
            d3 = Plane_getDistance(&pl, light->mRenderCorners[3]);
        } else {
            varying Vec3f corners[4];
            SpotLight_computeCorners(light, corners, time);
            d0 = Plane_getDistance(&pl, corners[0]);
            d1 = Plane_getDistance(&pl, corners[1]);
            d2 = Plane_getDistance(&pl, corners[2]);
            d3 = Plane_getDistance(&pl, corners[3]);
        }
        bool canIllum = d0 > threshold || d1 > threshold ||
                        d2 > threshold || d3 > threshold;
        if (!canIllum) return false;
    }

    if (lightFilterList) {
        CanIlluminateData cid;
        cid.lightPosition = Light_getPosition(li, time);
        cid.lightRadius = LocalParamLight_xformLocal2RenderScale(lpl, max(light->mLensRadius, light->mLensRadiusY), 
            time);
        cid.shadingPointPosition = p;
        cid.lightRender2LocalXform = LocalParamLight_getXformRender2Local(
            lpl, time, LightFilterList_getNeedsLightXform(lightFilterList));
        cid.shadingPointRadius = radius;
        cid.time = time;
        return LightFilterList_canIlluminate(lightFilterList, cid);
    }

    return true;
}

 
// Helper functions for converting between the spotlight's coordinate systems:
// Lens coordinates measure a local position on the lens, in world-space distance units
// Normalized lens coordinates do the same but range from -1 to 1 over each axis of the lens
// Focal coordinates measure a local position on the focal plane, in world-space distance units
// Normalized focal coordinates do the same but range from -1 to 1 over each axis of the focal plane
// uvs range from 0 to 1 in each axis of the relevant plane

varying Vec2f
SpotLight_lensToFocal(const uniform SpotLight * uniform light,
                      const Vec2f &lensCoords, const Vec3f &localP)
{
    float f  = light->mFocalDistance;
    float z  = localP.z;
    const Vec2f &xy = asAVec2f(localP);

    // Avoid zero denominator
    if (z == 0.0f) z = 1.0e-20f;

    return ((z-f) * lensCoords + f * xy) / z;
}

varying Vec2f
SpotLight_focalToLens(const uniform SpotLight * uniform light,
                      const Vec2f &focalCoords, const Vec3f &localP)
{
    float f  = light->mFocalDistance;
    float z  = localP.z;
    const Vec2f &xy = asAVec2f(localP);

    // Avoid zero denominator
    float denom = z - f;
    if (denom == 0.0f) denom = 1.0e-20f;

    return (z * focalCoords - f * xy) / denom;
}

varying Vec2f
SpotLight_getNormalizedLensCoords(const uniform SpotLight * uniform light,
                                  const Vec2f &lensCoords)
{
    Vec2f normalizedLensCoords;
    normalizedLensCoords.x = lensCoords.x * light->mRcpLensRadius;
    normalizedLensCoords.y = lensCoords.y * light->mRcpLensRadiusY;

    return normalizedLensCoords;
}

varying Vec2f
SpotLight_getNormalizedFocalCoords(const uniform SpotLight * uniform light,
                                   const Vec2f &focalCoords)
{
    Vec2f normalizedFocalCoords;
    normalizedFocalCoords.x = focalCoords.x * light->mRcpFocalRadius;
    normalizedFocalCoords.y = focalCoords.y * light->mRcpFocalRadiusY;

    return normalizedFocalCoords;
}

varying Vec2f
SpotLight_getLensCoords(const uniform SpotLight * uniform light,
                        const Vec2f &normalizedLensCoords)
{
    Vec2f lensCoords;
    lensCoords.x = normalizedLensCoords.x * light->mLensRadius;
    lensCoords.y = normalizedLensCoords.y * light->mLensRadiusY;

    return lensCoords;
}

varying Vec2f
SpotLight_getFocalCoords(const uniform SpotLight * uniform light,
                         const Vec2f &normalizedFocalCoords)
{
    Vec2f focalCoords;
    focalCoords.x = normalizedFocalCoords.x * light->mFocalRadius;
    focalCoords.y = normalizedFocalCoords.y * light->mFocalRadiusY;

    return focalCoords;
}

varying Vec2f
SpotLight_getUvsFromNormalized(const Vec2f& normalizedCoords)
{
    Vec2f uvs;
    uvs.x =  0.5f * normalizedCoords.x + 0.5f;
    uvs.y = -0.5f * normalizedCoords.y + 0.5f;

    return uvs;
}

varying bool
SpotLight_intersect(const uniform Light * uniform li,
        const varying Vec3f &p, const varying Vec3f& cullingNormal,
        const varying Vec3f &wi, varying float time,
        varying float maxDistance, varying LightIntersection &isect)
{
    const uniform LocalParamLight * uniform lpl = (const uniform LocalParamLight * uniform)li;
    const uniform SpotLight * uniform light = (const uniform SpotLight * uniform)li;

    MNRY_ASSERT(li->mOn);

    // Transform wi into local space.
    Vec3f localWi = LocalParamLight_xformVectorRender2LocalRot(lpl, wi, time);

    // Check that ray isn't traveling away from the light's surface.
    if (localWi.z >= 0.0f) {
        return false;
    }

    // Transform P into local space.
    Vec3f localP = LocalParamLight_xformPointRender2Local(lpl, p, time);

    // Check that ray origin is in front of light
    if (localP.z <= 0.0f) {
        return false;
    }

    // Get distance along ray to intersection with lens plane
    float localDistance = -localP.z / localWi.z;
    float renderDistance = LocalParamLight_xformLocal2RenderScale(lpl, localDistance, time);
    if (renderDistance > maxDistance) {
        return false;
    }

    // Compute and test lens intersection
    Vec2f lensCoords = asAVec2f(localP + localDistance * localWi);
    if (lengthSqr(SpotLight_getNormalizedLensCoords(light, lensCoords)) > 1.0f) {
           return false;
    }

    isect.N = Light_getDirection(li, time);
    isect.distance = renderDistance;

    // Map lens pos to focal plane for uvs
    Vec2f focalCoords = SpotLight_lensToFocal(light, lensCoords, localP);
    Vec2f normalizedFocalCoords = SpotLight_getNormalizedFocalCoords(light, focalCoords);
    isect.uv = SpotLight_getUvsFromNormalized(normalizedFocalCoords);

    return true;
}

// TODO: sample uniformly over the intersection between the 2 disks (lens, focal plane)
// TODO: put back the texture importance sampling and combine results using MIS
varying bool
SpotLight_sample_uniform(const uniform Light * uniform li,
        const varying Vec3f &p, const varying Vec3f &n, varying float time,
        const varying Vec3f &r, varying Vec3f &wi, varying LightIntersection &isect, varying float rayDirFootprint)
{
    const uniform LocalParamLight * uniform lpl = (const uniform LocalParamLight * uniform)li;
    const uniform SpotLight * uniform light = (const uniform SpotLight * uniform)li;

    MNRY_ASSERT(li->mOn);

    Vec3f localP = LocalParamLight_xformPointRender2Local(lpl, p, time);
    if (localP.z <= 0.0f) {
        return false;
    }

    // Both sampling strategies need a sample point on a disk
    Vec2f diskSample;
    squareSampleToCircle(r.x, r.y, &diskSample.x, &diskSample.y);

    // We'll compute the sample position in the lens' local frame
    Vec2f lensCoords;

    // Choose a sampling strategy based on which disk subtends a larger solid angle:
    // the lens, or the focal plane. Each strategy alone suffers from a region of extreme noise,
    // but when combined, each covers the problem region of the other.
    if (localP.z > light->mCrossOverDistance) {
        // Sample point on lens
        lensCoords = SpotLight_getLensCoords(light, diskSample);

        // Map lens pos to focal plane for uvs
        Vec2f focalCoords = SpotLight_lensToFocal(light, lensCoords, localP);
        Vec2f normalizedFocalCoords = SpotLight_getNormalizedFocalCoords(light, focalCoords);
        isect.uv = SpotLight_getUvsFromNormalized(normalizedFocalCoords);
    } else {

        // Sample point on focal plane
        Vec2f focalCoords = SpotLight_getFocalCoords(light, diskSample);

        // Map to lens pos
        lensCoords = SpotLight_focalToLens(light, focalCoords, localP);

        // Quit if projected point is outside lens
        if (lengthSqr(SpotLight_getNormalizedLensCoords(light, lensCoords)) > 1.0f) {
            return false;
        }

        isect.uv = SpotLight_getUvsFromNormalized(diskSample);
    }
 
    Vec3f lensPosLocal = {lensCoords.x, lensCoords.y, 0.0f};
    Vec3f lensPosRender = LocalParamLight_xformPointLocal2Render(lpl, lensPosLocal, time);
    wi = lensPosRender - p;
 
    if (isValidCullingNormal(n)  &&  dot(n, wi) <= 0.0f) {
        return false;
    }
    
    isect.distance = length(wi);
    isect.N = Light_getDirection(li, time);

    wi = wi / isect.distance;

    return true;
}

varying Color
SpotLight_eval_uniform(const uniform Light * uniform li, uniform ShadingTLState * uniform tls,
        const varying Vec3f &wi, const varying Vec3f &p, const varying LightFilterRandomValues& filterR, varying float time,
        const varying LightIntersection &isect, uniform bool fromCamera,
        const uniform LightFilterList * uniform lightFilterList, varying float rayDirFootprint,
        varying float * uniform pdf)
{
    const uniform LocalParamLight * uniform lpl = (const uniform LocalParamLight * uniform)li;
    const uniform SpotLight * uniform light = (const uniform SpotLight * uniform)li;

    MNRY_ASSERT(light->mOn);

    // Apply texture if present
    Color radiance = light->mDistribution ? ImageDistribution_eval(light->mDistribution,
                                                isect.uv.x, isect.uv.y, 0, light->mTextureFilter)
                                          : sWhite;
    
    // Apply spotlight falloff function at intersection with focal plane
    Vec2f normalizedFocalCoords;
    normalizedFocalCoords.x = 2.0f * isect.uv.x - 1.0f;
    normalizedFocalCoords.y = 2.0f * isect.uv.y - 1.0f;
    float r = length(normalizedFocalCoords);
    float falloffParam = (1.0f - r) * light->mFalloffGradient;
    float falloff = FalloffCurve_eval(&light->mFalloffCurve, falloffParam);
    if (fromCamera) {
        radiance = lerp(Color_ctor(light->mBlackLevel), radiance, falloff);
    } else {
        radiance = radiance * falloff;
    }

    // Apply light color
    radiance = radiance * light->mRadiance;

    // Apply light filter if present

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
        float scale = areaToSolidAngleScale(wi, isect.N, isect.distance);
        Vec3f localP  = LocalParamLight_xformPointRender2Local(lpl, p, time);
        if (localP.z > light->mCrossOverDistance) {
            // Compute pdf for sampling/intersecting the lens
            *pdf = light->mInvArea * scale;
        } else {
            // Compute pdf for sampling/intersecting the focal plane
            float dz = light->mFocalDistance - localP.z;
            float zRatio = dz / localP.z;
            *pdf = light->mFocalPlanePdfConst * scale * zRatio * zRatio;
        }
    }

    return radiance;
}

inline uniform float SpotLight_getThetaO(const uniform Light * uniform li) { return 0.f; }
inline uniform float SpotLight_getThetaE(const uniform Light * uniform li) {
    const uniform SpotLight * uniform light = (const uniform SpotLight * uniform)li;
    // find outer angle
    // if lens is elliptical, find larger outer angle
    uniform float aspectRatio = max(light->mRcpAspectRatio, 1.f); 
    // outer angle = atan( aspect ratio (if y > x) * opposite / adjacent )
    return atan(aspectRatio * (light->mFocalRadius + light->mLensRadius) / light->mFocalDistance);  
}

//----------------------------------------------------------------------------

export void
SpotLight_init(uniform SpotLight * uniform light)
{
    Light_init((uniform Light * uniform)light, SpotLight_canIlluminate,
                SpotLight_intersect, SpotLight_sample_uniform, SpotLight_eval_uniform,
                SpotLight_getThetaO, SpotLight_getThetaE);
}

//----------------------------------------------------------------------------

