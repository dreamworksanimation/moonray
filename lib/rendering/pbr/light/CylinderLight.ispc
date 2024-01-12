// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0


#include "Light.isph"
#include <moonray/rendering/pbr/core/Util.isph>

#include <scene_rdl2/common/platform/IspcUtil.isph>


//----------------------------------------------------------------------------

ISPC_UTIL_EXPORT_UNIFORM_STRUCT_TO_HEADER(CylinderLight);

export uniform uint32_t
CylinderLight_hudValidation(uniform bool verbose)
{
    CYLINDER_LIGHT_VALIDATION;
}


//----------------------------------------------------------------------------

// Compute local space position from uv cylinder parameterization.
inline varying Vec3f
uv2local(const uniform CylinderLight * uniform light, const varying Vec2f &uv)
{
    float phi = uv.y * sTwoPi;
    float sinPhi, cosPhi;
    sincos(phi, &sinPhi, &cosPhi);

    return Vec3f_ctor(light->mLocalRadius * cosPhi,
                     (uv.x - light->mUvOffset.x) * rcp(light->mUvScale.x),
                     light-> mLocalRadius * sinPhi);
}


// Compute uv cylinder parameterization from local space position.
inline varying Vec2f
local2uv(const uniform CylinderLight * uniform light, const varying Vec3f &pos)
{
    float u = pos.y * light->mUvScale.x + light->mUvOffset.x;

    float invRadius = rcp(light->mLocalRadius);
    float phi = atan2(pos.z * invRadius, pos.x * invRadius);
    float v = (phi < 0.0f  ?  phi + sTwoPi  :  phi) * sOneOverTwoPi;

    MNRY_ASSERT(isfinite(u));
    MNRY_ASSERT(isfinite(v));

    return Vec2f_ctor(u, v);
}


//----------------------------------------------------------------------------

varying bool
CylinderLight_canIlluminate(const uniform Light * uniform li,
        const varying Vec3f &p, const varying Vec3f &cullingNormal, varying float time,
        varying float radius, const uniform LightFilterList * uniform lightFilterList)
{
    const uniform CylinderLight * uniform light = (const uniform CylinderLight * uniform)li;
    const uniform LocalParamLight * uniform lpl = (const uniform LocalParamLight * uniform) li;
    MNRY_ASSERT(li->mOn);

    if (lightFilterList) {
        CanIlluminateData cid;
        cid.lightPosition = Light_getPosition(li, time);
        cid.lightRadius = max(light->mLocalHalfHeight * 2, light->mLocalRadius);
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
CylinderLight_intersect(const uniform Light * uniform li,
        const varying Vec3f &p, const varying Vec3f &cullingNormal,
        const varying Vec3f &wi, varying float time,
        varying float maxDistance, varying LightIntersection &isect)
{
    const uniform LocalParamLight * uniform lpl = (const uniform LocalParamLight * uniform) li;
    const uniform CylinderLight * uniform light = (const uniform CylinderLight * uniform)li;

    const Vec3f e = LocalParamLight_xformVectorLocal2RenderRot(lpl, (const varying Vec3f) Vec3f_ctor(0., 1., 0.), time);
    const Vec3f dp = Light_getPosition(li, time) - p;

    // Trivial rejection if ray is outside slab and travelling away from it
    const float A = dot(dp, e);
    const float B = dot(wi, e);
    if ((B >= 0.0f ? A : -A) <= -light->mActualHalfHeight) {
        return false;
    }

    const float C = dot(dp, wi);
    const float b = C - A*B;
    const float c = lengthSqr(dp) - A*A - light->mActualRadiusSquared;

    // Do trivial rejection for the following cases:
    // regular sidedness: reject if ray is traveling away from axis OR  ray origin is INSIDE cylinder
    // reverse sidedness: reject if ray is traveling away from axis AND ray origin is OUTSIDE cylinder
    if ((c > 0.0f) ? (b <= 0.0f) : (light->mSidedness == LIGHT_SIDEDNESS_REGULAR)) {
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
    const float s = sqrt(D);
    const float t = (light->mSidedness == LIGHT_SIDEDNESS_REGULAR) ? (b - s) / a : (b + s) / a;
    if (a == 0.0f || t < 0.0f || t > maxDistance) {
        return false;
    }

    // Compute intersection point relative to cylinder centre; check it's between end planes
    const float y = t * B - A;
    if (abs(y) > light->mActualHalfHeight) {
        return false;
    }

    // Fill in isect members
    const Vec3f hitPoint = t * wi - dp;
    // We don't need to worry about 2-sided here because that's not supported for CylinderLight
    isect.N = ((light->mSidedness == LIGHT_SIDEDNESS_REGULAR) ? light->mRcpActualRadius : -light->mRcpActualRadius)
                * (hitPoint - y * e);
    isect.uv = light->mDistribution ? local2uv(light, LocalParamLight_xformVectorRender2Local(lpl, hitPoint, time))
                                     :  Vec2f_ctor(0.0f);
    isect.distance = t;

    return true;
}

varying bool
CylinderLight_sample(const uniform Light * uniform li,
        const varying Vec3f &p, const varying Vec3f &cullingNormal, varying float time,
        const varying Vec3f &r, varying Vec3f &wi, varying LightIntersection &isect, varying float rayDirFootprint)
{
    const uniform LocalParamLight * uniform lpl = (const uniform LocalParamLight * uniform)li;
    const uniform CylinderLight * uniform light = (const uniform CylinderLight * uniform)li;

    MNRY_ASSERT(li->mOn);

    // Sample the image distribution if any, otherwise sample the cylinder
    // uv parameterization directly.
    Vec2f uv;
    if (light->mDistribution) {
        ImageDistribution_sample(light->mDistribution, r.x, r.y, 0, &uv, nullptr, light->mTextureFilter);
        isect.uv = uv;
    } else {
        uv = Vec2f_ctor(r.x, r.y);
        isect.uv = Vec2f_ctor(0.0f);
    }

    const Vec3f localHit = uv2local(light, uv);
    const Vec3f renderHit = LocalParamLight_xformPointLocal2Render(lpl, localHit, time);

    // Compute wi and d
    wi = renderHit - p;
    isect.distance = length(wi);
    if (isect.distance < sEpsilon) {
        return false;
    }
    wi = wi * rcp(isect.distance);
    if (isValidCullingNormal(cullingNormal) && dot(cullingNormal, wi) < sEpsilon) {
        return false;
    }

    const Vec3f localN = Vec3f_ctor(localHit.x, 0.0f, localHit.z);
    const Vec3f renderN = normalize(LocalParamLight_xformNormalLocal2Render(lpl, localN, time));
    // We don't need to worry about 2-sided here because that's not supported for CylinderLight
    isect.N = (light->mSidedness == LIGHT_SIDEDNESS_REGULAR) ? renderN : -1.0f * renderN;

    // Reject if the generated point is on the back-facing surface of the cylinder
    if (dot(isect.N, wi) > -sEpsilon) {
        return false;
    }

    return true;
}

varying Color
CylinderLight_eval(const uniform Light * uniform li, uniform ShadingTLState * uniform tls,
        const varying Vec3f &wi, const varying Vec3f &p, const varying LightFilterRandomValues& filterR, 
        varying float time, const varying LightIntersection &isect, uniform bool fromCamera,
        const uniform LightFilterList * uniform lightFilterList, varying float rayDirFootprint,
        varying float * uniform pdf)
{
    const uniform CylinderLight * uniform light = (const uniform CylinderLight * uniform)li;
    const uniform LocalParamLight * uniform lpl = (const uniform LocalParamLight * uniform) li;

    MNRY_ASSERT(li->mOn);

    Color radiance = light->mRadiance;

    if (light->mDistribution) {
        // Point sample the texture if any
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
        *pdf = light->mInvArea;
        if (light->mDistribution) {
            *pdf *= ImageDistribution_pdf(light->mDistribution, isect.uv.x, isect.uv.y, 0, light->mTextureFilter);
        }
        *pdf *= areaToSolidAngleScale(wi, isect.N, isect.distance);
    }

    return radiance;
}

inline uniform float CylinderLight_getThetaO(const uniform Light * uniform li) { return sPi; }
inline uniform float CylinderLight_getThetaE(const uniform Light * uniform li) { return sPi * 0.5f; }

//----------------------------------------------------------------------------

export void
CylinderLight_init(uniform CylinderLight * uniform light)
{
    Light_init((uniform Light * uniform)light, CylinderLight_canIlluminate,
            CylinderLight_intersect, CylinderLight_sample, CylinderLight_eval,
            CylinderLight_getThetaO, CylinderLight_getThetaE);
}

//----------------------------------------------------------------------------

