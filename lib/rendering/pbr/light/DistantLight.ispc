// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0


#include "Light.isph"

#include <moonray/rendering/shading/ispc/Util.isph>
#include <scene_rdl2/common/platform/IspcUtil.isph>


//----------------------------------------------------------------------------

ISPC_UTIL_EXPORT_UNIFORM_STRUCT_TO_HEADER(DistantLight);

export uniform uint32_t
DistantLight_hudValidation(uniform bool verbose)
{
    DISTANT_LIGHT_VALIDATION;
}


//----------------------------------------------------------------------------

static Vec3f
DistantLight_localToGlobal(const uniform DistantLight * uniform light,
                           const varying Vec3f &v, varying float time)
{
    if (!light->mMb) return localToGlobal(light->mFrame, v);

    // construct a new frame
    varying Mat3f m = Mat3f_ctor(slerp(light->mOrientation[0], light->mOrientation[1], time));
    return v * m;
}

static varying Xform3f
DistantLight_globalToLocalXform(const uniform DistantLight * uniform light, 
                                varying float time, bool needed = true)
{
    if (!needed) {
        Xform3f result;
        return result;
    }

    if (!light->mMb) {
        return Xform3f_ctor(transpose(Mat3f_ctor(light->mOrientation[0])),
                            Vec3f_ctor(0.f));
    }

    // construct a new frame
    return Xform3f_ctor(
        transpose(Mat3f_ctor(slerp(light->mOrientation[0], 
                                   light->mOrientation[1], time))),
        Vec3f_ctor(0.f));
}

//----------------------------------------------------------------------------

varying bool
DistantLight_canIlluminate(const uniform Light * uniform li,
        const varying Vec3f &p, const varying Vec3f &cullingNormal, varying float time,
        varying float radius, const uniform LightFilterList * uniform lightFilterList)
{
    const uniform LocalParamLight * uniform lpl = (const uniform LocalParamLight * uniform)li;
    const uniform DistantLight * uniform light = (const uniform DistantLight * uniform)li;

    MNRY_ASSERT(li->mOn);

    if (isValidCullingNormal(cullingNormal) && dot(neg(cullingNormal), Light_getDirection(li, time)) < light->mCullThreshold) {
        return false;
    }

    if (lightFilterList) {
        CanIlluminateData cid;
        cid.lightPosition = Light_getPosition(li, time);
        cid.lightRadius = sMaxValue;
        cid.shadingPointPosition = p;
        cid.lightRender2LocalXform = DistantLight_globalToLocalXform(
                light, time, LightFilterList_getNeedsLightXform(lightFilterList));
        cid.shadingPointRadius = radius;
        cid.time = time;
        return LightFilterList_canIlluminate(lightFilterList, cid);
    }

    return true;
}

varying bool
DistantLight_intersect(const uniform Light * uniform li,
        const varying Vec3f &p, const varying Vec3f &cullingNormal,
        const varying Vec3f &wi, varying float time,
        varying float maxDistance, varying LightIntersection &isect)
{
    const uniform DistantLight * uniform light = (const uniform DistantLight * uniform)li;

    if (dot(neg(wi), Light_getDirection(li, time)) < light->mCosThetaMax) {
        return false;
    }

    if (sDistantLightDistance > maxDistance) {
        return false;
    }

    isect.N = neg(wi);
    isect.distance = sDistantLightDistance;
    isect.uv = Vec2f_ctor(0.0f);

    return true;
}

varying bool
DistantLight_sample(const uniform Light * uniform li,
        const varying Vec3f &p, const varying Vec3f &cullingNormal, varying float time, const varying Vec3f &r,
        varying Vec3f &wi, varying LightIntersection &isect, varying float rayDirFootprint)
{
    const uniform DistantLight * uniform light = (const uniform DistantLight * uniform)li;

    MNRY_ASSERT(li->mOn);

    Vec3f sample = sampleLocalSphericalCapUniform2(r.x, r.y, light->mVersineThetaMax);
    wi = neg(DistantLight_localToGlobal(light, sample, time));

    if (isValidCullingNormal(cullingNormal)  &&  dot(cullingNormal, wi) < sEpsilon) {
        return false;
    }

    isect.N = neg(wi);
    isect.distance = sDistantLightDistance;
    isect.uv = Vec2f_ctor(0.0f);

    return true;
}


varying Color
DistantLight_eval(const uniform Light * uniform li, uniform ShadingTLState * uniform tls,
        const varying Vec3f &wi, const varying Vec3f &p, const varying LightFilterRandomValues& filterR, 
        varying float time, const varying LightIntersection &isect, uniform bool fromCamera,
        const uniform LightFilterList * uniform lightFilterList, varying float rayDirFootprint,
        varying float * uniform pdf)
{
    const uniform DistantLight * uniform light = (const uniform DistantLight * uniform)li;
    const uniform LocalParamLight * uniform lpl = (const uniform LocalParamLight * uniform) li;

    MNRY_ASSERT(light->mOn);

    Color radiance = light->mRadiance;

    if (lightFilterList) {
        EvalData evalData;
        evalData.tls = tls;
        evalData.isect = &isect;
        evalData.lightPosition = Light_getPosition(li, time);
        evalData.lightDirection = Light_getDirection(li, time);
        evalData.shadingPointPosition = p;
        evalData.randVar = filterR; 
        evalData.time = time;
        evalData.lightRender2LocalXform = DistantLight_globalToLocalXform(
            light, time, LightFilterList_getNeedsLightXform(lightFilterList));
        evalData.wi = wi;

        LightFilterList_eval(lightFilterList, evalData, &radiance);
    }

    if (pdf) {
        *pdf = light->mInvArea;
    }
    return radiance;
}

// Unbounded lights aren't included in the LightTree sampling BVH, so these values aren't needed
inline uniform float DistantLight_getThetaO(const uniform Light * uniform li) { return 0.f; }
inline uniform float DistantLight_getThetaE(const uniform Light * uniform li) { return 0.f; }

//----------------------------------------------------------------------------

export void
DistantLight_init(uniform DistantLight * uniform light, uniform bool uniformSampling)
{
    MNRY_ASSERT(!uniformSampling);

    Light_init((uniform Light * uniform)light, DistantLight_canIlluminate,
            DistantLight_intersect, DistantLight_sample, DistantLight_eval,
            DistantLight_getThetaO, DistantLight_getThetaE);
}


//----------------------------------------------------------------------------

