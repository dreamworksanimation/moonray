// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0


#include "Light.isph"

#include <moonray/rendering/shading/ispc/Util.isph>
#include <scene_rdl2/common/platform/IspcUtil.isph>


//----------------------------------------------------------------------------

ISPC_UTIL_EXPORT_UNIFORM_STRUCT_TO_HEADER(EnvLight);

export uniform uint32_t
EnvLight_hudValidation(uniform bool verbose)
{
    ENV_LIGHT_VALIDATION;
}


//----------------------------------------------------------------------------

// Compute local space dir from uv sphere parameterization. The output vector
// is defined in a z-up local coordinate system.
inline varying Vec3f
uv2local(const varying Vec2f &uv)
{
    // We are flipping the u coordinate here since we're on the inside of the
    // sphere. We're flipping v since local z and texture v are in opposite
    // directions.
    float phi = (1.0f - uv.x) * sTwoPi;
    float theta = (1.0f - uv.y) * sPi;

    float sinTheta, cosTheta, sinPhi, cosPhi;
    sincos(theta, &sinTheta, &cosTheta);
    sincos(phi, &sinPhi, &cosPhi);

    return Vec3f_ctor(sinTheta * cosPhi, sinTheta * sinPhi, cosTheta);
}


// Compute uv sphere parameterization from local space direction. The input dir
// is assumed to be defined in a z-up local coordinate system.
inline varying Vec2f
local2uv(const varying Vec3f &dir)
{
    MNRY_ASSERT(isNormalized(dir));

    float theta = dw_acos(clamp(dir.z, -1.0f, 1.0f));
    float phi = atan2(dir.y, dir.x);

    float u = phi * sOneOverTwoPi;
    float v = 1.0f - (theta * sOneOverPi);

    // Handle singularities at poles, to keep function consistent when inverted.
    if (v < 0.0005f || v > 0.9995f) {
        u = 0.0f;
    } else if (u < 0.0f) {
        u += 1.0f;
    }

    // undo the flip of the u coordinate that we did in uv2local
    u = 1.0f - u;

    // TODO:
    MNRY_ASSERT(isfinite(u));
    MNRY_ASSERT(isfinite(v));

    return Vec2f_ctor(u, v);
}


static varying Vec3f
EnvLight_localToGlobal(const uniform EnvLight * uniform light, const varying Vec3f &v, varying float time)
{
    if (!light->mMb) return localToGlobal(light->mFrame, v);

    // construct a new frame
    Mat3f m = Mat3f_ctor(slerp(light->mOrientation[0], light->mOrientation[1], time));
    return v * m;
}

static varying Vec3f
EnvLight_globalToLocal(const uniform EnvLight * uniform light, const varying Vec3f &v, varying float time)
{
    if (!light->mMb) return globalToLocal(light->mFrame, v);

    // construct a new frame
    Mat3f m = transpose(Mat3f_ctor(slerp(light->mOrientation[0], light->mOrientation[1], time)));
    return v * m;
}

static varying Xform3f
EnvLight_globalToLocalXform(const uniform EnvLight * uniform light, varying float time,
                            bool needed = true)
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
EnvLight_canIlluminate(const uniform Light * uniform li, const varying Vec3f &p, const varying Vec3f &cullingNormal,
                       varying float time, varying float radius,
                       const uniform LightFilterList * uniform lightFilterList)
{
    const uniform EnvLight * uniform light = (const uniform EnvLight * uniform)li;
    MNRY_ASSERT(li->mOn);
    return true;
}

varying bool
EnvLight_intersect(const uniform Light * uniform li, const varying Vec3f &p, const varying Vec3f &cullingNormal,
                   const varying Vec3f &wi, varying float time, varying float maxDistance,
                   varying LightIntersection &isect)
{
    const uniform EnvLight * uniform light = (const uniform EnvLight * uniform)li;

    if (light->mHemispherical && dot(wi, Light_getDirection(li, time)) < 0.0f) {
        return false;
    }

    if (sEnvLightDistance > maxDistance) {
        return false;
    }

    isect.N = neg(wi);
    isect.distance = sEnvLightDistance;
    isect.uv = light->mDistribution ? local2uv(EnvLight_globalToLocal(light, wi, time))
                                         : Vec2f_ctor(0.0f);

    return true;
}


varying float
EnvLight_getMipLevel(const uniform Light * uniform li, varying float rayDirFootprint)
{
    const uniform EnvLight * uniform light = (const uniform EnvLight * uniform)li;
    return rayDirFootprint - light->mLog2TexelAngle;
}


varying bool
EnvLight_sample(const uniform Light * uniform li, const varying Vec3f &p, const varying Vec3f &cullingNormal,
                varying float time, const varying Vec3f &r, varying Vec3f &wi, varying LightIntersection &isect,
                varying float rayDirFootprint)
{
    const uniform EnvLight * uniform light = (const uniform EnvLight * uniform)li;

    MNRY_ASSERT(li->mOn);

    if (light->mDistribution) {
        float mipLevel = EnvLight_getMipLevel(li, rayDirFootprint);
        ImageDistribution_sample(light->mDistribution, r.x, r.y, mipLevel, &isect.uv, nullptr, light->mTextureFilter);

        // Handle singularities at poles so that sample() and intersect()
        // will return identical values. if v == 0, then throw away u value
        // since it's meaningless.
        if (isect.uv.y < 0.0005f || isect.uv.y > 0.9995f) {
            isect.uv.x = 0.0f;
        }

        wi = uv2local(isect.uv);
        wi = EnvLight_localToGlobal(light, wi, time);

    } else if (light->mHemispherical) {

        isect.uv = Vec2f_ctor(0.0f);
        wi = sampleLocalHemisphereUniform(r.x, r.y);
        wi = EnvLight_localToGlobal(light, wi, time);

    } else {

        isect.uv = Vec2f_ctor(0.0f);
        wi = sampleSphereUniform(r.x, r.y);
    }

    if (isValidCullingNormal(cullingNormal)  &&  dot(cullingNormal, wi) < sEpsilon) {
        return false;
    }

    isect.N = neg(wi);
    isect.distance = sEnvLightDistance;

    return true;
}

varying Color
EnvLight_eval(const uniform Light * uniform li, uniform ShadingTLState * uniform tls, const varying Vec3f &wi,
              const varying Vec3f &p, const varying LightFilterRandomValues& filterR, varying float time,
              const varying LightIntersection &isect, uniform bool fromCamera,
              const uniform LightFilterList * uniform lightFilterList, varying float rayDirFootprint,
              varying float * uniform pdf)
{
    const uniform EnvLight * uniform light = (const uniform EnvLight * uniform)li;

    MNRY_ASSERT(light->mOn);

    float mipLevel = EnvLight_getMipLevel(li, rayDirFootprint);

    Color radiance = light->mRadiance;
    if (light->mDistribution) {
        // TODO: Use proper filtering with ray differentials and mip-mapping.
        radiance = radiance * ImageDistribution_eval(light->mDistribution,
                isect.uv.x, isect.uv.y, mipLevel, light->mTextureFilter);
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
        evalData.lightRender2LocalXform = EnvLight_globalToLocalXform(
            light, time, LightFilterList_getNeedsLightXform(lightFilterList));
        evalData.wi = wi;

        LightFilterList_eval(lightFilterList, evalData, &radiance);
    }

    if (pdf) {
        if (light->mDistribution) {
            // We must account for the mapping transformation so we express the
            // pdf density on the sphere in solid angles (see pbrt section 14.6.5).
            float sinTheta = sin((isect.uv.y) * sPi);
            sinTheta = max(sinTheta, sEpsilon);
            *pdf = ImageDistribution_pdf(light->mDistribution, isect.uv.x, isect.uv.y, mipLevel, light->mTextureFilter)
                 / (sTwoPiSqr * sinTheta);
            MNRY_ASSERT(isfinite(*pdf));
        } else {
            *pdf = light->mInvArea;
        }
    }

    return radiance;
}

// Unbounded lights aren't included in the LightTree sampling BVH, so these values aren't needed
inline uniform float EnvLight_getThetaO(const uniform Light * uniform li) { return 0.f; }
inline uniform float EnvLight_getThetaE(const uniform Light * uniform li) { return 0.f; }

//----------------------------------------------------------------------------

export void
EnvLight_init(uniform EnvLight * uniform light)
{
    Light_init((uniform Light * uniform)light, EnvLight_canIlluminate,
                EnvLight_intersect, EnvLight_sample, EnvLight_eval,
                EnvLight_getThetaO, EnvLight_getThetaE);
}

//----------------------------------------------------------------------------

