// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0


#include "LightFilter.isph"
#include <scene_rdl2/common/platform/IspcUtil.isph>

//----------------------------------------------------------------------------

ISPC_UTIL_EXPORT_UNIFORM_STRUCT_TO_HEADER(DecayLightFilter);

export uniform uint32_t
DecayLightFilter_hudValidation(uniform bool verbose)
{
    DECAY_LIGHT_FILTER_VALIDATION;
}


//----------------------------------------------------------------------------

varying bool
DecayLightFilter_canIlluminate(const uniform LightFilter * uniform lif,
        const varying CanIlluminateData &data)
{
    const uniform DecayLightFilter * uniform lightfilter = (const uniform DecayLightFilter * uniform)lif;

    // Approximate light and shading point as two spheres. We can compute the
    // minimum and maximum distances between any 2 points in those two spheres.
    float distance = length(data.lightPosition - data.shadingPointPosition);
    float buffer = data.lightRadius + data.shadingPointRadius;
    float minDistance = buffer > distance ? 0.f : distance - buffer;
    float maxDistance = distance + buffer;
    
    varying bool success = true;

    if (lightfilter->mFalloffNear) {
        success &= maxDistance >= lightfilter->mNearStart;
    }

    if (lightfilter->mFalloffFar) {
        success &= minDistance <= lightfilter->mFarEnd;
    }

    return success;
}

void
DecayLightFilter_eval(const uniform LightFilter * uniform lif,
        const varying EvalData& data,
        varying Color * uniform filterValue)
{
    const uniform DecayLightFilter * uniform lightfilter = (const uniform DecayLightFilter * uniform)lif;

    float dist = data.isect->distance;
    float fval = 1.f;

    if ((lightfilter->mFalloffNear && dist < lightfilter->mNearStart) ||
        (lightfilter->mFalloffFar && dist > lightfilter->mFarEnd)) {
        fval = 0.f;

    } else if (lightfilter->mFalloffNear && dist < lightfilter->mNearEnd) {
        fval = (dist - lightfilter->mNearStart) / 
               (lightfilter->mNearEnd - lightfilter->mNearStart);

    } else if (lightfilter->mFalloffFar && dist > lightfilter->mFarStart) {
        fval = (lightfilter->mFarEnd - dist) / 
               (lightfilter->mFarEnd - lightfilter->mFarStart);
    }

    *filterValue = Color_ctor(fval, fval, fval);
}

//----------------------------------------------------------------------------

export void
DecayLightFilter_init(uniform DecayLightFilter * uniform lightfilter)
{
    LightFilter_init((uniform LightFilter * uniform)lightfilter, 
            DecayLightFilter_canIlluminate, DecayLightFilter_eval);
}


//----------------------------------------------------------------------------

