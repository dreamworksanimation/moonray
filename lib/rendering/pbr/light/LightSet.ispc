// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0


#include "LightSet.isph"
#include <moonray/rendering/pbr/core/Util.isph>

#include <scene_rdl2/common/platform/IspcUtil.isph>


//----------------------------------------------------------------------------

ISPC_UTIL_EXPORT_UNIFORM_STRUCT_TO_HEADER(LightSet);


export uniform uint32_t
LightSet_hudValidation(uniform bool verbose)
{
    LIGHT_SET_VALIDATION;
}

varying int
LightSet_intersect(uniform PbrTLState * uniform pbrTls, const uniform LightSet * uniform lightSet,
                   const varying Vec3f &P, const varying Vec3f &cullingNormal, const varying Vec3f &wi,
                   varying float time, varying float maxDistance, uniform bool fromCamera,
                   varying bool includeRayTerminationLights, varying IntegratorSample1D &samples, varying int depth,
                   varying int visibilityMask, varying LightIntersection &chosenIsect, varying int &numHits)
{
    chosenIsect.distance = maxDistance;

    numHits = 0;

    // Use acceleration if it would be quicker
    const uniform LightAccelerator * uniform acc = lightSet->mAccelerator;
    if (acc && LightAccelerator_useAcceleration(acc)) {
        varying int idx = LightAccelerator_intersect(pbrTls, acc, P, cullingNormal, wi, time, maxDistance,
            includeRayTerminationLights, visibilityMask, samples, depth, chosenIsect, numHits,
            lightSet->mAcceleratorLightIdMap);
        return idx >= 0 ? lightSet->mAcceleratorLightIdMap[idx] : idx;
    }
    
    varying int chosenLightIdx = -1;

    // Test lights for intersection
    for (uniform int l = 0; l < lightSet->mLightCount; l++) {

        varying LightIntersection currentIsect;
        const uniform Light * uniform light = lightSet->mLights[l];

        if (!(visibilityMask & Light_getVisibilityMask(light))) {
            // skip light if it is masked
            continue;
        }

        if (!includeRayTerminationLights && Light_getIsRayTerminator(light)) {
            // Skip any ray termination lights if we were told not to include them
            continue;
        }

        if (Light_intersect(light, P, cullingNormal, wi, time, maxDistance, currentIsect)) {
            numHits++;

            if (chooseThisLight(pbrTls, samples, depth, numHits)) {
                chosenLightIdx = l;
                chosenIsect = currentIsect;
            }
        }
    }

    MNRY_ASSERT(numHits==0 || chosenLightIdx >= 0);

    return chosenLightIdx;
}


void
LightSet_intersectAndEval(uniform PbrTLState * uniform pbrTls,
                          const uniform LightSet * uniform lightSet,
                          uniform ShadingTLState * uniform tls,
                          const varying Vec3f &P, const varying Vec3f &cullingNormal, const varying Vec3f &wi,
                          const varying LightFilterRandomValues &filterR, varying float time, uniform bool fromCamera,
                          varying bool includeRayTerminationLights, varying IntegratorSample1D &samples,
                          varying int depth, varying int visibilityMask, varying LightContribution &lCo,
                          varying float rayDirFootprint)
{
    // Initialize contribution
    lCo.isInvalid = true;
    lCo.light = nullptr;
    lCo.distance = sMaxValue;
    lCo.Li = sBlack;
    lCo.pdf = 0.0f;

    // Compute intersection with a random light
    varying LightIntersection isect;
    varying int numLightsHit = 0;
    varying int lightIdx = LightSet_intersect(pbrTls, lightSet, P, cullingNormal, wi, time, sMaxValue, fromCamera,
        includeRayTerminationLights, samples, depth, visibilityMask, isect, numLightsHit);

    const uniform Light * varying light = nullptr;
    const uniform LightFilterList * varying lightFilterList = nullptr;

    if (lightIdx >= 0) {
        light = LightSet_getLight(lightSet, lightIdx);
        lightFilterList = LightSet_getLightFilterList(lightSet, lightIdx);
    }

    // Early return if we didn't hit a light
    if (numLightsHit == 0) {
        return;
    }

    lCo.light = light;
    lCo.distance = isect.distance;

    // Evaluate the intersected light Li and pdf
    lCo.Li = Light_eval(light, tls, wi, P, filterR, time, isect, fromCamera, lightFilterList, rayDirFootprint,
                        &lCo.pdf);
    lCo.isInvalid = isSampleInvalid(lCo.Li, lCo.pdf);

    // Radiance has to be scaled by number of lights hit because we're only sampling one
    lCo.Li = (float)numLightsHit * lCo.Li;
}




//----------------------------------------------------------------------------

