// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0


#include "LightUtil.isph"
#include "LightSet.isph"
#include <moonray/rendering/pbr/core/PbrTLState.isph>

#include <moonray/rendering/shading/ispc/bsdf/Bsdf.isph>
#include <moonray/rendering/shading/ispc/bssrdf/Bssrdf.isph>
#include <scene_rdl2/common/platform/IspcUtil.isph>
#include <scene_rdl2/render/util/Arena.isph>

//----------------------------------------------------------------------------

ISPC_UTIL_EXPORT_UNIFORM_STRUCT_TO_HEADER(FalloffCurve);
ISPC_UTIL_EXPORT_ENUM_TO_HEADER(FalloffCurveType);

ISPC_UTIL_EXPORT_UNIFORM_STRUCT_TO_HEADER(OldFalloffCurve);
ISPC_UTIL_EXPORT_ENUM_TO_HEADER(OldFalloffCurveType);


export uniform uint32_t
Plane_hudValidation(uniform bool verbose)
{
    PLANE_VALIDATION;
}

export uniform uint32_t
FalloffCurve_hudValidation(uniform bool verbose)
{
    FALLOFF_CURVE_VALIDATION;
}

export uniform uint32_t
OldFalloffCurve_hudValidation(uniform bool verbose)
{
    OLD_FALLOFF_CURVE_VALIDATION;
}

void
computeActiveLights(uniform Arena * uniform arena,
                    const uniform LightSet * uniform lightList,
                    const varying Vec3f &pos,
                    const varying Vec3f &normal,
                    const varying Bsdf * uniform bsdf,
                    varying float rayTime,
                    uniform LightSet * uniform lightSet,
                    uniform bool &hasRayTerminatorLights)
{
    MNRY_ASSERT(arena);

    uniform int activeLightCount = 0;
    hasRayTerminatorLights = false;
    const uniform Light * uniform * uniform activeLights = nullptr;
    const uniform LightFilterList * uniform * uniform activeLightFilterLists = nullptr;
    // The original light list from "lightList" is culled in this function. We must map the
    // original light list index to the reduced light set index. This is needed for the
    // LightAccelerator. The LightAccelerator, which is used for intersecting lights,
    // is constructed with the original light list. The LightSet, which is used for
    // sampling lights, contains the reduced light list. When we intersect a light in
    // the LightAccelerator, we want to know which light that corresponds to in the LightSet.
    // It is possible that we intersect a light in the light accelerator that does not exist
    // in the LightSet. We map those ids to -1.
    uniform int * uniform lightIdMap = nullptr;

    const varying Bssrdf * uniform bssrdf = Bsdf_getBssrdf(bsdf);
    const varying bool hasBssrdf = (bssrdf != NULL);
    const varying float radius = (hasBssrdf  ?  Bssrdf_getMaxRadius(bssrdf)  :  0.0f);

    uniform int * uniform activeLightIds = nullptr;

    if (lightList) {
        uniform int upperBound = LightSet_getLightCount(lightList);
        lightIdMap = (uniform int * uniform) Arena_allocArray(arena, upperBound, sizeof(uniform int));
        uniform uint8_t *uniform const memBookmark = Arena_getPtr(arena);
        activeLightIds = (uniform int * uniform) Arena_allocArray(arena, upperBound, sizeof(uniform int));

        for (uniform size_t i = 0; i < upperBound; ++i) {
            const uniform Light * uniform light = LightSet_getLight(lightList, i);
            const uniform LightFilterList * uniform lightFilterList = LightSet_getLightFilterList(lightList, i);

            // light culling is done in here
            if (any(Light_canIlluminate(light, pos, normal, rayTime, radius, lightFilterList))) {
                lightIdMap[i] = activeLightCount;
                activeLightIds[activeLightCount++] = i;
                hasRayTerminatorLights |= Light_getIsRayTerminator(light);
            } else {
                lightIdMap[i] = -1;
            }
        }

        if (activeLightCount > 0) {
            activeLights = (const uniform Light * uniform * uniform)
                Arena_allocArray(arena, activeLightCount, sizeof(varying Light *));
            activeLightFilterLists = (const uniform LightFilterList * uniform * uniform)
                Arena_allocArray(arena, activeLightCount, sizeof(varying LightFilterList *));

            for (uniform size_t i = 0; i < activeLightCount; ++i) {
                activeLights[i] = LightSet_getLight(lightList, activeLightIds[i]);
                activeLightFilterLists[i] = LightSet_getLightFilterList(lightList, activeLightIds[i]);
            }
        } else {
            // Reset arena pointer if there are no active lights. If there are active lights, we
            // must keep the arena pointer where is it because we need to keep activeLights and
            // activeLightFilterLists in the arena. This means that even though we don't use activeLighId
            // after this point, it is still hanging around in the arena. This is suboptimal.

            // The pointer is reset to the beginning of activeLightId,
            // because lightIdMap is always filled and used, even when there are no active lights.
            Arena_setPtr(arena, memBookmark);
        }
    }

    // The LightSet does NOT take ownership of the array, however this is not
    // a memory leak because the array is owned by the memory arena.
    LightSet_init(lightSet, activeLights, activeLightCount, activeLightFilterLists);
    LightSet_setAccelerator(lightSet, lightList->mAccelerator, lightIdMap);
}

bool 
chooseThisLight(uniform PbrTLState * uniform pbrTls,
                varying IntegratorSample1D &samples,
                varying int depth, 
                varying int numHits)
{
    // Choose the first light we hit
    varying bool chooseThisOne = true;

    // If we're hitting a second or subsequent light,
    // choose randomly from among those hit
    if (numHits > 1) {
        varying float randVar;
        getSample(samples, randVar, depth, *pbrTls->mFs);
        chooseThisOne = (randVar * (float)numHits < 1.0f);
    }
    
    return chooseThisOne;
}

//----------------------------------------------------------------------------

