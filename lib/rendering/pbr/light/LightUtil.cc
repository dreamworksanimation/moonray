// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0


#include <moonray/rendering/pbr/core/Scene.h>

#include "LightUtil.h"
#include "LightSet.h"

#include <moonray/rendering/bvh/shading/Intersection.h>
#include <moonray/rendering/pbr/light/LightUtil_ispc_stubs.h>
#include <moonray/rendering/shading/bsdf/Bsdf.h>
#include <moonray/rendering/shading/bssrdf/Bssrdf.h>
#include <moonray/rendering/shading/bssrdf/VolumeSubsurface.h>

#include <scene_rdl2/common/math/Vec3.h>

namespace moonray {
namespace pbr {


//----------------------------------------------------------------------------

HUD_VALIDATOR(Plane);
HUD_VALIDATOR(FalloffCurve);


uint32_t
FalloffCurve::hudValidation(bool verbose)
{
    FALLOFF_CURVE_TYPE_ENUM_VALIDATION;
    FALLOFF_CURVE_VALIDATION;
}


void 
computeActiveLights(scene_rdl2::alloc::Arena *arena,
                    const Scene *scene,
                    const shading::Intersection &isect,
                    const scene_rdl2::math::Vec3f *normal,
                    const shading::Bsdf &bsdf,
                    float rayTime,
                    LightSet &lightSet,
                    bool &hasRayTerminatorLights)
{
    MNRY_ASSERT(arena);

    int lightSetIdx = isect.getLayerAssignmentId();
    const LightPtrList *lightList = scene->getLightPtrList(lightSetIdx);
    const LightFilterLists* lightFilterLists = scene->getLightFilterLists(lightSetIdx);

    // null LightSets and LightFilterLists are supported
    const Light **activeLights = nullptr;
    const LightFilterList **activeLightFilterLists = nullptr;
    // The original light list from "scene" is culled in this function. We must map the
    // original light list index to the reduced light set index. This is needed for the
    // LightAccelerator. The LightAccelerator, which is used for intersecting lights,
    // is constructed with the original light list. The LightSet, which is used for
    // sampling lights, contains the reduced light list. When we intersect a light in
    // the LightAccelerator, we want to know which light that corresponds to in the LightSet.
    // It is possible that we intersect a light in the light accelerator that does not exist
    // in the LightSet. We map those ids to -1.
    int* lightIdMap = nullptr;

    const bool hasBssrdf = (bsdf.getBssrdf() != nullptr);
    const bool hasVolumeSubsurface = (bsdf.getVolumeSubsurface() != nullptr);
    const float radius = (hasBssrdf  ?  bsdf.getBssrdf()->getMaxRadius()  :  0.0f);

    int activeLightCount = 0;
    hasRayTerminatorLights = false;

    if (lightList) {
        size_t upperBound = lightList->size();
        lightIdMap = arena->allocArray<int>(upperBound);
        uint8_t * const memBookmark = arena->getPtr();
        int* activeLightId = arena->allocArray<int>(upperBound);

        const scene_rdl2::math::Vec3f& pos = isect.getP();

        for (size_t i = 0; i < upperBound; ++i) {
            const Light *light = (*lightList)[i];

            // light culling is done in here
            // due to the nature of random walk subsurface scattering,
            // we can not use the position/normal of first hit to cull light.
            // there may be chance the fist hit can't be lit by the light but
            // the random walk end point can
            if (hasVolumeSubsurface ||
                light->canIlluminate(pos, normal, rayTime, radius, (*lightFilterLists)[i])) {
                lightIdMap[i] = activeLightCount;
                activeLightId[activeLightCount++] = i;
                hasRayTerminatorLights |= light->getIsRayTerminator();
            } else {
                lightIdMap[i] = -1;
            }
        }

        if (activeLightCount > 0) {
            activeLights = arena->allocArray<const Light *>(activeLightCount);
            activeLightFilterLists = arena->allocArray<const LightFilterList *>(activeLightCount);

            for (size_t i = 0; i < activeLightCount; ++i) {
                activeLights[i] = (*lightList)[activeLightId[i]];
                activeLightFilterLists[i] = (*lightFilterLists)[activeLightId[i]];
            }
        } else {
            // Reset arena pointer if there are no active lights. If there are active lights, we
            // must keep the arena pointer where is it because we need to keep activeLights and
            // activeLightFilterLists in the arena. This means that even though we don't use activeLighId
            // after this point, it is still hanging around in the arena. This is suboptimal.

            // The pointer is reset to the beginning of activeLightId,
            // because lightIdMap is always filled and used, even when there are no active lights.
            arena->setPtr(memBookmark);
        }
    }

    // The LightSet does NOT take ownership of the array, however this is not
    // a memory leak because the array is owned by the memory arena.
    lightSet.init(activeLights, activeLightCount, activeLightFilterLists);

    // Augment light set with Embree-based acceleration structure
    const LightAccelerator *acc = scene->getLightAccelerator(lightSetIdx);
    lightSet.setAccelerator(acc, lightIdMap);
}

bool chooseThisLight(const IntegratorSample1D &samples, int depth, unsigned int numHits)
{
    // Choose the first light we hit
    bool chooseThisOne = true;

    // If we're hitting a second or subsequent light,
    // choose randomly from among those hit
    if (numHits > 1) {
        float randVar;
        samples.getSample(&randVar, depth);
        chooseThisOne = (randVar * (float)numHits < 1.0f);
    }

    return chooseThisOne;
}

//----------------------------------------------------------------------------

} // namespace pbr
} // namespace moonray

