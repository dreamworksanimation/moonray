// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

///
/// @file LightSet.h
/// $Id$
///

#pragma once

#include "Light.h"
#include "LightAccelerator.h"
#include "LightSet.hh"
#include <moonray/rendering/pbr/core/PbrTLState.h>
#include <moonray/rendering/pbr/sampler/IntegratorSample.h>
#include <moonray/rendering/pbr/sampler/SequenceID.h>

#include <scene_rdl2/common/math/Color.h>
#include <scene_rdl2/common/math/Vec3.h>
#include <scene_rdl2/scene/rdl2/LightSet.h>

// Forward declaration of the ISPC types
namespace ispc {
    struct LightSet;
}


namespace moonray {
namespace pbr {

class LightAccelerator;
class PathVertex;

//----------------------------------------------------------------------------

// - pdf is the pdf of this sample for the intersected light
// - Li is the emitted radiance of the intersected light
//   in direction wi, and 0 if there is no light in that direction
//   TODO: or if the intersected light is not in the subset
struct LightContribution {
    bool isInvalid;
    const Light* light;
    float distance;
    float pdf;
    scene_rdl2::math::Color Li;
};


// This class keeps track of the node lightsets encountered along a path.
// Node lightsets also apply to indirect illumination, so any light paths
// that collect indirect illumination must remember all bsdf node lightsets
// encountered along the way and apply them.
class Rdl2LightSetList
{
public:
    Rdl2LightSetList() : mNumLightSets(0) {}

    bool allLightSetsContain(const scene_rdl2::rdl2::Light *rdlLight) const {
        for (int i = 0; i < mNumLightSets; i++) {
            if (!mLightSets[i]->contains(rdlLight)) {
                return false;
            }
        }
        // note that an EMPTY list contains all lights
        return true;
    }
    void append(const scene_rdl2::rdl2::LightSet* lightSet) {
        // Note that only non-null lightsets should be appended.
        // We could check for duplicates as a potential optimization. That would
        // add more processing here and save processing checking the lightsets later.
        // It is not clear if this would be more efficient overall.
        if (mNumLightSets < mMaxLightSets) {
            mLightSets[mNumLightSets++] = lightSet;
        } // else just throw it away, it probably doesn't make much difference
          // after mMaxLightSets bounces
    }

private:
    int mNumLightSets;
    static constexpr int mMaxLightSets = 8;
    const scene_rdl2::rdl2::LightSet* mLightSets[mMaxLightSets];
};


///
/// @class LightSet LightSet.h <pbr/LightSet.h>
/// @brief A light set is a list of lights that should shine! This is distinct
///        from an rdl2 LightSet in that lights which can't illuminate the point/
///        normal we are shading won't get added to this set.
/// 
class LightSet
{
public:
    /// Constructor / Destructor
    LightSet() : mLights(nullptr), mLightCount(0), mAccelerator(nullptr) {}
    ~LightSet() {}


    /// HUD validation and type casting
    static uint32_t hudValidation(bool verbose) {
        LIGHT_SET_VALIDATION;
    }
    HUD_AS_ISPC_METHODS(LightSet);


    finline void init(const Light * const *lights, int lightCount, const LightFilterList * const * lightFilterLists)
    {
        mLights = lights;
        mLightCount = lightCount;
        mLightFilterLists = lightFilterLists;
        mAccelerator = nullptr;
        mAcceleratorLightIdMap = nullptr;
    }

    finline int getLightCount() const              {  return mLightCount;  }
    finline const Light *getLight(int index) const {  return mLights[index];  }
    finline const Light*const* getLights() const   {  return mLights;  }
    finline const LightFilterList *getLightFilterList(int index) const
    {
        return mLightFilterLists[index];
    }

    finline void addInactiveLightsToVisibilityAov(
        const std::function<void(const Light* const)>& addMissesToVisibilityAov) const
    {
        for (int i = 0; i < mAccelerator->getLightCount(); ++i) {
            if (mAcceleratorLightIdMap[i] == -1) {
                addMissesToVisibilityAov(mAccelerator->getLight(i));
            }
        }
    }

    // Returns the light index of a light chosen randomly from among those that
    // intersect the ray(P, wi), within maxDistance. It returns -1 if no light was
    // intersected.
    int intersect(const scene_rdl2::math::Vec3f &P, const scene_rdl2::math::Vec3f *N, const scene_rdl2::math::Vec3f &wi,
            float time, float maxDistance, bool includeRayTerminationLights,  IntegratorSample1D &samples,
            int depth, int visibilityMask, LightIntersection &chosenIsect, int &numHits,
            const scene_rdl2::rdl2::LightSet* bsdfLobeLightSet,
            const Rdl2LightSetList& parentLobeLightSets) const;

    // Intersects the given ray (P, wi), choosing randomly among the lights  in the
    // LightSet intersected by the ray. Fills up the LightContribution structure with the
    // results. The random samples must first be initialized with a call to initSamples().
    void intersectAndEval(mcrt_common::ThreadLocalState *shadingTls, const scene_rdl2::math::Vec3f &P, const scene_rdl2::math::Vec3f *N,
            const scene_rdl2::math::Vec3f &wi, const LightFilterRandomValues& filterR, float time, const PathVertex *pv, bool fromCamera,
            bool includeRayTerminationLights, IntegratorSample1D &samples, int depth,  int visibilityMask,
            LightContribution &lCo, float rayDirFootprint,
            const scene_rdl2::rdl2::LightSet* bsdfLobeLightSet,
            const Rdl2LightSetList& parentLobeLightSets) const;

    finline const LightAccelerator* getAccelerator() const { return mAccelerator; }
    finline const int* getLightIdMap() const { return mAcceleratorLightIdMap; }

    // Set the accelerator for this light set. The accelerator may contain more lights than the LightSet itself.
    // This is because the accelerator's light list is generated during render prep, and includes all of the original
    // lights specified in the rdl2::LightSet. The LightSet has some lights culled away during mcrt, so contains a
    // subset of those lights. We need to map the index of the lights in the LightAccelerator to the index of the light
    // in the LightSet. The lightIdMap does this. Lights that exist in the LightAccelerator but not in the LightSet are
    // mapped to -1.
    finline void setAccelerator(const LightAccelerator* acc, const int* lightIdMap)
    {
        mAccelerator = acc;
        mAcceleratorLightIdMap = lightIdMap;
    }

private:
    /// Copy is disabled
    LightSet(const LightSet &other);
    const LightSet &operator=(const LightSet &other);

    LIGHT_SET_MEMBERS;
};


//----------------------------------------------------------------------------

} // namespace pbr
} // namespace moonray

