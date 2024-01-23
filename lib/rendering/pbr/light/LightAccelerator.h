// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0


#pragma once

#include "Light.h"
#include "LightAccelerator.hh"

#include <scene_rdl2/common/math/Vec3.h>
#include <moonray/rendering/pbr/light/LightTree.h>
#include <moonray/rendering/pbr/sampler/IntegratorSample.h>

#include <embree4/rtcore.h>
#include <embree4/rtcore_ray.h>


namespace moonray {
namespace pbr {

class LightAccelerator
{
public:
    LightAccelerator();
    ~LightAccelerator();

    // HUD validation and type casting
    static uint32_t hudValidation(bool verbose) {
        LIGHT_ACCELERATOR_VALIDATION;
    }

    void init(const Light*const* lights, int lightCount, const RTCDevice& rtcDevice, 
              float sceneDiameter, float samplingThreshold);
    int intersect(const scene_rdl2::math::Vec3f &P, const scene_rdl2::math::Vec3f* N, const scene_rdl2::math::Vec3f &wi, float time,
        float maxDistance, bool includeRayTerminationLights, int visibilityMask, IntegratorSample1D &samples,
        int depth, LightIntersection &isect, int &numHits, const int* lightIdMap) const;
    // The 'self' parameter is used to avoid self-occlusion when light blocking is enabled
    bool occluded(const scene_rdl2::math::Vec3f &P, const scene_rdl2::math::Vec3f &wi, float time, float maxDistance,
        const Light* self=nullptr) const;
    finline const Light* getLight(int l) const { return mLights[l]; }
    finline int getLightCount() const { return mLightCount; }
    finline bool useAcceleration() const { return mBoundedLightCount >= SCALAR_THRESHOLD_COUNT; }
    void buildSamplingTree();

private:
    LIGHT_ACCELERATOR_MEMBERS;

    int intersectBounded(const scene_rdl2::math::Vec3f &P, const scene_rdl2::math::Vec3f* N, const scene_rdl2::math::Vec3f &wi,
        float time, float maxDistance, bool includeRayTerminationLights, int visibilityMask,
        IntegratorSample1D &samples, int depth, LightIntersection &isect, int &numHits, const int* lightIdMap) const;
    int intersectUnbounded(const scene_rdl2::math::Vec3f &P, const scene_rdl2::math::Vec3f &wi, float time,
        float maxDistance, bool includeRayTerminationLights, int visibilityMask, IntegratorSample1D &samples,
        int depth, LightIntersection &isect, int &numHits, const int* lightIdMap) const;
};


typedef std::vector<LightAccelerator> LightAccList;

//----------------------------------------------------------------------------

} // namespace pbr
} // namespace moonray

