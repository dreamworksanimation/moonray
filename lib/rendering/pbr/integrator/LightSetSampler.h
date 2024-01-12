// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

///
/// @file LightSetSampler.h
/// $Id$
///

#pragma once

#include <moonray/rendering/pbr/core/PbrTLState.h>
#include <moonray/rendering/pbr/light/Light.h>
#include <moonray/rendering/pbr/light/LightSet.h>

#include <moonray/rendering/shading/bsdf/Bsdf.h>

#include <scene_rdl2/common/math/Color.h>
#include <scene_rdl2/common/math/Vec3.h>

namespace scene_rdl2 {

namespace alloc {
    class Arena;
}
}

namespace moonray {

namespace pbr {


//----------------------------------------------------------------------------

/// Additional information per light sample
/// that is needed for light path event processing
struct LightSampleLPE {
    scene_rdl2::math::Color t[shading::Bsdf::maxLobes];
    const shading::BsdfLobe *lobe[shading::Bsdf::maxLobes];
};

struct LightSample {
    scene_rdl2::math::Vec3f wi;
    float distance;
    float pdf;
    scene_rdl2::math::Color Li;
    scene_rdl2::math::Color t;

    LightSampleLPE lp;

    static constexpr float sInvalidDistance = 0.0f;

    finline void setInvalid()       {  distance = sInvalidDistance;  }
    finline bool isValid() const    {  return distance > 0.0f;  }
    finline bool isInvalid() const  {  return distance <= 0.0f;  }
};

///
/// @class LightSetSampler LightSetSampler.h <pbr/LightSetSampler.h>
/// @brief A LightSetSampler object is used to efficiently sample a LightSet,
/// using Veach's multi-sample model.
/// It uses light importance sampling, but also enforces good stratification
/// of light sampling, giving at least one sample per active light, at most
/// one sample per delta light, and a variable number of samples for other
/// lights. It also rounds the number of sample per lobe to a power of two
/// and enforces other sample count constraints.
/// This class lets the integrator:
/// - Setup a maximum number of samples per light
/// - Generate / Iterate over light-set samples, retrieving light properties
///   for the current sample
///
class LightSetSampler
{
public:

    LightSetSampler(scene_rdl2::alloc::Arena *arena, const LightSet &lightSet, const shading::Bsdf &bsdf,
            const scene_rdl2::math::Vec3f &p, int maxSamplesPerLight);
    ~LightSetSampler();

    LightSetSampler(const LightSetSampler &) = delete;
    LightSetSampler &operator=(const LightSetSampler &) = delete;

    // Returns maximum samples per light
    finline int getMaxSamplesPerLight() const    {  return mMaxSamplesPerLight;  }

    // Returns light count
    finline int getLightCount() const
    {
        return mLightSet.getLightCount();
    }

    // Returns light
    finline const Light *getLight(int lightIndex) const
    {
        MNRY_ASSERT(lightIndex < mLightSet.getLightCount());
        return mLightSet.getLight(lightIndex);
    }

    finline const LightSet& getLightSet() const { return mLightSet; }

    // Returns light filter list
    finline const LightFilterList *getLightFilterList(int lightIndex) const
    {
        MNRY_ASSERT(lightIndex < mLightSet.getLightCount());
        return mLightSet.getLightFilterList(lightIndex);
    }

    // Returns total number of samples for all lights in the light set
    finline int getSampleCount() const          {  return mSampleCount;  }
    finline float getInvSampleCount() const     {  return mInvSampleCount;  }

    // Returns number of samples for one light
    finline int getLightSampleCount() const     {  return mLightSampleCount;  }
    finline float getInvLightSampleCount() const{  return mInvLightSampleCount;  }

    // Sample iteration should proceed as follows.  Given
    // an array of LightSample, lsmp...
    //
    // int s = 0;
    // for (int lightIndex = 0; lightIndex < getLightCount(); ++lightIndex) {
    //     for (int i = 0; i < getLightSampleCount(); ++i, ++s) {
    //         sampleIntersectAndEval(lightIndex, P, N, time, r1, r2, lsmp[s]);
    //     }
    // }
    void sampleIntersectAndEval(mcrt_common::ThreadLocalState* tls,
            const Light* light, const LightFilterList *lightFilterList,
            const scene_rdl2::math::Vec3f &P, const scene_rdl2::math::Vec3f *N, const LightFilterRandomValues& filterR,
            float time, const scene_rdl2::math::Vec3f& r, LightSample &sample, float rayDirFootprint) const;

    finline void intersectAndEval(mcrt_common::ThreadLocalState* tls,
            const scene_rdl2::math::Vec3f &P, const scene_rdl2::math::Vec3f* N, const scene_rdl2::math::Vec3f &wi,
            const LightFilterRandomValues& filterR, float time, bool fromCamera, bool includeRayTerminationLights,
            IntegratorSample1D &samples, int depth, int visibilityMask, LightContribution &lCo,
            float rayDirFootprint) const {
        mLightSet.intersectAndEval(tls, P, N, wi, filterR, time, fromCamera, includeRayTerminationLights, samples,
            depth, visibilityMask, lCo, rayDirFootprint);
    }

    finline const shading::Bsdf &getBsdf() const { return *mBsdf; }

private:
    const LightSet &mLightSet;
    int mMaxSamplesPerLight;

    int mSampleCount;
    float mInvSampleCount;

    int mLightSampleCount;
    float mInvLightSampleCount;

    const shading::Bsdf *mBsdf;
};


//----------------------------------------------------------------------------

} // namespace pbr
} // namespace moonray

