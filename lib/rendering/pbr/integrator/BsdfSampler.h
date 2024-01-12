// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

///
/// @file BsdfSampler.h
/// $Id$
///

#pragma once

#include <moonray/rendering/pbr/core/PbrTLState.h>
#include <moonray/rendering/pbr/core/Util.h>
#include <moonray/rendering/pbr/light/Light.h>
#include "PathGuide.h"

#include <moonray/rendering/pbr/integrator/BsdfSampler_ispc_stubs.h>
#include <moonray/rendering/shading/bsdf/Bsdf.h>
#include <moonray/rendering/shading/bsdf/BsdfSlice.h>

#include <scene_rdl2/common/math/Color.h>
#include <scene_rdl2/common/math/Vec2.h>
#include <scene_rdl2/common/math/Vec3.h>
#include <scene_rdl2/common/platform/IspcUtil.h>


namespace scene_rdl2 {

namespace alloc {
    class Arena;
}
}
namespace moonray {

namespace pbr {


//----------------------------------------------------------------------------

/// Additional information per bsdf sample
/// that is needed for light path event processing 
struct BsdfSampleLPE {
    const Light *light;
};

struct BsdfSample {
    scene_rdl2::math::Vec2f sample;
    scene_rdl2::math::Vec3f wi;
    float distance;
    float pdf;
    scene_rdl2::math::Color f;
    scene_rdl2::math::Color tDirect;
    scene_rdl2::math::Color tIndirect;

    BsdfSampleLPE lp;

    static constexpr float sInvalidDistance = 0.0f;

    finline void setInvalid()           {  distance = sInvalidDistance;  }
    finline bool isValid() const        {  return distance > 0.0f;  }
    finline bool isInvalid() const      {  return distance <= 0.0f;  }
    finline bool didHitLight() const    {  return distance < scene_rdl2::math::sMaxValue;  }
};

///
/// @class BsdfSampler BsdfSampler.h <pbr/integrator/BsdfSampler.h>
/// @brief A BsdfSampler object is a helper class used to efficiently sample
/// multi-lobe bsdfs, using Veach's multi-sample model.
/// It uses lobe importance sampling, but also helps the integrator to preserve
/// good stratification of lobe sampling, giving at least one
/// sample per active lobe, at most one sample per mirror lobe, and a
/// variable number of samples for other lobes. It also rounds the number of
/// sample per lobe to a power of two and enforces other sample count
/// constraints.
/// This class lets the integrator:
/// - Setup a maximum number of samples per lobe
/// - Generate / Iterate over bsdf samples, retrieving lobe properties for the
///   current sample
///
class BsdfSampler
{
public:

    BsdfSampler(scene_rdl2::alloc::Arena *arena, shading::Bsdf &bsdf, const shading::BsdfSlice &slice,
        int maxSamplesPerLobe, bool doIndirect,
        const PathGuide &pathGuide);
    ~BsdfSampler();


    finline shading::Bsdf &getBsdf() const                  {  return mBsdf;  }
    finline const shading::BsdfSlice &getBsdfSlice() const  {  return mSlice;  }
    finline const PathGuide &getPathGuide() const           {  return mPathGuide; }


    // Returns maximum number of samples per lobe
    finline int getMaxSamplesPerLobe() const     {  return mMaxSamplesPerLobe;  }

    // Returns whether we compute indirect illumination
    finline int getDoIndirect() const     {  return mDoIndirect;  }

    // Returns active lobe count in the sampler. All lobeIndex in this class
    // are in the range [0..getLobeCount()[
    finline int getLobeCount() const     {  return mLobeCount;  }

    // Returns lobe from the Bsdf
    finline const shading::BsdfLobe *getLobe(int lobeIndex) const
            {  MNRY_ASSERT(lobeIndex < mLobeCount);
               return mBsdf.getLobe(mLobeIndex[lobeIndex]);  }
    finline shading::BsdfLobe *getLobe(int lobeIndex)
            {  MNRY_ASSERT(lobeIndex < mLobeCount);
               return mBsdf.getLobe(mLobeIndex[lobeIndex]);  }

    // Returns total number of samples for all lobes in the sampler
    finline int getSampleCount() const  {  return mSampleCount;  }

    // Returns number of samples for the given lobe index
    finline int getLobeSampleCount(int lobeIndex) const
            {  MNRY_ASSERT(lobeIndex < mLobeCount);
               return mLobeSampleCount[lobeIndex];  }
    finline float getInvLobeSampleCount(int lobeIndex) const
            {  MNRY_ASSERT(lobeIndex < mLobeCount);
               return mInvLobeSampleCount[lobeIndex];  }

    // Sample iteration should proceed as follows.  Given
    // an array of BsdfSample, bsmp...
    //
    // int s = 0;
    // for (int lobeIndex = 0; lobeIndex < getLobeCount(); ++lobeIndex) {
    //     for (int i = 0; i < getLobeSampleCount(lobeIndex); ++i, ++s) {
    //         sample(lobeIndex, r1, r2, bsmp[s]);
    //     }
    // }
    bool sample(pbr::TLState *pbrTls, int lobeIndex, const scene_rdl2::math::Vec3f &p,
        float r1, float r2, BsdfSample &sample) const;

private:
    /// Copy disabled for now
    BsdfSampler(const BsdfSampler &other);
    BsdfSampler &operator=(const BsdfSampler &other);


    shading::Bsdf &mBsdf;
    const shading::BsdfSlice &mSlice;
    int mMaxSamplesPerLobe;
    bool mDoIndirect;

    int mLobeCount;     // number of lobes matching the flags
    int mSampleCount;

    /// Note: This supports up to 256 lobes which should be more than enough!
    char *mLobeIndex;
    int *mLobeSampleCount;
    float *mInvLobeSampleCount;

    // path guiding
    const PathGuide &mPathGuide;
};


//----------------------------------------------------------------------------

// ispc vector types
ISPC_UTIL_TYPEDEF_STRUCT(BsdfSampler, BsdfSamplerv)
ISPC_UTIL_TYPEDEF_STRUCT(BsdfSample, BsdfSamplev)

} // namespace pbr
} // namespace moonray

