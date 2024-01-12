// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

///
/// @file BsdfOneSampler.h
/// $Id$
///

#pragma once

#include <moonray/rendering/pbr/integrator/BsdfOneSampler_ispc_stubs.h>
#include <moonray/rendering/shading/bsdf/Bsdf.h>
#include <moonray/rendering/shading/bsdf/BsdfSlice.h>
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

///
/// @class BsdfOneSampler BsdfOneSampler.h <pbr/BsdfOneSampler.h>
/// @brief A BsdfOneSampler object is used to efficiently sample multi-lobe
/// bsdfs.
///
class BsdfOneSampler
{
public:
    struct LobesContribution
    {
        int mMatchedLobeCount;
        scene_rdl2::math::Color mFs[shading::Bsdf::maxLobes];
        const shading::BsdfLobe* mLobes[shading::Bsdf::maxLobes];
    };

    /// A bsdf slice is defined given a surface geometric normal Ng and
    /// observer direction wo. We are also narrowing down on only the lobes that
    /// match the given flags.
    BsdfOneSampler(const shading::Bsdf &bsdf, const shading::BsdfSlice &slice);
    ~BsdfOneSampler();

    const shading::Bsdf &getBsdf() const { return mBsdf; }
    const shading::BsdfSlice &getSlice() const { return mSlice; }

    /// The following sampling API implements Veach's one-sample model. The API
    /// follows mostly the same rules and conventions from the BsdfLobe class.
    scene_rdl2::math::Color eval(const scene_rdl2::math::Vec3f &wi, float &pdf,
            LobesContribution* lobesContribution = nullptr) const;

    scene_rdl2::math::Color sample(float r0, float r1, float r2, scene_rdl2::math::Vec3f &wi, float &pdf,
            LobesContribution* lobesContribution = nullptr) const;

    // Pick a lobe, sets pdf to the probability of picking this lobe
    const shading::BsdfLobe *sampleLobe(float r, float &pdf) const;

private:
    /// Copy is disabled (for now)
    BsdfOneSampler(const BsdfOneSampler &other);
    BsdfOneSampler &operator=(const BsdfOneSampler &other);

    int sampleCdf(float r, float &pdf) const;

    const shading::Bsdf &mBsdf;
    const shading::BsdfSlice &mSlice;

    /// The lobes' sampling CDF and index array are ordered with an increasing cdf
    /// Note: Using 8 bit lobe indices supports up to 256 lobes
    /// which should be more than enough!
    float mLobeCdf[shading::Bsdf::maxLobes];
    char mLobeIndex[shading::Bsdf::maxLobes];
    int mLobeCount;     // number of matching lobes with non-zero cdf
};


//----------------------------------------------------------------------------

// ispc vector type
ISPC_UTIL_TYPEDEF_STRUCT(BsdfOneSampler, BsdfOneSamplerv)

// Wrapper class used for single lane evaluation.
// It ensures that lane matches across init, sample, and eval calls
// and adds convenient ispc type casts to and from math, pbr, and shading.
class BsdfOneSamplervOneLane
{
public:
    struct LobesContribution
    {
        int mMatchedLobeCount;
        scene_rdl2::math::Color mFs[shading::Bsdf::maxLobes];
        const shading::BsdfLobev* mLobes[shading::Bsdf::maxLobes];
    };
    BsdfOneSamplervOneLane(const shading::Bsdfv &bsdfv, const shading::BsdfSlicev &slicev, int lane);
    scene_rdl2::math::Color eval(const scene_rdl2::math::Vec3f &wi, float &pdf,
                                 LobesContribution *lobesContribution) const;
    scene_rdl2::math::Color sample(float r0, float r1, float r2, scene_rdl2::math::Vec3f &wi, float &pdf,
                                   LobesContribution *lobesContribution) const;

private:
    BsdfOneSamplerv mBsampler;
    int mLane;
};

} // namespace pbr
} // namespace moonray

