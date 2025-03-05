// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

///
/// @file BsdfSampler.cc
/// $Id$
///


#include "BsdfSampler.h"

#include <scene_rdl2/render/util/Arena.h>
#include <scene_rdl2/render/util/BitUtils.h>


namespace moonray {
namespace pbr {


using namespace scene_rdl2::math;


//----------------------------------------------------------------------------

// Warning: this must return at least one sample for each lobe, otherwise
// the algorithm is biased (unless the lobe has zero albedo, but in this case
// the shader should never have added the lobe in the first place).
static finline int
computeLobeSampleCount(const shading::BsdfLobe *lobe, const shading::BsdfSlice &slice,
        int maxLobeSamples, int lobeCount)
{
    int sampleCount;

    if (lobe->matchesFlags(shading::BsdfLobe::ALL_MIRROR)  ||  maxLobeSamples == 1) {
        // We devote only a single sample to delta distributions
        sampleCount = 1;
    } else {
        // Compute lobes' approximated albedo to adjust the number of samples.
        // Note: since we use the approximated albedo, our sampling decisions might
        // be slightly off in cases where a lobe crosses the surface (at grazing
        // angles). In these cases, we may spend slightly more samples than we
        // should. Also since we don't know yet about the incident lighting, we may
        // spend fewer samples for a given lobe than we wish for (i.e. a low-albedo
        // lobe that reflects a very strong spike of light), or too many samples
        // (i.e. a high-albedo lobe that reflects no light at all).
        // Note: Use the mirror reflection direction assumption to compute the
        // albedo approximaion, which scales our lobe sampling decisions
        // TODO: Factor in heuristics based on roughness, open up to
        // other heuristics (pass in a functor ?)
#if 1
        sampleCount = maxLobeSamples;
#else
        // Note: we multiply by the number of lobes, because the number of samples
        // is adjusted by the albedo, which shouldn't add up to more than one over
        // all lobes, so this is to keep roughly the same sample count as before.
        float lumAlbedo = abs(luminance(lobe->albedo(slice)));
        sampleCount = static_cast<int>(ceil(lumAlbedo * maxLobeSamples * lobeCount));
        sampleCount = min(sampleCount, maxLobeSamples * lobeCount);
#endif
    }

    // We need at least 1 sample per lobe otherwise this algorithm is biased
    MNRY_ASSERT(maxLobeSamples == 0  ||  sampleCount > 0);

    return sampleCount;
}


//----------------------------------------------------------------------------

BsdfSampler::BsdfSampler(scene_rdl2::alloc::Arena *arena, shading::Bsdf &bsdf, const shading::BsdfSlice &slice,
        int maxSamplesPerLobe, bool doIndirect) :
    mBsdf(bsdf),
    mSlice(slice),
    mMaxSamplesPerLobe(maxSamplesPerLobe),
    mDoIndirect(doIndirect),
    mLobeCount(mBsdf.getLobeCount()),
    mSampleCount(0)
{
    // Allocate and initialize lobe arrays
    mLobeIndex = arena->allocArray<char>(mLobeCount);
    mLobeSampleCount = arena->allocArray<int>(mLobeCount);
    mInvLobeSampleCount = arena->allocArray<float>(mLobeCount);

    // Compute sample count per lobe, and keep track of lobes with samples
    int lKeep = 0;
    shading::BsdfLobe::Type flags = mSlice.getFlags();
    for (int l = 0; l < mLobeCount; ++l) {
        const shading::BsdfLobe *lobe = mBsdf.getLobe(l);
        if (!lobe->matchesFlags(flags)) {
            continue;
        }

        int sampleCount = computeLobeSampleCount(lobe, mSlice, maxSamplesPerLobe, mLobeCount);

        mLobeIndex[lKeep] = l;
        mLobeSampleCount[lKeep] = sampleCount;
        mInvLobeSampleCount[lKeep] = 1.0f / static_cast<float>(sampleCount);
        mSampleCount += sampleCount;

        lKeep++;
    }
    mLobeCount = lKeep;

    // Early return if no lobes match the given flag
    if (mLobeCount == 0) {
        MNRY_ASSERT(mSampleCount == 0);
        return;
    }
}


BsdfSampler::~BsdfSampler()
{
}

bool
BsdfSampler::sample(pbr::TLState *pbrTls, int lobeIndex, const Vec3f &p, float r1,
    float r2, BsdfSample &sample) const
{
    // Get lobe in turn and draw a sample from that lobe
    const shading::BsdfLobe *lobe = getLobe(lobeIndex);
    sample.sample = Vec2f(r1, r2);

    // Pdf computation needs to be kept in sync when integrating
    // light samples (see integrateLightSetSample() in PathIntegratorUtil.cc).
    sample.f = lobe->sample(mSlice, r1, r2, sample.wi, sample.pdf);

    // Check if sample is invalid
    bool isValid = isSampleValid(sample.f, sample.pdf);

    // Check surface flag
    // TODO: BUG ? Should this be called passing in the lobe instead ?
    shading::BsdfLobe::Type flags = mSlice.getSurfaceFlags(mBsdf, sample.wi);
    isValid &= lobe->matchesFlags(flags);

    // Initialize distance, possibly marking sample as invalid
    sample.distance = (isValid  ?  scene_rdl2::math::sMaxValue  :  BsdfSample::sInvalidDistance);

    if (isValid) {
        pbrTls->mStatistics.incCounter(STATS_BSDF_SAMPLES);
    }

    return isValid;
}

//----------------------------------------------------------------------------

} // namespace pbr
} // namespace moonray

