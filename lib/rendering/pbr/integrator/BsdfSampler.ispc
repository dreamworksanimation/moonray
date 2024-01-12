// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0


#include "BsdfSampler.isph"

#include <scene_rdl2/common/platform/IspcUtil.isph>
#include <scene_rdl2/render/util/Arena.isph>

#pragma ignore warning(all)
ISPC_UTIL_EXPORT_STRUCT_TO_HEADER(BsdfSampler);
#pragma ignore warning(all)
ISPC_UTIL_EXPORT_STRUCT_TO_HEADER(BsdfSample);

//----------------------------------------------------------------------------

// Warning: this must return at least one sample for each lobe, otherwise
// the algorithm is biased (unless the lobe has zero albedo, but in this case
// the shader should never have added the lobe in the first place).
inline varying int
computeLobeSampleCount(const varying BsdfLobe * uniform lobe,
        const varying BsdfSlice &slice, varying int maxLobeSamples,
        uniform int lobeCount)
{
    int sampleCount;

    if (BsdfLobe_matchesFlags(lobe, BSDF_LOBE_TYPE_ALL_MIRROR)  ||  maxLobeSamples == 1) {
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
        float lumAlbedo = abs(luminance(BsdfLobe_albedo(lobe, slice)));
        sampleCount = (int)(ceil(lumAlbedo * maxLobeSamples * lobeCount));
        sampleCount = min(sampleCount, maxLobeSamples * lobeCount);
#endif
    }

    // We need at least 1 sample per lobe otherwise this algorithm is biased
    MNRY_ASSERT(maxLobeSamples == 0  ||  sampleCount > 0);

    return sampleCount;
}


//----------------------------------------------------------------------------

void
BsdfSampler_init(varying BsdfSampler * uniform bSampler,
                 uniform Arena * uniform arena,
                 const varying Bsdf &bsdf,
                 const varying BsdfSlice &slice,
                 varying int maxSamplesPerLobe,
                 varying bool doIndirect)
{
    bSampler->mBsdf = &bsdf;
    bSampler->mSlice = &slice;
    bSampler->mMaxSamplesPerLobe = maxSamplesPerLobe;
    bSampler->mDoIndirect = doIndirect;
    bSampler->mLobeCount = Bsdf_getLobeCount(bSampler->mBsdf);
    bSampler->mSampleCount = 0;
    bSampler->mSampleCountVarying = 0;

    // Allocate and initialize lobe arrays
    bSampler->mLobeIndex = (uniform uint8_t * uniform)
            Arena_allocArray(arena, bSampler->mLobeCount, sizeof(uniform uint8_t));
    bSampler->mLobeSampleCount = (varying int * uniform)
            Arena_allocArray(arena, bSampler->mLobeCount, sizeof(varying int));
    bSampler->mInvLobeSampleCount = (varying float * uniform)
            Arena_allocArray(arena, bSampler->mLobeCount, sizeof(varying float));

    memset(bSampler->mLobeSampleCount, 0, bSampler->mLobeCount * sizeof(varying int));
    memset(bSampler->mInvLobeSampleCount, 0, bSampler->mLobeCount * sizeof(varying float));

    // Compute sample count per lobe, and keep track of lobes with samples
    uniform int lKeep = 0;
    const varying BsdfLobeType flags = bSampler->mSlice->mFlags;
    for (uniform int l = 0; l < bSampler->mLobeCount; ++l) {
        const varying BsdfLobe * uniform lobe = Bsdf_getLobe(bSampler->mBsdf, l);
        if (isActive(lobe) && BsdfLobe_matchesFlags(lobe, flags)) {
            int sampleCount = computeLobeSampleCount(lobe, *bSampler->mSlice,
                    maxSamplesPerLobe, bSampler->mLobeCount);
            bSampler->mLobeToSampleIndex[lKeep] = bSampler->mSampleCount;
            bSampler->mSampleCount += reduce_max(sampleCount);
            bSampler->mSampleCountVarying += sampleCount;
            bSampler->mLobeIndex[lKeep] = l;
            bSampler->mLobeSampleCount[lKeep] = sampleCount;
            bSampler->mInvLobeSampleCount[lKeep] = rcp((float)sampleCount);
            ++lKeep;
        }
    }
    bSampler->mLobeCount = lKeep;

    MNRY_ASSERT(bSampler->mSampleCount >= lKeep);

    if (lKeep == 0) {
        MNRY_ASSERT(bSampler->mSampleCount == 0);
    }
}


//----------------------------------------------------------------------------

