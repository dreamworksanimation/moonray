// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

///
/// @file BsdfOneSampler.cc
/// $Id$
///


#include "BsdfOneSampler.h"

#include <moonray/rendering/shading/bsdf/Bsdfv.h>

#include <scene_rdl2/render/util/Arena.h>
#include <scene_rdl2/render/util/BitUtils.h>


namespace moonray {
namespace pbr {


using namespace scene_rdl2::math;


//----------------------------------------------------------------------------

BsdfOneSampler::BsdfOneSampler(const shading::Bsdf &bsdf, const shading::BsdfSlice &slice) :
    mBsdf(bsdf),
    mSlice(slice)
{
    // Allocate and initialize lobe cdf and index arrays
    mLobeCount = bsdf.getLobeCount();
    memset(mLobeCdf, 0, sizeof(mLobeCdf[0]) * mLobeCount);
    for (int l=0; l < mLobeCount; l++) {
        mLobeIndex[l] = l;
    }

    // Bubble-up the lobes that match the flags at the beginning of the array,
    // and compute their approximated albedo.
    // Note: since we use the approximated albedo, our sampling decisions might
    // be slightly off in cases where a lobe crosses the surface (at grazing
    // angles). In these cases, we may spend slightly more samples than we
    // should.
    // TODO: We should find a way to devote only a single sample to mirror ?
    int last = mLobeCount - 1;
    int current = 0;
    float normalize = 0.0f;
    shading::BsdfLobe::Type flags = mSlice.getFlags();
    while (current <= last) {
        const shading::BsdfLobe *lobe = bsdf.getLobe(mLobeIndex[current]);
        if (lobe->matchesFlags(flags)) {
            // Use the mirror reflection direction assumption to compute the
            // albedo approximaion, which drives our lobe sampling decisions.
            // A negative albedo shouldn't happen, the abs() is for paranoia.
            mLobeCdf[current] = scene_rdl2::math::abs(luminance(lobe->albedo(mSlice)));
            mLobeCdf[current] = max(mLobeCdf[current], sEpsilon);
            normalize += mLobeCdf[current];
            current++;
        } else {
            std::swap(mLobeIndex[current], mLobeIndex[last]);
            mLobeCdf[current] = mLobeCdf[last];
            mLobeCdf[last] = 0;
            last--;
        }
    }
    mLobeCount = last + 1;

    // Handle if no lobes match
    if (mLobeCount == 0) {
        return;
    }

    // Normalize the albedos and integrate into a cdf, all in one pass
    normalize = 1.0f / normalize;
    float cdf = 0.0f;
    for (int l=0; l < mLobeCount; l++) {
        mLobeCdf[l] *= normalize;
        mLobeCdf[l] += cdf;
        cdf = mLobeCdf[l];
    }
    MNRY_ASSERT(isOne(cdf));
}


BsdfOneSampler::~BsdfOneSampler()
{
}


//----------------------------------------------------------------------------

Color
BsdfOneSampler::eval(const scene_rdl2::math::Vec3f &wi, float &pdf,
        LobesContribution* lobesContribution) const
{
    shading::BsdfLobe::Type flags = mSlice.getSurfaceFlags(mBsdf, wi);

    if (lobesContribution != nullptr) {
        lobesContribution->mMatchedLobeCount = 0;
    }
    // Add up all the matching lobe's contributions and pdfs
    Color f(zero);
    pdf = 0.0f;
    float prevCdf = 0.0f;
    for (int l=0; l < mLobeCount; l++) {
        const shading::BsdfLobe * const lobe = mBsdf.getLobe(mLobeIndex[l]);

        // We need to account for lobe pdf, even if the surface flag doesn't match
        float tmpPdf = 0.0f;
        Color color = lobe->eval(mSlice, wi, &tmpPdf);
        if (lobe->matchesFlags(flags)) {
            f += color;
            if (lobesContribution != nullptr) {
                int& index = lobesContribution->mMatchedLobeCount;
                lobesContribution->mFs[index] = color;
                lobesContribution->mLobes[index] = lobe;
                index++;
            }
        }
        pdf += tmpPdf * (mLobeCdf[l] - prevCdf);
        prevCdf = mLobeCdf[l];
    }

    return f;
}

int
BsdfOneSampler::sampleCdf(float r, float &pdf) const
{
    MNRY_ASSERT(mLobeCount > 0);

    const float *ptr = std::upper_bound(mLobeCdf, mLobeCdf + mLobeCount - 1, r);
    const int cdfIndex = ptr - mLobeCdf;
    MNRY_ASSERT(cdfIndex >= 0);
    MNRY_ASSERT(cdfIndex < mLobeCount);

    pdf = (cdfIndex == 0)  ?
        mLobeCdf[cdfIndex] :
        mLobeCdf[cdfIndex] - mLobeCdf[cdfIndex - 1];

    return cdfIndex;
}


const shading::BsdfLobe *
BsdfOneSampler::sampleLobe(float r, float &pdf) const
{
    if (mLobeCount == 0) {
        pdf = 0.0f;
        return nullptr;
    }

    const int cdfIndex = sampleCdf(r, pdf);
    const shading::BsdfLobe *lobe = mBsdf.getLobe(mLobeIndex[cdfIndex]);

    return lobe;
}

Color
BsdfOneSampler::sample(float r0, float r1, float r2, scene_rdl2::math::Vec3f &wi,
        float &pdf, LobesContribution* lobesContribution) const
{
    if (lobesContribution != nullptr) {
        lobesContribution->mMatchedLobeCount = 0;
    }
    // If there is no matching lobes, just return an invalid sample
    if (mLobeCount == 0) {
        pdf = 0.0f;
        return sBlack;
    }

    float lobePdf;
    const int cdfIndex = sampleCdf(r0, lobePdf);
    const shading::BsdfLobe *lobe = mBsdf.getLobe(mLobeIndex[cdfIndex]);

    // Sample that lobe
    Color f = lobe->sample(mSlice, r1, r2, wi, pdf);

    // Treat invalid samples
    if (pdf == 0.0f) {
        return sBlack;
    }

    // Include the discrete probability of picking this lobe
    pdf *= lobePdf;
    if (lobesContribution != nullptr) {
        lobesContribution->mFs[lobesContribution->mMatchedLobeCount] = f;
        lobesContribution->mLobes[lobesContribution->mMatchedLobeCount] = lobe;
        lobesContribution->mMatchedLobeCount++;
    }

    // Note: This code and the whole mirror sampling rules assume the lobe
    // types are mutually exclusive.
    // TODO: Do we change this by sampling mirror lobes separately (see other
    // comment related to that in BsdfOneSampler::init()).
    bool lobeIsNotMirror = !lobe->matchesFlag(shading::BsdfLobe::MIRROR);
    if (lobeIsNotMirror) {

        // Ignore btdfs or brdfs based on geometric surface normal
        shading::BsdfLobe::Type flags = mSlice.getSurfaceFlags(mBsdf, wi);

        // Compute the overall bsdf for the sampled direction

        // First make sure to re-test the sampled lobe with new flags
        // TODO: bug, like in PBRT, we're not testing this for mirror lobes
        if (!lobe->matchesFlags(flags)) {
            f = sBlack;
        }

        // Then add up all the other lobe's contributions
        float prevCdf = 0.0f;
        for (int cI = 0; cI < mLobeCount; ++cI) {
            if (cI != cdfIndex) {
                lobe = mBsdf.getLobe(mLobeIndex[cI]);

                // We need to account for lobe pdf, even if the surface flag
                // doesn't match
                float tmpPdf = 0.0f;
                Color color = lobe->eval(mSlice, wi, &tmpPdf);
                if (lobe->matchesFlags(flags)) {
                    f += color;
                    if (lobesContribution != nullptr) {
                        lobesContribution->mFs[lobesContribution->mMatchedLobeCount] = color;
                        lobesContribution->mLobes[lobesContribution->mMatchedLobeCount] = lobe;
                        lobesContribution->mMatchedLobeCount++;
                    }
                }
                pdf += tmpPdf * (mLobeCdf[cI] - prevCdf);
            }
            prevCdf = mLobeCdf[cI];
        }
    }

    return f;
}

// ----------------------------------------------------------------------------

BsdfOneSamplervOneLane::BsdfOneSamplervOneLane(const shading::Bsdfv &bsdfv, const shading::BsdfSlicev &slicev,
                                               int lane): mLane(lane)
{
    BsdfOneSamplerv_init(&mBsampler, &bsdfv, &slicev, mLane);
}

scene_rdl2::math::Color
BsdfOneSamplervOneLane::eval(const scene_rdl2::math::Vec3f &wi, float &pdf, LobesContribution *lobesContribution) const
{
    const ispc::Col3f c = BsdfOneSamplerv_eval(&mBsampler,
                                               reinterpret_cast<const ispc::Vec3f &>(wi),
                                               pdf,
                                               reinterpret_cast<ispc::LobesContribution *>(lobesContribution),
                                               mLane);
    return *reinterpret_cast<const scene_rdl2::math::Color *>(&c);
}

scene_rdl2::math::Color
BsdfOneSamplervOneLane::sample(float r0, float r1, float r2, scene_rdl2::math::Vec3f &wi, float &pdf,
                               LobesContribution *lobesContribution) const
{
    const ispc::Col3f c = BsdfOneSamplerv_sample(&mBsampler,
                                                 r0,
                                                 r1,
                                                 r2,
                                                 reinterpret_cast<ispc::Vec3f &>(wi),
                                                 pdf,
                                                 reinterpret_cast<ispc::LobesContribution *>(lobesContribution),
                                                 mLane);
    return *reinterpret_cast<const scene_rdl2::math::Color *>(&c);
}

} // namespace pbr
} // namespace moonray

