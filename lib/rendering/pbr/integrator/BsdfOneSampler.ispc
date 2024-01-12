// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

/// @file BsdfOneSampler.ispc

#include "BsdfOneSampler.isph"

#include <scene_rdl2/common/platform/IspcUtil.isph>

//
// A Note about the vector port from scalar.
// There are two arrays to keep in mind, the lobe array in the bsdf and
// the cdf value array in this class.  They are indexed separately.  In
// order to keep this straight, indices into the cdf array are called "cdfIndex" or "cI"
// while indices into the bsdf lobe array are called "bsdfIndex" or "bI".
// Cdf indices run from [0, mLobeCount) while bsdf lobe indices run
// from [0, Bsdf_getLobeCount(bsdf)).  The mLobeIndex member is used to map
// cdf index to bsdf index.
//

void
BsdfOneSampler_init(varying BsdfOneSampler * uniform this,
                    const varying Bsdf * uniform bsdf,
                    const varying BsdfSlice * uniform slice)
{
    this->mBsdf = bsdf;
    this->mSlice = slice;

    // Allocate and initialize lobe cdf and index arrays
    //    mLobeCount: total number of active lobes (see adjustment below)
    //    mLobeIndex[cdfIndex]: bsdf lobe index of mLobeCdf[cdfIndex]
    //    mLobeCdf[cdfIndex]: cdf value
    this->mLobeCount = Bsdf_getLobeCount(bsdf);
    int cdfIndex = 0;
    for (uniform int bsdfIndex = 0; bsdfIndex < Bsdf_getLobeCount(bsdf); ++bsdfIndex) {
        this->mLobeCdf[bsdfIndex] = 0.f; // ok to use bsdfIndex when initializing mLobeCdf
        const varying BsdfLobe * uniform lobe = Bsdf_getLobe(bsdf, bsdfIndex);
        if (!isActive(lobe)) {
            --this->mLobeCount;
        } else {
            this->mLobeIndex[cdfIndex++] = bsdfIndex;
        }
    }

    // Bubble-up the lobes that match the flags at the beginning of the array,
    // and compute their approximated albedo.
    // Note: since we use the approximated albedo, our sampling decisions might
    // be slightly off in cases where a lobe crosses the surface (at grazing
    // angles). In these cases, we may spend slightly more samples than we
    // should.
    // TODO: We should find a way to devote only a single sample to mirror ?
    int last = this->mLobeCount - 1;
    int current = 0;
    float normalize = 0.f;
    while (current <= last) {
        const int bsdfIndex = this->mLobeIndex[current];
        const varying BsdfLobe * varying lobe = Bsdf_getLobe(bsdf, bsdfIndex);
        if (BsdfLobe_matchesFlags(lobe, BsdfSlice_getFlags(slice))) {
            foreach_unique (lobePtr in lobe) {
                // Use the mirror reflection direction assumption to compute the
                // albedo approximaion, which drives our lobe sampling decisions.
                // A negative albedo shouldn't happen, the abs() is for paranoia.
                this->mLobeCdf[current] = abs(luminance(BsdfLobe_albedo(lobePtr, *slice)));
                this->mLobeCdf[current] = max(this->mLobeCdf[current], sEpsilon);
            }
            normalize += this->mLobeCdf[current];
            ++current;
        } else {
            // swap(this->mLobeIndex[current], this->mLobeIndex[last])
            const uint8_t tmp = this->mLobeIndex[last];
            this->mLobeIndex[last] = this->mLobeIndex[current];
            this->mLobeIndex[current] = tmp;

            this->mLobeCdf[current] = this->mLobeCdf[last];
            this->mLobeCdf[last] = 0.f;
            last--;
        }
    }
    // mLobeCount: now set to total number of active lobes that match the slice flags
    this->mLobeCount = last + 1;

    // Handle if no lobes match
    if (this->mLobeCount <= 0) {
        return;
    }

    // Normalize the albedos and integrate into a cdf, all in one pass
    normalize = 1.0f / normalize;
    float cdf = 0.0f;
    for (int l = 0; l < this->mLobeCount; ++l) {
        this->mLobeCdf[l] *= normalize;
        this->mLobeCdf[l] += cdf;
        cdf = this->mLobeCdf[l];
    }
    MNRY_ASSERT(isOne(cdf));
}

varying Color
BsdfOneSampler_eval(const varying BsdfOneSampler * uniform this,
                    const varying Vec3f &wi,
                    varying float &pdf,
                    uniform LobesContribution * varying lobesContribution)
{
    BsdfLobeType flags = BsdfSlice_getSurfaceFlags(this->mSlice, *this->mBsdf, wi);

    if (lobesContribution != nullptr) {
        lobesContribution->mMatchedLobeCount = 0;
    }
    // Add up all the matching lobe's contributions and pdfs
    Color f = sBlack;
    pdf = 0.f;
    float prevCdf = 0.f;
    for (int cdfIndex = 0; cdfIndex < this->mLobeCount; ++cdfIndex) {
        const int bsdfIndex = this->mLobeIndex[cdfIndex];
        const varying BsdfLobe * varying lobe = Bsdf_getLobe(this->mBsdf, bsdfIndex);
        // We need to account for lobe pdf, even if the surface flag doesn't match
        float tmpPdf = 0.f;
        Color color = sBlack;
        foreach_unique (lobePtr in lobe) {
            color = BsdfLobe_eval(lobePtr, *this->mSlice, wi, &tmpPdf);
        }
        if (BsdfLobe_matchesFlags(lobe, flags)) {
            f = f + color;
            if (lobesContribution != nullptr) {
                int index = lobesContribution->mMatchedLobeCount;
                lobesContribution->mFs[index] = color;
                lobesContribution->mLobes[index] = lobe;
                lobesContribution->mMatchedLobeCount++;
            }
        }
        pdf = pdf + tmpPdf * (this->mLobeCdf[cdfIndex] - prevCdf);
        prevCdf = this->mLobeCdf[cdfIndex];
    }
    return f;
}

static varying int
BsdfOneSampler_sampleCdf(const varying BsdfOneSampler * uniform this,
                         varying float r,
                         varying float &pdf)
{
    MNRY_ASSERT(this->mLobeCount > 0);

    // My theory is that for small lobe counts (< 16), a linear search
    // might actually outperform a binary search.
    const varying float * varying ptr = &this->mLobeCdf[0];
    const varying float * varying end = ptr + this->mLobeCount - 1;
    while (ptr < end && *ptr < r) {
        ++ptr;
    }
    const int cdfIndex = ptr - &this->mLobeCdf[0];
    MNRY_ASSERT(cdfIndex >= 0);
    MNRY_ASSERT(cdfIndex < this->mLobeCount);

    pdf = (cdfIndex == 0) ?
        this->mLobeCdf[cdfIndex] :
        this->mLobeCdf[cdfIndex] - this->mLobeCdf[cdfIndex - 1];

    return cdfIndex;
}

const varying BsdfLobe * varying
BsdfOneSampler_sampleLobe(const varying BsdfOneSampler * uniform this,
                          varying float r,
                          varying float &pdf)
{
    if (this->mLobeCount == 0) {
        pdf = 0.f;
        return nullptr;
    }

    const int cdfIndex = BsdfOneSampler_sampleCdf(this, r, pdf);
    const varying BsdfLobe * varying lobe = Bsdf_getLobe(this->mBsdf, this->mLobeIndex[cdfIndex]);

    return lobe;
}

varying Color
BsdfOneSampler_sample(const varying BsdfOneSampler * uniform this,
                      varying float r0,
                      varying float r1,
                      varying float r2,
                      varying Vec3f &wi,
                      varying float &pdf,
                      uniform LobesContribution * varying lobesContribution)
{
    if (lobesContribution != nullptr) {
        lobesContribution->mMatchedLobeCount = 0;
    }
    // If there are no matching lobes, just return an invalid sample
    if (this->mLobeCount == 0) {
        pdf = 0.f;
        return sBlack;
    }

    float lobePdf;
    const int cdfIndex = BsdfOneSampler_sampleCdf(this, r0, lobePdf);
    const varying BsdfLobe * varying lobe = Bsdf_getLobe(this->mBsdf, this->mLobeIndex[cdfIndex]);

    // Sample that lobe
    Color f = sBlack;
    foreach_unique(lobePtr in lobe) {
        f = BsdfLobe_sample(lobePtr, *this->mSlice, r1, r2, wi, pdf);
    }

    // Treat invalid samples
    if (pdf == 0.f) {
        return sBlack;
    }

    // Include the discrete probability of picking this lobe
    pdf *= lobePdf;
    if (lobesContribution != nullptr) {
        lobesContribution->mFs[lobesContribution->mMatchedLobeCount] = f;
        lobesContribution->mLobes[lobesContribution->mMatchedLobeCount] = lobe;
        lobesContribution->mMatchedLobeCount++;
    }

    // Note: This code and the whole mirror sampling rules assume the
    // lobe types are mutually exclusive.
    // TODO: Do we change this by sampling mirror lobes separately (see other
    // comment related to that in BsdfOneSampler_init()).
    const bool lobeIsNotMirror = !BsdfLobe_matchesFlags(lobe, BSDF_LOBE_TYPE_MIRROR);
    if (lobeIsNotMirror) {

        // Ignore btdfs or brdfs based on geometric surface normal
        BsdfLobeType flags = BsdfSlice_getSurfaceFlags(this->mSlice, *this->mBsdf, wi);

        // Compute the overall bsdf for the sampled direction

        // First make sure to re-test the sampled lobe with new flags
        // TODO: bug, like in PBRT, we're not testing this for mirror lobes
        if (!BsdfLobe_matchesFlags(lobe, flags)) {
            f = sBlack;
        }

        // Then add up all the other lobe's contributions
        float prevCdf = 0.f;
        for (int cI = 0; cI < this->mLobeCount; ++cI) {
            if (cI != cdfIndex) {
                lobe = Bsdf_getLobe(this->mBsdf, this->mLobeIndex[cI]);

                // We need to account for lobe pdf, even if the surface flag
                // doesn't match
                float tmpPdf = 0.f;
                Color color = sBlack;
                foreach_unique (lobePtr in lobe) {
                    color = BsdfLobe_eval(lobePtr, *this->mSlice, wi, &tmpPdf);
                }
                if (BsdfLobe_matchesFlags(lobe, flags)) {
                    f = f + color;
                    if (lobesContribution != nullptr) {
                        lobesContribution->mFs[lobesContribution->mMatchedLobeCount] = color;
                        lobesContribution->mLobes[lobesContribution->mMatchedLobeCount] = lobe;
                        lobesContribution->mMatchedLobeCount++;
                    }
                }
                pdf += tmpPdf * (this->mLobeCdf[cI] - prevCdf);
            }
            prevCdf = this->mLobeCdf[cI];
        }
    }

    return f;
}

// ----------------------------------------------------------------------------
// exports to support single lane sampling and evaluation from c++

export void
#pragma ignore warning(all)
BsdfOneSamplerv_init(varying BsdfOneSampler * uniform bSampler,
#pragma ignore warning(all)
                     const varying Bsdf * uniform bsdf,
#pragma ignore warning(all)
                     const varying BsdfSlice * uniform slice,
                     uniform int lane)
{
    if (programIndex == lane) {
        BsdfOneSampler_init(bSampler, bsdf, slice);
    }
}

export uniform Color
#pragma ignore warning(all)
BsdfOneSamplerv_eval(const varying BsdfOneSampler * uniform bSampler,
                     const uniform Vec3f &wi,
                     uniform float &pdf,
#pragma ignore warning(all)
                     uniform LobesContribution * uniform lobesContribution,
                     uniform int lane)
{
    Color fv;
    float pdfv;
    Vec3f wiv = wi;
    uniform LobesContribution * varying lobesContributionv = lobesContribution;
    if (programIndex == lane) {
        fv = BsdfOneSampler_eval(bSampler, wiv, pdfv, lobesContributionv);
    }

    pdf = extract(pdfv, lane);
    uniform Color f;
    f.r = extract(fv.r, lane);
    f.g = extract(fv.g, lane);
    f.b = extract(fv.b, lane);

    return f;
}

export uniform Color
#pragma ignore warning(all)
BsdfOneSamplerv_sample(const varying BsdfOneSampler * uniform bSampler,
                       uniform float r1, uniform float r2, uniform float r3,
                       uniform Vec3f &wi,
                       uniform float &pdf,
#pragma ignore warning(all)
                       uniform LobesContribution * uniform lobesContribution,
                       uniform int lane)
{
    const float r1v = r1;
    const float r2v = r2;
    const float r3v = r3;
    Vec3f wiv;
    Color fv;
    float pdfv;
    uniform LobesContribution * varying lobesContributionv = lobesContribution;
    if (programIndex == lane) {
        fv = BsdfOneSampler_sample(bSampler, r1v, r2v, r3v, wiv, pdfv, lobesContributionv);
    }

    wi.x = extract(wiv.x, lane);
    wi.y = extract(wiv.y, lane);
    wi.z = extract(wiv.z, lane);
    pdf = extract(pdfv, lane);
    uniform Color f;
    f.r = extract(fv.r, lane);
    f.g = extract(fv.g, lane);
    f.b = extract(fv.b, lane);

    return f;
}

