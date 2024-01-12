// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

///
/// @file LightSetSampler.cc
/// $Id$
///


#include "LightSetSampler.h"
#include <moonray/rendering/pbr/core/Util.h>

#include <scene_rdl2/render/util/Arena.h>
#include <scene_rdl2/render/util/BitUtils.h>


using namespace scene_rdl2::math;

namespace moonray {
namespace pbr {

//----------------------------------------------------------------------------

// Warning: this must return at least one sample for each light, otherwise
// the algorithm is biased.
static finline int
computeLightSampleCount(int maxLightSamples)
{
    int lightSampleCount;

    // TODO: Factor in heuristics based on power, solid angle, open up to
    // other heuristics (pass in a functor ?)
    lightSampleCount = maxLightSamples;
    return lightSampleCount;
}


//----------------------------------------------------------------------------

LightSetSampler::LightSetSampler(scene_rdl2::alloc::Arena *arena, const LightSet &lightSet, const shading::Bsdf &bsdf,
        const Vec3f &p, int maxSamplesPerLight) :
    mLightSet(lightSet),
    mMaxSamplesPerLight(maxSamplesPerLight),
    mSampleCount(0),
    mInvSampleCount(0),
    mLightSampleCount(0),
    mInvLightSampleCount(0),
    mBsdf(&bsdf)
{
    const int lightCount = mLightSet.getLightCount();

    mLightSampleCount = computeLightSampleCount(mMaxSamplesPerLight);
    mInvLightSampleCount = rcp(float(mLightSampleCount));

    mSampleCount = lightCount * mLightSampleCount;
    mInvSampleCount = (mSampleCount > 0  ?  rcp(float(mSampleCount))  :  0);
}


LightSetSampler::~LightSetSampler()
{
}


//----------------------------------------------------------------------------

void
LightSetSampler::sampleIntersectAndEval(mcrt_common::ThreadLocalState* tls,
        const Light* light, const LightFilterList* lightFilterList,
        const Vec3f &P, const Vec3f *N, const LightFilterRandomValues& filterR, float time,
        const Vec3f& r, LightSample &sample, float rayDirFootprint) const
{
    // Draw a sample from light
    LightIntersection isect;
    if (!light->sample(P, N, time, r, sample.wi, isect, rayDirFootprint)) {
        sample.setInvalid();
        return;
    }
    sample.distance = isect.distance;

    // Evaluate light sample
    sample.Li = light->eval(tls, sample.wi, P, filterR, time, isect, false, lightFilterList, rayDirFootprint,
                            &sample.pdf);
    if (isSampleInvalid(sample.Li, sample.pdf)) {
        sample.setInvalid();
    }
}

//----------------------------------------------------------------------------

} // namespace pbr
} // namespace moonray

