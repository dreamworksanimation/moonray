// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0


#include "LightSetSampler.isph"
#include <moonray/rendering/pbr/core/Util.isph>

#include <scene_rdl2/render/util/Arena.isph>


//----------------------------------------------------------------------------

// Warning: this must return at least one sample for each light, otherwise
// the algorithm is biased.
inline varying int
computeLightSampleCount(varying int maxLightSamples)
{
    varying int lightSampleCount;

    // TODO: Factor in heuristics based on power, solid angle, open up to
    // other heuristics (pass in a functor ?)
    lightSampleCount = maxLightSamples;
    return lightSampleCount;
}


//----------------------------------------------------------------------------

void
LightSetSampler_init(varying LightSetSampler * uniform lSampler,
                     uniform Arena * uniform arena,
                     const uniform LightSet * uniform lightSet,
                     const varying Bsdf * uniform bsdf,
                     const varying Vec3f &p,
                     varying int maxSamplesPerLight)
{
    lSampler->mLightSet = lightSet,
    lSampler->mMaxSamplesPerLight = maxSamplesPerLight;
    lSampler->mBsdf = bsdf;

    uniform int lightCount = LightSet_getLightCount(lightSet);

    varying int lightSampleCount = computeLightSampleCount(lSampler->mMaxSamplesPerLight);

    lSampler->mLightSampleCount = lightSampleCount;
    lSampler->mInvLightSampleCount = rcp((float)lightSampleCount);

    lSampler->mSampleCount = reduce_max(lightSampleCount) * lightCount;
    lSampler->mInvSampleCount = (lSampler->mSampleCount > 0  ?
                                 rcp((uniform float)lSampler->mSampleCount)  :  0);
}


//----------------------------------------------------------------------------

void
LightSetSampler_sampleIntersectAndEval(
        const varying LightSetSampler * uniform lSampler,
        uniform ShadingTLState * uniform tls,
        const uniform Light * uniform light,
        const uniform LightFilterList * uniform lightFilterList,
        const varying Vec3f &P, const varying Vec3f &cullingNormal, const varying LightFilterRandomValues &filterR, 
        varying float time, const varying Vec3f &r, varying LightSample &sample, varying float rayDirFootprint)
{
    // Draw a sample from light
    LightIntersection isect;
    // TODO: We compute the pdf in there but don't return it. Instead we compute
    // it again in eval()
    if (!Light_sample(light, P, cullingNormal, time, r, sample.wi, isect, rayDirFootprint)) {
        LightSample_setInvalid(&sample);
        return;
    }

    MNRY_ASSERT(isNormalized(sample.wi));

    sample.distance = isect.distance;

    // Evaluate light sample
    sample.Li = Light_eval(light, tls, sample.wi, P, filterR, time, isect, false, lightFilterList, rayDirFootprint,
                           &sample.pdf);
    if (isSampleInvalid(sample.Li, sample.pdf)) {
        LightSample_setInvalid(&sample);
    }
}


//----------------------------------------------------------------------------

