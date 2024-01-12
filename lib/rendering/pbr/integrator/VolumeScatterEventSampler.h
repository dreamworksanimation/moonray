// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

///
/// @file VolumeScatterEventSampler.h
///

#pragma once

#include <moonray/rendering/pbr/Types.h>
#include <moonray/rendering/pbr/core/RayState.h>
#include <moonray/rendering/pbr/sampler/IntegratorSample.h>
#include <moonray/rendering/pbr/sampler/SequenceID.h>

namespace moonray {
namespace pbr {

/// Draw 1D samples to determine scatter event along the ray.
/// Since it's used in multiple places with different sequence factors
/// Wrap into one utility for easier management
class VolumeScatterEventSampler {
public:
    VolumeScatterEventSampler() = default;

    VolumeScatterEventSampler(const Subpixel& sp, const PathVertex& pv,
            int nSamples, bool highQuality, SequenceType scatterStrategy,
            SequenceType wiStrategy, unsigned sequenceID,
            SequenceType emitterType, int emitterIndex = -1)
    {
        init(sp, pv, nSamples, highQuality, scatterStrategy, wiStrategy,
            sequenceID, emitterType, emitterIndex);
    }

    void init(const Subpixel& sp, const PathVertex& pv,
            int nSamples, bool highQuality, SequenceType scatterStrategy,
            SequenceType wiStrategy, unsigned sequenceID,
            SequenceType emitterType, int emitterIndex = -1)
    {
        int subpiexelIndex = highQuality ? 0 : sp.mSubpixelIndex;
        SequenceIDIntegrator scatterEventSid(
            pv.nonMirrorDepth, sp.mPixel, subpiexelIndex,
            SequenceType::VolumeScattering,
            scatterStrategy,
            wiStrategy,
            emitterType,
            emitterIndex, sequenceID);
        if (highQuality) {
            mScatterEventSamples.resume(scatterEventSid,
                sp.mSubpixelIndex * nSamples);
        } else {
            mScatterEventSamples.restart(scatterEventSid, nSamples);
        }
    }

    float getSample(int depth) const
    {
        float u;
        mScatterEventSamples.getSample(&u, depth);
        return u;
    }

private:
    IntegratorSample1D mScatterEventSamples;
};

} // namespace pbr
} // namespace moonray

