// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

///
/// @file VolumeEmissionBsdfSampler.h
///

#pragma once

#include <moonray/rendering/pbr/Types.h>
#include <moonray/rendering/pbr/core/RayState.h>
#include <moonray/rendering/pbr/sampler/IntegratorSample.h>
#include <moonray/rendering/pbr/sampler/SequenceID.h>

#include <scene_rdl2/common/math/Vec3.h>

namespace moonray {
namespace pbr {

/// Utility class that wraps multiple samplers used in volume emission
/// bsdf sample strategy (for easier sampling management)
class VolumeEmissionBsdfSampler {
public:
    VolumeEmissionBsdfSampler(const Subpixel& sp, const PathVertex& pv,
            int bsdfSamples, int emissionIndex, bool highQuality,
            unsigned sequenceID):
        mNonMirrorDepth(pv.nonMirrorDepth)
    {
        if (highQuality) {
            int samplesSoFar = sp.mSubpixelIndex * bsdfSamples;
            SequenceIDIntegrator indexSid(
                pv.nonMirrorDepth, sp.mPixel, 0,
                SequenceType::VolumeEmission,
                SequenceType::Bsdf,
                SequenceType::IndexSelection,
                sequenceID, emissionIndex);
            mSelectLobeSamples.resume(indexSid, samplesSoFar);

            SequenceIDIntegrator bsdfSid(
                pv.nonMirrorDepth, sp.mPixel, 0,
                SequenceType::VolumeEmission,
                SequenceType::Bsdf,
                sequenceID, emissionIndex);
            mWiSamples.resume(bsdfSid, samplesSoFar);
        } else {
            int samplesCount = bsdfSamples;
            SequenceIDIntegrator indexSid(
                pv.nonMirrorDepth, sp.mPixel, sp.mSubpixelIndex,
                SequenceType::VolumeEmission,
                SequenceType::Bsdf,
                SequenceType::IndexSelection,
                sequenceID, emissionIndex);
            mSelectLobeSamples.restart(indexSid, samplesCount);

            SequenceIDIntegrator bsdfSid(
                pv.nonMirrorDepth, sp.mPixel, sp.mSubpixelIndex,
                SequenceType::VolumeEmission,
                SequenceType::Bsdf,
                sequenceID, emissionIndex);
            mWiSamples.restart(bsdfSid, samplesCount);
        }
    }

    // constructor for volume scattering samples
    VolumeEmissionBsdfSampler(const Subpixel& sp, const PathVertex& pv,
            int nPhaseSamples, int nScatterSamples, int emissionIndex,
            bool highQuality, unsigned sequenceID):
        mNonMirrorDepth(pv.nonMirrorDepth)
    {
        if (highQuality) {
            int samplesSoFar = sp.mSubpixelIndex *
                nPhaseSamples * nScatterSamples;
            SequenceIDIntegrator indexSid(
                pv.nonMirrorDepth, sp.mPixel, 0,
                SequenceType::VolumeEmission,
                SequenceType::VolumePhase,
                SequenceType::VolumeScattering,
                SequenceType::IndexSelection,
                sequenceID, emissionIndex);
            mSelectLobeSamples.resume(indexSid, samplesSoFar);

            SequenceIDIntegrator bsdfSid(
                pv.nonMirrorDepth, sp.mPixel, 0,
                SequenceType::VolumeEmission,
                SequenceType::VolumePhase,
                SequenceType::VolumeScattering,
                sequenceID, emissionIndex);
            mWiSamples.resume(bsdfSid, samplesSoFar);
        } else {
            int samplesCount = nPhaseSamples * nScatterSamples;
            SequenceIDIntegrator indexSid(
                pv.nonMirrorDepth, sp.mPixel, sp.mSubpixelIndex,
                SequenceType::VolumeEmission,
                SequenceType::VolumePhase,
                SequenceType::VolumeScattering,
                SequenceType::IndexSelection,
                sequenceID, emissionIndex);
            mSelectLobeSamples.restart(indexSid, samplesCount);

            SequenceIDIntegrator bsdfSid(
                pv.nonMirrorDepth, sp.mPixel, sp.mSubpixelIndex,
                SequenceType::VolumeEmission,
                SequenceType::VolumePhase,
                SequenceType::VolumeScattering,
                sequenceID, emissionIndex);
            mWiSamples.restart(bsdfSid, samplesCount);
        }
    }

    void getSample(float& uLobe, scene_rdl2::math::Vec2f& uWi) const
    {
        mSelectLobeSamples.getSample(&uLobe, mNonMirrorDepth);
        mWiSamples.getSample(&uWi[0], mNonMirrorDepth);
    }

private:
    // use this 1d sample to sample lobe index (BsdfOneSampler requests)
    IntegratorSample1D mSelectLobeSamples;
    // use this 2d sample to sample wi direction with pdf in solid angle domain
    IntegratorSample2D mWiSamples;
    int mNonMirrorDepth;
};

} // namespace pbr
} // namespace moonray

