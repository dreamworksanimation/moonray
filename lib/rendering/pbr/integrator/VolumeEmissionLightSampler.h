// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

///
/// @file VolumeEmissionLightSampler.h
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
/// light sample strategy (for easier sampling management)
class VolumeEmissionLightSampler {
public:
    // constructor for hard surface samples
    VolumeEmissionLightSampler(const Subpixel& sp, const PathVertex& pv,
            int nLightSamples, int emissionIndex, bool highQuality,
            unsigned sequenceID):
        mNonMirrorDepth(pv.nonMirrorDepth)
    {
        if (highQuality) {
            int samplesSoFar = sp.mSubpixelIndex * nLightSamples;
            SequenceIDIntegrator xyzSid(
                pv.nonMirrorDepth, sp.mPixel, 0,
                SequenceType::VolumeEmission,
                SequenceType::Light,
                sequenceID, emissionIndex);
            mSamples.resume(xyzSid, samplesSoFar);
        } else {
            int samplesCount = nLightSamples;
            SequenceIDIntegrator xyzSid(
                pv.nonMirrorDepth, sp.mPixel, sp.mSubpixelIndex,
                SequenceType::VolumeEmission,
                SequenceType::Light,
                sequenceID, emissionIndex);
            mSamples.restart(xyzSid, samplesCount);
        }
    }

    // constructor for bssrdf subsurface scattering samples
    VolumeEmissionLightSampler(const Subpixel& sp, const PathVertex& pv,
            int nLightSamples, int emissionIndex, bool highQuality,
            int subsurfaceSplitFactor, int subsurfaceIndex,
            unsigned sssSampleID, bool isLocal):
        mNonMirrorDepth(pv.nonMirrorDepth)
    {
        int localFactor = isLocal ?
            static_cast<int>(SequenceType::BssrdfLocalLight) :
            static_cast<int>(SequenceType::BssrdfGlobalLight);
        if (highQuality) {
            // subsurfaceIndex range from [0, mBssrdfSamples)
            // subsurfaceSplitFactor = mBssrdfSamples
            // Using SSS / SP / L
            // Note: using SP / SSS / L caused more correlation
            int samplesSoFar = nLightSamples * (
                subsurfaceIndex * sp.mPixelSamples + sp.mSubpixelIndex);
            SequenceIDIntegrator xyzSid(
                pv.nonMirrorDepth, sp.mPixel, 0,
                SequenceType::VolumeEmission,
                SequenceType::Light,
                localFactor, sssSampleID, emissionIndex);
            mSamples.resume(xyzSid, samplesSoFar);
        } else {
            int samplesCount = nLightSamples;
            SequenceIDIntegrator xyzSid(
                pv.nonMirrorDepth, sp.mPixel,
                sp.mSubpixelIndex * subsurfaceSplitFactor + subsurfaceIndex,
                SequenceType::VolumeEmission,
                SequenceType::Light,
                localFactor, sssSampleID, emissionIndex);
            mSamples.restart(xyzSid, samplesCount);
        }
    }

    // constructor for random walk subsurface scattering samples
    VolumeEmissionLightSampler(const Subpixel& sp, const PathVertex& pv,
            int emissionIndex, int subsurfaceSplitFactor, int subsurfaceIndex,
            unsigned sssSampleID):
        mNonMirrorDepth(pv.nonMirrorDepth)
    {
        SequenceIDIntegrator xyzSid(
            pv.nonMirrorDepth, sp.mPixel, 0,
            SequenceType::VolumeEmission,
            SequenceType::Light,
            SequenceType::BssrdfLocalLight,
            sssSampleID, emissionIndex);
        mSamples.resume(xyzSid, subsurfaceIndex);
    }

    // constructor for volume scattering samples
    VolumeEmissionLightSampler(const Subpixel& sp, const PathVertex& pv,
            int nLightSamples, int nScatterSamples, int emissionIndex,
            bool highQuality, unsigned sequenceID):
        mNonMirrorDepth(pv.nonMirrorDepth)
    {
        if (highQuality) {
            int samplesSoFar = sp.mSubpixelIndex *
                nLightSamples * nScatterSamples;
            SequenceIDIntegrator xyzSid(
                pv.nonMirrorDepth, sp.mPixel, 0,
                SequenceType::VolumeEmission,
                SequenceType::Light,
                SequenceType::VolumeScattering,
                sequenceID, emissionIndex);
            mSamples.resume(xyzSid, samplesSoFar);
        } else {
            int samplesCount = nLightSamples * nScatterSamples;
            SequenceIDIntegrator xyzSid(
                pv.nonMirrorDepth, sp.mPixel, sp.mSubpixelIndex,
                SequenceType::VolumeEmission,
                SequenceType::Light,
                SequenceType::VolumeScattering,
                sequenceID, emissionIndex);
            mSamples.restart(xyzSid, samplesCount);
        }
    }

    // constructor for volume multiple scattering approximation
    VolumeEmissionLightSampler(const Subpixel& sp, const PathVertex& pv,
            int nScatterSamples, int emissionIndex, unsigned sequenceID):
        mNonMirrorDepth(pv.nonMirrorDepth)
    {

        SequenceIDIntegrator xyzSid(
            pv.nonMirrorDepth, pv.volumeDepth, sp.mPixel, sp.mSubpixelIndex,
            SequenceType::VolumeEmission,
            SequenceType::Light,
            SequenceType::VolumeScattering,
            sequenceID, emissionIndex);
        mSamples.restart(xyzSid, nScatterSamples);
    }


    void getSample(scene_rdl2::math::Vec3f& u3d) const
    {
        mSamples.getSample(&u3d[0], mNonMirrorDepth);
    }

private:
    IntegratorSample3D mSamples;
    int mNonMirrorDepth;
};

} // namespace pbr
} // namespace moonray

