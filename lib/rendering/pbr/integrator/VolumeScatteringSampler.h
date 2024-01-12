// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

///
/// @file VolumeScatteringSampler.h
///

#pragma once

#include "VolumeScatterEventSampler.h"
#include <moonray/rendering/pbr/Types.h>
#include <moonray/rendering/pbr/core/RayState.h>
#include <moonray/rendering/pbr/sampler/IntegratorSample.h>
#include <moonray/rendering/pbr/sampler/SequenceID.h>

#include <scene_rdl2/common/math/Vec3.h>

namespace moonray {
namespace pbr {

/// Utility class that wraps multiple samplers used in volume scattering
/// integration estimation (for easier sampling management)
class VolumeScatteringSampler {
public:
    VolumeScatteringSampler(const Subpixel& sp, const PathVertex& pv,
            int samplesPerLight, int scatterSampleCount, const Light& light,
            bool highQuality, bool doEquiAngular, unsigned sequenceID):
        mNonMirrorDepth(pv.nonMirrorDepth),
        mDoEquiAngular(doEquiAngular)
    {
        int samplesCount = samplesPerLight * scatterSampleCount;

        mDistanceSamples.init(sp, pv, samplesCount, highQuality,
            SequenceType::VolumeDistance, SequenceType::Light,
            sequenceID, SequenceType::Light, light.getHash());
        if (mDoEquiAngular) {
            mEquiAngularSamples.init(sp, pv, samplesCount, highQuality,
                SequenceType::VolumeEquiAngular, SequenceType::Light,
                sequenceID, SequenceType::Light, light.getHash());
        }

        if (highQuality) {
            int samplesSoFar = sp.mSubpixelIndex * samplesCount;
            SequenceIDIntegrator distanceLightSid(
                pv.nonMirrorDepth, sp.mPixel, 0,
                SequenceType::VolumeDistance,
                SequenceType::VolumeScattering,
                SequenceType::Light,
                sequenceID, light.getHash());
            mDistanceLightSamples.resume(distanceLightSid, samplesSoFar);
            SequenceIDIntegrator distanceLightFilterSid(
                pv.nonMirrorDepth, sp.mPixel, 0,
                SequenceType::VolumeDistance,
                SequenceType::VolumeScattering,
                SequenceType::LightFilter,
                sequenceID, light.getHash());
            mDistanceLightFilterSamples.resume(distanceLightFilterSid, samplesSoFar);
            SequenceIDIntegrator distanceLightFilter3DSid(
                pv.nonMirrorDepth, sp.mPixel, 0,
                SequenceType::VolumeDistance,
                SequenceType::VolumeScattering,
                SequenceType::LightFilter3D,
                sequenceID, light.getHash());
            mDistanceLightFilterSamples3D.resume(distanceLightFilter3DSid, samplesSoFar);
            if (mDoEquiAngular) {
                SequenceIDIntegrator equiAngularLightSid(
                    pv.nonMirrorDepth, sp.mPixel, 0,
                    SequenceType::VolumeEquiAngular,
                    SequenceType::VolumeScattering,
                    SequenceType::Light,
                    sequenceID, light.getHash());
                mEquiAngularLightSamples.resume(equiAngularLightSid, samplesSoFar);
                SequenceIDIntegrator equiAngularLightFilterSid(
                    pv.nonMirrorDepth, sp.mPixel, 0,
                    SequenceType::VolumeEquiAngular,
                    SequenceType::VolumeScattering,
                    SequenceType::LightFilter,
                    sequenceID, light.getHash());
                mEquiAngularLightFilterSamples.resume(equiAngularLightFilterSid, samplesSoFar);
                SequenceIDIntegrator equiAngularLightFilter3DSid(
                    pv.nonMirrorDepth, sp.mPixel, 0,
                    SequenceType::VolumeEquiAngular,
                    SequenceType::VolumeScattering,
                    SequenceType::LightFilter3D,
                    sequenceID, light.getHash());
                mEquiAngularLightFilterSamples3D.resume(equiAngularLightFilter3DSid, samplesSoFar);
            }
        } else {
            SequenceIDIntegrator distanceLightSid(
                pv.nonMirrorDepth, sp.mPixel, sp.mSubpixelIndex,
                SequenceType::VolumeDistance,
                SequenceType::VolumeScattering,
                SequenceType::Light,
                sequenceID, light.getHash());
            mDistanceLightSamples.restart(distanceLightSid, samplesCount);
            SequenceIDIntegrator distanceLightFilterSid(
                pv.nonMirrorDepth, sp.mPixel, sp.mSubpixelIndex,
                SequenceType::VolumeDistance,
                SequenceType::VolumeScattering,
                SequenceType::LightFilter,
                sequenceID, light.getHash());
            mDistanceLightFilterSamples.restart(distanceLightFilterSid, samplesCount);
            SequenceIDIntegrator distanceLightFilter3DSid(
                pv.nonMirrorDepth, sp.mPixel, sp.mSubpixelIndex,
                SequenceType::VolumeDistance,
                SequenceType::VolumeScattering,
                SequenceType::LightFilter3D,
                sequenceID, light.getHash());
            mDistanceLightFilterSamples3D.restart(distanceLightFilter3DSid, samplesCount);
            if (mDoEquiAngular) {
                SequenceIDIntegrator equiAngularLightSid(
                    pv.nonMirrorDepth, sp.mPixel, sp.mSubpixelIndex,
                    SequenceType::VolumeEquiAngular,
                    SequenceType::VolumeScattering,
                    SequenceType::Light,
                    sequenceID, light.getHash());
                mEquiAngularLightSamples.restart(equiAngularLightSid, samplesCount);
                SequenceIDIntegrator equiAngularLightFilterSid(
                    pv.nonMirrorDepth, sp.mPixel, sp.mSubpixelIndex,
                    SequenceType::VolumeEquiAngular,
                    SequenceType::VolumeScattering,
                    SequenceType::LightFilter,
                    sequenceID, light.getHash());
                mEquiAngularLightFilterSamples.restart(equiAngularLightFilterSid, samplesCount);
                SequenceIDIntegrator equiAngularLightFilter3DSid(
                    pv.nonMirrorDepth, sp.mPixel, sp.mSubpixelIndex,
                    SequenceType::VolumeEquiAngular,
                    SequenceType::VolumeScattering,
                    SequenceType::LightFilter3D,
                    sequenceID, light.getHash());
                mEquiAngularLightFilterSamples3D.restart(equiAngularLightFilter3DSid, samplesCount);
            }
        } 
    }

    void getEquiAngularSample(float& ut, scene_rdl2::math::Vec3f& ul, LightFilterRandomValues& ulFilter) const
    {
        MNRY_ASSERT(hasEquiAngularSamples());
        ut = mEquiAngularSamples.getSample(mNonMirrorDepth);
        mEquiAngularLightSamples.getSample(&ul[0], mNonMirrorDepth);
        mEquiAngularLightFilterSamples.getSample(&ulFilter.r2[0], mNonMirrorDepth);
        mEquiAngularLightFilterSamples3D.getSample(&ulFilter.r3[0], mNonMirrorDepth);
    }

    void getDistanceSample(float& ut, scene_rdl2::math::Vec3f& ul, LightFilterRandomValues& ulFilter) const
    {
        ut = mDistanceSamples.getSample(mNonMirrorDepth);
        mDistanceLightSamples.getSample(&ul[0], mNonMirrorDepth);
        mDistanceLightFilterSamples.getSample(&ulFilter.r2[0], mNonMirrorDepth);
        mDistanceLightFilterSamples3D.getSample(&ulFilter.r3[0], mNonMirrorDepth);
    }

    bool hasEquiAngularSamples() const
    {
        return mDoEquiAngular;
    }

private:
    VolumeScatterEventSampler mDistanceSamples;
    IntegratorSample3D mDistanceLightSamples;
    IntegratorSample2D mDistanceLightFilterSamples;
    IntegratorSample3D mDistanceLightFilterSamples3D;
    VolumeScatterEventSampler mEquiAngularSamples;
    IntegratorSample3D mEquiAngularLightSamples;
    IntegratorSample2D mEquiAngularLightFilterSamples;
    IntegratorSample3D mEquiAngularLightFilterSamples3D;
    int mNonMirrorDepth;
    bool mDoEquiAngular;
};

} // namespace pbr
} // namespace moonray

