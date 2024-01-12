// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

///
/// @file AmorphousVolume.cc
///

#include "AmorphousVolume.h"

#include <moonray/rendering/geom/PrimitiveVisitor.h>
#include <moonray/rendering/geom/ProceduralContext.h>
#include <moonray/rendering/geom/State.h>

#include <moonray/rendering/geom/prim/AmorphousVolume.h>
#include <scene_rdl2/render/util/stdmemory.h>

namespace moonray {
namespace geom {

struct AmorphousVolume::Impl
{
    explicit Impl(internal::AmorphousVolume* vol) : mAmorphousVolume(vol) {}
    std::unique_ptr<internal::AmorphousVolume> mAmorphousVolume;
};

AmorphousVolume::AmorphousVolume(const VdbInitData& vdbInitData,
        const MotionBlurParams& motionBlurParams,
        LayerAssignmentId&& layerAssignmentId,
        shading::PrimitiveAttributeTable&& primitiveAttributeTalble):
    mImpl(fauxstd::make_unique<Impl>(new internal::AmorphousVolume(
        vdbInitData.mVdbFilePath,
        vdbInitData.mDensityGridName,
        vdbInitData.mVelocityGridName,
        motionBlurParams,
        std::move(layerAssignmentId),
        std::move(primitiveAttributeTalble)))) {}

AmorphousVolume::~AmorphousVolume() = default;

Primitive::size_type
AmorphousVolume::getMemory() const
{
    return sizeof(AmorphousVolume) + mImpl->mAmorphousVolume->getMemory();
}

Primitive::size_type
AmorphousVolume::getMotionSamplesCount() const
{
    return mImpl->mAmorphousVolume->getMotionSamplesCount();
}

void
AmorphousVolume::setName(const std::string& name)
{
    mImpl->mAmorphousVolume->setName(name);
}

const std::string&
AmorphousVolume::getName() const
{
    return mImpl->mAmorphousVolume->getName();
}

void
AmorphousVolume::setInterpolation(Interpolation interpolation)
{
    mImpl->mAmorphousVolume->setInterpolation(interpolation);
}

void
AmorphousVolume::setVelocityScale(float velocityScale)
{
    mImpl->mAmorphousVolume->setVelocityScale(velocityScale);
}

void
AmorphousVolume::setVelocitySampleRate(float velocitySampleRate)
{
    mImpl->mAmorphousVolume->setVelocitySampleRate(velocitySampleRate);
}

void
AmorphousVolume::setEmissionSampleRate(float emissionSampleRate)
{
    mImpl->mAmorphousVolume->setEmissionSampleRate(emissionSampleRate);
}

void
AmorphousVolume::transformPrimitive(
        const MotionBlurParams& motionBlurParams,
        const shading::XformSamples& primToRender)
{
    shading::XformSamples p2r = primToRender;
    if (p2r.size() == 1) {
        p2r.resize(2, p2r[0]);
    }
    float shutterOpenDelta = 0.0f, shutterCloseDelta = 0.0f;
    if (motionBlurParams.isMotionBlurOn()) {
        motionBlurParams.getMotionBlurDelta(shutterOpenDelta, shutterCloseDelta);
    }
    mImpl->mAmorphousVolume->setTransform(p2r, shutterOpenDelta, shutterCloseDelta);
}

internal::Primitive*
AmorphousVolume::getPrimitiveImpl()
{
    return mImpl->mAmorphousVolume.get();
}

} // namespace geom
} // namespace moonray

