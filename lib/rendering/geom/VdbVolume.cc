// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

///
/// @file VdbVolume.cc
///

#include "VdbVolume.h"

#include <moonray/rendering/geom/PrimitiveVisitor.h>
#include <moonray/rendering/geom/ProceduralContext.h>
#include <moonray/rendering/geom/State.h>

#include <moonray/rendering/geom/prim/VdbVolume.h>
#include <scene_rdl2/render/util/stdmemory.h>

namespace moonray {
namespace geom {

struct VdbVolume::Impl
{
    explicit Impl(internal::VdbVolume* vol) : mVdbVolume(vol) {}
    std::unique_ptr<internal::VdbVolume> mVdbVolume;
};

VdbVolume::VdbVolume(const VdbInitData& vdbInitData,
        const MotionBlurParams& motionBlurParams,
        LayerAssignmentId&& layerAssignmentId,
        shading::PrimitiveAttributeTable&& primitiveAttributeTalble):
    mImpl(fauxstd::make_unique<Impl>(new internal::VdbVolume(
        vdbInitData.mVdbFilePath,
        vdbInitData.mDensityGridName,
        vdbInitData.mEmissionGridName,
        vdbInitData.mVelocityGridName,
        motionBlurParams,
        std::move(layerAssignmentId),
        std::move(primitiveAttributeTalble)))) {}

VdbVolume::VdbVolume()
{
}

VdbVolume::~VdbVolume() = default;

void
VdbVolume::accept(PrimitiveVisitor& v)
{
    v.visitVdbVolume(*this);
}

Primitive::size_type
VdbVolume::getMemory() const
{
    return sizeof(VdbVolume) + mImpl->mVdbVolume->getMemory();
}

Primitive::size_type
VdbVolume::getMotionSamplesCount() const
{
    return mImpl->mVdbVolume->getMotionSamplesCount();
}

void
VdbVolume::setName(const std::string& name)
{
    mImpl->mVdbVolume->setName(name);
}

const std::string&
VdbVolume::getName() const
{
    return mImpl->mVdbVolume->getName();
}

void
VdbVolume::setInterpolation(Interpolation interpolation)
{
    mImpl->mVdbVolume->setInterpolation(interpolation);
}

void
VdbVolume::setVelocityScale(float velocityScale)
{
    mImpl->mVdbVolume->setVelocityScale(velocityScale);
}

void
VdbVolume::setVelocitySampleRate(float velocitySampleRate)
{
    mImpl->mVdbVolume->setVelocitySampleRate(velocitySampleRate);
}

void
VdbVolume::setEmissionSampleRate(float emissionSampleRate)
{
    mImpl->mVdbVolume->setEmissionSampleRate(emissionSampleRate);
}

void
VdbVolume::transformPrimitive(
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
    mImpl->mVdbVolume->setTransform(p2r, shutterOpenDelta, shutterCloseDelta);
}

internal::Primitive*
VdbVolume::getPrimitiveImpl()
{
    return mImpl->mVdbVolume.get();
}

} // namespace geom
} // namespace moonray

