// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

///
/// @file Sphere.cc
/// $Id$
///

#include "Sphere.h"

#include <moonray/rendering/geom/PrimitiveVisitor.h>

#include <moonray/rendering/geom/prim/Sphere.h>
#include <scene_rdl2/render/util/stdmemory.h>

namespace moonray {
namespace geom {

struct Sphere::Impl
{
    explicit Impl(internal::Sphere* sphere) : mSphere(sphere) {}
    std::unique_ptr<internal::Sphere> mSphere;
};

Sphere::Sphere(float radius, LayerAssignmentId&& layerAssignmentId,
        shading::PrimitiveAttributeTable&& primitiveAttributeTable):
        mImpl(fauxstd::make_unique<Impl>(new internal::Sphere(radius,
        std::move(layerAssignmentId), std::move(primitiveAttributeTable))))
{
}

Sphere::~Sphere() = default;

void
Sphere::accept(PrimitiveVisitor& v)
{
    v.visitSphere(*this);
}

Primitive::size_type
Sphere::getMemory() const
{
    return sizeof(Sphere) + mImpl->mSphere->getMemory();
}

Primitive::size_type
Sphere::getMotionSamplesCount() const
{
    return mImpl->mSphere->getMotionSamplesCount();
}

void
Sphere::setName(const std::string& name)
{
    mImpl->mSphere->setName(name);
}

const std::string&
Sphere::getName() const
{
    return mImpl->mSphere->getName();
}

void
Sphere::setClippingRange(float zMin, float zMax, float sweepAngle)
{
    mImpl->mSphere->setClippingRange(zMin, zMax, sweepAngle);
}

void
Sphere::setIsSingleSided(bool isSingleSided)
{
    mImpl->mSphere->setIsSingleSided(isSingleSided);
}

bool
Sphere::getIsSingleSided() const
{
    return mImpl->mSphere->getIsSingleSided();
}

void
Sphere::setIsNormalReversed(bool isNormalReversed)
{
    mImpl->mSphere->setIsNormalReversed(isNormalReversed);
}

bool
Sphere::getIsNormalReversed() const
{
    return mImpl->mSphere->getIsNormalReversed();
}

void
Sphere::transformPrimitive(
        const MotionBlurParams& motionBlurParams,
        const shading::XformSamples& prim2render)
{
    mImpl->mSphere->setTransform(prim2render[0]);
}

internal::Primitive*
Sphere::getPrimitiveImpl()
{
    return mImpl->mSphere.get();
}

} // namespace geom
} // namespace moonray

