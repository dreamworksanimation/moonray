// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

///
/// @file Box.cc
/// $Id$
///

#include "Box.h"

#include <moonray/rendering/geom/PrimitiveVisitor.h>

#include <moonray/rendering/geom/prim/Box.h>
#include <scene_rdl2/render/util/stdmemory.h>

namespace moonray {
namespace geom {

struct Box::Impl
{
    explicit Impl(internal::Box* Box) : mBox(Box) {}
    std::unique_ptr<internal::Box> mBox;
};

Box::Box(float length, float width, float height, LayerAssignmentId&& layerAssignmentId,
        shading::PrimitiveAttributeTable&& primitiveAttributeTable):
        mImpl(fauxstd::make_unique<Impl>(new internal::Box(length, width, height,
        std::move(layerAssignmentId), std::move(primitiveAttributeTable))))
{
}

Box::~Box() = default;

void
Box::accept(PrimitiveVisitor& v)
{
    v.visitBox(*this);
}

Primitive::size_type
Box::getMemory() const
{
    return sizeof(Box) + mImpl->mBox->getMemory();
}

Primitive::size_type
Box::getMotionSamplesCount() const
{
    return mImpl->mBox->getMotionSamplesCount();
}

void
Box::setName(const std::string& name)
{
    mImpl->mBox->setName(name);
}

const std::string&
Box::getName() const
{
    return mImpl->mBox->getName();
}

void
Box::setIsSingleSided(bool isSingleSided)
{
    mImpl->mBox->setIsSingleSided(isSingleSided);
}

bool
Box::getIsSingleSided() const
{
    return mImpl->mBox->getIsSingleSided();
}

void
Box::setIsNormalReversed(bool isNormalReversed)
{
    mImpl->mBox->setIsNormalReversed(isNormalReversed);
}

bool
Box::getIsNormalReversed() const
{
    return mImpl->mBox->getIsNormalReversed();
}

void
Box::transformPrimitive(
        const MotionBlurParams& motionBlurParams,
        const shading::XformSamples& prim2render)
{
    mImpl->mBox->setTransform(prim2render[0]);
}

internal::Primitive*
Box::getPrimitiveImpl()
{
    return mImpl->mBox.get();
}

} // namespace geom
} // namespace moonray

