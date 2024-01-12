// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

///
/// @file Points.cc
/// $Id$
///

#include "Points.h"

#include <moonray/rendering/geom/PrimitiveVisitor.h>
#include <moonray/rendering/geom/ProceduralContext.h>
#include <moonray/rendering/geom/State.h>

#include <moonray/rendering/geom/prim/Points.h>
#include <scene_rdl2/render/util/stdmemory.h>

namespace moonray {
namespace geom {

using namespace moonray::shading;

struct Points::Impl
{
    explicit Impl(internal::Points* points) : mPoints(points) {}
    std::unique_ptr<internal::Points> mPoints;
};

Points::Points(VertexBuffer&& position, RadiusBuffer&& radius,
        LayerAssignmentId&& layerAssignmentId,
        PrimitiveAttributeTable&& primitiveAttributeTable):
        mImpl(fauxstd::make_unique<Impl>(new internal::Points(
        std::move(position), std::move(radius),
        std::move(layerAssignmentId), std::move(primitiveAttributeTable))))
{
}

Points::~Points() = default;

void
Points::accept(PrimitiveVisitor& v)
{
    v.visitPoints(*this);
}

Primitive::size_type
Points::getMemory() const
{
    return sizeof(Points) + mImpl->mPoints->getMemory();
}

Primitive::size_type
Points::getMotionSamplesCount() const
{
    return mImpl->mPoints->getMotionSamplesCount();
}

void
Points::setName(const std::string& name)
{
    mImpl->mPoints->setName(name);
}

const std::string&
Points::getName() const
{
    return mImpl->mPoints->getName();
}

void
Points::setCurvedMotionBlurSampleCount(int count)
{
    mImpl->mPoints->setCurvedMotionBlurSampleCount(count);
}

void
Points::transformPrimitive(
        const MotionBlurParams& motionBlurParams,
        const XformSamples& prim2render)
{
    const PrimitiveAttributeTable* primAttrTab = mImpl->mPoints->getPrimitiveAttributeTable();
    transformVertexBuffer(mImpl->mPoints->getVertexBuffer(), prim2render, motionBlurParams,
                          mImpl->mPoints->getMotionBlurType(), mImpl->mPoints->getCurvedMotionBlurSampleCount(),
                          primAttrTab);

    float shutterOpenDelta, shutterCloseDelta;
    motionBlurParams.getMotionBlurDelta(shutterOpenDelta, shutterCloseDelta);
    mImpl->mPoints->getAttributes()->transformAttributes(prim2render,
                                                         shutterOpenDelta,
                                                         shutterCloseDelta,
                                                         {{StandardAttributes::sNormal, Vec3Type::NORMAL},
                                                         {StandardAttributes::sdPds, Vec3Type::VECTOR},
                                                         {StandardAttributes::sdPdt, Vec3Type::VECTOR}});
}

internal::Primitive*
Points::getPrimitiveImpl()
{
    return mImpl->mPoints.get();
}


} // namespace geom
} // namespace moonray

