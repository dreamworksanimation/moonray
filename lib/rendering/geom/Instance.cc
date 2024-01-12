// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

///
/// @file Instance.cc
/// $Id$
///

#include "Instance.h"

#include <moonray/rendering/geom/Api.h>
#include <moonray/rendering/geom/PrimitiveVisitor.h>

#include <moonray/rendering/geom/prim/Instance.h>

namespace moonray {
namespace geom {

class Instance::Impl : public internal::Instance {
public:
    explicit Impl(const shading::XformSamples& xform,
            std::shared_ptr<SharedPrimitive> reference,
            shading::PrimitiveAttributeTable&& primitiveAttributeTable) :
        internal::Instance(xform, reference, std::move(primitiveAttributeTable))
    {}

};

Instance::Instance(const shading::XformSamples& xform,
        std::shared_ptr<SharedPrimitive> reference,
        shading::PrimitiveAttributeTable&& primitiveAttributeTable) :
    mImpl(new Instance::Impl(xform, reference,
    std::move(primitiveAttributeTable)))
{
}


Instance::~Instance() = default;

void
Instance::accept(PrimitiveVisitor& v)
{
    v.visitInstance(*this);
}

Primitive::size_type
Instance::getMemory() const
{
    return sizeof(Instance) + mImpl->getMemory();
}

Primitive::size_type
Instance::getMotionSamplesCount() const
{
    return getReference()->getPrimitive()->getMotionSamplesCount();
}

const std::shared_ptr<SharedPrimitive>&
Instance::getReference() const
{
    return mImpl->getReference();
}

void
Instance::transformPrimitive(
        const MotionBlurParams& motionBlurParams,
        const shading::XformSamples& prim2render)
{
    float shutterOpenDelta, shutterCloseDelta;
    motionBlurParams.getMotionBlurDelta(shutterOpenDelta, shutterCloseDelta);
    mImpl->appendXform(prim2render, shutterOpenDelta, shutterCloseDelta);

    if (mImpl->getAttributes() != nullptr) {
        mImpl->getAttributes()->transformAttributes(prim2render,
                                                    shutterOpenDelta,
                                                    shutterCloseDelta,
                                                    {{shading::StandardAttributes::sNormal, shading::Vec3Type::NORMAL},
                                                    {shading::StandardAttributes::sdPds, shading::Vec3Type::VECTOR},
                                                    {shading::StandardAttributes::sdPdt, shading::Vec3Type::VECTOR}});
    }
}

internal::Primitive*
Instance::getPrimitiveImpl()
{
    return mImpl.get();
}

} // namespace geom
} // namespace moonray

