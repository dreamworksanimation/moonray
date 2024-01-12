// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

///
/// @file TransformedPrimitive.cc
/// $Id$
///

#include "TransformedPrimitive.h"

#include <moonray/rendering/geom/Api.h>
#include <moonray/rendering/geom/PrimitiveVisitor.h>

#include <moonray/rendering/geom/prim/PrimitivePrivateAccess.h>

namespace moonray {
namespace geom {

struct TransformedPrimitive::Impl
{
    explicit Impl(const shading::XformSamples& xform,
            std::unique_ptr<Primitive> primitive) :
        mXform(xform), mPrimitive(std::move(primitive)),
        mHasBeenApplied(false) {
        if (mXform.empty()) {
            mXform.push_back(Mat43(scene_rdl2::math::one));
        }
    }

    Primitive::size_type getMemory() const {
        return sizeof(TransformedPrimitive::Impl) +
            scene_rdl2::util::getVectorElementsMemory(mXform) + mPrimitive->getMemory();
    }

    shading::XformSamples mXform;
    std::unique_ptr<Primitive> mPrimitive;
    // TransformedPrimitive should only apply mXform to mPrimitive once during
    // render prepare stage, flip this flag once transformPrimitive has been called
    bool mHasBeenApplied;
};

TransformedPrimitive::~TransformedPrimitive() = default;

TransformedPrimitive::TransformedPrimitive(const shading::XformSamples& xform,
        std::unique_ptr<Primitive> primitive) :
    mImpl(new Impl(xform, std::move(primitive))) {}

void
TransformedPrimitive::accept(PrimitiveVisitor& v)
{
    v.visitTransformedPrimitive(*this);
}

shading::XformSamples
TransformedPrimitive::getXformSamples() const
{
    return mImpl->mXform;
}

Primitive::size_type
TransformedPrimitive::getMemory() const
{
    return sizeof(TransformedPrimitive) + mImpl->getMemory();
}

Primitive::size_type
TransformedPrimitive::getMotionSamplesCount() const
{
    return mImpl->mXform.size();
}

const std::unique_ptr<Primitive>&
TransformedPrimitive::getPrimitive() const
{
    return mImpl->mPrimitive;
}

void
TransformedPrimitive::transformPrimitive(
        const MotionBlurParams& motionBlurParams,
        const shading::XformSamples& prim2render)
{
    if (!mImpl->mHasBeenApplied) {
        shading::XformSamples local2render = concatenate(mImpl->mXform, prim2render);
        internal::PrimitivePrivateAccess::transformPrimitive(
            mImpl->mPrimitive.get(), motionBlurParams,
            local2render);
        mImpl->mHasBeenApplied = true;
    }
}

internal::Primitive*
TransformedPrimitive::getPrimitiveImpl()
{
    MNRY_ASSERT_REQUIRE(false, "TransformedPrimitive is an opaque handle wrapper"
        " that has no internal implementation");
    return nullptr;
}

} // namespace geom
} // namespace moonray

