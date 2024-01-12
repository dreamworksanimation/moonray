// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

///
/// @file Primitive.cc
/// $Id$
///

#include <moonray/rendering/geom/Primitive.h>
#include <moonray/rendering/geom/PrimitiveVisitor.h>
#include <moonray/rendering/geom/State.h>

namespace moonray {
namespace geom {

struct Primitive::Impl
{
    Impl(): mUpdated(false), mModifiability(Primitive::Modifiability::STATIC) {}

    bool mUpdated;
    Modifiability mModifiability;
};

Primitive::Primitive() : mImpl(new Primitive::Impl())
{
}

Primitive::~Primitive() = default;

void
Primitive::accept(PrimitiveVisitor& v)
{
    v.visitPrimitive(*this);
}

void
Primitive::setUpdated(bool updated)
{
    mImpl->mUpdated = updated;
}

bool
Primitive::getUpdated() const
{
    return mImpl->mUpdated;
}

void
Primitive::setModifiability(Modifiability m)
{
    mImpl->mModifiability = m;
}

Primitive::Modifiability
Primitive::getModifiability() const
{
    return mImpl->mModifiability;
}

} // namespace geom
} // namespace moonray

