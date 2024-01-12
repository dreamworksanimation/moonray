// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

///
/// @file ProceduralLeaf.cc
/// $Id$
///
#include "ProceduralLeaf.h"

#include <moonray/rendering/geom/Api.h>
#include <moonray/rendering/geom/Primitive.h>
#include <moonray/rendering/geom/PrimitiveGroup.h>
#include <moonray/rendering/geom/PrimitiveVisitor.h>
#include <moonray/rendering/geom/prim/PrimitivePrivateAccess.h>

#include <tbb/parallel_for_each.h>

namespace moonray {
namespace geom {

class ProceduralLeaf::Impl
{
public:
    Impl() : mPrimitiveGroup(createPrimitiveGroup()) {}

    void clear() { mPrimitiveGroup->clear(); }

    bool isReference() const {
        return (mSharedPrimitiveGroup != nullptr);
    }

    const std::shared_ptr<SharedPrimitive>& getReference() const {
        return mSharedPrimitiveGroup;
    }

    void transformToReference() {
        mSharedPrimitiveGroup = createSharedPrimitive(
            std::move(mPrimitiveGroup));

        mPrimitiveGroup = createPrimitiveGroup();
    }

    std::unique_ptr<PrimitiveGroup> mPrimitiveGroup;

    std::shared_ptr<SharedPrimitive> mSharedPrimitiveGroup;
};


ProceduralLeaf::ProceduralLeaf(const State &state) :
        Procedural(state), mImpl(new Impl())
{
}

ProceduralLeaf::~ProceduralLeaf()
{
    clear();
}

void
ProceduralLeaf::forEachPrimitive(PrimitiveVisitor& visitor,
        bool parallel)
{
    mImpl->mPrimitiveGroup->forEachPrimitive(visitor, parallel);
}

void
ProceduralLeaf::forEachStatic(PrimitiveVisitor& visitor,
        bool parallel)
{
    mImpl->mPrimitiveGroup->forEachStatic(visitor, parallel);
}

void
ProceduralLeaf::forEachDynamic(PrimitiveVisitor& visitor,
        bool parallel)
{
    mImpl->mPrimitiveGroup->forEachDynamic(visitor, parallel);
}

void
ProceduralLeaf::forEachDeformable(PrimitiveVisitor& visitor,
        bool parallel)
{
    mImpl->mPrimitiveGroup->forEachDeformable(visitor, parallel);
}

Procedural::size_type
ProceduralLeaf::getPrimitivesCount() const
{
    return mImpl->mPrimitiveGroup->getPrimitivesCount();
}

void
ProceduralLeaf::clear()
{
    mImpl->clear();
}

bool
ProceduralLeaf::isReference() const
{
    return mImpl->isReference();
}

const std::shared_ptr<SharedPrimitive>&
ProceduralLeaf::getReference() const
{
    return mImpl->getReference();
}

void
ProceduralLeaf::addPrimitive(std::unique_ptr<Primitive> p,
        const MotionBlurParams& motionBlurParams,
        const shading::XformSamples& parent2render)
{
    if (p) {
        const Mat43& proc2parent = getState().getProc2Parent();
        shading::XformSamples prim2render;
        for (size_t i = 0; i < parent2render.size(); ++i) {
            prim2render.push_back(proc2parent * parent2render[i]);
        }
        // transform primitive to rendering space and interpolate it with motion blur info
        internal::PrimitivePrivateAccess::transformPrimitive(p.get(), motionBlurParams, prim2render);
        mImpl->mPrimitiveGroup->addPrimitive(std::move(p));
    }
}

void
ProceduralLeaf::reservePrimitive(Procedural::size_type n)
{
    mImpl->mPrimitiveGroup->reserve(n);
}

void
ProceduralLeaf::transformToReference()
{
    mImpl->transformToReference();
}

} // namespace geom
} // namespace moonray


