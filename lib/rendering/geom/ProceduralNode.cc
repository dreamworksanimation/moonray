// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

///
/// @file ProceduralNode.cc
/// $Id$
///

#include "ProceduralNode.h"

#include <moonray/rendering/geom/Api.h>
#include <moonray/rendering/geom/PrimitiveVisitor.h>

#include <moonray/rendering/geom/prim/PrimitivePrivateAccess.h>
#include <tbb/parallel_for.h>
#include <numeric>

namespace moonray {
namespace geom {

class ProceduralNode::Impl
{
public:
    bool isReference() const {
        return (mSharedPrimitiveGroup != nullptr);
    }

    const std::shared_ptr<SharedPrimitive>& getReference() const {
        return mSharedPrimitiveGroup;
    }

    void transformToReference() {
        MNRY_ASSERT_REQUIRE(false, "not implemented yet");
    }

    ProceduralArray mChildren;
    std::shared_ptr<SharedPrimitive> mSharedPrimitiveGroup;
};

ProceduralNode::ProceduralNode(const State &state) :
        Procedural(state), mImpl(new Impl())
{
}

ProceduralNode::~ProceduralNode()
{
    clear();
}

const ProceduralArray&
ProceduralNode::getChildren() const
{
    return mImpl->mChildren;
}

void
ProceduralNode::forEachPrimitive(PrimitiveVisitor& visitor,
        bool parallel)
{
    if (parallel) {
        tbb::blocked_range<size_t> range(0, mImpl->mChildren.size());
        tbb::parallel_for(range, [&](const tbb::blocked_range<size_t> &r) {
            for (size_t i = r.begin(); i < r.end(); ++i) {
                mImpl->mChildren[i]->forEachPrimitive(visitor, parallel);
            }
        });
    } else {
        for (auto& child : mImpl->mChildren) {
            child->forEachPrimitive(visitor, parallel);
        }
    }
}

void
ProceduralNode::forEachStatic(PrimitiveVisitor& visitor,
        bool parallel)
{
    if (parallel) {
        tbb::blocked_range<size_t> range(0, mImpl->mChildren.size());
        tbb::parallel_for(range, [&](const tbb::blocked_range<size_t> &r) {
            for (size_t i = r.begin(); i < r.end(); ++i) {
                mImpl->mChildren[i]->forEachStatic(visitor, parallel);
            }
        });
    } else {
        for (auto& child : mImpl->mChildren) {
            child->forEachStatic(visitor, parallel);
        }
    }
}

void
ProceduralNode::forEachDynamic(PrimitiveVisitor& visitor,
        bool parallel)
{
    if (parallel) {
        tbb::blocked_range<size_t> range(0, mImpl->mChildren.size());
        tbb::parallel_for(range, [&](const tbb::blocked_range<size_t> &r) {
            for (size_t i = r.begin(); i < r.end(); ++i) {
                mImpl->mChildren[i]->forEachDynamic(visitor, parallel);
            }
        });
    } else {
        for (auto& child : mImpl->mChildren) {
            child->forEachDynamic(visitor, parallel);
        }
    }
}

void
ProceduralNode::forEachDeformable(PrimitiveVisitor& visitor,
        bool parallel)
{
    if (parallel) {
        tbb::blocked_range<size_t> range(0, mImpl->mChildren.size());
        tbb::parallel_for(range, [&](const tbb::blocked_range<size_t> &r) {
            for (size_t i = r.begin(); i < r.end(); ++i) {
                mImpl->mChildren[i]->forEachDeformable(visitor, parallel);
            }
        });
    } else {
        for (auto& child : mImpl->mChildren) {
            child->forEachDeformable(visitor, parallel);
        }
    }
}

Procedural::size_type
ProceduralNode::getPrimitivesCount() const
{
    return std::accumulate(mImpl->mChildren.begin(), mImpl->mChildren.end(), 0,
        [](const Procedural::size_type n, Procedural* p) {
            return n + p->getPrimitivesCount();
        });
}

void
ProceduralNode::clear()
{
    for (auto p : mImpl->mChildren) {
        p->clear();
    }
    mImpl->mChildren.clear();
}

bool
ProceduralNode::isReference() const
{
    return mImpl->isReference();
}

const std::shared_ptr<SharedPrimitive>&
ProceduralNode::getReference() const
{
    return mImpl->getReference();
}

void
ProceduralNode::transformToReference()
{
    mImpl->transformToReference();
}

} // namespace geom
} // namespace moonray


