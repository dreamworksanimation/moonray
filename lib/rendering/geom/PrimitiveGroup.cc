// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

///
/// @file PrimitiveGroup.cc
/// $Id$
///
#include "PrimitiveGroup.h"

#include <moonray/rendering/geom/PrimitiveVisitor.h>
#include <moonray/rendering/geom/ProceduralContext.h>
#include <moonray/rendering/geom/prim/PrimitivePrivateAccess.h>

#include <tbb/parallel_for.h>

namespace moonray {
namespace geom {

struct PrimitiveGroup::Impl
{
    ~Impl() {
        clear();
    }

    void clear() {
        // TODO the "delete" part should be removed after we update tbb
        // and change tbb::concurrent_vector<Primitive*> to
        // tbb::concurrent_vector<std::unique_ptr<Primitive>>
        for (auto p : mChildren) {
            delete p;
            p = nullptr;
        }
        mChildren.clear();
    }

    Primitive::size_type getMemory() const {
        Primitive::size_type memory = sizeof(PrimitiveGroup::Impl) +
            mChildren.capacity() * sizeof(Primitive*);
        for (const auto& child : mChildren) {
            memory += child->getMemory();
        }
        return memory;
    }

    tbb::concurrent_vector<Primitive*> mChildren;
};

PrimitiveGroup::~PrimitiveGroup() = default;

PrimitiveGroup::PrimitiveGroup() : mImpl(new Impl())
{
}

void
PrimitiveGroup::accept(PrimitiveVisitor& v)
{
    v.visitPrimitiveGroup(*this);
}

Primitive::size_type
PrimitiveGroup::getMemory() const
{
    return sizeof(PrimitiveGroup) + mImpl->getMemory();
}

Primitive::size_type
PrimitiveGroup::getMotionSamplesCount() const
{
    return 1;
}

Primitive::size_type
PrimitiveGroup::getPrimitivesCount() const
{
    return mImpl->mChildren.size();
}

void
PrimitiveGroup::addPrimitive(std::unique_ptr<Primitive> primitive)
{
    if (primitive) {
        mImpl->mChildren.push_back(primitive.release());
    }
}

void
PrimitiveGroup::reserve(Primitive::size_type n)
{
    mImpl->mChildren.reserve(n);
}

void
PrimitiveGroup::clear()
{
    mImpl->clear();
}

void
PrimitiveGroup::forEachPrimitive(PrimitiveVisitor& visitor, bool parallel)
{
    if (parallel) {
        tbb::blocked_range<size_t> range(0, mImpl->mChildren.size());
        tbb::parallel_for(range, [&](const tbb::blocked_range<size_t> &r)
        {
            for (size_t i = r.begin(); i < r.end(); ++i) {
                mImpl->mChildren[i]->accept(visitor);
            }
        });
    } else {
        for (size_t i = 0; i < mImpl->mChildren.size(); ++i) {
            mImpl->mChildren[i]->accept(visitor);
        }
    }
}

void
PrimitiveGroup::forEachStatic(PrimitiveVisitor& visitor, bool parallel)
{
    if (parallel) {
        tbb::blocked_range<size_t> range(0, mImpl->mChildren.size());
        tbb::parallel_for(range, [&](const tbb::blocked_range<size_t> &r)
        {
            for (size_t i = r.begin(); i < r.end(); ++i) {
                if (mImpl->mChildren[i]->getModifiability() ==
                    Primitive::Modifiability::STATIC) {
                    mImpl->mChildren[i]->accept(visitor);
                }
            }
        });
    } else {
        for (size_t i = 0; i < mImpl->mChildren.size(); ++i) {
            if (mImpl->mChildren[i]->getModifiability() ==
                Primitive::Modifiability::STATIC) {
                mImpl->mChildren[i]->accept(visitor);
            }
        }
    }
}

void
PrimitiveGroup::forEachDynamic(PrimitiveVisitor& visitor, bool parallel)
{
    if (parallel) {
        tbb::blocked_range<size_t> range(0, mImpl->mChildren.size());
        tbb::parallel_for(range, [&](const tbb::blocked_range<size_t> &r)
        {
            for (size_t i = r.begin(); i < r.end(); ++i) {
                if (mImpl->mChildren[i]->getModifiability() ==
                    Primitive::Modifiability::DYNAMIC) {
                    mImpl->mChildren[i]->accept(visitor);
                }
            }
        });
    } else {
        for (size_t i = 0; i < mImpl->mChildren.size(); ++i) {
            if (mImpl->mChildren[i]->getModifiability() ==
                Primitive::Modifiability::DYNAMIC) {
                mImpl->mChildren[i]->accept(visitor);
            }
        }
    }
}

void
PrimitiveGroup::forEachDeformable(PrimitiveVisitor& visitor, bool parallel)
{
    if (parallel) {
        tbb::blocked_range<size_t> range(0, mImpl->mChildren.size());
        tbb::parallel_for(range, [&](const tbb::blocked_range<size_t> &r)
        {
            for (size_t i = r.begin(); i < r.end(); ++i) {
                if (mImpl->mChildren[i]->getModifiability() ==
                    Primitive::Modifiability::DEFORMABLE) {
                    mImpl->mChildren[i]->accept(visitor);
                }
            }
        });
    } else {
        for (size_t i = 0; i < mImpl->mChildren.size(); ++i) {
            if (mImpl->mChildren[i]->getModifiability() ==
                Primitive::Modifiability::DEFORMABLE) {
                mImpl->mChildren[i]->accept(visitor);
            }
        }
    }
}

void
PrimitiveGroup::transformPrimitive(
        const MotionBlurParams& motionBlurParams,
        const shading::XformSamples& prim2render)
{
    for (auto& child : mImpl->mChildren) {
        internal::PrimitivePrivateAccess::transformPrimitive(child,
            motionBlurParams, prim2render);
    }
}

internal::Primitive*
PrimitiveGroup::getPrimitiveImpl()
{
    MNRY_ASSERT_REQUIRE(false, "PrimitiveGroup is an opaque handle wrapper "
        "that has no internal implementation");
    return nullptr;
}

} // namespace geom
} // namespace moonray

