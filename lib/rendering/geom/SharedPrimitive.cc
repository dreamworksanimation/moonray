// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

///
/// @file SharedPrimitive.cc
/// $Id$
///

#include "SharedPrimitive.h"

#include <embree4/rtcore.h>

namespace moonray {
namespace geom {

struct SharedPrimitive::Impl {
    explicit Impl(std::unique_ptr<Primitive>&& primitive) :
        mPrimitive(std::move(primitive)), mBVHScene(nullptr),
        mHasSurfaceAssignment(true), mHasVolumeAssignment(false) {}

    ~Impl() {
        mPrimitive.reset();
       resetBVHScene();
    }

    void resetBVHScene(RTCScene bvhScene = nullptr) {
        if (mBVHScene) {
            rtcReleaseScene(mBVHScene);
        }
        mBVHScene = bvhScene;
    }

    std::unique_ptr<Primitive> mPrimitive;
    RTCScene mBVHScene;
    bool mHasSurfaceAssignment; // assumed to be yes
    bool mHasVolumeAssignment; // assumed to be no
};

SharedPrimitive::SharedPrimitive(std::unique_ptr<Primitive>&& primitive) :
    mImpl(new SharedPrimitive::Impl(std::move(primitive)))
{
}

SharedPrimitive::~SharedPrimitive() = default;

const std::unique_ptr<Primitive>&
SharedPrimitive::getPrimitive() const
{
    return mImpl->mPrimitive;
}

void
SharedPrimitive::setHasSurfaceAssignment(bool hasSurfaceAssignment)
{
    mImpl->mHasSurfaceAssignment = hasSurfaceAssignment;
}

bool
SharedPrimitive::getHasSurfaceAssignment() const
{
    return mImpl->mHasSurfaceAssignment;
}

void
SharedPrimitive::setHasVolumeAssignment(bool hasVolumeAssignment)
{
    mImpl->mHasVolumeAssignment = hasVolumeAssignment;
}

bool
SharedPrimitive::getHasVolumeAssignment() const
{
    return mImpl->mHasVolumeAssignment;
}

void
SharedPrimitive::setBVHScene(void* bvhScene) {
    mImpl->resetBVHScene(static_cast<RTCScene>(bvhScene));
}

void*
SharedPrimitive::getBVHScene() {
    return static_cast<void*>(mImpl->mBVHScene);
}

} // namespace geom
} // namespace moonray

