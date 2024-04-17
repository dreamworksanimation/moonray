// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <Metal/Metal.h>
#include "MetalGPUPrimitive.h"

namespace moonray {
namespace rt {

class MetalGPUPrimitiveGroup;

// An instance just references a shared GPUPrimitiveGroup and specifies an xform
// for the instance.

class MetalGPUInstance
{
public:
    MetalGPUInstance()
    : mIsBuilt{false}
    , mGroup{nullptr}
    , mHasMotionBlur{false} {}
    virtual ~MetalGPUInstance() {}

    bool build(id<MTLDevice> context,
               id<MTLCommandQueue> queue,
               std::vector<id<MTLAccelerationStructure>>* bottomLevelAS,
               std::atomic<int> &structuresBuilding,
               std::string* errorMsg);

    bool mIsBuilt;

    MetalGPUPrimitiveGroup* mGroup;

    bool mHasMotionBlur;

    // Our CPU side code uses slerp() to interpolate between two instance xforms
    // when there is motion blur.
    // Optix can only lerp() so we need to generate 64 transforms that it can lerp() between
    // instead and approximate the slerp().
    // Note that if there is no motion blur, we still use mXforms[0] as the static xform.
    static const int sNumMotionKeys = 64;
    MetalGPUXform mXforms[sNumMotionKeys];

    // If there's motion blur, the referenced GPUPrimitiveGroup is the child of an
    // OptixMatrixMotionTransform, and then we use that traversable
    // instead of the GPUPrimitiveGroup's mTopLevelIAS.
    id<MTLAccelerationStructure> mMMTTraversable;
};

} // namespace rt
} // namespace moonray

