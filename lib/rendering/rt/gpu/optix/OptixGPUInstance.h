// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "OptixGPUPrimitive.h"

namespace moonray {
namespace rt {

class OptixGPUPrimitiveGroup;

// An instance just references a shared OptixGPUPrimitiveGroup and specifies an xform
// for the instance.

class OptixGPUInstance
{
public:
    OptixGPUInstance() : mIsBuilt{false}, mGroup{nullptr}, mHasMotionBlur{false} {}
    virtual ~OptixGPUInstance() {}

    bool build(CUstream cudaStream,
               OptixDeviceContext context,
               std::string* errorMsg);

    bool mIsBuilt;

    OptixGPUPrimitiveGroup* mGroup;

    bool mHasMotionBlur;

    // Our CPU side code uses slerp() to interpolate between two instance xforms
    // when there is motion blur.
    // Optix can only lerp() so we need to generate 64 transforms that it can lerp() between
    // instead and approximate the slerp().
    // Note that if there is no motion blur, we still use mXforms[0] as the static xform.
    static const int sNumMotionKeys = 64;
    OptixGPUXform mXforms[sNumMotionKeys];

    // If there's motion blur, the referenced OptixGPUPrimitiveGroup is the child of an
    // OptixMatrixMotionTransform, and then we use that traversable
    // instead of the OptixGPUPrimitiveGroup's mTopLevelIAS.
    OptixTraversableHandle mMMTTraversable;
    OptixGPUBuffer<char> mMMTTraversableBuf;      // the actual memory buffer on the GPU
};

} // namespace rt
} // namespace moonray

