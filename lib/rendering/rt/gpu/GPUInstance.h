// Copyright 2023 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "GPUPrimitive.h"

namespace moonray {
namespace rt {

class GPUPrimitiveGroup;

// An instance just references a shared GPUPrimitiveGroup and specifies an xform
// for the instance.

class GPUInstance
{
public:
    GPUInstance() : mIsBuilt{false}, mGroup{nullptr}, mHasMotionBlur{false} {}
    virtual ~GPUInstance() {}

    bool build(CUstream cudaStream,
               OptixDeviceContext context,
               std::string* errorMsg);

    bool mIsBuilt;

    GPUPrimitiveGroup* mGroup;

    bool mHasMotionBlur;

    // Our CPU side code uses slerp() to interpolate between two instance xforms
    // when there is motion blur.
    // Optix can only lerp() so we need to generate 64 transforms that it can lerp() between
    // instead and approximate the slerp().
    // Note that if there is no motion blur, we still use mXforms[0] as the static xform.
    static const int sNumMotionKeys = 64;
    GPUXform mXforms[sNumMotionKeys];

    // If there's motion blur, the referenced GPUPrimitiveGroup is the child of an
    // OptixMatrixMotionTransform, and then we use that traversable
    // instead of the GPUPrimitiveGroup's mTopLevelIAS.
    OptixTraversableHandle mMMTTraversable;
    GPUBuffer<char> mMMTTraversableBuf;      // the actual memory buffer on the GPU
};

} // namespace rt
} // namespace moonray

