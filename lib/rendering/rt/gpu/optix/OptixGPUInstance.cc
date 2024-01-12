// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

#include "OptixGPUInstance.h"
#include "OptixGPUPrimitiveGroup.h"
#include "OptixGPUUtils.h"

namespace moonray {
namespace rt {

bool
OptixGPUInstance::build(CUstream cudaStream,
                   OptixDeviceContext context,
                   std::string* errorMsg)
{
    // If we have already visited, do nothing.
    if (mIsBuilt) {
        return true;
    }
    mIsBuilt = true;

    // build() the referenced group.  If it has already been built,
    // e.g. it is referenced by another group/instance, then
    // mGroup->build() does nothing.
    if (!mGroup->build(cudaStream, context, errorMsg)) {
        // error
        return false;
    }

    if (mHasMotionBlur) {
        // It's ugly, but this is how NVIDIA says you are supposed to set this up.
        // 12 * sizeof(float) = Optix 4x3 transform matrix size
        size_t transformSizeInBytes = sizeof(OptixMatrixMotionTransform)
                                      + (sNumMotionKeys - 2) * 12 * sizeof(float);
        OptixMatrixMotionTransform *transform =
            (OptixMatrixMotionTransform*) alloca(transformSizeInBytes);

        transform->child = mGroup->mTopLevelIAS;
        transform->motionOptions.numKeys = sNumMotionKeys;
        transform->motionOptions.timeBegin = 0.f;
        transform->motionOptions.timeEnd = 1.f;
        transform->motionOptions.flags = OPTIX_MOTION_FLAG_NONE;
        for (int i = 0; i < sNumMotionKeys; i++) {
            mXforms[i].toOptixTransform(transform->transform[i]);
        }

        mMMTTraversableBuf.allocAndUpload((char*)(transform), transformSizeInBytes);

        if (optixConvertPointerToTraversableHandle(context,
                                                   mMMTTraversableBuf.deviceptr(),
                                                   OPTIX_TRAVERSABLE_TYPE_MATRIX_MOTION_TRANSFORM,
                                                   &mMMTTraversable) != OPTIX_SUCCESS) {
            return false;
        }
    }

    return true;
}

} // namespace rt
} // namespace moonray

