// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

#include <Metal/Metal.h>
#include "MetalGPUInstance.h"
#include "MetalGPUPrimitiveGroup.h"
#include "MetalGPUUtils.h"

namespace moonray {
namespace rt {

bool
MetalGPUInstance::build(id<MTLDevice> context,
                   id<MTLCommandQueue> queue,
                   std::vector<id<MTLAccelerationStructure>>* bottomLevelAS,
                   std::atomic<int> &structuresBuilding,
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
    if (!mGroup->build(context, queue, 0, bottomLevelAS, errorMsg)) {

        // error
        return false;
    }

    return true;
}

} // namespace rt
} // namespace moonray

