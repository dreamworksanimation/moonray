// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "MetalGPUInstance.h"
#include "MetalGPUPrimitive.h"
#include "MetalGPUSBTRecord.h"

#include <map>

namespace moonray {
namespace rt {

class MetalGPUPrimitiveGroup
{
public:
    MetalGPUPrimitiveGroup(id<MTLDevice> context) :
        mIsBuilt(false),
        mAccelStructBaseID(0),
        mTopLevelIAS(nil)
    {}

    virtual ~MetalGPUPrimitiveGroup();

    // Sets the shader binding table offset and updates the current sbt offset
    // value.  This function is called in a loop that loops over the groups to
    // assign their sbt offsets.
    void setAccelStructBaseID(unsigned int& baseID);

    void hasMotionBlur(uint32_t *flags);
    
    bool build(id<MTLDevice> context,
               id<MTLCommandQueue> queue,
               uint32_t motionBlurFlags,
               std::vector<id<MTLAccelerationStructure>>* bottomLevelAS,
               std::string* errorMsg);

    void getSBTRecords(std::vector<HitGroupRecord>& hitgroupRecs,
                       std::vector<id<MTLBuffer>>& usedResources);

    // build() may be called multiple times if there is instancing as the MetalGPUInstances
    // need to build their primitive groups but the group may be shared across multiple
    // MetalGPUInstances.  We only want to actually build once, so we need a flag.
    bool mIsBuilt;

    // The baseID to use for assigning unique user IDs to the acceleration structs.
    unsigned int mAccelStructBaseID;

    // The MetalGPUTriMesh objects are the host-side representation of the triangle meshes
    // on the GPU.  They manage the memory on the GPU and provide functionality needed
    // on the host side to constuct the BVH, such as AABB queries.
    // TriMeshes are hardware-accelerated (ray-triangle intersection) and are a special
    // case in Optix that can't be mixed with other types of primitives.  You do not
    // need to specify an "intersection" program for triangle meshes because it is baked
    // into the GPU.
    std::vector<MetalGPUTriMesh*> mTriMeshes;
    std::vector<MetalGPUTriMesh*> mTriMeshesMB;

    // GAS = geometry acceleration structure.  A BVH (a.k.a. Traversable) containing
    // only primitives (not other acceleration structures (AS.)

    std::vector<id<MTLAccelerationStructure>> mTrianglesGAS;    // the handles to the GAS
    std::vector<id<MTLAccelerationStructure>> mTrianglesMBGAS;  // the handles to the motion blurred GAS

    // Similar to mTriMeshes, this is the host-side representation of the round curves.
    std::vector<MetalGPURoundCurves*> mRoundCurves;
    std::vector<id<MTLAccelerationStructure>> mRoundCurvesGAS;  // the handles to the GAS

    std::vector<MetalGPURoundCurves*> mRoundCurvesMB;
    std::vector<id<MTLAccelerationStructure>> mRoundCurvesMBGAS;  // the handles to the GAS

    // GPUCustomPrimitive is used for any non-TriMesh primitives.  These all must
    // have intersection programs.  We can support any kind of geometry that we can
    // write an intersection program for.
    std::vector<MetalGPUCustomPrimitive*> mCustomPrimitives;
    std::vector<id<MTLAccelerationStructure>> mCustomPrimitivesGAS;

    // Instances that reference other groups.
    std::vector<MetalGPUInstance*> mInstances;

    // Since you can't mix trimeshes and custom primitives together in a GAS, we create
    // two GAS (one for trimeshes and one for the custom primitives) and put these into
    // an IAS (Instance Acceleration Structure.)  This IAS also contains the instances
    // that reference other GPUPrimitiveGroups.
    id<MTLAccelerationStructure> mTopLevelIAS;

    struct AccelData {
        AccelData(id<MTLAccelerationStructure> _as, int _userID)
        : as(_as)
        , userID(_userID)
        , motionKeys(nullptr)
        , numMotionKeys(1)
        , options(MTLAccelerationStructureInstanceOptionNone) {
            float *t = (float*)&xForm;
            /* use identity matrix */
            t[0] = t[4] = t[8] = 1.0f;
        }

        MTLPackedFloat4x3 xForm;
        id<MTLAccelerationStructure> as;
        MetalGPUXform *motionKeys;
        int numMotionKeys; // =1 indicates no motion keys
        int userID;
        MTLAccelerationStructureInstanceOptions options;
    };
    std::vector<AccelData> mUserInstanceIDs;
};

} // namespace rt
} // namespace moonray
