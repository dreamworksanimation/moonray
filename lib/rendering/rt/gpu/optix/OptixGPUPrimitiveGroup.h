// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "OptixGPUInstance.h"
#include "OptixGPUPrimitive.h"
#include "OptixGPUSBTRecord.h"

#include <map>

namespace moonray {
namespace rt {

class OptixGPUPrimitiveGroup
{
public:
    OptixGPUPrimitiveGroup() : mIsBuilt(false), mSBTOffset(0) {}

    virtual ~OptixGPUPrimitiveGroup();

    // Sets the shader binding table offset and updates the current sbt offset
    // value.  This function is called in a loop that loops over the groups to
    // assign their sbt offsets.
    void setSBTOffset(unsigned int& sbtOffset);

    bool build(CUstream cudaStream,
               OptixDeviceContext context,
               std::string* errorMsg);

    void getSBTRecords(std::map<std::string, OptixProgramGroup>& pgs,
                       std::vector<HitGroupRecord>& hitgroupRecs);

    // build() may be called multiple times if there is instancing as the OptixGPUInstances
    // need to build their primitive groups but the group may be shared across multiple
    // OptixGPUInstances.  We only want to actually build once, so we need a flag.
    bool mIsBuilt;

    // The shader binding table offset for this group.  Each group has a different offset.
    unsigned int mSBTOffset;

    // The OptixGPUTriMesh objects are the host-side representation of the triangle meshes
    // on the GPU.  They manage the memory on the GPU and provide functionality needed
    // on the host side to constuct the BVH, such as AABB queries.
    // TriMeshes are hardware-accelerated (ray-triangle intersection) and are a special
    // case in Optix that can't be mixed with other types of primitives.  You do not
    // need to specify an "intersection" program for triangle meshes because it is baked
    // into the GPU.
    std::vector<OptixGPUTriMesh*> mTriMeshes;
    std::vector<OptixGPUTriMesh*> mTriMeshesMB;

    // GAS = geometry acceleration structure.  A BVH (a.k.a. Traversable) containing
    // only primitives (not other acceleration structures (AS.)
    OptixTraversableHandle mTrianglesGAS;    // the handle to the GAS
    OptixGPUBuffer<char> mTrianglesGASBuf;        // the actual memory buffer on the GPU
    OptixTraversableHandle mTrianglesMBGAS;  // the handle to the motion blurred GAS
    OptixGPUBuffer<char> mTrianglesMBGASBuf;      // the actual memory buffer on the GPU

    // Similar to mTriMeshes, this is the host-side representation of the round curves.
    std::vector<OptixGPURoundCurves*> mRoundCurves;
    OptixTraversableHandle mRoundCurvesGAS;  // the handle to the GAS
    OptixGPUBuffer<char> mRoundCurvesGASBuf;      // the actual memory buffer on the GPU

    std::vector<OptixGPURoundCurves*> mRoundCurvesMB;
    OptixTraversableHandle mRoundCurvesMBGAS;  // the handle to the GAS
    OptixGPUBuffer<char> mRoundCurvesMBGASBuf;      // the actual memory buffer on the GPU

    // OptixGPUCustomPrimitive is used for any non-TriMesh primitives.  These all must
    // have intersection programs.  We can support any kind of geometry that we can
    // write an intersection program for.
    std::vector<OptixGPUCustomPrimitive*> mCustomPrimitives;
    OptixTraversableHandle mCustomPrimitivesGAS;
    OptixGPUBuffer<char> mCustomPrimitivesGASBuf;

    // Instances that reference other groups.
    std::vector<OptixGPUInstance*> mInstances;

    // Since you can't mix trimeshes and custom primitives together in a GAS, we create
    // two GAS (one for trimeshes and one for the custom primitives) and put these into
    // an IAS (Instance Acceleration Structure.)  This IAS also contains the instances
    // that reference other OptixGPUPrimitiveGroups.
    OptixTraversableHandle mTopLevelIAS;
    OptixGPUBuffer<char> mTopLevelIASBuf;
};

} // namespace rt
} // namespace moonray

