// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <moonray/rendering/geom/prim/Instance.h>

#include "OptixGPUMath.h"
#include "OptixGPUParams.h"
#include "OptixGPUPrimitive.h"
#include "OptixGPUPrimitiveGroup.h"
#include "OptixGPUSBTRecord.h"
#include "OptixGPUUtils.h"

#include <moonray/rendering/geom/Types.h>
#include <moonray/rendering/geom/prim/Primitive.h>
#include <scene_rdl2/scene/rdl2/RootShader.h>

#include <tbb/concurrent_unordered_map.h>

namespace moonray {
namespace rt {

// Also in EmbreeAccelerator.cc
typedef tbb::concurrent_unordered_map<std::shared_ptr<geom::SharedPrimitive>,
        tbb::atomic<OptixGPUPrimitiveGroup*>, geom::SharedPtrHash> SharedGroupMap;


class OptixGPUAccelerator
{
public:
    OptixGPUAccelerator(bool allowUnsupportedFeatures,
                        const scene_rdl2::rdl2::Layer *layer,
                        const scene_rdl2::rdl2::SceneContext::GeometrySetVector& geometrySets,
                        const scene_rdl2::rdl2::Layer::GeometryToRootShadersMap* g2s,
                        std::vector<std::string>& warningMsgs,
                        std::string* errorMsg);
    ~OptixGPUAccelerator();

    std::string getGPUDeviceName() const;

    void intersect(const unsigned numRays, const GPURay* rays) const;

    GPURayIsect* getOutputIsectBuf() const { return nullptr; /* TODO */};

    void occluded(const unsigned numRays, const GPURay* rays) const;

    unsigned char* getOutputOcclusionBuf() const { return mOutputOcclusionBuf; }

    static unsigned int getRaysBufSize() { return mRaysBufSize; }

private:
    bool build(CUstream cudaStream,
               OptixDeviceContext context,
               const scene_rdl2::rdl2::Layer *layer,
               const scene_rdl2::rdl2::SceneContext::GeometrySetVector& geometrySets,
               const scene_rdl2::rdl2::Layer::GeometryToRootShadersMap* g2s,
               std::vector<std::string>& warningMsgs,
               std::string* errorMsg);

    bool createProgramGroups(std::string* errorMsg);

    bool createShaderBindingTable(std::string* errorMsg);

    bool mAllowUnsupportedFeatures;

    CUstream mCudaStream;
    std::string mGPUDeviceName;

    OptixDeviceContext mContext;

    // The module corresponds to one .ptx file of compiled CUDA code.  Everything is in
    // one .cu source file so there is only one module.
    OptixModule mModule;

    // There are special built-in modules for Optix round curves
    OptixModule mRoundLinearCurvesModule;
    OptixModule mRoundLinearCurvesMBModule;
    OptixModule mRoundCubicBsplineCurvesModule;
    OptixModule mRoundCubicBsplineCurvesMBModule;

    // Specifies the programs to call for ray generation and for each different type of
    // geometry (OptixGPUPrimitive):
    // "intersection" , "closest hit" and "any hit" programs called during BVH traversal.
    std::map<std::string, OptixProgramGroup> mProgramGroups;

    // All of the program groups are bound together into a pipeline.  Multiple pipelines
    // are possible, but we only have one set of program groups.
    OptixPipeline mPipeline;

    // Geometry is contained in primitive groups.  The root group is the root (non-shared/instanced)
    // geometry.  The shared groups are the same as the shared "scenes" in the CPU-side
    // EmbreeAccelerator.  These are referenced by OptixGPUInstances.
    // For instancing/multi-level instancing, a primitive group may contain instances
    // that reference other primitive groups.
    OptixGPUPrimitiveGroup* mRootGroup;
    SharedGroupMap mSharedGroups;

    // The Shader Binding Table provides data (records) for each of the OptixGPUPrimitives that is
    // used in the various programs (intersection, closest hit, any hit.)  It maps 1:1
    // to the OptixGPUPrimitives in the GAS objects above.  If you get these out of sync, you will
    // have a bad day.
    OptixShaderBindingTable mSBT;

    // Buffers for the records in the Shader Binding Table.  Note how these match the
    // program groups.
    OptixGPUBuffer<RaygenRecord> mRaygenRecordBuf;
    OptixGPUBuffer<MissRecord> mMissRecordBuf;
    OptixGPUBuffer<HitGroupRecord> mHitGroupRecordBuf;

    // Buffers needed to pass rays to the GPU and read back intersection results.
    // The rays buffer size is the same as the ray queue size in RenderDriver::createXPUQueue()
    // and was determined empirically through performance testing.
    static const unsigned int mRaysBufSize = 262144;
    mutable OptixGPUBuffer<GPURay> mRaysBuf;

    // pinned host memory to avoid an extra copy in the GPU driver
    // when copying occlusion results from the GPU
    mutable unsigned char* mOutputOcclusionBuf;

    // occluded() results
    mutable OptixGPUBuffer<unsigned char> mIsOccludedBuf;

    // A parameters object that is globally available on the GPU side.
    mutable OptixGPUBuffer<OptixGPUParams> mParamsBuf;
};

} // namespace rt
} // namespace moonray

