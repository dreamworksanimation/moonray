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
        std::shared_ptr<std::atomic<OptixGPUPrimitiveGroup*>>, geom::SharedPtrHash> SharedGroupMap;


class OptixGPUAccelerator
{
public:
    OptixGPUAccelerator(bool allowUnsupportedFeatures,
                        const uint32_t numCPUThreads,
                        const scene_rdl2::rdl2::Layer *layer,
                        const scene_rdl2::rdl2::SceneContext::GeometrySetVector& geometrySets,
                        const scene_rdl2::rdl2::Layer::GeometryToRootShadersMap* g2s,
                        std::vector<std::string>& warningMsgs,
                        std::string* errorMsg);
    ~OptixGPUAccelerator();

    std::string getGPUDeviceName() const;

    void intersect(const uint32_t queueIdx,
                   const uint32_t numRays,
                   const GPURay* rays) const;

    GPURayIsect* getOutputIsectBuf(const uint32_t queueIdx) const {
        return mOutputIsectBuf[queueIdx];
    };

    void occluded(const uint32_t queueIdx,
                  const uint32_t numRays,
                  const GPURay* rays,
                  const void* /* bundledOcclRaysUMA - unused for Optix*/,
                  const size_t /* bundledOcclRayStride - unused for Optix*/) const;

    GPURay* getGPURaysBufUMA(const uint32_t queueIdx) const {
        return nullptr;
    }

    void* getBundledOcclRaysBufUMA(const uint32_t queueIdx,
                                   const uint32_t numRays,
                                   const size_t stride) const {
        return nullptr;
    }

    unsigned char* getOutputOcclusionBuf(const uint32_t queueIdx) const {
        return mOutputOcclusionBuf[queueIdx];
    }

    size_t getCPUMemoryUsed() const;

    static uint32_t getRaysBufSize() { return mRaysBufSize; }

    void* instanceIdToInstancePtr(unsigned int instanceId) const;

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

    uint32_t mNumCPUThreads; // number of CPU threads that can call the GPU

    bool mAllowUnsupportedFeatures;

    CUstream mCudaStream;
    std::vector<CUstream> mCudaStreams; // per-thread (queue) streams

    std::string mGPUDeviceName;

    OptixDeviceContext mContext;

    // The module corresponds to one .ptx file of compiled CUDA code.  Everything is in
    // one .cu source file so there is only one module.
    OptixModule mModule;

    // Specifies the programs to call for ray generation and for each different type of
    // geometry (OptixGPUPrimitive):
    // "intersection" , "closest hit" and "any hit" programs called during BVH traversal.
    std::map<std::string, OptixProgramGroup> mProgramGroups;

    // All of the program groups are bound together into a pipeline.  Multiple pipelines
    // share one set of program groups.
    std::vector<OptixPipeline> mPipeline; // per-thread (queue) pipelines

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
    static const uint32_t mRaysBufSize = 65536;
     
    mutable std::vector<OptixGPUBuffer<GPURay>> mRaysBuf; // per-thread (queue) input ray buffers

    // pinned host memory to avoid an extra copy in the GPU driver
    // when copying results from the GPU
    mutable std::vector<unsigned char*> mOutputOcclusionBuf; // per-thread (or queue) output result buffers
    mutable std::vector<GPURayIsect*> mOutputIsectBuf; // per-thread (or queue) output result buffers

    // Results
    mutable std::vector<OptixGPUBuffer<unsigned char>> mIsOccludedBuf;
    mutable std::vector<OptixGPUBuffer<GPURayIsect>> mIsectBuf;

    // A parameters object that is globally available on the GPU side.
    mutable std::vector<OptixGPUBuffer<OptixGPUParams>> mParamsBuf; // per-thread (queue) param buffers

    // Optix only supports a 32-bit instance ID but we need the original 64-bit Instance* pointer from
    // the ray intersection.
    std::vector<void*> mInstanceIdToInstancePtr;
};

} // namespace rt
} // namespace moonray

