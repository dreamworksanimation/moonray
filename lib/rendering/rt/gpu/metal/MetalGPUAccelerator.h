// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <moonray/rendering/geom/prim/Instance.h>

#include "MetalGPUMath.h"
#include "MetalGPUParams.h"
#include "MetalGPUSBTRecord.h"
#include "MetalGPUPrimitive.h"
#include "MetalGPUPrimitiveGroup.h"
#include "MetalGPUUtils.h"

#include <moonray/rendering/geom/Types.h>
#include <moonray/rendering/geom/prim/Primitive.h>
#include <scene_rdl2/scene/rdl2/RootShader.h>

#include <tbb/concurrent_unordered_map.h>

namespace moonray {
namespace rt {

// Also in EmbreeAccelerator.cc
typedef tbb::concurrent_unordered_map<std::shared_ptr<geom::SharedPrimitive>,
        tbb::atomic<MetalGPUPrimitiveGroup*>, geom::SharedPtrHash> SharedGroupMap;


class MetalGPUAccelerator
{
public:
    MetalGPUAccelerator(bool allowUnsupportedFeatures,
                       const uint32_t numCPUThreads,
                       const scene_rdl2::rdl2::Layer *layer,
                       const scene_rdl2::rdl2::SceneContext::GeometrySetVector& geometrySets,
                       const scene_rdl2::rdl2::Layer::GeometryToRootShadersMap* g2s,
                       std::vector<std::string>& warningMsgs,
                       std::string* errorMsg);
    ~MetalGPUAccelerator();

    std::string getGPUDeviceName() const;

    void intersect(const uint32_t queueIdx,
                   const uint32_t numRays,
                   const GPURay* rays) const;

    GPURayIsect* getOutputIsectBuf(const uint32_t queueIdx) const { return nullptr; /* TODO */};

    unsigned char* getOutputOcclusionBuf(const uint32_t queueIdx) const {
        return mIsOccludedBuf[queueIdx].cpu_ptr();
    }
    GPURay* getGPURaysBufUMA(const uint32_t queueIdx) const {
        return mRaysBuf[queueIdx].cpu_ptr();
    }
    void* getBundledOcclRaysBufUMA(const uint32_t queueIdx,
                                   const uint32_t numRays,
                                   const size_t stride) const;

    void occluded(const uint32_t queueIdx,
                  const uint32_t numRays,
                  const GPURay* rays,
                  const void* bundledOcclRaysUMA,
                  const size_t bundledOcclRayStride) const;

    size_t getCPUMemoryUsed() const { return 0; }

    static uint32_t getRaysBufSize() { return mRaysBufSize; }

    void* instanceIdToInstancePtr(unsigned int instanceId) const;

private:
    bool build(const scene_rdl2::rdl2::Layer *layer,
               const scene_rdl2::rdl2::SceneContext::GeometrySetVector& geometrySets,
               const scene_rdl2::rdl2::Layer::GeometryToRootShadersMap* g2s,
               std::vector<std::string>& warningMsgs,
               std::string* errorMsg);

    bool createIntersectionFunctions(std::string* errorMsg);

    bool createShaderBindingTable(std::string* errorMsg);
    
    bool mAllowUnsupportedFeatures;

    void prepareEncoder(const uint32_t queueIdx) const;

    std::string mGPUDeviceName;

    id<MTLDevice> mContext;

    // The module corresponds to one .ptx file of compiled CUDA code.  Everything is in
    // one .metalLib source file so there is only one module.
    id<MTLLibrary> mModule;
    
    enum {
        PSO_BASE,
        PSO_MOTION_BLUR_INSTANCING,
        PSO_MOTION_BLUR_PRIMITIVES,
        PSO_MOTION_BLUR_INSTANCING_PRIMITIVES,
        PSO_NUM_GROUPS,
    };
    struct PSOGroup {
        id<MTLComputePipelineState> rayGenPSO = nil;
        id<MTLIntersectionFunctionTable> intersectFuncTable = nil;
        NSMutableArray* linkedFunctions = nil;
    };
    PSOGroup mPSOs[PSO_NUM_GROUPS];

    // Geometry is contained in primitive groups.  The root group is the root (non-shared/instanced)
    // geometry.  The shared groups are the same as the shared "scenes" in the CPU-side
    // EmbreeAccelerator.  These are referenced by GPUInstances.
    // For instancing/multi-level instancing, a primitive group may contain instances
    // that reference other primitive groups.
    MetalGPUPrimitiveGroup* mRootGroup;
    SharedGroupMap mSharedGroups;
    std::vector<id<MTLAccelerationStructure>> mBottomLevelAS;

    // The Shader Binding Table provides data (records) for each of the GPUPrimitives that is
    // used in the various programs (intersection, closest hit, any hit.)  It maps 1:1
    // to the GPUPrimitives in the GAS objects above.  If you get these out of sync, you will
    // have a bad day.
//    OptixShaderBindingTable mSBT;

    // Buffers for the records in the Shader Binding Table.  Note how these match the
    // program groups.
    MetalGPUBuffer<HitGroupRecord> mHitGroupRecordBuf;
    std::vector<id<MTLBuffer>> mUsedIndirectResources;

    // Buffers needed to pass rays to the GPU and read back intersection results.
    // The rays buffer size is the same as the ray queue size in RenderDriver::createXPUQueue()
    // and was determined empirically through performance testing.
    static const unsigned int mRaysBufSize = 262144;

    mutable std::vector<MetalGPUBuffer<GPURay>> mRaysBuf;
    mutable std::vector<MetalGPUBuffer<unsigned char>> mIsOccludedBuf;

    // A parameters object that is globally available on the GPU side.
    mutable std::vector<MetalGPUBuffer<MetalGPUParams>> mParamsBuf;

    uint32_t mHasMotionBlur;

    mutable MetalGPUBuffer<float> mDebugBuf;
    
    struct CommandEncodingState {
        id<MTLCommandQueue> mQueue = nil;
        id<MTLCommandBuffer> commandBuffer = nil;
        id<MTLComputeCommandEncoder> encoder = nil;
        id<MTLBuffer> cpuBuffer = nil;
    };
    
    // This is stores a GPU state for each CPU thread, allowning a lockless and
    // fully parallel 
    mutable std::vector<CommandEncodingState> mEncoderStates;
};

} // namespace rt
} // namespace moonray

