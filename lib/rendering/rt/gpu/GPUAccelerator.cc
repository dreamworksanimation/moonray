// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>



#if defined(MOONRAY_USE_OPTIX) || defined(MOONRAY_USE_METAL)

#ifdef MOONRAY_USE_OPTIX
    #include "optix/OptixGPUAccelerator.h"
#define GPUAcceleratorType OptixGPUAccelerator
#endif 

#ifdef MOONRAY_USE_METAL
    #include "metal/MetalGPUAccelerator.h"
#define GPUAcceleratorType MetalGPUAccelerator
#endif 

#include "GPUAccelerator.h"

#ifndef __APPLE__
// This header must be included in exactly one .cc file for the link to succeed
#include <optix_function_table_definition.h>
#endif

namespace moonray {
namespace rt {

GPUAccelerator::GPUAccelerator(bool allowUnsupportedFeatures,
                               const uint32_t numCPUThreads,
                               const scene_rdl2::rdl2::Layer *layer,
                               const scene_rdl2::rdl2::SceneContext::GeometrySetVector& geometrySets,
                               const scene_rdl2::rdl2::Layer::GeometryToRootShadersMap* g2s,
                               std::vector<std::string>& warningMsgs,
                               std::string* errorMsg)
{
    mImpl.reset(new GPUAcceleratorType(
        allowUnsupportedFeatures, numCPUThreads, layer, geometrySets, g2s, warningMsgs, errorMsg));
    if (!errorMsg->empty()) {
        // Something went wrong so free everything
        // Output the error to Logger::error so we are guaranteed to see it
        scene_rdl2::logging::Logger::error("GPU: " + *errorMsg + "   ...falling back to CPU vectorized mode");
        mImpl.reset();
    }
}

GPUAccelerator::~GPUAccelerator()
{
}

std::string
GPUAccelerator::getGPUDeviceName() const
{
    return mImpl->getGPUDeviceName();
}

void
GPUAccelerator::intersect(const uint32_t queueIdx,
                          const uint32_t numRays,
                          const GPURay* rays) const
{
    mImpl->intersect(queueIdx, numRays, rays);
}

GPURayIsect*
GPUAccelerator::getOutputIsectBuf(const uint32_t queueIdx) const
{
    return mImpl->getOutputIsectBuf(queueIdx);
}

GPURay*
GPUAccelerator::getGPURaysBufUMA(const uint32_t queueIdx) const
{
    return mImpl->getGPURaysBufUMA(queueIdx);
}

void*
GPUAccelerator::getBundledOcclRaysBufUMA(const uint32_t queueIdx, uint32_t numRays, size_t stride) const
{
    return mImpl->getBundledOcclRaysBufUMA(queueIdx, numRays, stride);
}

void
GPUAccelerator::occluded(const uint32_t queueIdx,
                         const uint32_t numRays,
                         const GPURay* rays,
                         const void* bundledOcclRaysUMA,
                         size_t bundledOcclRayStride) const
{
    mImpl->occluded(queueIdx, numRays, rays, bundledOcclRaysUMA, bundledOcclRayStride);
}

unsigned char*
GPUAccelerator::getOutputOcclusionBuf(const uint32_t queueIdx) const
{
    return mImpl->getOutputOcclusionBuf(queueIdx);
}

size_t
GPUAccelerator::getCPUMemoryUsed() const
{
    return mImpl->getCPUMemoryUsed();
}

uint32_t
GPUAccelerator::getRaysBufSize()
{
    return GPUAcceleratorType::getRaysBufSize();
}

void*
GPUAccelerator::instanceIdToInstancePtr(unsigned int instanceId) const
{
    return mImpl->instanceIdToInstancePtr(instanceId);
}

} // namespace rt
} // namespace moonray

#else // not MOONRAY_USE_OPTIX nor MOONRAY_USE_METAL

#include "GPUAccelerator.h"

namespace moonray {
namespace rt {

GPUAccelerator::GPUAccelerator(bool /*allowUnsupportedFeatures*/,
                               const uint32_t numCPUThreads,
                               const scene_rdl2::rdl2::Layer* /*layer*/,
                               const scene_rdl2::rdl2::SceneContext::GeometrySetVector& /*geometrySets*/,
                               const scene_rdl2::rdl2::Layer::GeometryToRootShadersMap* /*g2s*/,
                               std::vector<std::string>& /*warningMsgs*/,
                               std::string* errorMsg)
{
   *errorMsg = "GPU mode not enabled in this build";
   scene_rdl2::logging::Logger::error("GPU: " + *errorMsg + "   ...falling back to CPU vectorized mode");
}

GPUAccelerator::~GPUAccelerator()
{
}

std::string
GPUAccelerator::getGPUDeviceName() const
{
    return "";
}

void
GPUAccelerator::intersect(const uint32_t /*queueIdx*/,
                          const uint32_t /*numRays*/,
                          const GPURay* /*rays*/) const
{
}

GPURayIsect*
GPUAccelerator::getOutputIsectBuf(const uint32_t /*queueIdx*/) const
{
    return nullptr;
}

GPURay*
GPUAccelerator::getGPURaysBufUMA(const uint32_t /* queueIdx */) const
{
    return nullptr;
}

void*
GPUAccelerator::getBundledOcclRaysBufUMA(const uint32_t /* queueIdx */,
                                         const uint32_t /* numRays */,
                                         const size_t /* stride */) const
{
    return nullptr;
}

void
GPUAccelerator::occluded(const uint32_t /* queueIdx */,
                         const uint32_t /* numRays */,
                         const GPURay* /* rays */,
                         const void* /* bundledOcclRaysUMA */,
                         const size_t /* bundledOcclRayStride */) const
{
}

unsigned char*
GPUAccelerator::getOutputOcclusionBuf(const uint32_t /*queueIdx*/) const
{
    return nullptr;
}

size_t
GPUAccelerator::getCPUMemoryUsed() const
{
    return 0;
}

uint32_t
GPUAccelerator::getRaysBufSize()
{
    return 0;
}

void*
GPUAccelerator::instanceIdToInstancePtr(unsigned int instanceId) const
{
    return nullptr;
}

} // namespace rt
} // namespace moonray

#endif // not MOONRAY_USE_OPTIX || MOONRAY_USE_METAL
