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
GPUAccelerator::intersect(const unsigned numRays, const GPURay* rays) const
{
    mImpl->intersect(numRays, rays);
}

GPURayIsect*
GPUAccelerator::getOutputIsectBuf() const
{
    return mImpl->getOutputIsectBuf();
}

GPURay*
GPUAccelerator::getGPURaysBuf(const uint32_t queueIdx) const
{
    return mImpl->getGPURaysBuf(queueIdx);
}

void*
GPUAccelerator::getCPURayBuf(const uint32_t queueIdx, size_t numRays, size_t stride) const
{
    return mImpl->getCPURayBuf(queueIdx, numRays, stride);
}

void
GPUAccelerator::occluded(const uint32_t queueIdx,
                         const unsigned numRays,
                         const GPURay* rays,
                         const void* cpuRays,
                         size_t cpuRayStride) const
{
    mImpl->occluded(queueIdx, numRays, rays, cpuRays, cpuRayStride);
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

unsigned int
GPUAccelerator::getRaysBufSize()
{
    return GPUAcceleratorType::getRaysBufSize();
}

bool
GPUAccelerator::getUMAAvailable()
{
    return GPUAcceleratorType::getUMAAvailable();
}

bool
GPUAccelerator::supportsMultipleQueues()
{
    return GPUAcceleratorType::supportsMultipleQueues();
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
GPUAccelerator::intersect(const unsigned /*numRays*/, const GPURay* /*rays*/) const
{
}

GPURayIsect*
GPUAccelerator::getOutputIsectBuf() const
{
    return nullptr;
}

GPURay*
GPUAccelerator::getGPURaysBuf(const uint32_t /* queueIdx */) const
{
    return nullptr;
}

void*
GPUAccelerator::getCPURayBuf(const uint32_t /* queueIdx */,
                             size_t /* numRays */,
                             size_t /* stride */) const
{
    return nullptr;
}

void
GPUAccelerator::occluded(const uint32_t /* queueIdx */,
                         const unsigned /* numRays */,
                         const GPURay* /* rays */,
                         const void* /* cpuRays */,
                         size_t /* cpuRayStride */) const
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

unsigned int
GPUAccelerator::getRaysBufSize()
{
    return 0;
}

bool
GPUAccelerator::getUMAAvailable()
{
    return false;
}

bool
GPUAccelerator::supportsMultipleQueues()
{
    return false;
}

} // namespace rt
} // namespace moonray

#endif // not MOONRAY_USE_OPTIX || MOONRAY_USE_METAL
