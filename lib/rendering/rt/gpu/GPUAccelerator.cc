// Copyright 2023 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

#ifdef MOONRAY_USE_CUDA

#include "GPUAcceleratorImpl.h"
#include "GPUAccelerator.h"

// This header must be included in exactly one .cc file for the link to succeed
#include <optix_function_table_definition.h>

namespace moonray {
namespace rt {

GPUAccelerator::GPUAccelerator(const scene_rdl2::rdl2::Layer *layer,
                               const scene_rdl2::rdl2::SceneContext::GeometrySetVector& geometrySets,
                               const scene_rdl2::rdl2::Layer::GeometryToRootShadersMap* g2s,
                               std::string* errorMsg)
{
    mImpl.reset(new GPUAcceleratorImpl(layer, geometrySets, g2s, errorMsg));
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

unsigned char*
GPUAccelerator::getOutputOcclusionBuf() const
{
    return mImpl->getOutputOcclusionBuf();
}

void
GPUAccelerator::occluded(const unsigned numRays, const GPUOcclusionRay* rays) const
{
    mImpl->occluded(numRays, rays);
}

unsigned int
GPUAccelerator::getRaysBufSize()
{
    return GPUAcceleratorImpl::getRaysBufSize();
}

} // namespace rt
} // namespace moonray

#else // not MOONRAY_USE_CUDA

#include "GPUAccelerator.h"

namespace moonray {
namespace rt {

GPUAccelerator::GPUAccelerator(const scene_rdl2::rdl2::Layer* /*layer*/,
                               const scene_rdl2::rdl2::SceneContext::GeometrySetVector& /*geometrySets*/,
                               const scene_rdl2::rdl2::Layer::GeometryToRootShadersMap* /*g2s*/,
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

unsigned char*
GPUAccelerator::getOutputOcclusionBuf() const
{
    return nullptr;
}

void
GPUAccelerator::occluded(const unsigned /*numRays*/, const GPUOcclusionRay* /*rays*/) const
{
}

unsigned int
GPUAccelerator::getRaysBufSize()
{
    return 0;
}

} // namespace rt
} // namespace moonray

#endif // not MOONRAY_USE_CUDA
