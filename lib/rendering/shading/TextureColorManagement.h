// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <OpenColorIO/OpenColorIO.h>

#include <string>

namespace OCIO = OCIO_NAMESPACE;

namespace moonray {
namespace shading {
namespace texture_color_management {

struct ProcessorResult
{
    OCIO::ConstCPUProcessorRcPtr mProcessor;
    std::string mDiagnostic;
};

ProcessorResult createTextureProcessor(const std::string& filename,
                                       const std::string& sourceColorSpace);

void applyProcessor(intptr_t processorPtr, float* rgba);

} // namespace texture_color_management
} // namespace shading
} // namespace moonray
