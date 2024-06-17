// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "MetalGPUBuffer.h"


#include <string>
#include <vector>
#include <map>

namespace moonray {
namespace rt {

class MetalGPUTriMesh;
class MetalGPURoundCurves;
class MetalGPUCustomPrimitive;

bool
getNVIDIADriverVersion(int* major, int* minor);

// Wrappers for the verbose and low-level Optix 7 API.  This makes the code
// dramatically cleaner overall and adapts to our error handling convention.

id<MTLDevice>
createMetalContext();

bool
validateMetalContext(id<MTLDevice> device,
                     std::string* deviceName,
                     std::string* errorMsg);

// A module is the compiled code contained in a .ptx file
bool
createMetalLibrary(id<MTLDevice> context,
                   const std::string& metalLibPath,
                   id<MTLLibrary>* library,
                   std::string* errorMsg);

// Specifies the function to call to generate rays
bool
createMetalRaygenKernel(id<MTLDevice> context,
                        id<MTLLibrary> library,
                        NSArray* linkedFunctions,
                        const char* functionName,
                        id<MTLComputePipelineState>* pso,
                        std::string* errorMsg);

bool
createIntersectionFunctionTables(NSArray* linkedFunctions,
                                 id<MTLComputePipelineState> pso,
                                 id<MTLIntersectionFunctionTable>* intersectFuncTable,
                                 std::string* errorMsg);

// Creates an acceleration structure (BVH) that contains other acceleration structures
bool
createMetalAccel(id<MTLDevice> context,
                 id<MTLCommandQueue> queue,
                 std::atomic<int> &structuresBuilding,
                 MTLAccelerationStructureDescriptor* input,
                 NSString * aLabel,
                 bool compact,
                 id<MTLAccelerationStructure>* accelHandle,
                 id<MTLBuffer> tempBufferToFree,
                 std::string* errorMsg);

// GAS == Geometry Acceleration Structure (a BVH)
bool
createTrianglesGAS(id<MTLDevice> context,
                   id<MTLCommandQueue> queue,
                   std::atomic<int> &structuresBuilding,
                   const std::vector<MetalGPUTriMesh*>& triMeshes,
                   std::vector<id<MTLAccelerationStructure>>& accels,
                   std::string* errorMsg);

bool
createRoundCurvesGAS(id<MTLDevice> context,
                     id<MTLCommandQueue> queue,
                     std::atomic<int> &structuresBuilding,
                     const std::vector<MetalGPURoundCurves*>& roundCurves,
                     std::vector<id<MTLAccelerationStructure>>& accels,
                     std::string* errorMsg);

// Custom primitives -> any non-triangle geometry as triangles are built-in
// in Optix 7 and are a special HW-accelerated case.
bool
createCustomPrimitivesGAS(id<MTLDevice> context,
                          id<MTLCommandQueue> queue,
                          std::atomic<int> &structuresBuilding,
                          const std::vector<MetalGPUCustomPrimitive*>& primitives,
                          std::vector<id<MTLAccelerationStructure>>& accels,
                          std::string* errorMsg);

} // namespace rt
} // namespace moonray

