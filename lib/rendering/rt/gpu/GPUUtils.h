// Copyright 2023 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "GPUBuffer.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <optix.h>
#include <optix_stubs.h>

#include <string>
#include <vector>
#include <map>

namespace moonray {
namespace rt {

class GPUTriMesh;
class GPURoundCurves;
class GPUCustomPrimitive;

bool
getNVIDIADriverVersion(int* major, int* minor);

// Wrappers for the verbose and low-level Optix 7 API.  This makes the code
// dramatically cleaner overall and adapts to our error handling convention.

bool
createOptixContext(OptixLogCallback logCallback,
                   CUstream* cudaStream,
                   OptixDeviceContext* ctx,
                   std::string* deviceName,
                   std::string* errorMsg);

// A module is the compiled code contained in a .ptx file
bool
createOptixModule(OptixDeviceContext context,
                  const std::string& ptxPath,
                  const OptixModuleCompileOptions& moduleCompileOptions,
                  const OptixPipelineCompileOptions& pipelineCompileOptions,
                  OptixModule* module,
                  std::string* errorMsg);

// Primitives like round curves use a built-in module
bool
getBuiltinISModule(OptixDeviceContext context,
                   const OptixModuleCompileOptions& moduleCompileOptions,
                   const OptixPipelineCompileOptions& pipelineCompileOptions,
                   const OptixPrimitiveType primitiveType,
                   const bool motionBlur,
                   OptixModule* module,
                   std::string* errorMsg);

// Specifies the function to call to generate rays
bool
createOptixRaygenProgramGroup(OptixDeviceContext context,
                              OptixModule module,
                              const char* functionName,
                              OptixProgramGroup *pg,
                              std::string* errorMsg);

// Specifies the function to call when the ray doesn't hit geometry
bool
createOptixMissProgramGroup(OptixDeviceContext context,
                            OptixModule module,
                            const char* functionName,
                            OptixProgramGroup *pg,
                            std::string* errorMsg);

// Specifies the functions to call when a ray intersects geometry
bool
createOptixHitGroupProgramGroup(OptixDeviceContext context,
                                OptixModule anyHitModule,
                                const char* anyHitFunctionName,
                                OptixModule closestHitModule,
                                const char* closestHitFunctionName,
                                OptixModule intersectionModule,
                                const char* intersectionFunctionName,
                                OptixProgramGroup *pg,
                                std::string* errorMsg);

// Pipeline contains a set of program groups
bool
createOptixPipeline(OptixDeviceContext context,
                    const OptixPipelineCompileOptions& pipelineCompileOptions,
                    const OptixPipelineLinkOptions& pipelineLinkOptions,
                    const std::map<std::string, OptixProgramGroup>& programGroups,
                    OptixPipeline* pipeline,
                    std::string* errorMsg);

// Creates an acceleration structure (BVH) that contains other acceleration structures
bool
createOptixAccel(OptixDeviceContext context,
                 CUstream cudaStream,
                 const OptixAccelBuildOptions& accelOptions,
                 const std::vector<OptixBuildInput>& inputs,
                 bool compact,
                 GPUBuffer<char>* accelBuf,
                 OptixTraversableHandle* accelHandle,
                 std::string* errorMsg);

// GAS == Geometry Acceleration Structure (a BVH)
bool
createTrianglesGAS(CUstream cudaStream,
                   OptixDeviceContext optixContext,
                   const std::vector<GPUTriMesh*>& triMeshes,
                   OptixTraversableHandle* accel,
                   GPUBuffer<char>* accelBuf,
                   std::string* errorMsg);

bool
createRoundCurvesGAS(CUstream cudaStream,
                     OptixDeviceContext optixContext,
                     const std::vector<GPURoundCurves*>& roundCurves,
                     OptixTraversableHandle* accel,
                     GPUBuffer<char>* accelBuf,
                     std::string* errorMsg);

// Custom primitives -> any non-triangle geometry as triangles are built-in
// in Optix 7 and are a special HW-accelerated case.
bool
createCustomPrimitivesGAS(CUstream cudaStream,
                          OptixDeviceContext optixContext,
                          const std::vector<GPUCustomPrimitive*>& primitives,
                          OptixTraversableHandle* accel,
                          GPUBuffer<char>* accelBuf,
                          std::string* errorMsg);

} // namespace rt
} // namespace moonray



