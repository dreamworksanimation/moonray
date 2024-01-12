// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

#include "OptixGPUUtils.h"

#include "OptixGPUBuffer.h"
#include "OptixGPUPrimitive.h"

#undef min  // or compile error with std::min in optix_stack_size.h
#undef max
#include <optix_stack_size.h>

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

// These utility functions all follow the same pattern: fill in Optix struct(s),
// call optix API function, handle errors.  The functions condense Optix's
// verbose low-level API into something much easier to read in the main GPUAccelerator
// code.

#define MAX_LOGSTRING_SIZE 2048

namespace moonray {
namespace rt {

bool
getNVIDIADriverVersion(int* major, int* minor)
{
    *major = 0;
    *minor = 0;
    bool success = false;
    FILE *fp = fopen("/sys/module/nvidia/version", "r");
    if (fp != NULL) {
        if (fscanf(fp, "%d.%d", major, minor) == 2) {
            success = true;
        }
        fclose(fp);
    }
    return success;
}

bool
createOptixContext(OptixLogCallback logCallback,
                   CUstream* cudaStream,
                   OptixDeviceContext* ctx,
                   std::string* deviceName,
                   std::string* errorMsg)
{
    *errorMsg = "";

    int major, minor;
    if (!getNVIDIADriverVersion(&major, &minor)) {
        *errorMsg = "Unable to query NVIDIA driver version";
        return false;
    }
    if (major < 525) {
        *errorMsg = "NVIDIA driver too old, must be >= 525";
        return false;
    }

    cudaFree(0);    // init CUDA

    int numDevices = 0;
    cudaGetDeviceCount(&numDevices);
    if (numDevices == 0) {
        *errorMsg = "No CUDA capable devices found";
        return false;
    }

    const int deviceID = 0;
    if (cudaSetDevice(deviceID) != cudaSuccess) {
        *errorMsg = "Unable to set the CUDA device";
        return false;
    }

    cudaDeviceProp cudaDeviceProps;
    if (cudaGetDeviceProperties(&cudaDeviceProps, deviceID) != cudaSuccess) {
        *errorMsg = "Unable to get the CUDA device properties";
        return false;
    }
    *deviceName = cudaDeviceProps.name;

    if (cudaDeviceProps.major < 6) {
        *errorMsg = "GPU too old, must be compute capability 6 or greater";
        return false;
    }

    if (cudaStreamCreate(cudaStream) != cudaSuccess) {
        *errorMsg = "Unable to create the CUDA stream";
        return false;
    }

    if (optixInit() != OPTIX_SUCCESS) {
        *errorMsg = "Unable to initialize the Optix API";
        return false;
    }

    CUcontext cudaContext = 0;  // zero means take the current context
    OptixDeviceContextOptions options = {};
    if (optixDeviceContextCreate(cudaContext, &options, ctx) != OPTIX_SUCCESS) {
        cudaStreamDestroy(*cudaStream);
        *errorMsg = "Unable to create the Optix device context";
        return false;
    }

    // Log all messages, they can be filtered by level in the log callback function
    if (optixDeviceContextSetLogCallback(*ctx, logCallback, nullptr, 4) != OPTIX_SUCCESS) {
        optixDeviceContextDestroy(*ctx);
        cudaStreamDestroy(*cudaStream);
        *errorMsg = "Unable to set the Optix logging callback";
        return false;
    }

    return true;
}

bool
createOptixModule(OptixDeviceContext context,
                  const std::string& ptxPath,
                  const OptixModuleCompileOptions& moduleCompileOptions,
                  const OptixPipelineCompileOptions& pipelineCompileOptions,
                  OptixModule* module,
                  std::string* errorMsg)
{
    std::ifstream ptxFile(ptxPath);
    if (ptxFile.fail()) {
        *errorMsg = "Unable to load PTX file: " + ptxPath;
        return false;
    }

    std::string ptx = std::string((std::istreambuf_iterator<char>(ptxFile)),
                      std::istreambuf_iterator<char>());

    char logString[MAX_LOGSTRING_SIZE];
    size_t logStringSize = sizeof(logString);
    if (optixModuleCreateFromPTX(context,
                                 &moduleCompileOptions,
                                 &pipelineCompileOptions,
                                 ptx.c_str(),
                                 ptx.length(),
                                 logString,
                                 &logStringSize,
                                 module) != OPTIX_SUCCESS) {
        *errorMsg = "Unable to create Optix module.  Log: " + std::string(logString);
        return false;
    }

    return true;
}

bool
getBuiltinISModule(OptixDeviceContext context,
                   const OptixModuleCompileOptions& moduleCompileOptions,
                   const OptixPipelineCompileOptions& pipelineCompileOptions,
                   const OptixPrimitiveType primitiveType,
                   const bool motionBlur,
                   OptixModule* module,
                   std::string* errorMsg)
{
    OptixBuiltinISOptions builtinISOptions = {};
    builtinISOptions.builtinISModuleType = primitiveType;
    builtinISOptions.usesMotionBlur = motionBlur;

    if (optixBuiltinISModuleGet(context,
                                &moduleCompileOptions,
                                &pipelineCompileOptions,
                                &builtinISOptions,
                                module) != OPTIX_SUCCESS) {
        *errorMsg = "Unable to get builtin Optix module";
        return false;
    }

    return true;
}

bool
createOptixRaygenProgramGroup(OptixDeviceContext context,
                              OptixModule module,
                              const char* functionName,
                              OptixProgramGroup *pg,
                              std::string* errorMsg)
{
    OptixProgramGroupDesc pgDesc = {};
    pgDesc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    pgDesc.raygen.module = module;
    pgDesc.raygen.entryFunctionName = functionName;
    OptixProgramGroupOptions pgOptions = {}; // currently empty in Optix 7.0

    char logString[MAX_LOGSTRING_SIZE];
    size_t logStringSize = sizeof(logString);
    if (optixProgramGroupCreate(context,
                                &pgDesc,
                                1, // num groups
                                &pgOptions,
                                logString, &logStringSize,
                                pg) != OPTIX_SUCCESS) {
        *errorMsg = "Optix raygen program group creation failed: " + std::string(logString);
        return false;
    }
    return true;
}

bool
createOptixMissProgramGroup(OptixDeviceContext context,
                            OptixModule module,
                            const char* functionName,
                            OptixProgramGroup *pg,
                            std::string* errorMsg)
{
    OptixProgramGroupDesc pgDesc = {};
    pgDesc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
    pgDesc.raygen.module = module;
    pgDesc.raygen.entryFunctionName = functionName;
    OptixProgramGroupOptions pgOptions = {}; // currently empty in Optix 7.0

    char logString[MAX_LOGSTRING_SIZE];
    size_t logStringSize = sizeof(logString);
    if (optixProgramGroupCreate(context,
                                &pgDesc,
                                1, // num groups
                                &pgOptions,
                                logString, &logStringSize,
                                pg) != OPTIX_SUCCESS) {
        *errorMsg = "Optix miss program group creation failed: " + std::string(logString);
        return false;
    }
    return true;
}

bool
createOptixHitGroupProgramGroup(OptixDeviceContext context,
                                OptixModule anyHitModule,
                                const char* anyHitFunctionName,
                                OptixModule closestHitModule,
                                const char* closestHitFunctionName,
                                OptixModule intersectionModule,
                                const char* intersectionFunctionName,
                                OptixProgramGroup *pg,
                                std::string* errorMsg)
{
    OptixProgramGroupDesc pgDesc = {};
    pgDesc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    pgDesc.hitgroup.moduleCH = closestHitModule;
    pgDesc.hitgroup.entryFunctionNameCH = closestHitFunctionName;
    pgDesc.hitgroup.moduleAH = anyHitModule;
    pgDesc.hitgroup.entryFunctionNameAH = anyHitFunctionName;
    pgDesc.hitgroup.moduleIS = intersectionModule;
    pgDesc.hitgroup.entryFunctionNameIS = intersectionFunctionName;
    OptixProgramGroupOptions pgOptions = {}; // currently empty in Optix 7.0

    char logString[MAX_LOGSTRING_SIZE];
    size_t logStringSize = sizeof(logString);
    if (optixProgramGroupCreate(context,
                                &pgDesc,
                                1, // num groups
                                &pgOptions,
                                logString, &logStringSize,
                                pg) != OPTIX_SUCCESS) {
        *errorMsg = "Optix hitgroup program group creation failed: " + std::string(logString);
        return false;
    }
    return true;
}

bool
createOptixPipeline(OptixDeviceContext context,
                    const OptixPipelineCompileOptions& pipelineCompileOptions,
                    const OptixPipelineLinkOptions& pipelineLinkOptions,
                    const std::map<std::string, OptixProgramGroup>& programGroups,
                    OptixPipeline* pipeline,
                    std::string* errorMsg)
{
    std::vector<OptixProgramGroup> pgs;
    for (const auto& pgEntry : programGroups) {
        pgs.push_back(pgEntry.second);
    }

    char logString[MAX_LOGSTRING_SIZE];
    size_t logStringSize = sizeof(logString);
    if (optixPipelineCreate(context,
                            &pipelineCompileOptions,
                            &pipelineLinkOptions,
                            pgs.data(),
                            static_cast<unsigned int>(pgs.size()),
                            logString,
                            &logStringSize,
                            pipeline) != OPTIX_SUCCESS) {
        *errorMsg = "Unable to create Optix pipeline.  Log: " + std::string(logString);
        return false;
    }

    OptixStackSizes stackSizes = {};
    for (auto pg : pgs) {
        if (optixUtilAccumulateStackSizes(pg, &stackSizes) != OPTIX_SUCCESS) {
            optixPipelineDestroy(*pipeline);
            *errorMsg = "Unable to accumulate Optix stack sizes.";
            return false;
        }
    }

    unsigned int maxCCDepth = 0;
    unsigned int maxDCDepth = 0;
    unsigned int directCallableStackSizeFromTraversal;
    unsigned int directCallableStackSizeFromState;
    unsigned int continuationStackSize;
    if (optixUtilComputeStackSizes(&stackSizes,
                                   pipelineLinkOptions.maxTraceDepth,
                                   maxCCDepth,
                                   maxDCDepth,
                                   &directCallableStackSizeFromTraversal,
                                   &directCallableStackSizeFromState,
                                   &continuationStackSize) != OPTIX_SUCCESS) {
        optixPipelineDestroy(*pipeline);
        *errorMsg = "Unable to compute Optix stack sizes.";
        return false;
    }

    // Max scene graph depth (instances / motion transforms etc.)
    // NOT the max ray depth "maxTraceDepth"
    unsigned int maxTraversalGraphDepth = 8;  // 3-4 levels of instancing
    if (optixPipelineSetStackSize(*pipeline,
                                  directCallableStackSizeFromTraversal,
                                  directCallableStackSizeFromState,
                                  continuationStackSize,
                                  maxTraversalGraphDepth) != OPTIX_SUCCESS) {
        optixPipelineDestroy(*pipeline);
        *errorMsg = "Unable to set Optix stack size.";
        return false;
    }

    return true;
}

bool
createOptixAccel(OptixDeviceContext context,
                 CUstream cudaStream,
                 const OptixAccelBuildOptions& accelOptions,
                 const std::vector<OptixBuildInput>& inputs,
                 bool compact,
                 OptixGPUBuffer<char>* accelBuf,
                 OptixTraversableHandle* accelHandle,
                 std::string* errorMsg)
{
    OptixAccelBufferSizes accelBufferSizes = {};
    if (optixAccelComputeMemoryUsage(context,
                                     &accelOptions,
                                     inputs.data(),
                                     (unsigned int)inputs.size(),
                                     &accelBufferSizes) != OPTIX_SUCCESS) {
        *errorMsg = "Unable to compute Optix accel memory usage";
        return false;
    }

    OptixGPUBuffer<uint64_t> compactedSizeBuffer;
    if (compactedSizeBuffer.alloc(1) != cudaSuccess) {
        *errorMsg = "Unable to allocate Optix compacted size buffer";
        return false;
    }

    OptixAccelEmitDesc emitDesc = {};
    emitDesc.type   = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
    emitDesc.result = compactedSizeBuffer.deviceptr();

    OptixGPUBuffer<char> tempBuffer;
    if (tempBuffer.alloc(accelBufferSizes.tempSizeInBytes) != cudaSuccess) {
        *errorMsg = "Unable to allocate Optix accel temporary buffer";
        return false;
    }

    OptixGPUBuffer<char> outputBuffer;
    if (outputBuffer.alloc(accelBufferSizes.outputSizeInBytes) != cudaSuccess) {
        *errorMsg = "Unable to allocate Optix accel output buffer";
        return false;
    }

    if (optixAccelBuild(context,
                        cudaStream,
                        &accelOptions,
                        inputs.data(),
                        (unsigned int)inputs.size(),
                        tempBuffer.deviceptr(),
                        tempBuffer.sizeInBytes(),
                        outputBuffer.deviceptr(),
                        outputBuffer.sizeInBytes(),
                        accelHandle,
                        &emitDesc,
                        1) != OPTIX_SUCCESS) {
        *errorMsg = "Unable to build Optix accel";
        return false;
    }

    uint64_t compactedSize;
    compactedSizeBuffer.download(&compactedSize);

    if (compact && compactedSize < accelBufferSizes.outputSizeInBytes) {
        // We should compact the BVH
        if (accelBuf->alloc(compactedSize) != cudaSuccess) {
            // There's not enough space to have both the original BVH and compacted BVH
            // in memory at the same time, so we can't compact the BVH.
            // So, we just need to move the buffer.
            *accelBuf = std::move(outputBuffer);
            return true;
        }
        if (optixAccelCompact(context,
                              cudaStream,
                              *accelHandle,
                              accelBuf->deviceptr(),
                              accelBuf->sizeInBytes(),
                              accelHandle) != OPTIX_SUCCESS) {
            *errorMsg = "Unable to compact Optix accel";
            return false;
        }
    } else {
        // The BVH is already compact and we just need to move the buffer
        *accelBuf = std::move(outputBuffer);
    }

    return true;
}

bool
createTrianglesGAS(CUstream cudaStream,
                   OptixDeviceContext optixContext,
                   const std::vector<OptixGPUTriMesh*>& triMeshes,
                   OptixTraversableHandle* accel,
                   OptixGPUBuffer<char>* accelBuf,
                   std::string* errorMsg)
{
    std::vector<OptixBuildInput> inputs;

    bool enableMotionBlur = false;
    for (auto& tm : triMeshes) {
        OptixBuildInput input = {};  // zero everything
        input.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
        input.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
        input.triangleArray.numVertices = tm->mNumVertices;
        input.triangleArray.vertexBuffers = tm->mVerticesPtrs.data();
        input.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
        input.triangleArray.numIndexTriplets = tm->mNumFaces;
        input.triangleArray.indexBuffer = tm->mIndices.deviceptr();
        input.triangleArray.flags = &(tm->mInputFlags);
        input.triangleArray.numSbtRecords = 1;
        input.triangleArray.sbtIndexOffsetBuffer = 0;
        input.triangleArray.sbtIndexOffsetSizeInBytes = 0;
        input.triangleArray.sbtIndexOffsetStrideInBytes = 0;

        if (tm->mEnableMotionBlur) {
            enableMotionBlur = true;
        }

        inputs.push_back(input);
    }

    OptixAccelBuildOptions accelOptions = {};
    accelOptions.buildFlags             = OPTIX_BUILD_FLAG_ALLOW_COMPACTION |
                                          OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
    accelOptions.motionOptions.numKeys  = 0;
    accelOptions.operation              = OPTIX_BUILD_OPERATION_BUILD;

    if (enableMotionBlur) {
        accelOptions.motionOptions.numKeys   = 2;
        accelOptions.motionOptions.timeBegin = 0.f;
        accelOptions.motionOptions.timeEnd   = 1.f;
        accelOptions.motionOptions.flags     = OPTIX_MOTION_FLAG_NONE;
    }

    return createOptixAccel(optixContext,
                            cudaStream,
                            accelOptions,
                            inputs,
                            true,
                            accelBuf,
                            accel,
                            errorMsg);
}

bool
createRoundCurvesGAS(CUstream cudaStream,
                     OptixDeviceContext optixContext,
                     const std::vector<OptixGPURoundCurves*>& roundCurves,
                     OptixTraversableHandle* accel,
                     OptixGPUBuffer<char>* accelBuf,
                     std::string* errorMsg)
{
    std::vector<OptixBuildInput> inputs;

    bool enableMotionBlur = false;
    for (const auto& shape : roundCurves) {
        OptixBuildInput input = {};  // zero everything
        input.type = OPTIX_BUILD_INPUT_TYPE_CURVES;
        input.curveArray.curveType = shape->mType;
        input.curveArray.numPrimitives = static_cast<unsigned int>(shape->mIndices.count());
        input.curveArray.vertexBuffers = shape->mVerticesPtrs.data();
        input.curveArray.numVertices = static_cast<unsigned int>(shape->mVertices.count());
        input.curveArray.vertexStrideInBytes = sizeof(float3);
        input.curveArray.widthBuffers = shape->mWidthsPtrs.data();
        input.curveArray.widthStrideInBytes = sizeof(float);
        input.curveArray.indexBuffer = shape->mIndices.deviceptr();
        input.curveArray.indexStrideInBytes = sizeof(unsigned int);
        input.curveArray.flag = OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT;

        if (shape->mMotionSamplesCount > 1) {
            enableMotionBlur = true;
        }

        inputs.push_back(input);
    }

    OptixAccelBuildOptions accelOptions = {};
    accelOptions.buildFlags             = OPTIX_BUILD_FLAG_ALLOW_COMPACTION |
                                          OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
    accelOptions.motionOptions.numKeys  = 0;
    accelOptions.operation              = OPTIX_BUILD_OPERATION_BUILD;

    if (enableMotionBlur) {
        accelOptions.motionOptions.numKeys   = 2;
        accelOptions.motionOptions.timeBegin = 0.f;
        accelOptions.motionOptions.timeEnd   = 1.f;
        accelOptions.motionOptions.flags     = OPTIX_MOTION_FLAG_NONE;
    }

    // TODO: BVH compaction broken for curves?
    return createOptixAccel(optixContext,
                            cudaStream,
                            accelOptions,
                            inputs,
                            false,
                            accelBuf,
                            accel,
                            errorMsg);
}

bool
createCustomPrimitivesGAS(CUstream cudaStream,
                          OptixDeviceContext optixContext,
                          const std::vector<OptixGPUCustomPrimitive*>& primitives,
                          OptixTraversableHandle* accel,
                          OptixGPUBuffer<char>* accelBuf,
                          std::string* errorMsg)
{
    // Temporary data that is freed when this function returns
    struct CustomPrimitiveBuildData
    {
        unsigned int mInputFlags;
        unsigned int mNumPrimitives;
        OptixGPUBuffer<OptixAabb> mAabbs;  // per primitive
        CUdeviceptr mAabbsPtr;
    };
    std::vector<CustomPrimitiveBuildData> buildDatas;

    for (const auto& primitive : primitives) {
        std::vector<OptixAabb> aabbs;
        primitive->getPrimitiveAabbs(&aabbs);
        buildDatas.emplace_back();
        CustomPrimitiveBuildData& bd = buildDatas.back();
        bd.mInputFlags = 0;
        bd.mNumPrimitives = static_cast<unsigned int>(aabbs.size());
        if (bd.mAabbs.allocAndUpload(aabbs) != cudaSuccess) {
            *errorMsg = "Error uploading the AABBs to the GPU";
            return false;
        }
        bd.mAabbsPtr = bd.mAabbs.deviceptr();
    }

    std::vector<OptixBuildInput> inputs;

    for (auto& bd : buildDatas) {
        OptixBuildInput input = {};  // zero everything
        input.type = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
        input.customPrimitiveArray.aabbBuffers = &bd.mAabbsPtr;
        input.customPrimitiveArray.numPrimitives = bd.mNumPrimitives;
        input.customPrimitiveArray.flags = &bd.mInputFlags;
        input.customPrimitiveArray.numSbtRecords = 1;

        inputs.push_back(input);
    }

    OptixAccelBuildOptions accelOptions = {};
    accelOptions.buildFlags             = OPTIX_BUILD_FLAG_ALLOW_COMPACTION |
                                          OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
    accelOptions.motionOptions.numKeys  = 0;
    accelOptions.operation              = OPTIX_BUILD_OPERATION_BUILD;

    return createOptixAccel(optixContext,
                            cudaStream,
                            accelOptions,
                            inputs,
                            true,
                            accelBuf,
                            accel,
                            errorMsg);
}

} // namespace rt
} // namespace moonray


