// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

#include "MetalGPUUtils.h"

#include "MetalGPUBuffer.h"
#include "MetalGPUPrimitive.h"

// #undef min  // or compile error with std::min in optix_stack_size.h
// #undef max
// #include <optix_stack_size.h>

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

// These utility functions all follow the same pattern: fill in Optix struct(s),
// call optix API function, handle errors.  The functions condense Optix's
// verbose low-level API into something much easier to read in the main GPUAccelerator
// code.

#define MAX_LOGSTRING_SIZE 2048
#define metal_printf printf

namespace moonray {
namespace rt {

id<MTLDevice>
createMetalContext()
{
    id<MTLDevice> device = nil;
    int numDevices = 0;
    if (@available(macos 13.0, *)) {
        NSArray<id<MTLDevice>> *allDevices = MTLCopyAllDevices();
        for (id<MTLDevice> d in allDevices) {
            const char *device_name = [d.name UTF8String];
            if (strstr(device_name, "Apple")) {
                device = d;
                numDevices++;
            }
        }
    }

    return device;
}

bool
validateMetalContext(id<MTLDevice> device,
                     std::string* deviceName,
                     std::string* errorMsg)
{
    *errorMsg = "";

    int old_os = true;
    if (@available(macos 14.0, *)) {
        old_os = false;
    }

    if (old_os) {
        *errorMsg = "macOS is too old, must be 14.0 or greater";
        return false;
    }

    if (device == nil) {
        *errorMsg = "Only Apple Silicon GPU devices are supported";
        return false;
    }
    *deviceName = [device.name UTF8String];

    return true;
}

bool
createMetalLibrary(id<MTLDevice> device,
                  const std::string& metalLibPath,
                  id<MTLLibrary>* library,
                  std::string* errorMsg)
{
    std::ifstream metalLibFile(metalLibPath);
    if (metalLibFile.fail()) {
        *errorMsg = "Unable to load metallib file: " + metalLibPath;
        return false;
    }

    std::string lib = std::string((std::istreambuf_iterator<char>(metalLibFile)),
                      std::istreambuf_iterator<char>());

    char logString[MAX_LOGSTRING_SIZE];
    size_t logStringSize = sizeof(logString);
    
    NSURL *url = [NSURL fileURLWithPath:@(metalLibPath.c_str())];
    NSError *error = nil;

    *library = [device newLibraryWithURL:url
                                   error:&error];
    if (!*library) {
        const char *api_err = error ? [[error localizedDescription] UTF8String] : nullptr;
        *errorMsg = "Unable to load Metal Library.  Log: " + std::string(api_err?api_err:"nil");
        return false;
    }

    return true;
}

bool
createMetalRaygenKernel(id<MTLDevice> device,
                        id<MTLLibrary> library,
                        NSArray* linkedFunctions,
                        const char* functionName,
                        id<MTLComputePipelineState>* pso,
                        std::string* errorMsg)
{
    NSString *entryPointName = [@(functionName) copy];

    NSError *error = nil;
    MTLFunctionDescriptor *desc = [MTLFunctionDescriptor functionDescriptor];
    desc.name = entryPointName;
    id<MTLFunction> function = [library newFunctionWithDescriptor:desc
                                                            error:&error];

    if (function == nil) {
        NSString *api_err = [error localizedDescription];
        std::string err_str = [api_err UTF8String];
        metal_printf("Error getting function \"%s\": %s", functionName, err_str.c_str());
        return;
    }

    function.label = [entryPointName copy];
    [entryPointName release];
    error = nil;
    
    MTLPipelineOption pipelineOptions = MTLPipelineOptionNone;
    MTLComputePipelineDescriptor *computePipelineStateDescriptor =
        [[MTLComputePipelineDescriptor alloc] init];
    
    computePipelineStateDescriptor.linkedFunctions = [[MTLLinkedFunctions alloc] init];
    computePipelineStateDescriptor.linkedFunctions.functions = linkedFunctions;
    computePipelineStateDescriptor.threadGroupSizeIsMultipleOfThreadExecutionWidth = true;
    computePipelineStateDescriptor.computeFunction = function;

    *pso = [device newComputePipelineStateWithDescriptor:computePipelineStateDescriptor
                                                 options:pipelineOptions
                                              reflection:nullptr
                                                   error:&error];
    if (*pso == nil) {
        NSString *api_err = [error localizedDescription];
        std::string err_str = [api_err UTF8String];
        metal_printf("Error creating pso \"%s\": %s", functionName, err_str.c_str());
    }
    
    [computePipelineStateDescriptor release];
    computePipelineStateDescriptor = nil;
    return *pso != nil;
}

bool
createIntersectionFunctionTables(NSArray* linkedFunctions,
                                 id<MTLComputePipelineState> pso,
                                 id<MTLIntersectionFunctionTable>* intersectFuncTable,
                                 std::string* errorMsg)
{
    MTLIntersectionFunctionTableDescriptor *ift_desc =
        [[MTLIntersectionFunctionTableDescriptor alloc] init];
    
    ift_desc.functionCount = linkedFunctions.count;
    *intersectFuncTable = [pso newIntersectionFunctionTableWithDescriptor:ift_desc];
    if (*intersectFuncTable == nil) {
        metal_printf("Error creating intersection function table");
        return false;
    }
    
    // write the function handles into the table
    for (int i = 0; i < ift_desc.functionCount; i++) {
        id<MTLFunctionHandle> handle = [pso functionHandleWithFunction:linkedFunctions[i]];
        [*intersectFuncTable setFunction:handle atIndex:i];
    }

    [ift_desc release];
    ift_desc = nil;

    return true;
}

bool
createMetalAccel(id<MTLDevice> context,
                 id<MTLCommandQueue> queue,
                 std::atomic<int> &structuresBuilding,
                 MTLAccelerationStructureDescriptor* input,
                 NSString * aLabel,
                 bool compact,
                 id<MTLAccelerationStructure>* accelHandle,
                 id<MTLBuffer> tempBufferToFree,
                 std::string* errorMsg)
{
    MTLAccelerationStructureSizes accelSizes = [context
                                                accelerationStructureSizesWithDescriptor:input];
    id<MTLAccelerationStructure> accel_uncompressed = [context
                                                       newAccelerationStructureWithSize:accelSizes.accelerationStructureSize];
    [accel_uncompressed setLabel:[NSString stringWithFormat:@"%@ - Uncompressed", aLabel]];
    id<MTLBuffer> scratchBuf = [context newBufferWithLength:accelSizes.buildScratchBufferSize
                                                    options:MTLResourceStorageModePrivate];
    [scratchBuf setLabel:[NSString stringWithFormat:@"%@ - Scratch Buffer", aLabel]];
    id<MTLBuffer> sizeBuf = [context newBufferWithLength:8 options:MTLResourceStorageModeShared];
    [sizeBuf setLabel:[NSString stringWithFormat:@"%@ - Size Buffer", aLabel]];
    id<MTLCommandBuffer> accelCommands = [queue commandBuffer];
    id<MTLAccelerationStructureCommandEncoder> accelEnc =
        [accelCommands accelerationStructureCommandEncoder];
    [accelEnc buildAccelerationStructure:accel_uncompressed
                              descriptor:input
                           scratchBuffer:scratchBuf
                     scratchBufferOffset:0];
    [accelEnc writeCompactedAccelerationStructureSize:accel_uncompressed
                                             toBuffer:sizeBuf
                                               offset:0
                                         sizeDataType:MTLDataTypeULong];
    [accelEnc endEncoding];
    [accelCommands addCompletedHandler:^(id<MTLCommandBuffer> command_buffer) {
        /* free temp resources */
        [scratchBuf release];
        
        /* Compact the accel structure */
        uint64_t compressed_size = *(uint64_t*)sizeBuf.contents;
        [sizeBuf release];
        
        dispatch_async(dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0), ^{
            id<MTLCommandBuffer> accelCommands = [queue commandBuffer];
            id<MTLAccelerationStructureCommandEncoder> accelEnc =
                [accelCommands accelerationStructureCommandEncoder];
            id<MTLAccelerationStructure> accel = [context
                                                  newAccelerationStructureWithSize:compressed_size];
            [accel setLabel:[NSString stringWithFormat:@"%@ - Compressed", aLabel]];
            [accelEnc copyAndCompactAccelerationStructure:accel_uncompressed
                                  toAccelerationStructure:accel];
            [accelEnc endEncoding];

            [accelCommands addCompletedHandler:^(id<MTLCommandBuffer> command_buffer) {
                uint64_t allocated_size = [accel allocatedSize];
                *accelHandle = accel;
                [accel_uncompressed release];
                structuresBuilding--;
            }];
            [accelCommands commit];
        });
    }];
    
    structuresBuilding++;
    [accelCommands commit];

    return true;

}

bool
createTrianglesGAS(id<MTLDevice> context,
                   id<MTLCommandQueue> queue,
                   std::atomic<int> &structuresBuilding,
                   const std::vector<MetalGPUTriMesh*>& triMeshes,
                   std::vector<id<MTLAccelerationStructure>>& accels,
                   std::string* errorMsg)
{
    bool enableMotionBlur = false;
    for (auto& tm : triMeshes) {
        if (tm->mEnableMotionBlur) {
            enableMotionBlur = true;
            break;
        }
    }

    accels.reserve(triMeshes.size());
    int accelIndex = 0;
    for (auto& tm : triMeshes) {
        
        scene_rdl2::logging::Logger::info("GPU: Creating AccelStruct for GPUMesh: ", tm->mName.c_str());
        MTLAccelerationStructureGeometryDescriptor *geomDesc;
        
        if (enableMotionBlur) {
            std::vector<MTLMotionKeyframeData*> verticesPtrs;
            verticesPtrs.reserve(2);
            for (int i = 0; i < 2; i++) {
                MTLMotionKeyframeData *keyFrame = [MTLMotionKeyframeData data];
                keyFrame.buffer = tm->mVertices.deviceptr();
                keyFrame.offset = tm->mVerticesPtrs[i];
                verticesPtrs.push_back(keyFrame);
            }
            
            MTLAccelerationStructureMotionTriangleGeometryDescriptor *geomDescMotion =
                [MTLAccelerationStructureMotionTriangleGeometryDescriptor descriptor];
            geomDescMotion.vertexBuffers = [NSArray arrayWithObjects:verticesPtrs.data()
                                                               count:verticesPtrs.size()];
            geomDescMotion.vertexStride = sizeof(*tm->mVertices.ptr());
            geomDescMotion.indexBuffer = tm->mIndices.deviceptr();
            geomDescMotion.indexBufferOffset = 0;
            geomDescMotion.indexType = MTLIndexTypeUInt32;
            geomDescMotion.triangleCount = tm->mNumFaces;
            geomDescMotion.intersectionFunctionTableOffset = 0;
            
            geomDesc = geomDescMotion;
        }
        else {
            MTLAccelerationStructureTriangleGeometryDescriptor *geomDescNoMotion =
                [MTLAccelerationStructureTriangleGeometryDescriptor descriptor];
            
            geomDescNoMotion.vertexBuffer = tm->mVertices.deviceptr();
            geomDescNoMotion.vertexBufferOffset = 0;
            geomDescNoMotion.vertexStride = sizeof(*tm->mVertices.ptr());
            geomDescNoMotion.indexBuffer = tm->mIndices.deviceptr();
            geomDescNoMotion.indexBufferOffset = 0;
            geomDescNoMotion.indexType = MTLIndexTypeUInt32;
            geomDescNoMotion.triangleCount = tm->mNumFaces;
            geomDescNoMotion.intersectionFunctionTableOffset = 0;
            
            geomDesc = geomDescNoMotion;
        }
        MTLPrimitiveAccelerationStructureDescriptor *accelDesc =
            [MTLPrimitiveAccelerationStructureDescriptor descriptor];
        accelDesc.geometryDescriptors = @[geomDesc];

        if (enableMotionBlur) {
            accelDesc.motionStartTime = 0.0f;
            accelDesc.motionEndTime = 1.0f;
            accelDesc.motionStartBorderMode = MTLMotionBorderModeClamp;
            accelDesc.motionEndBorderMode = MTLMotionBorderModeClamp;
            accelDesc.motionKeyframeCount = 2;
        }
        accels.push_back(nil);
        if (!createMetalAccel(context,
                              queue,
                              structuresBuilding,
                              accelDesc,
                              @"GPUMesh: Accel Struct",
                              true,
                              &accels[accelIndex++],
                              nil,
                              errorMsg)) {
            return false;
        }
    }

    return true;

}

bool
createRoundCurvesGAS(id<MTLDevice> context,
                     id<MTLCommandQueue> queue,
                     std::atomic<int> &structuresBuilding,
                     const std::vector<MetalGPURoundCurves*>& roundCurves,
                     std::vector<id<MTLAccelerationStructure>>& accels,
                     std::string* errorMsg)
{
    bool enableMotionBlur = false;

    accels.reserve(roundCurves.size());
    int accelIndex = 0;

    for (const auto& shape : roundCurves) {
        
        scene_rdl2::logging::Logger::info("GPU: Creating AccelStruct for GPUCurve: ", shape->mName.c_str());
        
        MTLAccelerationStructureGeometryDescriptor *geomDesc;

        MTLAccelerationStructureCurveGeometryDescriptor  *geomDescNoMotion =
                [MTLAccelerationStructureCurveGeometryDescriptor descriptor];

        geomDescNoMotion.curveType = shape->mSubType;
        geomDescNoMotion.curveBasis = shape->mType;

        geomDescNoMotion.controlPointBuffer = shape->mVertices.deviceptr();
        geomDescNoMotion.controlPointCount = shape->mNumControlPoints;
        geomDescNoMotion.controlPointStride = sizeof(float3);
        geomDescNoMotion.controlPointFormat = MTLAttributeFormatFloat3;
        geomDescNoMotion.controlPointBufferOffset = 0;

        geomDescNoMotion.segmentCount = static_cast<unsigned int>(shape->mIndices.count());
        geomDescNoMotion.segmentControlPointCount = 4;

        geomDescNoMotion.indexBuffer = shape->mIndices.deviceptr();
        geomDescNoMotion.indexType = MTLIndexTypeUInt32;
        geomDescNoMotion.indexBufferOffset = 0;
        
        geomDescNoMotion.radiusBuffer = shape->mWidths.deviceptr();
        geomDescNoMotion.radiusStride = sizeof(*shape->mWidths.ptr());

        geomDesc = geomDescNoMotion;

        MTLPrimitiveAccelerationStructureDescriptor *accelDesc =
            [MTLPrimitiveAccelerationStructureDescriptor descriptor];
        accelDesc.geometryDescriptors = @[geomDesc];

        accels.push_back(nil);
        
        if (!createMetalAccel(context,
                              queue,
                              structuresBuilding,
                              accelDesc,
                              @"GPUCurve: Accel Struct",
                              true,
                              &accels[accelIndex++],
                              nil,
                              errorMsg)) {
            return false;
        }

    }




    return true;
}


bool
createCustomPrimitivesGAS(id<MTLDevice> context,
                          id<MTLCommandQueue> queue,
                          std::atomic<int> &structuresBuilding,
                          const std::vector<MetalGPUCustomPrimitive*>& primitives,
                          std::vector<id<MTLAccelerationStructure>>& accels,
                          std::string* errorMsg)
{
    accels.reserve(primitives.size());
    int accelIndex = 0;

    std::vector<std::vector<OptixAabb>> aabbs;
    aabbs.reserve(primitives.size());
    int numAabbs = 0;
    for (const auto& primitive : primitives) {
        std::vector<OptixAabb> prim_aabbs;
        primitive->getPrimitiveAabbs(&prim_aabbs);
        numAabbs += prim_aabbs.size();
        aabbs.push_back(prim_aabbs);
    }
    
    /* GPU buffer */
    id<MTLBuffer> aabbsBuf = [context
        newBufferWithLength:numAabbs * sizeof(MTLAxisAlignedBoundingBox)
                    options:MTLResourceStorageModeShared];
    [aabbsBuf setLabel:@"Axis Aligned Bounding Box Buffer"];

    // OptixAabb should be the same size and layout as MTLAxisAlignedBoundingBox
    static_assert(sizeof(MTLAxisAlignedBoundingBox) == sizeof(OptixAabb));
    OptixAabb *aabb_data = (OptixAabb*)[aabbsBuf contents];

    MTLAccelerationStructureBoundingBoxGeometryDescriptor *geomDesc =
        [MTLAccelerationStructureBoundingBoxGeometryDescriptor descriptor];
        
    int aabbIndex = 0;
    size_t bufferOffset = 0;
    for (const auto& primitive : primitives) {
        geomDesc.boundingBoxBuffer = aabbsBuf;
        geomDesc.boundingBoxBufferOffset = bufferOffset;
        geomDesc.boundingBoxCount = aabbs[aabbIndex].size();
        geomDesc.boundingBoxStride = sizeof(OptixAabb);
        geomDesc.intersectionFunctionTableOffset = primitive->getFuncTableOffset();

        memcpy(aabb_data + aabbIndex, aabbs[aabbIndex].data(),
               geomDesc.boundingBoxCount * geomDesc.boundingBoxStride);
        
        MTLPrimitiveAccelerationStructureDescriptor *accelDesc =
            [MTLPrimitiveAccelerationStructureDescriptor descriptor];
        accelDesc.geometryDescriptors = @[geomDesc];

        accels.push_back(nil);
        if (!createMetalAccel(context,
                              queue,
                              structuresBuilding,
                              accelDesc,
                              @"CustomPrim: Accel Struct",
                              true,
                              &accels[accelIndex++],
                              aabbsBuf,
                              errorMsg)) {
            return false;
        }

        bufferOffset += geomDesc.boundingBoxCount * geomDesc.boundingBoxStride;
        aabbIndex++;
    }
    
    return true;
}

} // namespace rt
} // namespace moonray


