// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

#include "MetalGPUPrimitiveGroup.h"
#include "MetalGPUUtils.h"

namespace moonray {
namespace rt {

MetalGPUPrimitiveGroup::~MetalGPUPrimitiveGroup()
{
    for (auto& prim : mTriMeshes) {
        delete prim;
    }
    for (auto& prim : mTriMeshesMB) {
        delete prim;
    }
    for (auto& prim : mRoundCurves) {
        delete prim;
    }
    for (auto& prim : mRoundCurvesMB) {
        delete prim;
    }
    for (auto& prim : mCustomPrimitives) {
        delete prim;
    }
    for (auto& prim : mInstances) {
        delete prim;
    }
}

void
MetalGPUPrimitiveGroup::setAccelStructBaseID(unsigned int& baseID)
{
    mAccelStructBaseID = baseID;

    baseID += mTriMeshes.size() + mTriMeshesMB.size() +
              mRoundCurves.size() + mRoundCurvesMB.size() +
              mCustomPrimitives.size();
}

void
MetalGPUPrimitiveGroup::hasMotionBlur(uint32_t *flags)
{
    if (mTriMeshesMB.size() || mRoundCurvesMB.size()) {
        *flags |= GPU_MOTION_PRIMITIVES;
    }
    for (auto& prim : mCustomPrimitives) {
        prim->hasMotionBlur(flags);
    }
    for (auto& inst : mInstances) {
        if (inst->mHasMotionBlur) {
            *flags |= GPU_MOTION_INSTANCES;
        }
    }
}

bool
MetalGPUPrimitiveGroup::build(id<MTLDevice> device,
                         id<MTLCommandQueue> queue,
                         uint32_t motionBlurFlags,
                         std::vector<id<MTLAccelerationStructure>>* bottomLevelAS,
                         std::string* errorMsg)
{
    
    if (mIsBuilt) {
        return true;
    }
    mIsBuilt = true;

    std::atomic<int> structuresBuilding(0);
    int userID = mAccelStructBaseID;

    mUserInstanceIDs.clear();

    if (!mTriMeshes.empty()) {
        if (!createTrianglesGAS(device,
                                queue,
                                structuresBuilding,
                                mTriMeshes,
                                mTrianglesGAS,
                                errorMsg)) {
            return false;
        }
    }

    if (!mTriMeshesMB.empty()) {
        if (!createTrianglesGAS(device,
                                queue,
                                structuresBuilding,
                                mTriMeshesMB,
                                mTrianglesMBGAS,
                                errorMsg)) {
            return false;
        }
    }

    if (!mRoundCurves.empty()) {
        if (!createRoundCurvesGAS(device,
                                  queue,
                                  structuresBuilding,
                                  mRoundCurves,
                                  mRoundCurvesGAS,
                                  errorMsg)) {
            return false;
        }
    }

    if (!mRoundCurvesMB.empty()) {
        if (!createRoundCurvesGAS(device,
                                  queue,
                                  structuresBuilding,
                                  mRoundCurvesMB,
                                  mRoundCurvesMBGAS,
                                  errorMsg)) {
            return false;
        }
    }

    if (!mCustomPrimitives.empty()) {
        if (!createCustomPrimitivesGAS(device,
                                       queue,
                                       structuresBuilding,
                                       mCustomPrimitives,
                                       mCustomPrimitivesGAS,
                                       errorMsg)) {
            return false;
        }
    }

    if (!mInstances.empty()) {
        for (size_t i = 0; i < mInstances.size(); i++) {
            if (!mInstances[i]->build(device, queue, bottomLevelAS, structuresBuilding, errorMsg)) {
                return false;
            }
        }
    }
    
    // Wait for all the BLAS to complete building
    while(structuresBuilding != 0) {
        usleep(1);
    }

    if (!mCustomPrimitives.empty()) {
        for (auto& prim : mCustomPrimitives) {
            prim->freeHostMemory();
        }
    }
    
    int meshIdx = 0;
    for (auto& triMesh : mTriMeshes) {
        AccelData instance(mTrianglesGAS[meshIdx++], userID++);
        if (!triMesh->mIsSingleSided) {
            instance.options |= MTLAccelerationStructureInstanceOptionDisableTriangleCulling;
        }
        if (!triMesh->mShadowLinkLights.count()) {
            instance.options |= MTLAccelerationStructureInstanceOptionOpaque;
        }
        if (triMesh->mVisibleShadow) {
            mUserInstanceIDs.push_back(instance);
        }
    }
    
    int meshMBIdx = 0;
    for (auto& triMesh : mTriMeshesMB) {
        AccelData instance(mTrianglesMBGAS[meshMBIdx++], userID++);
        if (!triMesh->mIsSingleSided) {
            instance.options |= MTLAccelerationStructureInstanceOptionDisableTriangleCulling;
        }
        if (!triMesh->mShadowLinkLights.count()) {
            instance.options |= MTLAccelerationStructureInstanceOptionOpaque;
        }
        if (triMesh->mVisibleShadow) {
            mUserInstanceIDs.push_back(instance);
        }
    }
    
    int shapeIdx = 0;
    for (auto& shape : mRoundCurves) {
        AccelData instance(mRoundCurvesGAS[shapeIdx++], userID++);
        if (!shape->mIsSingleSided) {
            instance.options |= MTLAccelerationStructureInstanceOptionDisableTriangleCulling;
        }
        if (!shape->mShadowLinkLights.count()) {
            instance.options |= MTLAccelerationStructureInstanceOptionOpaque;
        }
        if (shape->mVisibleShadow) {
            mUserInstanceIDs.push_back(instance);
        }
    }
    
    int shapeMBIdx = 0;
    for (auto& shape : mRoundCurvesMB) {
        AccelData instance(mTrianglesMBGAS[shapeMBIdx++], userID++);
        if (!shape->mIsSingleSided) {
            instance.options |= MTLAccelerationStructureInstanceOptionDisableTriangleCulling;
        }
        if (!shape->mShadowLinkLights.count()) {
            instance.options |= MTLAccelerationStructureInstanceOptionOpaque;
        }
        if (shape->mVisibleShadow) {
            mUserInstanceIDs.push_back(instance);
        }
    }

    int primIdx = 0;
    for (auto& prim : mCustomPrimitives) {
        AccelData instance(mCustomPrimitivesGAS[primIdx++], userID++);
        if (!prim->mIsSingleSided) {
            instance.options |= MTLAccelerationStructureInstanceOptionDisableTriangleCulling;
        }
        if (!prim->mShadowLinkLights.count()) {
            instance.options |= MTLAccelerationStructureInstanceOptionOpaque;
        }
        if (prim->mVisibleShadow) {
            mUserInstanceIDs.push_back(instance);
        }
    }

    int numTotalMotionKeys = mUserInstanceIDs.size();
    for (size_t i = 0; i < mInstances.size(); i++) {
        if (mInstances[i]->mGroup->mTopLevelIAS) {
            AccelData instance(mInstances[i]->mGroup->mTopLevelIAS, userID++);
            mInstances[i]->mXforms[0].toOptixTransform((float*)&instance.xForm);
            mUserInstanceIDs.push_back(instance);
        } else {
            size_t head = mUserInstanceIDs.size();
            
            mUserInstanceIDs.insert(mUserInstanceIDs.end(),
                                    mInstances[i]->mGroup->mUserInstanceIDs.begin(),
                                    mInstances[i]->mGroup->mUserInstanceIDs.end());
            if (!mInstances[i]->mHasMotionBlur) {
                // The instance's xform is specified on the Metal Instance and the child
                // node is the top level IAS node in the referenced group.
                for(; head < mUserInstanceIDs.size(); head++) {
                    mInstances[i]->mXforms[0].toOptixTransform((float*)&mUserInstanceIDs[head].xForm);
                    numTotalMotionKeys++;
                }
            }
            else {
                for(; head < mUserInstanceIDs.size(); head++) {
                    mUserInstanceIDs[head].motionKeys = mInstances[i]->mXforms;
                    mUserInstanceIDs[head].numMotionKeys = mInstances[i]->sNumMotionKeys;
                    numTotalMotionKeys+=mInstances[i]->sNumMotionKeys;
                }
            }
        }
    }

    if (bottomLevelAS && mUserInstanceIDs.size()) {
        
        size_t head = bottomLevelAS->size();
        
        size_t instance_size;
        if (motionBlurFlags & GPU_MOTION_INSTANCES) {
            instance_size = sizeof(MTLAccelerationStructureMotionInstanceDescriptor);
        }
        else {
            instance_size = sizeof(MTLAccelerationStructureUserIDInstanceDescriptor);
        }

        id<MTLBuffer> instanceBuf = [device newBufferWithLength:mUserInstanceIDs.size() * instance_size
                                                        options:MTLResourceStorageModeShared];
        [instanceBuf setLabel:@"Instance Buffer (Shared)"];
        
        id<MTLBuffer> motionKeysBuf = nil;
        MTLPackedFloat4x3 *motionKeys = nullptr;
        if (motionBlurFlags & GPU_MOTION_INSTANCES) {
            motionKeysBuf = [device
                newBufferWithLength:numTotalMotionKeys * sizeof(MTLPackedFloat4x3)
                             options:MTLResourceStorageModeShared];
            [motionKeysBuf setLabel:@"Motion Keys Buffer (Shared)"];
            motionKeys = (MTLPackedFloat4x3*)motionKeysBuf.contents;
        }
        
        int accelIndex = 0;
        uint32_t motionKeyIndex = 0;
        
        bottomLevelAS->reserve(mUserInstanceIDs.size());
        for (auto& inst : mUserInstanceIDs) {
            int currIndex = accelIndex++;
            bottomLevelAS->push_back(inst.as);
            
            /* Bake into the appropriate descriptor */
            if (motionBlurFlags & GPU_MOTION_INSTANCES) {
                MTLAccelerationStructureMotionInstanceDescriptor *instances =
                    (MTLAccelerationStructureMotionInstanceDescriptor *)[instanceBuf contents];
                MTLAccelerationStructureMotionInstanceDescriptor &desc = instances[currIndex];

                desc.accelerationStructureIndex = currIndex;
                desc.userID = inst.userID;
                desc.mask = 255;
                desc.options = inst.options;
                desc.motionStartTime = 0.0f;
                desc.motionEndTime = 1.0f;
                desc.motionTransformsStartIndex = motionKeyIndex;
                desc.motionStartBorderMode = MTLMotionBorderModeClamp;
                desc.motionEndBorderMode = MTLMotionBorderModeClamp;
                desc.intersectionFunctionTableOffset = 0;
                desc.motionTransformsCount = inst.numMotionKeys;
                
                if (desc.motionTransformsCount == 1) {
                    motionKeys[motionKeyIndex] = inst.xForm;
                }
                else {
                    for (int i = 0; i < inst.numMotionKeys; i++) {
                        inst.motionKeys[i].toOptixTransform((float*)&motionKeys[motionKeyIndex + i]);
                    }
                }
                motionKeyIndex += desc.motionTransformsCount;
            }
            else {
                MTLAccelerationStructureUserIDInstanceDescriptor *instance =
                    (MTLAccelerationStructureUserIDInstanceDescriptor *)[instanceBuf contents];
                MTLAccelerationStructureUserIDInstanceDescriptor &desc = instance[currIndex];
                
                desc.accelerationStructureIndex = currIndex;
                desc.userID = inst.userID;
                desc.mask = 255;
                desc.intersectionFunctionTableOffset = 0;
                desc.options = inst.options;
                desc.transformationMatrix = inst.xForm;

            }
        }
        
        NSMutableArray *blas = [[NSMutableArray alloc] init];
        
        for(; head < bottomLevelAS->size(); head++) {
            [blas addObject:bottomLevelAS->at(head)];
        }
        
        MTLInstanceAccelerationStructureDescriptor *accelDesc =
            [MTLInstanceAccelerationStructureDescriptor descriptor];

        accelDesc.instanceCount = mUserInstanceIDs.size();
        accelDesc.instanceDescriptorType = MTLAccelerationStructureInstanceDescriptorTypeUserID;
        accelDesc.instanceDescriptorBuffer = instanceBuf;
        accelDesc.instanceDescriptorBufferOffset = 0;
        accelDesc.instanceDescriptorStride = instance_size;
        accelDesc.instancedAccelerationStructures = blas;
        
        if (motionBlurFlags & GPU_MOTION_INSTANCES) {
            accelDesc.instanceDescriptorType = MTLAccelerationStructureInstanceDescriptorTypeMotion;
            accelDesc.motionTransformBuffer = motionKeysBuf;
            accelDesc.motionTransformCount = numTotalMotionKeys;
        }

        if (!createMetalAccel( device,
                            queue,
                            structuresBuilding,
                            accelDesc,
                            @"PrimGroup: Accel Struct",
                            true,
                            &mTopLevelIAS,
                            instanceBuf,
                            errorMsg)) {
            return false;
        }
            
        // Wait for all the BLAS to complete building
        while(structuresBuilding != 0) {
            usleep(1);
        }

        [instanceBuf release];
        
        if (motionKeysBuf) {
            [motionKeysBuf release];
        }

    }

    return true;
}

void
MetalGPUPrimitiveGroup::getSBTRecords(std::vector<HitGroupRecord>& hitgroupRecs,
                                 std::vector<id<MTLBuffer>>& usedResources)
{
    // We need to create one "HitGroup" object a.k.a. record in the Shader
    // Binding Table for each GPUPrimitive.  It is important that these HitGroup
    // records appear in memory in exactly the same order as they were added.
    // The HitGroup records tell Optix what kind of geometry is contained in the
    // BVH node, its properties, and what programs to call to perform ray-object
    // intersections.
    // HitGroup records have two parts.  1) A header that tells Optix what program
    // group to use and 2) custom data for the primitive such as properties, vertices, etc.
    // The properties differ between primitive types but we can only have one
    // HitGroup struct type, and all HitGroup records must be the same size,
    // so the differing properties are efficiently packed via an anonymous union.
    // See GPUSBTRecord.h.
    
    int userInstanceID = mAccelStructBaseID;

    for (size_t i = 0; i < mTriMeshes.size(); i++) {
        HitGroupRecord rec = {};
        MetalGPUTriMesh* triMesh = mTriMeshes[i];
        rec.mData.mIsSingleSided = triMesh->mIsSingleSided;
        rec.mData.mIsNormalReversed = triMesh->mIsNormalReversed;
        rec.mData.mVisibleShadow = triMesh->mVisibleShadow;
        rec.mData.mAssignmentIds = triMesh->mAssignmentIds.ptr();
        rec.mData.mNumShadowLinkLights = triMesh->mShadowLinkLights.count();
        rec.mData.mShadowLinkLights = triMesh->mShadowLinkLights.ptr();
        rec.mData.mNumShadowLinkReceivers = triMesh->mShadowLinkReceivers.count();
        rec.mData.mShadowLinkReceivers = triMesh->mShadowLinkReceivers.ptr();
        rec.mData.mUserInstanceID = userInstanceID++;

        usedResources.push_back(triMesh->mAssignmentIds.deviceptr());
        usedResources.push_back(triMesh->mShadowLinkReceivers.deviceptr());
        usedResources.push_back(triMesh->mShadowLinkLights.deviceptr());

        hitgroupRecs.push_back(rec);
    }

    for (size_t i = 0; i < mTriMeshesMB.size(); i++) {
        HitGroupRecord rec = {};
        MetalGPUTriMesh* triMesh = mTriMeshesMB[i];
        rec.mData.mIsSingleSided = triMesh->mIsSingleSided;
        rec.mData.mIsNormalReversed = triMesh->mIsNormalReversed;
        rec.mData.mVisibleShadow = triMesh->mVisibleShadow;
        rec.mData.mAssignmentIds = triMesh->mAssignmentIds.ptr();
        rec.mData.mNumShadowLinkLights = triMesh->mShadowLinkLights.count();
        rec.mData.mShadowLinkLights = triMesh->mShadowLinkLights.ptr();
        rec.mData.mNumShadowLinkReceivers = triMesh->mShadowLinkReceivers.count();
        rec.mData.mShadowLinkReceivers = triMesh->mShadowLinkReceivers.ptr();
        rec.mData.mUserInstanceID = userInstanceID++;

        usedResources.push_back(triMesh->mAssignmentIds.deviceptr());
        usedResources.push_back(triMesh->mShadowLinkReceivers.deviceptr());
        usedResources.push_back(triMesh->mShadowLinkLights.deviceptr());

        hitgroupRecs.push_back(rec);
    }

    for (size_t i = 0; i < mRoundCurves.size(); i++) {
        HitGroupRecord rec = {};
        MetalGPURoundCurves* curves = mRoundCurves[i];
        rec.mData.mIsSingleSided = curves->mIsSingleSided;
        rec.mData.mIsNormalReversed = curves->mIsNormalReversed;
        rec.mData.mVisibleShadow = curves->mVisibleShadow;
        rec.mData.mAssignmentIds = curves->mAssignmentIds.ptr();
        rec.mData.mNumShadowLinkLights = curves->mShadowLinkLights.count();
        rec.mData.mShadowLinkLights = curves->mShadowLinkLights.ptr();
        rec.mData.mNumShadowLinkReceivers = curves->mShadowLinkReceivers.count();
        rec.mData.mShadowLinkReceivers = curves->mShadowLinkReceivers.ptr();
        rec.mData.mUserInstanceID = userInstanceID++;

        usedResources.push_back(curves->mAssignmentIds.deviceptr());
        usedResources.push_back(curves->mShadowLinkReceivers.deviceptr());
        usedResources.push_back(curves->mShadowLinkLights.deviceptr());

        hitgroupRecs.push_back(rec);
    }

    for (size_t i = 0; i < mRoundCurvesMB.size(); i++) {
        HitGroupRecord rec = {};
        MetalGPURoundCurves* curves = mRoundCurvesMB[i];
        rec.mData.mIsSingleSided = curves->mIsSingleSided;
        rec.mData.mIsNormalReversed = curves->mIsNormalReversed;
        rec.mData.mVisibleShadow = curves->mVisibleShadow;
        rec.mData.mAssignmentIds = curves->mAssignmentIds.ptr();
        rec.mData.mNumShadowLinkLights = curves->mShadowLinkLights.count();
        rec.mData.mShadowLinkLights = curves->mShadowLinkLights.ptr();
        rec.mData.mNumShadowLinkReceivers = curves->mShadowLinkReceivers.count();
        rec.mData.mShadowLinkReceivers = curves->mShadowLinkReceivers.ptr();

        rec.mData.mUserInstanceID = userInstanceID++;

        usedResources.push_back(curves->mAssignmentIds.deviceptr());
        usedResources.push_back(curves->mShadowLinkReceivers.deviceptr());
        usedResources.push_back(curves->mShadowLinkLights.deviceptr());

        hitgroupRecs.push_back(rec);
    }

    for (size_t i = 0; i < mCustomPrimitives.size(); i++) {
        HitGroupRecord rec = {};

        // Fill in the common properties
        MetalGPUCustomPrimitive* prim = mCustomPrimitives[i];
        rec.mData.mIsSingleSided = prim->mIsSingleSided;
        rec.mData.mIsNormalReversed = prim->mIsNormalReversed;
        rec.mData.mVisibleShadow = prim->mVisibleShadow;
        rec.mData.mAssignmentIds = prim->mAssignmentIds.ptr();
        rec.mData.mNumShadowLinkLights = prim->mShadowLinkLights.count();
        rec.mData.mShadowLinkLights = prim->mShadowLinkLights.ptr();
        rec.mData.mNumShadowLinkReceivers = prim->mShadowLinkReceivers.count();
        rec.mData.mShadowLinkReceivers = prim->mShadowLinkReceivers.ptr();
        rec.mData.mUserInstanceID = userInstanceID++;

        usedResources.push_back(prim->mAssignmentIds.deviceptr());
        usedResources.push_back(prim->mShadowLinkReceivers.deviceptr());
        usedResources.push_back(prim->mShadowLinkLights.deviceptr());

        // Fill in the primitive type-specific properties
        MetalGPUCurve* curve = dynamic_cast<MetalGPUCurve*>(prim);
        if (curve) {
            rec.mData.curve.mMotionSamplesCount = curve->mMotionSamplesCount;
            rec.mData.curve.mSegmentsPerCurve = curve->mSegmentsPerCurve;
            rec.mData.curve.mIndices = curve->mIndices.ptr();
            rec.mData.curve.mNumControlPoints = curve->mNumControlPoints;
            rec.mData.curve.mControlPoints = curve->mControlPoints.ptr();
            
            usedResources.push_back(curve->mIndices.deviceptr());
            usedResources.push_back(curve->mControlPoints.deviceptr());

        }
        MetalGPUPoints* points = dynamic_cast<MetalGPUPoints*>(prim);
        if (points) {
            rec.mData.points.mMotionSamplesCount = points->mMotionSamplesCount;
            rec.mData.points.mPoints = points->mPoints.ptr();
            
            usedResources.push_back(points->mPoints.deviceptr());

        }
        MetalGPUSphere* sphere = dynamic_cast<MetalGPUSphere*>(prim);
        if (sphere) {
            rec.mData.sphere.mL2P = sphere->mL2P;
            rec.mData.sphere.mP2L = sphere->mP2L;
            rec.mData.sphere.mRadius = sphere->mRadius;
            rec.mData.sphere.mPhiMax = sphere->mPhiMax;
            rec.mData.sphere.mZMin = sphere->mZMin;
            rec.mData.sphere.mZMax = sphere->mZMax;
        }
        MetalGPUBox* box = dynamic_cast<MetalGPUBox*>(prim);
        if (box) {
            rec.mData.box.mL2P = box->mL2P;
            rec.mData.box.mP2L = box->mP2L;
            rec.mData.box.mLength = box->mLength;
            rec.mData.box.mHeight = box->mHeight;
            rec.mData.box.mWidth = box->mWidth;
        }

        hitgroupRecs.push_back(rec);
    }
}

} // namespace rt
} // namespace moonray
