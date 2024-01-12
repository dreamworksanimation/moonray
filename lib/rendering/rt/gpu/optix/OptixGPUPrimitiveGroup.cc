// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

#include "OptixGPUPrimitiveGroup.h"
#include "OptixGPUUtils.h"

namespace moonray {
namespace rt {

OptixGPUPrimitiveGroup::~OptixGPUPrimitiveGroup()
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
OptixGPUPrimitiveGroup::setSBTOffset(unsigned int& sbtOffset)
{
    mSBTOffset = sbtOffset;
    sbtOffset += mTriMeshes.size() + mTriMeshesMB.size() + 
                 mRoundCurves.size() + mRoundCurvesMB.size() +
                 mCustomPrimitives.size();
}

bool
OptixGPUPrimitiveGroup::build(CUstream cudaStream,
                         OptixDeviceContext context,
                         std::string* errorMsg)
{
    if (mIsBuilt) {
        return true;
    }
    mIsBuilt = true;

    std::vector<OptixInstance> instances;

    if (!mTriMeshes.empty()) {
        // Create the GAS (geometry acceleration structure) for all of the triangle
        // meshes in the scene and wrap them in an OptixInstance.
        if (!createTrianglesGAS(cudaStream,
                                context,
                                mTriMeshes,
                                &mTrianglesGAS,
                                &mTrianglesGASBuf,
                                errorMsg)) {
            return false;
        }
        OptixInstance oinstance;
        OptixGPUXform::identityXform().toOptixTransform(oinstance.transform);
        oinstance.instanceId = 0;
        oinstance.visibilityMask = 255;
        oinstance.sbtOffset = mSBTOffset;
        oinstance.flags = 0;
        oinstance.traversableHandle = mTrianglesGAS;
        instances.push_back(oinstance);
    }

    if (!mTriMeshesMB.empty()) {
        // Create the GAS (geometry acceleration structure) for all of the triangle
        // meshes in the scene and wrap them in an OptixInstance.
        if (!createTrianglesGAS(cudaStream,
                                context,
                                mTriMeshesMB,
                                &mTrianglesMBGAS,
                                &mTrianglesMBGASBuf,
                                errorMsg)) {
            return false;
        }
        OptixInstance oinstance;
        OptixGPUXform::identityXform().toOptixTransform(oinstance.transform);
        oinstance.instanceId = 0;
        oinstance.visibilityMask = 255;
        oinstance.sbtOffset = mSBTOffset + mTriMeshes.size();
        oinstance.flags = 0;
        oinstance.traversableHandle = mTrianglesMBGAS;
        instances.push_back(oinstance);
    }

    if (!mRoundCurves.empty()) {
        // Create the GAS (geometry acceleration structure) for all of the round
        // curves in the scene and wrap them in an OptixInstance.
        if (!createRoundCurvesGAS(cudaStream,
                                  context,
                                  mRoundCurves,
                                  &mRoundCurvesGAS,
                                  &mRoundCurvesGASBuf,
                                  errorMsg)) {
            return false;
        }
        OptixInstance oinstance;
        OptixGPUXform::identityXform().toOptixTransform(oinstance.transform);
        oinstance.instanceId = 0;
        oinstance.visibilityMask = 255;
        oinstance.sbtOffset = mSBTOffset + mTriMeshes.size() + mTriMeshesMB.size();
        oinstance.flags = 0;
        oinstance.traversableHandle = mRoundCurvesGAS;
        instances.push_back(oinstance);
    }

    if (!mRoundCurvesMB.empty()) {
        // Create the GAS (geometry acceleration structure) for all of the motion-blurred round
        // curves in the scene and wrap them in an OptixInstance.
        if (!createRoundCurvesGAS(cudaStream,
                                  context,
                                  mRoundCurvesMB,
                                  &mRoundCurvesMBGAS,
                                  &mRoundCurvesMBGASBuf,
                                  errorMsg)) {
            return false;
        }
        OptixInstance oinstance;
        OptixGPUXform::identityXform().toOptixTransform(oinstance.transform);
        oinstance.instanceId = 0;
        oinstance.visibilityMask = 255;
        oinstance.sbtOffset = mSBTOffset + mTriMeshes.size() + mTriMeshesMB.size() + mRoundCurves.size();
        oinstance.flags = 0;
        oinstance.traversableHandle = mRoundCurvesMBGAS;
        instances.push_back(oinstance);
    }

    if (!mCustomPrimitives.empty()) {
        // Create the GAS for all of the custom primitives in the scene and
        // wrap them in an OptixInstance.
        if (!createCustomPrimitivesGAS(cudaStream,
                                       context,
                                       mCustomPrimitives,
                                       &mCustomPrimitivesGAS,
                                       &mCustomPrimitivesGASBuf,
                                       errorMsg)) {
            return false;
        }
        OptixInstance oinstance;
        OptixGPUXform::identityXform().toOptixTransform(oinstance.transform);
        oinstance.instanceId = 0;
        oinstance.visibilityMask = 255;
        oinstance.sbtOffset = mSBTOffset + mTriMeshes.size() + mTriMeshesMB.size() +
                                           mRoundCurves.size() + mRoundCurvesMB.size();
        oinstance.flags = 0;
        oinstance.traversableHandle = mCustomPrimitivesGAS;
        instances.push_back(oinstance);

        for (auto& prim : mCustomPrimitives) {
            prim->freeHostMemory();
        }
    }

    if (!mInstances.empty()) {
        // Wrap all of the instances in OptixInstances.  But first we need to 
        // build the instances.  This causes the referenced groups/instances to be built
        // recursively.
        for (size_t i = 0; i < mInstances.size(); i++) {
            if (!mInstances[i]->build(cudaStream, context, errorMsg)) {
                return false;
            }
        }

        for (size_t i = 0; i < mInstances.size(); i++) {
            OptixInstance oinstance = {};
            oinstance.instanceId = 0;
            oinstance.visibilityMask = 255;
            oinstance.sbtOffset = 0;
            if (!mInstances[i]->mHasMotionBlur) {
                // The instance's xform is specified on the OptixInstance and the child
                // node is the top level IAS node in the referenced group.
                oinstance.flags = OPTIX_INSTANCE_FLAG_NONE;
                mInstances[i]->mXforms[0].toOptixTransform(oinstance.transform);
                oinstance.traversableHandle = mInstances[i]->mGroup->mTopLevelIAS;
            } else {
                // The xform is specified in the mMMTTraversable instead.  The child
                // node is the MatrixMotionTransform of the instance, which itself
                // has the group's top level IAS node as its child.
                oinstance.flags = 0;
                OptixGPUXform::identityXform().toOptixTransform(oinstance.transform);
                oinstance.traversableHandle = mInstances[i]->mMMTTraversable;
            }
            instances.push_back(oinstance);
        }
    }

    // Upload the instance objects and AABBs to the GPU.  Note this GPU data is
    // temporary and is freed when this function returns.
    OptixGPUBuffer<OptixInstance> instanceBuf;
    if (instanceBuf.allocAndUpload(instances) != cudaSuccess) {
        *errorMsg = "Error uploading the instance objects to the GPU";
        return false;
    }

    // We are building an IAS (instance acceleration structure), so create one that
    // contains the two instances we just created.
    OptixBuildInput input = {}; // zero initialize
    input.type = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
    input.instanceArray.instances = instanceBuf.deviceptr();
    input.instanceArray.numInstances = static_cast<unsigned int>(instanceBuf.count());

    std::vector<OptixBuildInput> inputs;
    inputs.push_back(input);

    OptixAccelBuildOptions accelOptions = {};
    accelOptions.buildFlags             = OPTIX_BUILD_FLAG_ALLOW_COMPACTION | OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
    accelOptions.motionOptions.numKeys  = 0;
    accelOptions.operation              = OPTIX_BUILD_OPERATION_BUILD;

    if (!createOptixAccel(context,
                          cudaStream,
                          accelOptions,
                          inputs,
                          true,
                          &mTopLevelIASBuf,
                          &mTopLevelIAS,
                          errorMsg)) {
        return false;
    }

    return true;
}

void
OptixGPUPrimitiveGroup::getSBTRecords(std::map<std::string, OptixProgramGroup>& pgs,
                                 std::vector<HitGroupRecord>& hitgroupRecs)
{
    // We need to create one "HitGroup" object a.k.a. record in the Shader
    // Binding Table for each OptixGPUPrimitive.  It is important that these HitGroup
    // records appear in memory in exactly the same order as they were added.
    // The HitGroup records tell Optix what kind of geometry is contained in the
    // BVH node, its properties, and what programs to call to perform ray-object
    // intersections.
    // HitGroup records have two parts.  1) A header that tells Optix what program
    // group to use and 2) custom data for the primitive such as properties, vertices, etc.
    // The properties differ between primitive types but we can only have one
    // HitGroup struct type, and all HitGroup records must be the same size,
    // so the differing properties are efficiently packed via an anonymous union.
    // See OptixGPUSBTRecord.h.

    for (size_t i = 0; i < mTriMeshes.size(); i++) {
        HitGroupRecord rec = {};
        OptixGPUTriMesh* triMesh = mTriMeshes[i];
        rec.mData.mIsSingleSided = triMesh->mIsSingleSided;
        rec.mData.mIsNormalReversed = triMesh->mIsNormalReversed;
        rec.mData.mVisibleShadow = triMesh->mVisibleShadow;
        rec.mData.mAssignmentIds = triMesh->mAssignmentIds.ptr();
        rec.mData.mNumShadowLinkLights = triMesh->mShadowLinkLights.count();
        rec.mData.mShadowLinkLights = triMesh->mShadowLinkLights.ptr();
        rec.mData.mNumShadowLinkReceivers = triMesh->mShadowLinkReceivers.count();
        rec.mData.mShadowLinkReceivers = triMesh->mShadowLinkReceivers.ptr();

        // Specify the program group to use
        optixSbtRecordPackHeader(pgs["triMeshHG"], &rec);

        hitgroupRecs.push_back(rec);
    }

    for (size_t i = 0; i < mTriMeshesMB.size(); i++) {
        HitGroupRecord rec = {};
        OptixGPUTriMesh* triMesh = mTriMeshesMB[i];
        rec.mData.mIsSingleSided = triMesh->mIsSingleSided;
        rec.mData.mIsNormalReversed = triMesh->mIsNormalReversed;
        rec.mData.mVisibleShadow = triMesh->mVisibleShadow;
        rec.mData.mAssignmentIds = triMesh->mAssignmentIds.ptr();
        rec.mData.mNumShadowLinkLights = triMesh->mShadowLinkLights.count();
        rec.mData.mShadowLinkLights = triMesh->mShadowLinkLights.ptr();
        rec.mData.mNumShadowLinkReceivers = triMesh->mShadowLinkReceivers.count();
        rec.mData.mShadowLinkReceivers = triMesh->mShadowLinkReceivers.ptr();

        // Specify the program group to use
        optixSbtRecordPackHeader(pgs["triMeshHG"], &rec);

        hitgroupRecs.push_back(rec);
    }

    for (size_t i = 0; i < mRoundCurves.size(); i++) {
        HitGroupRecord rec = {};
        OptixGPURoundCurves* curves = mRoundCurves[i];
        rec.mData.mIsSingleSided = curves->mIsSingleSided;
        rec.mData.mIsNormalReversed = curves->mIsNormalReversed;
        rec.mData.mVisibleShadow = curves->mVisibleShadow;
        rec.mData.mAssignmentIds = curves->mAssignmentIds.ptr();
        rec.mData.mNumShadowLinkLights = curves->mShadowLinkLights.count();
        rec.mData.mShadowLinkLights = curves->mShadowLinkLights.ptr();
        rec.mData.mNumShadowLinkReceivers = curves->mShadowLinkReceivers.count();
        rec.mData.mShadowLinkReceivers = curves->mShadowLinkReceivers.ptr();

        // Specify the program group to use
        switch (curves->mType) {
        case OPTIX_PRIMITIVE_TYPE_ROUND_LINEAR:
            optixSbtRecordPackHeader(pgs["roundLinearCurvesHG"], &rec);
        break;
        case OPTIX_PRIMITIVE_TYPE_ROUND_CUBIC_BSPLINE:
            optixSbtRecordPackHeader(pgs["roundCubicBsplineCurvesHG"], &rec);
        break;
        default:
            MNRY_ASSERT_REQUIRE(false);
        }

        hitgroupRecs.push_back(rec);
    }

    for (size_t i = 0; i < mRoundCurvesMB.size(); i++) {
        HitGroupRecord rec = {};
        OptixGPURoundCurves* curves = mRoundCurvesMB[i];
        rec.mData.mIsSingleSided = curves->mIsSingleSided;
        rec.mData.mIsNormalReversed = curves->mIsNormalReversed;
        rec.mData.mVisibleShadow = curves->mVisibleShadow;
        rec.mData.mAssignmentIds = curves->mAssignmentIds.ptr();
        rec.mData.mNumShadowLinkLights = curves->mShadowLinkLights.count();
        rec.mData.mShadowLinkLights = curves->mShadowLinkLights.ptr();
        rec.mData.mNumShadowLinkReceivers = curves->mShadowLinkReceivers.count();
        rec.mData.mShadowLinkReceivers = curves->mShadowLinkReceivers.ptr();

        // Specify the program group to use
        switch (curves->mType) {
        case OPTIX_PRIMITIVE_TYPE_ROUND_LINEAR:
            optixSbtRecordPackHeader(pgs["roundLinearCurvesMBHG"], &rec);
        break;
        case OPTIX_PRIMITIVE_TYPE_ROUND_CUBIC_BSPLINE:
            optixSbtRecordPackHeader(pgs["roundCubicBsplineCurvesMBHG"], &rec);
        break;
        default:
            MNRY_ASSERT_REQUIRE(false);
        }

        hitgroupRecs.push_back(rec);
    }

    for (size_t i = 0; i < mCustomPrimitives.size(); i++) {
        HitGroupRecord rec = {};

        // Fill in the common properties
        OptixGPUCustomPrimitive* prim = mCustomPrimitives[i];
        rec.mData.mIsSingleSided = prim->mIsSingleSided;
        rec.mData.mIsNormalReversed = prim->mIsNormalReversed;
        rec.mData.mVisibleShadow = prim->mVisibleShadow;
        rec.mData.mAssignmentIds = prim->mAssignmentIds.ptr();
        rec.mData.mNumShadowLinkLights = prim->mShadowLinkLights.count();
        rec.mData.mShadowLinkLights = prim->mShadowLinkLights.ptr();
        rec.mData.mNumShadowLinkReceivers = prim->mShadowLinkReceivers.count();
        rec.mData.mShadowLinkReceivers = prim->mShadowLinkReceivers.ptr();

        // Fill in the primitive type-specific properties
        OptixGPUCurve* curve = dynamic_cast<OptixGPUCurve*>(prim);
        if (curve) {
            rec.mData.curve.mMotionSamplesCount = curve->mMotionSamplesCount;
            rec.mData.curve.mSegmentsPerCurve = curve->mSegmentsPerCurve;
            rec.mData.curve.mIndices = curve->mIndices.ptr();
            rec.mData.curve.mNumControlPoints = curve->mNumControlPoints;
            rec.mData.curve.mControlPoints = curve->mControlPoints.ptr();
            switch (curve->mBasis) {
            case BEZIER:
                optixSbtRecordPackHeader(pgs["flatBezierCurveHG"], &rec);
            break;
            case BSPLINE:
                optixSbtRecordPackHeader(pgs["flatBsplineCurveHG"], &rec);
            break;
            case LINEAR:
                optixSbtRecordPackHeader(pgs["flatLinearCurveHG"], &rec);
            break;
            default:
            break;
            }
        }
        OptixGPUPoints* points = dynamic_cast<OptixGPUPoints*>(prim);
        if (points) {
            rec.mData.points.mMotionSamplesCount = points->mMotionSamplesCount;
            rec.mData.points.mPoints = points->mPoints.ptr();
            optixSbtRecordPackHeader(pgs["pointsHG"], &rec);
        }
        OptixGPUSphere* sphere = dynamic_cast<OptixGPUSphere*>(prim);
        if (sphere) {
            rec.mData.sphere.mL2P = sphere->mL2P;
            rec.mData.sphere.mP2L = sphere->mP2L;
            rec.mData.sphere.mRadius = sphere->mRadius;
            rec.mData.sphere.mPhiMax = sphere->mPhiMax;
            rec.mData.sphere.mZMin = sphere->mZMin;
            rec.mData.sphere.mZMax = sphere->mZMax;
            optixSbtRecordPackHeader(pgs["sphereHG"], &rec);
        }
        OptixGPUBox* box = dynamic_cast<OptixGPUBox*>(prim);
        if (box) {
            rec.mData.box.mL2P = box->mL2P;
            rec.mData.box.mP2L = box->mP2L;
            rec.mData.box.mLength = box->mLength;
            rec.mData.box.mHeight = box->mHeight;
            rec.mData.box.mWidth = box->mWidth;
            optixSbtRecordPackHeader(pgs["boxHG"], &rec);
        }

        hitgroupRecs.push_back(rec);
    }
}

} // namespace rt
} // namespace moonray

