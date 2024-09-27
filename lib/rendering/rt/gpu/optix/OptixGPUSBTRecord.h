// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "OptixGPUShadowLinking.h"
#include <optix.h>

namespace moonray {
namespace rt {

// This is standard NVIDIA boilerplate that should probably be part of the API
template <typename T>
struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) SBTRecord
{
    __align__(OPTIX_SBT_RECORD_ALIGNMENT) char mHeader[OPTIX_SBT_RECORD_HEADER_SIZE];
    T mData;
};


struct RaygenData
{
    // empty
};

typedef SBTRecord<RaygenData> RaygenRecord;


struct MissData
{
    // empty
};

typedef SBTRecord<MissData> MissRecord;

// The HitGroupData contains all of the Primitive data needed for Optix to execute the
// programs on the GPU.

struct HitGroupData
{
    enum Type {
        TRIANGLE_MESH,
        QUAD_MESH,
        FLAT_LINEAR_CURVES,
        FLAT_BSPLINE_CURVES,
        FLAT_BEZIER_CURVES,
        ROUND_LINEAR_CURVES,
        ROUND_BSPLINE_CURVES,
        POINTS,
        SPHERE,
        BOX
    } mType;

    // Properties common to all primitives
    bool mIsSingleSided;
    bool mIsNormalReversed;
    int mMask;
    int *mAssignmentIds;
    intptr_t mEmbreeUserData;
    unsigned int mEmbreeGeomID;

    // Whether this primitive will cast a shadow from specific lights
    unsigned mNumShadowLinkLights;
    ShadowLinkLight *mShadowLinkLights;

    // Whether this primitive will cast a shadow onto specific receivers
    unsigned mNumShadowLinkReceivers;
    ShadowLinkReceiver *mShadowLinkReceivers;

    // Primitive type-specific properties.  This is all unioned together because
    // we need a fixed-size data structure for the Shader Binding Table entries.
    // Also note that we only store raw pointers below while on the host side we
    // use OptixGPUBuffer objects.  These pointers are obtained from the host-side
    // OptixGPUBuffer objects.
    union {
        struct {
            OptixGPUXform mL2P;
            OptixGPUXform mP2L;
            float mLength;
            float mHeight;
            float mWidth;
        } box;
        struct {
            int mMotionSamplesCount;
            unsigned int mSegmentsPerCurve;
            int mNumIndices;
            const unsigned int* mIndices;
            int mNumControlPoints;
            const float4* mControlPoints;
        } curve;
        struct {
            int mMotionSamplesCount;
            const float4* mPoints;
        } points;
        struct {
            OptixGPUXform mL2P;
            OptixGPUXform mP2L;
            float mRadius;
            float mPhiMax;
            float mZMin;
            float mZMax;
        } sphere;
        struct {
            int mMotionSamplesCount;
            const unsigned int* mIndices;
            const float3* mVertices[16]; // MAX_MOTION_BLUR_SAMPLES
        } mesh;
    };
};

typedef SBTRecord<HitGroupData> HitGroupRecord;

} // namespace rt
} // namespace moonray
