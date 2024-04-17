// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#ifndef __APPLE__
#include <optix.h>
#endif

#include "MetalGPUShadowLinking.h"

namespace moonray {
namespace rt {

// This is standard NVIDIA boilerplate that should probably be part of the API
template <typename T>
struct
#ifndef __APPLE__
__align__(OPTIX_SBT_RECORD_ALIGNMENT)
#endif
SBTRecord
{
#ifndef __APPLE__
    __align__(OPTIX_SBT_RECORD_ALIGNMENT) char mHeader[OPTIX_SBT_RECORD_HEADER_SIZE];
#endif
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
    // Properties common to all primitives
    bool mIsSingleSided;
    bool mIsNormalReversed;
    bool mVisibleShadow;
    unsigned char mUserInstanceID; 
    int __gpu_device__ *mAssignmentIds;
 //   unsigned int mNumShadowLinkEntries;
 //   int __gpu_device__ *mShadowLinkAssignmentIds;
 //   uint64_t __gpu_device__ *mShadowLinkLightIds;

    // Whether this primitive will cast a shadow from specific lights
    unsigned mNumShadowLinkLights;
    ShadowLinkLight __gpu_device__ *mShadowLinkLights;

    // Whether this primitive will cast a shadow onto specific receivers
    unsigned mNumShadowLinkReceivers;
    ShadowLinkReceiver __gpu_device__ *mShadowLinkReceivers;


    // Primitive type-specific properties.  This is all unioned together because
    // we need a fixed-size data structure for the Shader Binding Table entries.
    // Note that there is no type field telling us the primitive type.  We don't
    // need it because each primitive type is handled by its own programs,
    // (see program groups), thus it is already known.
    // Also note that we only store raw pointers below while on the host side we
    // use GPUBuffer objects.  These pointers are obtained from the host-side
    // GPUBuffer objects.
    union {
        struct {
            MetalGPUXform mL2P;
            MetalGPUXform mP2L;
            float mLength;
            float mHeight;
            float mWidth;
        } box;
        struct {
            int mMotionSamplesCount;
            unsigned int mSegmentsPerCurve;
            const unsigned int __gpu_device__* mIndices;
            int mNumControlPoints;
            const float4 __gpu_device__* mControlPoints;
        } curve;
        struct {
            int mMotionSamplesCount;
            const float4 __gpu_device__* mPoints;
        } points;
        struct {
            MetalGPUXform mL2P;
            MetalGPUXform mP2L;
            float mRadius;
            float mPhiMax;
            float mZMin;
            float mZMax;
        } sphere;
        // There is no TriMesh or RoundCurves here because the intersection program that uses
        // these data is built in to Optix and these data have already been passed in via
        // the Optix API elsewhere.
#ifdef __METAL__
    } u;
#else
    };
#endif
};

typedef SBTRecord<HitGroupData> HitGroupRecord;

} // namespace rt
} // namespace moonray

