// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <Metal/Metal.h>
#include <simd/simd.h>
#include "MetalGPUBuffer.h"
#include "MetalGPUMath.h"
#include "MetalGPUShadowLinking.h"

using namespace simd;

#include <vector>

namespace moonray {
namespace rt {

static const int GPU_MOTION_NONE = 0;
static const int GPU_MOTION_INSTANCES = 1;
static const int GPU_MOTION_PRIMITIVES = 2;

// The host-side representation of the geometric primitives in the scene.
// Typically this is a few parameters and a set of buffers that live on the
// GPU (as it is up to the host to manage the GPU memory.)

class MetalGPUPrimitive
{
public:
    MetalGPUPrimitive(id<MTLDevice> context):
        mAssignmentIds(context),
        mShadowLinkReceivers(context),
        mShadowLinkLights(context) {}
    virtual ~MetalGPUPrimitive() {}

    std::string mName;
    float3 mBounds[2];
    unsigned int mInputFlags;
    bool mIsSingleSided;
    bool mIsNormalReversed;
    bool mVisibleShadow;

    // per sub-primitive (or just one item if no sub-primitives)
    MetalGPUBuffer<int> mAssignmentIds;

    MetalGPUBuffer<ShadowLinkLight> mShadowLinkLights;
    MetalGPUBuffer<ShadowLinkReceiver> mShadowLinkReceivers;
};

// Basic triangle mesh geometry.  Each triangle has 3 verts.  Moonray quads
// have been tessellated into tris.
// Triangle meshes are hardware accelerated and do not have intersection programs.

class MetalGPUTriMesh : public MetalGPUPrimitive
{
public:
    MetalGPUTriMesh(id<MTLDevice> context):
        MetalGPUPrimitive(context),
        mVertices(context),
        mIndices(context) {}

    MetalGPUBuffer<float3>          mVertices;
    // We need to keep a pointer to each motion sample's vertex buffer. If there is no motion blur, both pointers
    // will point to the same buffer.
    std::array<uint64_t, 2>    mVerticesPtrs;
    size_t                     mNumVertices;

    MetalGPUBuffer<unsigned int>    mIndices;
    size_t                     mNumFaces;

    bool                       mEnableMotionBlur;
};

// Linear or BSpline round curves.  This is supported as a built-in type by Optix 7.1,
// thus they do not need an intersection program specified.

class MetalGPURoundCurves : public MetalGPUPrimitive
{
public:
    MetalGPURoundCurves(id<MTLDevice> context):
        MetalGPUPrimitive(context),
        mIndices(context),
        mVertices(context),
        mWidths(context) {}
    MTLCurveType mSubType;
    MTLCurveBasis mType;

    int mMotionSamplesCount; // 1 or 2 - a current limitation of RoundCurves

    // each index points to the first control point in a curve segment
    MetalGPUBuffer<unsigned int> mIndices;

    int mNumControlPoints;
    // We need to keep a pointer to each motion sample's vertex buffer. If there is no motion blur, both pointers
    // will point to the same buffer.
    std::array<uint64_t, 2> mVerticesPtrs;
    MetalGPUBuffer<float3> mVertices;
    std::array<uint64_t, 2> mWidthsPtrs;
    MetalGPUBuffer<float> mWidths; // radius, but Optix calls it width
};

// Custom primitives for non-trimesh geometry.  These have custom intersection
// programs on the GPU.  You cannot mix trimeshes and custom primitives in the
// same traversable (BVH) so they are treated separately throughout the GPU code.

class MetalGPUCustomPrimitive : public MetalGPUPrimitive
{
public:
    MetalGPUCustomPrimitive(id<MTLDevice> context):
        MetalGPUPrimitive(context) {}
    virtual ~MetalGPUCustomPrimitive() {}

    // Optix/Metal needs the Aabb(s) to put the custom primitive into the BVH.
    // If this is something simple like a sphere, there are no sub-primitives and
    // aabbs.size() == 1.  If it is a set of points, there is a separate Aabb
    // for each point as each sub-primitive may land in a different BVH node.
    virtual void getPrimitiveAabbs(std::vector<OptixAabb>* aabbs) const = 0;

    // Metal uses this offset to index into the intersection_function_table to
    // call the correct kernel function to perform the intersection check
    virtual int getFuncTableOffset() const = 0;

    // Metal must know which types of acceleration structures are present in the
    // acceleration structure, so that the correct __raygen__ and  intersection
    // function tables are used against it
    virtual void hasMotionBlur(uint32_t *flags) const = 0;
    
    // Custom primitives hold onto the host-side geometry for a while so the
    // getPrimitiveAabbs() function can compute aabbs.  After the BVH is built
    // we can free this memory.
    virtual void freeHostMemory() {}
};

class MetalGPUBox : public MetalGPUCustomPrimitive
{
public:
    MetalGPUBox(id<MTLDevice> context) : MetalGPUCustomPrimitive(context) {}

    void getPrimitiveAabbs(std::vector<OptixAabb>* aabbs) const override;
    int getFuncTableOffset() const override;
    void hasMotionBlur(uint32_t *flags) const override {}

    MetalGPUXform mL2P;  // Local to Primitive
    MetalGPUXform mP2L;  // Primitive to Local
    float mLength;
    float mHeight;
    float mWidth;
};

class MetalGPUCurve : public MetalGPUCustomPrimitive
{
public:
    MetalGPUCurve(id<MTLDevice> context)
        : MetalGPUCustomPrimitive(context)
        , mIndices(context)
        , mControlPoints(context) {}

    void getPrimitiveAabbs(std::vector<OptixAabb>* aabbs) const override;
    int getFuncTableOffset() const override;
    void hasMotionBlur(uint32_t *flags) const override {
        if (mMotionSamplesCount > 1) {
            *flags = *flags | GPU_MOTION_PRIMITIVES;
        }
    }
    void freeHostMemory() override;

    unsigned int mSegmentsPerCurve;
    unsigned int mBasis;  // BEZIER, BSPLINE, LINEAR

    // Number of motion samples for motion blur.  1 = no motion blur.
    int mMotionSamplesCount;

    // Host-side copy of the curve indices
    std::vector<unsigned int> mHostIndices;

    int mNumControlPoints;
    // Host-side copy of the control points
    // Control points are XYZ and radius
    std::vector<float4> mHostControlPoints;

    // GPU-side copy of the indices
    MetalGPUBuffer<unsigned int> mIndices;

    // GPU-side copy of the control points
    MetalGPUBuffer<float4> mControlPoints;
};

class MetalGPUPoints : public MetalGPUCustomPrimitive
{
public:
    MetalGPUPoints(id<MTLDevice> context) :
        MetalGPUCustomPrimitive(context),
        mPoints(context) {}

    void getPrimitiveAabbs(std::vector<OptixAabb>* aabbs) const override;
    void hasMotionBlur(uint32_t *flags) const override {
        if (mMotionSamplesCount > 1) {
            *flags |= GPU_MOTION_PRIMITIVES;
        }
    }
    int getFuncTableOffset() const override;
    void freeHostMemory() override;

    // Number of motion samples for motion blur.  1 = no motion blur.
    int mMotionSamplesCount;

    // Points are XYZ and radius
    std::vector<float4> mHostPoints;

    // GPU-side copy of the points
    MetalGPUBuffer<float4> mPoints;
};

class MetalGPUSphere : public MetalGPUCustomPrimitive
{
public:
    MetalGPUSphere(id<MTLDevice> context) : MetalGPUCustomPrimitive(context) {}

    void getPrimitiveAabbs(std::vector<OptixAabb>* aabbs) const override;
    void hasMotionBlur(uint32_t *flags) const override {}
    int getFuncTableOffset() const override;

    MetalGPUXform mL2P;  // Local to Primitive
    MetalGPUXform mP2L;  // Primitive to Local
    float mRadius;
    float mPhiMax;
    float mZMin;
    float mZMax;
};

} // namespace rt
} // namespace moonray

