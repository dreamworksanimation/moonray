// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "OptixGPUBuffer.h"
#include "OptixGPUMath.h"
#include "OptixGPUShadowLinking.h"

#include <cuda.h>
#include <vector>

namespace moonray {
namespace rt {

// The host-side representation of the geometric primitives in the scene.
// Typically this is a few parameters and a set of buffers that live on the
// GPU (as it is up to the host to manage the GPU memory.)

class OptixGPUPrimitive
{
public:
    virtual ~OptixGPUPrimitive() {}

    std::string mName;
    unsigned int mInputFlags;
    bool mIsSingleSided;
    bool mIsNormalReversed;
    bool mVisibleShadow;

    // per sub-primitive (or just one item if no sub-primitives)
    OptixGPUBuffer<int> mAssignmentIds;

    OptixGPUBuffer<ShadowLinkLight> mShadowLinkLights;
    OptixGPUBuffer<ShadowLinkReceiver> mShadowLinkReceivers;
};

// Basic triangle mesh geometry.  Each triangle has 3 verts.  Moonray quads
// have been tessellated into tris.
// Triangle meshes are hardware accelerated and do not have intersection programs.

class OptixGPUTriMesh : public OptixGPUPrimitive
{
public:
    OptixGPUBuffer<float3>       mVertices;
    // We need to keep a pointer to each motion sample's vertex buffer. If there is no motion blur, both pointers
    // will point to the same buffer.
    std::array<CUdeviceptr, 2>   mVerticesPtrs;
    size_t                       mNumVertices;

    OptixGPUBuffer<unsigned int> mIndices;
    size_t                       mNumFaces;

    bool                         mEnableMotionBlur;
};

// Linear or BSpline round curves.  This is supported as a built-in type by Optix 7.1,
// thus they do not need an intersection program specified.

class OptixGPURoundCurves : public OptixGPUPrimitive
{
public:
    OptixPrimitiveType mType;

    int mMotionSamplesCount; // 1 or 2 - a current limitation of RoundCurves

    // each index points to the first control point in a curve segment
    OptixGPUBuffer<unsigned int> mIndices;

    int mNumControlPoints;
    // We need to keep a pointer to each motion sample's vertex buffer. If there is no motion blur, both pointers
    // will point to the same buffer.
    std::array<CUdeviceptr, 2> mVerticesPtrs;
    OptixGPUBuffer<float3> mVertices;
    std::array<CUdeviceptr, 2> mWidthsPtrs;
    OptixGPUBuffer<float> mWidths; // radius, but Optix calls it width
};

// Custom primitives for non-trimesh geometry.  These have custom intersection
// programs on the GPU.  You cannot mix trimeshes and custom primitives in the
// same traversable (BVH) so they are treated separately throughout the GPU code.

class OptixGPUCustomPrimitive : public OptixGPUPrimitive
{
public:
    virtual ~OptixGPUCustomPrimitive() {}

    // Optix needs the Aabb(s) to put the custom primitive into the BVH.
    // If this is something simple like a sphere, there are no sub-primitives and
    // aabbs.size() == 1.  If it is a set of points, there is a separate Aabb
    // for each point as each sub-primitive may land in a different BVH node.
    virtual void getPrimitiveAabbs(std::vector<OptixAabb>* aabbs) const = 0;

    // Custom primitives hold onto the host-side geometry for a while so the
    // getPrimitiveAabbs() function can compute aabbs.  After the BVH is built
    // we can free this memory.
    virtual void freeHostMemory() {}
};

class OptixGPUBox : public OptixGPUCustomPrimitive
{
public:
    void getPrimitiveAabbs(std::vector<OptixAabb>* aabbs) const override;

    OptixGPUXform mL2P;  // Local to Primitive
    OptixGPUXform mP2L;  // Primitive to Local
    float mLength;
    float mHeight;
    float mWidth;
};

class OptixGPUCurve : public OptixGPUCustomPrimitive
{
public:
    void getPrimitiveAabbs(std::vector<OptixAabb>* aabbs) const override;
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
    OptixGPUBuffer<unsigned int> mIndices;

    // GPU-side copy of the control points
    OptixGPUBuffer<float4> mControlPoints;
};

class OptixGPUPoints : public OptixGPUCustomPrimitive
{
public:
    void getPrimitiveAabbs(std::vector<OptixAabb>* aabbs) const override;
    void freeHostMemory() override;

    // Number of motion samples for motion blur.  1 = no motion blur.
    int mMotionSamplesCount;

    // Points are XYZ and radius
    std::vector<float4> mHostPoints;

    // GPU-side copy of the points
    OptixGPUBuffer<float4> mPoints;
};

class OptixGPUSphere : public OptixGPUCustomPrimitive
{
public:
    void getPrimitiveAabbs(std::vector<OptixAabb>* aabbs) const override;

    OptixGPUXform mL2P;  // Local to Primitive
    OptixGPUXform mP2L;  // Primitive to Local
    float mRadius;
    float mPhiMax;
    float mZMin;
    float mZMax;
};

} // namespace rt
} // namespace moonray

