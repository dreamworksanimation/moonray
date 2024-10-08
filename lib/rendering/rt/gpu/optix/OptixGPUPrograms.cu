// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

#include "../GPURay.h"
#include "OptixGPUMath.h"
#include "OptixGPUParams.h"
#include "OptixGPUSBTRecord.h"
#include "EmbreeRoundlineIntersector.h"
#include "EmbreeCurveIntersectorSweep.h"

using namespace moonray::rt;

extern "C" __constant__ static OptixGPUParams params;

#define EMBREE_INVALID_GEOMETRY_ID 0xffffffff
#define INVALID_LIGHT_ID 0xffffffff

// Each ray has a PerRayData associated with it that is globally
// available in all of the programs.  Inputs and outputs are passed via this
// struct.

struct PerRayData
{
    bool mIsOcclusionRay;
    int mMask;
    int mShadowReceiverId;        // used for shadow linking (input)
    unsigned long long mLightId;  // used for shadow linking (input)
    bool mDidHitGeom;             // did ray hit geometry? (output)

    float mTFar; // intersection distance
    float mNgX, mNgY, mNgZ; // geometry normal
    float mU, mV; // barycentric coords

    unsigned int mPrimID;
    unsigned int mEmbreeGeomID;
    intptr_t mEmbreeUserData;

    unsigned int mInstance0IdOrLight;
    unsigned int mInstance1Id;
    unsigned int mInstance2Id;
    unsigned int mInstance3Id;
    float mL2R[4][3]; // layout matches Mat43
};


// Optix can only pass 32-bit unsigned integer "payload" parameters to its programs.
// But, we want to pass a pointer to the PerRayData.  These utility functions
// split and reconstruct the pointer to/from 32-bit uints and retrieve it from
// the 32-bit "payload" parameters.

inline __device__
void splitPointer(const void* ptr, unsigned int& i0, unsigned int& i1)
{
    const unsigned long long uptr = reinterpret_cast<unsigned long long>(ptr);
    i0 = uptr >> 32;
    i1 = uptr & 0x00000000ffffffff;
}

inline __device__
void *reconstructPointer(const unsigned int i0, const unsigned int i1)
{
    const unsigned long long uptr = static_cast<unsigned long long>(i0) << 32 | i1;
    void* ptr = reinterpret_cast<void*>(uptr);
    return ptr;
}


union PackedFloat
{
    unsigned int ibits;
    float fbits;
};

inline __device__
float asFloat(const unsigned int i)
{
    PackedFloat pf;
    pf.ibits = i;
    return pf.fbits;
}

inline __device__
unsigned int asInt(const float f)
{
    PackedFloat pf;
    pf.fbits = f;
    return pf.ibits;
}


template<typename T>
inline __device__
T *getPRD()
{
    const unsigned int u0 = optixGetPayload_0();
    const unsigned int u1 = optixGetPayload_1();
    return reinterpret_cast<T*>(reconstructPointer(u0, u1));
}

// The __raygen__ program sets up the PerRayData and then calls optixTrace()
// which starts the BVH traversal and calling of other programs.  It copies 
// the prd.mDidHitGeom result to the appropriate location in the output buffer.
// A __raygen__ function normally implements a camera.

extern "C" __global__
void __raygen__()
{
    const uint3 idx = optixGetLaunchIndex();

    const moonray::rt::GPURay *ray = params.mRaysBuf + idx.x;

    if (params.mIsectBuf) {
        // intersection ray
        PerRayData prd;
        prd.mIsOcclusionRay = false;
        prd.mMask = ray->mMask;
        prd.mShadowReceiverId = -1;
        prd.mLightId = INVALID_LIGHT_ID;
        prd.mDidHitGeom = false;
        // other prd members don't need to be cleared

        unsigned int u0, u1;
        splitPointer(&prd, u0, u1);

        optixTrace(params.mAccel,
                {ray->mOriginX, ray->mOriginY, ray->mOriginZ},
                {ray->mDirX, ray->mDirY, ray->mDirZ},
                ray->mMinT,
                ray->mMaxT,
                ray->mTime,
                OptixVisibilityMask(ray->mMask & 0xff),
                OPTIX_RAY_FLAG_NONE,
                0,             // SBT offset
                1,             // SBT stride
                0,             // missSBTIndex
                u0, u1);

        GPURayIsect *isect = params.mIsectBuf + idx.x;
        if (prd.mDidHitGeom) {
            isect->mTFar = prd.mTFar;
            isect->mNgX = prd.mNgX;
            isect->mNgY = prd.mNgY;
            isect->mNgZ = prd.mNgZ;
            isect->mU = prd.mU;
            isect->mV = prd.mV;
            isect->mEmbreeGeomID = prd.mEmbreeGeomID;
            isect->mPrimID = prd.mPrimID;
            isect->mEmbreeUserData = prd.mEmbreeUserData;
            isect->mInstance0IdOrLight = prd.mInstance0IdOrLight;
            isect->mInstance1Id = prd.mInstance1Id;
            isect->mInstance2Id = prd.mInstance2Id;
            isect->mInstance3Id = prd.mInstance3Id;
            memcpy(isect->mL2R, prd.mL2R, 12 * sizeof(float));
        } else {
            isect->mTFar = ray->mMaxT; // must be set even if no intersection or volumes will fail
            isect->mEmbreeGeomID = EMBREE_INVALID_GEOMETRY_ID;
            isect->mPrimID = 0;
            isect->mEmbreeUserData = 0;
            isect->mInstance0IdOrLight = 0;
            isect->mInstance1Id = 0;
            isect->mInstance2Id = 0;
            isect->mInstance3Id = 0;
        }
    } else {
        // occlusion ray
        PerRayData prd;
        prd.mIsOcclusionRay = true;
        prd.mMask = ray->mMask;
        prd.mShadowReceiverId = ray->mShadowReceiverId;
        prd.mLightId = ray->mLightId;
        prd.mDidHitGeom = false;
        // other prd members don't need to be cleared

        unsigned int u0, u1;
        splitPointer(&prd, u0, u1);

        optixTrace(params.mAccel,
                {ray->mOriginX, ray->mOriginY, ray->mOriginZ},
                {ray->mDirX, ray->mDirY, ray->mDirZ},
                ray->mMinT,
                ray->mMaxT,
                ray->mTime,
                OptixVisibilityMask(ray->mMask & 0xff),
                OPTIX_RAY_FLAG_NONE, // OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT for occlusion rays?
                0,             // SBT offset
                1,             // SBT stride
                0,             // missSBTIndex
                u0, u1);

        params.mIsOccludedBuf[idx.x] = prd.mDidHitGeom ? 1 : 0;
    }
}

inline __device__
void getTriVerts(const moonray::rt::HitGroupData* data, float3 verts[3])
{
    int motionSamplesCount = data->mesh.mMotionSamplesCount;
    int primIdx = optixGetPrimitiveIndex();
    int idx0 = data->mesh.mIndices[primIdx * 3];
    int idx1 = data->mesh.mIndices[primIdx * 3 + 1];
    int idx2 = data->mesh.mIndices[primIdx * 3 + 2];

    if (motionSamplesCount == 1) {
        verts[0] = data->mesh.mVertices[0][idx0];
        verts[1] = data->mesh.mVertices[0][idx1];
        verts[2] = data->mesh.mVertices[0][idx2];
    } else {
        const float time = optixGetRayTime();
        const float sample0PlusT = time * (motionSamplesCount - 1);
        const unsigned int sample0 = static_cast<unsigned int>(sample0PlusT);
        const unsigned int sample1 = fminf(sample0 + 1, motionSamplesCount - 1); // clamp to time = 1
        const float t = sample0PlusT - static_cast<float>(sample0);
        float3 vtx0_0 = data->mesh.mVertices[sample0][idx0];
        float3 vtx1_0 = data->mesh.mVertices[sample0][idx1];
        float3 vtx2_0 = data->mesh.mVertices[sample0][idx2];
        float3 vtx0_1 = data->mesh.mVertices[sample1][idx0];
        float3 vtx1_1 = data->mesh.mVertices[sample1][idx1];
        float3 vtx2_1 = data->mesh.mVertices[sample1][idx2];
        verts[0] = lerp(vtx0_0, vtx0_1, t);
        verts[1] = lerp(vtx1_0, vtx1_1, t);
        verts[2] = lerp(vtx2_0, vtx2_1, t);
    }
}

// The __closesthit__ program is called after optixReportIntersection() has been
// called in one or more of the __intersection__ programs and Optix has finished 
// its BVH traversal.  It is called once for the closest intersection.

extern "C" __global__
void __closesthit__()
{
    PerRayData* prd = getPRD<PerRayData>();
    prd->mDidHitGeom = true;

    if (!prd->mIsOcclusionRay) {
        const moonray::rt::HitGroupData* data = (moonray::rt::HitGroupData*)optixGetSbtDataPointer();

        prd->mTFar = optixGetRayTmax();
        prd->mEmbreeGeomID = data->mEmbreeGeomID;
        prd->mEmbreeUserData = data->mEmbreeUserData;

        float objectToWorld[12];
        optixGetObjectToWorldTransformMatrix(objectToWorld);
        prd->mL2R[0][0] = objectToWorld[0];
        prd->mL2R[0][1] = objectToWorld[4];
        prd->mL2R[0][2] = objectToWorld[8];
        prd->mL2R[1][0] = objectToWorld[1];
        prd->mL2R[1][1] = objectToWorld[5];
        prd->mL2R[1][2] = objectToWorld[9];
        prd->mL2R[2][0] = objectToWorld[2];
        prd->mL2R[2][1] = objectToWorld[6];
        prd->mL2R[2][2] = objectToWorld[10];
        prd->mL2R[3][0] = objectToWorld[3];
        prd->mL2R[3][1] = objectToWorld[7];
        prd->mL2R[3][2] = objectToWorld[11];

        // Did we traverse any instances along the way to this hit?
        prd->mInstance0IdOrLight = 0;
        prd->mInstance1Id = 0;
        prd->mInstance2Id = 0;
        prd->mInstance3Id = 0;

        unsigned int instanceDepth = 0;
        for (unsigned int i = 0; i < optixGetTransformListSize(); ++i) {
            OptixTraversableHandle handle = optixGetTransformListHandle(i);
            switch (optixGetTransformTypeFromHandle(handle)) {
                case OPTIX_TRANSFORM_TYPE_INSTANCE:
                {
                    unsigned int instanceId = optixGetInstanceIdFromHandle(handle);
                    if (instanceId == 0) {
                        // not a regular MoonRay instance
                        continue;
                    }
                    // else this is a MoonRay instance, record the Id
                    // refer to lib/rendering/geom/prim/Instance.cc::intersectFunc()
                    switch (instanceDepth) {
                        case 0: prd->mInstance0IdOrLight = instanceId;
                        break;
                        case 1: prd->mInstance1Id = instanceId;
                        break;
                        case 2: prd->mInstance2Id = instanceId;
                        break;
                        case 3: prd->mInstance3Id = instanceId;
                        break;
                        // We ignore instances more than 4 deep.
                    }
                    instanceDepth++;
                }
                break;
            }
        }

        switch (data->mType) {
            case HitGroupData::TRIANGLE_MESH:
            case HitGroupData::QUAD_MESH:
            {
                float3 verts[3];
                getTriVerts(data, verts);
                float3 ng = cross(verts[1] - verts[0], verts[2] - verts[0]);
                prd->mNgX = ng.x;
                prd->mNgY = ng.y;
                prd->mNgZ = ng.z;

                float2 uv = optixGetTriangleBarycentrics();
                if (data->mType == HitGroupData::QUAD_MESH) {
                    unsigned int triIdx = optixGetPrimitiveIndex();
                    if ((triIdx & 1) == 0) {
                        prd->mU = uv.x;
                        prd->mV = uv.y;
                    } else {
                        prd->mU = 1.f - uv.x;
                        prd->mV = 1.f - uv.y;
                    }
                    prd->mPrimID = triIdx >> 1; // / 2
                } else {
                    prd->mU = uv.x;
                    prd->mV = uv.y;
                    prd->mPrimID = optixGetPrimitiveIndex();
                }
            }
            break;
            case HitGroupData::SPHERE:
            {
                prd->mNgX = asFloat(optixGetAttribute_0());
                prd->mNgY = asFloat(optixGetAttribute_1()); 
                prd->mNgZ = asFloat(optixGetAttribute_2());
                prd->mU = asFloat(optixGetAttribute_3());
                prd->mV = 0.f;  // computed in cpu side postIntersect()
                prd->mPrimID = 0;
            }
            break;
            case HitGroupData::BOX:
            {
                prd->mNgX = asFloat(optixGetAttribute_0());
                prd->mNgY = asFloat(optixGetAttribute_1()); 
                prd->mNgZ = asFloat(optixGetAttribute_2());
                prd->mU = asFloat(optixGetAttribute_3());
                prd->mV = asFloat(optixGetAttribute_4());
                prd->mPrimID = 0;
            }
            break;
            case HitGroupData::POINTS:
            {
                prd->mNgX = asFloat(optixGetAttribute_0());
                prd->mNgY = asFloat(optixGetAttribute_1()); 
                prd->mNgZ = asFloat(optixGetAttribute_2());
                prd->mU = 0.f; // always zeros for points
                prd->mV = 0.f;
                prd->mPrimID = optixGetPrimitiveIndex();
            }
            break;
            case HitGroupData::FLAT_LINEAR_CURVES:
            {
                prd->mNgX = 0.f; // unused by MoonRay
                prd->mNgY = 0.f; // unused by MoonRay
                prd->mNgZ = 1.f; // unused by MoonRay
                prd->mU = asFloat(optixGetAttribute_0());
                prd->mV = 0.f; // always zero
                prd->mPrimID = optixGetPrimitiveIndex();
            }
            break;
            case HitGroupData::FLAT_BEZIER_CURVES:
            case HitGroupData::FLAT_BSPLINE_CURVES:
            {
                prd->mNgX = 0.f; // unused by MoonRay
                prd->mNgY = 0.f; // unused by MoonRay
                prd->mNgZ = 1.f; // unused by MoonRay
                prd->mU = asFloat(optixGetAttribute_0());
                prd->mV = asFloat(optixGetAttribute_1());
                prd->mPrimID = optixGetPrimitiveIndex();
            }
            break;
            case HitGroupData::ROUND_LINEAR_CURVES:
            case HitGroupData::ROUND_BSPLINE_CURVES:
            {
                prd->mNgX = asFloat(optixGetAttribute_0());
                prd->mNgY = asFloat(optixGetAttribute_1());
                prd->mNgZ = asFloat(optixGetAttribute_2());
                prd->mU = asFloat(optixGetAttribute_3());
                prd->mV = 0.f;
                prd->mPrimID = optixGetPrimitiveIndex();
            }
            break;
            default:
                // Should never get here
                prd->mNgX = 0.f;
                prd->mNgY = 0.f; 
                prd->mNgZ = 1.f;
                prd->mU = 0.f;
                prd->mV = 0.f;
                prd->mPrimID = 0;
        };

    } else {
        // not used for occlusion tests, but clear them anyway
        prd->mTFar = 0.f;
        prd->mNgX = 0.f;
        prd->mNgY = 0.f;
        prd->mNgZ = 0.f;
        prd->mU = 0.f;
        prd->mV = 0.f;
        prd->mPrimID = 0;
        prd->mEmbreeGeomID = EMBREE_INVALID_GEOMETRY_ID;
        prd->mEmbreeUserData = 0;
        prd->mInstance0IdOrLight = 0;
        prd->mInstance1Id = 0;
        prd->mInstance2Id = 0;
        prd->mInstance3Id = 0;
    }
}


// The __anyhit__ program is called for all intersections reported by optixReportIntersection().
// Here we can ignore intersections based on sidedness, visible shadows, shadow linking.

extern "C" __global__
void __anyhit__()
{
    PerRayData* prd = getPRD<PerRayData>();
    const moonray::rt::HitGroupData* data = (moonray::rt::HitGroupData*)optixGetSbtDataPointer();

    if ((data->mMask & prd->mMask) == 0) {
        optixIgnoreIntersection();
        return;
    }

    if (data->mIsSingleSided && optixIsTriangleBackFaceHit()) {
        optixIgnoreIntersection();
        return;
    }

    if (prd->mIsOcclusionRay) {
        unsigned int primIdx = optixGetPrimitiveIndex();
        unsigned int casterId = data->mAssignmentIds[primIdx];

        for (unsigned i = 0; i < data->mNumShadowLinkLights; i++) {
            if (casterId != data->mShadowLinkLights[i].mCasterId) {
                // entry doesn't apply to this caster id
                continue;
            }
            if (prd->mLightId == data->mShadowLinkLights[i].mLightId) {
                // if there is a match, the current object can't cast a shadow
                // from this specific light
                optixIgnoreIntersection();
                return;
            }
        }

        bool receiverMatches = false;
        bool isComplemented = false;
        for (unsigned i = 0; i < data->mNumShadowLinkReceivers; i++) {
            if (casterId != data->mShadowLinkReceivers[i].mCasterId) {
                // entry doesn't apply to this caster id
                continue;
            }

            // this is the same for all [i] for this matching casterId,
            // we just need one of them
            isComplemented = data->mShadowLinkReceivers[i].mIsComplemented;

            if (prd->mShadowReceiverId == data->mShadowLinkReceivers[i].mReceiverId) {
                // if there is a match, the current object can't cast a shadow
                // onto this receiver
                receiverMatches = true;
                break;
            }
        }

        if (receiverMatches ^ isComplemented) {
            optixIgnoreIntersection();
            return;
        }
    }
}


// The __miss__ program is called if the ray does not intersect any geometry
// (or all intersections have been ignored in __anyhit__).

extern "C" __global__
void __miss__()
{
    PerRayData* prd = getPRD<PerRayData>();
    prd->mDidHitGeom = false;
}



// Next, we have all of the ray __intersection__ programs for the Custom Primitives.
// These are closely adapted from the existing Moonray code or from Embree 3.9.

extern "C" __global__
void __intersection__box()
{
    // See: lib/rendering/geom/prim/Box.cc::intersectFunc()

    const HitGroupData* data = (HitGroupData*)optixGetSbtDataPointer();
    const float xLength = data->box.mLength;
    const float yHeight = data->box.mHeight;
    const float zWidth = data->box.mWidth;

    const float3 rayOrigin = data->sphere.mP2L.transformPoint(optixGetObjectRayOrigin());
    const float3 rayDir = data->sphere.mP2L.transformVector(optixGetObjectRayDirection());
    const float rayTnear = optixGetRayTmin();
    const float rayTfar = optixGetRayTmax();

    const bool isSingleSided = data->mIsSingleSided;
    const bool isNormalReversed = data->mIsNormalReversed;
    const float3 pMin = {-xLength * 0.5f, -yHeight * 0.5f, -zWidth * 0.5f};
    const float3 pMax = {xLength * 0.5f, yHeight * 0.5f, zWidth * 0.5f};

    float t0 = -FLOAT_MAX;
    float t1 = FLOAT_MAX;
    {
        const float invDirX = 1.f / rayDir.x;
        float tNearX = (pMin.x - rayOrigin.x) * invDirX;
        float tFarX = (pMax.x - rayOrigin.x) * invDirX;
        // Note that tNearX or tFarX can be NaN if rayDir.x == 0 
        // and min/pMax.x == rayOrigin.x.  This is an indeterminate situation
        // where the ray lies entirely in the YZ plane and grazes the box at the
        // min/max X coordinate.  We make the simplification that the hit/miss
        // is "don't care" when this happens.
        // NaNs are OK in this logic because all comparisons with NaN are false,
        // so the code still does something defined, if inconsistent depending on which
        // values are NaN.
        if (tNearX > tFarX) {
            swapf(tNearX, tFarX);
        }
        t0 = tNearX > t0 ? tNearX : t0;
        t1 = tFarX < t1 ? tFarX : t1;
        if (t0 > t1) {
            return;
        }
    }
    {
        const float invDirY = 1.f / rayDir.y;
        float tNearY = (pMin.y - rayOrigin.y) * invDirY;
        float tFarY = (pMax.y - rayOrigin.y) * invDirY;
        if (tNearY > tFarY) {
            swapf(tNearY, tFarY);
        }
        t0 = tNearY > t0 ? tNearY : t0;
        t1 = tFarY < t1 ? tFarY : t1;
        if (t0 > t1) {
            return;
        }
    }
    {
        const float invDirZ = 1.f / rayDir.z;
        float tNearZ = (pMin.z - rayOrigin.z) * invDirZ;
        float tFarZ = (pMax.z - rayOrigin.z) * invDirZ;
        if (tNearZ > tFarZ) {
            swapf(tNearZ, tFarZ);
        }
        t0 = tNearZ > t0 ? tNearZ : t0;
        t1 = tFarZ < t1 ? tFarZ : t1;
        if (t0 > t1) {
            return;
        }
    }

    if (t0 > rayTfar || t1 < rayTnear) {
        return;
    }
    float tHit = t0;
    if (t0 < rayTnear || (isSingleSided && isNormalReversed)) {
        tHit = t1;
        if (tHit > rayTfar || (isSingleSided && !isNormalReversed)) {
            return;
        }
    }

    // compute hit position
    float3 pHit = rayOrigin + tHit * rayDir;

    // Needed to check on which face of the box a ray intersects
    static float constexpr sEpsilonEdge = 1e-5f;

    float3 Ng;
    float u, v;
    // local space normal will be transformed
    // to parent space at postIntersect
    if (pHit.x < pMin.x * (1 - sEpsilonEdge) ||
        pHit.x > pMax.x * (1 - sEpsilonEdge)) { // left or right face
        Ng.x = pHit.x >= 0.f ? 1.f : -1.f;
        Ng.y = 0.0f;
        Ng.z = 0.0f;
        u = (pHit.y - pMin.y) / (pMax.y - pMin.y);
        v = (pHit.z - pMin.z) / (pMax.z - pMin.z);
    } else if (pHit.y < pMin.y * (1 - sEpsilonEdge) ||
               pHit.y > pMax.y * (1 - sEpsilonEdge)) { // top or bottom face
        Ng.x = 0.0f;
        Ng.y = pHit.y >= 0.f ? 1.f : -1.f;
        Ng.z = 0.0f;
        u = (pHit.x - pMin.x) / (pMax.x - pMin.x);
        v = (pHit.z - pMin.z) / (pMax.z - pMin.z);
    } else { // front or back face
        Ng.x = 0.0f;
        Ng.y = 0.0f;
        Ng.z = pHit.z >= 0.f ? 1.f : -1.f;
        u = (pHit.x - pMin.x) / (pMax.x - pMin.x);
        v = (pHit.y - pMin.y) / (pMax.y - pMin.y);
    }

    optixReportIntersection(tHit,
                            0,
                            asInt(Ng.x),
                            asInt(Ng.y),
                            asInt(Ng.z),
                            asInt(u),
                            asInt(v));
}

__inline__ __device__
void projectCurveControlPoints(const float4* cp,
                               const unsigned int numPoints,
                               const float3& rayOrg,
                               const float3& rayDir,
                               float4* cp2d)
{
    // Rotate curve control points to Z plane perpendicular to ray
    OptixGPUXform curve2DXform = OptixGPUXform::rotateToZAxisXform(rayDir);
    for (int i = 0; i < numPoints; i++) {
        float3 pos2d = curve2DXform.transformVector(make_float3(cp[i]) - rayOrg);
        cp2d[i] = make_float4(pos2d, cp[i].w);
    }
    // The "projection" is just ignoring the Z value from here onward.

    // note that we can transformVector() instead of transformPoint() because we have already
    // applied the translation by subtracting ray.mOrg
}

__inline__ __device__
float computeCurveEpsilon(const float4 cp[4])
{
    // from embree
    return 4.f * FLOAT_EPSILON * fmaxf(maxAbsComponent(make_float3(cp[0])),
                                       maxAbsComponent(make_float3(cp[1])),
                                       maxAbsComponent(make_float3(cp[2])),
                                       maxAbsComponent(make_float3(cp[3])));
}

__inline__ __device__
bool intersectCurveQuad(const float3 vtx[4],
                        const float dirLength,
                        const float rayTMin,
                        const float rayTMax,
                        float *u, float *v, float *t)
{
    // From embree intersect_quad_backface_culling(), except we assume
    // the ray origin is at zero.

    // Figure out which triangle to test for intersection (0-1-3 or 2-3-1).
    // Form a plane with the ray origin, vtx[3], and vtx[1].  The cross product
    // gives the normal.  Dotting with the ray dir tells us which side of the plane
    // the ray is on and thus which triangle to test for intersection.
    float side = dot(cross(vtx[3], vtx[1]), {0.f, 0.f, dirLength});
    const float3 v0 = side <= 0.f ? vtx[0] : vtx[2];
    const float3 v1 = side <= 0.f ? vtx[1] : vtx[3];
    const float3 v2 = side <= 0.f ? vtx[3] : vtx[1];

    // Perform similar ray-plane side tests for the other two triangle edges
    const float3 e0 = v2 - v0;
    const float3 e1 = v0 - v1;
    float uu = dot(cross(v0, e0), {0.f, 0.f, dirLength});
    float vv = dot(cross(v1, e1), {0.f, 0.f, dirLength});
    if (uu > 0.f || vv > 0.f) {
        return false;
    }

    const float3 Ng = cross(e1, e0);  // geom normal
    float den = dot(Ng, {0.f, 0.f, dirLength});
    if (den == 0.f) {
        return false;
    }
    float invDen = 1.f / den;

    *t = invDen * dot(v0, Ng);
    if (rayTMin > *t || rayTMax < *t) {
        return false;
    }

    *u = uu * invDen;
    *v = vv * invDen;

    // Adjust u and v depending on which triangle we intersected in the quad
    *u = side <= 0.f ? *u : 1.f - *u;
    *v = side <= 0.f ? *v : 1.f - *v;

    return true;
}

__device__
void getCurveControlPoints(float4 cp[4])
{
    const HitGroupData* data = (HitGroupData*)optixGetSbtDataPointer();
    const unsigned int primIdx = optixGetPrimitiveIndex();

    const int motionSamplesCount = data->curve.mMotionSamplesCount;
    if (motionSamplesCount == 1) {
        const float4* cp0 = data->curve.mControlPoints + data->curve.mIndices[primIdx];
        cp[0] = cp0[0];
        cp[1] = cp0[1];
        cp[2] = cp0[2];
        cp[3] = cp0[3];
    } else {
        const float time = optixGetRayTime();
        const float sample0PlusT = time * (motionSamplesCount - 1);
        const unsigned int sample0 = static_cast<unsigned int>(sample0PlusT);
        const unsigned int sample1 = fminf(sample0 + 1, motionSamplesCount - 1); // clamp to time = 1
        const float t = sample0PlusT - static_cast<float>(sample0);
        const float4* cp0 = data->curve.mControlPoints + data->curve.mNumControlPoints * sample0 +
                                                         data->curve.mIndices[primIdx];
        const float4* cp1 = data->curve.mControlPoints + data->curve.mNumControlPoints * sample1 +
                                                         data->curve.mIndices[primIdx];
        cp[0] = lerp(cp0[0], cp1[0], t);
        cp[1] = lerp(cp0[1], cp1[1], t);
        cp[2] = lerp(cp0[2], cp1[2], t);
        cp[3] = lerp(cp0[3], cp1[3], t);
    }
}

extern "C" __global__
void __intersection__flat_bezier_curve()
{
    // Based on PBRT

    float4 cp[4];
    getCurveControlPoints(cp);

    const HitGroupData* data = (HitGroupData*)optixGetSbtDataPointer();
    const float3 rayOrg = optixGetObjectRayOrigin();
    const float3 rayDir = optixGetObjectRayDirection();
    const float rayTmin = optixGetRayTmin();
    const float rayTmax = optixGetRayTmax();
    const float dirLength = length(rayDir);
    const unsigned int segmentsPerCurve = data->curve.mSegmentsPerCurve;

    float4 cp2d[4];
    projectCurveControlPoints(cp, 4, rayOrg, rayDir, cp2d);

    float eps = computeCurveEpsilon(cp2d);

    bool hit = false;
    float closestHitT = rayTmax;
    float isectU = 0.f;
    float isectV = 0.f;

    float deltaT = 1.f / static_cast<float>(segmentsPerCurve);

    float4 p1 = cp2d[0]; // The bezier basis at u=0 is {1, 0, 0, 0}
    float3 tangentVec = 3.f * make_float3(cp2d[1] - cp2d[0]);
    if (maxAbsComponent(tangentVec) < eps) {
        float t2 = deltaT;
        tangentVec = make_float3(evalBezier(cp2d, t2)) - make_float3(p1);
    }
    float3 n1 = p1.w * normalize({tangentVec.y, -tangentVec.x, 0.f});

    for (int i = 0; i < segmentsPerCurve; i++) {

        float4 p0 = p1;
        float3 n0 = n1;

        float t0 = static_cast<float>(i) * deltaT;
        float t1 = static_cast<float>(i+1) * deltaT;
        p1 = evalBezier(cp2d, t1);

        tangentVec = evalBezierDerivative(cp2d, t1);
        if (maxAbsComponent(tangentVec) < eps) {
            tangentVec = make_float3(p1) - make_float3(p0);
        }
        n1 = p1.w * normalize({tangentVec.y, -tangentVec.x, 0.f});

        const float3 quadVerts[4] = {make_float3(p0) + n0,
                                     make_float3(p1) + n1,
                                     make_float3(p1) - n1,
                                     make_float3(p0) - n0};

        float u, v, t;
        if (!intersectCurveQuad(quadVerts,
                                dirLength,
                                rayTmin,
                                closestHitT,
                                &u, &v, &t)) {
            continue;
        }

        float r = lerpf(p0.w, p1.w, u);
        if (t < 2.f * r / dirLength) {
            continue;
        }

        hit = true;
        closestHitT = t;
        isectU = t0 + u * deltaT;
        isectV = 2.f * v - 1.f;
    }

    if (hit) {
        optixReportIntersection(closestHitT,
                                0,
                                asInt(isectU),
                                asInt(isectV));
    }
}

extern "C" __global__
void __intersection__flat_bspline_curve()
{
    // Based on PBRT

    float4 cp[4];
    getCurveControlPoints(cp);

    const HitGroupData* data = (HitGroupData*)optixGetSbtDataPointer();
    const float3 rayOrg = optixGetObjectRayOrigin();
    const float3 rayDir = optixGetObjectRayDirection();
    const float rayTmin = optixGetRayTmin();
    const float rayTmax = optixGetRayTmax();
    const float dirLength = length(rayDir);
    const unsigned int segmentsPerCurve = data->curve.mSegmentsPerCurve;

    float4 cp2d[4];
    projectCurveControlPoints(cp, 4, rayOrg, rayDir, cp2d);

    float eps = computeCurveEpsilon(cp2d);

    bool hit = false;
    float closestHitT = rayTmax;
    float isectU = 0.f;
    float isectV = 0.f;

    float deltaT = 1.f / static_cast<float>(segmentsPerCurve);

    float4 p1 = evalBspline(cp2d, 0.f);
    float3 tangentVec = evalBsplineDerivative(cp2d, 0.f);
    if (maxAbsComponent(tangentVec) < eps) {
        float t2 = deltaT;
        tangentVec = make_float3(evalBspline(cp2d, t2)) - make_float3(p1);
    }
    float3 n1 = p1.w * normalize({tangentVec.y, -tangentVec.x, 0.f});

    for (int i = 0; i < segmentsPerCurve; i++) {

        float4 p0 = p1;
        float3 n0 = n1;

        float t0 = static_cast<float>(i) * deltaT;
        float t1 = static_cast<float>(i+1) * deltaT;
        p1 = evalBspline(cp2d, t1);
        tangentVec = evalBsplineDerivative(cp2d, t1);
        if (maxAbsComponent(tangentVec) < eps) {
            tangentVec = make_float3(p1) - make_float3(p0);
        }
        n1 = p1.w * normalize({tangentVec.y, -tangentVec.x, 0.f});

        const float3 quadVerts[4] = {make_float3(p0) + n0,
                                     make_float3(p1) + n1,
                                     make_float3(p1) - n1,
                                     make_float3(p0) - n0};

        float u, v, t;
        if (!intersectCurveQuad(quadVerts,
                                dirLength,
                                rayTmin,
                                closestHitT,
                                &u, &v, &t)) {
            continue;
        }

        float r = lerpf(p0.w, p1.w, u);
        if (t < 2.f * r / dirLength) {
            continue;
        }

        hit = true;
        closestHitT = t;
        isectU = t0 + u * deltaT;
        isectV = 2.f * v - 1.f;
    }

    if (hit) {
        optixReportIntersection(closestHitT,
                                0,
                                asInt(isectU),
                                asInt(isectV));
    }
}

extern "C" __global__
void __intersection__round_bspline_curve()
{
    float4 cp[4];
    getCurveControlPoints(cp);

    const HitGroupData* data = (HitGroupData*)optixGetSbtDataPointer();
    const float3 rayOrg = optixGetObjectRayOrigin();
    const float3 rayDir = optixGetObjectRayDirection();
    const float rayTmin = optixGetRayTmin();
    const float rayTmax = optixGetRayTmax();

    EmbreeRayHit rayHit;
    rayHit.org = rayOrg;
    rayHit.dir = rayDir;
    rayHit.tnear = rayTmin;
    rayHit.tfar = rayTmax;
    rayHit.Ng = {0.f, 0.f, 0.f};
    rayHit.u = 0.f;
    rayHit.v = 0.f;

    RoundBezierCurveEpilog epilog(&rayHit);

    SweepCurveIntersector intersector;
    if (intersector.intersect(rayHit,
                              cp[0], cp[1], cp[2], cp[3],
                              epilog)) {
        optixReportIntersection(rayHit.tfar,
                                0,
                                asInt(rayHit.Ng.x),
                                asInt(rayHit.Ng.y),
                                asInt(rayHit.Ng.z),
                                asInt(rayHit.u));
    }
}

extern "C" __global__
void __intersection__flat_linear_curve()
{
    // Based on PBRT

    const HitGroupData* data = (HitGroupData*)optixGetSbtDataPointer();
    const unsigned int primIdx = optixGetPrimitiveIndex();

    const float3 rayOrg = optixGetObjectRayOrigin();
    const float3 rayDir = optixGetObjectRayDirection();
    const float rayTmin = optixGetRayTmin();
    const float rayTmax = optixGetRayTmax();
    const float dirLength = length(rayDir);

    const int motionSamplesCount = data->curve.mMotionSamplesCount;
    float4 cp[2];
    if (motionSamplesCount == 1) {
        const float4* cp0 = data->curve.mControlPoints + data->curve.mIndices[primIdx];
        cp[0] = cp0[0];
        cp[1] = cp0[1];
    } else {
        const float time = optixGetRayTime();
        const float sample0PlusT = time * (motionSamplesCount - 1);
        const unsigned int sample0 = static_cast<unsigned int>(sample0PlusT);
        const unsigned int sample1 = fminf(sample0 + 1, motionSamplesCount - 1); // clamp to time = 1
        const float t = sample0PlusT - static_cast<float>(sample0);
        const float4* cp0 = data->curve.mControlPoints + data->curve.mNumControlPoints * sample0 +
                                                         data->curve.mIndices[primIdx];
        const float4* cp1 = data->curve.mControlPoints + data->curve.mNumControlPoints * sample1 +
                                                         data->curve.mIndices[primIdx];
        cp[0] = lerp(cp0[0], cp1[0], t);
        cp[1] = lerp(cp0[1], cp1[1], t);
    }

    float4 cp2d[2];
    projectCurveControlPoints(cp, 2, rayOrg, rayDir, cp2d);

    // PBRT3 section 3.7 fig 3.23
    const float4 dv = cp2d[1] - cp2d[0];
    if (dv.x == 0.f && dv.y == 0.f && dv.z == 0.f) {
        // Ignore degenerate segments
        return;
    }
    const float d0 = -cp2d[0].x * dv.x - cp2d[0].y * dv.y;
    const float d1 = dv.x * dv.x + dv.y * dv.y;

    // u distance along line between cp2d[0] and cp2d[1]
    const float u = clampf(d0 / d1, 0.f, 1.f);

    // Closest point on the line cp2d[0]-cp2d[1] to (0,0)
    const float4 p = cp2d[0] + u * dv;

    // Ray intersection distance to xy plane at z=p.z
    const float t = p.z / dirLength;

    // Distance^2 of p to the ray (at origin pointing down Z axis)
    const float d2 = p.x * p.x + p.y * p.y;

    // r = radius of line at p
    const float r = p.w;
    const float r2 = r * r;

    if (d2 <= r2 && rayTmin <= t && t <= rayTmax && t > 2.f * r / dirLength) {
        optixReportIntersection(t,
                                0,
                                asInt(u));
    }
}

extern "C" __global__
void __intersection__round_linear_curve()
{
    // Calls into ported Embree code

    const HitGroupData* data = (HitGroupData*)optixGetSbtDataPointer();
    const unsigned int primIdx = optixGetPrimitiveIndex();

    const float3 rayOrg = optixGetObjectRayOrigin();
    const float3 rayDir = optixGetObjectRayDirection();
    const float rayTmin = optixGetRayTmin();
    const float rayTmax = optixGetRayTmax();
    const float dirLength = length(rayDir);

    const int motionSamplesCount = data->curve.mMotionSamplesCount;
    float4 cp[2];
    float4 cpL, cpR;
    if (motionSamplesCount == 1) {
        unsigned int cpIdx = data->curve.mIndices[primIdx];
        cp[0] = data->curve.mControlPoints[cpIdx];
        cp[1] = data->curve.mControlPoints[cpIdx + 1];
        cpL = { inf, inf, inf, inf };
        if (primIdx > 0) {
            unsigned int cpLeftIdx = data->curve.mIndices[primIdx - 1];
            if ((cpLeftIdx + 1) == cpIdx) {
                // Curve continues to the left
                // From embree docs: left segment exists if segment(id-1)+1 == segment(id)
                cpL = data->curve.mControlPoints[cpLeftIdx];
            }
        }
        cpR = { inf, inf, inf, inf };
        if (primIdx + 1 < data->curve.mNumIndices) {
            unsigned int cpRightIdx = data->curve.mIndices[primIdx + 1];
            if ((cpRightIdx - 1) == cpIdx) {
                // Curve continues to the right
                // From embree docs: right segment exists if segment(id+1)-1 == segment(id)
                cpR = data->curve.mControlPoints[cpRightIdx];
            }
        }
    } else {
        const float time = optixGetRayTime();
        const float sample0PlusT = time * (motionSamplesCount - 1);
        const unsigned int sample0 = static_cast<unsigned int>(sample0PlusT);
        const unsigned int sample1 = fminf(sample0 + 1, motionSamplesCount - 1); // clamp to time = 1
        const float t = sample0PlusT - static_cast<float>(sample0);
        unsigned int cpIdx = data->curve.mIndices[primIdx];
        const float4* cp0 = data->curve.mControlPoints + data->curve.mNumControlPoints * sample0 +
                                                         cpIdx;
        const float4* cp1 = data->curve.mControlPoints + data->curve.mNumControlPoints * sample1 +
                                                         cpIdx;
        cp[0] = lerp(cp0[0], cp1[0], t);
        cp[1] = lerp(cp0[1], cp1[1], t);
        cpL = { inf, inf, inf, inf };
        if (primIdx > 0) {
            unsigned int cpLeftIdx = data->curve.mIndices[primIdx - 1];
            if ((cpLeftIdx + 1) == cpIdx) {
                // Curve continues to the left
                float4 cpL0 = data->curve.mControlPoints[data->curve.mNumControlPoints * sample0 + cpLeftIdx];
                float4 cpL1 = data->curve.mControlPoints[data->curve.mNumControlPoints * sample1 + cpLeftIdx];
                cpL = lerp(cpL0, cpL1, t);
            }
        }
        cpR = { inf, inf, inf, inf };
        if (primIdx + 1 < data->curve.mNumIndices) {
            unsigned int cpRightIdx = data->curve.mIndices[primIdx + 1];
            if ((cpRightIdx - 1) == cpIdx) {
                // Curve continues to the right
                float4 cpR0 = data->curve.mControlPoints[data->curve.mNumControlPoints * sample0 + cpRightIdx];
                float4 cpR1 = data->curve.mControlPoints[data->curve.mNumControlPoints * sample1 + cpRightIdx];
                cpR = lerp(cpR0, cpR1, t);
            }
        }
    }

    EmbreeRayHit rayHit;
    rayHit.org = rayOrg;
    rayHit.dir = rayDir;
    rayHit.tnear = rayTmin;
    rayHit.tfar = rayTmax;
    rayHit.Ng = {0.f, 0.f, 0.f};
    rayHit.u = 0.f;
    rayHit.v = 0.f;

    RoundLinearCurveEpilog epilog(&rayHit);

    RoundLinearCurveIntersector intersector;
    if (intersector.intersect(true,
                              rayHit,
                              cp[0], cp[1],
                              cpL, cpR,
                              epilog)) {
        optixReportIntersection(rayHit.tfar,
                                0,
                                asInt(rayHit.Ng.x),
                                asInt(rayHit.Ng.y),
                                asInt(rayHit.Ng.z),
                                asInt(rayHit.u));
    }
}

extern "C" __global__
void __intersection__points()
{
    // See: lib/rendering/geom/prim/Points.cc::intersectFunc()

    const HitGroupData* data = (HitGroupData*)optixGetSbtDataPointer();
    const unsigned int primIdx = optixGetPrimitiveIndex();

    const int motionSamplesCount = data->points.mMotionSamplesCount;
    float4 p4;
    if (motionSamplesCount == 1) {
        p4 = data->points.mPoints[primIdx]; // xyzr
    } else {
        const float time = optixGetRayTime();
        const float idxPlusT = time * (motionSamplesCount - 1);
        const unsigned int idx0 = static_cast<unsigned int>(idxPlusT);
        const unsigned int idx1 = fminf(idx0 + 1, motionSamplesCount - 1); // clamp to time = 1
        const float t = idxPlusT - static_cast<float>(idx0);
        p4 = lerp(data->points.mPoints[primIdx * motionSamplesCount + idx0],
                  data->points.mPoints[primIdx * motionSamplesCount + idx1], t);
    }

    const float3 p = make_float3(p4);
    const float r = p4.w;
    const float r2 = r * r;
    const float3 o = optixGetObjectRayOrigin();
    const float3 d = optixGetObjectRayDirection();
    const float rayTnear = optixGetRayTmin();
    const float rayTfar = optixGetRayTmax();

    const float3 u = p - o;
    if (dot(d, u) < 0.0f) {
        // ray is travelling away from sphere centre
        return;
    }
    if (dot(u, u) < r2) {
        // ray origin is inside sphere
        return;
    }
    const float3 v = cross(d, u);
    float v2 = dot(v, v);
    float d2 = dot(d, d);
    float r2d2 = r2 * d2;
    float D = r2d2 - v2;
    if (D < 0.0f) {
        // no intersections
        return;
    }
    float s = sqrt(D);
    const float3 w = cross(d, v);
    const float3 rn = (1.0f / d2) * (w - s * d);
    float t = length(u + rn) * rsqrt(d2);
    if (rayTnear < t && t < rayTfar) {
        float3 Ng = (1.0f / r) * rn;
        optixReportIntersection(t,
                                0,
                                asInt(Ng.x),
                                asInt(Ng.y),
                                asInt(Ng.z));
    }
}

extern "C" __global__
void __intersection__sphere()
{
    // See: lib/rendering/geom/prim/Sphere.cc::intersectFunc()

    const HitGroupData* data = (HitGroupData*)optixGetSbtDataPointer();

    const float3 org = data->sphere.mP2L.transformPoint(optixGetObjectRayOrigin());
    const float3 dir = data->sphere.mP2L.transformVector(optixGetObjectRayDirection());
    const float rayTnear = optixGetRayTmin();
    const float rayTfar = optixGetRayTmax();

    const bool isSingleSided = data->mIsSingleSided;
    const bool isNormalReversed = data->mIsNormalReversed;
    const float radius = data->sphere.mRadius;

    // Compute quadratic sphere coefficients.
    // Note: we eliminate some multiplies by halving B relative to the pbrt treatment,
    // so that the equation we're solving is now A*t^2 + 2*B*t + C = 0.
    // Also, calcs for C and D are deferred till after testing for real roots.
    float A = dot(dir, dir);
    float B = dot(org, dir);
    // More precise discriminant calculation to replace D = B * B - A * C;
    // see PBRT4 section 6.8.3 for details.
    float3 v = org - (B / A) * dir;
    float vlen = length(v);
    // This tests if the discriminant D is negative, without actually calculating it
    if (radius < vlen) {
        return;
    }
    float C = dot(org, org) - radius * radius;
    float D = A * (radius + vlen) * (radius - vlen);
    float rootDiscrim = sqrtf(D);
    float q = B < 0.0f ? rootDiscrim - B : -rootDiscrim - B;
    float t0 = q / A;
    float t1 = C / q;
    if (t0 > t1) {
        swapf(t0, t1);
    }
    // compute intersection distance along ray
    if (t0 > rayTfar || t1 < rayTnear) {
        return;
    }
    float tHit = t0;
    if (t0 < rayTnear || (isSingleSided && isNormalReversed)) {
        tHit = t1;
        if (tHit > rayTfar || (isSingleSided && !isNormalReversed)) {
            return;
        }
    }
    // compute hit position and phi
    float3 pHit = org + tHit * dir;
    // Refine sphere intersection point (PBRT4)
    pHit = pHit * (radius / length(pHit));
    if (pHit.x == 0.0f && pHit.y == 0.0f) {
        pHit.x = 1e-5f * radius;
    }
    float phi = atan2f(pHit.y, pHit.x);
    if (phi < 0.0f) {
        phi += 2.f * static_cast<float>(M_PI);
    }
    float zMin = data->sphere.mZMin;
    float zMax = data->sphere.mZMax;
    float phiMax = data->sphere.mPhiMax;
    // test sphere intersection against clipping parameters
    if (pHit.z < zMin || pHit.z > zMax || phi > phiMax) {
        if (tHit == t1) {
            return;
        }
        if (t1 > rayTfar || (isSingleSided && !isNormalReversed)) {
            return;
        }
        tHit = t1;
        pHit = org + tHit * dir;
        // Refine sphere intersection point (PBRT4)
        pHit = pHit * (radius / length(pHit));
        if (pHit.x == 0.0f && pHit.y == 0.0f) {
            pHit.x = 1e-5f * radius;
        }
        phi = atan2f(pHit.y, pHit.x);
        if (phi < 0.0f) {
            phi += 2.f * static_cast<float>(M_PI);
        }
        if (pHit.z < zMin || pHit.z > zMax || phi > phiMax) {
            return;
        }
    }

    optixReportIntersection(tHit,
                            0,
                            asInt(pHit.x),
                            asInt(pHit.y),
                            asInt(pHit.z),
                            asInt(phi));
}
