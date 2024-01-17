// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

#include "../GPURay.h"
#include "OptixGPUMath.h"
#include "OptixGPUParams.h"
#include "OptixGPUSBTRecord.h"

using namespace moonray::rt;

extern "C" __constant__ static OptixGPUParams params;


// Each ray has a PerRayData associated with it that is globally
// available in all of the programs.  Inputs and outputs are passed via this
// struct.

struct PerRayData
{
    int mShadowReceiverId;        // used for shadow linking (input)
    unsigned long long mLightId;  // used for shadow linking (input)
    bool mDidHitGeom;             // did ray hit geometry? (output)
    // todo: intersect() results
};


// Optix can only pass 32-bit unsigned integer "payload" parameters to its programs.
// But, we want to pass a pointer to the PerRayData.  These utility functions
// split and reconstruct the pointer to/from 32-bit uints and retrieve it from
// the 32-bit "payload" parameters.

inline __device__
void packPointer(const void* ptr, unsigned int& i0, unsigned int& i1)
{
    const unsigned long long uptr = reinterpret_cast<unsigned long long>(ptr);
    i0 = uptr >> 32;
    i1 = uptr & 0x00000000ffffffff;
}

inline __device__
void *unpackPointer(const unsigned int i0, const unsigned int i1)
{
    const unsigned long long uptr = static_cast<unsigned long long>(i0) << 32 | i1;
    void* ptr = reinterpret_cast<void*>(uptr);
    return ptr;
}

template<typename T>
inline __device__
T *getPRD()
{
    const unsigned int u0 = optixGetPayload_0();
    const unsigned int u1 = optixGetPayload_1();
    return reinterpret_cast<T*>(unpackPointer(u0, u1));
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

    // todo: differentiate between intersect() and occluded() rays by checking
    // which params.mIsectBuf or params.mIsOccludedBuf is nullptr.

    PerRayData prd;
    prd.mShadowReceiverId = ray->mShadowReceiverId;
    prd.mLightId = ray->mLightId;
    prd.mDidHitGeom = false;
    unsigned int u0, u1;
    packPointer(&prd, u0, u1);

    optixTrace(params.mAccel,
               {ray->mOriginX, ray->mOriginY, ray->mOriginZ},
               {ray->mDirX, ray->mDirY, ray->mDirZ},
               ray->mMinT,
               ray->mMaxT,
               ray->mTime,
               OptixVisibilityMask( 255 ),
               OPTIX_RAY_FLAG_NONE, // OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT for occlusion rays?
               0,             // SBT offset
               1,             // SBT stride
               0,             // missSBTIndex 
               u0, u1);

    params.mIsOccludedBuf[idx.x] = prd.mDidHitGeom ? 1 : 0;
}


// The __closesthit__ program is called after optixReportIntersection() has been
// called in one or more of the __intersection__ programs and Optix has finished 
// its BVH traversal.  It is called once for the closest intersection.

extern "C" __global__
void __closesthit__()
{
    PerRayData* prd = getPRD<PerRayData>();
    prd->mDidHitGeom = true;
}


// The __anyhit__ program is called for all intersections reported by optixReportIntersection().
// Here we can ignore intersections based on sidedness, visible shadows, shadow linking.

extern "C" __global__
void __anyhit__()
{
    PerRayData* prd = getPRD<PerRayData>();
    const moonray::rt::HitGroupData* data = (moonray::rt::HitGroupData*)optixGetSbtDataPointer();

    if (data->mIsSingleSided && optixIsTriangleBackFaceHit()) {
        optixIgnoreIntersection();
        return;
    }
    if (!data->mVisibleShadow) {
        optixIgnoreIntersection();
        return;
    }

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
    const float3 minCoord = {-xLength * 0.5f, -yHeight * 0.5f, -zWidth * 0.5f};
    const float3 maxCoord = {xLength * 0.5f, yHeight * 0.5f, zWidth * 0.5f};

    float t0 = -FLOAT_MAX;
    float t1 = FLOAT_MAX;
    {
        const float invDirX = 1.f / rayDir.x;
        float tNearX = (minCoord.x - rayOrigin.x) * invDirX;
        float tFarX = (maxCoord.x - rayOrigin.x) * invDirX;
        // Note that tNearX or tFarX can be NaN if rayDir.x == 0 
        // and min/maxCoord.x == rayOrigin.x.  This is an indeterminate situation
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
        float tNearY = (minCoord.y - rayOrigin.y) * invDirY;
        float tFarY = (maxCoord.y - rayOrigin.y) * invDirY;
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
        float tNearZ = (minCoord.z - rayOrigin.z) * invDirZ;
        float tFarZ = (maxCoord.z - rayOrigin.z) * invDirZ;
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

    optixReportIntersection(tHit, 0);
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

extern "C" __global__
void __intersection__flat_bezier_curve()
{
    const HitGroupData* data = (HitGroupData*)optixGetSbtDataPointer();
    const unsigned int primIdx = optixGetPrimitiveIndex();
    const unsigned int segmentsPerCurve = data->curve.mSegmentsPerCurve;

    const float3 rayOrg = optixGetObjectRayOrigin();
    const float3 rayDir = optixGetObjectRayDirection();
    const float rayTmin = optixGetRayTmin();
    const float rayTmax = optixGetRayTmax();
    const float dirLength = length(rayDir);

    const int motionSamplesCount = data->curve.mMotionSamplesCount;
    float4 cp[4];
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

    float4 cp2d[4];
    projectCurveControlPoints(cp, 4, rayOrg, rayDir, cp2d);

    float eps = computeCurveEpsilon(cp2d);

    bool hit = false;
    float closestHitT = rayTmax;

    float4 p1 = cp2d[0]; // The bezier basis at u=0 is {1, 0, 0, 0}
    float3 tangentVec = 3.f * make_float3(cp2d[1] - cp2d[0]);
    if (maxAbsComponent(tangentVec) < eps) {
        float t2 = 1.f / static_cast<float>(segmentsPerCurve);
        tangentVec = make_float3(evalBezier(cp2d, t2)) - make_float3(p1);
    }
    float3 n1 = p1.w * normalize({tangentVec.y, -tangentVec.x, 0.f});

    for (int i = 0; i < segmentsPerCurve; i++) {

        float4 p0 = p1;
        float3 n0 = n1;

        float t1 = static_cast<float>(i+1) / static_cast<float>(segmentsPerCurve);
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
    }

    if (hit) {
        optixReportIntersection(closestHitT, 0);
    }
}

extern "C" __global__
void __intersection__flat_bspline_curve()
{
    const HitGroupData* data = (HitGroupData*)optixGetSbtDataPointer();
    const unsigned int primIdx = optixGetPrimitiveIndex();
    const unsigned int segmentsPerCurve = data->curve.mSegmentsPerCurve;

    const float3 rayOrg = optixGetObjectRayOrigin();
    const float3 rayDir = optixGetObjectRayDirection();
    const float rayTmin = optixGetRayTmin();
    const float rayTmax = optixGetRayTmax();
    const float dirLength = length(rayDir);

    const int motionSamplesCount = data->curve.mMotionSamplesCount;
    float4 cp[4];
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

    float4 cp2d[4];
    projectCurveControlPoints(cp, 4, rayOrg, rayDir, cp2d);

    float eps = computeCurveEpsilon(cp2d);

    bool hit = false;
    float closestHitT = rayTmax;

    float4 p1 = evalBspline(cp2d, 0.f);
    float3 tangentVec = evalBsplineDerivative(cp2d, 0.f);
    if (maxAbsComponent(tangentVec) < eps) {
        float t2 = 1.f / static_cast<float>(segmentsPerCurve);
        tangentVec = make_float3(evalBspline(cp2d, t2)) - make_float3(p1);
    }
    float3 n1 = p1.w * normalize({tangentVec.y, -tangentVec.x, 0.f});

    for (int i = 0; i < segmentsPerCurve; i++) {

        float4 p0 = p1;
        float3 n0 = n1;

        float t1 = static_cast<float>(i+1) / static_cast<float>(segmentsPerCurve);
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
    }

    if (hit) {
        optixReportIntersection(closestHitT, 0);
    }
}

extern "C" __global__
void __intersection__flat_linear_curve()
{
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
        optixReportIntersection(t, 0);
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

    const float3 rayOrg = optixGetObjectRayOrigin();
    float3 rayDir = optixGetObjectRayDirection();
    const float rayTnear = optixGetRayTmin();
    const float rayTfar = optixGetRayTmax();

    const float3 v = p - rayOrg;
    if (dot(rayDir, v) < 0.f) {
        // ray is travelling away from sphere centre
        return;
    }
    if (dot(v, v) < r2) {
        // ray origin is inside sphere
        return;
    }

    const float3 dxv = cross(rayDir, v);
    const float d2 = dot(rayDir, rayDir);
    const float D = d2 * r2 - dot(dxv, dxv);
    if (D < 0.f) {
        // no intersections
        return;
    }

    const float Q = sqrtf(D);
    const float3 dxdxv = cross(rayDir, dxv);
    float3 Ng = (dxdxv - Q * rayDir) / d2;
    const float t = length(v + Ng);
    if (rayTnear < t && t < rayTfar) {
        optixReportIntersection(t, 0);
    }
}

extern "C" __global__
void __intersection__sphere()
{
    // See: lib/rendering/geom/prim/Sphere.cc::intersectFunc()

    const HitGroupData* data = (HitGroupData*)optixGetSbtDataPointer();

    const float3 rayOrigin = data->sphere.mP2L.transformPoint(optixGetObjectRayOrigin());
    const float3 rayDir = data->sphere.mP2L.transformVector(optixGetObjectRayDirection());
    const float rayTnear = optixGetRayTmin();
    const float rayTfar = optixGetRayTmax();

    const bool isSingleSided = data->mIsSingleSided;
    const bool isNormalReversed = data->mIsNormalReversed;
    const float radius = data->sphere.mRadius;

    // compute quadratic sphere coefficients
    const float A = dot(rayDir, rayDir);
    const float B = dot(rayOrigin, rayDir);
    // Note all the 0.5 and 2 and 4 terms can (and have) been removed as they cancel
    const float C = dot(rayOrigin, rayOrigin) - radius * radius;
    // solve quadratic equation for t values
    const float D = B * B - A * C;
    if (D < 0.0f) {
        return;
    }
    const float rootDiscrim = sqrtf(D);
    float q = (B < 0.0f ? rootDiscrim : -rootDiscrim) - B;
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
    float3 pHit = rayOrigin + tHit * rayDir;

    float phi;
    if (pHit.x == 0.0f && pHit.y == 0.0f) {
        phi = 0.f;
    } else {
        phi = atan2f(pHit.y, pHit.x);
        if (phi < 0.0f) {
            phi += 2.f * M_PI;
        }
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
        pHit = rayOrigin + tHit * rayDir;

        float phi;
        if (pHit.x == 0.0f && pHit.y == 0.0f) {
            phi = 0.f;
        } else {
            phi = atan2f(pHit.y, pHit.x);
            if (phi < 0.0f) {
                phi += 2.f * M_PI;
            }
        }

        if (pHit.z < zMin || pHit.z > zMax || phi > phiMax) {
            return;
        }
    }

    optixReportIntersection(tHit, 0);
}
