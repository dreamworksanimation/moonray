// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

#ifndef __APPLE__
#define __APPLE__
#define __arm__
#endif

#define MLI_MAX_LEVELS_COUNT 16

// we use Constexpr if
#pragma clang diagnostic ignored "-Wc++17-extensions"

#include <metal_atomic>
#include <metal_pack>
#include <metal_stdlib>
#include <simd/simd.h>
using namespace metal;
using namespace metal::raytracing;

#define __host__
#define __device__
#define __global__ device
#define __gpu_device__ device
#define __gpu_thread__ thread
#define __constant__ constant

typedef intersection_function_table<curve_data, triangle_data, instancing, max_levels<MLI_MAX_LEVELS_COUNT>> ift_type_base;
typedef metal::raytracing::intersector<curve_data, triangle_data, instancing, max_levels<MLI_MAX_LEVELS_COUNT>> intersector_type_base;

typedef intersection_function_table<triangle_data, instancing, instance_motion> ift_type_im;
typedef metal::raytracing::intersector<triangle_data, instancing, instance_motion> intersector_type_im;

typedef intersection_function_table<triangle_data, instancing, primitive_motion> ift_type_pm;
typedef metal::raytracing::intersector<triangle_data, instancing, primitive_motion> intersector_type_pm;

typedef intersection_function_table<triangle_data, instancing, instance_motion, primitive_motion> ift_type_im_pm;
typedef metal::raytracing::intersector<triangle_data, instancing, instance_motion, primitive_motion> intersector_type_im_pm;

#include "MetalGPUMath.h"
#include "MetalGPUParams.h"
#include "../GPURay.h"
#include "MetalGPUSBTRecord.h"

using namespace moonray::rt;

/* For a bounding box intersection function. */
struct BoundingBoxIntersectionResult {
  bool accept [[accept_intersection]];
  bool continue_search [[continue_search]];
  float distance [[distance]];
};

/* For a triangle intersection function. */
struct TriangleIntersectionResult {
  bool accept [[accept_intersection]];
  bool continue_search [[continue_search]];
};

// Each occlusion ray has a PerRayData associated with it that is globally
// available in all of the programs.  Inputs and outputs are passed via this
// struct.

struct PerRayData
{
    int mShadowReceiverId;        // used for shadow linking (input)
    uint64_t mLightId;  // used for shadow linking (input)
    bool mDidHitGeom;             // did ray hit geometry? (output)
    // todo: intersect() results
};

// The __anyhit__ program is called for all intersections reported by optixReportIntersection().
// Here we can ignore intersections based on sidedness, visible shadows, shadow linking.
bool __anyhit__(const __gpu_device__ void* hgData,
                ray_data PerRayData &prd,
                bool isTriangleFrontFace,
                uint primIdx,
                uint instanceIdx)
{
    const HitGroupData __gpu_device__* data =
    (HitGroupData __gpu_device__*)hgData + instanceIdx;

    if (data->mIsSingleSided && !isTriangleFrontFace) {
        return false;
    }
    if (!data->mVisibleShadow) {
        return false;
    }

    // int assignmentId = data->mAssignmentIds[primIdx];
    // for (uint i = 0; i < data->mNumShadowLinkEntries; i++) {
    //     if (assignmentId == data->mShadowLinkAssignmentIds[i] &&
    //         prd.mLightId == data->mShadowLinkLightIds[i]) {

    //         return false;
    //     }

    unsigned int casterId = data->mAssignmentIds[primIdx];

    for (unsigned i = 0; i < data->mNumShadowLinkLights; i++) {
        if (casterId != data->mShadowLinkLights[i].mCasterId) {
            // entry doesn't apply to this caster id
            continue;
        }
        if (prd.mLightId == data->mShadowLinkLights[i].mLightId) {
            // if there is a match, the current object can't cast a shadow
            // from this specific light
            //optixIgnoreIntersection();
            return false;
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

        if (prd.mShadowReceiverId == data->mShadowLinkReceivers[i].mReceiverId) {
            // if there is a match, the current object can't cast a shadow
            // onto this receiver
            receiverMatches = true;
            break;
        }
    }

    if (receiverMatches ^ isComplemented) {
        //optixIgnoreIntersection();
        return false;


    }
    
    return true;
}

// The __raygen__ program sets up the PerRayData and then calls optixTrace()
// which starts the BVH traversal and calling of other programs.  It copies
// the prd.mIsOccluded result to the appropriate location in the output buffer.
// A __raygen__ function normally implements a camera.

template<typename accel_struct_type, typename ift_type, typename intersector_type, bool use_motion>
void raygen(constant MetalGPUParams &params,
            accel_struct_type accel,
            ift_type intersection_functions,
            const __gpu_device__ void* hgData,
            device float3 *debug_buffer,
            uint gridIndex)
{
    const uint idx = gridIndex;

    if (idx >= params.mNumRays) {
        return;
    }

    const moonray::rt::GPURay __gpu_device__ *ray = params.mRaysBuf + idx;
    const float __gpu_device__ *cpuRay =
        (const float __gpu_device__*)(params.mCPURays + (params.mCPURayStride * idx));

    PerRayData prd;
    prd.mShadowReceiverId = ray->mShadowReceiverId;
    prd.mLightId = ray->mLightId;

    typename intersector_type::result_type intersection;
    metal::raytracing::ray r(float3(cpuRay[0], cpuRay[1], cpuRay[2]),
                             float3(cpuRay[3], cpuRay[4], cpuRay[5]),
                             cpuRay[6], cpuRay[7]);
    intersector_type metalrt_intersect;
    metalrt_intersect.assume_geometry_type(geometry_type::curve | geometry_type::triangle | geometry_type::bounding_box);
    uint ray_mask = 255;
    if constexpr (use_motion) {
        intersection = metalrt_intersect.intersect(r,
                                                   accel,
                                                   ray_mask,
                                                   cpuRay[8],
                                                   intersection_functions,
                                                   prd);
    }
    else {
        intersection = metalrt_intersect.intersect(r,
                                                   accel,
                                                   ray_mask,
                                                   intersection_functions,
                                                   prd);
    }

    params.mIsOccludedBuf[idx] = intersection.type != intersection_type::none;
}

kernel
void __raygen__base(constant MetalGPUParams &params [[buffer(0)]],
                    acceleration_structure<instancing> accel [[buffer(1)]],
                    ift_type_base intersection_functions [[buffer(2)]],
                    const __gpu_device__ void* hgData [[buffer(3)]],
                    device float3 *debug_buffer [[buffer(4)]],
                    uint gridIndex [[thread_position_in_grid]])
{
    raygen<acceleration_structure<instancing>,
           ift_type_base,
           intersector_type_base,
           false>
         (params, accel, intersection_functions, hgData, debug_buffer, gridIndex);
}

kernel
void __raygen__im(constant MetalGPUParams &params [[buffer(0)]],
                  acceleration_structure<instancing, instance_motion> accel [[buffer(1)]],
                  ift_type_im intersection_functions [[buffer(2)]],
                  const __gpu_device__ void* hgData [[buffer(3)]],
                  device float3 *debug_buffer [[buffer(4)]],
                  uint gridIndex [[thread_position_in_grid]])
{
    raygen<acceleration_structure<instancing, instance_motion>,
           ift_type_im,
           intersector_type_im,
           true>
         (params, accel, intersection_functions, hgData, debug_buffer, gridIndex);
}

kernel
void __raygen__pm(constant MetalGPUParams &params [[buffer(0)]],
                  acceleration_structure<instancing, primitive_motion> accel [[buffer(1)]],
                  ift_type_pm intersection_functions [[buffer(2)]],
                  const __gpu_device__ void* hgData [[buffer(3)]],
                  device float3 *debug_buffer [[buffer(4)]],
                  uint gridIndex [[thread_position_in_grid]])
{
    raygen<acceleration_structure<instancing, primitive_motion>,
           ift_type_pm,
           intersector_type_pm,
           true>
         (params, accel, intersection_functions, hgData, debug_buffer, gridIndex);
}

kernel
void __raygen__im_pm(constant MetalGPUParams &params [[buffer(0)]],
                     acceleration_structure<instancing, instance_motion, primitive_motion> accel [[buffer(1)]],
                     ift_type_im_pm intersection_functions [[buffer(2)]],
                     const __gpu_device__ void* hgData [[buffer(3)]],
                     device float3 *debug_buffer [[buffer(4)]],
                     uint gridIndex [[thread_position_in_grid]])
{
    raygen<acceleration_structure<instancing, instance_motion, primitive_motion>,
           ift_type_im_pm,
           intersector_type_im_pm,
           true>
         (params, accel, intersection_functions, hgData, debug_buffer, gridIndex);
}

// First, we have the ray __intersection__ program for the triangle primitives.
// Closely adapted from the existing Moonray code or from Embree 3.9.


// Triangles

[[intersection(triangle, triangle_data, instancing, max_levels<MLI_MAX_LEVELS_COUNT>)]]
TriangleIntersectionResult
__intersection__triangle(const __gpu_device__ void* hgData [[buffer(0)]],
                         ray_data PerRayData &prd [[payload]],
                         bool isTriangleFrontFace [[front_facing]],
                         uint primIdx [[primitive_id]],
                         array_ref<uint> instanceIdx [[user_instance_id]])
{
    TriangleIntersectionResult result;
    result.continue_search = false;

    result.accept = __anyhit__(hgData, prd, isTriangleFrontFace, primIdx, instanceIdx[instanceIdx.size() - 1]);

    return result;
}

[[intersection(triangle, triangle_data, instancing, max_levels<MLI_MAX_LEVELS_COUNT>, instance_motion)]]
TriangleIntersectionResult
__intersection__triangle_im(const __gpu_device__ void* hgData [[buffer(0)]],
                            ray_data PerRayData &prd [[payload]],
                            bool isTriangleFrontFace [[front_facing]],
                            uint primIdx [[primitive_id]],
                            array_ref<uint> instanceIdx [[user_instance_id]])
{
    TriangleIntersectionResult result;
    result.continue_search = false;

    result.accept = __anyhit__(hgData, prd, isTriangleFrontFace, primIdx, instanceIdx[instanceIdx.size() - 1]);
    
    return result;
}


[[intersection(triangle, triangle_data, instancing, max_levels<MLI_MAX_LEVELS_COUNT>, primitive_motion)]]
TriangleIntersectionResult
__intersection__triangle_pm(const __gpu_device__ void* hgData [[buffer(0)]],
                            ray_data PerRayData &prd [[payload]],
                            bool isTriangleFrontFace [[front_facing]],
                            uint primIdx [[primitive_id]],
                            array_ref<uint> instanceIdx [[user_instance_id]])
{
    TriangleIntersectionResult result;
    result.continue_search = false;

    result.accept = __anyhit__(hgData, prd, isTriangleFrontFace, primIdx, instanceIdx[instanceIdx.size() - 1]);
    
    return result;
}

[[intersection(triangle, triangle_data, instancing, max_levels<MLI_MAX_LEVELS_COUNT>, instance_motion, primitive_motion)]]
TriangleIntersectionResult
__intersection__triangle_im_pm(const __gpu_device__ void* hgData [[buffer(0)]],
                               ray_data PerRayData &prd [[payload]],
                               bool isTriangleFrontFace [[front_facing]],
                               uint primIdx [[primitive_id]],
                               array_ref<uint> instanceIdx [[user_instance_id]])
{
    TriangleIntersectionResult result;
    result.continue_search = false;
    result.accept = __anyhit__(hgData, prd, isTriangleFrontFace, primIdx, instanceIdx[instanceIdx.size() - 1]);
    return result;
}


// Next, we have all of the ray __intersection__ programs for the Custom Primitives.
// These are closely adapted from the existing Moonray code or from Embree 3.9.


// Boxes

BoundingBoxIntersectionResult
intersectionBox(const __gpu_device__ void* hgData,
                ray_data PerRayData &prd,
                uint primIdx,
                const float3 rayOriginIn,
                const float3 rayDirIn,
                const float rayTnear,
                const float rayTfar,
                const uint instanceIdx)
{
    // See: lib/rendering/geom/prim/Box.cc::intersectFunc()
    BoundingBoxIntersectionResult result;
    result.continue_search = false;
    
    const HitGroupData __gpu_device__* data =
        (const HitGroupData __gpu_device__*)hgData + instanceIdx;
    const float xLength = data->u.box.mLength;
    const float yHeight = data->u.box.mHeight;
    const float zWidth = data->u.box.mWidth;

    const float3 rayOrigin = data->u.sphere.mP2L.transformPoint(rayOriginIn);
    const float3 rayDir = data->u.sphere.mP2L.transformVector(rayDirIn);

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
            result.accept = false;
            return result;
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
            result.accept = false;
            return result;
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
            result.accept = false;
            return result;
        }
    }

    if (t0 > rayTfar || t1 < rayTnear) {
        result.accept = false;
        return result;
    }
    float tHit = t0;
    if (t0 < rayTnear || (isSingleSided && isNormalReversed)) {
        tHit = t1;
        if (tHit > rayTfar || (isSingleSided && !isNormalReversed)) {
            result.accept = false;
            return result;
        }
    }

    result.accept = __anyhit__(hgData, prd, true, primIdx, instanceIdx);
    result.distance = tHit;
    return result;
}

[[intersection(bounding_box, triangle_data, instancing, max_levels<MLI_MAX_LEVELS_COUNT>)]]
BoundingBoxIntersectionResult
__intersection__box(const __gpu_device__ void* hgData [[buffer(0)]],
                    ray_data PerRayData &prd [[payload]],
                    uint primIdx [[primitive_id]],
                    const float3 rayOriginIn [[origin]],
                    const float3 rayDirIn [[direction]],
                    const float rayTnear [[min_distance]],
                    const float rayTfar [[max_distance]],
                    array_ref<uint> instanceIdx [[user_instance_id]])
{
    return intersectionBox(hgData, prd, primIdx, rayOriginIn, rayDirIn, rayTnear, rayTfar, instanceIdx[instanceIdx.size() - 1]);
}

[[intersection(bounding_box, triangle_data, instancing, max_levels<MLI_MAX_LEVELS_COUNT>, instance_motion)]]
BoundingBoxIntersectionResult
__intersection__box_im(const __gpu_device__ void* hgData [[buffer(0)]],
                       ray_data PerRayData &prd [[payload]],
                       uint primIdx [[primitive_id]],
                       const float3 rayOriginIn [[origin]],
                       const float3 rayDirIn [[direction]],
                       const float rayTnear [[min_distance]],
                       const float rayTfar [[max_distance]],
                       array_ref<uint> instanceIdx [[user_instance_id]])
{
    return intersectionBox(hgData, prd, primIdx, rayOriginIn, rayDirIn, rayTnear, rayTfar, instanceIdx[instanceIdx.size() - 1]);
}

[[intersection(bounding_box, triangle_data, instancing, max_levels<MLI_MAX_LEVELS_COUNT>, primitive_motion)]]
BoundingBoxIntersectionResult
__intersection__box_pm(const __gpu_device__ void* hgData [[buffer(0)]],
                       ray_data PerRayData &prd [[payload]],
                       uint primIdx [[primitive_id]],
                       const float3 rayOriginIn [[origin]],
                       const float3 rayDirIn [[direction]],
                       const float rayTnear [[min_distance]],
                       const float rayTfar [[max_distance]],
                       array_ref<uint> instanceIdx [[user_instance_id]])
{
    return intersectionBox(hgData, prd, primIdx, rayOriginIn, rayDirIn, rayTnear, rayTfar, instanceIdx[instanceIdx.size() - 1]);
}

[[intersection(bounding_box, triangle_data, instancing,  max_levels<MLI_MAX_LEVELS_COUNT>, instance_motion, primitive_motion)]]
BoundingBoxIntersectionResult
__intersection__box_im_pm(const __gpu_device__ void* hgData [[buffer(0)]],
                          ray_data PerRayData &prd [[payload]],
                          uint primIdx [[primitive_id]],
                          const float3 rayOriginIn [[origin]],
                          const float3 rayDirIn [[direction]],
                          const float rayTnear [[min_distance]],
                          const float rayTfar [[max_distance]],
                          array_ref<uint> instanceIdx [[user_instance_id]])
{
    return intersectionBox(hgData, prd, primIdx, rayOriginIn, rayDirIn, rayTnear, rayTfar, instanceIdx[instanceIdx.size() - 1]);
}


// Points

BoundingBoxIntersectionResult
intersectionPoints(__gpu_device__ void* hgData [[buffer(0)]],
                   ray_data PerRayData &prd,
                   const unsigned int primIdx,
                   const float3 rayOrg,
                   const float3 rayDir,
                   const float rayTmin,
                   const float rayTmax,
                   const uint instanceIdx,
                   const float time)
{
    BoundingBoxIntersectionResult result;
    result.continue_search = false;
    result.accept = false;
    // See: lib/rendering/geom/prim/Points.cc::intersectFunc()

    const HitGroupData __gpu_device__* data =
        (const HitGroupData __gpu_device__*)hgData + instanceIdx;

    const int motionSamplesCount = data->u.points.mMotionSamplesCount;
    float4 p4;
    if (motionSamplesCount == 1) {
        p4 = data->u.points.mPoints[primIdx]; // xyzr
    } else {
        const float idxPlusT = time * (motionSamplesCount - 1);
        const unsigned int idx0 = static_cast<unsigned int>(idxPlusT);
        const unsigned int idx1 = min(idx0 + 1u, motionSamplesCount - 1u); // clamp to time = 1
        const float t = idxPlusT - static_cast<float>(idx0);
        p4 = lerp(data->u.points.mPoints[primIdx * motionSamplesCount + idx0],
                  data->u.points.mPoints[primIdx * motionSamplesCount + idx1], t);
    }

    const float3 p = make_float3(p4);
    const float r = p4.w;
    const float r2 = r * r;

    const float3 v = p - rayOrg;
    if (dot(rayDir, v) < 0.f) {
        // ray is travelling away from sphere centre
        return result;
    }
    if (dot(v, v) < r2) {
        // ray origin is inside sphere
        return result;
    }

    const float3 dxv = cross(rayDir, v);
    const float d2 = dot(rayDir, rayDir);
    const float D = d2 * r2 - dot(dxv, dxv);
    if (D < 0.f) {
        // no intersections
        return result;
    }

    const float Q = sqrtf(D);
    const float3 dxdxv = cross(rayDir, dxv);
    float3 Ng = (dxdxv - Q * rayDir) / d2;
    const float t = length(v + Ng);
    if (rayTmin < t && t < rayTmax) {
        result.distance = t;
        result.accept = __anyhit__(hgData, prd, true, primIdx, instanceIdx);
    }
    
    return result;
}

[[intersection(bounding_box, triangle_data, instancing, max_levels<MLI_MAX_LEVELS_COUNT>)]]
BoundingBoxIntersectionResult
__intersection__points(__gpu_device__ void* hgData [[buffer(0)]],
                       ray_data PerRayData &prd [[payload]],
                       const unsigned int primIdx [[primitive_id]],
                       const float3 rayOrg [[origin]],
                       const float3 rayDir [[direction]],
                       const float rayTmin [[min_distance]],
                       const float rayTmax [[max_distance]],
                       array_ref<uint> instanceIdx [[user_instance_id]])
{
    return intersectionPoints(hgData, prd, primIdx, rayOrg, rayDir, rayTmin, rayTmax, instanceIdx[instanceIdx.size() - 1], 0.0f);
}

[[intersection(bounding_box, triangle_data, instancing, max_levels<MLI_MAX_LEVELS_COUNT>, instance_motion)]]
BoundingBoxIntersectionResult
__intersection__points_im(__gpu_device__ void* hgData [[buffer(0)]],
                          ray_data PerRayData &prd [[payload]],
                          const unsigned int primIdx [[primitive_id]],
                          const float3 rayOrg [[origin]],
                          const float3 rayDir [[direction]],
                          const float rayTmin [[min_distance]],
                          const float rayTmax [[max_distance]],
                          array_ref<uint> instanceIdx [[user_instance_id]])
{
    return intersectionPoints(hgData, prd, primIdx, rayOrg, rayDir, rayTmin, rayTmax, instanceIdx[instanceIdx.size() - 1], 0.0f);
}

[[intersection(bounding_box, triangle_data, instancing, max_levels<MLI_MAX_LEVELS_COUNT>, primitive_motion)]]
BoundingBoxIntersectionResult
__intersection__points_pm(__gpu_device__ void* hgData [[buffer(0)]],
                          ray_data PerRayData &prd [[payload]],
                          const unsigned int primIdx [[primitive_id]],
                          const float3 rayOrg [[origin]],
                          const float3 rayDir [[direction]],
                          const float rayTmin [[min_distance]],
                          const float rayTmax [[max_distance]],
                          array_ref<uint> instanceIdx [[user_instance_id]],
                          const float time [[time]])
{
    return intersectionPoints(hgData, prd, primIdx, rayOrg, rayDir, rayTmin, rayTmax, instanceIdx[instanceIdx.size() - 1], time);
}

[[intersection(bounding_box, triangle_data, instancing, max_levels<MLI_MAX_LEVELS_COUNT>, instance_motion, primitive_motion)]]
BoundingBoxIntersectionResult
__intersection__points_im_pm(__gpu_device__ void* hgData [[buffer(0)]],
                             ray_data PerRayData &prd [[payload]],
                             const unsigned int primIdx [[primitive_id]],
                             const float3 rayOrg [[origin]],
                             const float3 rayDir [[direction]],
                             const float rayTmin [[min_distance]],
                             const float rayTmax [[max_distance]],
                             array_ref<uint> instanceIdx [[user_instance_id]],
                             const float time [[time]])
{
    return intersectionPoints(hgData, prd, primIdx, rayOrg, rayDir, rayTmin, rayTmax, instanceIdx[instanceIdx.size() - 1], time);
}


// Spheres

BoundingBoxIntersectionResult
intersectionSphere(__gpu_device__ void* hgData,
                   ray_data PerRayData &prd,
                   const unsigned int primIdx,
                   const float3 rayOrginIn,
                   const float3 rayDirIn,
                   const float rayTmin,
                   const float rayTmax,
                   const uint instanceIdx,
                   const float time)
{
    BoundingBoxIntersectionResult result;
    result.continue_search = false;
    result.accept = false;

    // See: lib/rendering/geom/prim/Sphere.cc::intersectFunc()

    const HitGroupData __gpu_device__* data =
        (const HitGroupData __gpu_device__*)hgData + instanceIdx;

    const float3 rayOrg = data->u.sphere.mP2L.transformPoint(rayOrginIn);
    const float3 rayDir = data->u.sphere.mP2L.transformVector(rayDirIn);

    const bool isSingleSided = data->mIsSingleSided;
    const bool isNormalReversed = data->mIsNormalReversed;
    const float radius = data->u.sphere.mRadius;

    // compute quadratic sphere coefficients
    const float A = dot(rayDir, rayDir);
    const float B = dot(rayOrg, rayDir);
    // Note all the 0.5 and 2 and 4 terms can (and have) been removed as they cancel
    const float C = dot(rayOrg, rayOrg) - radius * radius;
    // solve quadratic equation for t values
    const float D = B * B - A * C;
    if (D < 0.0f) {
        return result;
    }
    const float rootDiscrim = sqrtf(D);
    float q = (B < 0.0f ? rootDiscrim : -rootDiscrim) - B;
    float t0 = q / A;
    float t1 = C / q;
    if (t0 > t1) {
        swapf(t0, t1);
    }

    // compute intersection distance along ray
    if (t0 > rayTmax || t1 < rayTmin) {
        return result;
    }
    float tHit = t0;
    if (t0 < rayTmin || (isSingleSided && isNormalReversed)) {
        tHit = t1;
        if (tHit > rayTmax || (isSingleSided && !isNormalReversed)) {
            return result;
        }
    }
    // compute hit position and phi
    float3 pHit = rayOrg + tHit * rayDir;

    float phi;
    if (pHit.x == 0.0f && pHit.y == 0.0f) {
        phi = 0.f;
    } else {
        phi = atan2f(pHit.y, pHit.x);
        if (phi < 0.0f) {
            phi += 2.f * M_PI_F;
        }
    }

    float zMin = data->u.sphere.mZMin;
    float zMax = data->u.sphere.mZMax;
    float phiMax = data->u.sphere.mPhiMax;
    // test sphere intersection against clipping parameters
    if (pHit.z < zMin || pHit.z > zMax || phi > phiMax) {
        if (tHit == t1) {
            return result;
        }
        if (t1 > rayTmax || (isSingleSided && !isNormalReversed)) {
            return result;
        }
        tHit = t1;
        pHit = rayOrg + tHit * rayDir;

        float phi;
        if (pHit.x == 0.0f && pHit.y == 0.0f) {
            phi = 0.f;
        } else {
            phi = atan2f(pHit.y, pHit.x);
            if (phi < 0.0f) {
                phi += 2.f * M_PI_F;
            }
        }

        if (pHit.z < zMin || pHit.z > zMax || phi > phiMax) {
            return result;
        }
    }

    result.distance = tHit;
    result.accept = __anyhit__(hgData, prd, true, primIdx, instanceIdx);
    
    return result;
}

[[intersection(bounding_box, triangle_data, instancing, max_levels<MLI_MAX_LEVELS_COUNT>)]]
BoundingBoxIntersectionResult
__intersection__sphere(__gpu_device__ void* hgData [[buffer(0)]],
                       ray_data PerRayData &prd [[payload]],
                       const unsigned int primIdx [[primitive_id]],
                       const float3 rayOriginIn [[origin]],
                       const float3 rayDirIn [[direction]],
                       const float rayTmin [[min_distance]],
                       const float rayTmax [[max_distance]],
                       array_ref<uint> instanceIdx [[user_instance_id]])
{
    return intersectionSphere(hgData, prd, primIdx, rayOriginIn, rayDirIn, rayTmin, rayTmax, instanceIdx[instanceIdx.size() - 1], 0.0f);
}

[[intersection(bounding_box, triangle_data, instancing, max_levels<MLI_MAX_LEVELS_COUNT>, instance_motion)]]
BoundingBoxIntersectionResult
__intersection__sphere_im(__gpu_device__ void* hgData [[buffer(0)]],
                          ray_data PerRayData &prd [[payload]],
                          const unsigned int primIdx [[primitive_id]],
                          const float3 rayOriginIn [[origin]],
                          const float3 rayDirIn [[direction]],
                          const float rayTmin [[min_distance]],
                          const float rayTmax [[max_distance]],
                          array_ref<uint> instanceIdx [[user_instance_id]])
{
    return intersectionSphere(hgData, prd, primIdx, rayOriginIn, rayDirIn, rayTmin, rayTmax, instanceIdx[instanceIdx.size() - 1], 0.0f);
}

[[intersection(bounding_box, triangle_data, instancing, max_levels<MLI_MAX_LEVELS_COUNT>, primitive_motion)]]
BoundingBoxIntersectionResult
__intersection__sphere_pm(__gpu_device__ void* hgData [[buffer(0)]],
                          ray_data PerRayData &prd [[payload]],
                          const unsigned int primIdx [[primitive_id]],
                          const float3 rayOriginIn [[origin]],
                          const float3 rayDirIn [[direction]],
                          const float rayTmin [[min_distance]],
                          const float rayTmax [[max_distance]],
                          array_ref<uint> instanceIdx [[user_instance_id]],
                          const float time [[time]])
{
    return intersectionSphere(hgData, prd, primIdx, rayOriginIn, rayDirIn, rayTmin, rayTmax, instanceIdx[instanceIdx.size() - 1], time);
}

[[intersection(bounding_box, triangle_data, instancing, max_levels<MLI_MAX_LEVELS_COUNT>, instance_motion, primitive_motion)]]
BoundingBoxIntersectionResult
__intersection__sphere_im_pm(__gpu_device__ void* hgData [[buffer(0)]],
                             ray_data PerRayData &prd [[payload]],
                             const unsigned int primIdx [[primitive_id]],
                             const float3 rayOriginIn [[origin]],
                             const float3 rayDirIn [[direction]],
                             const float rayTmin [[min_distance]],
                             const float rayTmax [[max_distance]],
                             array_ref<uint> instanceIdx [[user_instance_id]],
                             const float time [[time]])
{
    return intersectionSphere(hgData, prd, primIdx, rayOriginIn, rayDirIn, rayTmin, rayTmax, instanceIdx[instanceIdx.size() - 1], time);
}
