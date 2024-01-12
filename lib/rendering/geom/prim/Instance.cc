// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

///
/// @file Instance.cc
/// $Id$
///

#include "Instance.h"

#include <moonray/rendering/geom/prim/BVHUserData.h>
#include <moonray/rendering/geom/prim/GeomTLState.h>
#include <moonray/rendering/geom/prim/PrimitivePrivateAccess.h>
#include <moonray/rendering/geom/prim/VolumeAssignmentTable.h>

#include <scene_rdl2/common/math/ispc/Typesv.h>
#include <scene_rdl2/common/math/ispc/Xformv.h>

namespace moonray {
namespace geom {
namespace internal {

static void
boundsFunc(const RTCBoundsFunctionArguments* args)
{
    const BVHUserData* userData = (const BVHUserData*)args->geometryUserPtr;
    const Instance* instance = (const Instance*)userData->mPrimitive;
    BBox3f bound = instance->computeAABB();
    RTCBounds* output = args->bounds_o;
    output->lower_x = bound.lower.x;
    output->lower_y = bound.lower.y;
    output->lower_z = bound.lower.z;
    output->upper_x = bound.upper.x;
    output->upper_y = bound.upper.y;
    output->upper_z = bound.upper.z;
}

RTCBoundsFunction
Instance::getBoundsFunction() const
{
    return &boundsFunc;
}

// Intersection is a bit tricky with instancing.
// We need to track if there's a new intersection found at each instance level.
// So we need to store the original ray state before trace new ray,
// set ray.geomID to invalid, which will be assigned to a valid geom id
// when a new intersection is found.
static void
intersectFunc(const RTCIntersectFunctionNArguments* args)
{
    int* valid = (int*)args->valid;
    unsigned int N = args->N;
    RTCRayHitN* rayhit = (RTCRayHitN*)args->rayhit;
    RTCRayN* rays = RTCRayHitN_RayN(rayhit, N);
    RTCHitN* hits = RTCRayHitN_HitN(rayhit, N);
    const BVHUserData* userData = (const BVHUserData*)args->geometryUserPtr;
    const Instance* instance = (const Instance*)userData->mPrimitive;
    const MotionTransform& xform = instance->getLocal2Parent();
    const RTCScene referenceScene = instance->getReferenceScene();
    int geomID = instance->getGeomID();
    mcrt_common::IntersectContext* context =
        (mcrt_common::IntersectContext*)args->context;
    for (unsigned int index = 0; index < N; ++index) {
        int rayId = RTCRayN_id(rays, N, index);
        mcrt_common::RayExtension& rayExtension =
            context->mRayExtension[rayId];
        if (valid[index] == 0) {
            continue;
        }
        const Mat43 inL2R = rayExtension.l2r;
        void* inUserData = rayExtension.userData;
        const void* inInstance0 = rayExtension.instance0OrLight;
        const void* inInstance1 = rayExtension.instance1;
        const void* inInstance2 = rayExtension.instance2;
        const void* inInstance3 = rayExtension.instance3;
        const int inInstanceAttributesDepth = rayExtension.instanceAttributesDepth;
        const int inVolumeInstanceState = rayExtension.volumeInstanceState;
        
        float time = RTCRayN_time(rays, N, index);
        Mat43 L2P, P2L;
        if (xform.isStatic()) {
            L2P = xform.getStaticXform();
            P2L = xform.getStaticInverse();
        } else {
            L2P = xform.eval(time);
            P2L = L2P.inverse();
        }

        Vec3f localOrg = scene_rdl2::math::transformPoint(P2L, Vec3f(
            RTCRayN_org_x(rays, N, index),
            RTCRayN_org_y(rays, N, index),
            RTCRayN_org_z(rays, N, index)));
        Vec3f localDir = scene_rdl2::math::transformVector(P2L, Vec3f(
            RTCRayN_dir_x(rays, N, index),
            RTCRayN_dir_y(rays, N, index),
            RTCRayN_dir_z(rays, N, index)));

        RTCRayHit localRay;
        localRay.ray.org_x = localOrg.x;
        localRay.ray.org_y = localOrg.y;
        localRay.ray.org_z = localOrg.z;
        localRay.ray.tnear = RTCRayN_tnear(rays, N, index);
        localRay.ray.dir_x = localDir.x;
        localRay.ray.dir_y = localDir.y;
        localRay.ray.dir_z = localDir.z;
        localRay.ray.time  = time;
        localRay.ray.tfar  = RTCRayN_tfar(rays, N, index);
        localRay.ray.mask  = RTCRayN_mask(rays, N, index);
        localRay.ray.id    = rayId;
        localRay.hit.geomID = RTC_INVALID_GEOMETRY_ID;
        localRay.hit.instID[0] = RTC_INVALID_GEOMETRY_ID;
        rayExtension.l2r = Mat43(scene_rdl2::math::one);
        rayExtension.userData = nullptr;
        // Clear instance pointers at and beyond the current depth
        memset(&rayExtension.instance0OrLight + rayExtension.instanceAttributesDepth, 0,
           (Instance::sMaxInstanceAttributesDepth - rayExtension.instanceAttributesDepth) * sizeof(void*));

        // Increase the instance attribute depth so we know
        // which instanceN member to set if the ray intersects the instance.
        // This intersectFunc is called recursively with multi-level instancing.
        // Note that we only increment the depth if we are at the top level or
        // the instance has attributes.  We always need to set the top level
        // instance for motion vector support.
        bool topLevel = (rayExtension.instanceAttributesDepth == 0);
        if (topLevel || instance->getAttributes()) {
            rayExtension.instanceAttributesDepth++;
        }

        // volume instance ids
        // top level instance primitives are initialized with state = 0
        // rayExtension.volumeInstanceState will be >= 0 if the volume instance
        // state is valid, -1 otherwise (invalid transition, or not a volume instance)
        // currently, only volume rays have a non-null geomTls
        const int state = topLevel ? 0 : rayExtension.volumeInstanceState;
        const geom::internal::TLState *geomTls = static_cast<geom::internal::TLState *>(rayExtension.geomTls);
        if (state >= 0 && geomTls) {
            MNRY_ASSERT(geomTls->mVolumeRayState.getVolumeAssignmentTable() != nullptr);
            const geom::internal::VolumeIdFSM &fsm =
                geomTls->mVolumeRayState.getVolumeAssignmentTable()->getInstanceVolumeIds();
            rayExtension.volumeInstanceState = fsm.transition(state, instance);
        }

        RTCIntersectArguments args;
        rtcInitIntersectArguments(&args);
        args.context = &(context->mRtcContext);

        rtcIntersect1(referenceScene, &localRay, &args);

        if (localRay.hit.geomID == RTC_INVALID_GEOMETRY_ID) {
            // no new intersection found, restore original intersection state
            rayExtension.userData = inUserData;
            rayExtension.l2r = inL2R;
            // we restore the instanceAttributesDepth below
            rayExtension.instance0OrLight = inInstance0;
            rayExtension.instance1 = inInstance1;
            rayExtension.instance2 = inInstance2;
            rayExtension.instance3 = inInstance3;
        } else {
            // intersection is found, update the intersection state
            if (localRay.hit.instID[0] != localRay.hit.geomID) {
                // ray intersect a leaf Primitive
                rayExtension.userData = rtcGetGeometryUserData(
                    rtcGetGeometry(referenceScene, localRay.hit.geomID));
            }
            // the way we use instID is different from how embree's official
            // single level instance geometry. If the ray hits an instance,
            // we mark the instID = geomID. We use this mechanic to identify
            // whether the ray hit an instance or not after rtcIntersect call
            localRay.hit.geomID = geomID;
            localRay.hit.instID[0] = geomID;
            // we found a hit, concat scene transform to ray l2r
            RTCRayN_tfar(rays, N, index) = localRay.ray.tfar;
            rayExtension.l2r *= L2P;
            rtcCopyHitToHitN(hits, &localRay.hit, N, index);
            switch (rayExtension.instanceAttributesDepth) {
            case 1:
                if (topLevel) {
                    rayExtension.instance0OrLight = instance;
                }
                break;
            case 2:
                if (instance->getAttributes()) {
                    rayExtension.instance1 = instance;
                }
                break;
            case 3:
                if (instance->getAttributes()) {
                    rayExtension.instance2 = instance;
                }
                break;
            case 4:
                if (instance->getAttributes()) {
                    rayExtension.instance3 = instance;
                }
                break;
            default:
                // We ignore instances more than sMaxInstanceDepth deep.
                // This means prim attrs are ignored on those instances.
                break;
            }
        }
        // restore the previous instance depth before the recursion unwinds
        rayExtension.instanceAttributesDepth = inInstanceAttributesDepth;
        // restore volume instance state
        rayExtension.volumeInstanceState = inVolumeInstanceState;
    }
}

RTCIntersectFunctionN
Instance::getIntersectFunction() const
{
    return &intersectFunc;
}

static void
occludedFunc(const RTCOccludedFunctionNArguments* args)
{
    int* valid = (int*)args->valid;
    unsigned int N = args->N;
    RTCRayN* rays = args->ray;
    const BVHUserData* userData = (const BVHUserData*)args->geometryUserPtr;
    const Instance* instance = (const Instance*)userData->mPrimitive;
    const MotionTransform& xform = instance->getLocal2Parent();
    const RTCScene referenceScene = instance->getReferenceScene();
    mcrt_common::IntersectContext* context =
        (mcrt_common::IntersectContext*)args->context;
    for (unsigned int index = 0; index < N; ++index) {
        if (valid[index] == 0) {
            continue;
        }
        Vec3f localOrg;
        Vec3f localDir;
        float time = RTCRayN_time(rays, N, index);
        const Mat43 P2L = xform.isStatic() ?
            xform.getStaticInverse() : xform.eval(time).inverse();
        // transform ray to local space
        localOrg = scene_rdl2::math::transformPoint(P2L, Vec3f(
            RTCRayN_org_x(rays, N, index),
            RTCRayN_org_y(rays, N, index),
            RTCRayN_org_z(rays, N, index)));
        localDir = scene_rdl2::math::transformVector(P2L, Vec3f(
            RTCRayN_dir_x(rays, N, index),
            RTCRayN_dir_y(rays, N, index),
            RTCRayN_dir_z(rays, N, index)));
        RTCRay localRay;
        localRay.org_x = localOrg.x;
        localRay.org_y = localOrg.y;
        localRay.org_z = localOrg.z;
        localRay.tnear = RTCRayN_tnear(rays, N, index);
        localRay.dir_x = localDir.x;
        localRay.dir_y = localDir.y;
        localRay.dir_z = localDir.z;
        localRay.time  = time;
        localRay.tfar  = RTCRayN_tfar(rays, N, index);
        localRay.mask  = RTCRayN_mask(rays, N, index);
        localRay.id    = RTCRayN_id(rays, N, index);
        context->mRtcContext.instID[0] = instance->getGeomID();

        RTCOccludedArguments oargs;
        rtcInitOccludedArguments(&oargs);
        oargs.context = &(context->mRtcContext);

        rtcOccluded1(referenceScene, &localRay, &oargs);

        context->mRtcContext.instID[0] = RTC_INVALID_GEOMETRY_ID;
        RTCRayN_tfar(rays, N, index) = localRay.tfar;
    }
}

RTCOccludedFunctionN
Instance::getOccludedFunction() const
{
    return &occludedFunc;
}

BBox3f
Instance::computeAABB() const
{
    const RTCScene& referenceScene = getReferenceScene();
    const MotionTransform& local2Parent = getLocal2Parent();
    RTCBounds refBound;
    rtcGetSceneBounds(referenceScene, &refBound);
    BBox3f localBound;
    localBound.lower = Vec3f(refBound.lower_x, refBound.lower_y, refBound.lower_z);
    localBound.upper = Vec3f(refBound.upper_x, refBound.upper_y, refBound.upper_z);
    // Instance with empty source scene (usually due to incorrect user setup)
    // Make it a valid (but meaningless) bounding box so it won't break the
    // partition process during BVH construction
    if (!scene_rdl2::math::isFinite(localBound.lower) ||
        !scene_rdl2::math::isFinite(localBound.upper)) {
        return BBox3f(0.0f);
    }

    // MOONRAY-4472
    // Handle case where the local2Parent MotionTransform has not been
    // initialized to prevent corrupting the BVH
    if (!local2Parent.isInitialized()) {
        return localBound;
    }

    BBox3f bound;
    if (local2Parent.isStatic()) {
        bound = transformBounds(local2Parent.getStaticXform(), localBound);
    } else {
        // TODO this sample based bounding box calculation may generate
        // insufficiently conservative result(though in practice it works well)
        // pbrt-v3 has an implementation that formulating bbox corner movement
        // as time space function and evaluate the derivative zero point.
        // Might worth investigating and port it in the future
        bound = merge(
            transformBounds(local2Parent.eval(0), localBound),
            transformBounds(local2Parent.eval(1), localBound));
        constexpr size_t sampleCount = 128;
        const float sampleDelta = 1.0f / sampleCount;
        for (size_t i = 1; i < sampleCount; ++i) {
            float t = i * sampleDelta;
            bound = merge(bound,
                transformBounds(local2Parent.eval(t), localBound));
        }
    }
    return bound;
}

BBox3f
Instance::computeAABBAtTimeStep(int timeStep) const
{
    const RTCScene& referenceScene = getReferenceScene();
    const MotionTransform& local2Parent = getLocal2Parent();
    RTCBounds refBound;
    rtcGetSceneBounds(referenceScene, &refBound);
    BBox3f localBound;
    localBound.lower = Vec3f(refBound.lower_x, refBound.lower_y, refBound.lower_z);
    localBound.upper = Vec3f(refBound.upper_x, refBound.upper_y, refBound.upper_z);
    // Instance with empty source scene (usually due to incorrect user setup)
    // Make it a valid (but meaningless) bounding box so it won't break the
    // partition process during BVH construction
    if (!scene_rdl2::math::isFinite(localBound.lower) ||
        !scene_rdl2::math::isFinite(localBound.upper)) {
        return BBox3f(0.0f);
    }

    if (local2Parent.isStatic()) {
        return transformBounds(local2Parent.getStaticXform(), localBound);
    } else {
        MNRY_ASSERT(timeStep >= 0 && timeStep < static_cast<int>(getMotionSamplesCount()), "timeStep out of range");
        return transformBounds(local2Parent.eval(timeStep), localBound);
    }
}

bool
Instance::hasAssignment(int assignmentId) const
{
    const auto& pImpl = PrimitivePrivateAccess::getPrimitiveImpl(
        getReference()->getPrimitive().get());
    return pImpl->hasAssignment(assignmentId);
}

} // namespace internal
} // namespace geom
} // namespace rendering


