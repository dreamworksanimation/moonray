// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

#include <moonray/rendering/geom/prim/NamedPrimitive.h>

#include <moonray/rendering/bvh/shading/RootShader.h>
#include <moonray/rendering/geom/prim/BVHUserData.h>
#include <moonray/rendering/geom/prim/GeomTLState.h>
#include <moonray/rendering/geom/prim/VdbVolume.h>

namespace moonray {
namespace rt {

using namespace scene_rdl2::math;

bool
originVolumeTest(uint32_t threadIdx,
                 int volumeId, const RTCRay &ray, geom::internal::VdbVolume *vdbVolume)
{
    Vec3f rayOrg(ray.org_x, ray.org_y, ray.org_z);
    Vec3f rayDir(ray.dir_x, ray.dir_y, ray.dir_z);
    Vec3f pStart = rayOrg + rayDir * ray.tnear;
    if (vdbVolume->isInActiveField(threadIdx, pStart, ray.time)) {
        return true; // ray origin is inside the active volume. We consider this volume as originVolume.
    }

    //
    // queryIntersections() returns all possible intersection (= volume interval) with active volumes.
    // However, in some cases, volume is not active at the front-most volume interval position.
    // Even if the front-most volume interval distance from occlusion ray origin is almost 0.0f,
    // isInActiveField() might return false and it is pretty difficult to robustly make this point is
    // inside volume or not when only using isInActiveField()
    // Following logic try to decide whether occlusion ray origin is reasonably close to active volume.
    // If ray origin is reasonably close to active volume, we consider this volumeId as originVolumeId.
    //
    geom::internal::VolumeRayState tmpVolState;
    tmpVolState.resetState(ray.tfar, true);
    bool intersectVoxel =
        vdbVolume->queryIntersections(rayOrg, rayDir, ray.tnear, ray.time, threadIdx, volumeId, tmpVolState,
                                      true);
    if (!intersectVoxel || tmpVolState.getIntervalCount() == 0) {
        // It's possible to intersect the volume but have it beyond the tfar distance
        // of the ray, in which case there are no intervals.
        return false; // no shadow occlusion -> no originVolume.
    }

    geom::internal::VolumeTransition* volTransition = tmpVolState.getVolumeIntervals();
    if (volTransition[0].mVolumeId != volumeId) {
        return false; // probably never happen. 
    }

    float t0 = volTransition[0].mTRenderSpace[0];
    float t1 = volTransition[0].mTRenderSpace[1];
    if (t0 < 0.0 || t1 < 0.0) {
        return false; // probably never happen
    }

    //
    // Using the first interval's t0, t1 (both are renderSpace distance), we can auto-calibrate the
    // decision distance to the closest volume as its size.
    // Currently, 10% (heuristically decided value : offsetScale) of the closest volume size
    // (= distance of t0 and t1) is used and so far so good (based on the production scene test).
    // We might need to control by the user in the future if needed.
    //
    constexpr float offsetScale = 0.1;
    float t = t0 + (t1 - t0) * offsetScale;
    pStart = rayOrg + rayDir * t; // New detection point for origin volume judgment.
                                  // This point is offsetScale farther from the ray origin of its interval.
    if (vdbVolume->isInActiveField(threadIdx, pStart, ray.time)) {
        // The ray origin is reasonably close to active volume or inside volume.
        // We can consider this volume as originVolume.
        return true;
    }
    return false; // This volume is not a originVolume
}

void
vdbVolumeIntervalFilter(const RTCFilterFunctionNArguments* args)
{
    if (args->N == 1) {
        RTCRay* ray = (RTCRay*)args->ray;
        RTCHit* hit = (RTCHit*)args->hit;
        // Immediately exit if this is not a volume intersection
        mcrt_common::IntersectContext* context =
            (mcrt_common::IntersectContext*)args->context;
        if (context->mRayExtension->materialID >= 0) {
            return;
        }
        auto tls = (geom::internal::TLState*)context->mRayExtension->geomTls;
        const int volumeMask = scene_rdl2::rdl2::ALL_VISIBLE <<
            scene_rdl2::rdl2::sNumVisibilityTypes;
        if ((tls == nullptr) || (ray->mask & volumeMask) == 0) {
            return;
        }
        const geom::internal::BVHUserData* userData =
            (const geom::internal::BVHUserData*)args->geometryUserPtr;
        auto vdbVolume = const_cast<geom::internal::VdbVolume*>(
            (const geom::internal::VdbVolume*)userData->mPrimitive);
        int assignmentId = vdbVolume->getIntersectionAssignmentId(hit->primID);
        if (assignmentId < 0) {
            return;
        }
        // invalidate the intersection so this ray continues to collect other
        // volume intersections
        args->valid[0] = 0;
        auto& volumeRayState = tls->mVolumeRayState;
        const int volumeId = volumeRayState.getVolumeId(assignmentId, context->mRayExtension->volumeInstanceState);
        bool eval = !volumeRayState.isVisited(volumeId);
        if (ray->mask & (scene_rdl2::rdl2::SHADOW << scene_rdl2::rdl2::sNumVisibilityTypes)) {
            // volume shadow ray
            const geom::internal::ShadowLinking* shadowLinking = vdbVolume->getShadowLinking(assignmentId);
            if (shadowLinking && shadowLinking->getLights().size()) {
                //
                // We are adjusting volume interval construction depending on the condition of this call is
                // for light transmittance computation of volume or not (i.e. return value of
                // volumeRayState.getEstimateInScatter()). 
                // If we are not estimateInScatter condition, we simply construct every volume interval
                // without any special case.
                //
                // If we are estimateInScatter condition, we want to skip construction of volume interval
                // if this volume does not cast shadow controlled by ShadowSet attribute.
                // In order to judge this decision (i.e. construct volume interval or not), we need volumeId
                // at ray origin position (= origin volumeId).
                // If origin volumeId is equal of current volume, we have to construct volume interval in
                // order to compute self-shadowing because ray origin volume is myself (we need self-shadowing
                // for this volume even ShadowSet attribute says this volume does not cast). 
                // If origin volumeId is empty or not the same as current volumeId, this is a case of this
                // volume cast shadow to ray origin volume/surface.
                // In this case, we can safely skip this volume interval for transmittance computation if
                // ShadowSet attribute has this light (i.e. this volume does not cast a shadow on the ray
                // origin).

                float volumeDistance = ray->tfar; // tfar is the intersection distance to the volume's bbox
                if (volumeRayState.getOriginVolumeId() == geom::internal::VolumeRayState::ORIGIN_VOLUME_INIT ||
                    (volumeDistance < volumeRayState.getOriginVolumeDistance())) {
                    // If we have found an origin volume, check the intersection distance to see if this
                    // one is actually closer.  This fixes an issue where the BVH traversal may not be
                    // in distance order along the ray and handles the case where multiple volumes
                    // overlap the origin.
                    if (volumeRayState.getEstimateInScatter()) {
                        if (originVolumeTest(tls->mThreadIdx, volumeId, *ray, vdbVolume)) {
                            volumeRayState.setOriginVolumeId(volumeId, volumeDistance);
                        } else {
                            // not set origin volume for the next vdbVolumeIntervalFilter() call.
                        }
                    } else {
                        volumeRayState.setOriginVolumeId(geom::internal::VolumeRayState::ORIGIN_VOLUME_EMPTY, 0.f);
                    }
                }

                if (volumeId != volumeRayState.getOriginVolumeId()) {
                    int rayId = RTCRayN_id((RTCRayN *)args->ray, 1, 0);
                    mcrt_common::RayExtension& rayExtension = context->mRayExtension[rayId];            
                    if (!shadowLinking->
                        canCastShadow((const scene_rdl2::rdl2::Light*)rayExtension.instance0OrLight)) {
                        eval = false;
                    }
                }
            }
        }
        if (eval) {
            Vec3f rayOrg(ray->org_x, ray->org_y, ray->org_z);
            Vec3f rayDir(ray->dir_x, ray->dir_y, ray->dir_z);
            bool intersectVoxel =
                vdbVolume->queryIntersections(rayOrg, rayDir, ray->tnear, ray->time, tls->mThreadIdx,
                                              volumeId, volumeRayState, false);
            // whether the starting point of ray segment is inside volume
            // there seems to be rare edge cases that the ray starting point
            // is inside active voxel but the vdb VolumeRayIntersector doesn't
            // intersect any voxel. In this case we treat the ray starting point
            // outside of volume
            Vec3f pStart = rayOrg + rayDir * ray->tnear;
            if (vdbVolume->isInActiveField(tls->mThreadIdx, pStart, ray->time) &&
                intersectVoxel) {
                volumeRayState.turnOn(volumeId, vdbVolume);
            }
            vdbVolume->initVolumeSampleInfo(
                &volumeRayState.getVolumeSampleInfo(volumeId),
                rayOrg, rayDir, ray->time,
                volumeRayState.getVolumeAssignmentTable()->lookupWithAssignmentId(
                assignmentId), volumeId);
            volumeRayState.setVisited(volumeId, vdbVolume);
        }
    } else {
        MNRY_ASSERT_REQUIRE(false, "vectorized volume is not available yet");
    }
}

void
manifoldVolumeIntervalFilter(const RTCFilterFunctionNArguments* args)
{
    if (args->N == 1) {
        RTCRay* ray = (RTCRay*)args->ray;
        RTCHit* hit = (RTCHit*)args->hit;
        // Immediately exit if this is not a volume intersection
        mcrt_common::IntersectContext* context =
            (mcrt_common::IntersectContext*)args->context;
        if (context->mRayExtension->materialID >= 0) {
            return;
        }
        auto tls = (geom::internal::TLState*)context->mRayExtension->geomTls;
        const int volumeMask = scene_rdl2::rdl2::ALL_VISIBLE <<
            scene_rdl2::rdl2::sNumVisibilityTypes;
        if ((tls == nullptr) || (ray->mask & volumeMask) == 0) {
            return;
        }
        const geom::internal::BVHUserData* userData =
            (const geom::internal::BVHUserData*)args->geometryUserPtr;
        auto primitive = (const geom::internal::NamedPrimitive*)userData->mPrimitive;
        int assignmentId = primitive->getIntersectionAssignmentId(hit->primID);
        if (assignmentId < 0) {
            return;
        }
        // invalidate the intersection so this ray continues to collect other
        // volume intersections
        args->valid[0] = 0;
        auto& volumeRayState = tls->mVolumeRayState;
        const int volumeId = volumeRayState.getVolumeId(assignmentId, context->mRayExtension->volumeInstanceState);
        // ray casting method to determine whether ray origin is inside
        // this volume. If the ray hits this primitive even times,
        // it's out of this primitive, otherwise it's in this primitive.
        // This algorithm has the assumption that primitive with
        // volume assignment is a closed manifold.
        Vec3f rayDir(ray->dir_x, ray->dir_y, ray->dir_z);
        float rayTfar = ray->tfar;
        float dotProduct =
            rayDir.x * hit->Ng_x +
            rayDir.y * hit->Ng_y +
            rayDir.z * hit->Ng_z;
        bool isEntry = (dotProduct < 0.0f);

        // Eliminate repeated entry or exit points with identical t-values, which Embree generates in abundance.
        // Similar elimination can be achieved by switching on RTC_SCENE_FLAG_ROBUST, but we don't want to pay the
        // cost for that flag unless we must.
        int count = volumeRayState.getIntervalCount();
        geom::internal::VolumeTransition* intervals = volumeRayState.getVolumeIntervals();
        if (count > 0) {
                if (intervals[count-1].mT == rayTfar         &&
                    intervals[count-1].mVolumeId == volumeId &&
                    intervals[count-1].mIsEntry == isEntry) {
                return;
            }
        }

        volumeRayState.addInterval(primitive, rayTfar, volumeId, isEntry);
        Vec3f rayOrg(ray->org_x, ray->org_y, ray->org_z);

        if (!volumeRayState.isVisited(volumeId)) {
            primitive->initVolumeSampleInfo(
                &volumeRayState.getVolumeSampleInfo(volumeId), rayOrg, rayDir, ray->time,
                volumeRayState.getVolumeAssignmentTable()->lookupWithAssignmentId(
                assignmentId), volumeId);
            volumeRayState.setVisited(volumeId, primitive);
        }
    } else {
        MNRY_ASSERT_REQUIRE(false, "vectorized volume is not available yet");
    }
}

void
bssrdfTraceSetIntersectionFilter(const RTCFilterFunctionNArguments* args)
{
    // TODO can we make this filter only get called while firing bssrdf
    // projection ray instead of register it to every intersection call?
    // (it seems embree can do some kind of per ray filter function call)
    if (args->N == 1) {
        // If this is not a subsurface intersection,
        // immediately exit filter function.
        mcrt_common::IntersectContext* context =
            (mcrt_common::IntersectContext*)args->context;
        mcrt_common::RayExtension* rayExtension = context->mRayExtension;
        if (rayExtension->materialID == -1) {
            return;
        }
        auto geomTls = (geom::internal::TLState*)rayExtension->geomTls;
        RTCRay* ray = (RTCRay*)args->ray;
        RTCHit* hit = (RTCHit*)args->hit;
        const geom::internal::BVHUserData* userData =
            (const geom::internal::BVHUserData*)args->geometryUserPtr;
        auto primitive = (const geom::internal::NamedPrimitive*)userData->mPrimitive;
        int assignmentId = primitive->getIntersectionAssignmentId(hit->primID);
        if (assignmentId < 0) {
            return;
        }
        if (geomTls->mSubsurfaceTraceSet != nullptr) {
            // there is a user specified trace set
            const scene_rdl2::rdl2::TraceSet* traceSet =
                (const scene_rdl2::rdl2::TraceSet *)geomTls->mSubsurfaceTraceSet;
            const auto& gp = userData->mLayer->lookupGeomAndPart(assignmentId);
            // getAssignmentId(Geometry, part) is a slow function.
            // If there is ever a complaint that using a trace set
            // slows down the render, this is the place to optimize.
            if (traceSet->getAssignmentId(gp.first, gp.second) < 0) {
                args->valid[0] = 0;
            }
        } else {
            // default case: we compare material ids of subsurface ray
            // and intersection
            const scene_rdl2::rdl2::Material* rdl2Material =
                userData->mLayer->lookupMaterial(assignmentId);
            const shading::RootShader* material =
                &rdl2Material->get<const shading::RootShader>();
            if (rayExtension->materialID != material->getMaterialId()) {
                args->valid[0] = 0;
            }
        }
    }
}

void
backFaceCullingFilter(const RTCFilterFunctionNArguments* args)
{
    int* valid = args->valid;
    unsigned int N = args->N;
    RTCRayN* rays = args->ray;
    RTCHitN* hits = args->hit;
    for (unsigned int index = 0; index < N; ++index) {
        if (valid[index] == 0) {
            continue;
        }
        float dotProduct =
            RTCRayN_dir_x(rays, N, index) * RTCHitN_Ng_x(hits, N, index) +
            RTCRayN_dir_y(rays, N, index) * RTCHitN_Ng_y(hits, N, index) +
            RTCRayN_dir_z(rays, N, index) * RTCHitN_Ng_z(hits, N, index);
        // reject back facing intersection
        if (dotProduct >= 0.0f) {
            valid[index] = 0;
        }
    }
}

void
skipOcclusionFilter(const RTCFilterFunctionNArguments* args)
{

    // Although "args" points to a packet of N rays, in practice we always set N=1 (and there are parts of the code
    // base which already rely on this assumption). So here we hardcode for that case - thus we only need to inspect
    // valid[0] here, which we can write as *valid.
    int* valid = args->valid;
    if (*valid == 0) {
        return;
    }

    // We continue to assume N=1 here and cast the various data types to their single-ray counterparts.
    const geom::internal::BVHUserData* userData = static_cast<const geom::internal::BVHUserData*>(args->geometryUserPtr);
    const geom::internal::NamedPrimitive* prim  = static_cast<const geom::internal::NamedPrimitive*>(userData->mPrimitive);
    RTCRay* ray = (RTCRay*)args->ray;
    RTCHit* hit = (RTCHit*)args->hit;
    const mcrt_common::IntersectContext* context = reinterpret_cast<const mcrt_common::IntersectContext*>(args->context);
    mcrt_common::RayExtension& rayExtension = context->mRayExtension[ray->id];
    int casterId = prim->getIntersectionAssignmentId(hit->primID);
    int receiverId = rayExtension.shadowReceiverId;

    // If the occlusion ray was cast from a volume, suppress shadowing by the geometry it's assigned to (see
    // MOONRAY-4130). This test reuses the volumeInstanceState member of RayExtension, which is otherwise unused
    // in occlusion tests.
    if (rayExtension.volumeInstanceState && (receiverId == casterId)) {
        *valid = 0;
        return;
    }

    // Shadow linking
    // We currently don't support per instance shadow linking.
    MNRY_ASSERT(userData->mPrimitive->getType() != geom::internal::Primitive::INSTANCE);
    const geom::internal::ShadowLinking* shadowLinking = prim->getShadowLinking(casterId);
    if (shadowLinking != nullptr) {
        // Suppress shadows if this light is marked as not casting shadows from the caster geometry
        if (!shadowLinking->canCastShadow((const scene_rdl2::rdl2::Light*)rayExtension.instance0OrLight)) {
            *valid = 0;
            return;
        }

        // Suppress shadows if this receiver is marked as not receiving shadows from the caster geometry
        // (See MOONRAY-4130 and MOONRAY-4663)
        if (!shadowLinking->canReceiveShadow(receiverId)) {
            *valid = 0;
            return;
        }
    }
}

} // namespace rt
} // namespace moonray

