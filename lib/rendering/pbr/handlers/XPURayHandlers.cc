// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

#include "XPURayHandlers.h"

#include "RayHandlerUtils.h"
#include <moonray/rendering/pbr/core/Aov.h>

#include <moonray/rendering/geom/prim/BVHUserData.h>
#include <moonray/rendering/geom/prim/Instance.h>
#include <moonray/rendering/geom/prim/Primitive.h>
#include <moonray/rendering/mcrt_common/Clock.h>
#include <moonray/rendering/mcrt_common/ProfileAccumulatorHandles.h>
#include <moonray/rendering/mcrt_common/ThreadLocalState.h>
#include <moonray/rendering/pbr/core/PbrTLState.h>
#include <moonray/rendering/pbr/core/RayState.h>
#include <moonray/rendering/pbr/integrator/PathIntegrator.h>
#include <moonray/rendering/pbr/integrator/PathIntegratorUtil.h>
#include <moonray/rendering/pbr/integrator/VolumeTransmittance.h>
#include <moonray/rendering/rt/gpu/GPUAccelerator.h>
#include <moonray/rendering/rt/EmbreeAccelerator.h>
#include <moonray/rendering/shading/Material.h>
#include <moonray/rendering/shading/Types.h>

#include <scene_rdl2/common/math/Color.h>
#include <scene_rdl2/common/math/Vec3.h>
#include <scene_rdl2/scene/rdl2/VisibilityFlags.h>

#define RAY_HANDLER_STD_SORT_CUTOFF     200

namespace moonray {
namespace pbr {

// warning #1684: conversion from pointer to
// same-sized integral type (potential portability problem)
#pragma warning push
#pragma warning disable 1684

namespace {

unsigned
computeXPUOcclusionQueriesOnGPU(mcrt_common::ThreadLocalState *tls,
                                unsigned numRays,
                                BundledOcclRay* rays,
                                const rt::GPURay* gpuRays,
                                BundledRadiance *results,
                                std::atomic<int>& threadsUsingGPU)
{
    pbr::TLState *pbrTls = tls->mPbrTls.get();
    scene_rdl2::alloc::Arena *arena = &tls->mArena;

    // Update ray stats.
    pbrTls->mStatistics.addToCounter(STATS_OCCLUSION_RAYS, numRays);
    pbrTls->mStatistics.addToCounter(STATS_BUNDLED_OCCLUSION_RAYS, numRays);
    pbrTls->mStatistics.addToCounter(STATS_BUNDLED_GPU_OCCLUSION_RAYS, numRays);

    const FrameState &fs = *pbrTls->mFs;
    rt::GPUAccelerator *accel = const_cast<rt::GPUAccelerator*>(fs.mGPUAccel);
    const bool disableShadowing = !fs.mIntegrator->getEnableShadowing();

    threadsUsingGPU++;
    {
        EXCL_ACCUMULATOR_PROFILE(pbrTls, EXCL_ACCUM_GPU_OCCLUSION);

        // Call the GPU and wait for it to finish processing these rays.
        accel->occluded(tls->mThreadIdx, numRays, gpuRays, rays, sizeof(rays[0]));
    }
    threadsUsingGPU--;

    unsigned char *isOccluded = accel->getOutputOcclusionBuf(tls->mThreadIdx);

/*
    {
        // SW debug mode for comparison
        unsigned numIncorrect = 0;

        for (size_t i = 0; i < numRays; ++i) {
            const BundledOcclRay &occlRay = raysCopy[i];
            MNRY_ASSERT(occlRay.isValid());

            occlRaysCopy[i] = occlRay;

            mcrt_common::Ray rtRay;

            rtRay.org[0]  = occlRay.mOrigin.x;
            rtRay.org[1]  = occlRay.mOrigin.y;
            rtRay.org[2]  = occlRay.mOrigin.z;
            rtRay.dir[0]  = occlRay.mDir.x;
            rtRay.dir[1]  = occlRay.mDir.y;
            rtRay.dir[2]  = occlRay.mDir.z;
            rtRay.tnear   = occlRay.mMinT;
            rtRay.tfar    = occlRay.mMaxT;
            rtRay.time    = occlRay.mTime;
            rtRay.mask    = scene_rdl2::rdl2::SHADOW;
            rtRay.geomID  = RT_INVALID_RAY_ID;
            rtRay.ext.instance0OrLight = static_cast<BundledOcclRayData *>(
                pbrTls->getListItem(occlRay.mDataPtrHandle, 0))->mLight->getRdlLight();

            //isOccluded[i] = accel->occluded(rtRay);
            if (isOccluded[i] != accel->occluded(rtRay)) {
                 numIncorrect++;
            }
        }

        if (numIncorrect > 0) {
            std::cout << "numIncorrect: " << numIncorrect << std::endl;
        }
    }
*/

    // Create the BundledRadiance objects as required based on the occlusion
    // test results.
    unsigned numRadiancesFilled = 0;
    for (unsigned i = 0; i < numRays; ++i) {
        BundledOcclRay &occlRay = rays[i];

        if (occlRay.mOcclTestType == OcclTestType::FORCE_NOT_OCCLUDED) {
            // See forceSingleRaysUnoccluded()
            scene_rdl2::math::Color tr = getTransmittance(pbrTls, occlRay);
            occlRay.mRadiance = occlRay.mRadiance * tr;
            BundledRadiance *result = &results[numRadiancesFilled++];
            fillBundledRadiance(pbrTls, result, occlRay);

            // LPE
            if (occlRay.mDataPtrHandle != nullHandle) {
                EXCL_ACCUMULATOR_PROFILE(pbrTls, EXCL_ACCUM_AOVS);

                const int numItems = pbrTls->getNumListItems(occlRay.mDataPtrHandle);
                accumLightAovs(pbrTls, occlRay, fs, numItems, tr, nullptr, AovSchema::sLpePrefixNone);
            }

        } else if (!isOccluded[i] || disableShadowing) {
            scene_rdl2::math::Color tr = getTransmittance(pbrTls, occlRay);
            occlRay.mRadiance = occlRay.mRadiance * tr;
            BundledRadiance *result = &results[numRadiancesFilled++];
            fillBundledRadiance(pbrTls, result, occlRay);

            // LPE
            if (occlRay.mDataPtrHandle != nullHandle) {
                EXCL_ACCUMULATOR_PROFILE(pbrTls, EXCL_ACCUM_AOVS);

                const int numItems = pbrTls->getNumListItems(occlRay.mDataPtrHandle);
                accumLightAovs(pbrTls, occlRay, fs, numItems, scene_rdl2::math::sWhite, &tr,
                               AovSchema::sLpePrefixUnoccluded);
                accumVisibilityAovs(pbrTls, occlRay, fs, numItems, reduceTransparency(tr));
            }

        } else {
            // LPE: visibility aovs when we don't hit light
            if (occlRay.mDataPtrHandle != nullHandle) {

                const Light *light = static_cast<BundledOcclRayData *>(
                    pbrTls->getListItem(occlRay.mDataPtrHandle, 0))->mLight;

                // see PathIntegrator::addDirectVisibleLightSampleContributions()
                if (light->getClearRadiusFalloffDistance() != 0.f &&
                    occlRay.mMaxT < light->getClearRadius() + light->getClearRadiusFalloffDistance()) {
                    scene_rdl2::math::Color tr = getTransmittance(pbrTls, occlRay);
                    occlRay.mRadiance = calculateShadowFalloff(light, occlRay.mMaxT, tr * occlRay.mRadiance);
                    BundledRadiance *result = &results[numRadiancesFilled++];
                    fillBundledRadiance(pbrTls, result, occlRay);
                }

                EXCL_ACCUMULATOR_PROFILE(pbrTls, EXCL_ACCUM_AOVS);

                const int numItems = pbrTls->getNumListItems(occlRay.mDataPtrHandle);
                accumVisibilityAovsOccluded(pbrTls, occlRay, fs, numItems);

                // We only accumulate here if we were occluded but we have the flag on. Otherwise
                // it would already have been filled by the previous call.
                if (fs.mAovSchema->hasLpePrefixFlags(AovSchema::sLpePrefixUnoccluded)) {
                    accumLightAovs(pbrTls, occlRay, fs, numItems, scene_rdl2::math::sWhite, nullptr,
                                   AovSchema::sLpePrefixUnoccluded);
                }
            }
        }

        // LPE
        // we are responsible for freeing LPE memory
        if (occlRay.mDataPtrHandle != nullHandle) {
            pbrTls->freeList(occlRay.mDataPtrHandle);
        }
        pbrTls->releaseDeepData(occlRay.mDeepDataHandle);
    }

    return numRadiancesFilled;
}

}


void
xpuRayBundleHandler(mcrt_common::ThreadLocalState *tls,
                    unsigned numEntries,
                    RayState **rayStates,
                    const rt::GPURay *gpuRays,
                    std::atomic<int>& threadsUsingGPU)
{
    pbr::TLState *pbrTls = tls->mPbrTls.get();

    EXCL_ACCUMULATOR_PROFILE(pbrTls, EXCL_ACCUM_RAY_HANDLER);

    MNRY_ASSERT(numEntries);

    // By convention, if userData is null then rayState contains an array of raw
    // RayState pointers.
    RayHandlerFlags handlerFlags = RayHandlerFlags((uint64_t)nullptr); // TODO: is userData needed?

    const FrameState &fs = *pbrTls->mFs;

    scene_rdl2::alloc::Arena *arena = &tls->mArena;
    SCOPED_MEM(arena);

    // heat map
    int64_t ticks = 0;
    MCRT_COMMON_CLOCK_OPEN(fs.mRequiresHeatMap ? &ticks : nullptr);

    // Perform all intersection checks.
    if (numEntries) {

        pbrTls->mStatistics.addToCounter(STATS_INTERSECTION_RAYS, numEntries);
        pbrTls->mStatistics.addToCounter(STATS_BUNDLED_INTERSECTION_RAYS, numEntries);
        pbrTls->mStatistics.addToCounter(STATS_BUNDLED_GPU_INTERSECTION_RAYS, numEntries);

        rt::GPUAccelerator *accel = const_cast<rt::GPUAccelerator*>(fs.mGPUAccel);

        threadsUsingGPU++;
        {
            EXCL_ACCUMULATOR_PROFILE(pbrTls, EXCL_ACCUM_GPU_INTERSECTION);

            // Call the GPU and wait for it to finish processing these rays.
            accel->intersect(tls->mThreadIdx, numEntries, gpuRays);
        }
        threadsUsingGPU--;

        rt::GPURayIsect* isects = accel->getOutputIsectBuf(tls->mThreadIdx);

        for (unsigned i = 0; i < numEntries; ++i) {
            RayState *rs = rayStates[i];
            RayState rsCPU = *rs; // copy for validation below

            MNRY_ASSERT(isValid(rs));
            rs->mRay.tfar = isects[i].mTFar;
            rs->mRay.Ng.x = isects[i].mNgX;
            rs->mRay.Ng.y = isects[i].mNgY;
            rs->mRay.Ng.z = isects[i].mNgZ;
            rs->mRay.u = isects[i].mU;
            rs->mRay.v = isects[i].mV;
            rs->mRay.geomID = isects[i].mEmbreeGeomID;
            rs->mRay.primID = isects[i].mPrimID;
            rs->mRay.instID = -1;
            rs->mRay.ext.userData = reinterpret_cast<void*>(isects[i].mEmbreeUserData);

            void* topLevelInstance = accel->instanceIdToInstancePtr(isects[i].mInstance0IdOrLight);
            rs->mRay.ext.instance0OrLight = topLevelInstance;
            rs->mRay.ext.instance1 = accel->instanceIdToInstancePtr(isects[i].mInstance1Id);
            rs->mRay.ext.instance2 = accel->instanceIdToInstancePtr(isects[i].mInstance2Id);
            rs->mRay.ext.instance3 = accel->instanceIdToInstancePtr(isects[i].mInstance3Id);
            rs->mRay.ext.l2r = scene_rdl2::math::Xform3f(
                isects[i].mL2R[0][0], isects[i].mL2R[0][1], isects[i].mL2R[0][2],
                isects[i].mL2R[1][0], isects[i].mL2R[1][1], isects[i].mL2R[1][2],
                isects[i].mL2R[2][0], isects[i].mL2R[2][1], isects[i].mL2R[2][2],
                isects[i].mL2R[3][0], isects[i].mL2R[3][1], isects[i].mL2R[3][2]);

            if (topLevelInstance != nullptr) {
                geom::internal::Instance* inst = reinterpret_cast<geom::internal::Instance*>(topLevelInstance);
                int geomID = inst->getGeomID();
                // both geomID and instID are set to the instance's geomID
                rs->mRay.geomID = geomID;
                rs->mRay.instID = geomID;
            }

#if 0 // debugging code
            // Validate the GPU intersection results against CPU Embree
            {
                const rt::EmbreeAccelerator *embreeAccel = fs.mEmbreeAccel;
                embreeAccel->intersect(rsCPU.mRay);

                if (rs->mRay.geomID != rsCPU.mRay.geomID) {
                    std::cout << "ray: " << i << " embree geomID: " << rsCPU.mRay.geomID << " optix geomID: " << rs->mRay.geomID << std::endl;
                } else {
                    if (rs->mRay.geomID != -1) {
                        if (rs->mRay.primID != rsCPU.mRay.primID) {
                            std::cout << "ray: " << i << " embree primID: " << rsCPU.mRay.primID << " optix primID: " << rs->mRay.primID << std::endl;
                        }
                        if (rs->mRay.instID != rsCPU.mRay.instID) {
                            std::cout << "ray: " << i << " embree instID: " << rsCPU.mRay.instID << " optix instID: " << rs->mRay.instID << std::endl;
                        }
                        if (rs->mRay.ext.userData != rsCPU.mRay.ext.userData) {
                            std::cout << "ray: " << i << " embree ext.userData: " << rsCPU.mRay.ext.userData << 
                                        " optix ext.userData: " << rs->mRay.ext.userData << std::endl;
                        }
                        if (rs->mRay.instID != -1) {
                            if (rs->mRay.ext.instance0OrLight != rsCPU.mRay.ext.instance0OrLight) {
                                std::cout << "ray: " << i << " embree ext.instance0OrLight: " << rsCPU.mRay.ext.instance0OrLight <<
                                            " optix ext.instance0OrLight: " << rs->mRay.ext.instance0OrLight << std::endl;
                            }
                            if ((!scene_rdl2::math::isEqual(rs->mRay.ext.l2r.l.vx.x, rsCPU.mRay.ext.l2r.l.vx.x, 0.001f)) ||
                                (!scene_rdl2::math::isEqual(rs->mRay.ext.l2r.l.vx.y, rsCPU.mRay.ext.l2r.l.vx.y, 0.001f)) ||
                                (!scene_rdl2::math::isEqual(rs->mRay.ext.l2r.l.vx.z, rsCPU.mRay.ext.l2r.l.vx.z, 0.001f)) ||
                                (!scene_rdl2::math::isEqual(rs->mRay.ext.l2r.l.vy.x, rsCPU.mRay.ext.l2r.l.vy.x, 0.001f)) ||
                                (!scene_rdl2::math::isEqual(rs->mRay.ext.l2r.l.vy.y, rsCPU.mRay.ext.l2r.l.vy.y, 0.001f)) ||
                                (!scene_rdl2::math::isEqual(rs->mRay.ext.l2r.l.vy.z, rsCPU.mRay.ext.l2r.l.vy.z, 0.001f)) ||
                                (!scene_rdl2::math::isEqual(rs->mRay.ext.l2r.l.vz.x, rsCPU.mRay.ext.l2r.l.vz.x, 0.001f)) ||
                                (!scene_rdl2::math::isEqual(rs->mRay.ext.l2r.l.vz.y, rsCPU.mRay.ext.l2r.l.vz.y, 0.001f)) ||
                                (!scene_rdl2::math::isEqual(rs->mRay.ext.l2r.l.vz.z, rsCPU.mRay.ext.l2r.l.vz.z, 0.001f)) ||
                                (!scene_rdl2::math::isEqual(rs->mRay.ext.l2r.p.x, rsCPU.mRay.ext.l2r.p.x, 0.001f)) ||
                                (!scene_rdl2::math::isEqual(rs->mRay.ext.l2r.p.y, rsCPU.mRay.ext.l2r.p.y, 0.001f)) ||
                                (!scene_rdl2::math::isEqual(rs->mRay.ext.l2r.p.z, rsCPU.mRay.ext.l2r.p.z, 0.001f))) {
                                    std::cout << "ray: " << i << " embree ext.l2r: " << rsCPU.mRay.ext.l2r << std::endl <<
                                            "     optix ext.l2r: " << rs->mRay.ext.l2r << std::endl;
                            }
                        }
                        if (rs->mRay.primID == rsCPU.mRay.primID) {
                            if (!scene_rdl2::math::isEqual(rs->mRay.tfar, rsCPU.mRay.tfar, 0.001f)) {
                                std::cout << "ray: " << i << " embree tfar: " << rsCPU.mRay.tfar << " optix tfar: " << rs->mRay.tfar << std::endl;
                            }
                            if (!scene_rdl2::math::isEqual(rs->mRay.u, rsCPU.mRay.u, 0.005f)) {
                                std::cout << "ray: " << i << " embree u: " << rsCPU.mRay.u << " optix u: " << rs->mRay.u << std::endl;
                            }
                            if (!scene_rdl2::math::isEqual(rs->mRay.v, rsCPU.mRay.v, 0.005f)) {
                                std::cout << "ray: " << i << " embree v: " << rsCPU.mRay.v << " optix v: " << rs->mRay.v << std::endl;
                            }
                            if (rs->mRay.Ng.x != 0.f || rs->mRay.Ng.y != 0.f || rs->mRay.Ng.z != 1.f) {
                                // Ignore unused curves Ng when Ng = 0 0 1
                                if (!scene_rdl2::math::isEqual(rs->mRay.Ng.x, rsCPU.mRay.Ng.x, 0.005f)) {
                                    std::cout << "ray: " << i << " embree Ng.x: " << rsCPU.mRay.Ng.x << " optix Ng.x: " << rs->mRay.Ng.x << std::endl;
                                }
                                if (!scene_rdl2::math::isEqual(rs->mRay.Ng.y, rsCPU.mRay.Ng.y, 0.005f)) {
                                    std::cout << "ray: " << i << " embree Ng.y: " << rsCPU.mRay.Ng.y << " optix Ng.y: " << rs->mRay.Ng.y << std::endl;
                                }
                                if (!scene_rdl2::math::isEqual(rs->mRay.Ng.z, rsCPU.mRay.Ng.z, 0.005f)) {
                                    std::cout << "ray: " << i << " embree Ng.z: " << rsCPU.mRay.Ng.z << " optix Ng.z: " << rs->mRay.Ng.z << std::endl;
                                }
                            }
                        }
                    }
                }
            }
#endif
        }
    }

    // ******* Below here is the same as the regular vector mode ray handler
    // TODO: find a way to avoid this duplication

    // Volumes - compute volume radiance and transmission for each ray
    for (unsigned i = 0; i < numEntries; ++i) {
        RayState &rs = *rayStates[i];
        const mcrt_common::Ray &ray = rs.mRay;
        const Subpixel &sp = rs.mSubpixel;
        PathVertex &pv = rs.mPathVertex;
        const int lobeType = pv.nonMirrorDepth == 0 ? 0 : pv.lobeType;
        const unsigned sequenceID = rs.mSequenceID;
        float *aovs = nullptr;
        PathIntegrator::DeepParams *deepParams = nullptr; // TODO: MOONRAY-3133 support deep output of volumes
        rs.mVolRad = scene_rdl2::math::sBlack;
        VolumeTransmittance vt;
        vt.reset();
        float volumeSurfaceT = scene_rdl2::math::sMaxValue;
        rs.mVolHit = fs.mIntegrator->computeRadianceVolume(pbrTls, ray, sp, pv, lobeType,
            rs.mVolRad, sequenceID, vt, aovs, deepParams, &rs, &volumeSurfaceT);
        rs.mVolTr = vt.mTransmittanceE;
        rs.mVolTh = vt.mTransmittanceH;
        rs.mVolTalpha = vt.mTransmittanceAlpha;
        rs.mVolTm = vt.mTransmittanceMin;
        rs.mVolumeSurfaceT = volumeSurfaceT;
    }

    CHECK_CANCELLATION(pbrTls, return);

    // heat maps
    MCRT_COMMON_CLOCK_CLOSE();
    if (fs.mRequiresHeatMap) {
        heatMapBundledUpdate(pbrTls, ticks, rayStates, numEntries);
    }

    //
    // Sort by material to minimize locks when adding to shared shade queues.
    //
    struct SortedEntry
    {
        uint32_t mSortKey;                      // Material bundled id is stored in here.
        uint32_t mRsIdx;                        // Global ray state index.
        const shading::Material *mMaterial;
    };
    SortedEntry *sortedEntries = arena->allocArray<SortedEntry>(numEntries, CACHE_LINE_SIZE);
    unsigned numSortedEntries = 0;
    uint32_t maxSortKey = 0;

    // Allocate memory to gather raystates so we can bulk free them later in the function.
    unsigned numRayStatesToFree = 0;
    RayState **rayStatesToFree = arena->allocArray<RayState *>(numEntries);

    RayState *baseRayState = indexToRayState(0);
    const scene_rdl2::rdl2::Layer *layer = fs.mLayer;

    for (unsigned i = 0; i < numEntries; ++i) {

        SortedEntry &sortedEntry = sortedEntries[numSortedEntries];
        mcrt_common::Ray &ray = rayStates[i]->mRay;
        PathVertex &pv = rayStates[i]->mPathVertex;

        if (ray.geomID == -1) {
            // We didn't hit anything.
            sortedEntry.mSortKey = 0;
            sortedEntry.mRsIdx = rayStates[i] - baseRayState;
            sortedEntry.mMaterial = nullptr;
            ++numSortedEntries;

            // Prevent aliasing in the visibility aov by accounting for 
            // primary rays that don't hit anything 
            if (ray.getDepth() == 0) {
                const AovSchema &aovSchema = *fs.mAovSchema;

                // If we're on the edge of the geometry, some rays should count as "hits", some as "misses". Here, 
                // we're adding light_sample_count * lights number of "misses" to the visibility aov to account for 
                // the light samples that couldn't be taken because the primary ray doesn't hit anything. 
                // This improves aliasing on the edges.
                if (!aovSchema.empty()) {
                    const LightAovs &lightAovs = *fs.mLightAovs;
                    
                    // predict the number of light samples that would have been taken if the ray hit geom
                    int totalLightSamples = fs.mIntegrator->getLightSampleCount() * fs.mScene->getLightCount();

                    // Doesn't matter what the lpe is -- if there are subpixels that hit a surface that isn't included
                    // in the lpe, this would be black anyway. If there are subpixels that DO hit a surface that is
                    // included in the lpe, this addition prevents aliasing. 
                    aovAccumVisibilityAttemptsBundled(pbrTls, aovSchema, lightAovs, totalLightSamples, 
                                                      rayStates[i]->mSubpixel.mPixel, rayStates[i]->mDeepDataHandle);
                }
            }
        } else {
            geom::internal::BVHUserData* userData =
                static_cast<geom::internal::BVHUserData*>(ray.ext.userData);
            const geom::internal::Primitive* prim = userData->mPrimitive;
            const scene_rdl2::rdl2::Material *rdl2Material = MNRY_VERIFY(prim)->getIntersectionMaterial(layer, ray);
            const shading::Material *material = &rdl2Material->get<const shading::Material>();

            if (material) {
                // perform material substitution if needed
                scene_rdl2::rdl2::RaySwitchContext switchCtx;
                switchCtx.mRayType = lobeTypeToRayType(pv.lobeType);
                rdl2Material = rdl2Material->raySwitch(switchCtx);
                material = &rdl2Material->get<const shading::Material>();

                sortedEntry.mSortKey = MNRY_VERIFY(material->getMaterialId());
                sortedEntry.mRsIdx = rayStates[i] - baseRayState;
                sortedEntry.mMaterial = material;
                maxSortKey = std::max(maxSortKey, sortedEntry.mSortKey);
                ++numSortedEntries;
            } else {
                // No material is assigned to this hit point, just skip entry
                // and free up associated RayState resource.
                rayStatesToFree[numRayStatesToFree++] = rayStates[i];

                // We may still have volume radiance to consider
                const RayState &rs = *rayStates[i];
                if (rs.mVolHit) {
                    // We passed through a volume and then hit a geometry.
                    // But there is no material assigned to the geometry, so
                    // there will be no further processing on this ray.  It will
                    // not be passed to the shade queue.
                    // We will add the radiance from the volume to the radiance
                    // queue and set the alpha based on the volume alpha.
                    const float alpha = ray.getDepth() == 0 ?
                        rs.mPathVertex.pathPixelWeight * (1.0f - reduceTransparency(rs.mVolTalpha)) : 0.f;
                    BundledRadiance rad;
                    rad.mRadiance = RenderColor(rs.mVolRad.r, rs.mVolRad.g, rs.mVolRad.b, alpha);
                    rad.mPathPixelWeight = rs.mPathVertex.pathPixelWeight;
                    rad.mPixel = rs.mSubpixel.mPixel;
                    rad.mSubPixelIndex = rs.mSubpixel.mSubpixelIndex;
                    rad.mDeepDataHandle = pbrTls->acquireDeepData(rs.mDeepDataHandle);
                    rad.mCryptomatteDataHandle = pbrTls->acquireCryptomatteData(rs.mCryptomatteDataHandle);
                    rad.mCryptoRefP = rs.mCryptoRefP;
                    rad.mCryptoP0 = rs.mCryptoP0;
                    rad.mCryptoRefN = rs.mCryptoRefN;
                    rad.mCryptoUV = rs.mCryptoUV;
                    rad.mTilePass = rs.mTilePass;
                    pbrTls->addRadianceQueueEntries(1, &rad);
                }
            }
        }
    }

    // Do the actual sorting.
    sortedEntries = scene_rdl2::util::smartSort32<SortedEntry, 0, RAY_HANDLER_STD_SORT_CUTOFF>(numSortedEntries,
                                                                                               sortedEntries,
                                                                                               maxSortKey, arena);

    //
    // The SortedEntry array in now sorted by material, with all the entries
    // which didn't hit anything or have a null material assigned at the start.
    //

    SortedEntry *endEntry = sortedEntries + numSortedEntries;
    unsigned numMisses = 0;

    // Aovs.
    float *aovs = nullptr;
    unsigned aovNumChannels = 0;
    if (!fs.mAovSchema->empty()) {
        // scratch space storage for per-pixel aov packing
        aovNumChannels = fs.mAovSchema->numChannels();
        aovs = arena->allocArray<float>(aovNumChannels);
        fs.mAovSchema->initFloatArray(aovs);
    }

    // Check if rays which didn't intersect anything hit any lights in the scene.
    if (sortedEntries->mSortKey == 0) {

        SortedEntry *spanEnd = sortedEntries + 1;
        while (spanEnd != endEntry && spanEnd->mSortKey == 0) {
            ++spanEnd;
        }

        numMisses = unsigned(spanEnd - sortedEntries);

        if (numMisses) {

            EXCL_ACCUMULATOR_PROFILE(pbrTls, EXCL_ACCUM_INTEGRATION);

            BundledRadiance *radiances = arena->allocArray<BundledRadiance>(numMisses, CACHE_LINE_SIZE);

            for (unsigned i = 0; i < numMisses; ++i) {

                RayState *rs = &baseRayState[sortedEntries[i].mRsIdx];
                // if the ray is not a primary ray and hit a light,
                // its radiance contribution will be tested in occlusion or
                // presence shadow ray queue
                scene_rdl2::math::Color radiance = scene_rdl2::math::sBlack;
                float alpha = 0.0f;
                BundledRadiance *rad = &radiances[i];

                // This code path only gets executed for rays which didn't intersect with any geometry
                // in the scene. We can discard such rays at this point but first need to check if
                // primary rays intersected any lights, so we can include their contribution if visible.

                const Light *hitLight = nullptr;
                if (rs->mRay.getDepth() == 0) {

                    LightIntersection hitLightIsect;
                    int numHits = 0;
                    const Light *hitLight;

                    SequenceIDIntegrator sid(rs->mSubpixel.mPixel,
                                             rs->mSubpixel.mSubpixelIndex,
                                             fs.mInitialSeed);
                    IntegratorSample1D lightChoiceSamples(sid);
                    hitLight = fs.mScene->intersectVisibleLight(rs->mRay,
                        sInfiniteLightDistance, lightChoiceSamples, hitLightIsect, numHits);

                    if (hitLight) {
                        // Evaluate the radiance on the selected light in camera.
                        // Note: we multiply the radiance contribution by the number of lights hit. This is
                        // because we want to compute the sum of all contributing lights, but we're
                        // stochastically sampling just one.

                        LightFilterRandomValues lightFilterR = {
                            scene_rdl2::math::Vec2f(0.f, 0.f),
                            scene_rdl2::math::Vec3f(0.f, 0.f, 0.f)}; // light filters don't apply to camera rays
                        radiance = rs->mPathVertex.pathThroughput *
                            hitLight->eval(tls, rs->mRay.getDirection(), rs->mRay.getOrigin(),
                                           lightFilterR, rs->mRay.getTime(), hitLightIsect, true, nullptr, nullptr,
                                           rs->mRay.getDirFootprint(), nullptr, nullptr) * numHits;
                        // attenuate based on volume transmittance
                        if (rs->mVolHit) radiance *= (rs->mVolTr * rs->mVolTh);

                        // alpha depends on volumes
                        if (rs->mVolHit) {
                            // We hit a visible light, but the light is not
                            // opaque in alpha (e.g. a distant or env light).
                            // There is a volume along this ray.  The volume
                            // alpha transmission determines the alpha contribution.
                            alpha = rs->mPathVertex.pathPixelWeight * (1.f - reduceTransparency(rs->mVolTalpha));
                        } else {
                            // We hit a visible light, but the light is not
                            // opaque in alpha (e.g. a distant or env light).
                            // There is no volume along the ray.
                            // The alpha contribution is 0.
                            alpha = 0.f;
                        }
                    } else if (rs->mVolHit) {
                        // We didn't hit a visible light.  We didn't hit geometry.
                        // But we did pass through a volume.
                        // The volume alpha transmission determines the alpha contribution.
                        alpha = rs->mPathVertex.pathPixelWeight * (1.f - reduceTransparency(rs->mVolTalpha));
                    }
                }

                // add in any volume radiance
                radiance += rs->mVolRad;

                rad->mRadiance = RenderColor(radiance.r, radiance.g, radiance.b, alpha);
                rad->mPathPixelWeight = rs->mPathVertex.pathPixelWeight;
                rad->mPixel = rs->mSubpixel.mPixel;
                rad->mSubPixelIndex = rs->mSubpixel.mSubpixelIndex;
                rad->mDeepDataHandle = pbrTls->acquireDeepData(rs->mDeepDataHandle);
                rad->mCryptomatteDataHandle = pbrTls->acquireCryptomatteData(rs->mCryptomatteDataHandle);
                rad->mCryptoRefP = rs->mCryptoRefP;
                rad->mCryptoP0 = rs->mCryptoP0;
                rad->mCryptoRefN = rs->mCryptoRefN;
                rad->mCryptoUV = rs->mCryptoUV;
                rad->mTilePass = rs->mTilePass;

                // LPE
                if (!fs.mAovSchema->empty()) {
                    EXCL_ACCUMULATOR_PROFILE(pbrTls, EXCL_ACCUM_AOVS);

                    // accumulate background aovs
                    aovAccumBackgroundExtraAovsBundled(pbrTls, fs, rs);

                    // Did we hit a volume and do we have volume depth/position AOVs?
                    if (rs->mRay.getDepth() == 0 && rs->mVolHit && rs->mVolumeSurfaceT < scene_rdl2::math::sMaxValue) {
                        aovSetStateVarsVolumeOnly(pbrTls,
                                                  *fs.mAovSchema,
                                                  rs->mVolumeSurfaceT,
                                                  rs->mRay,
                                                  *fs.mScene,
                                                  rs->mPathVertex.pathPixelWeight,
                                                  aovs);
                        aovAddToBundledQueueVolumeOnly(pbrTls,
                                                       *fs.mAovSchema,
                                                       rs->mRay,
                                                       AOV_TYPE_STATE_VAR,
                                                       aovs,
                                                       rs->mSubpixel.mPixel,
                                                       rs->mDeepDataHandle);
                    }

                    const LightAovs &lightAovs = *fs.mLightAovs;
                    // This is complicated.
                    // Case 1:
                    // rayDepth() == 0.  In this case, the ray left
                    // the camera, and hit a light.  We use the lpeStateId in
                    // in the ray state.
                    //
                    // Case 2:
                    // We expect that PathIntegratorBundled set lpeStateId to
                    // the scattering event, and lpeStateIdLight to the light
                    // event.  In this case we hit no geometry, so we hit the light.
                    // For this reason, we use lpeStateIdLight rather than lpeStateId
                    //
                    int lpeStateId = -1;
                    if (rs->mRay.getDepth() == 0) {
                        if (hitLight) {
                            // case 1
                            int lpeStateId = rs->mPathVertex.lpeStateId;
                            if (lpeStateId >= 0) {
                                // transition to light event
                                lpeStateId = lightAovs.lightEventTransition(pbrTls, lpeStateId, hitLight);
                            }
                        }
                    } else {
                        // case 2
                        // transition already computed in PathIntegratorBundled
                        lpeStateId = rs->mPathVertex.lpeStateIdLight;
                    }

                    if (lpeStateId >= 0) {
                        // accumulate results. As this has to do with directly hitting a light, we don't have
                        // to worry about pre-occlusion LPEs here.
                        aovAccumLightAovsBundled(pbrTls, *fs.mAovSchema,
                                                 lightAovs, radiance, nullptr, AovSchema::sLpePrefixNone, lpeStateId,
                                                 rad->mPixel, rad->mDeepDataHandle);
                    }
                }

                // It's critical that we don't leak ray states.
                rayStatesToFree[numRayStatesToFree++] = rs;
            }

            pbrTls->addRadianceQueueEntries(numMisses, radiances);

            CHECK_CANCELLATION(pbrTls, return);
        }
    }

    // Bulk free raystates.
    MNRY_ASSERT(numRayStatesToFree <= numEntries);
    pbrTls->freeRayStates(numRayStatesToFree, rayStatesToFree);

    //
    // Route remaining sortedEntries to their associated materials in batches.
    // Shade queues are thread safe, multiple threads can add to them simultaneously.
    //

    uint8_t *memBookmark = arena->getPtr();

    SortedEntry *currEntry = sortedEntries + numMisses;
    while (currEntry != endEntry) {

        const uint32_t currBundledMatId = MNRY_VERIFY(currEntry->mSortKey);

        SortedEntry *spanEnd = currEntry + 1;
        while (spanEnd != endEntry && spanEnd->mSortKey == currBundledMatId) {
            ++spanEnd;
        }

        // Create entries for shade queue.
        unsigned numShadeEntries = MNRY_VERIFY(spanEnd - currEntry);
        shading::SortedRayState *shadeEntries = arena->allocArray<shading::SortedRayState>(numShadeEntries,
                                                                                           CACHE_LINE_SIZE);

        for (unsigned i = 0; i < numShadeEntries; ++i) {
            uint32_t rsIdx = currEntry[i].mRsIdx;
            RayState *rs = &baseRayState[rsIdx];
            const mcrt_common::Ray &ray = rs->mRay;
            shadeEntries[i].mRsIdx = rsIdx;

            // Sort first by geometry and then by primitive within that geometry.
            // This is to improve locality for postIntersection calls.
            shadeEntries[i].mSortKey = ((ray.geomID & 0xfff) << 20) | (ray.primID & 0xfffff);
        }

        // Submit to destination queue.
        shading::ShadeQueue *shadeQueue = MNRY_VERIFY(currEntry->mMaterial->getShadeQueue());
        shadeQueue->addEntries(tls, numShadeEntries, shadeEntries, arena);

        CHECK_CANCELLATION(pbrTls, return);

        // Free up entry memory from arena.
        arena->setPtr(memBookmark);

        currEntry = spanEnd;
    }
}

void
xpuOcclusionQueryBundleHandler(mcrt_common::ThreadLocalState *tls,
                               unsigned numRays,
                               BundledOcclRay *rays,
                               const rt::GPURay *gpuRays,
                               std::atomic<int>& threadsUsingGPU)
{
    pbr::TLState *pbrTls = tls->mPbrTls.get();

    EXCL_ACCUMULATOR_PROFILE(pbrTls, EXCL_ACCUM_OCCL_QUERY_HANDLER);

    scene_rdl2::alloc::Arena *arena = &tls->mArena;
    SCOPED_MEM(arena);

    BundledRadiance *results = arena->allocArray<BundledRadiance>(numRays, CACHE_LINE_SIZE);

    unsigned numRadiancesFilled =
        computeXPUOcclusionQueriesOnGPU(tls,
                                        numRays,
                                        rays,
                                        gpuRays,
                                        results,
                                        threadsUsingGPU);

    MNRY_ASSERT(numRadiancesFilled <= numRays);

    CHECK_CANCELLATION(pbrTls, return);

    pbrTls->addRadianceQueueEntries(numRadiancesFilled, results);
}

#pragma warning pop

} // namespace pbr
} // namespace moonray
