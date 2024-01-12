// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

///
/// @file PathIntegratorVolume.cc
///

#include "PathIntegrator.h"
#include "PathIntegratorUtil.h"
#include "BsdfOneSampler.h"
#include "VolumeProperties.h"
#include "VolumeScatterEventSampler.h"
#include "VolumeScatteringSampler.h"
#include "VolumeTransmittance.h"
#include "VolumeEmissionBsdfSampler.h"
#include "VolumeEmissionLightSampler.h"
#include <moonray/rendering/pbr/core/Constants.h>
#include <moonray/rendering/pbr/core/DeepBuffer.h>
#include <moonray/rendering/pbr/core/PbrTLState.h>
#include <moonray/rendering/pbr/core/RayState.h>
#include <moonray/rendering/pbr/core/Scene.h>
#include <moonray/rendering/pbr/core/Util.h>
#include <moonray/rendering/pbr/core/VolumePhase.h>
#include <moonray/rendering/geom/prim/GeomTLState.h>
#include <moonray/rendering/geom/prim/Primitive.h>
#include <moonray/rendering/geom/prim/VolumeRegions.h>
#include <moonray/rendering/geom/prim/VolumeAssignmentTable.h>
#include <moonray/rendering/rt/EmbreeAccelerator.h>
#include <moonray/rendering/bvh/shading/AttributeTable.h>
#include <moonray/rendering/bvh/shading/Intersection.h>
#include <moonray/rendering/bvh/shading/ShadingTLState.h>
#include <moonray/rendering/shading/bsdf/Bsdf.h>
#include <moonray/rendering/shading/bsdf/Bsdfv.h>
#include <moonray/rendering/shading/Util.h>

#include <scene_rdl2/common/math/Constants.h>

// using namespace scene_rdl2::math; // can't use this as it breaks openvdb in clang.

namespace moonray {
namespace pbr {

static size_t
estimateStepCountSubinterval(const geom::internal::VolumeRegions& volumeRegions,
        int* volumeIds,
        const std::vector<geom::internal::VolumeSampleInfo>& volumeSampleInfo,
        float t0, float t1, int depth, float invVolumeQuality)
{
    int volumeRegionsCount = volumeRegions.getVolumeIds(volumeIds);
    if (volumeRegionsCount == 0) {
        return 0;
    }
    float minFeatureSize = scene_rdl2::math::inf;
    bool isHomogenous = true;
    for (int i = 0; i < volumeRegionsCount; ++i) {
        const auto& sampleInfo = volumeSampleInfo[volumeIds[i]];
        isHomogenous &= sampleInfo.isHomogenous();
        // when there are multiple volume regions in this interval,
        // use the smallest non-homogenous feature size for stepping
        // (stepping only happens when there is a non-homogenous volume region,
        //  and homogenous volumes do not have a valid feature size)
        if (!sampleInfo.isHomogenous()) {
            minFeatureSize = scene_rdl2::math::min(minFeatureSize, sampleInfo.getFeatureSize());
        }
    }
    if (isHomogenous) {
        return 1;
    }
    const float stepSize = minFeatureSize * invVolumeQuality * scene_rdl2::math::max(depth - 1, 1);
    // (t1 - t0) / stepSize + 1 should be tight bound, +2 for safety
    return static_cast<int>((t1 - t0) / stepSize) + 2;
}

static size_t
estimateStepCount(const mcrt_common::Ray& ray, size_t intervalCount,
        const geom::internal::VolumeTransition* intervals,
        int* volumeIds, float invVolumeQuality,
        const geom::internal::VolumeRayState& volumeRayState,
        size_t maxSteps)
{
    // we need to flip bits of volume regions while estimating the total
    // number of marching step, so make a copy instead of using the original one
    geom::internal::VolumeRegions volumeRegions =
        volumeRayState.getCurrentVolumeRegions();
    const auto& volumeSampleInfo = volumeRayState.getVolumeSampleInfo();
    size_t stepCount = 0;
    float tStart = ray.tnear;
    for (size_t i = 0; i < intervalCount; ++i) {
        if (intervals[i].mT > ray.tfar) {
            break;
        }
        if ((intervals[i].mT - tStart) > scene_rdl2::math::sEpsilon) {
            stepCount += estimateStepCountSubinterval(volumeRegions, volumeIds,
                volumeSampleInfo, tStart, intervals[i].mT,
                ray.getDepth(), invVolumeQuality);
            tStart = intervals[i].mT;
        }
        if (intervals[i].mIsEntry) {
            volumeRegions.turnOn(intervals[i].mVolumeId, intervals[i].mPrimitive);
        }
        if (intervals[i].mIsExit) {
            volumeRegions.turnOff(intervals[i].mVolumeId);
        }
        // Take no action here if intervals[i] is marked as neither an entry nor an exit.
        // This can happen when the volume has self-intersecting geometry. In such cases, the initial ray-trace
        // marks each intersection as an entry or an exit according to the surface's facing direction. But where
        // such intervals overlap, we only want to count the outermosst entry and exit points of the union of
        // the intervals - otherwise it is not possible to determine whether the ray origin is inside or outside
        // the volume. So all the interior entry/exit points have their entry/exit status switched off.
        // See MOONRAY-4292 for context.
    }

    if ((ray.tfar - tStart) > scene_rdl2::math::sEpsilon) {
        size_t lastStepCount = estimateStepCountSubinterval(volumeRegions, volumeIds,
            volumeSampleInfo, tStart,  ray.tfar,
            ray.getDepth(), invVolumeQuality);

        // Sometimes we hit the volume once, causing an enter event, but never hit anything again.
        // In this case, ray.tfar is very large and we request far too many steps. We should disregard
        // those steps.
        if (lastStepCount + stepCount < maxSteps) {
            stepCount += lastStepCount;
        }
    }
    return stepCount;
}

scene_rdl2::math::Color
PathIntegrator::estimateInScatteringSourceTerm(pbr::TLState *pbrTls, const mcrt_common::Ray& ray,
        const scene_rdl2::math::Vec3f& scatterPoint, const Light* light, int assignmentId,
        const VolumePhase& phaseFunction, const scene_rdl2::math::Vec3f& ul, const LightFilterRandomValues& ulFilter,
        const Subpixel &sp, unsigned sequenceID, float scaleFactor) const
{
    const LightFilterList* lightFilterList = pbrTls->mFs->mScene->getLightFilterList(assignmentId, light);
    if (!light->canIlluminate(scatterPoint, nullptr, ray.getTime(), /* radius = */ 0.f, lightFilterList)) {
        return scene_rdl2::math::Color(0.0f);
    }
    scene_rdl2::math::Color Ls(0.0f);
    LightIntersection lIsect;
    scene_rdl2::math::Vec3f lWi;
    // TODO: replace the 0.0f with a proper footprint value? (We currently have no RayDifferential available)
    if (light->sample(scatterPoint, nullptr, ray.getTime(), ul, lWi, lIsect, 0.0f)) {
        float pdfLight;
        // TODO: replace the 0.0f with a proper footprint value? (We currently have no RayDifferential available)
        scene_rdl2::math::Color Li = light->eval(pbrTls->mTopLevelTls,
            lWi, scatterPoint, ulFilter, ray.getTime(), lIsect, false, lightFilterList, 0.0f, &pdfLight);
        if (!isSampleInvalid(Li, pdfLight) && scene_rdl2::math::isfinite(pdfLight)) {
            mcrt_common::Ray shadowRay(scatterPoint, lWi, 0,
                sHitEpsilonEnd * lIsect.distance,
                ray.getTime(), ray.getDepth() + 1);
            float presence = 0.0f;
            if (!isRayOccluded(pbrTls, light, shadowRay,
                sHitEpsilonStart, 0.f /* shadow ray epsilon */, presence, assignmentId, true)) {
                // shadowRay can be modified in occlusion query
                mcrt_common::Ray trRay(scatterPoint, lWi, 0,
                    sHitEpsilonEnd * lIsect.distance,
                    ray.getTime(), ray.getDepth() + 1);
                scene_rdl2::math::Color trToLight =
                    transmittance(pbrTls, trRay,
                                  sp.mPixel, sp.mSubpixelIndex, sequenceID, light, scaleFactor, true);
                float ph = phaseFunction.eval(-ray.dir, lWi);
                Ls += (1.0f - presence) * trToLight * ph * Li /
                    pdfLight;
            }
        }
    }
    return Ls;
}

scene_rdl2::math::Color
PathIntegrator::equiAngularVolumeScattering(pbr::TLState *pbrTls,
        const mcrt_common::Ray& ray, int lightIndex,
        float ue, const scene_rdl2::math::Vec3f& ul, const LightFilterRandomValues& ulFilter,
        float D, float thetaA, float thetaB, float offset,
        const VolumeProperties* volumeProperties,
        const GuideDistribution1D& densityDistribution,
        const Subpixel &sp, unsigned& sequenceID, bool doMIS) const
{
    EXCL_ACCUMULATOR_PROFILE(pbrTls, EXCL_ACCUM_VOL_INTEGRATION);

    float te = sampleEquiAngular(ue, D, thetaA, thetaB);
    float pdfTe = pdfEquiAngular(te, D, thetaA, thetaB);
    if (pdfTe == 0.0f || !scene_rdl2::math::isfinite(pdfTe)) {
        return scene_rdl2::math::Color(0.0f);
    }
    float tePrime = te + offset;
    // figure out which block tePrime fall in
    uint32_t marchingStepsCount = densityDistribution.getSize();
    uint32_t densityIndex = geom::findInterval(marchingStepsCount + 1,
        [&](int index) {
            return volumeProperties[index].mTStart <= tePrime;
        });
    const auto& vp = volumeProperties[densityIndex];
    // no in-scattering at all
    if (isBlack(vp.mSigmaS)) {
        return scene_rdl2::math::Color(0.0f);
    }
    // the light is not active in the light set assigned to this volume region
    if (!pbrTls->mFs->mScene->isLightActive(vp.mAssignmentId, lightIndex)) {
        return scene_rdl2::math::Color(0.0f);
    }
    // the transmittance of this scatter point
    scene_rdl2::math::Color trScatter = vp.mTransmittance * vp.mTransmittanceH *
        exp(-vp.mSigmaT * (tePrime - vp.mTStart)) *
        exp(-vp.mSigmaTh * (tePrime - vp.mTStart));
    scene_rdl2::math::Vec3f scatterPoint = ray.org + ray.dir * tePrime;
    const Light* light = pbrTls->mFs->mScene->getLight(lightIndex);

    float misWeight = 1.0f;
    if (doMIS) {
        float pdfIndex = densityDistribution.pdfDiscrete(densityIndex);
        float pdfDistance = pdfDistanceExponential(tePrime,
            luminance(vp.mSigmaT), vp.mTStart, vp.mTStart + vp.mDelta);
        float pdfTd = pdfIndex * pdfDistance;
        if (scene_rdl2::math::isfinite(pdfTd)) {
            misWeight = powerHeuristic(pdfTe, pdfTd);
        }
    }
    scene_rdl2::math::Color Ls = estimateInScatteringSourceTerm(pbrTls, ray, scatterPoint,
        light, vp.mAssignmentId, VolumePhase(vp.mG), ul, ulFilter, sp, sequenceID);
    return misWeight * vp.mSigmaS * trScatter * Ls / pdfTe;
}

scene_rdl2::math::Color
PathIntegrator::distanceVolumeScattering(pbr::TLState *pbrTls,
        const mcrt_common::Ray& ray, int lightIndex,
        float ud, const scene_rdl2::math::Vec3f& ul, const LightFilterRandomValues& ulFilter,
        float D, float thetaA, float thetaB, float offset,
        const VolumeProperties* volumeProperties,
        const GuideDistribution1D& densityDistribution,
        const Subpixel &sp, unsigned& sequenceID, bool doMIS,
        float& deepTd, scene_rdl2::math::Color& radiance, scene_rdl2::math::Color& transmittance) const
{
    EXCL_ACCUMULATOR_PROFILE(pbrTls, EXCL_ACCUM_VOL_INTEGRATION);

    radiance = scene_rdl2::math::Color(0.0f);

    float pdfDensity, udRemapped;
    int densityIndex = densityDistribution.sampleDiscrete(
        ud, &pdfDensity, &udRemapped);
    if (pdfDensity == 0.0f) {
        return scene_rdl2::math::Color(0.0f);
    }
    const auto& vp = volumeProperties[densityIndex];
    // no in-scattering at all
    if (isBlack(vp.mSigmaS)) {
        return scene_rdl2::math::Color(0.0f);
    }
    // the light is not active in the light set assigned to this volume region
    if (!pbrTls->mFs->mScene->isLightActive(vp.mAssignmentId, lightIndex)) {
        return scene_rdl2::math::Color(0.0f);
    }
    float sigmaT = luminance(vp.mSigmaT);
    float t0 = vp.mTStart;
    float t1 = vp.mTStart + vp.mDelta;
    float td = sampleDistanceExponential(udRemapped, sigmaT, t0, t1);
    float pdfTd = pdfDensity * pdfDistanceExponential(td, sigmaT, t0, t1);
    if (pdfTd == 0.0f || !scene_rdl2::math::isfinite(pdfTd)) {
        return scene_rdl2::math::Color(0.0f);
    }
    // the transmittance of this scatter point
    scene_rdl2::math::Color trScatter = vp.mTransmittance * vp.mTransmittanceH *
        exp(-vp.mSigmaT * (td - t0)) * exp(-vp.mSigmaTh * (td - t0));
    scene_rdl2::math::Vec3f scatterPoint = ray.org + ray.dir * td;
    const Light* light = pbrTls->mFs->mScene->getLight(lightIndex);

    float misWeight = 1.0f;
    if (doMIS) {
        float pdfTe = pdfEquiAngular(td - offset, D, thetaA, thetaB);
        if (scene_rdl2::math::isfinite(pdfTe)) {
            misWeight = powerHeuristic(pdfTd, pdfTe);
        }
    }
    scene_rdl2::math::Color Ls = estimateInScatteringSourceTerm(pbrTls, ray, scatterPoint,
        light, vp.mAssignmentId, VolumePhase(vp.mG), ul, ulFilter, sp, sequenceID);

    // For deep volume support, we need to extract the t distance, radiance,
    //  and transmittance.
    deepTd = td;
    radiance = vp.mSigmaS * Ls / pdfTd;
    transmittance = trScatter;

    return misWeight * vp.mSigmaS * trScatter * Ls / pdfTd;
}

scene_rdl2::math::Color
PathIntegrator::integrateVolumeScattering(pbr::TLState *pbrTls, const mcrt_common::Ray& ray,
        const VolumeProperties* volumeProperties,
        const GuideDistribution1D& densityDistribution,
        const Subpixel &sp, const PathVertex& pv, unsigned& sequenceID,
        float* aovs, DeepParams* deepParams, const RayState *rs) const
{
    EXCL_ACCUMULATOR_PROFILE(pbrTls, EXCL_ACCUM_VOL_INTEGRATION);
    // now drawing distance/lighting sample for scattering
    bool highQualitySample = pv.nonMirrorDepth == 0;
    const Scene *scene = MNRY_VERIFY(pbrTls->mFs->mScene);
    // we loop through "every" light in the scene for volume scattering
    // integration instead of light set assigned to particular volume region.
    // The main reason is equi-angular sampling needs to use a sampled light as
    // pivot point to draw distance samples and we don't know which light set
    // cover the whole volume regions across the ray segments. Once we know
    // which volume region the scattering point lands on, we then query whether
    // one particular light is active in the light set assigned to this
    // volume region. (only active light in that light set will contribute
    // in-scattering radiance)
    int lightCount = scene->getLightCount();
    int samplesPerLight = highQualitySample ? mLightSamples : 1;
    int scatterSampleCount = highQualitySample ? mVolumeIlluminationSamples : 1;
    if (scatterSampleCount <= 0) {
        return scene_rdl2::math::sBlack;
    }
    scene_rdl2::math::Color LSingleScatter(0.0f);
    float invN = 1.0f / (samplesPerLight * scatterSampleCount);
    for (int lightIndex = 0; lightIndex < lightCount; ++lightIndex) {
        const Light* light = scene->getLight(lightIndex);
        // equi-angular sampling is not effective for infinite lights
        // no eqi-angular sampling for deep volumes
        bool doEquiAngular = light->isBounded() && !deepParams;
        VolumeScatteringSampler volumeScatteringSampler(sp, pv,
            samplesPerLight, scatterSampleCount, *light,
            highQualitySample, doEquiAngular, sequenceID);
        scene_rdl2::math::Color LDirect(0.0f);
        for (int ls = 0; ls < samplesPerLight; ++ls) {
            for (int ts = 0; ts < scatterSampleCount; ++ts) {
                scene_rdl2::math::Vec3f pivot(0.0f);
                float offset = 0.0f;
                float D = 0.0f;
                float thetaA = 0.0f;
                float thetaB = 0.0f;
                if (doEquiAngular) {
                    // intentionally reuse sample for light scattering to query
                    // pivot point (this is what the original paper do too)
                    float ue;
                    scene_rdl2::math::Vec3f uel;
                    LightFilterRandomValues uelFilter;
                    volumeScatteringSampler.getEquiAngularSample(ue, uel, uelFilter);
                    pivot = light->getEquiAngularPivot(uel, ray.getTime());
                    // project pivot point to ray
                    offset = scene_rdl2::math::dot(pivot - ray.org, ray.dir);
                    // the distance from pivot point to ray
                    D = length(pivot - (ray.org + ray.dir * offset));
                    const auto& firstVp = volumeProperties[0];
                    // TODO it is possible that we can further clipping the
                    // sample range by light plane, cone angle...etc
                    float ta = firstVp.mTStart - offset;
                    uint32_t marchingStepsCount = densityDistribution.getSize();
                    const auto& lastVp =
                        volumeProperties[marchingStepsCount - 1];
                    float tb = lastVp.mTStart + lastVp.mDelta - offset;
                    thetaA = scene_rdl2::math::atan2(ta, D);
                    thetaB = scene_rdl2::math::atan2(tb, D);
                    scene_rdl2::math::Color contribution = invN * pv.pathThroughput *
                        equiAngularVolumeScattering(
                        pbrTls, ray, lightIndex, ue, uel, uelFilter, D,
                        thetaA, thetaB, offset, volumeProperties,
                        densityDistribution, sp, sequenceID, true);
                    // we always do distance sampling so set doMIS to true
                    LDirect += contribution;
                }
                float ud;
                scene_rdl2::math::Vec3f udl;
                LightFilterRandomValues udlFilter;
                volumeScatteringSampler.getDistanceSample(ud, udl, udlFilter);
                // if no equi-angular sampling, then no MIS
                // (only distance sampling)

                float td;
                scene_rdl2::math::Color radiance;
                scene_rdl2::math::Color transmittance;

                scene_rdl2::math::Color contribution = invN * pv.pathThroughput *
                    distanceVolumeScattering(
                    pbrTls, ray, lightIndex, ud, udl, udlFilter, D,
                    thetaA, thetaB, offset, volumeProperties,
                    densityDistribution, sp, sequenceID, doEquiAngular,
                    td, radiance, transmittance);
                LDirect += contribution;

                if (deepParams) {
                    scene_rdl2::math::Color deepContribution = radiance * invN;
                    if (deepParams->mVolumeAovs) {
                        const FrameState &fs = *pbrTls->mFs;
                        const LightAovs &lightAovs = *fs.mLightAovs;
                        // transition
                        int lpeStateId = pv.lpeStateId;
                        lpeStateId = lightAovs.volumeEventTransition(pbrTls, lpeStateId);
                        lpeStateId = lightAovs.lightEventTransition(pbrTls, lpeStateId, light);
                        // retrieve AOV data for this sample
                        fs.mAovSchema->initFloatArray(deepParams->mVolumeAovs);
                        aovAccumLightAovs(pbrTls, *fs.mAovSchema, *fs.mLightAovs,
                            deepContribution, nullptr, AovSchema::sLpePrefixNone, lpeStateId, deepParams->mVolumeAovs);
                    }
                    deepParams->mDeepBuffer->addVolumeSample(
                        pbrTls,
                        deepParams->mPixelX,
                        deepParams->mPixelY,
                        td,
                        transmittance,
                        deepContribution,
                        deepParams->mVolumeAovs);
                }
            }
        }
        LSingleScatter += LDirect;
        // LPE
        if (pbrTls->mFs->mLightAovs->hasEntries() && (aovs || rs)) {
            EXCL_ACCUMULATOR_PROFILE(pbrTls, EXCL_ACCUM_AOVS);
            const FrameState &fs = *pbrTls->mFs;
            const LightAovs &lightAovs = *fs.mLightAovs;
            // transition
            int lpeStateId = pv.lpeStateId;
            lpeStateId = lightAovs.volumeEventTransition(pbrTls, lpeStateId);
            lpeStateId = lightAovs.lightEventTransition(pbrTls, lpeStateId, light);
            // accumulate matching aovs
            if (aovs) {
                aovAccumLightAovs(pbrTls, *fs.mAovSchema, *fs.mLightAovs,
                    LDirect, nullptr, AovSchema::sLpePrefixNone, lpeStateId, aovs);
            } else {
                MNRY_ASSERT(rs && fs.mExecutionMode == mcrt_common::ExecutionMode::VECTORIZED);
                aovAccumLightAovsBundled(pbrTls, *fs.mAovSchema, lightAovs,
                    LDirect, nullptr, AovSchema::sLpePrefixNone, lpeStateId,
                    sp.mPixel, rs->mDeepDataHandle);
            }
        }
    }
    // treat emissive volumes as light sources and compute their in-scattering
    // contribution, similar to what we do on bsdf/bssrdf surface reflection
    LSingleScatter += computeRadianceEmissiveRegionsVolumes(pbrTls, sp, pv, ray,
        volumeProperties, densityDistribution, sequenceID, aovs, rs);
    return LSingleScatter;
}

static size_t
collectVolumeIntervals(pbr::TLState *pbrTls, const mcrt_common::Ray& ray, int rayMask,
                       const void* light = nullptr, bool estimateInScatter = false)
{
    const Scene *scene = MNRY_VERIFY(pbrTls->mFs->mScene);
    if (!scene->hasVolume()) {
        return 0;
    }

    EXCL_ACCUMULATOR_PROFILE(pbrTls, EXCL_ACCUM_VOL_INTEGRATION);

    mcrt_common::ThreadLocalState* tls = pbrTls->mTopLevelTls;
    // collect all volume hits along the ray
    scene->intersectVolumes(tls, ray, rayMask, light, estimateInScatter);
    auto& volumeRayState = tls->mGeomTls->mVolumeRayState;
    scene_rdl2::alloc::Arena* arena = &(tls->mArena);
    SCOPED_MEM(arena);

    size_t intervalCount = volumeRayState.getIntervalCount();
    if (intervalCount == 0) {
        return 0;
    }

    geom::internal::VolumeTransition* intervals = volumeRayState.getVolumeIntervals();
    // sort the volume intervals
    intervals = scene_rdl2::util::smartSort32<geom::internal::VolumeTransition>(
            intervalCount, intervals, 0xffffffff, arena);

    // Eliminate the spurious duplicate intersections which Embree generates
    intervalCount = std::unique(intervals, intervals + intervalCount) - intervals;

    // TODO: rewrite this comment
    // the following interiority check is not necessary if every geometry
    // with volume assignment is closed manifold. Unfortunately, for cases
    // that a mesh has faces cross over each other, the volume region check
    // in earlier BVH traversal time can result to weird artifact like:
    // org      Front      Back       Back
    // -------->|--------->|--------->|---------->
    // air        volGeom    volGeom    air
    //
    // the ray casting algorithm would resolve ray origin inside the volume
    // since it hits the geometry with odd number(3 in this case) intersections
    //
    // we try our best to fix above case by checking the nearest intersection
    // for each intersected volume, if it's front faces then we assume
    // ray origin is out of this volume (vice versa). there are still
    // potentially other artifacts that can be introduced due to assign volume
    // to open geometry though...

    // Note: the following section has been rewritten to support self-penetrating geometry, as requested by
    // MOONRAY-4292.
    // The original version of this code examined the intersections starting at the ray origin, and used the
    // facing direction of the first hit to determine whether the ray origin was inside the volume.
    // With self-intersecting geometry, though, that test is no longer sufficient to determine if the origin is
    // in the volume, since a front-facing surface might not indicate the first entry into the volume.
    // To remedy this situation, we keep a count of how many times the ray has entered each volume (the array
    // 'interiorityCount') in order to switch off the entry and exit status of all but the outermost ones.
    // We must also now trace from the far end of the ray (which we know is outside all volumes) backwards towards
    // the ray origin, since we don't know the interiority count of the ray origin upfront.

    int volumeCount = volumeRayState.getVolumeAssignmentTable()->getVolumeCount();
    int* interiorityCount = arena->allocArray<int>(volumeCount);
    memset(interiorityCount, 0, sizeof(int) * volumeCount);

    bool hasRedundantIntersections = false;
    for (int i = intervalCount - 1; i >= 0; --i) {
        int volumeId = intervals[i].mVolumeId;

        if (intervals[i].mIsEntry) {
            // Crossing backwards out of an entry point
            --interiorityCount[volumeId];
            if (interiorityCount[volumeId] == 0) {
                // Leaving outermost entry point, so no longer inside this volume, so switch it off
                volumeRayState.turnOff(volumeId);
            } else {
                // Leaving an entry point but still inside volume
                if (interiorityCount[volumeId] > 0) {
                    // Switch of its "entry" status so only the outermost entry will remain
                    intervals[i].mIsEntry = false;
                    hasRedundantIntersections = true;
                }
            }
        }

        if (intervals[i].mIsExit) {
            // Crossing backwards into an exit point
            if (interiorityCount[volumeId] == 0) {
                // Entering outermost exit point, so now inside this volume, so switch it on
                volumeRayState.turnOn(volumeId, intervals[i].mPrimitive);
            } else {
                // Entering an exit point but we were already inside inside volume
                if (interiorityCount[volumeId] > 0) {
                    // Switch off its "exit" status so only the outermost exit will remain
                    intervals[i].mIsExit = false;
                    hasRedundantIntersections = true;
                }
            }
            ++interiorityCount[volumeId];
        }

    }

    // Strip out intersections which don't have entry or exit status
    if (hasRedundantIntersections) {
        int reducedIntervalCount = 0;
        for (int i = 0; i < intervalCount; i++) {
            if (intervals[i].mIsEntry || intervals[i].mIsExit) {
                intervals[reducedIntervalCount++] = intervals[i];
            }
        }
        return reducedIntervalCount;
    }

    return intervalCount;
}

//==---------------------------------------------------------------------------
// Volume shader evaluation.
// Conceptually there are many different ways to evaluate volume shaders in
// ovelapping regions:
//   SUM: add all transmittance and scattering contributions
//   MAX: only consider the volume with maximum transmittance (i.e. the thickest)
//   RND: randomly pick a volume from a distribution based on transmittance
//
// If the overlap mode is RND, then rndVar is used to select the volume
// from the distribution.  Otherwise it is unused.
// sigmaT, sigmaS, and g are single values computed based on overlap mode.
// emission is returned per volume if mode is SUM, otherwise it is just the
// emission of the chosen volume
// If the mode is MAX or RND then the chosen volume id is returned in volumeIdSampled,
// otherwise it is set to -1.
//

static void
evalVolumeShaders(pbr::TLState *pbrTls, int volumeRegionsCount, int* volumeIds,
        VolumeOverlapMode overlapMode, float rndVar,
        const std::vector<geom::internal::VolumeSampleInfo>& volumeSampleInfo,
        float t, float time, bool enableCutouts,
        scene_rdl2::math::Color* sigmaT, scene_rdl2::math::Color* sigmaS, scene_rdl2::math::Color* sigmaTh,
        scene_rdl2::math::Color* sigmaSh, float* surfaceOpacityThreshold, float* g,
        scene_rdl2::math::Color* emission, int* volumeIdSampled, float rayVolumeDepth)
{
    MNRY_ASSERT(volumeRegionsCount > 0);
    const bool highQuality = true;

    *sigmaT = scene_rdl2::math::Color(0.0f);
    *sigmaS = scene_rdl2::math::Color(0.0f);
    *sigmaTh = scene_rdl2::math::Color(0.0f);
    *sigmaSh = scene_rdl2::math::Color(0.0f);
    *surfaceOpacityThreshold = 1.f;
    *volumeIdSampled = -1;
    scene_rdl2::math::Color anisotropy(0.0f);

    scene_rdl2::math::Color *sigmaTs = nullptr;
    scene_rdl2::math::Color *sigmaSs = nullptr;
    scene_rdl2::math::Color *sigmaThs = nullptr;
    scene_rdl2::math::Color *sigmaShs = nullptr;
    scene_rdl2::math::Color *anisotropies = nullptr;
    scene_rdl2::math::Color *emissions = nullptr;
    float *surfaceOpacityThresholds = nullptr;
    float *cdf = nullptr;

    scene_rdl2::alloc::Arena *arena = pbrTls->mArena;
    SCOPED_MEM(arena);
    mcrt_common::ThreadLocalState* tls = pbrTls->mTopLevelTls;
    shading::TLState *shadingTls = tls->mShadingTls.get();

    if (overlapMode == VolumeOverlapMode::RND) {
        // scratch space needed for the distribution
        sigmaTs = arena->allocArray<scene_rdl2::math::Color>(volumeRegionsCount);
        sigmaSs = arena->allocArray<scene_rdl2::math::Color>(volumeRegionsCount);
        sigmaThs = arena->allocArray<scene_rdl2::math::Color>(volumeRegionsCount);
        sigmaShs = arena->allocArray<scene_rdl2::math::Color>(volumeRegionsCount);
        anisotropies = arena->allocArray<scene_rdl2::math::Color>(volumeRegionsCount);
        emissions = arena->allocArray<scene_rdl2::math::Color>(volumeRegionsCount);
        surfaceOpacityThresholds = arena->allocArray<float>(volumeRegionsCount);
        cdf = arena->allocArray<float>(volumeRegionsCount);
    }

    scene_rdl2::math::Color sigmaTSum(0.f);

    float maxSigmaTLum = scene_rdl2::math::neg_inf;

    for (int i = 0; i < volumeRegionsCount; ++i) {
        const geom::internal::VolumeSampleInfo& sampleInfo =
            volumeSampleInfo[volumeIds[i]];
        const scene_rdl2::rdl2::VolumeShader* volumeShader = sampleInfo.getShader();
        bool isCutout = enableCutouts && volumeShader->isCutout();
        scene_rdl2::math::Vec3f p = sampleInfo.getSamplePosition(t);
        scene_rdl2::math::Color extinction, albedo;
        scene_rdl2::math::Color temperature(0.0f);
        scene_rdl2::math::Color* temperaturePtr = nullptr;
        if ((volumeShader->getProperties() & scene_rdl2::rdl2::VolumeShader::IS_EMISSIVE) && emission) {
            temperaturePtr = &temperature;
        }

        shading::Intersection isect;
        auto& volumeRayState = tls->mGeomTls->mVolumeRayState;
        const geom::internal::Primitive* prim = volumeRayState.getCurrentVolumeRegions().getPrimitive(volumeIds[i]);
        isect.init(prim->getRdlGeometry());
        const scene_rdl2::math::Vec3f evalP = prim->evalVolumeSamplePosition(tls, volumeIds[i], p, time);
        isect.setP(prim->transformVolumeSamplePosition(evalP, time));
        const shading::State state(&isect);
        prim->evalVolumeCoefficients(tls, volumeIds[i], evalP,
            &extinction, &albedo, temperaturePtr, highQuality, rayVolumeDepth, volumeShader);
        scene_rdl2::math::Color sigmaTLocal = extinction;

        switch (overlapMode) {
        case VolumeOverlapMode::RND:
            // random behavior
            {
                // build a cdf with sigmaT
                sigmaTSum += sigmaTLocal;
                cdf[i] = scene_rdl2::math::luminance(sigmaTSum);
                scene_rdl2::math::Color scatter(volumeShader->albedo(shadingTls, state, albedo, rayVolumeDepth) * sigmaTLocal);
                if (!isCutout) {
                    sigmaTs[i] = sigmaTLocal;
                    sigmaThs[i] = scene_rdl2::math::Color(0.f);
                    sigmaSs[i] = scatter;
                    sigmaShs[i] = scene_rdl2::math::Color(0.f);
                    if (emission) {
                        emissions[i] = volumeShader->emission(shadingTls, state, temperature);
                    }
                    anisotropies[i] = scatter * volumeShader->anisotropy(shadingTls, state);
                } else {
                    sigmaTs[i] = scene_rdl2::math::Color(0.f);
                    sigmaThs[i] = sigmaTLocal;
                    sigmaSs[i] = scene_rdl2::math::Color(0.f);
                    sigmaShs[i] = scatter;
                    anisotropies[i] = scene_rdl2::math::Color(0.f);
                    if (emission) {
                        emissions[i] = scene_rdl2::math::Color(0.f);
                    }
                    anisotropies[i] = scene_rdl2::math::Color(0.f);
                }
                surfaceOpacityThresholds[i] = volumeShader->surfaceOpacityThreshold();
            }
            break;
        case VolumeOverlapMode::MAX:
            // max behavior
            {
                const float sigmaTLumLocal = scene_rdl2::math::luminance(sigmaTLocal);
                if (sigmaTLumLocal > maxSigmaTLum) {
                    maxSigmaTLum = sigmaTLumLocal;
                    scene_rdl2::math::Color scatter(volumeShader->albedo(shadingTls, state, albedo, rayVolumeDepth) * sigmaTLocal);
                    if (!isCutout) {
                        *sigmaT = sigmaTLocal;
                        *sigmaTh = scene_rdl2::math::Color(0.f);
                        *sigmaS = scatter;
                        *sigmaSh = scene_rdl2::math::Color(0.f);
                        if (emission) {
                            *emission = volumeShader->emission(shadingTls, state, temperature);
                        }
                        anisotropy = scatter * volumeShader->anisotropy(shadingTls, state);
                    } else {
                        *sigmaT = scene_rdl2::math::Color(0.f);
                        *sigmaTh = sigmaTLocal;
                        *sigmaS = scene_rdl2::math::Color(0.f);
                        *sigmaSh = scatter;
                        if (emission) {
                            *emission = scene_rdl2::math::Color(0.f);
                        }
                        anisotropy = scene_rdl2::math::Color(0.f);
                    }
                    *surfaceOpacityThreshold = volumeShader->surfaceOpacityThreshold();
                    *volumeIdSampled = volumeIds[i];
                }
            }
            break;
        case VolumeOverlapMode::SUM:
            // old behavior
            {
                scene_rdl2::math::Color scatter(volumeShader->albedo(shadingTls, state, albedo, rayVolumeDepth) * sigmaTLocal);

                if (!isCutout) {
                    *sigmaT += sigmaTLocal;
                    *sigmaS += scatter;
                    if (emission) {
                        emission[volumeIds[i]] = volumeShader->emission(shadingTls, state, temperature);
                    }
                    anisotropy += scatter * volumeShader->anisotropy(shadingTls, state);
                } else {
                    *sigmaTh += sigmaTLocal;
                    *sigmaSh += scatter;
                    if (emission) {
                        emission[volumeIds[i]] = scene_rdl2::math::Color(0.f);
                    }
                }

                // The volume id we pick is from the volume with the maximum sigmaT
                const float sigmaTLumLocal = scene_rdl2::math::luminance(sigmaTLocal);
                if (sigmaTLumLocal > maxSigmaTLum) {
                    maxSigmaTLum = sigmaTLumLocal;
                    *surfaceOpacityThreshold = volumeShader->surfaceOpacityThreshold();
                    *volumeIdSampled = volumeIds[i];
                }
            }
            break;
        }
    }

    if (overlapMode == VolumeOverlapMode::RND) {
        // if we are in a vacuum, we are done
        if (scene_rdl2::math::luminance(sigmaTSum) == 0.f) {
            *sigmaT = sigmaTs[0];
            *sigmaS = sigmaSs[0];
            *sigmaTh = sigmaThs[0];
            *sigmaSh = sigmaShs[0];
            anisotropy = anisotropies[0];
            if (emission) {
                *emission = emissions[0];
            }
            *volumeIdSampled = volumeIds[0];
            *g = scene_rdl2::math::luminance(anisotropy);
            float normalization = scene_rdl2::math::luminance(*sigmaS);
            if (normalization > 0.0f) {
                *g /= normalization;
            }
            return;
        }

        // normalize the cdf
        for (int i = 0; i < volumeRegionsCount; ++i) {
            cdf[i] /= scene_rdl2::math::luminance(sigmaTSum);
        }

        // choose a volume from the cdf with rndVar
        int volIdx;
        for (volIdx = 0; volIdx < volumeRegionsCount; volIdx++) {
            if (rndVar < cdf[volIdx]) break;
        }

        // Need to adjust the values to compensate for the fact that we are
        // choosing only one volume, but all the volumes have an effect in the region.
        // Kind of similar to russian roulette logic.
        // If we choose a volume 10% of the time, it needs to have its contribution
        // boosted when we choose it because its contribution is missing for the other
        // 90%.
        float weight = scene_rdl2::math::luminance(sigmaTSum) /
                       scene_rdl2::math::luminance(sigmaTs[volIdx] + sigmaThs[volIdx]);
        *sigmaT = sigmaTs[volIdx] * weight;
        *sigmaS = sigmaSs[volIdx] * weight;
        *sigmaTh = sigmaThs[volIdx] * weight;
        *sigmaSh = sigmaShs[volIdx] * weight;
        anisotropy = anisotropies[volIdx] * weight;
        if (emission) {
            *emission = emissions[volIdx] * weight;
        }
        *surfaceOpacityThreshold = surfaceOpacityThresholds[volIdx];
        *volumeIdSampled = volumeIds[volIdx];
    }

    *g = scene_rdl2::math::luminance(anisotropy);
    float normalization = scene_rdl2::math::luminance(*sigmaS);
    if (normalization > 0.0f) {
        *g /= normalization;
    }
}

static scene_rdl2::math::Color
evalSigmaT(pbr::TLState *pbrTls, int volumeRegionsCount, int* volumeIds,
        VolumeOverlapMode overlapMode, float rndVar,
        const std::vector<geom::internal::VolumeSampleInfo>& volumeSampleInfo,
        float t, float time, const Light* light, float rayVolumeDepth)
{
    scene_rdl2::math::Color sigmaT(0.0f);

    scene_rdl2::math::Color *sigmaTs = nullptr;
    float *cdf = nullptr;

    scene_rdl2::alloc::Arena *arena = pbrTls->mArena;
    SCOPED_MEM(arena);

    if (overlapMode == VolumeOverlapMode::RND) {
        // scratch space needed for the distribution
        sigmaTs = arena->allocArray<scene_rdl2::math::Color>(volumeRegionsCount);
        cdf = arena->allocArray<float>(volumeRegionsCount);
    }

    scene_rdl2::math::Color sigmaTSum(0.f);

    mcrt_common::ThreadLocalState* tls = pbrTls->mTopLevelTls;
    const geom::internal::VolumeAssignmentTable* vTable = tls->mGeomTls->
            mVolumeRayState.getVolumeAssignmentTable();

    for (int i = 0; i < volumeRegionsCount; ++i) {
        if (light) {
            bool eval = true;
            auto& volumeRayState = tls->mGeomTls->mVolumeRayState;
            int originVolumeId = volumeRayState.getOriginVolumeId();
            if (originVolumeId == geom::internal::VolumeRayState::ORIGIN_VOLUME_INIT) {
                // non shadow volume occlusion
                if (!vTable->lookupShadowLinkingWithVolumeId(volumeIds[i]).
                    canCastShadow(light->getRdlLight())) {
                    eval = false;
                }
            } else {
                // shadow volume occlusion
                if (volumeIds[i] != originVolumeId) {
                    if (!vTable->lookupShadowLinkingWithVolumeId(volumeIds[i]).
                        canCastShadow(light->getRdlLight())) {
                        eval = false;
                    }
                }
            }
            if (!eval) {
                continue;
            }
        }
        const geom::internal::VolumeSampleInfo& sampleInfo =
            volumeSampleInfo[volumeIds[i]];
        unsigned int property = sampleInfo.getProperties();
        scene_rdl2::math::Vec3f p = sampleInfo.getSamplePosition(t);

        if (property & scene_rdl2::rdl2::VolumeShader::IS_EXTINCTIVE) {
            auto& volumeRayState = tls->mGeomTls->mVolumeRayState;
            const geom::internal::Primitive* prim = volumeRayState.getCurrentVolumeRegions().getPrimitive(volumeIds[i]);
            const scene_rdl2::math::Vec3f evalP = prim->evalVolumeSamplePosition(tls, volumeIds[i], p, time);

            const scene_rdl2::rdl2::VolumeShader* volumeShader = sampleInfo.getShader();
            scene_rdl2::math::Color density = prim->evalDensity(tls, volumeIds[i], evalP, rayVolumeDepth, volumeShader);
            switch (overlapMode) {
            case VolumeOverlapMode::RND:
                {
                    // random behavior
                    scene_rdl2::math::Color sigmaT = density;
                    sigmaTs[i] = sigmaT;
                    sigmaTSum += sigmaT;
                }
                break;
            case VolumeOverlapMode::MAX:
                {
                    // max behavior
                    scene_rdl2::math::Color shaderSigmaT = density;
                    if (scene_rdl2::math::luminance(shaderSigmaT) > scene_rdl2::math::luminance(sigmaT)) {
                        sigmaT = shaderSigmaT;
                    }
                }
                break;
            case VolumeOverlapMode::SUM:
                // sum behavior
                sigmaT += density;
                break;
            }
        }

        if (overlapMode == VolumeOverlapMode::RND) {
            cdf[i] = scene_rdl2::math::luminance(sigmaTSum);
        }
    }

    if (overlapMode == VolumeOverlapMode::RND) {
        // Build a normalized cdf based on the overlapping segment's sigmaT values
        if (scene_rdl2::math::luminance(sigmaTSum) == 0.f) { return sigmaTs[0]; }

        for (int i = 0; i < volumeRegionsCount; ++i) {
            cdf[i] /= scene_rdl2::math::luminance(sigmaTSum);
        }

        // Use the random variable to choose the volume
        int volIdx;
        for (volIdx = 0; volIdx < volumeRegionsCount; volIdx++) {
            if (rndVar < cdf[volIdx]) break; // using <= instead of < can give Nans if rndVar == 0 and cdf[0] == 0
        }

        sigmaT = sigmaTs[volIdx] / (scene_rdl2::math::luminance(sigmaTs[volIdx]) / scene_rdl2::math::luminance(sigmaTSum));
    }

    return sigmaT;
}

// Volume shader evaluation
//==---------------------------------------------------------------------------

scene_rdl2::math::Color
PathIntegrator::approximateVolumeMultipleScattering(pbr::TLState *pbrTls,
        const mcrt_common::Ray& ray, const VolumeProperties* volumeProperties,
        const GuideDistribution1D& densityDistribution,
        const Subpixel &sp, const PathVertex& pv, const int rayMask,
        unsigned sequenceID, float* aovs, DeepParams* deepParams, const RayState *rs) const
{
    // The approximation implementation is based on
    // Wrenninge M. 15 "Art-Directable Multiple Volumetric Scattering"
    // it uses primary ray's existing densityDistribution to draw
    // scatter events(illustrated as * below), and then bounce scatter events
    // around through "approximated" Woodcock tracking
    //
    //     *
    //      \            * --- *
    //       *          /     /
    //       |         *     /
    //       |              /               primary ray
    // --*---*-----*-------*-------------------------->
    //  /         /
    // *         /     *-------*
    // |        /      |
    // |       *-------*
    // *
    const Scene *scene = MNRY_VERIFY(pbrTls->mFs->mScene);
    int lightCount = scene->getLightCount();
    int scatterSampleCount = mVolumeIlluminationSamples;
    // use these samples to decide how far ray goes before
    // the next scatter event happens
    IntegratorSample1D freePathSamples;
    SequenceIDIntegrator freePathSid(pv.nonMirrorDepth, sp.mPixel, pv.volumeDepth, SequenceType::VolumeDistance, sequenceID);
    freePathSamples.resume(freePathSid,
        sp.mSubpixelIndex * scatterSampleCount *
        (mMaxVolumeDepth - pv.volumeDepth + 1));

    IntegratorSample1D freePathSamples2;
    SequenceIDIntegrator freePathSid2(pv.nonMirrorDepth, sp.mPixel, pv.volumeDepth, SequenceType::IndexSelection, sequenceID);
    freePathSamples2.resume(freePathSid2,
        sp.mSubpixelIndex * scatterSampleCount *
        (mMaxVolumeDepth - pv.volumeDepth + 1));

    // use these samples to decide how a scatter event alters the ray direction
    IntegratorSample2D phaseSamples;
    SequenceIDIntegrator phaseSid(pv.nonMirrorDepth, sp.mPixel, pv.volumeDepth, SequenceType::VolumePhase, sequenceID);
    phaseSamples.resume(phaseSid,
        sp.mSubpixelIndex * scatterSampleCount *
        (mMaxVolumeDepth - pv.volumeDepth + 1));
    // use these samples to estimate light contribution on each scatter event
    // (light strategy)
    IntegratorSample3D scatterLightSamples;
    SequenceIDIntegrator scatterLightSid(pv.nonMirrorDepth, sp.mPixel,
        pv.volumeDepth, SequenceType::VolumeScattering, SequenceType::Light, sequenceID);
    scatterLightSamples.resume(scatterLightSid,
            sp.mSubpixelIndex * scatterSampleCount *
            (mMaxVolumeDepth - pv.volumeDepth) * lightCount);
    IntegratorSample2D scatterLightFilterSamples;
    SequenceIDIntegrator scatterLightFilterSid(pv.nonMirrorDepth, sp.mPixel,
        pv.volumeDepth, SequenceType::VolumeScattering, SequenceType::LightFilter, sequenceID);
    scatterLightFilterSamples.resume(scatterLightFilterSid,
            sp.mSubpixelIndex * scatterSampleCount *
            (mMaxVolumeDepth - pv.volumeDepth) * lightCount);
    IntegratorSample3D scatterLightFilterSamples3D;
    SequenceIDIntegrator scatterLightFilter3DSid(pv.nonMirrorDepth, sp.mPixel,
        pv.volumeDepth, SequenceType::VolumeScattering, SequenceType::LightFilter3D, sequenceID);
    scatterLightFilterSamples3D.resume(scatterLightFilter3DSid,
            sp.mSubpixelIndex * scatterSampleCount *
            (mMaxVolumeDepth - pv.volumeDepth) * lightCount);
    mcrt_common::ThreadLocalState* tls = pbrTls->mTopLevelTls;
    scene_rdl2::alloc::Arena* arena = &(tls->mArena);
    SCOPED_MEM(arena);
    // use these samples to estimate emission contribution on each scatter event
    // (light strategy)
    const std::vector<geom::internal::EmissiveRegion>& emissiveRegions =
        scene->getEmissiveRegions();
    int emissiveRegionsCount = emissiveRegions.size();
    VolumeEmissionLightSampler* emissionSamples =
        arena->allocArray<VolumeEmissionLightSampler>(emissiveRegionsCount);
    for (int eIndex = 0; eIndex < emissiveRegionsCount; ++eIndex) {
        emissionSamples[eIndex] = VolumeEmissionLightSampler(sp, pv,
            (mMaxVolumeDepth - pv.volumeDepth), eIndex, sequenceID);
    }
    scene_rdl2::math::Color lMultiScatter(0.0f);
    float invN = 1.0f / (float)scatterSampleCount;
    for (int i = 0; i < scatterSampleCount; ++i) {
        // draw free path sample to determine where the scatter event starts
        // along the primary ray
        float ut;
        freePathSamples.getSample(&ut, pv.nonMirrorDepth);
        float pdfDensity, utRemapped;
        int densityIndex = densityDistribution.sampleDiscrete(
            ut, &pdfDensity, &utRemapped);
        const auto& vp = volumeProperties[densityIndex];
        float sigmaT = luminance(vp.mSigmaT + vp.mSigmaTh);
        float t0 = vp.mTStart;
        float t1 = vp.mTStart + vp.mDelta;
        float td = sampleDistanceExponential(utRemapped, sigmaT, t0, t1);
        float pdfT = pdfDensity * pdfDistanceExponential(td, sigmaT, t0, t1);
        if (pdfT == 0.0f || !scene_rdl2::math::isfinite(pdfT)) {
            continue;
        }
        // the transmittance from ray start point to this scatter point
        scene_rdl2::math::Color trScatter = vp.mTransmittance * vp.mTransmittanceH *
            exp(-vp.mSigmaT * (td - t0)) * exp(-vp.mSigmaTh * (td - t0));
        scene_rdl2::math::Vec3f scatterPoint = ray.org + ray.dir * td;
        scene_rdl2::math::Color throughput(pv.pathThroughput * vp.mSigmaS * trScatter / pdfT);
        // We don't want the scattering or pathThroughput for deep volumes,
        //  these are handled elsewhere.
        scene_rdl2::math::Color deepThroughput = vp.mSigmaS / pdfT;
        // draw phase sample to determine the new ray direction after this
        // scatter event
        float up[2];
        phaseSamples.getSample(up, pv.nonMirrorDepth);
        // theoretically throughput need to be updated here with
        // throughput *= phase(theta) / pdf(phase(theta))
        // but since our current phase model is perfect sampled
        // the phase eval is equal to pdf and we can get just skip it
        scene_rdl2::math::Vec3f scatterDir = VolumePhase(vp.mG).sample(-ray.dir, up[0], up[1]);

        // TODO computeRadianceRecurse
        int primaryRayDepth = ray.getDepth();
        int volumeDepth = pv.volumeDepth;
        float time = ray.time;
        int scatterStateId = pv.lpeStateId;
        // LPE
        if (pbrTls->mFs->mLightAovs->hasEntries()) {
            EXCL_ACCUMULATOR_PROFILE(pbrTls, EXCL_ACCUM_AOVS);
            // transition for volume scattering event
            scatterStateId = pbrTls->mFs->mLightAovs->volumeEventTransition(
                pbrTls, scatterStateId);
        }
        while (volumeDepth < mMaxVolumeDepth) {
            if (luminance(throughput) < mVolumeTransmittanceThreshold) {
                break;
            }
            // magic numbers in the original paper which are used to reduced the
            // scatter/extinction/anisotropic effect as scatter events go
            // to higher order
            float ai = scene_rdl2::math::pow(mVolumeAttenuationFactor, volumeDepth);
            float bi = scene_rdl2::math::pow(mVolumeContributionFactor, volumeDepth);
            float ci = scene_rdl2::math::pow(mVolumePhaseAttenuationFactor,
                volumeDepth);
            // draw free path
            freePathSamples.getSample(&ut, pv.nonMirrorDepth);
            // the reference paper makes the approximation that local
            // sigmaT is rougly equal to global max sigmaT in order to
            // select the next step length
            // (which is definitely not true in highly heterogeneous case...)
            float tFreePath = -scene_rdl2::math::log(1.0f - ut) / sigmaT;
            // intersectVolumes and get intervals
            mcrt_common::Ray scatterRay(scatterPoint, scatterDir, scene_rdl2::math::sEpsilon,
                tFreePath, time, primaryRayDepth + volumeDepth);
            size_t intervalCount = collectVolumeIntervals(pbrTls, scatterRay, rayMask);

            // figure out which volume regions this free path sample lands on
            auto& volumeRayState = tls->mGeomTls->mVolumeRayState;
            const geom::internal::VolumeTransition* intervals =
                volumeRayState.getVolumeIntervals();
            for (size_t j = 0; j < intervalCount; ++j) {
                // collectVolumeIntervals can return VolumeTransitions that are
                // past the end point of the ray, (tFreePath). This is because
                // tFreePath might be inside the volume, but we must intersect
                // the edge of the volume for the interiority check.
                if (intervals[j].mT > tFreePath) {
                    // Ensure the on /off state for this volumeId is correct.
                    // If we entered the volume after tFreePath, the volume region is off,
                    // if we exited, the volume region is on.
                    MNRY_ASSERT((intervals[j].mIsEntry && volumeRayState.isOff(intervals[j].mVolumeId)) ||
                               (intervals[j].mIsExit  && volumeRayState.isOn(intervals[j].mVolumeId)));

                    // If the transition comes after the end of the ray, we do not need to
                    // check any subsequent transitions because the on / off state should be
                    // correctly initialized already. Transitions are ordered front to back.
                    //
                    // Example 1: Scatter ray starts and ends inside a volume region.
                    //
                    // | = volume edge
                    // * = scatter ray start or end point
                    // ---> = normal at ray intersection point / direction of ray.
                    //
                    // |                    |
                    // |    *--------*------|--->  ray is exiting volume, but end point is inside volume.
                    // |                    |
                    //
                    // Example 2: scatter ray starts and ends before the first intersection with the volume.
                    //
                    //           ______
                    //          |      |
                    // *----*---|------|--->
                    //          |______|
                    //
                    // Example 3: scatter ray enters and exits the volume multiple times,
                    //            but the end point of the ray is outside the volume.
                    //
                    //           ______           ______
                    //          |      |         |      |
                    // *--------|------|-----*---|------|--->
                    //          |      |_________|      |
                    //          |_______________________|
                    //
                    // All volume region states that we have not yet set here,
                    // have been turned on or off appropriately in the collectVolumeIntervals function.
                    // collectVolumeIntervals sets the on /off state relative to the location of the
                    // origin of the scatter ray. If we have not visited the volume by this point, then
                    // the origin and the end of the scatter ray are on the same side of the volume.
                    // See Example 2 above.
                    break;
                }

                if (intervals[j].mIsEntry) {
                    volumeRayState.turnOn(intervals[j].mVolumeId, intervals[j].mPrimitive);
                }
                if (intervals[j].mIsExit) {
                    volumeRayState.turnOff(intervals[j].mVolumeId);
                }
            }
            const auto& volumeRegions = volumeRayState.getCurrentVolumeRegions();
            int* volumeIds = arena->allocArray<int>(
                volumeRayState.getVolumeAssignmentTable()->getVolumeCount());
            int volumeRegionsCount = volumeRegions.getVolumeIds(volumeIds);
            scatterPoint = scatterPoint + tFreePath * scatterDir;
            // the free path sample is out of volume
            if (volumeRegionsCount == 0) {
                break;
            }
            const auto& volumeSampleInfo = volumeRayState.getVolumeSampleInfo();
            scene_rdl2::math::Color newSigmaT, newSigmaS;
            // for cutouts/holdouts, but not used here because cutouts/holdouts
            //  only apply to primary rays
            scene_rdl2::math::Color newSigmaTh, newSigmaSh;
            float newSurfaceTransmittance;
            float newG;
            int volumeIdSampled;
            float rndVar = 0.f;
            if (mVolumeOverlapMode == VolumeOverlapMode::RND) {
                freePathSamples2.getSample(&rndVar, 0);
            }

            evalVolumeShaders(pbrTls, volumeRegionsCount, volumeIds,
                mVolumeOverlapMode, rndVar, volumeSampleInfo,
                tFreePath, time, false,
                &newSigmaT, &newSigmaS, &newSigmaTh, &newSigmaSh,
                &newSurfaceTransmittance, &newG, nullptr, &volumeIdSampled, -1);
            VolumePhase phase(newG * ci);
            sigmaT = luminance(newSigmaT);
            if (scene_rdl2::math::isZero(sigmaT)) {
                break;
            }
            // the throughput update below stands for:
            // throughput *= sigmaS(current) * transmittance(prev, current) /
            //     pdf(tFreePath)
            // since pdf(tFreePath) = sigmaT(current) * transmittance(prev, current)
            // we get the below result
            // For pdf derivation part please reference Raab M. 06
            // "Unbiased Global Illumination with Participating Media"
            throughput *= newSigmaS / sigmaT;
            deepThroughput *= newSigmaS / sigmaT;
            // LPE
            if (pbrTls->mFs->mLightAovs->hasEntries()) {
                EXCL_ACCUMULATOR_PROFILE(pbrTls, EXCL_ACCUM_AOVS);
                // transition for volume scattering event
                scatterStateId = pbrTls->mFs->mLightAovs->volumeEventTransition(
                    pbrTls, scatterStateId);
            }
            int assignmentId = volumeRayState.getAssignmentId(volumeIds[0]);
            // integrate the lighting contribution
            for (int lightIndex = 0; lightIndex < lightCount; ++lightIndex) {
                const Light* light = scene->getLight(lightIndex);
                // draw samples to integrate in-scattering source term
                scene_rdl2::math::Vec3f usl;
                scatterLightSamples.getSample(&usl[0], pv.nonMirrorDepth);
                LightFilterRandomValues uslFilter;
                scatterLightFilterSamples.getSample(&uslFilter.r2[0], pv.nonMirrorDepth);
                scatterLightFilterSamples3D.getSample(&uslFilter.r3[0], pv.nonMirrorDepth);
                // skip this light if it's not in the current lightset
                if (!scene->isLightActive(assignmentId, lightIndex)) {
                    continue;
                }

                scene_rdl2::math::Color scatteringSourceTerm = estimateInScatteringSourceTerm(
                    pbrTls, scatterRay, scatterPoint, light, assignmentId,
                    phase, usl, uslFilter, sp, sequenceID, ai);

                scene_rdl2::math::Color contribution = invN * throughput * bi *
                    scatteringSourceTerm;
                lMultiScatter += contribution;

                if (deepParams) {
                    // Same as above but without the pv.pathThroughput or
                    // trScatter terms
                    scene_rdl2::math::Color deepContribution = invN * deepThroughput * bi *
                        scatteringSourceTerm;

                    if (deepParams->mVolumeAovs) {
                        EXCL_ACCUMULATOR_PROFILE(pbrTls, EXCL_ACCUM_AOVS);
                        const FrameState &fs = *pbrTls->mFs;
                        const LightAovs &lightAovs = *fs.mLightAovs;
                        // transition
                        int lpeStateId = lightAovs.lightEventTransition(pbrTls,
                                                scatterStateId, light);
                        // accumulate matching aovs
                        fs.mAovSchema->initFloatArray(deepParams->mVolumeAovs);
                        aovAccumLightAovs(pbrTls, *fs.mAovSchema, *fs.mLightAovs,
                            deepContribution, nullptr, AovSchema::sLpePrefixNone, lpeStateId, deepParams->mVolumeAovs);
                    }

                    deepParams->mDeepBuffer->addVolumeSample(
                        pbrTls,
                        deepParams->mPixelX,
                        deepParams->mPixelY,
                        td,
                        trScatter,
                        deepContribution,
                        deepParams->mVolumeAovs);
                }

                // LPE
                if (pbrTls->mFs->mLightAovs->hasEntries() && (aovs || rs)) {
                    EXCL_ACCUMULATOR_PROFILE(pbrTls, EXCL_ACCUM_AOVS);
                    const FrameState &fs = *pbrTls->mFs;
                    const LightAovs &lightAovs = *fs.mLightAovs;
                    // transition
                    int lpeStateId = lightAovs.lightEventTransition(pbrTls,
                        scatterStateId, light);
                    // accumulate matching aovs
                    if (aovs) {
                        aovAccumLightAovs(pbrTls, *fs.mAovSchema, *fs.mLightAovs,
                            contribution, nullptr, AovSchema::sLpePrefixNone, lpeStateId, aovs);
                    } else {
                        MNRY_ASSERT(rs && fs.mExecutionMode == mcrt_common::ExecutionMode::VECTORIZED);
                        aovAccumLightAovsBundled(pbrTls, *fs.mAovSchema, *fs.mLightAovs,
                            contribution, nullptr, AovSchema::sLpePrefixNone, lpeStateId, sp.mPixel, rs->mDeepDataHandle);
                    }
                }
            }
            // integrate the emissive volume contribution
            for (int eIndex = 0; eIndex < emissiveRegionsCount; ++eIndex) {
                const auto& emissiveRegion = emissiveRegions[eIndex];
                int emissiveVolumeId = emissiveRegion.mVolumeId;
                if (!emissiveRegion.canIlluminateVolume()) {
                    continue;
                }
                scene_rdl2::math::Vec3f uls;
                emissionSamples[eIndex].getSample(uls);
                scene_rdl2::math::Vec3f wi;
                float pdfWi;
                float tEnd;
                emissiveRegion.sample(scatterPoint, uls[0], uls[1], uls[2],
                    wi, pdfWi, tEnd, time);
                if (pdfWi == 0.0f || !scene_rdl2::math::isfinite(pdfWi)) {
                    continue;
                }
                mcrt_common::Ray wiRay(scatterPoint, wi, 0.0f, tEnd, ray.time,
                    ray.getDepth() + 1);

                scene_rdl2::math::Color LVe = computeEmissiveVolumeIntegral(pbrTls, wiRay,
                    emissiveVolumeId, sp, sequenceID);
                float ph = phase.eval(-scatterRay.dir, wi);
                scene_rdl2::math::Color contribution = invN * throughput * bi * ph * LVe / pdfWi;
                lMultiScatter += contribution;
                if (deepParams) {
                    // Same as above but without the pv.pathThroughput or
                    // trScatter terms
                    scene_rdl2::math::Color deepContribution = invN * deepThroughput * bi *
                        ph * LVe / pdfWi;

                    if (deepParams->mVolumeAovs) {
                        EXCL_ACCUMULATOR_PROFILE(pbrTls, EXCL_ACCUM_AOVS);
                        const FrameState &fs = *pbrTls->mFs;
                        const LightAovs &lightAovs = *fs.mLightAovs;
                        // transition
                        int lpeStateId = lightAovs.emissionEventTransition(pbrTls,
                            scatterStateId,
                            scene->getVolumeLabelId(emissiveRegion.mVolumeId));
                        // accumulate matching aovs
                        fs.mAovSchema->initFloatArray(deepParams->mVolumeAovs);
                        aovAccumLightAovs(pbrTls, *fs.mAovSchema, *fs.mLightAovs,
                                          deepContribution, nullptr, AovSchema::sLpePrefixNone,
                                          lpeStateId, deepParams->mVolumeAovs);
                    }

                    deepParams->mDeepBuffer->addVolumeSample(
                        pbrTls,
                        deepParams->mPixelX,
                        deepParams->mPixelY,
                        td,
                        trScatter,
                        deepContribution,
                        deepParams->mVolumeAovs);
                }

                // LPE
                if (pbrTls->mFs->mLightAovs->hasEntries() && (aovs || rs)) {
                    EXCL_ACCUMULATOR_PROFILE(pbrTls, EXCL_ACCUM_AOVS);
                    const FrameState &fs = *pbrTls->mFs;
                    const LightAovs &lightAovs = *fs.mLightAovs;
                    // transition
                    int lpeStateId = lightAovs.emissionEventTransition(pbrTls,
                        scatterStateId,
                        scene->getVolumeLabelId(emissiveRegion.mVolumeId));
                    // accumulate matching aovs
                    if (aovs) {
                        aovAccumLightAovs(pbrTls, *fs.mAovSchema, *fs.mLightAovs,
                            contribution, nullptr, AovSchema::sLpePrefixNone, lpeStateId, aovs);
                    } else {
                        MNRY_ASSERT(rs && fs.mExecutionMode == mcrt_common::ExecutionMode::VECTORIZED);
                        aovAccumLightAovsBundled(pbrTls, *fs.mAovSchema, *fs.mLightAovs,
                            contribution, nullptr, AovSchema::sLpePrefixNone, lpeStateId, sp.mPixel, rs->mDeepDataHandle);
                    }
                }
            }
            // draw phase sample to determine the new ray direction after this
            // scatter event
            phaseSamples.getSample(up, pv.nonMirrorDepth);
            scatterDir = phase.sample(-scatterRay.dir, up[0], up[1]);
            volumeDepth++;
        }
    }
    return lMultiScatter;
}

bool
PathIntegrator::computeRadianceVolume(pbr::TLState *pbrTls, const mcrt_common::Ray& ray,
        const Subpixel& sp, PathVertex& pv, const int lobeType,
        scene_rdl2::math::Color& radiance, unsigned sequenceID, VolumeTransmittance& vt,
        float* aovs, DeepParams* deepParams, const RayState *rs,
        float* surfaceT) const
{
    EXCL_ACCUMULATOR_PROFILE(pbrTls, EXCL_ACCUM_VOL_INTEGRATION);

    vt.reset();

    // Set the proper ray mask. If lobeType is 0,
    // it's a primary ray from camera.
    // Otherwise, we figure out the ray mask based on the lobe type.
    // for ray mask propagation, we simply merge with existing ray mask.
    int rayMask = lobeType == 0 ?
        scene_rdl2::rdl2::CAMERA : lobeTypeToRayMask(lobeType);
    size_t intervalCount = collectVolumeIntervals(pbrTls, ray, rayMask);
    mcrt_common::ThreadLocalState* tls = pbrTls->mTopLevelTls;
    auto& volumeRayState = tls->mGeomTls->mVolumeRayState;
    // vacuum case, no participating media contribution and full transmission
    if (intervalCount == 0 &&
        volumeRayState.getCurrentVolumeRegions().isVacuum()) {
        return false;
    }
    scene_rdl2::alloc::Arena* arena = &(tls->mArena);
    SCOPED_MEM(arena);
    geom::internal::VolumeTransition* intervals =
        volumeRayState.getVolumeIntervals();
    int visitedVolumeCount = volumeRayState.getVisitedVolumeRegionsCount();
    int* volumeIds = arena->allocArray<int>(visitedVolumeCount);
    // estimate how many entries we need for building distribution table
    // for decoupled ray marching. We cannot allocate more entries than
    // there is memory in an Arena block.
    size_t maxSteps = arena->getBlockSize() / sizeof(VolumeProperties);
    size_t estimatedStepCount = estimateStepCount(ray, intervalCount, intervals,
        volumeIds, mInvVolumeQuality, volumeRayState, maxSteps);
    // the appended one is added to guarantee upper bound exist for later
    // binary search
    VolumeProperties* volumeProperties =
        arena->allocArray<VolumeProperties>(estimatedStepCount + 1);
    // start the decoupled ray marching through all the intervals collected
    // along this ray, this would give us the transmittance value across this
    // ray and an array of VolumeProperty for later 1d distribution construction
    size_t marchingStepsCount = 0;
    SequenceIDIntegrator transmittanceSid(sp.mPixel, sp.mSubpixelIndex, SequenceType::VolumeAttenuation, sequenceID);
    IntegratorSample1D trSamples(transmittanceSid);

    SequenceIDIntegrator transmittanceSid2(sp.mPixel, sp.mSubpixelIndex, SequenceType::IndexSelection, sequenceID);
    IntegratorSample1D trSamples2(transmittanceSid2);

    float tStart = ray.tnear;
    // integrate primary ray volume emission during decoupledRayMarching
    // for non primary ray, we'll sample emission explicitly like light sources
    // instead relying on bounce ray blindly hitting them. This is similar to
    // direct lighting for primary ray: when ray hits the light, we only add
    // light contribution if the ray is a primary ray
    bool sampleEmission = (ray.getDepth() == 0);
    scene_rdl2::math::Color* perVolumeLVe = nullptr;
    if (sampleEmission) {
        int volumeCount =
            volumeRayState.getVolumeAssignmentTable()->getVolumeCount();
        perVolumeLVe = arena->allocArray<scene_rdl2::math::Color>(volumeCount);
        memset(perVolumeLVe, 0, sizeof(scene_rdl2::math::Color) * volumeCount);
    }
    bool reachTransmittanceThreshold = false;
    for (size_t i = 0; i < intervalCount; ++i) {
        if (intervals[i].mT > ray.tfar || reachTransmittanceThreshold) {
            break;
        }
        if ((intervals[i].mT - tStart) > scene_rdl2::math::sEpsilon) {
            decoupledRayMarching(pbrTls, vt, perVolumeLVe,
                estimatedStepCount, marchingStepsCount,
                tStart, intervals[i].mT, ray.time, ray.getDepth(),
                volumeRayState, volumeIds, volumeProperties, trSamples, trSamples2,
                reachTransmittanceThreshold, deepParams);
            tStart = intervals[i].mT;
        }
        if (intervals[i].mIsEntry) {
            volumeRayState.turnOn(intervals[i].mVolumeId, intervals[i].mPrimitive);
        }
        if (intervals[i].mIsExit) {
            volumeRayState.turnOff(intervals[i].mVolumeId);
        }
    }
    if ((ray.tfar - tStart) > scene_rdl2::math::sEpsilon && !reachTransmittanceThreshold) {
        decoupledRayMarching(pbrTls, vt, perVolumeLVe,
            estimatedStepCount, marchingStepsCount,
            tStart, ray.tfar, ray.time, ray.getDepth(),
            volumeRayState, volumeIds, volumeProperties, trSamples, trSamples2,
            reachTransmittanceThreshold, deepParams);
    }

    if (marchingStepsCount == 0) {
        return false;
    }

    if (deepParams) {
        // We have collected all the volume segments for this ray, add them to
        //  the deep buffer.
        deepParams->mDeepBuffer->addVolumeSegments(
            pbrTls,
            deepParams->mPixelX,
            deepParams->mPixelY,
            deepParams->mSampleX,
            deepParams->mSampleY,
            ray.dir.z,
            volumeProperties,
            marchingStepsCount);
    }

    // Compute the T distance of the volume's "surface"
    float accumulatedTransmittance = 1.f;
    for (size_t i = 0; i < marchingStepsCount; i++) {
        float sigmaT = scene_rdl2::math::luminance(volumeProperties[i].mSigmaT);
        float stepTransmittance = exp(-sigmaT * volumeProperties[i].mDelta); // Beer's Law
        float desiredTransmittance = 1.f - volumeProperties[i].mSurfaceOpacityThreshold;
        if (desiredTransmittance > accumulatedTransmittance * stepTransmittance) {
            // This is the step we want because we have found the range of
            //  transmittance that we are searching for.
            // Compute the amount of additional transmittance we need to apply within this
            //  step to achieve the desired transmittance.
            float additionalTransmittance = desiredTransmittance / accumulatedTransmittance;

            // From Beer's Law: tr = exp(-sigmaT * t)
            // We solve for the t distance that matches transmittance tr:
            //   ln(tr) = -sigmaT * t
            //   -ln(tr) / sigmaT = t
            float stepT = -log(additionalTransmittance) / sigmaT;

            // The total distance is the start of the step plus the distance within the step
            *surfaceT = volumeProperties[i].mTStart + stepT;

            break;
        }
        accumulatedTransmittance *= stepTransmittance;
    }

    // when volume attenuation pass through transmittance threshold, treat the
    // volume as fully opaque
    if (reachTransmittanceThreshold) {
        vt.mTransmittanceE = scene_rdl2::math::Color(0.0f);
        // The alpha is computed from transmittanceA.
        // In this case we want to use the accumulated minimum transmission
        // due to the holdouts (>= 0) instead of just being opaque (0).
        // This early termination due to transmittance threshold is an ugly
        // biased hack anyway which is why we have this weird special case.
        vt.mTransmittanceAlpha = vt.mTransmittanceMin;
    }

    // add emission contributions
    if (sampleEmission) {
        int* visitedVolumeIds = arena->allocArray<int>(visitedVolumeCount);
        volumeRayState.getVisitedVolumeIds(visitedVolumeIds);
        for (int i = 0; i < visitedVolumeCount; ++i) {
            int volumeId = visitedVolumeIds[i];
            scene_rdl2::math::Color emissionContribution = pv.pathThroughput *
                perVolumeLVe[volumeId];
            if (!isBlack(emissionContribution)) {
                radiance += emissionContribution;
                // LPE
                if (pbrTls->mFs->mLightAovs->hasEntries() && (aovs || rs)) {
                    EXCL_ACCUMULATOR_PROFILE(pbrTls, EXCL_ACCUM_AOVS);
                    const FrameState &fs = *pbrTls->mFs;
                    const LightAovs &lightAovs = *fs.mLightAovs;
                    // transition
                    int lpeStateId = pv.lpeStateId;
                    lpeStateId = lightAovs.emissionEventTransition(pbrTls,
                        lpeStateId, fs.mScene->getVolumeLabelId(volumeId));
                    // accumulate matching aovs
                    if (aovs) {
                        aovAccumLightAovs(pbrTls, *fs.mAovSchema, *fs.mLightAovs,
                            emissionContribution, nullptr, AovSchema::sLpePrefixNone, lpeStateId, aovs);
                    } else {
                        MNRY_ASSERT(rs && fs.mExecutionMode == mcrt_common::ExecutionMode::VECTORIZED);
                        aovAccumLightAovsBundled(pbrTls, *fs.mAovSchema, *fs.mLightAovs,
                            emissionContribution, nullptr, AovSchema::sLpePrefixNone,
                            lpeStateId, sp.mPixel, rs->mDeepDataHandle);
                    }
                }
            }
        }
    }

    if (pv.volumeDepth < mMaxVolumeDepth) {
        // start computing volume in-scattering integration along the ray
        const auto& lastVp = volumeProperties[marchingStepsCount - 1];
        volumeProperties[marchingStepsCount].mTStart =
            lastVp.mTStart + lastVp.mDelta;
        // build a 1D distribution based on the volume property we collected
        // from decoupledRayMarching
        float* discreteDensity = arena->allocArray<float>(marchingStepsCount);
        for (size_t i = 0; i < marchingStepsCount; ++i) {
            auto& vp = volumeProperties[i];
            discreteDensity[i] = scene_rdl2::math::luminance(vp.mDelta * vp.mSigmaS * vp.mTransmittance) +
                                 scene_rdl2::math::luminance(vp.mDelta * vp.mSigmaSh * vp.mTransmittanceH);
        }
        uint32_t *guide = arena->allocArray<uint32_t>(marchingStepsCount);
        GuideDistribution1D densityDistribution(marchingStepsCount, discreteDensity, guide);
        densityDistribution.tabulateCdf();
        // draw some samples to integrate volume single scattering
        radiance += integrateVolumeScattering(pbrTls, ray,
            volumeProperties, densityDistribution, sp, pv, sequenceID, aovs, deepParams, rs);
        pv.volumeDepth++;
        if (pv.volumeDepth < mMaxVolumeDepth) {
            radiance += approximateVolumeMultipleScattering(
                pbrTls, ray, volumeProperties, densityDistribution, sp, pv,
                rayMask, sequenceID, aovs, deepParams, rs);
        }
    }

    return true;
}

scene_rdl2::math::Color
PathIntegrator::computeEmissiveVolumeIntegral(pbr::TLState *pbrTls, mcrt_common::Ray& ray,
        int emissiveVolumeId, const Subpixel& sp, unsigned sequenceID) const
{
    // figure out the integral end point with hard surface intersection test
    {
        EXCL_ACCUMULATOR_PROFILE(pbrTls, EXCL_ACCUM_EMBREE_INTERSECTION);
        ray.mask = scene_rdl2::rdl2::SHADOW;
        pbrTls->mFs->mScene->getEmbreeAccelerator()->intersect(ray);
    }
    float tStart = ray.tnear;
    float tEnd = ray.tfar;
    // The reason to pass in a all on ray mask is because this ray
    // serves the purpose of both collecting emissive volume and
    // transmittance volume. If we make the ray collects only volume
    // contributes to transmittance, we can miss the volume contributes
    // emission when the volume set the visbility flag for shadow off
    size_t intervalCount = collectVolumeIntervals(pbrTls, ray, scene_rdl2::rdl2::ALL_VISIBLE);
    mcrt_common::ThreadLocalState* tls = pbrTls->mTopLevelTls;
    auto& volumeRayState = tls->mGeomTls->mVolumeRayState;
    // vacuum case, no participating media contribution and full transmision
    if (intervalCount == 0 &&
        volumeRayState.getCurrentVolumeRegions().isVacuum()) {
        return scene_rdl2::math::sBlack;
    }
    // the ray doesn't hit the volume with specified emissiveVolumeId
    if (!volumeRayState.isVisited(emissiveVolumeId)) {
        return scene_rdl2::math::sBlack;
    }
    scene_rdl2::alloc::Arena* arena = &(tls->mArena);
    SCOPED_MEM(arena);
    geom::internal::VolumeTransition* intervals =
        volumeRayState.getVolumeIntervals();
    int visitedVolumeCount = volumeRayState.getVisitedVolumeRegionsCount();
    int* volumeIds = arena->allocArray<int>(visitedVolumeCount);
    // we only need to integrate from ray start to the end of specified
    // emissive volume
    for (int i = intervalCount - 1; i >= 0; --i) {
        if (intervals[i].mVolumeId == emissiveVolumeId) {
            // no more volume interval that we need to integrate from here
            // trim down tEnd to avoid unnecessary computation
            if (intervals[i].mIsExit) {
                intervalCount = i + 1;
                tEnd = intervals[i].mT;
            }
            break;
        }
    }
    scene_rdl2::math::Color LVe(0.0f);
    scene_rdl2::math::Color tr(1.0f);
    IntegratorSample1D trSamples(SequenceIDIntegrator(
        sp.mPixel, sp.mSubpixelIndex,
        SequenceType::VolumeAttenuation, sequenceID));
    IntegratorSample1D trSamples2(SequenceIDIntegrator(
            sp.mPixel, sp.mSubpixelIndex,
            SequenceType::IndexSelection, sequenceID));
    bool reachTransmittanceThreshold = false;
    for (size_t i = 0; i < intervalCount; ++i) {
        if (intervals[i].mT > tEnd || reachTransmittanceThreshold) {
            break;
        }
        if ((intervals[i].mT - tStart) > scene_rdl2::math::sEpsilon) {
            LVe += computeEmissiveVolumeIntegralSubInterval(pbrTls,
                emissiveVolumeId, tr, tStart, intervals[i].mT,
                ray.time, ray.getDepth(),
                volumeRayState, volumeIds, trSamples, trSamples2,
                reachTransmittanceThreshold);
            tStart = intervals[i].mT;
        }
        if (intervals[i].mIsEntry) {
            volumeRayState.turnOn(intervals[i].mVolumeId, intervals[i].mPrimitive);
        }
        if (intervals[i].mIsExit) {
            volumeRayState.turnOff(intervals[i].mVolumeId);
        }
    }
    if ((tEnd - tStart) > scene_rdl2::math::sEpsilon && !reachTransmittanceThreshold) {
        LVe += computeEmissiveVolumeIntegralSubInterval(pbrTls,
            emissiveVolumeId, tr, tStart, tEnd,
            ray.time, ray.getDepth(),
            volumeRayState, volumeIds, trSamples, trSamples2,
            reachTransmittanceThreshold);
    }
    return LVe;
}

scene_rdl2::math::Color
PathIntegrator::computeEmissiveVolumeIntegralSubInterval(pbr::TLState *pbrTls,
        int emissiveVolumeId, scene_rdl2::math::Color& transmittance, float t0, float t1,
        float time, int depth, const geom::internal::VolumeRayState& volumeRayState,
        int* volumeIds, const IntegratorSample1D& trSamples, const IntegratorSample1D& trSamples2,
        bool& reachTransmittanceThreshold) const
{
    int volumeRegionsCount = volumeRayState.getCurrentVolumeRegions().getVolumeIds(
        volumeIds);
    if (volumeRegionsCount == 0) {
        return scene_rdl2::math::sBlack;
    }
    const auto& volumeSampleInfo = volumeRayState.getVolumeSampleInfo();
    bool isHomogenous = true;
    float minFeatureSize = scene_rdl2::math::inf;
    for (int i = 0; i < volumeRegionsCount; ++i) {
        const auto& sampleInfo = volumeSampleInfo[volumeIds[i]];
        isHomogenous &= sampleInfo.isHomogenous();
        // when there are multiple volume regions in this interval,
        // use the smallest non-homogenous feature size for stepping
        // (stepping only happens when there is a non-homogenous volume region,
        //  and homogenous volumes do not have a valid feature size)
        if (!sampleInfo.isHomogenous()) {
            minFeatureSize = scene_rdl2::math::min(minFeatureSize, sampleInfo.getFeatureSize());
        }
    }
    bool sampleEmission =
        volumeRayState.getCurrentVolumeRegions().isOn(emissiveVolumeId);

    scene_rdl2::alloc::Arena* arena = &(pbrTls->mTopLevelTls->mArena);
    SCOPED_MEM(arena);
    mcrt_common::ThreadLocalState* tls = pbrTls->mTopLevelTls;

    // only volume with casting shadow turned on will contribute transmittance
    int trVolumeRegionsCount = 0;
    int* trVolumeIds = arena->allocArray<int>(volumeRegionsCount);
    for (int i = 0; i < volumeRegionsCount; ++i) {
        const geom::internal::VolumeSampleInfo& sampleInfo =
            volumeSampleInfo[volumeIds[i]];
        if (sampleInfo.canCastShadow()) {
            trVolumeIds[trVolumeRegionsCount++] = volumeIds[i];
        }
    }

    scene_rdl2::math::Color LVe(0.0f);
    if (isHomogenous) {
        float rndVal = 0.f;
        if (mVolumeOverlapMode == VolumeOverlapMode::RND) {
            trSamples2.getSample(&rndVal, 0);
        }
        float rayVolumeDepth = t1 - t0; // total distance ray travels through the volume
        scene_rdl2::math::Color sigmaT = evalSigmaT(pbrTls, trVolumeRegionsCount, trVolumeIds,
            mVolumeOverlapMode, rndVal, volumeSampleInfo, t0, time, nullptr, rayVolumeDepth);
        scene_rdl2::math::Color tr = exp(-sigmaT * (t1 - t0));
        // integrate emission * e^{-sigmaT * t) from t0 to t1
        if (sampleEmission) {
            const geom::internal::VolumeSampleInfo& sampleInfo =
                volumeSampleInfo[emissiveVolumeId];
            const scene_rdl2::rdl2::VolumeShader* volumeShader = sampleInfo.getShader();

            shading::Intersection isect;
            const geom::internal::Primitive* prim = volumeRayState.getCurrentVolumeRegions().getPrimitive(emissiveVolumeId);
            isect.init(prim->getRdlGeometry());
            const scene_rdl2::math::Vec3f p = sampleInfo.getSamplePosition(t0);
            const scene_rdl2::math::Vec3f evalP = prim->evalVolumeSamplePosition(tls, emissiveVolumeId, p, time);
            isect.setP(prim->transformVolumeSamplePosition(evalP, time));
            const shading::State state(&isect);

            scene_rdl2::math::Color emission = volumeShader->emission(
                pbrTls->mTopLevelTls->mShadingTls.get(), state,
                prim->evalTemperature(tls, emissiveVolumeId, evalP));
            LVe = transmittance * emission * (scene_rdl2::math::Color(1.0f) - tr);
            LVe.r = scene_rdl2::math::isZero(sigmaT.r) ? 0.0f : LVe.r / sigmaT.r;
            LVe.g = scene_rdl2::math::isZero(sigmaT.g) ? 0.0f : LVe.g / sigmaT.g;
            LVe.b = scene_rdl2::math::isZero(sigmaT.b) ? 0.0f : LVe.b / sigmaT.b;
        }
        transmittance *= tr;
    } else {
        shading::TLState* shadingTls = pbrTls->mTopLevelTls->mShadingTls.get();

        float u;
        trSamples.getSample(&u, 0);
        float tStart = t0;
        float stepSize = minFeatureSize * mInvVolumeQuality * scene_rdl2::math::max(depth - 1, 1);
        float t = t0 + stepSize * u;
        while (t < t1) {
            if (luminance(transmittance) < mVolumeTransmittanceThreshold) {
                reachTransmittanceThreshold = true;
                return LVe;
            }
            float rndVal = 0.f;
            if (mVolumeOverlapMode == VolumeOverlapMode::RND) {
                trSamples2.getSample(&rndVal, 0);
            }
            scene_rdl2::math::Color sigmaT = evalSigmaT(pbrTls, trVolumeRegionsCount, trVolumeIds,
                mVolumeOverlapMode, rndVal, volumeSampleInfo, t, time, nullptr, -1);
            float delta = t - tStart;
            if (sampleEmission) {
                const geom::internal::VolumeSampleInfo& sampleInfo =
                    volumeSampleInfo[emissiveVolumeId];
                const scene_rdl2::rdl2::VolumeShader* volumeShader = sampleInfo.getShader();

                shading::Intersection isect;
                const geom::internal::Primitive* prim = volumeRayState.getCurrentVolumeRegions().getPrimitive(emissiveVolumeId);
                isect.init(prim->getRdlGeometry());
                const scene_rdl2::math::Vec3f p = sampleInfo.getSamplePosition(t);
                const scene_rdl2::math::Vec3f evalP = prim->evalVolumeSamplePosition(tls, emissiveVolumeId, p, time);
                isect.setP(prim->transformVolumeSamplePosition(evalP, time));
                const shading::State state(&isect);

                scene_rdl2::math::Color emission = volumeShader->emission(
                    shadingTls, state,
                    prim->evalTemperature(tls, emissiveVolumeId, evalP));
                LVe += transmittance * emission * delta;
            }
            transmittance *= exp(-sigmaT * delta);
            tStart = t;
            t += stepSize;
        }
        if (luminance(transmittance) < mVolumeTransmittanceThreshold) {
            reachTransmittanceThreshold = true;
            return LVe;
        }
        // last step
        if ((t - t1) > scene_rdl2::math::sEpsilon) {
            float rndVal = 0.f;
            if (mVolumeOverlapMode == VolumeOverlapMode::RND) {
                trSamples2.getSample(&rndVal, 0);
            }

            scene_rdl2::math::Color sigmaT = evalSigmaT(pbrTls, trVolumeRegionsCount, trVolumeIds,
                mVolumeOverlapMode, rndVal, volumeSampleInfo, t1, time, nullptr, -1);
            float delta = t1 - tStart;
            if (sampleEmission) {
                const geom::internal::VolumeSampleInfo& sampleInfo =
                    volumeSampleInfo[emissiveVolumeId];
                const scene_rdl2::rdl2::VolumeShader* volumeShader = sampleInfo.getShader();

                shading::Intersection isect;
                const geom::internal::Primitive* prim = volumeRayState.getCurrentVolumeRegions().getPrimitive(emissiveVolumeId);
                isect.init(prim->getRdlGeometry());
                const scene_rdl2::math::Vec3f p = sampleInfo.getSamplePosition(t1);
                const scene_rdl2::math::Vec3f evalP = prim->evalVolumeSamplePosition(tls, emissiveVolumeId, p, time);
                isect.setP(prim->transformVolumeSamplePosition(evalP, time));
                const shading::State state(&isect);

                scene_rdl2::math::Color emission = volumeShader->emission(
                    shadingTls, state,
                    prim->evalTemperature(tls, emissiveVolumeId, evalP));
                LVe += transmittance * emission * delta;
            }
            transmittance *= exp(-sigmaT * delta);
        }
    }
    return LVe;
}

scene_rdl2::math::Color
PathIntegrator::computeRadianceEmissiveRegionsBundled(pbr::TLState *pbrTls, const RayState &rs,
        const shading::Bsdfv& bsdfv, const shading::BsdfSlicev& slicev,
        float rayEpsilon, unsigned int lane) const
{
    EXCL_ACCUMULATOR_PROFILE(pbrTls, EXCL_ACCUM_VOL_INTEGRATION);
    MNRY_ASSERT(pbrTls->mFs->mExecutionMode == mcrt_common::ExecutionMode::VECTORIZED ||
               pbrTls->mFs->mExecutionMode == mcrt_common::ExecutionMode::XPU);

    const Subpixel &sp = rs.mSubpixel;
    const PathVertex &pv = rs.mPathVertex;
    const mcrt_common::Ray &ray = rs.mRay;
    const shading::Intersection &isect = *rs.mAOSIsect;
    const unsigned int sequenceID = rs.mSequenceID;

    const FrameState &fs = *pbrTls->mFs;
    const Scene *scene = MNRY_VERIFY(fs.mScene);
    const std::vector<geom::internal::EmissiveRegion>& emissiveRegions =
        scene->getEmissiveRegions();
    if (emissiveRegions.empty()) {
        return scene_rdl2::math::sBlack;
    }
    bool highQualitySample = pv.nonMirrorDepth == 0;
    int nLightSamples = highQualitySample ? mLightSamples : 1;
    int nBsdfSamples = highQualitySample ? mBsdfSamples : 1;
    scene_rdl2::math::Color LVe(0.0f);
    const scene_rdl2::math::Vec3f& p = isect.getP();
    const scene_rdl2::math::Vec3f& n = isect.getN();
    for (size_t eIndex = 0; eIndex < emissiveRegions.size(); ++eIndex) {
        const auto& emissiveRegion = emissiveRegions[eIndex];
        int emissiveVolumeId = emissiveRegion.mVolumeId;
        // filtered out bsdf lobes that are flagged invisible on emissiveRegion
        int validReflection = rayMaskToReflectionLobes(
            emissiveRegion.mVisibilityMask);
        int validTransmission = rayMaskToTransmissionLobes(
            emissiveRegion.mVisibilityMask);
        shading::Bsdfv validBsdfv;
        shading::Bsdfv_init(&validBsdfv);
        bool hasValidLobe = false;
        for (int lIndex = 0; lIndex < shading::getLobeCount(bsdfv); ++lIndex) {
            shading::BsdfLobev* lobev = shading::getLobe(const_cast<shading::Bsdfv &>(bsdfv), lIndex);
            if (shading::isActive(*lobev, lane)) {
                if (shading::matchesFlags(*lobev, validReflection) ||
                    shading::matchesFlags(*lobev, validTransmission)) {
                    shading::addLobe(validBsdfv, lobev);
                    hasValidLobe = true;
                }
            }
        }
        if (!hasValidLobe) {
            continue;
        }
        BsdfOneSamplervOneLane bsdfSamplerv(validBsdfv, slicev, lane);
        // light strategy sampling for emissive volume
        VolumeEmissionLightSampler lSamples(sp, pv, nLightSamples, eIndex,
            highQualitySample, sequenceID);
        for (int ls = 0; ls < nLightSamples; ++ls) {
            scene_rdl2::math::Vec3f uls;
            lSamples.getSample(uls);
            scene_rdl2::math::Vec3f wi;
            float pdfLight;
            float tEnd;
            emissiveRegion.sample(p, uls[0], uls[1], uls[2], wi, pdfLight, tEnd, ray.time);
            if (pdfLight == 0.0f) {
                continue;
            }
            BsdfOneSamplervOneLane::LobesContribution lobesContribution;
            float pdfBsdf;
            scene_rdl2::math::Color f = bsdfSamplerv.eval(wi, pdfBsdf, &lobesContribution);
            if (isEqual(f, scene_rdl2::math::sBlack)) {
                continue;
            }
            mcrt_common::Ray wiRay(p, wi, rayEpsilon, tEnd, ray.time,
                ray.getDepth() + 1);
            scene_rdl2::math::Color contribution = computeEmissiveVolumeIntegral(pbrTls, wiRay,
                emissiveVolumeId, sp, sequenceID);
            float cosineTerm = shading::getIncludeCosineTerm(slicev, lane) ?
                1.0f : scene_rdl2::math::abs(scene_rdl2::math::dot(wi, n));
            float misWeight = 1.0f;
            if (nBsdfSamples != 0) {
                misWeight = powerHeuristic(
                    nLightSamples * pdfLight, nBsdfSamples * pdfBsdf);
            }
            contribution *= (misWeight * cosineTerm * pv.pathThroughput) /
                (pdfLight * nLightSamples);
            int nMatchedLobes = lobesContribution.mMatchedLobeCount;
            for (int lobeIndex = 0; lobeIndex < nMatchedLobes; ++lobeIndex) {
                scene_rdl2::math::Color lobeContribution = lobesContribution.mFs[lobeIndex] *
                    contribution;
                LVe += lobeContribution;
                // LPE
                const LightAovs &lightAovs = *fs.mLightAovs;
                if (lightAovs.hasEntries()) {
                    EXCL_ACCUMULATOR_PROFILE(pbrTls, EXCL_ACCUM_AOVS);
                    const AovSchema &aovSchema = *fs.mAovSchema;
                    // transition
                    int lpeStateId = pv.lpeStateId;
                    lpeStateId = lightAovs.scatterEventTransitionVector(pbrTls, lpeStateId,
                        bsdfv, *(lobesContribution.mLobes[lobeIndex]));
                    lpeStateId = lightAovs.emissionEventTransition(pbrTls,
                        lpeStateId,
                        scene->getVolumeLabelId(emissiveRegion.mVolumeId));
                    aovAccumLightAovsBundled(pbrTls, aovSchema, lightAovs, lobeContribution,
                        nullptr, AovSchema::sLpePrefixNone,
                        lpeStateId, sp.mPixel, rs.mDeepDataHandle);

                }
            }
        }
        // bsdf strategy sampling for emissive volume
        VolumeEmissionBsdfSampler bSamples(sp, pv, nBsdfSamples, eIndex,
            highQualitySample, sequenceID);
        for (int bs = 0; bs < nBsdfSamples; ++bs) {
            BsdfOneSamplervOneLane::LobesContribution lobesContribution;
            float uLobe;
            scene_rdl2::math::Vec2f ubs;
            bSamples.getSample(uLobe, ubs);
            scene_rdl2::math::Vec3f wi;
            float pdfBsdf;
            scene_rdl2::math::Color f = bsdfSamplerv.sample(uLobe, ubs[0], ubs[1], wi, pdfBsdf,
                &lobesContribution);
            if (pdfBsdf == 0.0f || isEqual(f, scene_rdl2::math::sBlack)) {
                continue;
            }
            float tEnd;
            float pdfLight = emissiveRegion.pdf(p, wi, tEnd, ray.time);
            // if pdfLight is 0, it means the ray doesn't hit any active
            // region in this EmissiveRegion
            if (pdfLight == 0.0f) {
                continue;
            }
            mcrt_common::Ray wiRay(p, wi, rayEpsilon, tEnd, ray.time,
                ray.getDepth() + 1);
            scene_rdl2::math::Color contribution = computeEmissiveVolumeIntegral(pbrTls, wiRay,
                emissiveVolumeId, sp, sequenceID);
            float cosineTerm = shading::getIncludeCosineTerm(slicev, lane) ?
                1.0f : scene_rdl2::math::abs(scene_rdl2::math::dot(wi, n));
            float misWeight = 1.0f;
            if (nLightSamples != 0) {
                misWeight = powerHeuristic(
                    nBsdfSamples * pdfBsdf, nLightSamples * pdfLight);
            }
            contribution *= (misWeight * cosineTerm * pv.pathThroughput) /
                (pdfBsdf * nBsdfSamples);
            int nMatchedLobes = lobesContribution.mMatchedLobeCount;
            for (int lobeIndex = 0; lobeIndex < nMatchedLobes; ++lobeIndex) {
                scene_rdl2::math::Color lobeContribution = lobesContribution.mFs[lobeIndex] *
                    contribution;
                LVe += lobeContribution;
                // LPE
                const LightAovs &lightAovs = *fs.mLightAovs;
                if (lightAovs.hasEntries()) {
                    EXCL_ACCUMULATOR_PROFILE(pbrTls, EXCL_ACCUM_AOVS);
                    const AovSchema &aovSchema = *fs.mAovSchema;
                    // transition
                    int lpeStateId = pv.lpeStateId;
                    lpeStateId = lightAovs.scatterEventTransitionVector(pbrTls, lpeStateId,
                        bsdfv, *(lobesContribution.mLobes[lobeIndex]));
                    lpeStateId = lightAovs.emissionEventTransition(pbrTls,
                        lpeStateId,
                        scene->getVolumeLabelId(emissiveRegion.mVolumeId));
                    aovAccumLightAovsBundled(pbrTls, aovSchema, lightAovs, lobeContribution,
                        nullptr, AovSchema::sLpePrefixNone,
                        lpeStateId, sp.mPixel, rs.mDeepDataHandle);
                }
            }
        }
    }
    return LVe;
}

scene_rdl2::math::Color
PathIntegrator::computeRadianceEmissiveRegionsScalar(pbr::TLState *pbrTls,
        const Subpixel& sp, const PathVertex& pv, const mcrt_common::Ray& ray,
        const shading::Intersection& isect, shading::Bsdf& bsdf, const shading::BsdfSlice& slice,
        float rayEpsilon, unsigned sequenceID, float* aovs) const
{
    EXCL_ACCUMULATOR_PROFILE(pbrTls, EXCL_ACCUM_VOL_INTEGRATION);

    const FrameState &fs = *pbrTls->mFs;
    const Scene *scene = MNRY_VERIFY(fs.mScene);
    const std::vector<geom::internal::EmissiveRegion>& emissiveRegions =
        scene->getEmissiveRegions();
    if (emissiveRegions.empty()) {
        return scene_rdl2::math::sBlack;
    }
    bool highQualitySample = pv.nonMirrorDepth == 0;
    int nLightSamples = highQualitySample ? mLightSamples : 1;
    int nBsdfSamples = highQualitySample ? mBsdfSamples : 1;
    scene_rdl2::math::Color LVe(0.0f);
    const scene_rdl2::math::Vec3f& p = isect.getP();
    const scene_rdl2::math::Vec3f& n = isect.getN();
    for (size_t eIndex = 0; eIndex < emissiveRegions.size(); ++eIndex) {
        const auto& emissiveRegion = emissiveRegions[eIndex];
        int emissiveVolumeId = emissiveRegion.mVolumeId;
        // filtered out bsdf lobes that are flagged invisible on emissiveRegion
        int validReflection = rayMaskToReflectionLobes(
            emissiveRegion.mVisibilityMask);
        int validTransmission = rayMaskToTransmissionLobes(
            emissiveRegion.mVisibilityMask);
        shading::Bsdf validBsdf;
        bool hasValidLobe = false;
        for (int lIndex = 0; lIndex < bsdf.getLobeCount(); ++lIndex) {
            shading::BsdfLobe* lobe = bsdf.getLobe(lIndex);
            if (lobe->matchesFlags(validReflection) ||
                lobe->matchesFlags(validTransmission)) {
                validBsdf.addLobe(lobe);
                hasValidLobe = true;
            }
        }
        if (!hasValidLobe) {
            continue;
        }
        BsdfOneSampler bsdfSampler(validBsdf, slice);
        // light strategy sampling for emissive volume
        VolumeEmissionLightSampler lSamples(sp, pv, nLightSamples, eIndex,
            highQualitySample, sequenceID);
        for (int ls = 0; ls < nLightSamples; ++ls) {
            scene_rdl2::math::Vec3f uls;
            lSamples.getSample(uls);
            scene_rdl2::math::Vec3f wi;
            float pdfLight;
            float tEnd;
            emissiveRegion.sample(p, uls[0], uls[1], uls[2], wi, pdfLight, tEnd, ray.time);
            if (pdfLight == 0.0f) {
                continue;
            }
            BsdfOneSampler::LobesContribution lobesContribution;
            float pdfBsdf;
            scene_rdl2::math::Color f = bsdfSampler.eval(wi, pdfBsdf, &lobesContribution);
            if (isEqual(f, scene_rdl2::math::sBlack)) {
                continue;
            }
            mcrt_common::Ray wiRay(p, wi, rayEpsilon, tEnd, ray.time,
                ray.getDepth() + 1);
            scene_rdl2::math::Color contribution = computeEmissiveVolumeIntegral(pbrTls, wiRay,
                emissiveVolumeId, sp, sequenceID);
            float cosineTerm = slice.getIncludeCosineTerm() ?
                1.0f : scene_rdl2::math::abs(scene_rdl2::math::dot(wi, n));
            float misWeight = 1.0f;
            if (nBsdfSamples != 0) {
                misWeight = powerHeuristic(
                    nLightSamples * pdfLight, nBsdfSamples * pdfBsdf);
            }
            contribution *= (misWeight * cosineTerm * pv.pathThroughput) /
                (pdfLight * nLightSamples);
            int nMatchedLobes = lobesContribution.mMatchedLobeCount;
            for (int lobeIndex = 0; lobeIndex < nMatchedLobes; ++lobeIndex) {
                scene_rdl2::math::Color lobeContribution = lobesContribution.mFs[lobeIndex] *
                    contribution;
                LVe += lobeContribution;
                // LPE
                if (aovs) {
                    EXCL_ACCUMULATOR_PROFILE(pbrTls, EXCL_ACCUM_AOVS);
                    const LightAovs &lightAovs = *fs.mLightAovs;
                    // transition
                    int lpeStateId = pv.lpeStateId;
                    lpeStateId = lightAovs.scatterEventTransition(pbrTls, lpeStateId,
                        bsdf, *(lobesContribution.mLobes[lobeIndex]));
                    lpeStateId = lightAovs.emissionEventTransition(pbrTls,
                        lpeStateId,
                        scene->getVolumeLabelId(emissiveRegion.mVolumeId));
                    // accumulate matching aovs
                    aovAccumLightAovs(pbrTls, *fs.mAovSchema, *fs.mLightAovs,
                        lobeContribution, nullptr, AovSchema::sLpePrefixNone, lpeStateId, aovs);
                }
            }
        }
        // bsdf strategy sampling for emissive volume
        VolumeEmissionBsdfSampler bSamples(sp, pv, nBsdfSamples, eIndex,
            highQualitySample, sequenceID);
        for (int bs = 0; bs < nBsdfSamples; ++bs) {
            BsdfOneSampler::LobesContribution lobesContribution;
            float uLobe;
            scene_rdl2::math::Vec2f ubs;
            bSamples.getSample(uLobe, ubs);
            scene_rdl2::math::Vec3f wi;
            float pdfBsdf;
            scene_rdl2::math::Color f = bsdfSampler.sample(uLobe, ubs[0], ubs[1], wi, pdfBsdf,
                &lobesContribution);
            if (pdfBsdf == 0.0f || isEqual(f, scene_rdl2::math::sBlack)) {
                continue;
            }
            float tEnd;
            float pdfLight = emissiveRegion.pdf(p, wi, tEnd, ray.time);
            // if pdfLight is 0, it means the ray doesn't hit any active
            // region in this EmissiveRegion
            if (pdfLight == 0.0f) {
                continue;
            }
            mcrt_common::Ray wiRay(p, wi, rayEpsilon, tEnd, ray.time,
                ray.getDepth() + 1);
            scene_rdl2::math::Color contribution = computeEmissiveVolumeIntegral(pbrTls, wiRay,
                emissiveVolumeId, sp, sequenceID);
            float cosineTerm = slice.getIncludeCosineTerm() ?
                1.0f : scene_rdl2::math::abs(scene_rdl2::math::dot(wi, n));
            float misWeight = 1.0f;
            if (nLightSamples != 0) {
                misWeight = powerHeuristic(
                    nBsdfSamples * pdfBsdf, nLightSamples * pdfLight);
            }
            contribution *= (misWeight * cosineTerm * pv.pathThroughput) /
                (pdfBsdf * nBsdfSamples);
            int nMatchedLobes = lobesContribution.mMatchedLobeCount;
            for (int lobeIndex = 0; lobeIndex < nMatchedLobes; ++lobeIndex) {
                scene_rdl2::math::Color lobeContribution = lobesContribution.mFs[lobeIndex] *
                    contribution;
                LVe += lobeContribution;
                // LPE
                if (aovs) {
                    EXCL_ACCUMULATOR_PROFILE(pbrTls, EXCL_ACCUM_AOVS);
                    const LightAovs &lightAovs = *fs.mLightAovs;
                    // transition
                    int lpeStateId = pv.lpeStateId;
                    lpeStateId = lightAovs.scatterEventTransition(pbrTls, lpeStateId,
                        bsdf, *(lobesContribution.mLobes[lobeIndex]));
                    lpeStateId = lightAovs.emissionEventTransition(pbrTls,
                        lpeStateId,
                        scene->getVolumeLabelId(emissiveRegion.mVolumeId));
                    // accumulate matching aovs
                    aovAccumLightAovs(pbrTls, *fs.mAovSchema, *fs.mLightAovs,
                        lobeContribution, nullptr, AovSchema::sLpePrefixNone, lpeStateId, aovs);
                }
            }
        }
    }
    return LVe;
}

scene_rdl2::math::Color
PathIntegrator::computeRadianceEmissiveRegionsSSS(pbr::TLState *pbrTls,
        const Subpixel& sp, const PathVertex& pv, const mcrt_common::Ray& ray,
        const scene_rdl2::math::Color& pathThroughput, const shading::Fresnel* transmissionFresnel,
        const shading::Bsdf& bsdf, const shading::BsdfLobe &lobe, const shading::BsdfSlice &slice,
        const scene_rdl2::math::Vec3f& p, const scene_rdl2::math::Vec3f& n,
        int subsurfaceSplitFactor, int subsurfaceIndex,
        float rayEpsilon, unsigned sssSampleID, bool isLocal,
        float* aovs) const
{
    EXCL_ACCUMULATOR_PROFILE(pbrTls, EXCL_ACCUM_VOL_INTEGRATION);

    const FrameState &fs = *pbrTls->mFs;
    const Scene *scene = MNRY_VERIFY(fs.mScene);
    const std::vector<geom::internal::EmissiveRegion>& emissiveRegions =
        scene->getEmissiveRegions();
    if (emissiveRegions.empty()) {
        return scene_rdl2::math::sBlack;
    }
    bool highQualitySample = pv.nonMirrorDepth == 0;
    // similar to splitFactor in computeRadianceSubsurfaceSample,
    // each projection sample use 1 light sample
    // seems expensive enought that we don't want further splitting
    int nLightSamples = 1;
    // the lobe passed into computeRadianceSubsurfaceSample is a LambertLobe
    // that light sampling should always be the better strategy.
    // As the result, we don't do bsdf strategy sampling and corresponding MIS
    // for bssrdf case
    scene_rdl2::math::Color LVe(0.0f);
    // light strategy sampling for emissive volume
    for (size_t eIndex = 0; eIndex < emissiveRegions.size(); ++eIndex) {
        const auto& emissiveRegion = emissiveRegions[eIndex];
        int emissiveVolumeId = emissiveRegion.mVolumeId;
        if (!emissiveRegion.canIlluminateSSS()) {
            continue;
        }
        VolumeEmissionLightSampler lSamples(sp, pv, nLightSamples, eIndex,
            highQualitySample, subsurfaceSplitFactor, subsurfaceIndex,
            sssSampleID, isLocal);
        scene_rdl2::math::Color emissionContribution(0.0f);
        for (int ls = 0; ls < nLightSamples; ++ls) {
            scene_rdl2::math::Vec3f uls;
            lSamples.getSample(uls);
            scene_rdl2::math::Vec3f wi;
            float pdfLight;
            float tEnd;
            emissiveRegion.sample(p, uls[0], uls[1], uls[2], wi, pdfLight, tEnd, ray.time);
            if (pdfLight == 0.0f) {
                continue;
            }
            //------------------------------
            // Evaluate lobe pdf. Note we do not use its contribution.
            // Note: We don't want a geometric surface check like we do when
            // evaluating bsdf because the subsurface transport makes wo irrelevant
            float pdfBsdf;
            scene_rdl2::math::Color f = lobe.eval(slice, wi, &pdfBsdf);
            if (isSampleInvalid(f, pdfBsdf)) {
                continue;
            }
            float cosineTerm = scene_rdl2::math::dot(wi, n);
            if (cosineTerm < scene_rdl2::math::sEpsilon) {
                continue;
            }
            mcrt_common::Ray wiRay(p, wi, rayEpsilon, tEnd, ray.time,
                ray.getDepth() + 1);
            scene_rdl2::math::Color contribution = computeEmissiveVolumeIntegral(pbrTls, wiRay,
                emissiveVolumeId, sp, sssSampleID);
            // Compute contribution applying incoming fresnel transmission
            scene_rdl2::math::Color fresnel = transmissionFresnel != nullptr ?
                transmissionFresnel->eval(scene_rdl2::math::abs(scene_rdl2::math::dot(n, wi))) :
                scene_rdl2::math::sWhite;
            contribution *=
                (pathThroughput * fresnel * f) /
                (pdfLight * nLightSamples);
            emissionContribution += contribution;
        }
        LVe += emissionContribution;
        // LPE
        if (aovs) {
            EXCL_ACCUMULATOR_PROFILE(pbrTls, EXCL_ACCUM_AOVS);
            const LightAovs &lightAovs = *fs.mLightAovs;
            // transition
            int lpeStateId = pv.lpeStateId;
            lpeStateId = lightAovs.subsurfaceEventTransition(pbrTls, lpeStateId, bsdf);
            lpeStateId = lightAovs.emissionEventTransition(pbrTls, lpeStateId,
                scene->getVolumeLabelId(emissiveRegion.mVolumeId));
            // accumulate matching aovs
            aovAccumLightAovs(pbrTls, *fs.mAovSchema, *fs.mLightAovs,
                emissionContribution, nullptr, AovSchema::sLpePrefixNone, lpeStateId, aovs);
        }
    }
    return LVe;
}

scene_rdl2::math::Color
PathIntegrator::computeRadianceEmissiveRegionsVolumes(pbr::TLState *pbrTls,
        const Subpixel& sp, const PathVertex& pv,
        const mcrt_common::Ray& ray,
        const VolumeProperties* volumeProperties,
        const GuideDistribution1D& densityDistribution,
        unsigned sequenceID, float* aovs, const RayState *rs) const
{
    EXCL_ACCUMULATOR_PROFILE(pbrTls, EXCL_ACCUM_VOL_INTEGRATION);

    const FrameState &fs = *pbrTls->mFs;
    const Scene *scene = MNRY_VERIFY(fs.mScene);
    const std::vector<geom::internal::EmissiveRegion>& emissiveRegions =
        scene->getEmissiveRegions();
    if (emissiveRegions.empty()) {
        return scene_rdl2::math::sBlack;
    }

    const scene_rdl2::math::Vec3f& wo = -ray.dir;
    bool highQualitySample = pv.nonMirrorDepth == 0;
    int nLightSamples = highQualitySample ? mLightSamples : 1;
    int scatterSampleCount = highQualitySample ? mVolumeIlluminationSamples : 1;
    float invN = 1.0f / (nLightSamples * scatterSampleCount);
    scene_rdl2::math::Color LVe(0.0f);
    for (size_t eIndex = 0; eIndex < emissiveRegions.size(); ++eIndex) {
        const auto& emissiveRegion = emissiveRegions[eIndex];
        int emissiveVolumeId = emissiveRegion.mVolumeId;
        if (!emissiveRegion.canIlluminateVolume()) {
            continue;
        }
        // equi-angular sampling
        VolumeScatterEventSampler equiAngularSamples(sp, pv,
            nLightSamples * scatterSampleCount, highQualitySample,
            SequenceType::VolumeEquiAngular, SequenceType::Light,
            sequenceID, SequenceType::VolumeEmission, eIndex);
        VolumeEmissionLightSampler lSamples(sp, pv, nLightSamples,
            scatterSampleCount, eIndex, highQualitySample, sequenceID);
        scene_rdl2::math::Color emissionContribution(0.0f);
        for (int ls = 0; ls < nLightSamples; ++ls) {
            for (int ts = 0; ts < scatterSampleCount; ++ts) {
                scene_rdl2::math::Vec3f uls;
                lSamples.getSample(uls);
                scene_rdl2::math::Vec3f pivot = emissiveRegion.getEquiAngularPivot(
                    uls[0], uls[1], uls[2], ray.time);
                float ue = equiAngularSamples.getSample(pv.nonMirrorDepth);
                // project pivot point to ray
                float offset = scene_rdl2::math::dot(pivot - ray.org, ray.dir);
                // the distance from pivot point to ray
                float D = length(pivot - (ray.org + ray.dir * offset));
                const auto& firstVp = volumeProperties[0];
                float ta = firstVp.mTStart - offset;
                uint32_t marchingStepsCount = densityDistribution.getSize();
                const auto& lastVp =
                    volumeProperties[marchingStepsCount - 1];
                float tb = lastVp.mTStart + lastVp.mDelta - offset;
                float thetaA = scene_rdl2::math::atan2(ta, D);
                float thetaB = scene_rdl2::math::atan2(tb, D);
                float te = sampleEquiAngular(ue, D, thetaA, thetaB);
                float pdfTe = pdfEquiAngular(te, D, thetaA, thetaB);
                if (pdfTe == 0.0f || !scene_rdl2::math::isfinite(pdfTe)) {
                    continue;
                }
                float tePrime = te + offset;
                uint32_t densityIndex = geom::findInterval(
                    marchingStepsCount + 1,
                    [&](int index) {
                        return volumeProperties[index].mTStart <= tePrime;
                    });
                const auto& vp = volumeProperties[densityIndex];
                if (isBlack(vp.mSigmaS)) {
                    continue;
                }
                // the transmittance of this scatter point
                scene_rdl2::math::Color trScatter = vp.mTransmittance * vp.mTransmittanceH *
                    exp(-vp.mSigmaT * (tePrime - vp.mTStart)) *
                    exp(-vp.mSigmaTh * (tePrime - vp.mTStart));
                scene_rdl2::math::Vec3f scatterPoint = ray.org + ray.dir * tePrime;
                scene_rdl2::math::Vec3f wi = scene_rdl2::math::normalize(pivot - scatterPoint);
                float tEnd;
                float pdfWi = emissiveRegion.pdf(scatterPoint, wi, tEnd, ray.time);
                if (pdfWi == 0.0f || !scene_rdl2::math::isfinite(pdfWi)) {
                    continue;
                }
                mcrt_common::Ray wiRay(scatterPoint, wi, 0.0f, tEnd, ray.time,
                    ray.getDepth() + 1);
                scene_rdl2::math::Color contribution = computeEmissiveVolumeIntegral(pbrTls, wiRay,
                    emissiveVolumeId, sp, sequenceID);
                float ph = VolumePhase(vp.mG).eval(wo, wi);
                contribution *= invN * pv.pathThroughput *
                    vp.mSigmaS * trScatter * ph / (pdfTe * pdfWi);
                emissionContribution += contribution;
            }
        }
        LVe += emissionContribution;
        // LPE
        if (fs.mLightAovs->hasEntries() && (aovs || rs)) {
            EXCL_ACCUMULATOR_PROFILE(pbrTls, EXCL_ACCUM_AOVS);
            const LightAovs &lightAovs = *fs.mLightAovs;
            // transition
            int lpeStateId = pv.lpeStateId;
            lpeStateId = lightAovs.volumeEventTransition(pbrTls, lpeStateId);
            lpeStateId = lightAovs.emissionEventTransition(pbrTls, lpeStateId,
                scene->getVolumeLabelId(emissiveRegion.mVolumeId));
            // accumulate matching aovs
            if (aovs) {
                aovAccumLightAovs(pbrTls, *fs.mAovSchema, *fs.mLightAovs,
                    emissionContribution, nullptr, AovSchema::sLpePrefixNone, lpeStateId, aovs);
            } else {
                MNRY_ASSERT(rs && fs.mExecutionMode == mcrt_common::ExecutionMode::VECTORIZED);
                aovAccumLightAovsBundled(pbrTls, *fs.mAovSchema, *fs.mLightAovs,
                    emissionContribution, nullptr, AovSchema::sLpePrefixNone, lpeStateId, sp.mPixel, rs->mDeepDataHandle);
            }
        }
    }
    return LVe;
}

scene_rdl2::math::Color
PathIntegrator::transmittance(pbr::TLState *pbrTls, const mcrt_common::Ray& ray,
                              uint32_t pixel, int subpixelIndex, unsigned sequenceID,
                              const Light* light, float scaleFactor, bool estimateInScatter) const
{

    EXCL_ACCUMULATOR_PROFILE(pbrTls, EXCL_ACCUM_VOL_INTEGRATION);

    size_t intervalCount =
        collectVolumeIntervals(pbrTls, ray,
                               scene_rdl2::rdl2::SHADOW,
                               (light) ? static_cast<const void *>(light->getRdlLight()) : nullptr,
                               estimateInScatter);

    mcrt_common::ThreadLocalState* tls = pbrTls->mTopLevelTls;
    auto& volumeRayState = tls->mGeomTls->mVolumeRayState;

    if (intervalCount > 1 &&
        volumeRayState.getOriginVolumeId() == geom::internal::VolumeRayState::ORIGIN_VOLUME_INIT) {
        // In this case, we have volume along the ray but ray origin is assumed outside of the volume.
        // So, we set ORIGIN_VOLUME_EMPTY to originVolumeId.
        volumeRayState.setOriginVolumeId(geom::internal::VolumeRayState::ORIGIN_VOLUME_EMPTY, 0.f);
    }

    // no volume across this ray
    if (intervalCount == 0 &&
        volumeRayState.getCurrentVolumeRegions().isVacuum()) {
        return scene_rdl2::math::Color(1.0f);
    }
    geom::internal::VolumeTransition* intervals =
        volumeRayState.getVolumeIntervals();

    // now integrate the transmittance for each non-overlapping interval
    SequenceIDIntegrator sid(pixel, subpixelIndex, SequenceType::VolumeAttenuation, sequenceID);
    IntegratorSample1D trSamples(sid);
    scene_rdl2::math::Color tr(1.0f);
    float tStart = ray.tnear;
    for (size_t i = 0; i < intervalCount; ++i) {
        if (intervals[i].mT > ray.tfar) {
            break;
        }
        if ((intervals[i].mT - tStart) > scene_rdl2::math::sEpsilon) {
            // looking for a tauThreshold that
            // tr * exp(-tauThreshold) = transmittanceThreshold ->
            // tauThreshold = -log(transmittanceThreshold / tr)
            float tauThreshold = scene_rdl2::math::isZero(mVolumeTransmittanceThreshold) ?
                scene_rdl2::math::inf :
                -scene_rdl2::math::log(mVolumeTransmittanceThreshold / luminance(tr));
            tr *= transmittanceSubinterval(pbrTls, tStart, intervals[i].mT,
                volumeRayState.getCurrentVolumeRegions(), ray.time, ray.getDepth(),
                trSamples, tauThreshold, light, scaleFactor);
            if (luminance(tr) <= mVolumeTransmittanceThreshold) {
                return scene_rdl2::math::Color(0.0f);
            }
            tStart = intervals[i].mT;
        }
        if (intervals[i].mIsEntry) {
            volumeRayState.turnOn(intervals[i].mVolumeId, intervals[i].mPrimitive);
        }
        if (intervals[i].mIsExit) {
            volumeRayState.turnOff(intervals[i].mVolumeId);
        }
    }
    if ((ray.tfar - tStart) > scene_rdl2::math::sEpsilon) {
        float tauThreshold = scene_rdl2::math::isZero(mVolumeTransmittanceThreshold) ?
            scene_rdl2::math::inf :
            -scene_rdl2::math::log(mVolumeTransmittanceThreshold / luminance(tr));
        tr *= transmittanceSubinterval(pbrTls, tStart, ray.tfar,
            volumeRayState.getCurrentVolumeRegions(), ray.time, ray.getDepth(),
            trSamples, tauThreshold, light, scaleFactor);
        if (luminance(tr) <= mVolumeTransmittanceThreshold) {
            return scene_rdl2::math::Color(0.0f);
        }
    }

    return tr;
}

scene_rdl2::math::Color
PathIntegrator::transmittanceSubinterval(pbr::TLState *pbrTls,
        float t0, float t1,
        const geom::internal::VolumeRegions& volumeRegions, float time,
        int depth, const IntegratorSample1D& trSamples,
        float tauThreshold, const Light* light, float scaleFactor) const
{
    mcrt_common::ThreadLocalState* tls = pbrTls->mTopLevelTls;

    EXCL_ACCUMULATOR_PROFILE(tls, EXCL_ACCUM_VOL_INTEGRATION);

    scene_rdl2::alloc::Arena* arena = &(tls->mArena);
    SCOPED_MEM(arena);
    auto& volumeRayState = tls->mGeomTls->mVolumeRayState;
    // TODO maybe we can pass in visitedVolumeCount instead
    int* volumeIds = arena->allocArray<int>(
        volumeRayState.getVolumeAssignmentTable()->getVolumeCount());
    int volumeRegionsCount = volumeRegions.getVolumeIds(volumeIds);
    scene_rdl2::math::Color tr(1.0f);

    // vacuum space
    if (volumeRegionsCount == 0) {
        return tr;
    }
    if (volumeRegionsCount > 0 && t1 >= sHitEpsilonEnd * sDistantLightDistance) {
        return scene_rdl2::math::Color(0.0f);
    }

    const auto& volumeSampleInfo = volumeRayState.getVolumeSampleInfo();
    bool isHomogenous = true;
    for (int i = 0; i < volumeRegionsCount; ++i) {
        isHomogenous &= volumeSampleInfo[volumeIds[i]].isHomogenous();
    }

    if (isHomogenous) {
        // analytical solution (Beer's law) is available
        float rndVal = 0.f;
        if (mVolumeOverlapMode == VolumeOverlapMode::RND) {
            trSamples.getSample(&rndVal, 0);
        }
        float rayVolumeDepth = t1 - t0; // total distance ray travels through the volume
        scene_rdl2::math::Color sigmaT = evalSigmaT(pbrTls, volumeRegionsCount, volumeIds, mVolumeOverlapMode, rndVal,
            volumeSampleInfo, t0, time, light, rayVolumeDepth);
        tr = exp(-sigmaT * (t1 - t0) * scaleFactor);
    } else {
        // For now using traditional ray marching approach.
        // Once we have the ability to query averge value on leaf node,
        // we can potentially replace this with residual ratio tracking

        // figure out the step size: when there are multiple volume regions
        // in this interval, use the smallest feature size for stepping
        float minFeatureSize = scene_rdl2::math::inf;
        for (int i = 0; i < volumeRegionsCount; ++i) {
            // when there are multiple volume regions in this interval,
            // use the smallest non-homogenous feature size for stepping
            // (stepping only happens when there is a non-homogenous volume region,
            //  and homogenous volumes do not have a valid feature size)
            const auto& sampleInfo = volumeSampleInfo[volumeIds[i]];
            if (!sampleInfo.isHomogenous()) {
                minFeatureSize = scene_rdl2::math::min(minFeatureSize, volumeSampleInfo[volumeIds[i]].getFeatureSize());
            }
        }
        float u;
        trSamples.getSample(&u, 0);
        float stepSize = minFeatureSize * mInvVolumeShadowQuality * scene_rdl2::math::max(depth - 1, 1);
        float t = t0 + stepSize * u;
        scene_rdl2::math::Color tau(0.0f);

        while (t < t1) {
            float rndVal = 0.f;
            if (mVolumeOverlapMode == VolumeOverlapMode::RND) {
                trSamples.getSample(&rndVal, 0);
            }
            scene_rdl2::math::Color sigmaT = evalSigmaT(pbrTls, volumeRegionsCount, volumeIds, mVolumeOverlapMode,
                                                        rndVal, volumeSampleInfo, t, time, light, -1);
            tau += sigmaT;
            if (luminance(tau * stepSize * scaleFactor) > tauThreshold) {
                return scene_rdl2::math::Color(0.0f);
            }
            t += stepSize;
        }
        // the estimator for integrate optical thickness from t0 to t1 is
        // (1 / N) * (sum(sigmaT(Pi) / pdf(Pi))) where pdf(Pi) = 1 / (t1 - t0)
        // and stepSize = (t1 - t0) / N
        tau *= stepSize;
        // this is the "volume attenuation factor" in
        // multiple scattering approximation
        tau *= scaleFactor;
        tr = exp(-tau);
    }
    return tr;
}

void
PathIntegrator::decoupledRayMarching(pbr::TLState *pbrTls,
        VolumeTransmittance& vt, scene_rdl2::math::Color* perVolumeLVe,
        const size_t maxStepCount, size_t& marchingStepsCount, float t0, float t1,
        float time, int depth, const geom::internal::VolumeRayState& volumeRayState,
        int* volumeIds, VolumeProperties* volumeProperties,
        const IntegratorSample1D& trSamples, const IntegratorSample1D& trSamples2,
        bool& reachTransmittanceThreshold,
        DeepParams* deepParams) const
{
    // due to the float point precission accumulated error,
    // the marching step count can still pass max step count we estimated
    // earlier slightly...in this case we'll just skip the further ray marching
    // (otherwise the exceeding steps can result to memory corruption)
    if (marchingStepsCount >= maxStepCount) {
        return;
    }
    int volumeRegionsCount = volumeRayState.getCurrentVolumeRegions().getVolumeIds(
        volumeIds);
    if (volumeRegionsCount == 0) {
        return;
    }
    // This shouldn't happen in proper setup, but can be triggered with
    // incorrect volume assignment (for example, assign a face with volume)
    // The only handling we can do is ingnoring this case
    if (volumeRegionsCount > 0 && t1 >= sHitEpsilonEnd * sDistantLightDistance) {
        return;
    }

    EXCL_ACCUMULATOR_PROFILE(pbrTls, EXCL_ACCUM_VOL_INTEGRATION);

    const auto& volumeSampleInfo = volumeRayState.getVolumeSampleInfo();
    bool isHomogenous = true;
    float minFeatureSize = scene_rdl2::math::inf;
    for (int i = 0; i < volumeRegionsCount; ++i) {
        const auto& sampleInfo = volumeSampleInfo[volumeIds[i]];
        isHomogenous &= sampleInfo.isHomogenous();
        // when there are multiple volume regions in this interval,
        // use the smallest non-homogenous feature size for stepping
        // (stepping only happens when there is a non-homogenous volume region,
        //  and homogenous volumes do not have a valid feature size)
        if (!sampleInfo.isHomogenous()) {
            minFeatureSize = scene_rdl2::math::min(minFeatureSize, sampleInfo.getFeatureSize());
        }
    }

    bool sampleEmission = perVolumeLVe != nullptr;
    scene_rdl2::alloc::Arena* arena = pbrTls->mArena;
    SCOPED_MEM(arena);
    scene_rdl2::math::Color* emissions = nullptr;
    if (sampleEmission) {
        int volCount = mVolumeOverlapMode == VolumeOverlapMode::SUM ?
            volumeRayState.getVolumeAssignmentTable()->getVolumeCount() : 1;
        emissions = arena->allocArray<scene_rdl2::math::Color>(volCount);
    }

    if (isHomogenous) {
        scene_rdl2::math::Color sigmaT, sigmaS;
        // sigmas for holdouts (cutouts)
        scene_rdl2::math::Color sigmaTh, sigmaSh;
        float surfaceOpacityThreshold;
        float g;
        int volumeIdSampled;
        float rndVal = 0.f;
        if (mVolumeOverlapMode == VolumeOverlapMode::RND) {
            trSamples2.getSample(&rndVal, 0);
        }
        float rayVolumeDepth = t1 - t0; // total distance ray travels through the volume
        evalVolumeShaders(pbrTls, volumeRegionsCount, volumeIds, mVolumeOverlapMode, rndVal,
            volumeSampleInfo, t0, time, (depth == 0),
            &sigmaT, &sigmaS, &sigmaTh, &sigmaSh,
            &surfaceOpacityThreshold, &g, emissions, &volumeIdSampled, rayVolumeDepth);
        int assignmentId = volumeRayState.getAssignmentId(volumeIdSampled);
        volumeProperties[marchingStepsCount++] = VolumeProperties(
            sigmaT, sigmaS, sigmaTh, sigmaSh, vt.mTransmittanceE, vt.mTransmittanceH,
            surfaceOpacityThreshold, g, t0, t1 - t0, assignmentId);

        scene_rdl2::math::Color tr = exp(-sigmaT * (t1 - t0));
        scene_rdl2::math::Color trh = exp(-sigmaTh * (t1 - t0));

        // integrate emission * e^{-sigmaT * t) from t0 to t1
        if (sampleEmission) {
            if (mVolumeOverlapMode == VolumeOverlapMode::SUM) {
                for (int i = 0; i < volumeRegionsCount; ++i) {
                    const scene_rdl2::math::Color& emission = emissions[volumeIds[i]];
                    scene_rdl2::math::Color LVeLocal = vt.transmittance() * emission * (scene_rdl2::math::Color(1.0f) -
                                                                                        tr);
                    LVeLocal.r = scene_rdl2::math::isZero(sigmaT.r) ? 0.0f : LVeLocal.r / sigmaT.r;
                    LVeLocal.g = scene_rdl2::math::isZero(sigmaT.g) ? 0.0f : LVeLocal.g / sigmaT.g;
                    LVeLocal.b = scene_rdl2::math::isZero(sigmaT.b) ? 0.0f : LVeLocal.b / sigmaT.b;
                    perVolumeLVe[volumeIds[i]] += LVeLocal;
                }
            } else {
                MNRY_ASSERT(volumeIdSampled != -1);
                scene_rdl2::math::Color LVeLocal = vt.transmittance() * emissions[0] * (scene_rdl2::math::Color(1.0f) -
                                                                                        tr);
                LVeLocal.r = scene_rdl2::math::isZero(sigmaT.r) ? 0.0f : LVeLocal.r / sigmaT.r;
                LVeLocal.g = scene_rdl2::math::isZero(sigmaT.g) ? 0.0f : LVeLocal.g / sigmaT.g;
                LVeLocal.b = scene_rdl2::math::isZero(sigmaT.b) ? 0.0f : LVeLocal.b / sigmaT.b;
                perVolumeLVe[volumeIdSampled] += LVeLocal;
            }
        }

        vt.update(tr, trh);

    } else {
        float u;
        trSamples.getSample(&u, 0);
        float tStart = t0;
        float stepSize = minFeatureSize * mInvVolumeQuality * scene_rdl2::math::max(depth - 1, 1);
        float t = t0 + stepSize * u;
        while (t < t1) {
            if (luminance(vt.mTransmittanceAlpha) < mVolumeTransmittanceThreshold) {
                reachTransmittanceThreshold = true;
                return;
            }
            scene_rdl2::math::Color sigmaT, sigmaS;
            scene_rdl2::math::Color sigmaTh, sigmaSh;
            float surfaceOpacityThreshold;
            float g;
            int volumeIdSampled;
            float rndVal = 0.f;
            if (mVolumeOverlapMode == VolumeOverlapMode::RND) {
                trSamples2.getSample(&rndVal, 0);
            }
            evalVolumeShaders(pbrTls, volumeRegionsCount, volumeIds, mVolumeOverlapMode, rndVal,
                volumeSampleInfo, t, time, (depth == 0),
                &sigmaT, &sigmaS, &sigmaTh, &sigmaSh,
                &surfaceOpacityThreshold, &g, emissions, &volumeIdSampled, -1);
            float delta = t - tStart;
            int assignmentId = volumeRayState.getAssignmentId(volumeIdSampled);
            volumeProperties[marchingStepsCount++] = VolumeProperties(
                sigmaT, sigmaS, sigmaTh, sigmaSh, vt.mTransmittanceE, vt.mTransmittanceH,
                surfaceOpacityThreshold, g, tStart, delta, assignmentId);
            if (sampleEmission) {
                if (mVolumeOverlapMode == VolumeOverlapMode::SUM) {
                    for (int i = 0; i < volumeRegionsCount; ++i) {
                        const scene_rdl2::math::Color& emission = emissions[volumeIds[i]];
                        perVolumeLVe[volumeIds[i]] += vt.transmittance() * emission * delta;

                        if (deepParams) {
                            deepParams->mDeepBuffer->addVolumeSample(
                                pbrTls,
                                deepParams->mPixelX,
                                deepParams->mPixelY,
                                t,
                                vt.transmittance(),
                                emission * delta,
                                nullptr);
                        }
                    }
                } else {
                    // Only one volume emitting now.  Its emission has been scaled
                    // properly to compensate for only one volume emitting at a time.
                    MNRY_ASSERT(volumeIdSampled != -1);
                    const scene_rdl2::math::Color& emission = emissions[0];
                    perVolumeLVe[volumeIdSampled] += vt.transmittance() * emission * delta;

                    if (deepParams) {
                        deepParams->mDeepBuffer->addVolumeSample(
                            pbrTls,
                            deepParams->mPixelX,
                            deepParams->mPixelY,
                            t,
                            vt.transmittance(),
                            emission * delta,
                            nullptr);
                    }
                }
            }

            scene_rdl2::math::Color tr = exp(-sigmaT * delta);
            scene_rdl2::math::Color trh = exp(-sigmaTh * delta);
            vt.update(tr, trh);

            tStart = t;
            t += stepSize;
            if (marchingStepsCount >= maxStepCount) {
                return;
            }
        }
        if (luminance(vt.mTransmittanceAlpha) < mVolumeTransmittanceThreshold) {
            reachTransmittanceThreshold = true;
            return;
        }
        if (marchingStepsCount >= maxStepCount) {
            return;
        }
        // last step
        if ((t - t1) > scene_rdl2::math::sEpsilon) {
            scene_rdl2::math::Color sigmaT, sigmaS;
            scene_rdl2::math::Color sigmaTh, sigmaSh;
            float surfaceOpacityThreshold;
            float g;
            int volumeIdSampled;
            float rndVal = 0.f;
            if (mVolumeOverlapMode == VolumeOverlapMode::RND) {
                trSamples2.getSample(&rndVal, 0);
            }
            evalVolumeShaders(pbrTls, volumeRegionsCount, volumeIds, mVolumeOverlapMode, rndVal,
                volumeSampleInfo, t1, time, (depth == 0),
                &sigmaT, &sigmaS, &sigmaTh, &sigmaSh,
                &surfaceOpacityThreshold, &g, emissions, &volumeIdSampled, -1);
            float delta = t1 - tStart;
            if (sampleEmission) {
                if (mVolumeOverlapMode == VolumeOverlapMode::SUM) {
                    for (int i = 0; i < volumeRegionsCount; ++i) {
                        const scene_rdl2::math::Color& emission = emissions[volumeIds[i]];
                        perVolumeLVe[volumeIds[i]] += vt.transmittance() * emission * delta;

                        if (deepParams) {
                            deepParams->mDeepBuffer->addVolumeSample(
                                pbrTls,
                                deepParams->mPixelX,
                                deepParams->mPixelY,
                                t,
                                vt.transmittance(),
                                emission * delta,
                                nullptr);
                        }
                    }
                } else {
                    MNRY_ASSERT(volumeIdSampled != -1);
                    const scene_rdl2::math::Color& emission = emissions[0];
                    perVolumeLVe[volumeIdSampled] += vt.transmittance() * emission * delta;

                    if (deepParams) {
                        deepParams->mDeepBuffer->addVolumeSample(
                            pbrTls,
                            deepParams->mPixelX,
                            deepParams->mPixelY,
                            t,
                            vt.transmittance(),
                            emission * delta,
                            nullptr);
                    }
                }
            }
            int assignmentId = volumeRayState.getAssignmentId(volumeIdSampled);
            volumeProperties[marchingStepsCount++] = VolumeProperties(
                sigmaT, sigmaS, sigmaTh, sigmaSh, vt.mTransmittanceE, vt.mTransmittanceH,
                surfaceOpacityThreshold, g, tStart, delta, assignmentId);

            scene_rdl2::math::Color tr = exp(-sigmaT * delta);
            scene_rdl2::math::Color trh = exp(-sigmaTh * delta);
            vt.update(tr, trh);
        }
    }
}

} // namespace pbr
} // namespace moonray

