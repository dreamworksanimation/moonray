// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

///
/// @file PathIntegratorSubsurface.cc
/// $Id$
///

#include "PathIntegrator.h"
#include "PathIntegratorUtil.h"
#include "VolumeTransmittance.h"
#include <moonray/rendering/pbr/camera/Camera.h>
#include <moonray/rendering/pbr/core/Aov.h>
#include <moonray/rendering/pbr/core/Constants.h>
#include <moonray/rendering/pbr/core/DebugRay.h>
#include <moonray/rendering/pbr/core/PbrTLState.h>
#include <moonray/rendering/pbr/core/RayState.h>
#include <moonray/rendering/pbr/core/Scene.h>
#include <moonray/rendering/pbr/core/Util.h>
#include <moonray/rendering/pbr/core/VolumePhase.h>
#include <moonray/rendering/pbr/light/Light.h>
#include <moonray/rendering/pbr/light/LightSet.h>

#include <moonray/rendering/bvh/shading/State.h>
#include <moonray/rendering/geom/IntersectionInit.h>
#include <moonray/rendering/geom/prim/GeomTLState.h>
#include <moonray/rendering/mcrt_common/Ray.h>
#include <moonray/rendering/bvh/shading/Intersection.h>
#include <moonray/rendering/shading/bsdf/BsdfLambert.h>
#include <moonray/rendering/shading/bsdf/BsdfSlice.h>
#include <moonray/rendering/shading/bsdf/Fresnel.h>
#include <moonray/rendering/shading/bssrdf/Bssrdf.h>
#include <moonray/rendering/shading/bssrdf/VolumeSubsurface.h>
#include <moonray/rendering/shading/Material.h>
#include <scene_rdl2/scene/rdl2/VisibilityFlags.h>

// using namespace scene_rdl2::math; // can't use this as it breaks openvdb in clang.

namespace moonray {

namespace pbr {

using namespace moonray::mcrt_common;
using namespace moonray::shading;

//----------------------------------------------------------------------------

finline static scene_rdl2::math::Color
evalFresnel(const Fresnel *fresnel, const scene_rdl2::math::Vec3f &N, const scene_rdl2::math::Vec3f &w)
{
    return (fresnel != nullptr ? fresnel->eval(scene_rdl2::math::abs(dot(N, w))) : scene_rdl2::math::sWhite);
}

// Utility Function to decide if we can substitute expensive subsurface computation
// with a cheaper approximation instead. Right now, we check if
// the ray-footprint is bigger than the 'scattering radius' and if so,
// we replace subsurface with Lambertian.
// Note a similar user-controlled substitution happens with a
// 'max subsurface per path' parameter.
// This substitution is an optimization for when the full subsurface computation
// has no visual effect since the ray-footprint being approximated is
// bigger than the scattering radius.
finline bool
canSubstituteSubsurface(float sssRadius,
                        const shading::Intersection &isect)
{
    const scene_rdl2::math::Vec3f dPds = isect.getdPds();
    const scene_rdl2::math::Vec3f dPdt = isect.getdPdt();
    const scene_rdl2::math::Vec3f dPdx = dPds * isect.getdSdx() + dPdt * isect.getdTdx();
    const scene_rdl2::math::Vec3f dPdy = dPds * isect.getdSdy() + dPdt * isect.getdTdy();
    // Avg ray footprint length (faster than actual area computation)
    float rayFootprintRSqr = (lengthSqr(dPdx) + lengthSqr(dPdy)) * 0.5f;
    float ratioFootprintToRadius = rayFootprintRSqr / (sssRadius * sssRadius);
    // 0.85f is an empirical threshold based on local tests.
    // It could be modified in the future if we encounter any issues.
    // So far, disabling subsurface when the ray-footprint's around 85% of the
    // scattering radius seems like a good enough approximation for optimum speed.
    static const float SSS_THRESHOLD = 0.85f;
    return (ratioFootprintToRadius > SSS_THRESHOLD);
}

// References for bssrdf based subsurface scattering:
// [1] BSSRDF Importance Sampling - King et al
// [2] PBRTv3 Section 15.4.1 - Sampling the SeparableBSSRDF

//----------------------------------------------------------------------------
// TODO: Review all rays and their transfer status

// Sample direct and indirect lighting at a given subsurface sample and compute
// the reflected radiance. We apply the given pathThroughput and
// transmissionFresnel, if any, to the contribution of the samples.
scene_rdl2::math::Color
PathIntegrator::computeRadianceSubsurfaceSample(pbr::TLState *pbrTls,
        const Bsdf &bsdf, const Subpixel &sp,
        const PathVertex &pv, const RayDifferential &parentRay,
        const scene_rdl2::math::Vec3f &dNdx, const scene_rdl2::math::Vec3f &dNdy,
        const scene_rdl2::math::Color &pathThroughput, const Fresnel *transmissionFresnel,
        const LightSet &lightSet, BsdfLobe &lobe,
        const BsdfSlice &slice, const scene_rdl2::math::Vec3f &P, const scene_rdl2::math::Vec3f &N,
        int subsurfaceSplitFactor, int computeRadianceSplitFactor,
        int subsurfaceIndex, bool doIndirect, float rayEpsilon, float shadowRayEpsilon,
        unsigned sssSampleID, unsigned &sequenceID, bool isLocal,
        float *aovs, const shading::Intersection &isect) const
{
    scene_rdl2::math::Color radiance = scene_rdl2::math::sBlack;
    // Estimate emissive volume region energy contribution
    radiance += computeRadianceEmissiveRegionsSSS(pbrTls, sp, pv, parentRay,
        pathThroughput, transmissionFresnel, bsdf, lobe, slice, P, N,
        subsurfaceSplitFactor, subsurfaceIndex,
        rayEpsilon, sssSampleID, isLocal, aovs);
    MNRY_ASSERT(pbrTls->isIntegratorAccumulatorRunning());

    Statistics &stats = pbrTls->mStatistics;
    scene_rdl2::math::Color pt = pathThroughput / float(computeRadianceSplitFactor);

    // Sample from the lights
    stats.addToCounter(STATS_LIGHT_SAMPLES,
        lightSet.getLightCount() * computeRadianceSplitFactor);

    IntegratorSample3D lightSamples;
    IntegratorSample2D lightFilterSamples;
    IntegratorSample3D lightFilterSamples3D;

    const Scene *scene = MNRY_VERIFY(pbrTls->mFs->mScene);
    bool lightFilterNeedsSamples = scene->lightFilterNeedsSamples();

    for (int lightIndex = 0; lightIndex < lightSet.getLightCount(); lightIndex++) {
        const Light* const light = lightSet.getLight(lightIndex);

        // Sampling WIP
        // We want one shared sequence for depth 0 across subpixels and local
        // or global sub-surface samples
        const SequenceIDIntegrator lightSid(
            pv.nonMirrorDepth,
            sp.mPixel,
            light->getHash(),
            (isLocal) ? SequenceType::BssrdfLocalLight : SequenceType::BssrdfGlobalLight,
            (pv.nonMirrorDepth == 0) ? 0 : sp.mSubpixelIndex * subsurfaceSplitFactor + subsurfaceIndex,
            sssSampleID);

        // Using SSS / SP / L
        // Note: using SP / SSS / L caused more correlation
        int samplesSoFar = (pv.nonMirrorDepth == 0) ?
            subsurfaceIndex * sp.mPixelSamples * computeRadianceSplitFactor +
            sp.mSubpixelIndex * computeRadianceSplitFactor : 0;
        lightSamples.resume(lightSid, samplesSoFar);

        if (lightFilterNeedsSamples) {
            const SequenceIDIntegrator lightFilterSid(
                pv.nonMirrorDepth,
                sp.mPixel,
                light->getHash(),
                SequenceType::LightFilter,
                (pv.nonMirrorDepth == 0) ? 0 : sp.mSubpixelIndex * subsurfaceSplitFactor + subsurfaceIndex,
                sssSampleID);
            const SequenceIDIntegrator lightFilter3DSid(
                pv.nonMirrorDepth,
                sp.mPixel,
                light->getHash(),
                SequenceType::LightFilter3D,
                (pv.nonMirrorDepth == 0) ? 0 : sp.mSubpixelIndex * subsurfaceSplitFactor + subsurfaceIndex,
                sssSampleID);

            lightFilterSamples.resume(lightFilterSid, samplesSoFar);
            lightFilterSamples3D.resume(lightFilter3DSid, samplesSoFar);
        }

        // if light's visibility mask does not match, skip this light.
        if (!(scene_rdl2::rdl2::DIFFUSE_REFLECTION & light->getVisibilityMask())) {
            continue;
        }

        const LightFilterList *lightFilterList = lightSet.getLightFilterList(lightIndex);

        for (int sampleIndex = 0; sampleIndex < computeRadianceSplitFactor;
            sampleIndex++) {
            // Draw a sample for this light
            scene_rdl2::math::Vec3f lightSample;
            lightSamples.getSample(&lightSample[0], pv.nonMirrorDepth);
            LightFilterRandomValues lightFilterSample = {
                            scene_rdl2::math::Vec2f(0.f, 0.f), 
                            scene_rdl2::math::Vec3f(0.f, 0.f, 0.f)};
            if (lightFilterNeedsSamples) {
                lightFilterSamples.getSample(&lightFilterSample.r2[0], pv.nonMirrorDepth);
                lightFilterSamples3D.getSample(&lightFilterSample.r3[0], pv.nonMirrorDepth);
            }

            // Warp the sample from this light
            scene_rdl2::math::Vec3f wi;
            LightIntersection lightIsect;
            if (!light->sample(P, &N, parentRay.getTime(), lightSample,
                wi, lightIsect, parentRay.getDirFootprint())) {
                continue;
            }

            // Evaluate the light sample and pdf
            float lightPdf;
            scene_rdl2::math::Color Li = light->eval(pbrTls->mTopLevelTls, wi, P, lightFilterSample,
                parentRay.getTime(), lightIsect, false, lightFilterList, parentRay.getDirFootprint(), &lightPdf);

            if (isSampleInvalid(Li, lightPdf)) {
                continue;
            }
            // Evaluate the lobe pdf. Note we only use the lobe for sampling
            // to apply MIS, not for its contribution.
            // Note: We don't want a geometric surface check like we do when
            // evaluating bsdf because the subsurface transport makes wo irrelevant
            float bsdfPdf;
            scene_rdl2::math::Color f = lobe.eval(slice, wi, &bsdfPdf);
            if (isSampleInvalid(f, bsdfPdf)) {
                continue;
            }
            // Check orientation with surface normal. This is a bit unneccessary
            // due to the call to isSampleInvalid() above.
            float cosWi = dot(N, wi);
            if (cosWi < scene_rdl2::math::sEpsilon) {
                continue;
            }
            // Compute contribution applying incoming fresnel transmission
            scene_rdl2::math::Color fresnel = evalFresnel(transmissionFresnel, N, wi);
            // Apply MIS power heuristic
            scene_rdl2::math::Color contribution = pt * Li * fresnel * f / lightPdf *
                powerHeuristic(lightPdf, bsdfPdf);
            // TODO: Apply russian roulette
            // Check shadowing
            float tfar = lightIsect.distance * sHitEpsilonEnd;
            float time = parentRay.getTime();
            int rayDepth = parentRay.getDepth() + 1;
            Ray shadowRay(P, wi, rayEpsilon, tfar, time, rayDepth);
            float presence = 0.0f;
            int32_t assignmentId = isect.getLayerAssignmentId();
            if (isRayOccluded(pbrTls, light, shadowRay, rayEpsilon, shadowRayEpsilon, presence, assignmentId)) {
                // LPE
                if (aovs) {
                    EXCL_ACCUMULATOR_PROFILE(pbrTls, EXCL_ACCUM_AOVS);
                    const FrameState &fs = *pbrTls->mFs;
                    const LightAovs &lightAovs = *fs.mLightAovs;
                    // transition
                    int lpeStateId = pv.lpeStateId;
                    lpeStateId = lightAovs.subsurfaceEventTransition(pbrTls,
                        lpeStateId, bsdf);
                    // intentionally ignore the the internal lambert
                    // lobe event (see MOONRAY-1957).
                    lpeStateId = lightAovs.lightEventTransition(pbrTls,
                        lpeStateId, light);
                    // accumulate matching aovs
                    aovAccumVisibilityAovs(pbrTls, *fs.mAovSchema, lightAovs,
                        scene_rdl2::math::Vec2f(0.0f, 1.0f), lpeStateId, aovs);
                }
                continue;
            }
            // Raydb debug
            if (lightIsect.distance < sDistantLightDistance) {
                Ray debugRay(P, wi, 0.0f);
                RAYDB_EXTEND_RAY_NO_HIT(pbrTls, debugRay, lightIsect.distance);
                RAYDB_SET_CONTRIBUTION(pbrTls, Li);
                RAYDB_ADD_TAGS(pbrTls, TAG_AREALIGHT);
            } else {
                Ray debugRay(P, wi, 0.0f);
                RAYDB_EXTEND_RAY_NO_HIT(pbrTls, debugRay, 40.0f);
                RAYDB_SET_CONTRIBUTION(pbrTls, Li);
                RAYDB_ADD_TAGS(pbrTls, TAG_ENVLIGHT);
            }
            // shadowRay can be modified in occlusion query
            Ray trRay(P, wi, scene_rdl2::math::max(rayEpsilon, shadowRayEpsilon), tfar, time, rayDepth);
            // volume transmittance from P to light intersection
            scene_rdl2::math::Color volumeTransmittance = transmittance(pbrTls, trRay, sp.mPixel, sp.mSubpixelIndex, sequenceID, light);
            contribution *= (1.0f - presence) * volumeTransmittance;
            // Sum things up
            radiance += contribution;
            // LPE
            if (aovs) {
                EXCL_ACCUMULATOR_PROFILE(pbrTls, EXCL_ACCUM_AOVS);
                const FrameState &fs = *pbrTls->mFs;
                const LightAovs &lightAovs = *fs.mLightAovs;
                const AovSchema &aovSchema = *fs.mAovSchema;
                // transition
                int lpeStateId = pv.lpeStateId;
                lpeStateId = lightAovs.subsurfaceEventTransition(pbrTls, lpeStateId, bsdf);
                // intentionally ignore the the internal lambert
                // lobe event (see MOONRAY-1957).
                lpeStateId = lightAovs.lightEventTransition(pbrTls, lpeStateId, light);
                // accumulate matching aovs
                aovAccumLightAovs(pbrTls, aovSchema, lightAovs, contribution, nullptr, 
                                  AovSchema::sLpePrefixNone, lpeStateId, aovs);
                aovAccumVisibilityAovs(pbrTls, aovSchema, lightAovs,
                    scene_rdl2::math::Vec2f((1.0f - presence) * reduceTransparency(volumeTransmittance), 1.0f),
                    lpeStateId, aovs);
            }
        }
    }

    // Sample from the lobe, sharing the sample for direct and indirect estimator
    IntegratorSample2D bsdfSamples;
    IntegratorSample2D bsdfLightFilterSamples;
    IntegratorSample3D bsdfLightFilterSamples3D;
    // Sampling WIP
    // We want one shared sequence for depth 0 across subpixels and
    // local sub-surface samples
    // We want one shared sequence for depth 0 across subpixels and
    // global sub-surface samples
    // TODO: We still get weird visible bias over progressive passes
    const SequenceIDIntegrator bsdfSid(
        pv.nonMirrorDepth,
        sp.mPixel,
        0,
        (isLocal) ? SequenceType::BssrdfLocalBsdf : SequenceType::BssrdfGlobalBsdf,
        (pv.nonMirrorDepth == 0) ? 0 : sp.mSubpixelIndex * subsurfaceSplitFactor + subsurfaceIndex,
        sssSampleID);

    // Using SSS / SP / Bsdf
    // Note: using SP / SSS / Bsdf caused more correlation
    const int samplesSoFar = (pv.nonMirrorDepth == 0) ?
        (subsurfaceIndex * sp.mPixelSamples +
         sp.mSubpixelIndex) * computeRadianceSplitFactor : 0;

    bsdfSamples.resume(bsdfSid, samplesSoFar);

    if (lightFilterNeedsSamples) {
        const SequenceIDIntegrator bsdfLightFilterSid(
            pv.nonMirrorDepth,
            sp.mPixel,
            0,
            SequenceType::LightFilter,
            (pv.nonMirrorDepth == 0) ? 0 : sp.mSubpixelIndex * subsurfaceSplitFactor + subsurfaceIndex,
            sssSampleID);
        const SequenceIDIntegrator bsdfLightFilter3DSid(
            pv.nonMirrorDepth,
            sp.mPixel,
            0,
            SequenceType::LightFilter3D,
            (pv.nonMirrorDepth == 0) ? 0 : sp.mSubpixelIndex * subsurfaceSplitFactor + subsurfaceIndex,
            sssSampleID);

        bsdfLightFilterSamples.resume(bsdfLightFilterSid, samplesSoFar);
        bsdfLightFilterSamples3D.resume(bsdfLightFilter3DSid, samplesSoFar);
    }

    // Set up sample sequence for stochastic light choice (overlapping lights)
    SequenceIDIntegrator lightChoiceSid(pv.nonMirrorDepth, sp.mPixel,
        SequenceType::IndexSelection,
        sp.mSubpixelIndex * subsurfaceSplitFactor + subsurfaceIndex,
        sssSampleID);
    IntegratorSample1D lightChoiceSamples(lightChoiceSid);

    // Prepare our next potential path vertex
    PathVertex nextPv;
    nextPv.pathPixelWeight = pv.pathPixelWeight;
    nextPv.aovPathPixelWeight = pv.pathPixelWeight;
    nextPv.pathDistance = pv.pathDistance + parentRay.getEnd();
    nextPv.minRoughness = computeMinRoughness(lobe, mRoughnessClampingFactor,
        pv.minRoughness);
    nextPv.diffuseDepth = pv.diffuseDepth + 1;
    nextPv.subsurfaceDepth = pv.subsurfaceDepth;
    nextPv.glossyDepth = pv.glossyDepth;
    nextPv.mirrorDepth = pv.mirrorDepth;
    nextPv.nonMirrorDepth = pv.nonMirrorDepth + 1;
    nextPv.presenceDepth = pv.presenceDepth;
    nextPv.totalPresence = pv.totalPresence;
    nextPv.volumeDepth = pv.volumeDepth;
    nextPv.hairDepth = pv.hairDepth;
    nextPv.lobeType = lobe.getType();

    // LPE
    if (aovs) {
        EXCL_ACCUMULATOR_PROFILE(pbrTls, EXCL_ACCUM_AOVS);
        const FrameState &fs = *pbrTls->mFs;
        const LightAovs &lightAovs = *fs.mLightAovs;
        // transition
        int lpeStateId = pv.lpeStateId;
        lpeStateId = lightAovs.subsurfaceEventTransition(pbrTls, lpeStateId, bsdf);
        // intentionally ignore the the internal lambert
        // lobe event (see MOONRAY-1957).
        nextPv.lpeStateId = lpeStateId;
        nextPv.lpeStateIdLight = -1;
    }

    stats.addToCounter(STATS_BSDF_SAMPLES, computeRadianceSplitFactor);
    for (int sampleIndex = 0; sampleIndex < computeRadianceSplitFactor; sampleIndex++) {
        // Draw a sample
        float bsdfSample[2];
        bsdfSamples.getSample(bsdfSample, pv.nonMirrorDepth);
        LightFilterRandomValues bsdfLightFilterSample = {
                            scene_rdl2::math::Vec2f(0.f, 0.f), 
                            scene_rdl2::math::Vec3f(0.f, 0.f, 0.f)};
        if (lightFilterNeedsSamples) {
            bsdfLightFilterSamples.getSample(&bsdfLightFilterSample.r2[0], pv.nonMirrorDepth);
            bsdfLightFilterSamples3D.getSample(&bsdfLightFilterSample.r3[0], pv.nonMirrorDepth);
        }

        // Warp the sample and evaluate lobe pdf. Note we only use the lobe for
        // sampling to apply BIS/MIS, not for its contribution.
        // Note: We don't want a geometric surface check like we do when
        // evaluating bsdf because the subsurface transport makes wo irrelevant
        scene_rdl2::math::Vec3f wi;
        float bsdfPdf;
        scene_rdl2::math::Color f = lobe.sample(slice, bsdfSample[0], bsdfSample[1], wi, bsdfPdf);
        if (isSampleInvalid(f, bsdfPdf)) {
            continue;
        }
        // Check orientation with surface normal. This is a bit unneccessary
        // due to the call to isSampleInvalid() above.
        float cosWi = dot(N, wi);
        if (cosWi < scene_rdl2::math::sEpsilon) {
            continue;
        }
        int numLightsHit = 0;
        LightIntersection lightIsect;
        int hitIdx = lightSet.intersect(P, &N, wi, parentRay.getTime(),
            scene_rdl2::math::sMaxValue, false, lightChoiceSamples, pv.nonMirrorDepth,
            scene_rdl2::rdl2::DIFFUSE_REFLECTION, lightIsect, numLightsHit);
        const Light* hitLight = hitIdx >= 0 ? lightSet.getLight(hitIdx) : nullptr;
        const LightFilterList* hitLightFilterList = hitIdx >= 0 ? lightSet.getLightFilterList(hitIdx) : nullptr;
        bool doDirect = (numLightsHit > 0);
        // Evaluate the light contribution and pdf
        float lightPdf = 0.0f;
        scene_rdl2::math::Color Li = scene_rdl2::math::sBlack;
        if (doDirect) {
            // We multiply by numLightsHit because we're stochastically sampling
            // just one, thus computing an average
            Li = hitLight->eval(pbrTls->mTopLevelTls, wi, P, bsdfLightFilterSample,
                parentRay.getTime(), lightIsect, false, hitLightFilterList, parentRay.getDirFootprint(), &lightPdf) *
                (float)numLightsHit;
            if (isSampleInvalid(Li, lightPdf)) {
                doDirect = false;
            }
        }
        // Compute incoming fresnel transmission
        scene_rdl2::math::Color fresnel = evalFresnel(transmissionFresnel, N, wi);
        // TODO: Apply russian roulette
        // the volume attenuation along this ray to the first hit (or infinity)

        VolumeTransmittance vt;
        vt.reset();

        float presence = 0.0f;
        // Compute indirect illumination if needed, trace a shadow ray if not
        if (doIndirect) {
            // Apply BIS
            nextPv.pathThroughput = pt * fresnel * f / bsdfPdf;
            // Build our next ray differential to see if there is an indirect
            // lighting contribution closer than the intersected light, if any.
            // TODO: We had some self-intersections when rays leave at grazing
            // angle, so we adjust the rayEpsilon accordingly
            const float denom = scene_rdl2::math::abs(dot(slice.getNg(), wi));
            // slice.getNg() itself or the dot product above can be zero.
            const float start = scene_rdl2::math::isZero(denom) ? rayEpsilon : rayEpsilon / denom;
            RayDifferential ray(
                    parentRay, start,
                    lightIsect.distance * sHitEpsilonEnd);
            scatterAndScale(dNdx, dNdy,
                            lobe,
                            -parentRay.getDirection(),
                            wi,
                            1.0f,
                            bsdfSample[0],
                            bsdfSample[1],
                            ray);

            // Recurse
            scene_rdl2::math::Color contribution;
            float transparency;
            ++sequenceID;
            bool hitVolume;

            IndirectRadianceType indirectRadianceType = computeRadianceRecurse(
                    pbrTls, ray, sp, nextPv, &lobe,
                    contribution, transparency, vt,
                    sequenceID, aovs, nullptr, nullptr, nullptr, nullptr, false, hitVolume);
            if (indirectRadianceType != NONE) {
                // Accumulate radiance, but only accumulate indirect or direct
                // contribution
                radiance += contribution;
            }
        }
        //------------------------------
        // Compute direct lighting contribution with MIS if needed.
        if (doDirect) {
            float tfar = lightIsect.distance * sHitEpsilonEnd;
            float time = parentRay.getTime();
            int rayDepth = parentRay.getDepth() + 1;
            const FrameState &fs = *pbrTls->mFs;
            Ray shadowRay(P, wi, rayEpsilon, tfar, time, rayDepth);
            const bool hasUnoccludedFlag = fs.mAovSchema->hasLpePrefixFlags(AovSchema::sLpePrefixUnoccluded);
            int32_t assignmentId = isect.getLayerAssignmentId();
            if (isRayOccluded(pbrTls, hitLight, shadowRay, rayEpsilon, shadowRayEpsilon, presence, assignmentId)) {
                // LPE
                if (aovs) {
                    EXCL_ACCUMULATOR_PROFILE(pbrTls, EXCL_ACCUM_AOVS);
                    const AovSchema &aovSchema = *fs.mAovSchema;
                    const LightAovs &lightAovs = *fs.mLightAovs;
                    // transition
                    int lpeStateId = pv.lpeStateId;
                    lpeStateId = lightAovs.subsurfaceEventTransition(pbrTls,
                        lpeStateId, bsdf);
                    // intentionally ignore the the internal lambert
                    // lobe event (see MOONRAY-1957).
                    lpeStateId = lightAovs.lightEventTransition(pbrTls,
                        lpeStateId, hitLight);
                    // accumulate matching aovs
                    aovAccumVisibilityAovs(pbrTls, aovSchema, lightAovs,
                        scene_rdl2::math::Vec2f(0.0f, 1.0f), lpeStateId, aovs);

                    // unoccluded prefix LPEs
                    if (hasUnoccludedFlag) {
                        // If it's occluded but we have the unoccluded flag set, only contribute this to any 
                        // pre-occlusion aovs.
                        scene_rdl2::math::Color contribution = pt * Li * fresnel * f / bsdfPdf * powerHeuristic(bsdfPdf, lightPdf);
                        aovAccumLightAovs(pbrTls, aovSchema, lightAovs, contribution, nullptr,
                                          AovSchema::sLpePrefixUnoccluded, lpeStateId, aovs);
                    }
                }
                continue;
            }
            // shadowRay can be modified in occlusion query
            Ray trRay(P, wi, scene_rdl2::math::max(rayEpsilon, shadowRayEpsilon), tfar, time, rayDepth);
            vt.mTransmittanceE = transmittance(pbrTls, trRay, sp.mPixel, sp.mSubpixelIndex, sequenceID, hitLight);
            // Apply MIS power heuristic
            scene_rdl2::math::Color contribution = pt * Li * fresnel * f / bsdfPdf *
                powerHeuristic(bsdfPdf, lightPdf);
            const scene_rdl2::math::Color unoccludedContribution = contribution;
            contribution *= (1.0f - presence) * vt.mTransmittanceE;
            // Sum things up
            radiance += contribution;
            // LPE
            if (aovs) {
                EXCL_ACCUMULATOR_PROFILE(pbrTls, EXCL_ACCUM_AOVS);
                const FrameState &fs = *pbrTls->mFs;
                const AovSchema &aovSchema = *fs.mAovSchema;
                const LightAovs &lightAovs = *fs.mLightAovs;
                // transition
                int lpeStateId = nextPv.lpeStateId;
                lpeStateId = lightAovs.lightEventTransition(pbrTls, lpeStateId, hitLight);
                // accumulate matching aovs
                aovAccumVisibilityAovs(pbrTls, aovSchema, lightAovs,
                    scene_rdl2::math::Vec2f((1-presence) * reduceTransparency(vt.mTransmittanceE), 1.0f),
                    lpeStateId, aovs);

                // Accumulate aovs depending on whether or not the unoccluded flag is set.
                if (hasUnoccludedFlag) {
                    // If the unoccluded flag is set we have to add occluded and unoccluded 
                    // (without presence and volume transmittance) separately.
                    aovAccumLightAovs(pbrTls, aovSchema, lightAovs, unoccludedContribution, &contribution, 
                                      AovSchema::sLpePrefixUnoccluded, lpeStateId, aovs);
                } else {
                    // Otherwise, just add the contribution to all non-pre-occlusion aovs.
                    aovAccumLightAovs(pbrTls, aovSchema, lightAovs, contribution, nullptr, 
                                      AovSchema::sLpePrefixNone, lpeStateId, aovs);
                }
            }
            // Raydb debug
            if (lightIsect.distance < sDistantLightDistance) {
                Ray debugRay(P, wi, 0.0f);
                RAYDB_EXTEND_RAY_NO_HIT(pbrTls, debugRay, lightIsect.distance);
                RAYDB_SET_CONTRIBUTION(pbrTls, Li);
                RAYDB_ADD_TAGS(pbrTls, TAG_AREALIGHT);
            } else {
                Ray debugRay(P, wi, 0.0f);
                RAYDB_EXTEND_RAY_NO_HIT(pbrTls, debugRay, 40.0f);
                RAYDB_SET_CONTRIBUTION(pbrTls, Li);
                RAYDB_ADD_TAGS(pbrTls, TAG_ENVLIGHT);
            }
        }
    }
    return radiance;
}

scene_rdl2::math::Color
PathIntegrator::computeRadianceDiffusionSubsurface(pbr::TLState *pbrTls,
        const Bsdf &bsdf, const Subpixel &sp, const PathVertex &pv,
        const RayDifferential &ray, const Intersection &isect,
        const BsdfSlice &slice, const Bssrdf &bssrdf,
        const LightSet &lightSet, bool doIndirect,
        float rayEpsilon, float shadowRayEpsilon,
        unsigned &sequenceID, scene_rdl2::math::Color &ssAov, float *aovs) const
{
    scene_rdl2::math::Color radiance = scene_rdl2::math::sBlack;
    if (mBssrdfSamples < 1 || !mEnableSSS) {
        return radiance;
    }

    EXCL_ACCUMULATOR_PROFILE(pbrTls, EXCL_ACCUM_SSS_INTEGRATION);

    // TODO: We need to use a "lobe type" that tells computeRadianceRecurse() and
    // the rest of the system that this is a bssrdf "lobe type" (used for ray
    // masking and ray debugging)
    const shading::BsdfLobe::Type lobeType =
        shading::BsdfLobe::Type(shading::BsdfLobe::REFLECTION | shading::BsdfLobe::DIFFUSE);
    Statistics &stats = pbrTls->mStatistics;

    const Scene *scene = MNRY_VERIFY(pbrTls->mFs->mScene);

    // Search limits
    float maxRadius = bssrdf.getMaxRadius();
    float minRadius = maxRadius * 0.1f;

    // Where are we ?
    const scene_rdl2::math::Vec3f P = isect.getP();
    const scene_rdl2::math::Vec3f N = isect.getN();
    const scene_rdl2::math::Vec3f wo = -(ray.getDirection());
    // Local Reference Frame
    scene_rdl2::math::ReferenceFrame localF(N);

    // The scale * outgoing fresnel
    const Fresnel *transmissionFresnel = bssrdf.getTransmissionFresnel();
    const scene_rdl2::math::Color scaleFresnelWo = evalFresnel(transmissionFresnel, N, wo) *
        bssrdf.getScale();

    // We disable the transmission fresnel on the wi side.
    // If the material is using a OneMinusRoughSchlickFresnel, this matches the
    // energy conservation we are doing with other types of "under" lobes.
    // Otherise it darkens the SSS somewhat a bit when the specular factor of
    // the "top" lobe is in the 0.2 .. 0.7 range.
    transmissionFresnel = nullptr;

    //---------------------------------------------------------------------
    // TODO: We ignore single scattering for now since its relative contribution
    // is small. We'll implement this in the future, its relatively very easy
    // compared to multiple scattering below.

    // Check if we can substitute subsurface with a cheaper approximation
    if (canSubstituteSubsurface(maxRadius, isect)) {

        shading::BsdfSlice sliceLocal(isect.getNg(), slice.getWo(), true, true, slice.getShadowTerminatorFix());
        LambertBsdfLobe lobeLocal(N, scene_rdl2::math::sWhite, true);

        // Not doing any subsurface computations
        int subsurfaceSplitFactor = 1;
        // But split to compute Radiance
        int computeRadianceSplitFactor = (pv.nonMirrorDepth > 0)  ?
            1  :  mBsdfSamples;

        /// SSS throughput
        scene_rdl2::math::Color throughput = pv.pathThroughput * bssrdf.diffuseReflectance() *
            scaleFresnelWo;

        // Lambertian Reflection instead of SSS
        radiance += computeRadianceSubsurfaceSample(
            pbrTls, bsdf, sp, pv,
            ray, isect.getdNdx(), isect.getdNdy(), throughput,
            transmissionFresnel, lightSet, lobeLocal, sliceLocal, P, N,
            subsurfaceSplitFactor,           //subsurfaceSplitFactor
            computeRadianceSplitFactor,      //regular split factor
            0,                               //split index
            doIndirect, rayEpsilon, shadowRayEpsilon, sequenceID, sequenceID, true, aovs, isect);


        // We cannot approximate the forward scattering with a
        // Lambertian lobe, so evaluate this explicitly.
        radiance += computeDiffusionForwardScattering(pbrTls, bsdf, sp, pv,
            ray, isect, slice, transmissionFresnel, scaleFresnelWo, lightSet,
            bssrdf, P, N, localF, subsurfaceSplitFactor, doIndirect,
            rayEpsilon, shadowRayEpsilon, sequenceID, sequenceID, ssAov, aovs);

        return radiance;
    }


    // Compute Full Subsurface

    //---------------------------------------------------------------------
    // Compute local multiple scattering, as in paper [2]
    bool dontSplit = (pv.nonMirrorDepth > 0);

    // Make sure we do all local sss sampling with input sequenceID, which will
    // get incremented in the process.
    unsigned sssSampleID = sequenceID;

    // Triplanar projection sampling needs at least 2 samples.
    int subsurfaceSplitFactor = (dontSplit ? 2 : mBssrdfSamples);

    scene_rdl2::math::Color ptSubsurface = pv.pathThroughput / subsurfaceSplitFactor;

    // Setup Bssrdf local samples.
    // We want one shared sequence for non-mirror-depth 0 across sub-pixels
    IntegratorSample2D bssrdfLocalSamples;
    const SequenceIDBssrdf bssrdfLocalSid(
        pv.nonMirrorDepth, sp.mPixel, SequenceType::BssrdfLocal,
        pv.nonMirrorDepth == 0 ? 0 : sp.mSubpixelIndex, sssSampleID);
    bssrdfLocalSamples.resume(bssrdfLocalSid,
        pv.nonMirrorDepth == 0 ? sp.mSubpixelIndex * subsurfaceSplitFactor : 0);

    stats.addToCounter(STATS_SSS_SAMPLES, subsurfaceSplitFactor);

    // We measure the diffuse reflectance through sampling to
    // calculate the area compensation factor based on how much we
    // deviate from the semi-infinite plane assumption
    scene_rdl2::math::Color measuredDiffuseReflectance = scene_rdl2::math::sBlack;

    // Collect a list of sample points to evaluate the subsurface integral
    struct SubsurfaceSample {
        scene_rdl2::math::Vec3f mP, mN, mNg;
        scene_rdl2::math::Vec3f mdNdx, mdNdy;
        scene_rdl2::math::Color mBssrdfEval;
    };
    std::vector<SubsurfaceSample> subsurfaceSamples;

    // Bssrdf input normal
    const scene_rdl2::rdl2::Material* sssMaterial = bssrdf.getMaterial();
    const scene_rdl2::rdl2::EvalNormalFunc evalSubsurfaceNormal = bssrdf.getEvalNormalFunc();
    bool validInputNormal = sssMaterial && evalSubsurfaceNormal;

    for (int sampleIndex = 0; sampleIndex < subsurfaceSplitFactor; sampleIndex++) {

        // Draw a low discrepancy sample
        float sample[2];
        bssrdfLocalSamples.getSample(sample, pv.nonMirrorDepth);

        // Step 1 - Select a projection axis & remap the random number to be
        // in [0,1) Axis of Projection
        scene_rdl2::math::Vec3f directionProj;
        int axisIndex = bssrdfSelectAxisAndRemapSample(
            localF, sample[0], directionProj);

        // Sample position according to Bssrdf
        scene_rdl2::math::Vec3f PiTangent;
        float r;
        float pdf = bssrdf.sampleLocal(sample[0], sample[1], PiTangent, r);
        if (scene_rdl2::math::isZero(pdf, 0.0f)) { // exactly zero
            continue;
        }

        PiTangent = P + bssrdfOffsetLocalToGlobal(localF, axisIndex, PiTangent);

        // We need to limit the distance to search for nearby surfaces to
        // project onto. Otherwise we might go through and get blocked by
        // other nearby objects.
        float search = scene_rdl2::math::min(r, maxRadius);
        search = scene_rdl2::math::max(search, minRadius);
        scene_rdl2::math::Vec3f originProj = PiTangent - directionProj * search;

        shading::Intersection isectProj;
        Ray rayProj(originProj,
                    directionProj,
                    0.0f,
                    2.0f * search,
                    ray.getTime(),
                    ray.getDepth() + 1);

        // get trace set for sss
        auto geomTls = pbrTls->mTopLevelTls->mGeomTls.get();
        geomTls->mSubsurfaceTraceSet = bssrdf.getTraceSet();
        rayProj.ext.geomTls = (void*)geomTls;

        // set default material for trace set;
        const scene_rdl2::rdl2::Material* isectMaterial = isect.getMaterial();
        if (isectMaterial) {
            const shading::Material *materialExt =
                &isectMaterial->get<const shading::Material>();
            rayProj.ext.materialID = materialExt->getMaterialId();
        }

        // TODO: We need position and normal only in the intersection
        bool intersected = scene->intersectRay(pbrTls->mTopLevelTls, rayProj,
            isectProj, lobeType);

        const scene_rdl2::rdl2::Material* isectProjMaterial = isectProj.getMaterial();

        if (!intersected || isectProj.getMaterial() == nullptr) {
            continue;
        }
        geom::initIntersectionPhase2(isectProj,
                                     pbrTls->mTopLevelTls,
                                     pv.mirrorDepth,
                                     pv.glossyDepth,
                                     pv.diffuseDepth,
                                     isSubsurfaceAllowed(pv.subsurfaceDepth),
                                     pv.minRoughness,
                                     -rayProj.getDirection());
        RayDifferential rayDiff(ray, isectProj.getEpsilonHint(), rayProj.tfar);
        isectProj.transferAndComputeDerivatives(pbrTls->mTopLevelTls, &rayDiff,
            sp.mTextureDiffScale);

        // Keep track of projected position and normal
        scene_rdl2::math::Vec3f PiProj = isectProj.getP();
        scene_rdl2::math::Vec3f NiProj = isectProj.getN();

        scene_rdl2::math::Vec3f NiProjMap = NiProj;
        // The material of the intersected point must match the
        // isectMaterial to evaluate the normal map. This ensures that
        // the intersection state has all the attributes requested by the
        // normal map. There may not be a match when using trace sets.
        // In case of a mismatch, we do not attempt to evaluate the subsurface normal map.
        if (validInputNormal && isectProjMaterial == isectMaterial) {
            // We evaluate using the sssMaterial, which does not always match the isectMaterial.
            // Example: sssMaterial is undermaterial of isectMaterial, which is a GlitterFlakeMaterial.
            NiProjMap = evalSubsurfaceNormal(sssMaterial,
                                             pbrTls->mTopLevelTls->mShadingTls.get(),
                                             shading::State(&isectProj));
        }

        // Only count surface hits facing the direction of projection.
        // This avoids double contribution across local / global scattering
        const float cosTheta = -dot(NiProj, directionProj);
        if (cosTheta < scene_rdl2::math::sEpsilon) {
            continue;
        }

        RAYDB_EXTEND_RAY(pbrTls, rayProj, isectProj);
        RAYDB_SET_TAGS(pbrTls, lobeType);

        // Update r from projected position and make sure the projected
        // position is still within maxRadius, otherwise this may clash
        // with light culling (see where the lightSet is computed).
        scene_rdl2::math::Vec3f dp = P - PiProj;
        r = dp.length();
        if (r > maxRadius) {
            continue;
        }
        // MIS One-Sample Projection Weight
        float misWeight = bssrdfGetMISAxisWeight(localF, NiProj, dp, bssrdf);
        scene_rdl2::math::Color bssrdfEval = bssrdf.eval(r) * misWeight;

        // Calculate the BSSRDF DiffuseReflectance Integral via Sampling
        measuredDiffuseReflectance += bssrdfEval;

        // Subsurface Integral
        scene_rdl2::math::Color pt = ptSubsurface * bssrdfEval;

        // Apply the out-point transmission fresnel
        pt *= scaleFresnelWo;

        // used for subsurface material aovs
        ssAov += pt;

        // Collect this sample
        SubsurfaceSample s;
        s.mP             = PiProj;
        s.mN             = NiProjMap;
        s.mNg            = isectProj.getNg();
        s.mdNdx          = isectProj.getdNdx();
        s.mdNdy          = isectProj.getdNdy();
        s.mBssrdfEval    = pt;
        subsurfaceSamples.push_back(std::move(s));
    }

    // DiffuseReflectance Integral
    measuredDiffuseReflectance /= subsurfaceSplitFactor;
    // Area Compensation Term
    const scene_rdl2::math::Color areaCompensationFactor = bssrdfAreaCompensation(
        measuredDiffuseReflectance, bssrdf.diffuseReflectance());
    ssAov *= areaCompensationFactor;

    //For each sample, evaluate irradiance and compute the subsurface integral
    // TODO: For now, we don't want to split here at all
    #if 0
        int computeRadianceSplitFactor = pv.nonMirrorDepth == 0 ? mDirectSamples:1;
    #else
        int computeRadianceSplitFactor = 1;
    #endif

    for (size_t i = 0; i < subsurfaceSamples.size(); ++i) {
        // Compute the direct irradiance for this sample
        // TODO: compute woiProj
        shading::BsdfSlice sliceLocal(subsurfaceSamples[i].mNg, slice.getWo(), true, true,
            slice.getShadowTerminatorFix());
        LambertBsdfLobe lobeLocal(subsurfaceSamples[i].mN, scene_rdl2::math::sWhite, true);

        // TODO: Need to pass rayProj and updated pv instead of ray, pv
        radiance += computeRadianceSubsurfaceSample(
            pbrTls, bsdf, sp, pv, ray,
            subsurfaceSamples[i].mdNdx, subsurfaceSamples[i].mdNdy,
            subsurfaceSamples[i].mBssrdfEval * areaCompensationFactor,
            transmissionFresnel, lightSet, lobeLocal, sliceLocal,
            subsurfaceSamples[i].mP, subsurfaceSamples[i].mN,
            subsurfaceSplitFactor, computeRadianceSplitFactor, i,
            doIndirect, rayEpsilon, shadowRayEpsilon, sssSampleID,
            sequenceID, true, aovs, isect);
        RAYDB_SET_CONTRIBUTION(pbrTls, radiance / pv.pathThroughput);
    }

    /// Back-scattering Term
    radiance += computeDiffusionForwardScattering(pbrTls, bsdf, sp, pv, ray,
        isect, slice, transmissionFresnel, scaleFresnelWo, lightSet, bssrdf,
        P, N, localF, subsurfaceSplitFactor, doIndirect, rayEpsilon, shadowRayEpsilon,
        sssSampleID, sequenceID, ssAov, aovs);

    return radiance;
}

scene_rdl2::math::Color
PathIntegrator::computeDiffusionForwardScattering(pbr::TLState *pbrTls, const Bsdf &bsdf,
        const Subpixel &sp, const PathVertex &pv,
        const RayDifferential &ray, const Intersection &isect,
        const BsdfSlice &slice, const Fresnel *transmissionFresnel,
        const scene_rdl2::math::Color& scaleFresnelWo, const LightSet &lightSet,
        const Bssrdf &bssrdf, const scene_rdl2::math::Vec3f &P, const scene_rdl2::math::Vec3f &N,
        const scene_rdl2::math::ReferenceFrame &localF, int subsurfaceSplitFactor,
        bool doIndirect, float rayEpsilon, float shadowRayEpsilon,
        unsigned sssSampleID, unsigned &sequenceID, scene_rdl2::math::Color &ssAov, float *aovs) const
{
    scene_rdl2::math::Color radiance = scene_rdl2::math::sBlack;
    const Scene *scene = MNRY_VERIFY(pbrTls->mFs->mScene);
    // Bssrdf input normal
    const scene_rdl2::rdl2::Material* sssMaterial = bssrdf.getMaterial();
    const scene_rdl2::rdl2::EvalNormalFunc evalSubsurfaceNormal = bssrdf.getEvalNormalFunc();
    bool validInputNormal = sssMaterial && evalSubsurfaceNormal;

    scene_rdl2::math::Color ptSubsurface = pv.pathThroughput / subsurfaceSplitFactor;

    // Compute global multiple scattering (back of the ear, leaf, etc.)
    /// Ref - "Efficient rendering of human skin", D'Eon et al., EGSR 2007
    // Setup Bssrdf global samples.
    // We want one shared sequence for non-mirror-depth 0 across sub-pixels
    IntegratorSample2D bssrdfGlobalSamples;

    Statistics &stats = pbrTls->mStatistics;
    float maxRadius = bssrdf.getMaxRadius();
    // TODO: We need to use a "lobe type" that tells computeRadianceRecurse() and
    // the rest of the system that this is a bssrdf "lobe type" (used for ray
    // masking and ray debugging)
    const BsdfLobe::Type lobeType =
        BsdfLobe::Type(BsdfLobe::REFLECTION | BsdfLobe::DIFFUSE);

    const SequenceIDBssrdf bssrdfGlobalSid(pv.nonMirrorDepth, sp.mPixel,
        SequenceType::BssrdfGlobal,
        (pv.nonMirrorDepth == 0 ? 0 : sp.mSubpixelIndex), sssSampleID);
    bssrdfGlobalSamples.resume(bssrdfGlobalSid,
        (pv.nonMirrorDepth == 0 ? sp.mSubpixelIndex * subsurfaceSplitFactor : 0));

    // Prepare a path vertex
    PathVertex nextPv;
    nextPv.pathThroughput  = scene_rdl2::math::sWhite;     // Unused
    nextPv.pathPixelWeight = 0.0f;
    // Use previous path pixel weight for aovPathPixelWeight as there's existing logic
    // in vector mode that sometimes assumes that pv.pathPixelWeight = 0.  Thus, we must seperately
    // keep track of the pathPixelWeight for aovs.  See comment in PathIntegratorMultiSampler.ispc::
    // addIndirectOrDirectVisibleContributionsBundled().
    nextPv.aovPathPixelWeight = pv.pathPixelWeight;
    nextPv.pathDistance    = pv.pathDistance + ray.getEnd();
    nextPv.minRoughness    = computeMinRoughness(
        1.0f, mRoughnessClampingFactor, pv.minRoughness);
    nextPv.diffuseDepth    = pv.diffuseDepth + 1;
    nextPv.subsurfaceDepth = pv.subsurfaceDepth;
    nextPv.glossyDepth     = pv.glossyDepth;
    nextPv.mirrorDepth     = pv.mirrorDepth;
    nextPv.nonMirrorDepth  = pv.nonMirrorDepth + 1;
    nextPv.presenceDepth   = pv.presenceDepth;
    nextPv.totalPresence   = pv.totalPresence;
    nextPv.volumeDepth     = pv.volumeDepth;
    nextPv.hairDepth       = pv.hairDepth;

    // LPE: not sure how to characterize this event
    if (aovs) {
        nextPv.lpeStateId = pv.lpeStateId;
        nextPv.lpeStateIdLight = -1;
    }

    stats.addToCounter(STATS_SSS_SAMPLES, subsurfaceSplitFactor);

    // Max Spherical Cap
    static const float cosThetaMax = scene_rdl2::math::cos(85.0f / 180.0f * scene_rdl2::math::sPi);

    for (int sampleIndex = 0; sampleIndex < subsurfaceSplitFactor; sampleIndex++) {
        // Draw a low discrepancy sample
        float sample[2];
        bssrdfGlobalSamples.getSample(sample, pv.nonMirrorDepth);
        // Use the bssrdf to sample a direction into the subsurface
        scene_rdl2::math::Vec3f direction;
        float pdf = bssrdf.sampleGlobal(sample[0], sample[1],
            cosThetaMax, direction);
        direction = -localF.localToGlobal(direction);
        // Trace to search a nearby back-facing surface
        Intersection isectBack;
        Ray rayBack(P, direction, rayEpsilon, maxRadius,
            ray.getTime(), ray.getDepth() + 1);
        // get trace set for sss
        geom::internal::TLState* geomTls = pbrTls->mTopLevelTls->mGeomTls.get();
        geomTls->mSubsurfaceTraceSet = bssrdf.getTraceSet();
        rayBack.ext.geomTls = geomTls;
        // get default material for trace set
        const scene_rdl2::rdl2::Material* isectMaterial = isect.getMaterial();
        if (isectMaterial) {
            const shading::Material *materialExt =
                &isectMaterial->get<const shading::Material>();
            rayBack.ext.materialID = materialExt->getMaterialId();
        }
        // TODO: We need position and normal only in the intersection
        if (!scene->intersectRay(pbrTls->mTopLevelTls, rayBack, isectBack, lobeType) ||
            isectBack.getMaterial() == nullptr) {
            continue;
        }
        geom::initIntersectionPhase2(isectBack,
                                     pbrTls->mTopLevelTls,
                                     pv.mirrorDepth,
                                     pv.glossyDepth,
                                     pv.diffuseDepth,
                                     isSubsurfaceAllowed(pv.subsurfaceDepth),
                                     pv.minRoughness,
                                     -rayBack.getDirection());
        RayDifferential rayDiff(ray, isectBack.getEpsilonHint(), rayBack.tfar);
        isectBack.transferAndComputeDerivatives(pbrTls->mTopLevelTls, &rayDiff,
            sp.mTextureDiffScale);

        // TODO: Should we use the un-flipped N ?
        scene_rdl2::math::Vec3f Ni = -isectBack.getN();
        scene_rdl2::math::Vec3f Pi = isectBack.getP();
        float r = rayBack.getEnd();

        scene_rdl2::math::Vec3f NiMap = Ni;
        // The material of the intersected point must match the
        // isectMaterial to evaluate the normal map. This ensures that
        // the intersection state has all the attributes requested by the
        // normal map. There may not be a match when using trace sets.
        // In case of a mismatch, we do not attempt to evaluate the subsurface normal map.
        if (validInputNormal && isectBack.getMaterial() == isectMaterial) {
            // We evaluate using the sssMaterial, which does not always match the isectMaterial.
            // Example: sssMaterial is undermaterial of isectMaterial, which is a GlitterFlakeMaterial.
            NiMap = -evalSubsurfaceNormal(sssMaterial,
                                          pbrTls->mTopLevelTls->mShadingTls.get(),
                                          shading::State(&isectBack));
        }
        // Only count surface hits facing away from the direction of projection
        // that we used for local scattering.
        // This avoids double contribution across local / global scattering
        float cosTheta = -dot(Ni, N);
        if (cosTheta < scene_rdl2::math::sEpsilon) {
            continue;
        }

        RAYDB_EXTEND_RAY(pbrTls, rayBack, isectBack);
        RAYDB_SET_TAGS(pbrTls, lobeType);

        // We compute the jacobian to convert between this integral over solid
        // angle into the integral we want over surface area
        cosTheta = scene_rdl2::math::abs(dot(Ni, direction));
        float jacobian = r * r / scene_rdl2::math::max(cosTheta, 0.05f);

        // Account for Bssrdf
        scene_rdl2::math::Color pt = ptSubsurface * bssrdf.eval(r, true) * jacobian / pdf;
        // Apply the out-point transmission fresnel and 1/pi term
        pt *= scaleFresnelWo;
        // used for subsurface aov
        ssAov += pt;
        // TODO: For now, we don't want to split here at all
        const int computeRadianceSplitFactor = 1;
        // Compute the direct irradiance for this sample
        // TODO: compute woiBack
        shading::BsdfSlice sliceGlobal(-isectBack.getNg(), slice.getWo(), true, true,
            slice.getShadowTerminatorFix());
        LambertBsdfLobe lobeGlobal(NiMap, scene_rdl2::math::sWhite, true);
        // TODO: Need to pass rayBack and updated pv instead of ray, pv
        radiance += computeRadianceSubsurfaceSample(pbrTls, bsdf,
            sp, nextPv, ray, isectBack.getdNdx(), isectBack.getdNdy(),
            pt, transmissionFresnel, lightSet, lobeGlobal, sliceGlobal,
            Pi, NiMap, subsurfaceSplitFactor, computeRadianceSplitFactor,
            sampleIndex, doIndirect, rayEpsilon, shadowRayEpsilon,
            sssSampleID, sequenceID, false, aovs, isect);

        RAYDB_SET_CONTRIBUTION(pbrTls, radiance / pv.pathThroughput);
    }
    return radiance;
}

//-----------------------------------------------------------------------------

// utility function to sample distance t from 0 to inf with distribution
// propotional to exp(-sigmaT[channel] * t)
// channel is selected based on discrete pdf provided through channelWeights
// this is similar to the bsdf one sample idea that picking out a lobe first
// and then eval the total lobes contribution
// the idea is explored by multiple publication addressing chromatic volume
// coefficient importance sampling, see
// "Practical and Controllable Subsurface Scattering for Production Path Tracing"
// Matt Jen-Yuan Chiang el al
// 3. Sampling for detail reference
// pbrt3 also contains an implementation that selecting channel with uniform pdf
// http://www.pbr-book.org/3ed-2018/Light_Transport_II_Volume_Rendering/Sampling_Volume_Scattering.html#HomogeneousMedium
float
sampleDistanceZeroScatter(const scene_rdl2::math::Color& channelWeights, const scene_rdl2::math::Color& sigmaT,
        float uChannel, float uDistance, scene_rdl2::math::Color& tr, float& pdfDistance)
{
    float normalization =
        scene_rdl2::math::rcp(channelWeights[0] + channelWeights[1] + channelWeights[2]);
    scene_rdl2::math::Color pdfChannel = normalization * channelWeights;

    float distance;
    if (uChannel < pdfChannel[0]) {
        distance = -scene_rdl2::math::log(1.0f - uDistance) / sigmaT[0];
    } else if (uChannel < (pdfChannel[0] + pdfChannel[1])) {
        distance = -scene_rdl2::math::log(1.0f - uDistance) / sigmaT[1];
    } else {
        distance = -scene_rdl2::math::log(1.0f - uDistance) / sigmaT[2];
    }
    tr = scene_rdl2::math::exp(-sigmaT * distance);
    pdfDistance =
        pdfChannel[0] * sigmaT[0] * tr[0] +
        pdfChannel[1] * sigmaT[1] * tr[1] +
        pdfChannel[2] * sigmaT[2] * tr[2];
    return distance;
}

// Distance Sampling with either:
// (a) sampling transmittance or (b) Dwivedi distance sampling
// For Dwivedi Distance Sampling, ref "Improved Dwivedi Sampling"
// Return the resulting transmittance for that distance
// & the PDF for both methods for one-sample PDF computation
float
sampleDistanceMultiScatter(const scene_rdl2::math::Color& sigmaT, const scene_rdl2::math::Color& sigmaTPrime,
        const scene_rdl2::math::Color& channelSamplingWeights, int chIndex, float uDistance,
        float NdotDir, const scene_rdl2::math::Color& dwivediV0, bool choseDwivediSampling,
        scene_rdl2::math::Color& tr, scene_rdl2::math::Color& pdfDistance, scene_rdl2::math::Color& pdfDistanceDwivedi)
{
    float distance;
    if (choseDwivediSampling) {
        distance = -scene_rdl2::math::log(1.0f - uDistance) / sigmaTPrime[chIndex];
    } else {
        distance = -scene_rdl2::math::log(1.0f - uDistance) / sigmaT[chIndex];
    }
    tr = scene_rdl2::math::exp(-sigmaT * distance);
    pdfDistance = sigmaT*tr;
    pdfDistanceDwivedi = sigmaTPrime*exp(-sigmaTPrime*distance);
    return distance;
}

// One-sample between RGB using weights
// Returns the selected channel index
int
sampleChannel(const scene_rdl2::math::Color& channelWeights,
        float uChannel, scene_rdl2::math::Color& channelSamplingWeights)
{
    float normalization =
        scene_rdl2::math::rcp(channelWeights[0] + channelWeights[1] + channelWeights[2]);

    channelSamplingWeights = normalization * channelWeights;
    int chIndex;
    if (uChannel < channelSamplingWeights[0]) {
        chIndex = 0;
    } else if (uChannel < (channelSamplingWeights[0]+channelSamplingWeights[1])) {
        chIndex = 1;
    } else {
        chIndex = 2;
    }
    return chIndex;
}

// One-Sample Between PhaseFunction and Dwivedi Direction Sampling
// Returns a boolean for that choice, the outputcosine between chosen direction
// and the referenceFrame normal and the output direction
void
sampleDirection(const scene_rdl2::math::Vec2f& uDir,
                float backwardDirectionSample, int chIndex, float phaseSamplingWeight,
                float wBackwardSampling, const VolumePhase& sssPhaseFunction,
                const Ray& sssRay, const scene_rdl2::math::Color& dwivediV0,
                const scene_rdl2::math::ReferenceFrame& inReferenceFrame,
                bool& choseDwivediSampling, float& outputCosine,
                scene_rdl2::math::Vec3f& outputDir)
{
    if (uDir[0] < phaseSamplingWeight) {
        // choose isotropic sampling
        choseDwivediSampling = false;
        // re-normalize random number
        float r = uDir[0] / phaseSamplingWeight;

        outputDir = sssPhaseFunction.sample(-sssRay.dir,
                                            r,
                                            uDir[1]);
        // choose appropriate cosine for one-sample afterwards
        if (backwardDirectionSample < wBackwardSampling) {
            outputCosine = dot(outputDir, inReferenceFrame.getN());
        } else {
            // go 'forwards' along -N
            outputCosine = -dot(outputDir, inReferenceFrame.getN());
        }
    } else {
        // choose Dwivedi Sampling
        choseDwivediSampling = true;
        // re-normalize random number
        float r = (uDir[0] - phaseSamplingWeight) / (1.0f - phaseSamplingWeight);
        const float v0 = dwivediV0[chIndex];
        // Ref "Improved Dwivedi Sampling" Eqn 10
        outputCosine = v0 - (v0+1.0f)*scene_rdl2::math::pow((v0-1.0f)/(v0+1.0f), r);
        float sine = scene_rdl2::math::sqrt(scene_rdl2::math::max(0.0f, 1.0f - outputCosine*outputCosine));
        float phi = scene_rdl2::math::sTwoPi * uDir[1];
        float cosPhi, sinPhi;
        scene_rdl2::math::sincos(phi, &sinPhi, &cosPhi);
        scene_rdl2::math::Vec3f localDir(sine * cosPhi,
                             sine * sinPhi,
                             outputCosine);

        if (backwardDirectionSample < wBackwardSampling) {
            // go backwards along N
            outputDir = normalize(inReferenceFrame.localToGlobal(localDir));
        } else {
            // go 'forwards' along -N
            outputDir = -normalize(inReferenceFrame.localToGlobal(localDir));
        }
    }
}

// One Sample PDF from sampling Random-Walk between
// RGB, Direction and Distance sampling
float
calculateOneSamplePDF(float NdotDir, const scene_rdl2::math::Vec3f& wo, const scene_rdl2::math::Vec3f& wi,
        const scene_rdl2::math::Color& dwivediV0, const scene_rdl2::math::Color& dwivediNormalization,
        const VolumePhase& sssPhaseFunction, const scene_rdl2::math::Color& wChannelSampling,
        float phaseSamplingWeight, const scene_rdl2::math::Color& pdfDistance, const scene_rdl2::math::Color& pdfDistanceDwivedi)
{
    const float pdfPhaseFunction = sssPhaseFunction.pdf(wo, wi);
    // dwivedi PDF
    // Ref "Improved Dwivedi Sampling" Eqn 12
    const scene_rdl2::math::Color pdfDwivedi = scene_rdl2::math::sOneOverTwoPi * dwivediNormalization /
                   (dwivediV0 - NdotDir);

    float oneSamplePdf = 0.0f;
    for (int c = 0; c < 3; ++c) {
        oneSamplePdf += (wChannelSampling[c]*
                (phaseSamplingWeight*pdfPhaseFunction*pdfDistance[c] +
                (1.0f-phaseSamplingWeight)*pdfDwivedi[c]*pdfDistanceDwivedi[c]));
    }
    return oneSamplePdf;
}

// Utility Function to Check if the Path Traced SSS Intersection is Valid
bool
isValidIntersection(const Intersection& isect,
        const VolumeSubsurface& volumeSubsurface)
{
    // If the intersection has a material
    bool isValid = (isect.getMaterial() != nullptr);

    if (volumeSubsurface.resolveSelfIntersections()) {
        // We Ignore intersections that are NOT exiting the subsurface
        // random walk but entering another one.
        isValid = isValid && (isect.isEntering() == false);
    }
    return isValid;
}

scene_rdl2::math::Color
PathIntegrator::computeRadiancePathTraceSubsurface(pbr::TLState* pbrTls,
        const Bsdf& bsdf, const Subpixel &sp, const PathVertex &pv,
        const RayDifferential& ray, const Intersection& isect,
        const VolumeSubsurface& volumeSubsurface, const LightSet& lightSet,
        bool doIndirect, float rayEpsilon, float shadowRayEpsilon, unsigned &sequenceID,
        scene_rdl2::math::Color &ssAov, float* aovs) const
{
    scene_rdl2::math::Color radiance(0.0f);
    // Check if samples are zero OR
    // Global Subsurface is Toggled OFF
    if (mBssrdfSamples < 1 || !mEnableSSS) {
        return radiance;
    }

    // -----------------------------------------------------------------------------------------------------------------
    // Fix (part 1) for MOONRAY-3352:
    // Backup the current aov values. For explanation, search "MOONRAY-3352" lower in this function.
    float *backupAovs = nullptr;
    const int aovNumChannels = pbrTls->mFs->mAovSchema->numChannels();
    scene_rdl2::alloc::Arena *arena = pbrTls->mArena;
    SCOPED_MEM(arena);

    if (aovs != nullptr) {
        backupAovs = arena->allocArray<float>(aovNumChannels);
        memcpy(backupAovs, aovs, aovNumChannels * sizeof(float));
    }
    // -----------------------------------------------------------------------------------------------------------------

    EXCL_ACCUMULATOR_PROFILE(pbrTls, EXCL_ACCUM_SSS_INTEGRATION);
    const Scene *scene = MNRY_VERIFY(pbrTls->mFs->mScene);
    // volume coefficient for scatter event sampling and transmittance
    const scene_rdl2::math::Color albedo = volumeSubsurface.getAlbedo();
    const scene_rdl2::math::Color sigmaT = volumeSubsurface.getSigmaT();
    const scene_rdl2::math::Color sigmaS = sigmaT * albedo;

    // Dwivedi Sampling Params
    // Ref Improved Dwivedi Sampling
    // Ref A Zero-Variance based Sampling Scheme for Monte Carlo SSS
    // Largest Discrete Eigenvalue of the transport operator
    const scene_rdl2::math::Color dwivediV0 = volumeSubsurface.getDwivediV0();

    // subsurface scattering is considered as diffuse reflection for now
    const BsdfLobe::Type lobeType =
        BsdfLobe::Type(BsdfLobe::REFLECTION | BsdfLobe::DIFFUSE);
    int nSubsurfaceSample = (pv.nonMirrorDepth == 0) ? mBssrdfSamples : 1;
    float time = ray.time;
    // potential normal map binding
    const scene_rdl2::rdl2::Material* sssMaterial = volumeSubsurface.getMaterial();
    const scene_rdl2::rdl2::EvalNormalFunc evalNormalFunc = volumeSubsurface.getEvalNormalFunc();
    bool evalNormalBinding = sssMaterial && evalNormalFunc;
    // trace set for sss
    auto geomTls = pbrTls->mTopLevelTls->mGeomTls.get();
    geomTls->mSubsurfaceTraceSet = volumeSubsurface.getTraceSet();
    // Fresnel evaluation
    scene_rdl2::math::Vec3f nIn = evalNormalBinding ?
        evalNormalFunc(sssMaterial, pbrTls->mTopLevelTls->mShadingTls.get(),
            shading::State(&isect)) :
            isect.getN();
    scene_rdl2::math::Color fresnel = evalFresnel(volumeSubsurface.getTransmissionFresnel(),
        nIn, -ray.dir);

    // We disable the transmission fresnel on the wi side.
    // If the material is using a OneMinusRoughSchlickFresnel, this matches the
    // energy conservation we are doing with other types of "under" lobes.
    // Otherise it darkens the SSS somewhat a bit when the specular factor of
    // the "top" lobe is in the 0.2 .. 0.7 range.
    const Fresnel* transmissionFresnel = nullptr;

    // Make sure we do all local sss sampling with input sequenceID, which will
    // get incremented in the process.
    unsigned sssSampleID = sequenceID;

    // Local Reference Frame
    // Using the smooth shading normal here instead of Ng to avoid
    // faceting artifacts (similar to the diffusion integrator above)
    scene_rdl2::math::ReferenceFrame inReferenceFrame(isect.getN());
    const scene_rdl2::math::ReferenceFrame inReferenceFrameNg(isect.getNg());
    pbrTls->mStatistics.addToCounter(STATS_SSS_SAMPLES, nSubsurfaceSample);
 

    // Counter to calculate all the *valid* subsurface samples evaluated to
    // Divide the throughput correctly
    unsigned int samplesEvaluated = 0;
    // Weight for Isotropic Sampling
    float phaseSamplingWeight = 1.0f;
    // Weight for sampling 'back' towards the surface along N for Dwivedi
    // Sampling. Save some samples for going 'forwards'
    // for thin-geom/back-lighting scenes.
    // Ref "Improved Dwivedi Sampling"
    const float wBackwardSampling = 0.9f;

    for (int s = 0; s < nSubsurfaceSample; ++s) {
        // sample sequence for random walk direction
        IntegratorSample2D directionSamples(SequenceIDIntegrator(
            sp.mPixel, sp.mSubpixelIndex,
            SequenceType::BssrdfGlobal,
            SequenceType::VolumePhase,
            s, sssSampleID));
        // sample sequence for selecting RGB channel to do distance sampling
        IntegratorSample1D channelSamples(SequenceIDIntegrator(
            sp.mPixel, sp.mSubpixelIndex,
            SequenceType::BssrdfLocalChannel,
            SequenceType::VolumeScattering,
            s, sssSampleID));
        // sample sequence for random walk scatter distance
        IntegratorSample1D distanceSamples(SequenceIDIntegrator(
            sp.mPixel, sp.mSubpixelIndex,
            SequenceType::BssrdfGlobal,
            SequenceType::VolumeScattering,
            s, sssSampleID));
        // sample sequence for random walk forward/backward sampling
        IntegratorSample1D dwivediDirectionSamples(SequenceIDIntegrator(
            sp.mPixel, sp.mSubpixelIndex,
            SequenceType::BssrdfGlobal,
            SequenceType::VolumeScattering,
            s, sssSampleID));
#if 0
        // NOT using RR at the moment
        // sample sequence for russian roulette
        IntegratorSample1D rrSamples(SequenceIDIntegrator(
            sp.mPixel, sp.mSubpixelIndex,
            SequenceType::BssrdfGlobal,
            SequenceType::RussianRouletteBsdf,
            s, sssSampleID));
#endif
        // setup initial path throughput
        scene_rdl2::math::Color pt = volumeSubsurface.getScale() * pv.pathThroughput *
                fresnel / nSubsurfaceSample;
        if (isBlack(pt)) {
            continue;
        }
        // sample a random walk direction using hemisphere cosine distribution
        // along the flip side of shading Normal
        scene_rdl2::math::Vec2f uDir;
        directionSamples.getSample(&uDir[0], pv.nonMirrorDepth);
        scene_rdl2::math::Vec3f dir = -inReferenceFrame.localToGlobal(
            sampleLocalHemisphereCosine(uDir[0], uDir[1]));

        // Use reverse geometric normal with minimum of rayEpsilon and sHitEpsilonStart (10e-5)  
        // min(rayEpsilon, sHitEpsilonStart) is used instead of just rayEpsilon for the following reason
        // If camera is far away from surface, rayEpsilon becomes much larger than the least scatter_color component
        // that's non zero. This results in loss of samples when selecting that color channel during importance sampling.
        // This ends up "tinting" the transmittance more to the largest scatter_color component.     
        // MOONRAY-4260 is an example of this
        // Reverse geometric normal is used to avoid self intersection with same triangle. 
        // This is better than using rayDir or shading normal as grazing angles can cause more self intersections. 
        // We could compensate by increasing epsilon based on rayDir and geometric normal but grazing angles can cause
        // us to use a much larger epsilon leading back to the original issue rayEpsilon issue above.   
        // https://link.springer.com/content/pdf/10.1007%2F978-1-4842-4427-2_6.pdf has more insight on this
        // TODO: do the random walk in volumesubsurface space so we can reduce precision loss even further when camera is 
        // far away. this involves generating a BVH with just the volume subsurface at render prep time.   
        scene_rdl2::math::Vec3f org = isect.getP() - inReferenceFrameNg.getN() * scene_rdl2::math::min(rayEpsilon, sHitEpsilonStart);
        bool reachSurface = false;
        scene_rdl2::math::Vec3f pOut, nOut;
        Intersection isectOut;

        // ZERO Scatter Integrator
        // Direct Transmission Events using mZeroScatterSigmaT
        constexpr size_t maxZeroScatterEvents = 1;
        for (size_t nScatter = 0; nScatter < maxZeroScatterEvents; ++nScatter) {
            if (isBlack(pt)) {
                break;
            }
            // sample the next scattering event based on path throughput and sigmaT
            float uChannel, uDistance;
            channelSamples.getSample(&uChannel, pv.nonMirrorDepth);
            distanceSamples.getSample(&uDistance, pv.nonMirrorDepth);
            scene_rdl2::math::Color tr;
            float pdfT;
            scene_rdl2::math::Color zeroScatterSigmaT = volumeSubsurface.getZeroScatterSigmaT();
            float tfar = sampleDistanceZeroScatter(pt, zeroScatterSigmaT,
                uChannel, uDistance, tr, pdfT);
            // setup random walk ray
            Ray sssRay(org, dir, 0.0f, tfar, time);
            // setup trace set
            sssRay.ext.geomTls = (void*)geomTls;
            const scene_rdl2::rdl2::Material* isectMaterial = isect.getMaterial();
            if (isectMaterial) {
                const shading::Material *materialExt =
                    &isectMaterial->get<const shading::Material>();
                sssRay.ext.materialID = materialExt->getMaterialId();
            }
            bool intersected = scene->intersectRay(pbrTls->mTopLevelTls, sssRay,
                isectOut, lobeType);
            if (intersected && isValidIntersection(isectOut, volumeSubsurface)) {
                // update throughput
                tr = exp(-zeroScatterSigmaT * sssRay.tfar);
                // the pdf of tfar sample from the intersection point
                // see pbrt3
                // http://www.pbr-book.org/3ed-2018/Light_Transport_II_Volume_Rendering/Sampling_Volume_Scattering.html#eq:homogeneous-medium-psurf
                // for derivation
                // 15.11, 1/n is for uniform sampling
                // we use MIS between channels based on throughput instead
                float normalization = 1.0f / (pt[0] + pt[1] + pt[2]);
                float pdfExitSurf =
                    pt[0] * normalization * tr[0] +
                    pt[1] * normalization * tr[1] +
                    pt[2] * normalization * tr[2];
                pt *= tr / pdfExitSurf;

                pOut = isectOut.getP();
                // construct intersection

                // The material of the exiting intersection must match the
                // sssMaterial to evaluate the normal map. This ensures that
                // the intersection state has all the attributes requested by the
                // normal map. There may not be a match when using trace sets.
                // In case of a mismatch, we do not attempt to evaluate the subsurface normal map.
                if (evalNormalBinding && isectOut.getMaterial() == isectMaterial) {

                    // Invert Intersection Normals here because even though we'll hit the surfaces
                    // from the back-side, we want to shade the point on the 'outside' facing geo.
                    // One of the applications is computing the correct ReferenceFrame to
                    // for evaluating tangent space normal maps and for
                    // correct transformations to render space. This should also helps for projected
                    // normals where we use the surface normal for calculating the correct projection.
                    // Note that computing tangent space normal on the back side of the geo (-N)
                    // and inverting the resulting normal is not mathematically equivalent.
                    isectOut.invertIntersectionNormals();
                    geom::initIntersectionPhase2(isectOut, pbrTls->mTopLevelTls,
                            pv.mirrorDepth, pv.glossyDepth, pv.diffuseDepth,
                            isSubsurfaceAllowed(pv.subsurfaceDepth), pv.minRoughness,
                            sssRay.dir);
                    RayDifferential rayDiff(ray,
                                            isectOut.getEpsilonHint(),
                                            sssRay.tfar);
                    isectOut.transferAndComputeDerivatives(pbrTls->mTopLevelTls,
                                                           &rayDiff,
                                                           sp.mTextureDiffScale);

                    // We evaluate using the sssMaterial, which does not always match the isectMaterial.
                    // Example: sssMaterial is undermaterial of isectMaterial, which is a GlitterFlakeMaterial.
                    nOut = evalNormalFunc(sssMaterial,
                        pbrTls->mTopLevelTls->mShadingTls.get(),
                        shading::State(&isectOut));
                } else {
                    geom::initIntersectionPhase2(isectOut, pbrTls->mTopLevelTls,
                            pv.mirrorDepth, pv.glossyDepth, pv.diffuseDepth,
                            isSubsurfaceAllowed(pv.subsurfaceDepth), pv.minRoughness,
                            -sssRay.dir);
                    nOut = -isectOut.getN();
                }
                // used for subsurface material aovs
                ssAov += pt;
                reachSurface = true;
                samplesEvaluated += 1;
                // we are getting out of surface
                // calculate the radiance contribution of random walk exit point
                // Count The Samples Evaluated
                const bool includeCosineTerm = true, isEntering = true;
                shading::BsdfSlice sliceLocal(nOut, -ray.getDirection(),
                                              includeCosineTerm, isEntering,
                                              pbrTls->mFs->mShadowTerminatorFix);
                const bool isReflection = true;
                LambertBsdfLobe lobeLocal(nOut, scene_rdl2::math::sWhite, isReflection);
                radiance += computeRadianceSubsurfaceSample(
                    pbrTls, bsdf, sp, pv, ray, isectOut.getdNdx(), isectOut.getdNdy(),
                    pt, transmissionFresnel, lightSet, lobeLocal, sliceLocal, pOut, nOut,
                    nSubsurfaceSample, 1, s, doIndirect,  rayEpsilon, shadowRayEpsilon,
                    sequenceID, sequenceID, true, aovs, isect);
            }
        }

        if (reachSurface == true) {
            // Had a Zero Scatter Event
            // Try another sample
            continue;
        }

        // 1+ Scatter Event
        // For now using isotropic phase function (what Hyperion does)
        VolumePhase sssPhaseFunction = VolumePhase(0.0f);
        // set a maximum scattering depth in case we go into
        // infinite random walk (due to bad geometry setup)
        // the hardcoded number is mentioned in Pixar's memo

        scene_rdl2::math::Color tr;
        bool choseDwivediSampling = false;
        // set a maximum scattering depth in case we go into
        // infinite random walk (due to bad geometry setup)
        // the hardcoded number is mentioned in Pixar's memo
        constexpr size_t maxScatterEvents = 256;
        bool hadScatterEvent = false;
        for (size_t nScatter = 0; nScatter < maxScatterEvents; ++nScatter) {
            if (isBlack(pt)) {
                break;
            }

            // Start using Dwivedi Sampling after a few
            // Phase-Function related bounces inside
            // Ref : A Zero-Variance Based Sampling for MonteCarlo SSS
            // Paper suggests a weight distribution of (0.25f, 0.75f) towards
            // Dwivedi Sampling, but in our tests (0.1f, 0.9f) seems to
            // work best after a couple of steps inside. Starting Dwivedi
            // at the first steps seems to result in much less saturated results
            phaseSamplingWeight = (nScatter > 1) ? 0.1f : 1.0f;

            // sample the next scattering event based on path throughput and sigmaT
            // Sample between RGB
            float uChannel;
            channelSamples.getSample(&uChannel, pv.nonMirrorDepth);
            scene_rdl2::math::Color channelSamplingWeights;
            int chIndex = sampleChannel(pt, uChannel, channelSamplingWeights);

            // Sample Distance
            float uDistance;
            distanceSamples.getSample(&uDistance, pv.nonMirrorDepth);
            float NdotDir = dot(dir, inReferenceFrame.getN());
            // sigmaTP for Dwivedi Sampling
            // Dwivedi Distance Sampling - take larger steps when going back
            // towards the normal
            // Ref "Improved Dwivedi Sampling" Eqn 13
            scene_rdl2::math::Color sigmaTPrime = sigmaT*(1.0f - NdotDir / dwivediV0);
            scene_rdl2::math::Color pdfDistance, pdfDistanceDwivedi;
            float tfar = sampleDistanceMultiScatter(sigmaT, sigmaTPrime,
                channelSamplingWeights, chIndex, uDistance, NdotDir, dwivediV0,
                choseDwivediSampling, tr, pdfDistance, pdfDistanceDwivedi);

            // setup random walk ray
            Ray sssRay(org, dir, 0.0f, tfar, time);
            // setup trace set
            sssRay.ext.geomTls = (void*)geomTls;
            const scene_rdl2::rdl2::Material* isectMaterial = isect.getMaterial();
            if (isectMaterial) {
                const shading::Material *materialExt =
                    &isectMaterial->get<const shading::Material>();
                sssRay.ext.materialID = materialExt->getMaterialId();
            }
            bool intersected = scene->intersectRay(pbrTls->mTopLevelTls, sssRay,
                isectOut, lobeType);
            if (intersected && isValidIntersection(isectOut, volumeSubsurface)) {
                // update throughput
                if (hadScatterEvent == false) {
                    // we've already accounted for zero scattering in the loop above.
                    // we should not count this sample because the sigmaT here is not
                    // calibrated for zeroScatter and will cause hue-shifts for thin geo.
                    break;
                } else {
                    tr = exp(-sigmaT * sssRay.tfar);
                }

                reachSurface = true;

                // transmittance per dwivedi sampling
                scene_rdl2::math::Color trp = exp(-sigmaTPrime*sssRay.tfar);

                // the pdf of tfar sample from the intersection point
                // see pbrt3
                // http://www.pbr-book.org/3ed-2018/Light_Transport_II_Volume_Rendering/Sampling_Volume_Scattering.html#eq:homogeneous-medium-psurf
                // for derivation
                // 15.11, 1/n is for uniform sampling
                // we use MIS between channels based on throughput instead
                float normalization = 1.0f / (pt[0] + pt[1] + pt[2]);
                float pdfSurf =
                        pt[0] * normalization * (phaseSamplingWeight*tr[0] +(1.0f-phaseSamplingWeight)*trp[0])+
                        pt[1] * normalization * (phaseSamplingWeight*tr[1] +(1.0f-phaseSamplingWeight)*trp[1])+
                        pt[2] * normalization * (phaseSamplingWeight*tr[2] +(1.0f-phaseSamplingWeight)*trp[2]);

                pt *= tr / pdfSurf;

                pOut = isectOut.getP();
                // construct intersection

                // The material of the exiting intersection must match the
                // sssMaterial to evaluate the normal map. This ensures that
                // the intersection state has all the attributes requested by the
                // normal map. There may not be a match when using trace sets.
                // In case of a mismatch, we do not attempt to evaluate the subsurface normal map.
                if (evalNormalBinding && isectOut.getMaterial() == isectMaterial) {

                    // Invert Intersection Normals here because even though we'll hit the surfaces
                    // from the back-side, we want to shade the point on the 'outside' facing geo.
                    // One of the applications is computing the correct ReferenceFrame to
                    // for evaluating tangent space normal maps and for
                    // correct transformations to render space. This should also helps for projected
                    // normals where we use the surface normal for calculating the correct projection.
                    // Note that computing tangent space normal on the back side of the geo (-N)
                    // and inverting the resulting normal is not mathematically equivalent.
                    isectOut.invertIntersectionNormals();
                    geom::initIntersectionPhase2(isectOut, pbrTls->mTopLevelTls,
                            pv.mirrorDepth, pv.glossyDepth, pv.diffuseDepth,
                            isSubsurfaceAllowed(pv.subsurfaceDepth), pv.minRoughness,
                            sssRay.dir);
                    RayDifferential rayDiff(ray,
                                            isectOut.getEpsilonHint(),
                                            sssRay.tfar);
                    isectOut.transferAndComputeDerivatives(pbrTls->mTopLevelTls,
                                                           &rayDiff,
                                                           sp.mTextureDiffScale);

                    // We evaluate using the sssMaterial, which does not always match the isectMaterial.
                    // Example: sssMaterial is undermaterial of isectMaterial, which is a GlitterFlakeMaterial.
                    nOut = evalNormalFunc(sssMaterial,
                        pbrTls->mTopLevelTls->mShadingTls.get(),
                        shading::State(&isectOut));
                } else {
                    geom::initIntersectionPhase2(isectOut, pbrTls->mTopLevelTls,
                            pv.mirrorDepth, pv.glossyDepth, pv.diffuseDepth,
                            isSubsurfaceAllowed(pv.subsurfaceDepth), pv.minRoughness,
                            -sssRay.dir);
                    nOut = -isectOut.getN();
                }
                break;
            } else {
                // Scatter Event
                hadScatterEvent = true;
                org = sssRay.org + tfar * sssRay.dir;

                pt *= tr*sigmaS;
                pt *= sssPhaseFunction.eval(sssRay.dir, dir);
                // Sample scattering direction
                directionSamples.getSample(&uDir[0], pv.nonMirrorDepth);

                // Select between front/back sampling
                float backwardDirectionSample;
                dwivediDirectionSamples.getSample(&backwardDirectionSample, pv.nonMirrorDepth);

                sampleDirection(uDir, backwardDirectionSample, chIndex,
                    phaseSamplingWeight, wBackwardSampling, sssPhaseFunction,
                    sssRay, dwivediV0, inReferenceFrame, choseDwivediSampling, NdotDir, dir);

                float pdfSampling = calculateOneSamplePDF(NdotDir, sssRay.dir, dir,
                    dwivediV0, volumeSubsurface.getDwivediNormPDF(), sssPhaseFunction,
                    channelSamplingWeights, phaseSamplingWeight, pdfDistance, pdfDistanceDwivedi);
                pt /= pdfSampling;

#if 0
                // TODO: Fix this.
                // Following RR code doesn't work correctly.
                // Increasing/Decreasing Pixel/Bssrdf Samples creates unpredictable,
                // wildly different results.
                float lum = luminance(pt);
                if (lum < mRussianRouletteThreshold) {
                    float continueProbability = std::max(
                        sEpsilon, lum * mInvRussianRouletteThreshold);
                    float uRR;
                    rrSamples.getSample(&uRR, pv.nonMirrorDepth);
                    if (uRR > continueProbability) {
                        break;
                    } else {
                        pt /= continueProbability;
                    }
                }
#endif
            }
        }

        // We should only skip the samples discarded because of
        // zero-scattering
        // Reminder: we distinguish between zero and multi scatter to avoid
        // hue shifts
        if (hadScatterEvent) {
            // count the sample if multi-scatter
            samplesEvaluated += 1;
        }

        if (!reachSurface) {
            // the sample never reached the surface
            continue;
        }

        // used for subsurface material aovs
        ssAov += pt;

        const bool includeCosineTerm = true, isEntering = true;
        shading::BsdfSlice sliceLocal(nOut, -ray.getDirection(),
                                      includeCosineTerm, isEntering,
                                      pbrTls->mFs->mShadowTerminatorFix);
        const bool isReflection = true;
        LambertBsdfLobe lobeLocal(nOut, scene_rdl2::math::sWhite, isReflection);
        radiance += computeRadianceSubsurfaceSample(
            pbrTls, bsdf, sp, pv, ray, isectOut.getdNdx(), isectOut.getdNdy(),
            pt, transmissionFresnel, lightSet, lobeLocal, sliceLocal, pOut, nOut,
            nSubsurfaceSample, 1, s, doIndirect,  rayEpsilon, shadowRayEpsilon,
            sequenceID, sequenceID, true, aovs, isect);

    }

    // Radiance tweak added by Priyamvad as part of his work for MOONRAY-2049.
    // However, this led to a bug, MOONRAY-3352, since he didn't do the corresponding thing for the aovs,
    // causing a mismatch. This is now fixed - details below.
    if (samplesEvaluated > 0 && samplesEvaluated < nSubsurfaceSample) {
        // Only divide for the valid samples evaluated
        // Helps with geometry with holes in it, like eyeballs, where a
        // few samples can potentially not reach the surface.
        float scaleFactor = nSubsurfaceSample / (float)samplesEvaluated;
        radiance *= scaleFactor;
        ssAov    *= scaleFactor;

        // -------------------------------------------------------------------------------------------------------------
        // Fix for MOONRAY-3352:
        // Use the backed-up aov values to modify the aovs consistently with the scaled radiance.
        // 
        // Essentially, we only scale the part of each aov that's been accumulated since the start of this function.
        // But there's an additional consideration - this function can be called recursively. So lets discuss how this
        // scaling needs to work.
        //
        // Consider 2 alternative software models for accumulating radiance across a sequence of nested function calls:
        // 
        // (1) Each function has a local radiance variable which it initialises to 0 and then proceeds to add to it.
        // Each summand could either be a simple directly evaluated expression, or the value returned by a nested
        // function call. At the end of the function, the locally accumulated radiance is returned, so that the calling
        // function can accumulate it to its own scoped value using radiance += func(); where func() is the function
        // we just exited and which returned the locally accumulated radiance.
        // 
        // (2) There's a single global radiance accumulator, which lives in the topmost function, and is passed by
        // reference into lower level functions. All functions modify this radiance variable directly, and do not
        // therefore need to return a radiance value.
        //
        // Moonray actually uses a mixture of both models to compute a final radiance value, but clearly the present
        // function is using model 1. The additional complication it introduces is that the radiance is being scaled
        // (sometimes) by a scale factor just before being returned.
        //
        // Let's consider how we would perform the same scaling if we were to (hypothetically) modify this function
        // to use model 2 instead. If the global radiance has value r0 at the beginning of the function call, and after
        // the various accumulations performed by the function, it has value r1 at the end, then the difference in
        // radiance r1-r0 corresponds to the locally accumulated value r that gets returned by the function under model
        // 1. If model 1 now applies a scale factor s to r, and returns the final value r*s, under model 2 the portion
        // of the accumulated radiance to be scaled by s is r1-r0, so the adjusted value at the end of the function
        // should be r0 + s*(r1-r0) = lerp(r0, r1, s). Note that this is true regardless of what expressions any
        // lower-level functions are using to compute their radiance contributions, including, crucially, the same
        // type of scaling performed by lower level recursive instantiations of the same function. These details are
        // irrelevant to the present scope of the function because they are all hidden internal details of the nested
        // function calls. So the lerp is still the appropriate thing to do even in the recursive case.
        //
        // The reason it was important to work through that hypothetical case is that the aovs exclusively use model 2
        // to perform their accumulations. This makes good sense because some renders use large numbers of aovs, and it
        // would require lots of memory plus the additional execution cost of zeroing that memory at the start of each
        // function to support model 1 for aovs. Hence to support the local scaling of aovs, using model 2, we apply
        // the lerp indicated above on each aov channel. Note that we backed up the values of all aovs on entry to this
        // function.

        if (aovs != nullptr) {
            for (int i = 0; i < aovNumChannels; ++i) {
                aovs[i] = scene_rdl2::math::lerp(backupAovs[i], aovs[i], scaleFactor);
            }
        }

        // -------------------------------------------------------------------------------------------------------------
    }
    return radiance;
}

//----------------------------------------------------------------------------

} // namespace pbr
} // namespace moonray

