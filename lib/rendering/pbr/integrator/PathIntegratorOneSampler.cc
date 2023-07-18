// Copyright 2023 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

/// @file PathIntegratorOneSampler.cc

// Integration implementation based on the BsdfOneSampler
// 
// When making changes to this file, you'll likely also need
// to update the vector implementation and the multi-sample
// integrator (both vector and scalar).
//   PathIntegratorOneSampler.ispc
//   PathIntegratorMultiSampler.cc
//   PathIntegratorMultiSampler.ispc

#include "PathIntegrator.h"

#include <moonray/rendering/pbr/core/Aov.h>
#include <moonray/rendering/pbr/core/Constants.h>
#include <moonray/rendering/pbr/core/DebugRay.h>
#include <moonray/rendering/pbr/core/RayState.h>
#include "PathIntegratorUtil.h"
#include "VolumeTransmittance.h"

// using namespace scene_rdl2::math; // can't use this as it breaks openvdb in clang.

namespace moonray {
namespace pbr {

// Debug ray recording:
// Scalar mode has the seldom used feature "moonray -record_rays"
// which allows us to create a .mm that can be used to visualize
// ray origin and directions.  We store rays that leave a surface
// and hit a light.  Several RaTS tests make use of this feature.
// I suspect that this feature will just go away when .mm is finally
// removed.  But for now, we support it.
static void
debugRayRecordLightHit(
    pbr::TLState *pbrTls,
    const scene_rdl2::math::Vec3f &P,
    const scene_rdl2::math::Vec3f &wi,
    float distance,
    const scene_rdl2::math::Color &Li)
{
    // If the intersection distance is closer than the distant light, then
    // assume the hit wasn't due to a distant or env light.
    if (DebugRayRecorder::isRecordingEnabled()) {
        if (distance < sDistantLightDistance) {
            mcrt_common::Ray debugRay(P, wi, 0.0f);
            RAYDB_EXTEND_RAY_NO_HIT(pbrTls, debugRay, distance);
            RAYDB_SET_CONTRIBUTION(pbrTls, Li);
            RAYDB_ADD_TAGS(pbrTls, TAG_AREALIGHT);
        } else {
            mcrt_common::Ray debugRay(P, wi, 0.0f);
            RAYDB_EXTEND_RAY_NO_HIT(pbrTls, debugRay, 40.0f);
            RAYDB_SET_CONTRIBUTION(pbrTls, Li);
            RAYDB_ADD_TAGS(pbrTls, TAG_ENVLIGHT);
        }
    }
}

// Helper function that abstracts the differences between
// one lobe and one sample sampling within the context of path guiding.
// The key abstractions are the bsdf sample and eval functions.
// When doing one sampling, we use the BsdfOneSamplerObject, when
// doing one lobe, we already have the lobe and use the lobe's
// sample and eval functions directly.
static scene_rdl2::math::Color
sampleWithPathGuide(const PathGuide &pg,
                    bool isMirror,
                    const scene_rdl2::math::Vec3f &P,
                    float &r0, float &r1, float &r2,
                    std::function<scene_rdl2::math::Color(float, float, float, scene_rdl2::math::Vec3f&, float&)> bsdfSampleFn,
                    std::function<scene_rdl2::math::Color(const scene_rdl2::math::Vec3f &, float&)> bsdfEvalFn,
                    scene_rdl2::math::Vec3f &wi,
                    float &pdfBsdf)
{
    scene_rdl2::math::Color f;
    if (pg.canSample() && pg.getPercentage() > 0 && !isMirror) {
        float pdfPg = 0.f;
        const float u = pg.getPercentage();
        if (r1 > u) {
            // use bsdf sampling direction
            r1 = (r1 - u) / (1.0f - u); // remap r1 into [0, 1)
            f = bsdfSampleFn(r0, r1, r2, wi, pdfBsdf);
            pdfPg = pg.getPdf(P, wi);
        } else {
            // use path guide sampling direction
            r1 = r1 / u; // remap r1 into [0, 1)
            wi = pg.sampleDirection(P, r1, r2, &pdfPg);
            f = bsdfEvalFn(wi, pdfBsdf);
        }
        if (isSampleValid(f, pdfBsdf)) {
            // blending pdf values seems to work well enough in practice, and
            // allows a potential user percentage control
            pdfBsdf = u * pdfPg + (1.0 - u) * pdfBsdf;
        }
    } else {
        f = bsdfSampleFn(r0, r1, r2, wi, pdfBsdf);
    }

    return f;
}

// Helper function that applies russian roulette.
// Determine to either cull c or keep and bump of c's contribution.
// Returns true if c is kept (and bumped up)
// Returns false if the sample should be culled
static bool
applyRussianRoulette(float r, float threshold, float invThreshold, int nonMirrorDepth, float lum, scene_rdl2::math::Color &c)
{
    // The direct comparison against 0.0f is intentional.
    // A threshold of exactly zero means that russian roulette is off.
    // But we always want to cull 0 luminance samples.
    if (lum == 0.0f) {
        return false; // cull
    }

    if (nonMirrorDepth > 0 && lum < threshold) {
        const float continueProbability = scene_rdl2::math::max(scene_rdl2::math::sEpsilon, lum * invThreshold);
        if (r > continueProbability) {
            return false; // cull
        }
        c *= scene_rdl2::math::rcp(continueProbability); // bump it up
    }

    return true; // keep it
}

scene_rdl2::math::Color
PathIntegrator::oneSamplerDirectLight(pbr::TLState *pbrTls,
    const Subpixel &sp, int cameraId, const PathVertex &pv, const mcrt_common::RayDifferential &ray,
    const shading::Intersection &isect, const Light *light, const LightFilterList *lightFilterList,
    const scene_rdl2::math::Vec3f *cullingNormal, const BsdfOneSampler &bsdfOneSampler, const scene_rdl2::math::Color &pt,
    const float r[9], float rayEpsilon, float shadowRayEpsilon, unsigned sequenceID, float *aovs) const
{
    // r[0:2] used for light sample, r[3] used for russian roulette, r[4:5] for the light filter

    // Sample light direction
    LightIntersection lightIsect;
    scene_rdl2::math::Vec3f wi;
    const scene_rdl2::math::Vec3f rLight(r[0], r[1], r[2]); // first 3 samples used for light sampling
    if (!light->sample(isect.getP(), cullingNormal, ray.getTime(), rLight, wi, lightIsect, ray.getDirFootprint())) {
        return scene_rdl2::math::sBlack;
    }

    // Evaluate light radiance and pdf
    float lightPdf = 0.0f;
    const LightFilterRandomValues rLightFilter = {
        scene_rdl2::math::Vec2f(r[4], r[5]), 
        scene_rdl2::math::Vec3f(r[6], r[7], r[8])};
    const scene_rdl2::math::Color Li = light->eval(pbrTls->mTopLevelTls, wi, isect.getP(), rLightFilter, ray.getTime(), lightIsect,
        /* fromCamera = */ false, lightFilterList, ray.getDirFootprint(), &lightPdf);
    if (isSampleInvalid(Li, lightPdf)) {
        return scene_rdl2::math::sBlack;
    }

    // sample is valid - count it
    pbrTls->mStatistics.incCounter(STATS_LIGHT_SAMPLES);

    // evaluate the bsdf, we'll need individual lobe contributions for aovs
    BsdfOneSampler::LobesContribution lobesContribution;
    float bsdfPdf;
    const scene_rdl2::math::Color f = bsdfOneSampler.eval(wi, bsdfPdf, &lobesContribution);
    if (isSampleInvalid(f, bsdfPdf)) {
        return scene_rdl2::math::sBlack;
    }

    // if we are path guiding, adjust the bsdf pdf accordingly
    if (mPathGuide.canSample()) {
        MNRY_ASSERT(lobesContribution.mMatchedLobeCount > 0);
        const shading::BsdfLobe *lobe = lobesContribution.mLobes[0];
        MNRY_ASSERT(lobe);
        if (!(lobe->getType() & shading::BsdfLobe::MIRROR)) {
            const float u = mPathGuide.getPercentage();
            const float pgPdf = mPathGuide.getPdf(isect.getP(), wi);
            // blending pdf values seems to work well enough in practice, and
            // allows for a potential user percentage control
            bsdfPdf = u * pgPdf + (1.0 - u) * bsdfPdf;
        }
    }

    const float misWeight = powerHeuristic(lightPdf, bsdfPdf);

    // Compute light contribution, taking path throughput into account
    scene_rdl2::math::Color lightContribution = pt * misWeight * Li / lightPdf;

    // Apply RR based on luminance
    const float lum = luminance(f * lightContribution);
    if (!applyRussianRoulette(r[3], mRussianRouletteThreshold, mInvRussianRouletteThreshold,
                              pv.nonMirrorDepth, lum, lightContribution)) {
        return scene_rdl2::math::sBlack;
    }

    // resolve occlusion
    const float tfar = lightIsect.distance - rayEpsilon;
    const int rayDepth = ray.getDepth() + 1;
    // THINK: Would a standard ray, rather than a ray differential work better?
    // Using a ray differential produces more occlusion (i.e. darker images).
    // Since the multi-sample integrator uses ray differentials, we do so here
    // as well.  It might be worth changing this behavior in both integrators.
    mcrt_common::RayDifferential shadowRay(isect.getP(), wi,
                                           ray.getOriginX(), wi,
                                           ray.getOriginY(), wi,
                                           rayEpsilon, tfar, ray.getTime(), rayDepth);
    float presence = 0.0f;

    const FrameState &fs = *pbrTls->mFs;
    const bool hasUnoccludedFlag = fs.mAovSchema->hasLpePrefixFlags(AovSchema::sLpePrefixUnoccluded);
    int32_t assignmentId = isect.getLayerAssignmentId();
    if (isRayOccluded(pbrTls, light, shadowRay, rayEpsilon, shadowRayEpsilon, presence, assignmentId)) {
        // Visibility LPE
        if (aovs) {
            EXCL_ACCUMULATOR_PROFILE(pbrTls, EXCL_ACCUM_AOVS);
            const LightAovs &lightAovs = *fs.mLightAovs;

            // If there is no visibility AOV or if we don't have unoccluded flags, then we don't need to bother
            // with accumulating these values here.
            if (lightAovs.hasVisibilityEntries() || hasUnoccludedFlag) {

                const AovSchema &aovSchema = *fs.mAovSchema;
                bool addVisibilityAov = true;
                for (int lobeIndex = 0; lobeIndex < lobesContribution.mMatchedLobeCount; ++lobeIndex) {
                    int lpeStateId = pv.lpeStateId;
                    lpeStateId = lightAovs.scatterEventTransition(pbrTls, lpeStateId, bsdfOneSampler.getBsdf(),
                        *(lobesContribution.mLobes[lobeIndex]));
                    lpeStateId = lightAovs.lightEventTransition(pbrTls, lpeStateId, light);

                    // Add any visibility information:
                    if (addVisibilityAov) {
                        if (lightAovs.hasVisibilityEntries()) {
                            if (aovAccumVisibilityAovs(pbrTls, aovSchema, cameraId, lightAovs,
                                    scene_rdl2::math::Vec2f(0.0f, 1.0f), lpeStateId, aovs)) {
                                addVisibilityAov = false; // add visibility aov at most once per shadow ray
                            }
                        }
                    }
                    
                    if (hasUnoccludedFlag) {
                        scene_rdl2::math::Color unoccludedLobeContribution = lobesContribution.mFs[lobeIndex] * lightContribution;
                        if (pv.nonMirrorDepth >= mSampleClampingDepth) {
                            unoccludedLobeContribution = smartClamp(unoccludedLobeContribution, sp.mSampleClampingValue);
                        }
                        aovAccumLightAovs(pbrTls, aovSchema, cameraId, lightAovs, unoccludedLobeContribution, nullptr, 
                            AovSchema::sLpePrefixUnoccluded, lpeStateId, aovs);
                    }
                }
            }
        }

        return scene_rdl2::math::sBlack;
    }

    scene_rdl2::math::Color unoccludedLightContribution = lightContribution;
    mcrt_common::Ray trRay(isect.getP(), wi, scene_rdl2::math::max(rayEpsilon, shadowRayEpsilon), tfar, ray.getTime(), rayDepth);
    scene_rdl2::math::Color tr = transmittance(pbrTls, trRay, sp.mPixel, sp.mSubpixelIndex, sequenceID, light);
    lightContribution *= tr * (1.0f - presence);

    // Now add in the lobe contributions
    scene_rdl2::math::Color LDirect = scene_rdl2::math::sBlack;
    bool addVisibilityAov = true;
    for (int lobeIndex = 0; lobeIndex < lobesContribution.mMatchedLobeCount; ++lobeIndex) {
        // skip lobe if light is marked as not visible from this lobe
        if (!(lobeTypeToRayMask(lobesContribution.mLobes[lobeIndex]->getType()) & light->getVisibilityMask())) {
            continue;
        }
        scene_rdl2::math::Color lobeContribution = lobesContribution.mFs[lobeIndex] * lightContribution;
        scene_rdl2::math::Color unoccludedLobeContribution = lobesContribution.mFs[lobeIndex] * unoccludedLightContribution;
        // sample clamping gah!
        if (pv.nonMirrorDepth >= mSampleClampingDepth) {
            lobeContribution = smartClamp(lobeContribution, sp.mSampleClampingValue);
            unoccludedLobeContribution = smartClamp(unoccludedLobeContribution, sp.mSampleClampingValue);
        }
        LDirect += lobeContribution;
        // LPE
        if (aovs) {
            EXCL_ACCUMULATOR_PROFILE(pbrTls, EXCL_ACCUM_AOVS);
            const FrameState &fs = *pbrTls->mFs;
            const AovSchema &aovSchema = *fs.mAovSchema;
            const LightAovs &lightAovs = *fs.mLightAovs;
            // transition
            int lpeStateId = pv.lpeStateId;
            lpeStateId = lightAovs.scatterEventTransition(pbrTls, lpeStateId, bsdfOneSampler.getBsdf(),
                *(lobesContribution.mLobes[lobeIndex]));
            lpeStateId = lightAovs.lightEventTransition(pbrTls, lpeStateId, light);

            // Accumulate aovs depending on whether or not the unoccluded flag is set.
            if (hasUnoccludedFlag) {
                // If the unoccluded flag is set we have to add unoccluded and occluded (with presence and volumes) separately.
                aovAccumLightAovs(pbrTls, aovSchema, cameraId, lightAovs, unoccludedLobeContribution, 
                                  &lobeContribution, AovSchema::sLpePrefixUnoccluded, lpeStateId, aovs);
            } else {
                // Otherwise, just add the contribution to all non-pre-occlusion aovs.
                aovAccumLightAovs(pbrTls, aovSchema, cameraId, lightAovs, lobeContribution, nullptr, 
                                  AovSchema::sLpePrefixNone, lpeStateId, aovs);
            }

            // visibilty aov
            if (addVisibilityAov && lightAovs.hasVisibilityEntries()) {
                if (aovAccumVisibilityAovs(pbrTls, aovSchema, cameraId, lightAovs,
                        scene_rdl2::math::Vec2f(reduceTransparency(tr) * (1 - presence), 1.0f), lpeStateId, aovs)) {
                    addVisibilityAov = false; // add visibility aov at most once per shadow ray
                }
            }
        }
    }

    // Debug ray record
    debugRayRecordLightHit(pbrTls, isect.getP(), wi, lightIsect.distance, Li);
    return LDirect;
}

scene_rdl2::math::Color
PathIntegrator::oneSamplerDirectBsdf(pbr::TLState *pbrTls,
    const Subpixel &sp, int cameraId, const PathVertex &pv, const mcrt_common::RayDifferential &ray,
    const shading::Intersection &isect, const Light *light, const LightFilterList *lightFilterList,
    const scene_rdl2::math::Vec3f *cullingNormal, const BsdfOneSampler &bsdfOneSampler, const scene_rdl2::math::Color &pt,
    const float r[9], float rayEpsilon, float shadowRayEpsilon, unsigned sequenceID, float *aovs) const
{
    // r[0:2] used for bsdf sample, r[3] used for russian roulette, r[4:5] used for light filter

    // draw a sample and see if it hits the light
    BsdfOneSampler::LobesContribution lobesContribution;
    scene_rdl2::math::Vec3f wi;
    float pdfBsdf;
    float cr[3] = { r[0], r[1], r[2] }; // need a copy or r since sampleWithPathGuide may modify r[1]
    const scene_rdl2::math::Color f = sampleWithPathGuide(mPathGuide,
                                        bsdfOneSampler.getBsdf().getType() & shading::BsdfLobe::MIRROR,
                                        isect.getP(),
                                        cr[0], cr[1], cr[2],
                                        [&](float r0, float r1, float r2, scene_rdl2::math::Vec3f &dir, float &pdf)
                                        {
                                            return bsdfOneSampler.sample(r0, r1, r2, dir, pdf, &lobesContribution);
                                        },
                                        [&](const scene_rdl2::math::Vec3f &dir, float &pdf)
                                        {
                                            return bsdfOneSampler.eval(dir, pdf, &lobesContribution);
                                        },
                                        wi,
                                        pdfBsdf);

    if (isSampleInvalid(f, pdfBsdf)) {
        return scene_rdl2::math::sBlack;
    }
    LightIntersection lightIsect;
    if (!light->intersect(isect.getP(), cullingNormal, wi, ray.getTime(), scene_rdl2::math::sMaxValue, lightIsect)) {
        return scene_rdl2::math::sBlack;
    }

    // ok we hit the light, add it to the stats, now start the expensive stuff
    // eventhough this was sampled from the bsdf, we still report it as a "light sample".
    // For purposes of statistics, "light samples" mean NEE samples.
    pbrTls->mStatistics.incCounter(STATS_LIGHT_SAMPLES);

    // evaluate the light and get a pdf
    float pdfLight;
    const LightFilterRandomValues rLightFilter = {
        scene_rdl2::math::Vec2f(r[4], r[5]), 
        scene_rdl2::math::Vec3f(r[6], r[7], r[8])};
    const scene_rdl2::math::Color Li = light->eval(pbrTls->mTopLevelTls, wi, isect.getP(), rLightFilter, ray.getTime(), lightIsect,
        /* fromCamera = */ false, lightFilterList, ray.getDirFootprint(), &pdfLight);
    if (isSampleInvalid(Li, pdfLight)) {
        return scene_rdl2::math::sBlack;
    }

    // compute the mis weight
    // mirror lobes require special treatment
    const float misWeight = (lobesContribution.mMatchedLobeCount == 1 &&
        lobesContribution.mLobes[0]->matchesFlag(shading::BsdfLobe::MIRROR)) ?
        1.0 : powerHeuristic(pdfBsdf, pdfLight);

    // total value of this sample
    scene_rdl2::math::Color lightContribution = pt * misWeight * Li / pdfBsdf;
    scene_rdl2::math::Color unoccludedLightContribution = lightContribution;

    // apply russian roulette
    const float lum = luminance(f * lightContribution);
    if (!applyRussianRoulette(r[3], mRussianRouletteThreshold, mInvRussianRouletteThreshold,
                              pv.nonMirrorDepth, lum, lightContribution)) {
        return scene_rdl2::math::sBlack;
    }

    // still here?  ok resolve occlusion
    const float tfar = lightIsect.distance - rayEpsilon;
    const int rayDepth = ray.getDepth() + 1;
    // THINK: Would a standard ray, rather than a ray differential work better?
    // Using a ray differential produces more occlusion (i.e. darker images).
    // Since the multi-sample integrator uses ray differentials, we do so here
    // as well.  It might be worth changing this behavior in both integrators.
    mcrt_common::RayDifferential shadowRay(isect.getP(), wi,
                                           ray.getOriginX(), wi,
                                           ray.getOriginY(), wi,
                                           rayEpsilon, tfar, ray.getTime(), rayDepth);
    float presence = 0.0f;

    int32_t assignmentId = isect.getLayerAssignmentId();
    if (!isRayOccluded(pbrTls, light, shadowRay, rayEpsilon, shadowRayEpsilon, presence, assignmentId)) {
        // not occluded, compute volume transmittance, take presence into account
        // shadowRay can be modified in occlusion query, so make a new one
        mcrt_common::Ray trRay(isect.getP(), wi, scene_rdl2::math::max(rayEpsilon, shadowRayEpsilon), tfar, ray.getTime(), rayDepth);
        scene_rdl2::math::Color tr = transmittance(pbrTls, trRay, sp.mPixel, sp.mSubpixelIndex, sequenceID, light);
        lightContribution *= tr * (1.0f - presence);

        // Now add in the direct lobe contributions
        scene_rdl2::math::Color LDirect = scene_rdl2::math::sBlack;
        for (int lobeIndex = 0; lobeIndex < lobesContribution.mMatchedLobeCount; ++lobeIndex) {
            // skip lobe if light is marked as not visible from this lobe
            if (!(lobeTypeToRayMask(lobesContribution.mLobes[lobeIndex]->getType()) & light->getVisibilityMask())) {
                continue;
            }
            scene_rdl2::math::Color lobeContribution = lobesContribution.mFs[lobeIndex] * lightContribution;
            scene_rdl2::math::Color unoccludedLobeContribution = lobesContribution.mFs[lobeIndex] * unoccludedLightContribution;
            // sample clamping gah!
            if (pv.nonMirrorDepth >= mSampleClampingDepth) {
                lobeContribution = smartClamp(lobeContribution, sp.mSampleClampingValue);
                unoccludedLobeContribution = smartClamp(unoccludedLobeContribution, sp.mSampleClampingValue);
            }
            LDirect += lobeContribution;
            // LPE
            if (aovs) {
                EXCL_ACCUMULATOR_PROFILE(pbrTls, EXCL_ACCUM_AOVS);
                const FrameState &fs = *pbrTls->mFs;
                const AovSchema &aovSchema = *fs.mAovSchema;
                const LightAovs &lightAovs = *fs.mLightAovs;
                // transition
                int lpeStateId = pv.lpeStateId;
                lpeStateId = lightAovs.scatterEventTransition(pbrTls, lpeStateId, bsdfOneSampler.getBsdf(),
                    *(lobesContribution.mLobes[lobeIndex]));
                lpeStateId = lightAovs.lightEventTransition(pbrTls, lpeStateId, light);
                // accumulate matching aovs
                if (fs.mAovSchema->hasLpePrefixFlags(AovSchema::sLpePrefixUnoccluded)) {
                    aovAccumLightAovs(pbrTls, aovSchema, cameraId, lightAovs, unoccludedLobeContribution, &lobeContribution,
                                      AovSchema::sLpePrefixUnoccluded, lpeStateId, aovs);
                } else {
                    aovAccumLightAovs(pbrTls, aovSchema, cameraId, lightAovs, lobeContribution, nullptr,
                                      AovSchema::sLpePrefixNone, lpeStateId, aovs);
                }
            }
        }

        // Debug ray record
        // TODO: why do we store sWhite, rather than Li for direct bsdf samples?
        debugRayRecordLightHit(pbrTls, isect.getP(), wi, lightIsect.distance, scene_rdl2::math::sWhite);

        return LDirect;
    } else {
        // Rcord unocclusion aovs if we have any:
        const FrameState &fs = *pbrTls->mFs;
        const AovSchema &aovSchema = *fs.mAovSchema;
        const LightAovs &lightAovs = *fs.mLightAovs;
        if (aovs && fs.mAovSchema->hasLpePrefixFlags(AovSchema::sLpePrefixUnoccluded)) {
            for (int lobeIndex = 0; lobeIndex < lobesContribution.mMatchedLobeCount; ++lobeIndex) {
                // skip lobe if light is marked as not visible from this lobe
                if (!(lobeTypeToRayMask(lobesContribution.mLobes[lobeIndex]->getType()) & light->getVisibilityMask())) {
                    continue;
                }
                scene_rdl2::math::Color unoccludedLobeContribution = lobesContribution.mFs[lobeIndex] * unoccludedLightContribution;
                // sample clamping gah!
                if (pv.nonMirrorDepth >= mSampleClampingDepth) {
                    unoccludedLobeContribution = smartClamp(unoccludedLobeContribution, sp.mSampleClampingValue);
                }

                // LPE

                // transition
                int lpeStateId = pv.lpeStateId;
                lpeStateId = lightAovs.scatterEventTransition(pbrTls, lpeStateId, bsdfOneSampler.getBsdf(),
                    *(lobesContribution.mLobes[lobeIndex]));
                lpeStateId = lightAovs.lightEventTransition(pbrTls, lpeStateId, light);
                // accumulate matching aovs
                aovAccumLightAovs(pbrTls, aovSchema, cameraId, lightAovs, unoccludedLobeContribution, nullptr,
                                  AovSchema::sLpePrefixUnoccluded, lpeStateId, aovs);
            }
        }
        return scene_rdl2::math::sBlack;
    }
}

scene_rdl2::math::Color
PathIntegrator::oneSamplerDirect(pbr::TLState *pbrTls,
    const Subpixel &sp, int cameraId, const PathVertex &pv, const mcrt_common::RayDifferential &ray,
    const shading::Intersection &isect, const BsdfOneSampler &bsdfOneSampler,
    const LightSet &activeLightSet, const scene_rdl2::math::Vec3f *cullingNormal, float rayEpsilon, float shadowRayEpsilon,
    IntegratorSample1D &rrSamples, unsigned sequenceID, float *aovs) const
{
    // It might not be clear, but this function's outputs are
    // radiance (returned), and aovs (added into).
    scene_rdl2::math::Color radiance = scene_rdl2::math::sBlack;

    const bool highQuality = pv.nonMirrorDepth == 0;
    const int nDirect = highQuality ? mLightSamples : scene_rdl2::math::min(mLightSamples, 1);

    // TODO: we need a strategy to choose a good light subset
    // But that task is outside the scope of the initial one-sampler implementation
    // task.  We have a similar need with the multi-sampler integrator.
    for (int lightIndex = 0; lightIndex < activeLightSet.getLightCount(); ++lightIndex) {
        const Light *light = activeLightSet.getLight(lightIndex);
        const LightFilterList *lightFilterList = activeLightSet.getLightFilterList(lightIndex);

        // Skip ray termination lights
        if (light->getIsRayTerminator()) continue;
        
        // Set up our sample sequences
        // We could use IntegratorSample3D sequences, rather than a 2D and 1D.
        // When I tried this, I found some results in RaTS (such as material/dwalayer/transmission)
        // were a little bit worse.  This is exactly the opposite of the result
        // I saw on the vector side, where using a 3D sampler improved the results and
        // eliminated correlated noise.
        IntegratorSample1D lightSamples1D;
        IntegratorSample2D lightSamples2D;
        IntegratorSample2D lightFilterSamples2D;
        IntegratorSample3D lightFilterSamples3D;
        IntegratorSample1D lobeSamples;
        IntegratorSample2D bsdfSamples;

        const Scene *scene = MNRY_VERIFY(pbrTls->mFs->mScene);
        bool lightFilterNeedsSamples = scene->lightFilterNeedsSamples();

        if (highQuality) {
            // We want one shared sequence for depth 0
            const int samplesSoFar = sp.mSubpixelIndex * nDirect;
            SequenceIDIntegrator lSid(pv.nonMirrorDepth, sp.mPixel, 0,
                        SequenceType::NextEventEstimation,
                        SequenceType::Light, light->getHash(),
                        sequenceID);
            lightSamples1D.resume(lSid, samplesSoFar);
            lightSamples2D.resume(lSid, samplesSoFar);

            if (lightFilterNeedsSamples) {
                SequenceIDIntegrator lFilterSid(pv.nonMirrorDepth, sp.mPixel, 0,
                            SequenceType::NextEventEstimation,
                            SequenceType::LightFilter, light->getHash(),
                            sequenceID);
                lightFilterSamples2D.resume(lFilterSid, samplesSoFar);
                SequenceIDIntegrator lFilter3DSid(pv.nonMirrorDepth, sp.mPixel, 0,
                            SequenceType::NextEventEstimation,
                            SequenceType::LightFilter3D, light->getHash(),
                            sequenceID);
                lightFilterSamples3D.resume(lFilter3DSid, samplesSoFar);
            }

            SequenceIDIntegrator bSid(pv.nonMirrorDepth, sp.mPixel, 0,
                        SequenceType::NextEventEstimation,
                        SequenceType::Bsdf, light->getHash(),
                        sequenceID);
            lobeSamples.resume(bSid, samplesSoFar);
            bsdfSamples.resume(bSid, samplesSoFar);
        } else {
            SequenceIDIntegrator lSid(pv.nonMirrorDepth, sp.mPixel, sp.mSubpixelIndex,
                        SequenceType::NextEventEstimation,
                        SequenceType::Light, light->getHash(),
                        sequenceID);
            lightSamples1D.restart(lSid, nDirect);
            lightSamples2D.restart(lSid, nDirect);

            if (lightFilterNeedsSamples) {
                SequenceIDIntegrator lFilterSid(pv.nonMirrorDepth, sp.mPixel, sp.mSubpixelIndex,
                            SequenceType::NextEventEstimation,
                            SequenceType::LightFilter, light->getHash(),
                            sequenceID);
                lightFilterSamples2D.restart(lFilterSid, nDirect);
                SequenceIDIntegrator lFilter3DSid(pv.nonMirrorDepth, sp.mPixel, sp.mSubpixelIndex,
                            SequenceType::NextEventEstimation,
                            SequenceType::LightFilter3D, light->getHash(),
                            sequenceID);
                lightFilterSamples3D.restart(lFilter3DSid, nDirect);
            }

            SequenceIDIntegrator bSid(pv.nonMirrorDepth, sp.mPixel, sp.mSubpixelIndex,
                        SequenceType::NextEventEstimation,
                        SequenceType::Bsdf, light->getHash(),
                        sequenceID);
            lobeSamples.restart(bSid, nDirect);
            bsdfSamples.restart(bSid, nDirect);
        }

        const scene_rdl2::math::Color pt = pv.pathThroughput / nDirect;

        // Loop over each sample
        for (int s = 0; s < nDirect; ++s) {
            // light sampling
            float r[9];
            lightSamples2D.getSample(&r[0], pv.nonMirrorDepth);
            lightSamples1D.getSample(&r[2], pv.nonMirrorDepth);
            rrSamples.getSample(&r[3], pv.nonMirrorDepth);

            if (lightFilterNeedsSamples) {
                lightFilterSamples2D.getSample(&r[4], pv.nonMirrorDepth);
                lightFilterSamples3D.getSample(&r[6], pv.nonMirrorDepth);
            }

            radiance += oneSamplerDirectLight(pbrTls, sp, cameraId, pv, ray,
                isect, light, lightFilterList, cullingNormal, bsdfOneSampler, pt, r,
                rayEpsilon, shadowRayEpsilon, sequenceID, aovs);

            // bsdf sampling
            lobeSamples.getSample(&r[0], pv.nonMirrorDepth);
            bsdfSamples.getSample(&r[1], pv.nonMirrorDepth);
            rrSamples.getSample(&r[3], pv.nonMirrorDepth);

            if (lightFilterNeedsSamples) {
                lightFilterSamples2D.getSample(&r[4], pv.nonMirrorDepth);
                lightFilterSamples3D.getSample(&r[6], pv.nonMirrorDepth);
            }

            radiance += oneSamplerDirectBsdf(pbrTls, sp, cameraId, pv, ray,
                isect, light, lightFilterList, cullingNormal, bsdfOneSampler, pt, r,
                rayEpsilon, shadowRayEpsilon, sequenceID, aovs);
        }
    }

    return radiance;
}

scene_rdl2::math::Color
PathIntegrator::oneSamplerRayTerminatorLights(pbr::TLState *pbrTls,
    int cameraId, const PathVertex &pv, const mcrt_common::RayDifferential &ray,
    const shading::Intersection &isect, const scene_rdl2::math::Vec3f &wi, float pdfBsdf,
    const shading::BsdfLobe &lobe, const shading::Bsdf &bsdf,
    const LightSet &activeLightSet, const LightFilterRandomValues& lightFilterR,
    const scene_rdl2::math::Vec3f *cullingNormal, const scene_rdl2::math::Color &pt,
    float *aovs) const
{
    // output radiance and aovs (added into)
    scene_rdl2::math::Color radiance = scene_rdl2::math::sBlack;

    // TODO: we need a better way to get just the ray-terminator lights
    for (int lightIndex = 0; lightIndex < activeLightSet.getLightCount(); ++lightIndex) {
        const Light *light = activeLightSet.getLight(lightIndex);
        const LightFilterList *lightFilterList = activeLightSet.getLightFilterList(lightIndex);

        // Skip non ray terminator lights
        if (!light->getIsRayTerminator()) continue;

        // Does the sample direction hit the light?
        LightIntersection lightIsect;
        if (!light->intersect(isect.getP(), cullingNormal, wi, ray.getTime(), scene_rdl2::math::sMaxValue, lightIsect)) {
            continue;
        }

        // It does, evaluate it and get a pdf
        float pdfLight;
        const scene_rdl2::math::Color Li = light->eval(pbrTls->mTopLevelTls, wi, isect.getP(), lightFilterR, ray.getTime(), lightIsect,
            /* fromCamera = */ false, lightFilterList, ray.getDirFootprint(), &pdfLight);

        // Compute the mis weight
        const float misWeight = (lobe.matchesFlag(shading::BsdfLobe::MIRROR)) ?
            1.0f : powerHeuristic(pdfBsdf, pdfLight);

        const scene_rdl2::math::Color contribution = misWeight * Li;
        radiance += contribution;

        // LPE
        if (aovs) {
            EXCL_ACCUMULATOR_PROFILE(pbrTls, EXCL_ACCUM_AOVS);
            const FrameState &fs = *pbrTls->mFs;
            const AovSchema &aovSchema = *fs.mAovSchema;
            const LightAovs &lightAovs = *fs.mLightAovs;
            // transition
            int lpeStateId = pv.lpeStateId;
            lpeStateId = lightAovs.scatterEventTransition(pbrTls, lpeStateId, bsdf, lobe);
            lpeStateId = lightAovs.lightEventTransition(pbrTls, lpeStateId, light);
            // Accumulate matching aovs. Don't have to worry about pre-occlusion aovs.
            aovAccumLightAovs(pbrTls, aovSchema, cameraId, lightAovs, contribution, nullptr,
                              AovSchema::sLpePrefixNone, lpeStateId, aovs);
        }
    }

    return radiance;
}

scene_rdl2::math::Color
PathIntegrator::computeRadianceBsdfOneSampler(pbr::TLState *pbrTls,
    const Subpixel &sp, int cameraId, const PathVertex &pv, const mcrt_common::RayDifferential &ray,
    const shading::Intersection &isect, const shading::Bsdf &bsdf, const shading::BsdfSlice &slice,
    bool doIndirect, const shading::BsdfLobe::Type indirectFlags, const scene_rdl2::rdl2::Material *newPriorityList[4],
    int newPriorityListCount[4], const LightSet &activeLightSet, const scene_rdl2::math::Vec3f *cullingNormal,
    bool hasRayTerminatorLights, float rayEpsilon, float shadowRayEpsilon, const scene_rdl2::math::Color &ssAov,
    unsigned &sequenceID, float *aovs) const
{
    // It might not be clear, but this function's outputs are
    // radiance (returned) and aovs (added into).
    // Additionally, the sequenceID is updated as needed by the sampling framework
    scene_rdl2::math::Color radiance = scene_rdl2::math::sBlack;

    // we are responsible for filling out material aovs
    if (aovs) {
        const FrameState &fs = *pbrTls->mFs;
        const AovSchema &aovSchema = *fs.mAovSchema;
        const LightAovs &lightAovs = *fs.mLightAovs;
        const MaterialAovs &materialAovs = *fs.mMaterialAovs;
        const Scene &scene = *fs.mScene;

        // Since we don't produce BsdfSample objects like multi-sample,
        // we can pack the material aovs now.  The albedo aov will use the slice
        // and the lobe albedo methods.
        aovSetMaterialAovs(pbrTls, aovSchema, cameraId, lightAovs, materialAovs, isect,
            ray, scene, bsdf, ssAov, &slice, pv.aovPathPixelWeight, pv.lpeStateId, aovs);
    }

    const BsdfOneSampler bsdfOneSampler(bsdf, slice);

    // Both direct and indirect sampling share russian roulette samples
    IntegratorSample1D rrSamples(SequenceID(
                sp.mPixel, sp.mSubpixelIndex, pv.nonMirrorDepth,
                SequenceType::RussianRouletteBsdf, sequenceID));

    // First do NEE
    const bool doNEE = mLightSamples != 0;
    if (doNEE) {
        radiance += oneSamplerDirect(pbrTls, sp, cameraId, pv, ray, isect, bsdfOneSampler,
            activeLightSet, cullingNormal, rayEpsilon, shadowRayEpsilon, rrSamples, sequenceID, aovs);
    }

    // We can exit early if doIndirect is false and we have no ray terminator lights
    if (!doIndirect && !hasRayTerminatorLights) {
        return radiance;
    }

    // Spawn new rays through bsdf sampling
    // Respect the user bsdf samples up to the first non-mirror bounce.
    // After that, we spawn only a single bsdf sample.  If the bsdf
    // contains mirror lobes, we ignore bsdf samples.
    const bool highQuality = pv.nonMirrorDepth == 0 &&
        !(bsdf.getType() & shading::BsdfLobe::MIRROR);
    const int nIndirect = highQuality ? mBsdfSamples : scene_rdl2::math::min(mBsdfSamples, 1);

    IntegratorSample1D lobeSamples; // picks the lobe
    IntegratorSample2D bsdfSamples; // samples bsdf
    IntegratorSample2D lightFilterSamples; // samples light filter
    IntegratorSample3D lightFilterSamples3D; // samples light filter

    const Scene *scene = MNRY_VERIFY(pbrTls->mFs->mScene);
    bool lightFilterNeedsSamples = scene->lightFilterNeedsSamples();

    if (highQuality) {
        // We want one shared sequence for depth 0
        int samplesSoFar = sp.mSubpixelIndex * nIndirect;
        SequenceIDIntegrator bSid(
            pv.nonMirrorDepth, sp.mPixel, 0,
            SequenceType::IndirectLighting,
            SequenceType::Bsdf, sequenceID);
        lobeSamples.resume(bSid, samplesSoFar);
        bsdfSamples.resume(bSid, samplesSoFar);

        if (lightFilterNeedsSamples) {
            SequenceIDIntegrator bFilterSid(
                pv.nonMirrorDepth, sp.mPixel, 0,
                SequenceType::IndirectLighting,
                SequenceType::LightFilter, sequenceID);
            lightFilterSamples.resume(bFilterSid, samplesSoFar);
            SequenceIDIntegrator bFilter3DSid(
                pv.nonMirrorDepth, sp.mPixel, 0,
                SequenceType::IndirectLighting,
                SequenceType::LightFilter3D, sequenceID);
            lightFilterSamples3D.resume(bFilter3DSid, samplesSoFar);
        }
    } else {
        SequenceIDIntegrator bSid(
            pv.nonMirrorDepth, sp.mPixel, sp.mSubpixelIndex,
            SequenceType::IndirectLighting,
            SequenceType::Bsdf, sequenceID);
        lobeSamples.restart(bSid, nIndirect);
        bsdfSamples.restart(bSid, nIndirect);

        if (lightFilterNeedsSamples) {
            SequenceIDIntegrator bFilterSid(
                pv.nonMirrorDepth, sp.mPixel, sp.mSubpixelIndex,
                SequenceType::IndirectLighting,
                SequenceType::LightFilter, sequenceID);
            lightFilterSamples.restart(bFilterSid, nIndirect);
            SequenceIDIntegrator bFilter3DSid(
                pv.nonMirrorDepth, sp.mPixel, sp.mSubpixelIndex,
                SequenceType::IndirectLighting,
                SequenceType::LightFilter3D, sequenceID);
            lightFilterSamples3D.restart(bFilter3DSid, nIndirect);
        }
    }

    // loop over samples
    for (int s = 0; s < nIndirect; ++s) {
        float r[9]; // r[0] lobe select, r[1:2] bsdf, r[3] russian roulette, r[4:5] light filter, r[6:9] light filter 3D
        lobeSamples.getSample(&r[0], pv.nonMirrorDepth);
        bsdfSamples.getSample(&r[1], pv.nonMirrorDepth);

        if (lightFilterNeedsSamples) {
            lightFilterSamples.getSample(&r[4], pv.nonMirrorDepth);
            lightFilterSamples3D.getSample(&r[6], pv.nonMirrorDepth);
        }

        scene_rdl2::math::Vec3f wi;
        float pdfBsdf;
        scene_rdl2::math::Color f;
        // The sampled lobe will be the one we use for all lobe dependent
        // decisions, such as path depth, aovs, etc...
        const shading::BsdfLobe *lobe = nullptr;
        if (mBsdfSamplerStrategy == BSDF_SAMPLER_STRATEGY_ONE_LOBE) {
            float pdfLobe; // probability of picking this lobe
            lobe = bsdfOneSampler.sampleLobe(r[0], pdfLobe);
            if (!lobe || pdfLobe == 0.0f) {
                continue;
            }
            f = sampleWithPathGuide(mPathGuide,
                                    lobe->getType() & shading::BsdfLobe::MIRROR,
                                    isect.getP(),
                                    r[0], r[1], r[2],
                                    [&](float /*r0*/, float r1, float r2, scene_rdl2::math::Vec3f &dir, float &pdf)
                                    {
                                        return lobe->sample(slice, r1, r2, dir, pdf);
                                    },
                                    [&](const scene_rdl2::math::Vec3f &dir, float &pdf)
                                    {
                                        return lobe->eval(slice, dir, &pdf);
                                    },
                                    wi,
                                    pdfBsdf);
            pdfBsdf *= pdfLobe; // include probability of selecting lobe

            // We are responsible for checking if this lobe matches the flags
            shading::BsdfLobe::Type flags = slice.getSurfaceFlags(bsdf, wi);
            if (!lobe->matchesFlags(flags)) {
                f = scene_rdl2::math::sBlack;
                pdfBsdf = 0.f;
            }
        } else {
            MNRY_ASSERT(mBsdfSamplerStrategy == BSDF_SAMPLER_STRATEGY_ONE_SAMPLE);
            BsdfOneSampler::LobesContribution lobesContribution;
            f = sampleWithPathGuide(mPathGuide,
                                    bsdf.getType() & shading::BsdfLobe::MIRROR,
                                    isect.getP(),
                                    r[0], r[1], r[2],
                                    [&](float r0, float r1, float r2, scene_rdl2::math::Vec3f &dir, float &pdf)
                                    {
                                        return bsdfOneSampler.sample(r0, r1, r2, dir, pdf, &lobesContribution);
                                    },
                                    [&](const scene_rdl2::math::Vec3f &dir, float &pdf)
                                    {
                                        return bsdfOneSampler.eval(dir, pdf, &lobesContribution);
                                    },
                                    wi,
                                    pdfBsdf);
            if (!lobesContribution.mMatchedLobeCount) {
                continue;
            }
            // The chosen lobe is always at index 0.
            lobe = lobesContribution.mLobes[0];
        }

        if (!isSampleValid(f, pdfBsdf)) {
            continue;
        }

        // Compute the total throughput potential for this sample.
        // Terminate the sample if it's throughput is 0.
        scene_rdl2::math::Color pt = f / (pdfBsdf * nIndirect);
        pt *= pv.pathThroughput;
        float lum = luminance(pt);

        // Apply russian roulette
        rrSamples.getSample(&r[3], pv.nonMirrorDepth);
        if (!applyRussianRoulette(r[3], mRussianRouletteThreshold, mInvRussianRouletteThreshold,
                                  pv.nonMirrorDepth, lum, pt)) {
            continue;
        }

        // Terminate path due to depth limits.
        // FIXME: we should not need to special case the hair lobe depth check.
        // Make it work with indirectFlags.
        const bool hairLobe = lobe->getIsHair();
        if (!doIndirect || !lobe->matchesFlags(indirectFlags)) {
            // The lobe flags say the depth limit has been meet.  But
            // if we are a hair lobe, me may need to keep going anyway.
            if (!hairLobe || (pv.hairDepth >= mMaxHairDepth)) {
                // apply ray terminator lights to non-hair transmission lobes
                if (hasRayTerminatorLights && !hairLobe &&
                    lobe->matchesFlags(shading::BsdfLobe::ALL_TRANSMISSION)) {
                    LightFilterRandomValues lightFilterR = { 
                        scene_rdl2::math::Vec2f(r[4], r[5]), 
                        scene_rdl2::math::Vec3f(r[6], r[7], r[8])};
                    radiance += oneSamplerRayTerminatorLights(pbrTls, cameraId, pv, ray, isect, wi, pdfBsdf,
                        *lobe, bsdf, activeLightSet, lightFilterR, cullingNormal, pt, aovs);
                }
                continue; // terminate the path
            }
        }

        // At this point, count this sample in the stats
        Statistics &stats = pbrTls->mStatistics;
        stats.incCounter(STATS_BSDF_SAMPLES);

        const bool diffuseLobe = lobe->matchesFlags(shading::BsdfLobe::ALL_DIFFUSE);
        const bool glossyLobe = lobe->matchesFlags(shading::BsdfLobe::ALL_GLOSSY);
        const bool mirrorLobe = lobe->matchesFlags(shading::BsdfLobe::ALL_MIRROR);
        const bool transmissionLobe = lobe->matchesFlags(shading::BsdfLobe::ALL_TRANSMISSION);

        // Check transparency threshold
        float newAccumOpacity;
        if (mirrorLobe && transmissionLobe) {
            float lobeTransparency = reduceTransparency(f);
            newAccumOpacity = pv.accumOpacity + (1 - lobeTransparency) * (1 - pv.accumOpacity);
            if (newAccumOpacity > mTransparencyThreshold) {
                continue;
            }
        } else {
            newAccumOpacity = pv.accumOpacity;
        }

        // Prepare a path vertex
        PathVertex nextPv;
        nextPv.pathThroughput = pt;
        nextPv.pathPixelWeight = 0.0f;
        // Use previous path pixel weight for aovPathPixelWeight as there's existing logic
        // in vector mode that sometimes assumes that pv.pathPixelWeight = 0.  Thus, we must seperately
        // keep track of the pathPixelWeight for aovs.  See comment in PathIntegratorMultiSampler.ispc::
        // addIndirectOrDirectVisibleContributionsBundled().
        nextPv.aovPathPixelWeight = pv.pathPixelWeight;
        nextPv.pathDistance = pv.pathDistance + ray.getEnd();
        nextPv.minRoughness = shading::computeMinRoughness(*lobe, mRoughnessClampingFactor, pv.minRoughness);
        nextPv.diffuseDepth = pv.diffuseDepth + (diffuseLobe ? 1 : 0);
        nextPv.glossyDepth = pv.glossyDepth + (glossyLobe ? 1 : 0);
        nextPv.mirrorDepth = pv.mirrorDepth + (mirrorLobe ? 1 : 0);
        nextPv.nonMirrorDepth = pv.nonMirrorDepth + (mirrorLobe ? 0 : 1);
        nextPv.hairDepth = pv.hairDepth + (hairLobe ? 1 : 0);
        nextPv.volumeDepth = pv.volumeDepth + 1;
        nextPv.presenceDepth = pv.presenceDepth;
        nextPv.subsurfaceDepth = pv.subsurfaceDepth;
        nextPv.accumOpacity = newAccumOpacity;
        nextPv.lobeType = lobe->getType();

        // LPE
        // TODO: Big issue when doing BSDF_SAMPLER_STRATEGY_ONE_SAMPLE.
        // We can only set a single scatter transition event,
        // so we'll use the selected lobe - but f contains the evaluation of the all lobes.
        // so the aovs will bleed into each other.
        if (aovs) {
            const FrameState &fs = *pbrTls->mFs;
            const LightAovs &lightAovs = *fs.mLightAovs;
            // transition
            nextPv.lpeStateId = lightAovs.scatterEventTransition(pbrTls, pv.lpeStateId, bsdf, *lobe);
            // Accumulate post scatter extra aovs
            aovAccumPostScatterExtraAovs(pbrTls, fs, pv, cameraId, bsdf, aovs);
        }

        // We have some self-intersections when rays leave at grazing
        // angle, so we adjust the rayEpsilon accordingly.
        const float denom = scene_rdl2::math::abs(dot(isect.getNg(), wi));
        // isect.getNg() itself or the dot product above can be zero.
        const float start = scene_rdl2::math::isZero(denom) ? rayEpsilon : rayEpsilon / denom;
        // For NEE to work properly with de-coupled direct and indirect
        // samples we must skip samples that would have been counted if sampled
        // as direct bsdf samples.  So that means we need to find the shortest
        // distance to an active light and set the end of our ray to this distance.
        // TODO: find a faster way to get the shortest distance to a light
        // Perhaps the light accelerator could add an intersectNearestLight function.
        // Note that because we update the ray end with the distance to the
        // nearest light, we eliminate the possibility of intersecting geometry
        // that is beyond the nearest light.
        float end = scene_rdl2::math::sMaxValue;
        int hitLightIndex = -1;
        LightIntersection hitLightIsect;
        if (doNEE) {
            for (int lightIndex = 0; lightIndex < activeLightSet.getLightCount(); ++lightIndex) {
                const Light *light = activeLightSet.getLight(lightIndex);
                // skip ray termination lights
                if (light->getIsRayTerminator()) continue;
                if (light->intersect(isect.getP(), cullingNormal, wi, ray.getTime(), end, hitLightIsect)) {
                    MNRY_ASSERT(hitLightIsect.distance <= end);
                    end = hitLightIsect.distance;
                }
            }
            if (end <= start) {
                continue;
            }
        } else {
            // we are not doing NEE, so we must include a light evaluation if we happen
            // to hit a light.  We'll select a light at random along our ray path
            // and set our ray end to this distance.
            int lobeIndex = 0;
            for (; lobeIndex < bsdf.getLobeCount(); ++lobeIndex) {
                if (bsdf.getLobe(lobeIndex) == lobe) break;
            }
            SequenceIDIntegrator lightChoiceSid(pv.nonMirrorDepth,
                        sp.mPixel,
                        lobeIndex,
                        SequenceType::IndexSelection,
                        sp.mSubpixelIndex,
                        sequenceID);
            IntegratorSample1D lightChoiceSamples(lightChoiceSid);
            int numHits;
            hitLightIndex = activeLightSet.intersect(isect.getP(), cullingNormal, wi,
                ray.getTime(), end, /* includeRayTerminationLights = */ false, lightChoiceSamples,
                pv.nonMirrorDepth, lobeTypeToRayMask(lobe->getType()), hitLightIsect, numHits);
            if (hitLightIndex != -1) {
                MNRY_ASSERT(hitLightIndex < activeLightSet.getLightCount());
                end = hitLightIsect.distance;
            }
        }
        // Prepare a RayDifferential
        mcrt_common::RayDifferential nextRay(ray, start, end);
        if (transmissionLobe) {
            // copy in the new priority list into the ray
            setPriorityList(nextRay, newPriorityList, newPriorityListCount);
        }
        // Scatter and scale our next ray differential
        // We scatter the ray based on the sampled lobe
        float scale = (lobe->getDifferentialFlags() & shading::BsdfLobe::IGNORES_INCOMING_DIFFERENTIALS) ?
            sp.mPrimaryRayDiffScale : 1.0f;
        scatterAndScale(isect, *lobe, -ray.dir, wi, scale, r[1], r[2], nextRay);

        // Recurse
        scene_rdl2::math::Color radianceIndirect;
        float transparencyIndirect;
        // the volume attenuation along this ray to the first hit (or infinity)
        VolumeTransmittance vtIndirect;
        ++sequenceID;
        bool hitVolume;
        bool isStereoscopic = false;
        IndirectRadianceType result = computeRadianceRecurse(pbrTls, nextRay, nextRay, sp, cameraId, nextPv,
            lobe, radianceIndirect, transparencyIndirect,
            vtIndirect, sequenceID, aovs, /* depth = */ nullptr, /* deepParams = */ nullptr,
            /* cryptomatteParams = */ nullptr, /* refractCryptomatteParams = */ nullptr, /* ignoreVolumes = */ false, hitVolume,
            isStereoscopic);
        if (result != NONE) {
            radiance += radianceIndirect;

            // path guiding is trained on indirect radiance contributions.
            // it turns out that including any other radiance contributions in the
            // training causes the path guide to overwhelmingly favor direct lighting
            // directions.  this is the exact opposite of what we want.  we are trying
            // to build a distribution that favors important indirect lighting.
            mPathGuide.recordRadiance(nextRay.getOrigin(), nextRay.getDirection(), radianceIndirect);
        } else {
            // We didn't bounce
            // If we are not doing NEE, and our ray hit a light, we must
            // add contribution now
            if (!doNEE && hitLightIndex != -1) {
                const Light *hitLight = activeLightSet.getLight(hitLightIndex);
                const LightFilterList *hitLightFilterList = activeLightSet.getLightFilterList(hitLightIndex);
                float pdfLight;
                const LightFilterRandomValues lightFilterR = { 
                    scene_rdl2::math::Vec2f(r[4], r[5]), 
                    scene_rdl2::math::Vec3f(r[6], r[7], r[8])}; 
                const scene_rdl2::math::Color Li = hitLight->eval(pbrTls->mTopLevelTls, wi, isect.getP(), lightFilterR, ray.getTime(), 
                    hitLightIsect, /* fromCamera = */ false, hitLightFilterList, ray.getDirFootprint(), &pdfLight);                   
                const scene_rdl2::math::Color contribution = Li * pt;
                radiance += contribution;
                // LPE
                if (aovs) {
                    EXCL_ACCUMULATOR_PROFILE(pbrTls, EXCL_ACCUM_AOVS);
                    const FrameState &fs = *pbrTls->mFs;
                    const AovSchema &aovSchema = *fs.mAovSchema;
                    const LightAovs &lightAovs = *fs.mLightAovs;
                    // transition
                    int lpeStateId = nextPv.lpeStateId; // already includes the scatter event
                    lpeStateId = lightAovs.lightEventTransition(pbrTls, lpeStateId, hitLight);
                    // Accumulate matching aovs. Don't have to worry about pre-occlusion aovs.
                    aovAccumLightAovs(pbrTls, aovSchema, cameraId, lightAovs, contribution, nullptr, 
                                      AovSchema::sLpePrefixNone, lpeStateId, aovs);
                }

                // Debug ray record
                debugRayRecordLightHit(pbrTls, isect.getP(), wi, hitLightIsect.distance, Li);
            }
        }
    }


    return radiance;
}

} // namespace pbr
} // namespace moonray

