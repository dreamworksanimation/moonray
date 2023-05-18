// Copyright 2023 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

/// @file PathIntegratorOneSampler.ispc

// Integration implementation based on the BsdfOneSampler

// When making changes to this file, you'll likely also need
// to update the scalar implementation and the multi-sample
// integrator (both vector and scalar).
//   PathIntegratorOneSampler.cc
//   PathIntegratorMultiSampler.cc
//   PathIntegratorMultiSampler.ispc

#include "PathIntegrator.isph"

#include "BsdfOneSampler.isph"
#include "PathIntegratorUtil.isph"
#include <moonray/rendering/pbr/Types.isph>
#include <moonray/rendering/pbr/core/Aov.isph>
#include <moonray/rendering/pbr/core/PbrTLState.isph>
#include <moonray/rendering/pbr/core/RayState.isph>
#include <moonray/rendering/pbr/core/Statistics.isph>
#include <moonray/rendering/pbr/core/Util.isph>
#include <moonray/rendering/pbr/light/LightSet.isph>
#include <moonray/rendering/pbr/sampler/SequenceID.isph>

// I started encountering this strange ISPC bug
// where code would continue to execute even if the
// lanemask was 0.  Ideally this check would not be needed
#define CHECK_LANEMASK_AND_CONTINUE \
    if (lanemask() == 0) continue;  \
    MNRY_ASSERT(lanemask() != 0);


struct Scene;
extern "C" uniform bool CPP_LightFilterNeedsSamples(const uniform Scene *uniform scene);

// Helper function that applies russian roulette.
// Determine to either cull c or keep and bump of c's contribution.
// Returns true if c is kept (and bumped up)
// Returns false if the sample should be culled
static varying bool
applyRussianRoulette(varying float r,
                     uniform float threshold,
                     uniform float invThreshold,
                     varying int nonMirrorDepth,
                     varying float lum,
                     varying Color &c)
{
    // The direct comparison against 0.0f is intentional.
    // A threshold of exactly zero means that russian roulette is off.
    // But we always want to cull 0 luminance samples.
    if (lum == 0.0f) {
        return false; // cull
    }

    if (nonMirrorDepth > 0 && lum < threshold) {
        const float continueProbability = max(sEpsilon, lum * invThreshold);
        if (r > continueProbability) {
            return false; // cull
        }
        c = c * rcp(continueProbability); // bump it up
    }

    return true; // keep it
}

static void
oneSamplerSubmitOcclusionRays(const uniform PathIntegrator * uniform this,
                              uniform PbrTLState * uniform pbrTls,
                              const varying RayState * uniform rs,
                              const varying Intersection &isect,
                              const uniform Light * uniform light,
                              const varying Bsdf &bsdf,
                              varying float rayEpsilon,
                              varying float shadowRayEpsilon,
                              varying uint32_t sequenceID,
                              const varying LightIntersection &lightIsect,
                              const varying Color &lightContribution,
                              const uniform LobesContribution * varying lobesContribution,
                              const varying Vec3f &wi,
                              varying OcclTestType occlTestType)
{
    // This is a rather complicated funcion with different repsonsibilities.
    // Given light and lobe contributions, compute a total direct lighting value along
    // with all contributing aovs. This includes taking rather unrelated ad-hoc features
    // into account as well such as sample clamping and light visiblity masks.
    // Finally, build an occlusion ray from this information (from isect along wi)
    // and submit it to the occlusion queue.
    Color LDirect = sBlack;
    varying uint32_t data = PBR_nullHandle;
    const uniform FrameState &fs = *pbrTls->mFs;
    const uniform LightAovs * uniform lightAovs = nullptr;
    const uniform AovSchema &aovSchema = *((const uniform AovSchema * uniform) fs.mAovSchema);
    if (!AovSchema_empty(aovSchema)) {
        lightAovs = (const uniform LightAovs * uniform) pbrTls->mFs->mLightAovs;
        if (LightAovs_hasEntries(lightAovs)) {
            // allocate data to handle aovs
            data = PbrTLState_allocList(pbrTls, sizeof(uniform BundledOcclRayData), lobesContribution->mMatchedLobeCount);
            uniform BundledOcclRayData * varying b =
                (uniform BundledOcclRayData * varying) PbrTLState_getListItem(pbrTls, data, 0);
            BundledOcclRayData_init(b, LPE | LIGHT_SAMPLE, rs->mCameraId, light);
        }
    }
    if (data == PBR_nullHandle) {
        // need 1 data item allocated for non-aov cases
        data = PbrTLState_allocList(pbrTls, sizeof(uniform BundledOcclRayData), 1);
        uniform BundledOcclRayData * varying b =
            (uniform BundledOcclRayData * varying) PbrTLState_getListItem(pbrTls, data, 0);
        BundledOcclRayData_init(b, /* flags = */ 0, rs->mCameraId, light);
    }
    uint32_t currAov = 0;
    for (int lobeIndex = 0; lobeIndex < lobesContribution->mMatchedLobeCount; ++lobeIndex) {
        // skip lobe if light is marked as not visible from this lobe
        // FIXME: not sure why ispc complains when using this more obivous syntax:
        //     const varying BsdfLobe * varying lobe = lobesContribution->mLobes[lobeIndex];
        const varying BsdfLobe * varying lobe = *(&lobesContribution->mLobes[0] + lobeIndex);
        const uniform int visibilityMask = Light_getVisibilityMask(light);
        const int bsdfLobeType = (int) BsdfLobe_getType(lobe);
        if (!(bsdfLobeType & visibilityMask)) continue;

        Color lobeContribution = lobesContribution->mFs[lobeIndex] * lightContribution;
        // sample clamping gah!
        const varying Subpixel &sp = rs->mSubpixel;
        const PathVertex &pv = rs->mPathVertex;
        if (pv.nonMirrorDepth >= this->mSampleClampingDepth) {
            lobeContribution = smartClamp(lobeContribution, sp.mSampleClampingValue);
        }
        LDirect = LDirect + lobeContribution;
        if (lightAovs) {
            // transition
            int lpeStateId = pv.lpeStateId;
            foreach_unique(lobePtr in lobe) {
                lpeStateId = LightAovs_scatterEventTransition(lightAovs, lpeStateId, bsdf, *lobePtr);
            }
            if (lpeStateId >= 0) {
                lpeStateId = LightAovs_lightEventTransition(lightAovs, lpeStateId, light);
            }
            // FIXME: ISPC bug? why does this continue statement not work?
            // if (lpeStateId < 0) {
            //    continue;
            // }
            if (lpeStateId >= 0) {
                // fill out currAov
                uniform BundledOcclRayData * varying b =
                    (uniform BundledOcclRayData * varying) PbrTLState_getListItem(pbrTls, data, currAov);
                b->mLpeRadiance = lobeContribution;
                b->mLpeStateId = lpeStateId;

                // initialize next aov, if we have one
                ++currAov;
                if (currAov < lobesContribution->mMatchedLobeCount) {
                    b = (uniform BundledOcclRayData * varying) PbrTLState_getListItem(pbrTls, data, currAov);
                    BundledOcclRayData_init(b, (uniform uint32_t) (LPE | LIGHT_SAMPLE), rs->mCameraId, light);
                }
            }
        }
    }
    rayEpsilon = max(rayEpsilon, shadowRayEpsilon);
    // Add presence shadow specific items to the first data item, if needed
    const RayDifferential &ray = rs->mRay;
    if (light->mPresenceShadows) {
        uniform BundledOcclRayData * varying b =
            (uniform BundledOcclRayData * varying) PbrTLState_getListItem(pbrTls, data, 0);
        b->mRayEpsilon = rayEpsilon;
    }

    // resolve occlusion
    BundledOcclRay shadowRay;
    const float maxT = lightIsect.distance * sHitEpsilonEnd;
    const float minT = rayEpsilon < maxT ? rayEpsilon : maxT;
    const int rayDepth = Ray_getDepth(ray) + 1;
    const float rayTime = Ray_getTime(ray);
    PbrTLState_acquireDeepData(pbrTls, rs->mDeepDataHandle);
    PbrTLState_acquireCryptomatteData(pbrTls, rs->mCryptomatteDataHandle);
    PbrTLState_acquireCryptomatteData2(pbrTls, rs->mCryptomatteDataHandle2);
    const varying Subpixel &sp = rs->mSubpixel;
    int32_t assignmentId = isect.mLayerAssignmentId;
    BundledOcclRay_init(&shadowRay,
                        getP(isect),
                        wi,
                        minT,
                        maxT,
                        rayTime,
                        rayDepth,
                        LDirect,
                        sp.mPixel,
                        sp.mSubpixelIndex,
                        sequenceID,
                        rs->mTilePassAndFilm,
                        data,
                        rs->mDeepDataHandle,
                        rs->mCryptomatteDataHandle,
                        rs->mCryptomatteDataHandle2,
                        occlTestType,
                        assignmentId);
    // packed_store_active2() "may write one extra element to the output array"
    // according to ISPC documentation https://ispc.github.io/ispc.html.
    // Please make sure that 1 extra element was allocated for the output array.
    uniform uint32_t activeLaneIndices[VLEN + 1];
    uniform uint32_t numEntries = packed_store_active2(activeLaneIndices, programIndex);
    snapshotLaneUtilization(pbrTls->mStatistics, STATS_VEC_FILL_OCCL_RAY);
    if (numEntries > 0) {
        shadowRay.mMaxT = max(shadowRay.mMaxT - light->mClearRadius, rayEpsilon);
        if (light->mMaxShadowDistance > 0) {
            shadowRay.mMaxT = min(shadowRay.mMaxT, light->mMaxShadowDistance);
        }
        if (light->mPresenceShadows) {
            CPP_addPresenceShadowsQueueEntries(pbrTls, &shadowRay, numEntries, activeLaneIndices);
        } else {
            CPP_addOcclusionQueueEntries(pbrTls, &shadowRay, numEntries, activeLaneIndices);
        }
    } else {
        // free the data handle, since it wasn't turned over to the ray handler
        PbrTLState_freeList(pbrTls, data);

        // TODO: should we reduce the ref count on the deep data?
    }
}


static void
oneSamplerDirectLight(const uniform PathIntegrator * uniform this,
                      uniform PbrTLState * uniform pbrTls,
                      uniform ShadingTLState * uniform shadingTls,
                      const varying RayState * uniform rs,
                      const varying Intersection &isect,
                      const uniform Light * uniform light,
                      const uniform LightFilterList * uniform lightFilterList,
                      const varying Vec3f &cullingNormal,
                      const varying BsdfOneSampler &bsdfOneSampler,
                      const Color &pt,
                      const float r[9],
                      const float rayEpsilon,
                      const float shadowRayEpsilon,
                      varying uint32_t sequenceID)
{
    // r[0:2] used for light sample, r[3] used for russian roulette, r[4:5] for the light filter

    // Sample light direction
    LightIntersection lightIsect;
    Vec3f wi;
    const RayDifferential &ray = rs->mRay;
    const varying Vec3f rVec = Vec3f_ctor(r[0], r[1], r[2]);
    bool isValidForVisAov = false;
    if (!Light_sample(light, getP(isect), cullingNormal, Ray_getTime(ray), rVec, wi, lightIsect,
                      Ray_getDirFootprint(ray), isValidForVisAov)) {
        return;
    }


    // Evaluate light radiance and pdf
    float lightPdf = 0.0f;
    const varying LightFilterRandomValues rLightFilter = {Vec2f_ctor(r[4], r[5]), Vec3f_ctor(r[6], r[7], r[8])};
    const Color Li = Light_eval(light,
                                shadingTls,
                                wi,
                                getP(isect),
                                rLightFilter,
                                Ray_getTime(ray),
                                lightIsect,
                                /* fromCamera = */ false,
                                lightFilterList,
                                Ray_getDirFootprint(ray),
                                &lightPdf);
    if (isSampleInvalid(Li, lightPdf)) {
        return;
    }

    // sample is valid - count it
    addToCounter(pbrTls->mStatistics, STATS_LIGHT_SAMPLES, getActiveLaneCount());

    // evaluate the bsdf, we'll need individual lobe contributions for aovs
    uniform LobesContribution lobesContributionData[programCount];
    uniform LobesContribution * varying lobesContribution = &lobesContributionData[programIndex];
    float bsdfPdf;
    const Color f = BsdfOneSampler_eval(&bsdfOneSampler, wi, bsdfPdf, lobesContribution);
    if (isSampleInvalid(f, bsdfPdf)) {
        return;
    }

    // TODO: path guide

    const float misWeight = powerHeuristic(lightPdf, bsdfPdf);

    // Compute light contribution, taking path throughput into account
    Color lightContribution = pt * misWeight * Li * rcp(lightPdf);

    // Apply RR based on luminance
    const float lum = luminance(f * lightContribution);
    const PathVertex &pv = rs->mPathVertex;
    if (!applyRussianRoulette(r[3],
                              this->mRussianRouletteThreshold,
                              this->mInvRussianRouletteThreshold,
                              pv.nonMirrorDepth,
                              lum,
                              lightContribution)) {
        return;
    }

    // resolve occlusion
    oneSamplerSubmitOcclusionRays(this,
                                  pbrTls,
                                  rs,
                                  isect,
                                  light,
                                  BsdfOneSampler_getBsdf(&bsdfOneSampler),
                                  rayEpsilon,
                                  shadowRayEpsilon,
                                  sequenceID,
                                  lightIsect,
                                  lightContribution,
                                  lobesContribution,
                                  wi,
                                  STANDARD);

}

static void
oneSamplerDirectBsdf(const uniform PathIntegrator * uniform this,
                     uniform PbrTLState * uniform pbrTls,
                     uniform ShadingTLState * uniform shadingTls,
                     const varying RayState * uniform rs,
                     const varying Intersection &isect,
                     const uniform Light * uniform light,
                     const uniform LightFilterList * uniform lightFilterList,
                     const varying Vec3f &cullingNormal,
                     const varying BsdfOneSampler &bsdfOneSampler,
                     const Color &pt,
                     const float r[9],
                     const float rayEpsilon,
                     const float shadowRayEpsilon,
                     varying uint32_t sequenceID)
{
    // r[0] lobe select, r[1:2] bsdf sample, r[3] used for russian roulette, r[4:5] used for light filter

    // draw a sample and see if it hits the light
    uniform LobesContribution lobesContributionData[programCount];
    uniform LobesContribution * varying lobesContribution = &lobesContributionData[programIndex];
    Vec3f wi;
    float pdfBsdf;
    Color f;
    // TODO: path guide
    f = BsdfOneSampler_sample(&bsdfOneSampler, r[0], r[1], r[2], wi, pdfBsdf, lobesContribution);
    if (isSampleInvalid(f, pdfBsdf)) {
        return;
    }
    LightIntersection lightIsect;
    const RayDifferential &ray = rs->mRay;
    const float rayTime = Ray_getTime(ray);
    const Vec3f &P = getP(isect);
    if (!Light_intersect(light, P, cullingNormal, wi, rayTime, sMaxValue, lightIsect)) {
        return;
    }

    // ok we hit the light, add it to the stats, now start the expensive stuff
    // eventhough this was sampled from the bsdf, we still report it as a "light sample".
    // For purposes of statistics, "light samples" mean NEE samples.
    addToCounter(pbrTls->mStatistics, STATS_LIGHT_SAMPLES, getActiveLaneCount());

    // evaluate the light and get a pdf
    float pdfLight;
    
    const varying LightFilterRandomValues rLightFilter = {Vec2f_ctor(r[4], r[5]), Vec3f_ctor(r[6], r[7], r[8])};
    const Color Li = Light_eval(light, shadingTls, wi, P, rLightFilter, rayTime, lightIsect,
        /* fromCamera = */ false, lightFilterList, Ray_getDirFootprint(ray), &pdfLight);
    if (isSampleInvalid(Li, pdfLight)) {
        return;
    }

    // compute the mis weight
    // mirror lobes require special treatment
    const float misWeight = (lobesContribution->mMatchedLobeCount == 1 &&
        BsdfLobe_matchesFlags(lobesContribution->mLobes[0], BSDF_LOBE_TYPE_MIRROR)) ?
        1.0f : powerHeuristic(pdfBsdf, pdfLight);

    // total value of this sample
    Color lightContribution = pt * misWeight * Li * rcp(pdfBsdf);

    // apply russian roulette
    const float lum = luminance(f * lightContribution);
    const PathVertex &pv = rs->mPathVertex;
    if (!applyRussianRoulette(r[3],
                              this->mRussianRouletteThreshold,
                              this->mInvRussianRouletteThreshold,
                              pv.nonMirrorDepth,
                              lum,
                              lightContribution)) {
        return;
    }

    // resolve occlusion
    oneSamplerSubmitOcclusionRays(this,
                                  pbrTls,
                                  rs,
                                  isect,
                                  light,
                                  BsdfOneSampler_getBsdf(&bsdfOneSampler),
                                  rayEpsilon,
                                  shadowRayEpsilon,
                                  sequenceID,
                                  lightIsect,
                                  lightContribution,
                                  lobesContribution,
                                  wi,
                                  STANDARD);
}

static void
oneSamplerDirect(const uniform PathIntegrator * uniform this,
                 uniform PbrTLState * uniform pbrTls,
                 uniform ShadingTLState * uniform shadingTls,
                 const varying RayState * uniform rs,
                 const varying Intersection &isect,
                 const varying BsdfOneSampler &bsdfOneSampler,
                 const uniform LightSet &activeLightSet,
                 const varying Vec3f &cullingNormal,
                 varying float rayEpsilon,
                 const float shadowRayEpsilon,
                 varying IntegratorSample1D &rrSamples,
                 varying uint32_t &sequenceID)
{
    const PathVertex &pv = rs->mPathVertex;
    const bool highQuality = pv.nonMirrorDepth == 0;
    const int nDirect = highQuality ? this->mLightSamples : min(this->mLightSamples, 1);

    // TODO: we need a strategy to choose a good light subset
    // But that task is outside the scope of the initial one-sampler implementation
    // task.  We have a similar need with the multi-sampler integrator.
    const uniform int lightCount = LightSet_getLightCount(&activeLightSet);
    for (uniform int lightIndex = 0; lightIndex < lightCount; ++lightIndex) {
        const uniform Light * uniform light = LightSet_getLight(&activeLightSet, lightIndex);
        const uniform LightFilterList * uniform lightFilterList =
            LightSet_getLightFilterList(&activeLightSet, lightIndex);

        // Skip ray termination lights
        if (Light_getIsRayTerminator(light)) continue;

        // Set up our sample sequences
        // The scalar code uses 2D and 1D samplers rather than a 3D sampler.
        // In vector mode, using a 1D and 2D sampler for bsdf samples leads
        // to correlated noise.  Using the 3D sampler fixes this.  We seem to have
        // the opposite issue in scalar.
        IntegratorSample2D lightSamples;
        IntegratorSample3D bsdfSamples;
        IntegratorSample2D lightFilterSamples;
        IntegratorSample3D lightFilterSamples3D;
        const varying Subpixel &sp = rs->mSubpixel;

        bool lightFilterNeedsSamples = CPP_LightFilterNeedsSamples(
            (const uniform Scene *uniform)(pbrTls->mFs->mScene));

        if (highQuality) {
            // We want one shared sequence for depth 0
            const int samplesSoFar = sp.mSubpixelIndex * nDirect;
            SequenceID lSid;
            SequenceID_init(lSid,
                            pv.nonMirrorDepth,
                            sp.mPixel,
                            0,
                            SequenceTypeNextEventEstimation,
                            SequenceTypeLight,
                            Light_getHash(light),
                            sequenceID);
            IntegratorSample2D_init(lightSamples, lSid);
            lightSamples.mSampleNumber = samplesSoFar; // resume

            SequenceID bSid;
            SequenceID_init(bSid,
                            pv.nonMirrorDepth,
                            sp.mPixel,
                            0,
                            SequenceTypeNextEventEstimation,
                            SequenceTypeBsdf,
                            Light_getHash(light),
                            sequenceID);
            IntegratorSample3D_init(bsdfSamples, bSid);
            bsdfSamples.mSampleNumber = samplesSoFar; // resume

            if (lightFilterNeedsSamples) {
                SequenceID lFilterSid;
                SequenceID_init(lFilterSid,
                                pv.nonMirrorDepth,
                                sp.mPixel,
                                0,
                                SequenceTypeNextEventEstimation,
                                SequenceTypeLightFilter,
                                Light_getHash(light),
                                sequenceID);
                IntegratorSample2D_init(lightFilterSamples, lFilterSid);
                lightFilterSamples.mSampleNumber = samplesSoFar; // resume
                
                SequenceID lFilter3DSid;
                SequenceID_init(lFilter3DSid,
                                pv.nonMirrorDepth,
                                sp.mPixel,
                                0,
                                SequenceTypeNextEventEstimation,
                                SequenceTypeLightFilter3D,
                                Light_getHash(light),
                                sequenceID);
                IntegratorSample3D_init(lightFilterSamples3D, lFilter3DSid);
                lightFilterSamples3D.mSampleNumber = samplesSoFar; // resume
            }
        } else {
            SequenceID lSid;
            SequenceID_init(lSid,
                            pv.nonMirrorDepth,
                            sp.mPixel,
                            sp.mSubpixelIndex,
                            SequenceTypeNextEventEstimation,
                            SequenceTypeLight,
                            Light_getHash(light),
                            sequenceID);
            IntegratorSample2D_init(lightSamples, lSid);
            lightSamples.mTotalSamples = nDirect; // restart

            SequenceID bSid;
            SequenceID_init(bSid,
                            pv.nonMirrorDepth,
                            sp.mPixel,
                            sp.mSubpixelIndex,
                            SequenceTypeNextEventEstimation,
                            SequenceTypeBsdf,
                            Light_getHash(light),
                            sequenceID);
            IntegratorSample3D_init(bsdfSamples, bSid);
            bsdfSamples.mTotalSamples = nDirect; // restart

            if (lightFilterNeedsSamples) {
                SequenceID lFilterSid;
                SequenceID_init(lFilterSid,
                                pv.nonMirrorDepth,
                                sp.mPixel,
                                sp.mSubpixelIndex,
                                SequenceTypeNextEventEstimation,
                                SequenceTypeLightFilter,
                                Light_getHash(light),
                                sequenceID);
                IntegratorSample2D_init(lightFilterSamples, lFilterSid);
                lightFilterSamples.mTotalSamples = nDirect; // restart
                
                SequenceID lFilter3DSid;
                SequenceID_init(lFilter3DSid,
                                pv.nonMirrorDepth,
                                sp.mPixel,
                                sp.mSubpixelIndex,
                                SequenceTypeNextEventEstimation,
                                SequenceTypeLightFilter3D,
                                Light_getHash(light),
                                sequenceID);
                IntegratorSample3D_init(lightFilterSamples3D, lFilter3DSid);
                lightFilterSamples3D.mTotalSamples = nDirect; // restart
            }
        }

        const Color pt = pv.pathThroughput * (1.0f / nDirect);

        // Loop over each sample
        for (int s = 0; s < nDirect; ++s) {
            // light sampling
            float r[9];
            getSample(lightSamples, &r[0], pv.nonMirrorDepth, *pbrTls->mFs);
            getSample(rrSamples, r[3], pv.nonMirrorDepth, *pbrTls->mFs);
            if (lightFilterNeedsSamples) {
                getSample(lightFilterSamples, &r[4], pv.nonMirrorDepth, *pbrTls->mFs);
                getSample(lightFilterSamples3D, &r[6], pv.nonMirrorDepth, *pbrTls->mFs);
            }

            oneSamplerDirectLight(this,
                                  pbrTls,
                                  shadingTls,
                                  rs,
                                  isect,
                                  light,
                                  lightFilterList,
                                  cullingNormal,
                                  bsdfOneSampler,
                                  pt,
                                  r,
                                  rayEpsilon,
                                  shadowRayEpsilon,
                                  sequenceID);

            // bsdf sampling
            getSample(bsdfSamples, &r[0], pv.nonMirrorDepth, *pbrTls->mFs);
            getSample(rrSamples, r[3], pv.nonMirrorDepth, *pbrTls->mFs);
            if (lightFilterNeedsSamples) {
                getSample(lightFilterSamples, &r[4], pv.nonMirrorDepth, *pbrTls->mFs);
                getSample(lightFilterSamples3D, &r[6], pv.nonMirrorDepth, *pbrTls->mFs);
            }

            oneSamplerDirectBsdf(this,
                                 pbrTls,
                                 shadingTls,
                                 rs,
                                 isect,
                                 light,
                                 lightFilterList,
                                 cullingNormal,
                                 bsdfOneSampler,
                                 pt,
                                 r,
                                 rayEpsilon,
                                 shadowRayEpsilon,
                                 sequenceID);
        }
    }
}

static void
oneSamplerRayTerminatorLights(const uniform PathIntegrator * uniform this,
                              uniform PbrTLState * uniform pbrTls,
                              uniform ShadingTLState * uniform shadingTls,
                              const varying RayState * uniform rs,
                              const varying Intersection &isect,
                              const varying Vec3f &wi,
                              varying float pdfBsdf,
                              const varying BsdfLobe * varying lobe,
                              const varying Bsdf &bsdf,
                              const uniform LightSet &activeLightSet,
                              const varying LightFilterRandomValues &lightFilterR,
                              const varying Vec3f &cullingNormal,
                              varying float rayEpsilon,
                              const float shadowRayEpsilon,
                              varying uint32_t sequenceID,
                              const varying Color &pt)
{
    // TODO: we need a better way to get just the ray-terminator lights
    const uniform int lightCount = LightSet_getLightCount(&activeLightSet);
    for (uniform int lightIndex = 0; lightIndex < lightCount; ++lightIndex) {
        const uniform Light * uniform light = LightSet_getLight(&activeLightSet, lightIndex);
        const uniform LightFilterList * uniform lightFilterList =
            LightSet_getLightFilterList(&activeLightSet, lightIndex);

        // Skip non ray terminator lights
        if (!Light_getIsRayTerminator(light)) continue;
        CHECK_LANEMASK_AND_CONTINUE;

        // Does the sample direction hit the light?
        LightIntersection lightIsect;
        const RayDifferential &ray = rs->mRay;
        if (!Light_intersect(light, getP(isect), cullingNormal, wi, Ray_getTime(ray), sMaxValue, lightIsect)) {
            continue;
        }
        CHECK_LANEMASK_AND_CONTINUE;

        // It does, evaluate it and get a pdf
        float pdfLight;
        const Color Li = Light_eval(light, shadingTls, wi, getP(isect), lightFilterR, Ray_getTime(ray),
            lightIsect, /* fromCamera = */ false, lightFilterList, Ray_getDirFootprint(ray), &pdfLight);
        if (!isSampleValid(Li, pdfLight)) {
            continue;
        }
        CHECK_LANEMASK_AND_CONTINUE;

        // Compute the mis weight
        const float misWeight = BsdfLobe_matchesFlags(lobe, BSDF_LOBE_TYPE_MIRROR) ?
            1.0f : powerHeuristic(pdfBsdf, pdfLight);

        const Color contribution = misWeight * Li;

        // spawn a no-op occlusion ray
        uniform LobesContribution lc[programCount];
        lc[programIndex].mMatchedLobeCount = 1;
        lc[programIndex].mFs[0] = pt;
        lc[programIndex].mLobes[0] = lobe;
        oneSamplerSubmitOcclusionRays(this,
                                      pbrTls,
                                      rs,
                                      isect,
                                      light,
                                      bsdf,
                                      rayEpsilon,
                                      shadowRayEpsilon,
                                      sequenceID,
                                      lightIsect,
                                      contribution,
                                      &lc[programIndex],
                                      wi,
                                      FORCE_NOT_OCCLUDED);

    }
}

void
computeRadianceBsdfOneSampler(const uniform PathIntegrator * uniform this,
                              uniform PbrTLState * uniform pbrTls,
                              uniform ShadingTLState * uniform shadingTls,
                              const varying RayState * uniform rs,
                              const varying Intersection &isect,
                              const varying Bsdf &bsdf,
                              const varying BsdfSlice &slice,
                              varying bool doIndirect,
                              varying BsdfLobeType indirectFlags,
                              const uniform LightSet &activeLightSet,
                              const varying Vec3f &cullingNormal,
                              uniform bool hasRayTerminatorLights,
                              varying float rayEpsilon,
                              varying float shadowRayEpsilon,
                              const Color &ssAov,
                              varying uint32_t &sequenceID)
{

    const varying PathVertex &pv = rs->mPathVertex;
    const varying Subpixel &sp = rs->mSubpixel;
    const RayDifferential &ray = rs->mRay;

    // we are responsible for filling out material aovs
    const uniform FrameState &fs = *pbrTls->mFs;
    const uniform AovSchema &aovSchema = *((const uniform AovSchema * uniform) fs.mAovSchema);
    if (!AovSchema_empty(aovSchema)) {
        // Since we don't produce BsdfSample objects like multi-sample,
        // we can pack the material aovs now.  The albedo aov will use the slice
        // and the lobe albedo methods.
        const uniform LightAovs &lightAovs = *((const uniform LightAovs * uniform) fs.mLightAovs);
        const uniform MaterialAovs &materialAovs = *((const uniform MaterialAovs * uniform) fs.mMaterialAovs);
        aovAccumMaterialAovs(pbrTls, aovSchema, rs->mCameraId, lightAovs, materialAovs, isect, ray, fs.mScene, bsdf,
                             ssAov, slice, pv.aovPathPixelWeight, sp.mPixel, rs->mDeepDataHandle,
                             getFilmISPC(rs->mTilePassAndFilm), pv.lpeStateId);
    }

    BsdfOneSampler bsdfOneSampler;
    BsdfOneSampler_init(&bsdfOneSampler, &bsdf, &slice);

    // Both direct and indirect sampling share russian roulette samples
    SequenceID rrSamplesSid;
    SequenceID_init(rrSamplesSid,
                    sp.mPixel,
                    sp.mSubpixelIndex,
                    pv.nonMirrorDepth,
                    SequenceTypeRussianRouletteBsdf,
                    sequenceID);
    IntegratorSample1D rrSamples;
    IntegratorSample1D_init(rrSamples, rrSamplesSid);

    // First do NEE
    const uniform bool doNEE = this->mLightSamples != 0;
    if (doNEE) {
        oneSamplerDirect(this,
                         pbrTls,
                         shadingTls,
                         rs,
                         isect,
                         bsdfOneSampler,
                         activeLightSet,
                         cullingNormal,
                         rayEpsilon,
                         shadowRayEpsilon,
                         rrSamples,
                         sequenceID);
    }

    // We can exit early if doIndirect is false and we have no ray terminator lights
    if (!doIndirect && !hasRayTerminatorLights) {
        return;
    }

    // Spawn new rays through bsdf sampling
    // Respect the user bsdf samples up to the first non-mirror bounce.
    // After that, we spawn only a single bsdf sample.  If the bsdf
    // contains mirror lobes, we ignore bsdf samples.
    const bool highQuality = pv.nonMirrorDepth == 0 &&
        !(Bsdf_getType(&bsdf) & BSDF_LOBE_TYPE_MIRROR);
    const int nIndirect = highQuality ? this->mBsdfSamples : min(this->mBsdfSamples, 1);

    IntegratorSample3D bsdfSamples;
    IntegratorSample2D lightFilterSamples;
    IntegratorSample3D lightFilterSamples3D;

    bool lightFilterNeedsSamples = CPP_LightFilterNeedsSamples(
        (const uniform Scene *uniform)(pbrTls->mFs->mScene));

    if (highQuality) {
        // We want one shared sequence for depth 0
        const int samplesSoFar = sp.mSubpixelIndex * nIndirect;
        SequenceID bSid;
        SequenceID_init(bSid,
                        pv.nonMirrorDepth,
                        sp.mPixel,
                        0,
                        SequenceTypeIndirectLighting,
                        SequenceTypeBsdf,
                        sequenceID);
        IntegratorSample3D_init(bsdfSamples, bSid);
        bsdfSamples.mSampleNumber = samplesSoFar; // resume

        if (lightFilterNeedsSamples) {
            SequenceID bFilterSid;
            SequenceID_init(bFilterSid,
                            pv.nonMirrorDepth,
                            sp.mPixel,
                            0,
                            SequenceTypeIndirectLighting,
                            SequenceTypeLightFilter,
                            sequenceID);
            IntegratorSample2D_init(lightFilterSamples, bFilterSid);
            lightFilterSamples.mSampleNumber = samplesSoFar; // resume

            SequenceID bFilter3DSid;
            SequenceID_init(bFilter3DSid,
                            pv.nonMirrorDepth,
                            sp.mPixel,
                            0,
                            SequenceTypeIndirectLighting,
                            SequenceTypeLightFilter3D,
                            sequenceID);
            IntegratorSample3D_init(lightFilterSamples3D, bFilter3DSid);
            lightFilterSamples3D.mSampleNumber = samplesSoFar; // resume
        }
    } else {
        SequenceID bSid;
        SequenceID_init(bSid,
                        pv.nonMirrorDepth,
                        sp.mPixel,
                        sp.mSubpixelIndex,
                        SequenceTypeIndirectLighting,
                        SequenceTypeBsdf,
                        sequenceID);
        IntegratorSample3D_init(bsdfSamples, bSid);
        bsdfSamples.mTotalSamples = nIndirect; // restart

        if (lightFilterNeedsSamples) {
            SequenceID bFilterSid;
            SequenceID_init(bFilterSid,
                            pv.nonMirrorDepth,
                            sp.mPixel,
                            sp.mSubpixelIndex,
                            SequenceTypeIndirectLighting,
                            SequenceTypeLightFilter,
                            sequenceID);
            IntegratorSample2D_init(lightFilterSamples, bFilterSid);
            lightFilterSamples.mTotalSamples = nIndirect; // restart

            SequenceID bFilter3DSid;
            SequenceID_init(bFilter3DSid,
                            pv.nonMirrorDepth,
                            sp.mPixel,
                            sp.mSubpixelIndex,
                            SequenceTypeIndirectLighting,
                            SequenceTypeLightFilter3D,
                            sequenceID);
            IntegratorSample3D_init(lightFilterSamples3D, bFilter3DSid);
            lightFilterSamples3D.mTotalSamples = nIndirect; // restart
        }
    }

    // loop over samples
    for (int s = 0; s < nIndirect; ++s) {
        float r[9]; // r[0] lobe select, r[1:2] bsdf, r[3] russian roulette, r[4:5] light filter, r[6:9] light filter 3D
        getSample(bsdfSamples, &r[0], pv.nonMirrorDepth, *pbrTls->mFs);
        if (lightFilterNeedsSamples) {
            getSample(lightFilterSamples, &r[4], pv.nonMirrorDepth, *pbrTls->mFs);
            getSample(lightFilterSamples3D, &r[6], pv.nonMirrorDepth, *pbrTls->mFs);
        }

        Vec3f wi;
        float pdfBsdf;
        Color f;
        // The sampled lobe will be the one we use fo all lobe dependent
        // decision, such as path depth, aovs, etc...
        const varying BsdfLobe * varying lobe = nullptr;
        if (this->mBsdfSamplerStrategy == BSDF_SAMPLER_STRATEGY_ONE_LOBE) {
            float pdfLobe; // probability of picking this lobe
            lobe = BsdfOneSampler_sampleLobe(&bsdfOneSampler, r[0], pdfLobe);
            if (!lobe || pdfLobe == 0.0f) {
                continue;
            }
            CHECK_LANEMASK_AND_CONTINUE;

            // TODO: path guide
            foreach_unique(l in lobe) {
                f = BsdfLobe_sample(l, slice, r[1], r[2], wi, pdfBsdf);
            }
            pdfBsdf *= pdfLobe; // include probability of selecting lobe

            // We are responsible for checking if this lobe matches the flags
            BsdfLobeType flags = BsdfSlice_getSurfaceFlags(&slice, bsdf, wi);
            if (!BsdfLobe_matchesFlags(lobe, flags)) {
                f = sBlack;
                pdfBsdf = 0.f;
            }
        } else {
            MNRY_ASSERT(this->mBsdfSamplerStrategy == BSDF_SAMPLER_STRATEGY_ONE_SAMPLE);
            uniform LobesContribution lobesContributionData[programCount];
            uniform LobesContribution * varying lobesContribution = &lobesContributionData[programIndex];
            // TODO: path guide
            f = BsdfOneSampler_sample(&bsdfOneSampler, r[0], r[1], r[2], wi, pdfBsdf, lobesContribution);
            if (!lobesContribution->mMatchedLobeCount) {
                continue;
            }
            CHECK_LANEMASK_AND_CONTINUE;

            lobe = lobesContribution->mLobes[0];
        }
        CHECK_LANEMASK_AND_CONTINUE;

        if (!isSampleValid(f, pdfBsdf)) {
            continue;
        }
        CHECK_LANEMASK_AND_CONTINUE;

        // Compute the total throughput potential for this sample.
        // Terminate the sample if it's throughput is 0.
        Color pt = f * rcp(pdfBsdf * nIndirect);
        pt = pt * pv.pathThroughput;
        float lum = luminance(pt);

        // Apply russian roulette
        getSample(rrSamples, r[3], pv.nonMirrorDepth, *pbrTls->mFs);
        if (!applyRussianRoulette(r[3],
                                  this->mRussianRouletteThreshold,
                                  this->mInvRussianRouletteThreshold,
                                  pv.nonMirrorDepth,
                                  lum,
                                  pt)) {
            continue;
        }
        CHECK_LANEMASK_AND_CONTINUE;

        // Terminate path due to depth limits.
        // FIXME: we should not need to special case the hair lobe depth check.
        // Make it work with indirectFlags.
        // ISPC BUG?: need to cast hairLobe to int8_t to avoid incorrect
        // comparison at runtime.
        const bool hairLobe = BsdfLobe_getIsHair(lobe);
        if (!doIndirect || !BsdfLobe_matchesFlags(lobe, indirectFlags)) {
            // The lobe flags say the depth limit has been meet.  But
            // if we are a hair lobe, me may need to keep going anyway.
            if (!((int8_t)hairLobe) || (pv.hairDepth >= this->mMaxHairDepth)) {
                if (hasRayTerminatorLights && !((int8_t) hairLobe) &&
                    BsdfLobe_matchesFlags(lobe, BSDF_LOBE_TYPE_ALL_TRANSMISSION)) {             
                    const varying LightFilterRandomValues lightFilterR = { 
                        Vec2f_ctor(r[4], r[5]), 
                        Vec3f_ctor(r[6], r[7], r[8])};
                    oneSamplerRayTerminatorLights(this,
                                                  pbrTls,
                                                  shadingTls,
                                                  rs,
                                                  isect,
                                                  wi,
                                                  pdfBsdf,
                                                  lobe,
                                                  bsdf,
                                                  activeLightSet,
                                                  lightFilterR,
                                                  cullingNormal,
                                                  rayEpsilon,
                                                  shadowRayEpsilon,
                                                  sequenceID,
                                                  pt);
                }
                continue; // terminate the path
            }
        }
        CHECK_LANEMASK_AND_CONTINUE;

        // At this point, count this sample in the stats
        addToCounter(pbrTls->mStatistics, STATS_BSDF_SAMPLES, getActiveLaneCount());

        const bool diffuseLobe = BsdfLobe_matchesFlags(lobe, BSDF_LOBE_TYPE_ALL_DIFFUSE);
        const bool glossyLobe = BsdfLobe_matchesFlags(lobe, BSDF_LOBE_TYPE_ALL_GLOSSY);
        const bool mirrorLobe = BsdfLobe_matchesFlags(lobe, BSDF_LOBE_TYPE_ALL_MIRROR);
        const bool transmissionLobe = BsdfLobe_matchesFlags(lobe, BSDF_LOBE_TYPE_ALL_TRANSMISSION);

        // There have been enough ISPC bugs with the above ray depth
        // termination code to warrant some asserts.
        if (hairLobe) {
            MNRY_ASSERT(!diffuseLobe || pv.diffuseDepth < this->mMaxDiffuseDepth || pv.hairDepth < this->mMaxHairDepth);
            MNRY_ASSERT(!glossyLobe || pv.glossyDepth < this->mMaxGlossyDepth || pv.hairDepth < this->mMaxHairDepth);
            MNRY_ASSERT(!mirrorLobe || pv.mirrorDepth < this->mMaxMirrorDepth || pv.hairDepth < this->mMaxHairDepth);
        } else {
            MNRY_ASSERT(!diffuseLobe || pv.diffuseDepth < this->mMaxDiffuseDepth);
            MNRY_ASSERT(!glossyLobe || pv.glossyDepth < this->mMaxGlossyDepth);
            MNRY_ASSERT(!mirrorLobe || pv.mirrorDepth < this->mMaxMirrorDepth);
        }

        // Check transparency threshold
        float newAccumOpacity;
        if (mirrorLobe && transmissionLobe) {
            float lobeTransparency = reduceTransparency(f);
            newAccumOpacity = pv.accumOpacity + (1 - lobeTransparency) * (1 - pv.accumOpacity);
            if (newAccumOpacity > this->mTransparencyThreshold) {
                continue;
            }
        } else {
            newAccumOpacity = pv.accumOpacity;
        }
        CHECK_LANEMASK_AND_CONTINUE;

        // Prepare a path vertex
        RayState nextRayState;
        PathVertex &nextPv = nextRayState.mPathVertex;
        nextPv.pathThroughput = pt;
        nextPv.pathPixelWeight = 0.0f;
        // Use previous path pixel weight for aovPathPixelWeight as there's existing logic
        // in vector mode that sometimes assumes that pv.pathPixelWeight = 0.  Thus, we must seperately
        // keep track of the pathPixelWeight for aovs.  See comment in PathIntegratorMultiSampler.ispc::
        // addIndirectOrDirectVisibleContributionsBundled().
        nextPv.aovPathPixelWeight = pv.pathPixelWeight;
        nextPv.pathDistance = pv.pathDistance + Ray_getEnd(ray);
        foreach_unique(lobePtr in lobe) {
            nextPv.minRoughness = computeMinRoughness(lobePtr, this->mRoughnessClampingFactor, pv.minRoughness);
        }
        nextPv.diffuseDepth = pv.diffuseDepth + (diffuseLobe ? 1 : 0);
        nextPv.glossyDepth = pv.glossyDepth + (glossyLobe ? 1 : 0);
        nextPv.mirrorDepth = pv.mirrorDepth + (mirrorLobe ? 1 : 0);
        nextPv.nonMirrorDepth = pv.nonMirrorDepth + (mirrorLobe ? 0 : 1);
        nextPv.hairDepth = pv.hairDepth + (hairLobe ? 1 : 0);
        nextPv.volumeDepth = pv.volumeDepth + 1;
        nextPv.presenceDepth = pv.presenceDepth;
        nextPv.subsurfaceDepth = pv.subsurfaceDepth;
        nextPv.accumOpacity = newAccumOpacity;
        nextPv.lobeType = BsdfLobe_getType(lobe);

        // LPE
        // TODO: Big issue when doing BSDF_SAMPLER_STRATEGY_ONE_SAMPLE.
        // We can only set a single scatter transition event,
        // so we'll use the selected lobe - but f contains the evaluation of the all lobes.
        // so the aovs will bleed into each other.
        nextPv.lpeStateId = -1;
        nextPv.lpeStateIdLight = -1; // no need for the ray bundle handler to handle
        const uniform LightAovs * uniform lightAovs = (const uniform LightAovs * uniform) pbrTls->mFs->mLightAovs;
        if (LightAovs_hasEntries(lightAovs)) {
            // transition
            foreach_unique(lobePtr in lobe) {
                nextPv.lpeStateId = LightAovs_scatterEventTransition(lightAovs, pv.lpeStateId, bsdf, *lobePtr);
            }
        }

        // We have some self-intersections when rays leave at grazing
        // angle, so we adjust the rayEpsilon accordingly
        const float denom = abs(dot(getNg(isect), wi));
        // isect.getNg() itself or the dot product above can be zero.
        const float start = isZero(denom) ? rayEpsilon : rayEpsilon / denom;
        // For NEE to work properly with de-coupled direct and indirect
        // samples we must skip samples that would have been counted if sampled
        // as direct bsdf samples.  So that means we need to find the shortest
        // distance to an active light and set the end of our ray to this distance.
        // TODO: find a faster way to get the shortest distance to a light
        // Perhaps the light accelerator could add an intersectNearestLight function.
        // Note that because we update the ray end with the distance to the
        // nearest light, we eliminate the possibility of intersecting geometry
        // that is beyond the nearest light.
        float end = sMaxValue;
        int hitLightIndex = -1;
        LightIntersection hitLightIsect;
        if (doNEE) {
            const uniform int lightCount = LightSet_getLightCount(&activeLightSet);
            for (uniform int lightIndex = 0; lightIndex < lightCount; ++lightIndex) {
                const uniform Light * uniform light = LightSet_getLight(&activeLightSet, lightIndex);
                // skip ray termination lights
                if (!Light_getIsRayTerminator(light)) {
                    if (Light_intersect(light, getP(isect), cullingNormal, wi, Ray_getTime(ray), end, hitLightIsect)) {
                        MNRY_ASSERT(hitLightIsect.distance <= end);
                        end = hitLightIsect.distance;
                    }
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
            for (; lobeIndex < Bsdf_getLobeCount(&bsdf); ++lobeIndex) {
                if (Bsdf_getLobe(&bsdf, lobeIndex) == lobe) break;
            }
            SequenceID lightChoiceSid;
            SequenceID_init(lightChoiceSid,
                            pv.nonMirrorDepth,
                            sp.mPixel,
                            lobeIndex,
                            SequenceTypeIndexSelection,
                            sp.mSubpixelIndex,
                            sequenceID);
            IntegratorSample1D lightChoiceSamples;
            IntegratorSample1D_init(lightChoiceSamples, lightChoiceSid);
            int numHits;
            hitLightIndex = LightSet_intersect(pbrTls,
                                               &activeLightSet,
                                               getP(isect),
                                               cullingNormal,
                                               wi,
                                               Ray_getTime(ray),
                                               end,
                                               /* fromCamera = */ false,
                                               /* includeRayTerminationLights = */ false,
                                               lightChoiceSamples,
                                               pv.nonMirrorDepth,
                                               lobeTypeToRayMask(BsdfLobe_getType(lobe)),
                                               hitLightIsect,
                                               numHits);
            if (hitLightIndex != -1) {
                MNRY_ASSERT(hitLightIndex < LightSet_getLightCount(&activeLightSet));
                end = hitLightIsect.distance;
            }
        }
        CHECK_LANEMASK_AND_CONTINUE;

        // Prepare a RayDifferential
        RayDifferential &nextRay = nextRayState.mRay;
        Ray_init(&nextRay, ray, start, end);

        // Scatter and scale our next ray differential
        // We scatter the ray based on the sampled lobe
        float scale = (BsdfLobe_getDifferentialFlags(lobe) & BSDF_LOBE_IGNORES_INCOMING_DIFFERENTIALS) ?
            sp.mPrimaryRayDiffScale : 1.0f;
        foreach_unique (lobePtr in lobe) {
            scatterAndScale(isect, *lobePtr, -1.0f * ray.dir, wi, scale, r[1], r[2], nextRay);
        }

        // Finish filling in the RayState data

        // We figure out the ray mask based on the lobe type.
        // For ray mask propagation, we simply merge with existing ray mask.
        // We should exclude the camera ray mask when propagating.
        int lobeMask = lobeTypeToRayMask(BsdfLobe_getType(lobe));
        ++sequenceID;
        nextRay.mask = fs.mPropagateVisibilityBounceType
            ? ((ray.mask | lobeMask) & ~RTC_MASK_RAY_CAMERA) : lobeMask;
        nextRayState.mSequenceID = sequenceID;
        nextRayState.mSubpixel.mPixel = rs->mSubpixel.mPixel;
        nextRayState.mSubpixel.mSubpixelIndex = sp.mSubpixelIndex;
        nextRayState.mSubpixel.mSubpixelX = sp.mSubpixelX;
        nextRayState.mSubpixel.mSubpixelY = sp.mSubpixelY;
        nextRayState.mSubpixel.mPixelSamples = sp.mPixelSamples;
        nextRayState.mSubpixel.mSampleClampingValue = sp.mSampleClampingValue;
        nextRayState.mSubpixel.mPrimaryRayDiffScale = sp.mPrimaryRayDiffScale;
        nextRayState.mSubpixel.mTextureDiffScale = sp.mTextureDiffScale;
        nextRayState.mCameraId = rs->mCameraId;
        nextRayState.mTilePassAndFilm = rs->mTilePassAndFilm;
        nextRayState.mDeepDataHandle = rs->mDeepDataHandle;
        nextRayState.mCryptomatteDataHandle = rs->mCryptomatteDataHandle;
        nextRayState.mCryptomatteDataHandle2 = rs->mCryptomatteDataHandle2;
        PbrTLState_acquireDeepData(pbrTls, nextRayState.mDeepDataHandle);
        PbrTLState_acquireCryptomatteData(pbrTls, nextRayState.mCryptomatteDataHandle);
        PbrTLState_acquireCryptomatteData2(pbrTls, nextRayState.mCryptomatteDataHandle2);

        // Accumulate post scatter extra aovs
        if (LightAovs_hasEntries(lightAovs)) {
            aovAccumPostScatterExtraAovs(pbrTls, fs, &nextRayState, bsdf);
        }

        snapshotLaneUtilization(pbrTls->mStatistics, STATS_VEC_INDIRECT_F);

        // queue the ray
        // packed_store_active2() "may write one extra element to the output array"
        // according to ISPC documentation https://ispc.github.io/ispc.html.
        // Please make sure that 1 extra element was allocated for the output array.
        uniform uint32_t activeLaneIndices[VLEN + 1];
        uniform uint32_t numEntries = packed_store_active2(activeLaneIndices, programIndex);
        if (numEntries > 0) {
            CPP_addIncoherentRayQueueEntries(pbrTls, &nextRayState, numEntries, activeLaneIndices);
        }

        // If we are not performaning NEE, then add an occlusion ray
        // Unlike the scalar code, we don't know if the submitted indirect ray
        // will bounce or not.  So the vector stats will show occlusion rays
        // while the scalar code will not.  We assume that if the indirect ray
        // bounces (i.e. hits geometry) then the occlusion ray will be occluded.
        if (!doNEE && hitLightIndex != -1) {
            uniform LobesContribution lc[programCount];
            MNRY_ASSERT(lobe);
            lc[programIndex].mMatchedLobeCount = 1;
            lc[programIndex].mFs[0] = pt;
            lc[programIndex].mLobes[0] = lobe;

            const uniform Light * varying hitLight = LightSet_getLight(&activeLightSet, hitLightIndex);
            const uniform LightFilterList * varying hitLightFilterList =
                LightSet_getLightFilterList(&activeLightSet, hitLightIndex);
            foreach_unique(light in hitLight) {
                Color Li;
                float pdfLight;
                const varying LightFilterRandomValues lightFilterR = {Vec2f_ctor(r[4], r[5]), Vec3f_ctor(r[6], r[7], r[8])};
                Li = Light_eval(light, shadingTls, wi, getP(isect), lightFilterR, Ray_getTime(ray), hitLightIsect,
                                /* fromCamera = */ false, hitLightFilterList, Ray_getDirFootprint(ray), &pdfLight);
                if (!isSampleInvalid(Li, pdfLight)) {
                    oneSamplerSubmitOcclusionRays(this,
                                                  pbrTls,
                                                  rs,
                                                  isect,
                                                  light,
                                                  bsdf,
                                                  rayEpsilon,
                                                  shadowRayEpsilon,
                                                  sequenceID,
                                                  hitLightIsect,
                                                  Li,
                                                  &lc[programIndex],
                                                  wi,
                                                  STANDARD);
                }
            }
        }
    }

}



