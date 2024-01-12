// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0


#include "PathIntegratorUtil.isph"
#include "PathIntegrator.isph"
#include <moonray/rendering/pbr/core/Util.isph>
#include <moonray/rendering/pbr/core/Aov.isph>

#include <moonray/rendering/mcrt_common/Ray.isph>
#include <moonray/rendering/pbr/sampler/SequenceID.isph>
#include <scene_rdl2/common/platform/Platform.isph>

//-----------------------------------------------------------------------------

inline void
integrateBsdfSample(const varying BsdfSampler &bSampler, uniform int lobeIndex,
        const varying LightSetSampler &lSampler, const varying Color &pathThroughput,
        const varying LightContribution &lCo, varying BsdfSample &bsmp)
{
    // Mark the sample valid by setting the distance
    // TODO: Isn't it better to not add the sample in the first place ?
    bsmp.distance = (!lCo.isInvalid  ?  lCo.distance  :
            (BsdfSampler_getDoIndirect(&bSampler)  ?  sMaxValue  :
                    sInvalidBsdfSampleDistance));
    if (BsdfSample_isInvalid(&bsmp)) {
        return;
    }

    const int ni = BsdfSampler_getLobeSampleCount(&bSampler, lobeIndex);
    const float invNi = BsdfSampler_getInvLobeSampleCount(&bSampler, lobeIndex);
    const Color pt = pathThroughput * invNi * bsmp.f * rcp(bsmp.pdf);

    // Compute direct tentative contribution (omit shadowing)
    const bool lobeIsMirror = BsdfLobe_matchesFlags(
            BsdfSampler_getLobe(&bSampler, lobeIndex), BSDF_LOBE_TYPE_ALL_MIRROR);
    const int nl = (lCo.isInvalid  ?  0  :
            LightSetSampler_getLightSampleCount(&lSampler));
    bsmp.tDirect = (lCo.isInvalid  ?  sBlack  :  (lobeIsMirror  ?
            // Bsdf importance sampling
            lCo.Li * pt  :
            // Multiple importance sampling
            lCo.Li * pt * powerHeuristic(ni * bsmp.pdf, nl * lCo.pdf)
        ));

    // Compute indirect tentative contribution (omit Lindirect)
    bsmp.tIndirect = (BsdfSampler_getDoIndirect(&bSampler)  ?  pt  :  sBlack);
}

struct Scene;
extern "C" uniform bool CPP_LightFilterNeedsSamples(const uniform Scene *uniform scene);

void
drawBsdfSamples(uniform PbrTLState * uniform pbrTls, uniform ShadingTLState * uniform shadingTls,
        const varying BsdfSampler &bSampler, const varying LightSetSampler &lSampler,
        const varying Subpixel &sp,  const varying PathVertex &pv,
        const varying Vec3f &P, const varying Vec3f &cullingNormal, varying float time,
        varying uint32_t sequenceID, varying BsdfSample * uniform bsmp, uniform int clampingDepth,
        varying float clampingValue, varying Flags indirectFlags, varying float rayDirFootprint)
{
    SequenceID bsdfSid, lightFilterSid, lightFilter3DSid, lightChoiceSid;
    IntegratorSample2D bsdfSamples;
    IntegratorSample2D lightFilterSamples;
    IntegratorSample3D lightFilterSamples3D;
    IntegratorSample1D lightChoiceSamples;
    const int spIndex = (pv.nonMirrorDepth == 0) ? 0 : sp.mSubpixelIndex;

    bool lightFilterNeedsSamples = CPP_LightFilterNeedsSamples(
        (const uniform Scene *uniform)(pbrTls->mFs->mScene));

    LightContribution lCo;

    // Loop over active lobes
    varying int lobeSeed = 0;
    uniform int lobeCount = BsdfSampler_getLobeCount(&bSampler);
    for (uniform int lobeIndex = 0; lobeIndex < lobeCount; ++lobeIndex) {
        const varying BsdfLobe * uniform lobe = BsdfSampler_getLobe(&bSampler, lobeIndex);

        if (!isActive(lobe)) {
            continue;
        }

        snapshotLaneUtilization(pbrTls->mStatistics, STATS_VEC_BSDF_LOBES);

        int lobeSampleCount = BsdfSampler_getLobeSampleCount(&bSampler, lobeIndex);
        MNRY_ASSERT(reduce_max(lobeSampleCount) > 0);

        // We can't use lobeIndex to seed the sequence since the lobeCount, and
        // therefore the lobe index, can change depending on which lobes are
        // active on samples in adjacent lanes. To keep everything deterministic
        // we'll keep a separate lobe seed for each lane which only gets
        // incremented when a lobe is actually active on that lane.
        lobeSeed += 1;

        // Setup sampler sequence
        // We want one shared sequence for depth 0.
        SequenceID_init(bsdfSid, (uint32_t)pv.nonMirrorDepth, sp.mPixel, lobeSeed,
                (uniform uint32_t)SequenceTypeBsdf, spIndex, sequenceID);
        IntegratorSample2D_init(bsdfSamples, bsdfSid);

        if (lightFilterNeedsSamples) {
            SequenceID_init(lightFilterSid, (uint32_t)pv.nonMirrorDepth, sp.mPixel, lobeSeed,
                    (uniform uint32_t)SequenceTypeLightFilter, spIndex, sequenceID);
            IntegratorSample2D_init(lightFilterSamples, lightFilterSid);

            SequenceID_init(lightFilter3DSid, (uint32_t)pv.nonMirrorDepth, sp.mPixel, lobeSeed,
                    (uniform uint32_t)SequenceTypeLightFilter3D, spIndex, sequenceID);
            IntegratorSample3D_init(lightFilterSamples3D, lightFilter3DSid);
        }

        if (pv.nonMirrorDepth == 0) {
            const int samplesSoFar = sp.mSubpixelIndex * lobeSampleCount;
            bsdfSamples.mSampleNumber = samplesSoFar;
            if (lightFilterNeedsSamples) {
                lightFilterSamples.mSampleNumber = samplesSoFar;
                lightFilterSamples3D.mSampleNumber = samplesSoFar;
            }
        } else {
            bsdfSamples.mTotalSamples = lobeSampleCount;
            if (lightFilterNeedsSamples) {
                lightFilterSamples.mTotalSamples = lobeSampleCount;
                lightFilterSamples3D.mTotalSamples = lobeSampleCount;
            }
        }

        // Set up sampler sequence for stochastic light intersection
        SequenceID_init(lightChoiceSid, (uint32_t)pv.nonMirrorDepth, sp.mPixel, lobeSeed,
                (uniform uint32_t)SequenceTypeBsdf, sp.mSubpixelIndex, sequenceID);
        IntegratorSample1D_init(lightChoiceSamples, lightChoiceSid);

        // lobeType is used for conditionally including ray termination lights.
        // Ray termination lights provide a way to cheaply fill in the zeros which result from terminating
        // ray paths too early. This is done by forcing the occlusion test to fail unconditionally.
        // A path can only be terminated when the relevant indirectFlags bit for this lobeType is set to false.
        const uniform BsdfLobeType lobeType = BsdfLobe_getType(lobe);
        varying bool includeRayTerminationLights = (lobeType & ~indirectFlags.mBits & BSDF_LOBE_TYPE_ALL_LOBES)
                                                       != 0;

        // lobeMask is used for comparison against a light's visibility flags.
        // This comparison is made in the LightSampler's intersectAndEval() function inside the loop below.
        const varying int lobeMask = lobeTypeToRayMask(lobeType);

        uniform int s = bSampler.mLobeToSampleIndex[lobeIndex];

        // Loop over each lobe's samples
        for (int i = 0; i < lobeSampleCount; ++i, ++s) {

            snapshotLaneUtilization(pbrTls->mStatistics, STATS_VEC_BSDF_LOBE_SAMPLES_PRE);

            // Draw the sample and test validity
            float bsdfSample[2];
            getSample(bsdfSamples, bsdfSample, pv.nonMirrorDepth, *pbrTls->mFs);

            LightFilterRandomValues lightFilterSample;
            if (lightFilterNeedsSamples) {
                getSample(lightFilterSamples, &lightFilterSample.r2.x, pv.nonMirrorDepth, *pbrTls->mFs);            
                getSample(lightFilterSamples3D, &lightFilterSample.r3.x, pv.nonMirrorDepth, *pbrTls->mFs);
            }

            bool isValid = BsdfSampler_sample(pbrTls, &bSampler, lobeIndex,
                    bsdfSample[0], bsdfSample[1], bsmp[s]);
            if (!isValid) {
                continue;
            }

            snapshotLaneUtilization(pbrTls->mStatistics, STATS_VEC_BSDF_LOBE_SAMPLES_POST);

            MNRY_ASSERT(isNormalized(bsmp[s].wi));

            // Compute lCo, which is the un-occluded light contribution for a
            // random physical light in the LightSet in the direction of the sample.
            // It should set Li and pdf to 0 if there is no light in that
            // direction, and return the distance to the chosen light if
            // there is. The distance is set to infinity if there is no light or
            // to sInfiniteLightDistance if there is an InfiniteAreaLight.
            LightSetSampler_intersectAndEval(&lSampler, pbrTls, shadingTls, P, cullingNormal, bsmp[s].wi,
                                             lightFilterSample, time, false, includeRayTerminationLights,
                                             lightChoiceSamples, pv.nonMirrorDepth, lobeMask, lCo, rayDirFootprint);

            integrateBsdfSample(bSampler, lobeIndex,
                    lSampler, pv.pathThroughput, lCo, bsmp[s]);
            // save the light, we'll need it (for its labels) when processing lpes
            if (lCo.isInvalid) {
                BsdfSampleLPE_setLightPtr(bsmp[s].lp, NULL);
            } else {
                BsdfSampleLPE_setLightPtr(bsmp[s].lp, lCo.light);
            }

            // Selective clamp of tDirect with clampingValue
            if (pv.nonMirrorDepth >= clampingDepth) {
                bsmp[s].tDirect = smartClamp(bsmp[s].tDirect, clampingValue);
            }
        }
    }
}


//-----------------------------------------------------------------------------

inline void
integrateLightSetSample(const varying LightSetSampler &lSampler,
        uniform int lightIndex, const varying BsdfSampler &bSampler,
        const varying PathVertex &pv, varying LightSample &lsmp,
        uniform int clampingDepth, varying float clampingValue)
{
    // Mark the sample valid only if we have valid lobe contributions and
    // initialize contribution for summing
    bool isInvalid = true;
    lsmp.t = sBlack;

    const int ni = LightSetSampler_getLightSampleCount(&lSampler);
    const float invNi = LightSetSampler_getInvLightSampleCount(&lSampler);

    const Color factor = pv.pathThroughput * invNi * lsmp.Li * rcp(lsmp.pdf);

    // Integrate with all the matching lobes
    const varying BsdfSlice * uniform slice = BsdfSampler_getBsdfSlice(&bSampler);
    BsdfLobeType flags = BsdfSlice_getSurfaceFlags(slice,
            *BsdfSampler_getBsdf(&bSampler), lsmp.wi);
    const uniform int lobeCount = BsdfSampler_getLobeCount(&bSampler);
    MNRY_ASSERT(lobeCount > 0);

    // initialize lpe member, setting each lobe entry to a null value
    for (uniform int k = 0; k < BSDF_MAX_LOBE; ++k) lsmp.lp.lobe[k] = nullptr;

    for (uniform int k = 0; k < lobeCount; ++k) {
        const varying BsdfLobe* const uniform lobe = BsdfSampler_getLobe(&bSampler, k);

        // TODO: Should we still go through MIS calculations if
        //  !lobe->matchesFlags(flags)
        if (!isActive(lobe) || !BsdfLobe_matchesFlags(lobe, flags)) {
            continue;
        }

        // skip lobe if light is marked as not visible from this lobe
        const varying int lobeMask = lobeTypeToRayMask(BsdfLobe_getType(lobe));
        const uniform Light* uniform light = LightSetSampler_getLight(&lSampler, lightIndex);
        if (!(lobeMask & Light_getVisibilityMask(light))) {
            continue;
        }

        // Evaluate the lobe
        // TODO: Should we still go through MIS calculations if
        // isSampleInvalid() because of pdf = 0
        float pdf;
        Color f = BsdfLobe_eval(lobe, *slice, lsmp.wi, &pdf);
        if (isSampleInvalid(f, pdf)) {
            continue;
        }

        // Direct lighting tentative contribution (omits shadowing)
        // using multiple importance sampling:
        const int nk = BsdfSampler_getLobeSampleCount(&bSampler, k);
        Color t = factor * f * powerHeuristic(ni * lsmp.pdf, nk * pdf);

        // Selective clamp of t with clampingValue
        if (pv.nonMirrorDepth >= clampingDepth) {
            t = smartClamp(t, clampingValue);
        }

        lsmp.t = lsmp.t + t;

        // save off per-lobe t info, we'll need these for LPEs
        MNRY_ASSERT(k < BSDF_MAX_LOBE);
        lsmp.lp.t[k] = t;
        lsmp.lp.lobe[k] = lobe;

        isInvalid = false;
    }

    if (isInvalid) {
        LightSample_setInvalid(&lsmp);
    }
}

extern "C" bool
CPP_aovAccumVisibilityAovsBundled(const uniform PbrTLState * uniform pbrTls,
                                  const uniform AovSchema& aovSchema,
                                  const uniform LightAovs& lightAovs,
                                  const uniform Vec2f& value,
                                  const uniform int lpeStateId,
                                  const uniform uint32_t pixel,
                                  const uniform uint32_t deepDataHandle,
                                  uniform bool lpePassthrough);

/// This helper function adds "misses" to the visibility aov. We do this here because there are some light
/// samples that are thrown out early because they are facing away from the point, and they need to be 
/// added to the visibility aov before they are discarded.
void accumVisibilityAovsOccluded(const uniform PbrTLState * uniform pbrTls,
                         const varying BsdfSampler& bSampler,
                         const varying LightSetSampler& lSampler,
                         const varying PathVertex &pv,
                         const varying uint32_t pixel,
                         const varying uint32_t deepDataHandle,
                         const uniform Light * light,
                         uniform int attempts) 
{
    // Only include direct light samples
    if (pv.nonMirrorDepth > 0) { return; }

    const uniform FrameState &fs = *pbrTls->mFs;
    const uniform AovSchema &aovSchema = *((const uniform AovSchema * uniform) fs.mAovSchema);
    const uniform LightAovs& lightAovs = *((const uniform LightAovs* uniform) fs.mLightAovs);

    if (!AovSchema_empty(aovSchema) && LightAovs_hasVisibilityEntries(&lightAovs)) {
        varying bool addVisibility = true;
        for (uniform int k = 0; k < BsdfSampler_getLobeCount(&bSampler); ++k) {
            if (!addVisibility) { break; }

            const varying BsdfLobe& lobe = *((const varying BsdfLobe * uniform) BsdfSampler_getLobe(&bSampler, k));
            const varying Bsdf &bsdf = *LightSetSampler_getBsdf(&lSampler);

            int varying lpeStateId = pv.lpeStateId;
            lpeStateId = LightAovs_scatterEventTransition(&lightAovs, lpeStateId, bsdf, lobe);
            lpeStateId = LightAovs_lightEventTransition(&lightAovs, lpeStateId, light);
    
            foreach_active(lane) {
                uniform Vec2f value = Vec2f_ctor(0.0f, attempts);
                if (CPP_aovAccumVisibilityAovsBundled(pbrTls, 
                                                      aovSchema, 
                                                      lightAovs, 
                                                      value,
                                                      extract(lpeStateId, lane),
                                                      extract(pixel, lane),
                                                      extract(deepDataHandle, lane),
                                                      false)) {
                    insert(addVisibility, lane, false);
                }
            }
        }
    }
}

void
drawLightSetSamples(uniform PbrTLState * uniform pbrTls, uniform ShadingTLState * uniform shadingTls,
        const varying LightSetSampler &lSampler, const varying BsdfSampler &bSampler,
        const varying Subpixel &sp, const varying PathVertex &pv,
        const varying Vec3f &P, const varying Vec3f &cullingNormal, varying float time,
        varying uint32_t sequenceID, varying LightSample * uniform lsmp,
        uniform int clampingDepth, varying float clampingValue, varying float rayDirFootprint,
        const varying RayState* uniform rs, const uniform int lightIndex)
{
    SequenceID sid, sidFilter, sidFilter3D;
    IntegratorSample3D lightSamples;
    IntegratorSample2D lightFilterSamples;
    IntegratorSample3D lightFilterSamples3D;
    const int spIndex = (pv.nonMirrorDepth == 0) ? 0 : sp.mSubpixelIndex;

    bool lightFilterNeedsSamples = CPP_LightFilterNeedsSamples(
        (const uniform Scene *uniform)(pbrTls->mFs->mScene));

    uniform int lightCount = LightSetSampler_getLightCount(&lSampler);
    varying int lightSampleCount = LightSetSampler_getLightSampleCount(&lSampler);

    {
        const uniform Light * uniform light = LightSetSampler_getLight(&lSampler, lightIndex);
        const uniform LightFilterList * uniform lightFilterList =
            LightSetSampler_getLightFilterList(&lSampler, lightIndex);

        // Setup sampler sequence
        // We want one shared sequence for depth 0.
        SequenceID_init(sid, (uint32_t)pv.nonMirrorDepth, sp.mPixel,
                Light_getHash(light), (uniform uint32_t)SequenceTypeLight,
                spIndex, sequenceID);
        IntegratorSample3D_init(lightSamples, sid);

        if (lightFilterNeedsSamples) {
            SequenceID_init(sidFilter, (uint32_t)pv.nonMirrorDepth, sp.mPixel,
                    Light_getHash(light), (uniform uint32_t)SequenceTypeLightFilter,
                    spIndex, sequenceID);
            IntegratorSample2D_init(lightFilterSamples, sidFilter);

            SequenceID_init(sidFilter3D, (uint32_t)pv.nonMirrorDepth, sp.mPixel,
                    Light_getHash(light), (uniform uint32_t)SequenceTypeLightFilter3D,
                    spIndex, sequenceID);
            IntegratorSample3D_init(lightFilterSamples3D, sidFilter3D);
        }

        if (pv.nonMirrorDepth == 0) {
            const int samplesSoFar = sp.mSubpixelIndex * lightSampleCount;
            lightSamples.mSampleNumber = samplesSoFar;
            if (lightFilterNeedsSamples) {
                lightFilterSamples.mSampleNumber = samplesSoFar;
                lightFilterSamples3D.mSampleNumber = samplesSoFar;
            }
        } else {
            lightSamples.mTotalSamples = lightSampleCount;
            if (lightFilterNeedsSamples) {
                lightFilterSamples.mTotalSamples = lightSampleCount;
                lightFilterSamples3D.mTotalSamples = lightSampleCount;
            }
        }

        // Loop over each light' samples
        for (uniform int i = 0, s = 0; i < lightSampleCount; ++i, ++s) {

            MNRY_ASSERT(s < LightSetSampler_getSampleCount(&lSampler));

            varying LightSample &currSamp = lsmp[s];

            snapshotLaneUtilization(pbrTls->mStatistics, STATS_VEC_LIGHT_SAMPLES_PRE);

            // Draw the sample and test validity
            varying Vec3f lightSample;
            getSample(lightSamples, &lightSample.x, pv.nonMirrorDepth, *pbrTls->mFs);

            varying LightFilterRandomValues lightFilterSample;
            if (lightFilterNeedsSamples) {
                getSample(lightFilterSamples, &lightFilterSample.r2.x, pv.nonMirrorDepth, *pbrTls->mFs);
                getSample(lightFilterSamples3D, &lightFilterSample.r3.x, pv.nonMirrorDepth, *pbrTls->mFs);
            }

            LightSetSampler_sampleIntersectAndEval(&lSampler, shadingTls, light, lightFilterList,
                    P, cullingNormal, lightFilterSample, time, lightSample, currSamp, rayDirFootprint);
            if (LightSample_isInvalid(&(currSamp))) {
                // These samples occur on the shadow terminator -- they are invalid because they face
                // away from the point (dot(n, wi) < epsilon). They should count as "misses" in the visibility aov.
                accumVisibilityAovsOccluded(pbrTls, bSampler, lSampler, 
                                            pv, rs->mSubpixel.mPixel, rs->mDeepDataHandle,
                                            light, 1);
                continue;
            }

            snapshotLaneUtilization(pbrTls->mStatistics, STATS_VEC_LIGHT_SAMPLES_POST);

            MNRY_ASSERT(isNormalized(currSamp.wi));

            integrateLightSetSample(lSampler, lightIndex, bSampler, pv, currSamp,
                    clampingDepth, clampingValue);

            addToCounter(pbrTls->mStatistics, STATS_LIGHT_SAMPLES, getActiveLaneCount());
        }
    }
}



//-----------------------------------------------------------------------------

void
applyRussianRoulette(
        uniform PbrTLState * uniform pbrTls,
        const varying BsdfSampler &bSampler, varying BsdfSample * uniform bsmp,
        const varying Subpixel &sp, const varying PathVertex &pv,
        varying uint32_t sequenceID,
        uniform float threshold, uniform float invThreshold)
{
    const varying int sampleCountVarying = BsdfSampler_getSampleCountVarying(&bSampler);
    const uniform int sampleCount = BsdfSampler_getSampleCount(&bSampler);
    SequenceID sid;
    SequenceID_init(sid, sp.mPixel, (uniform uint32_t)SequenceTypeRussianRouletteBsdf,
            (uint32_t)sp.mSubpixelIndex, sequenceID);
    IntegratorSample1D rrSamples;
    IntegratorSample1D_init(rrSamples, sid);
    rrSamples.mSampleNumber = pv.nonMirrorDepth * sampleCountVarying;

    // Cull rays from bsdf samples
    uniform int s = 0;
    for (int i = 0; i < sampleCount; ++i, ++s) {
        if (BsdfSample_isInvalid(&(bsmp[s]))) {
            continue;
        }

        const float lumDirect = luminance(bsmp[s].tDirect);
        const float lumIndirect = luminance(bsmp[s].tIndirect);
        const float lum = max(lumDirect, lumIndirect);
        if (lum < threshold) {
            // This should always be < 1
            //
            // The rcp function (SSE version) produces a NaN when the value is
            // less than 0x1p-64f (the version I tested, anyway). FLT_EPSILON
            // is much greater than this, but still probably a good threshold
            // for our minimum probability.
            const float continueProbability = max(sEpsilon, lum * invThreshold);
            float sample;
            getSample(rrSamples, sample, pv.nonMirrorDepth, *pbrTls->mFs);
            if (sample > continueProbability) {
                BsdfSample_setInvalid(&(bsmp[s]));
            } else {
                const float invContinueProbability = rcp(continueProbability);
                bsmp[s].tDirect = bsmp[s].tDirect * invContinueProbability;
                bsmp[s].tIndirect = bsmp[s].tIndirect * invContinueProbability;
            }
        }
    }
}


void
applyRussianRoulette(
        uniform PbrTLState * uniform pbrTls,
        const varying LightSetSampler &lSampler, varying LightSample * uniform lsmp,
        const varying Subpixel &sp, const varying PathVertex &pv,
        varying uint32_t sequenceID,
        uniform float threshold, uniform float invThreshold,
        IntegratorSample1D& rrSamples)
{
    const varying int lightSampleCount = LightSetSampler_getLightSampleCount(&lSampler);

    // Cull shadow rays from the light samples
    uniform int s = 0;
    for (int i = 0; i < lightSampleCount; ++i, ++s) {
        if (LightSample_isInvalid(&(lsmp[s]))) {
            continue;
        }

        const float lum = luminance(lsmp[s].t);
        if (lum < threshold) {
            // This should always be < 1
            //
            // The rcp function (SSE version) produces a NaN when the value is
            // less than 0x1p-64f (the version I tested, anyway). FLT_EPSILON
            // is much greater than this, but still probably a good threshold
            // for our minimum probability.
            const float continueProbability = max(sEpsilon, lum * invThreshold);
            float sample;
            getSample(rrSamples, sample, pv.nonMirrorDepth, *pbrTls->mFs);
            if (sample > continueProbability) {
                LightSample_setInvalid(&(lsmp[s]));
            } else {
                const float continueProbabilityInv = rcp(continueProbability);
                lsmp[s].t = lsmp[s].t * continueProbabilityInv;

                // adjust per lobe values, if needed (see integrateLightSetSample())
                for (uniform unsigned int k = 0; k < BSDF_MAX_LOBE; ++k) {
                    if (lsmp[s].lp.lobe[k]) {
                        lsmp[s].lp.t[k] = lsmp[s].lp.t[k] * continueProbabilityInv;
                    }
                }
            }
        }
    }
}

//-----------------------------------------------------------------------------

void
scatterAndScale(const varying Intersection &    isect,
                const varying BsdfLobe &        lobe,
                const varying Vec3f &           wo,
                const varying Vec3f &           wi,
                const varying float             scale,
                const varying float             r1,
                const varying float             r2,
                varying RayDifferential &       rd)
{
    MNRY_ASSERT(isNormalized(wi));

#ifdef FORCE_SKIP_RAY_DIFFERENTIALS

    Flags_clear(&rd.mFlags, RAY_HAS_DIFFERENTIALS);

#else

    if (Ray_hasDifferentials(rd)) {

        MNRY_ASSERT(isNormalized(rd.mDirX));
        MNRY_ASSERT(isNormalized(rd.mDirY));

        varying Vec3f dDdx = Ray_getdDdx(rd);
        varying Vec3f dDdy = Ray_getdDdy(rd);

        BsdfLobe_differentials(&lobe, wo, wi, r1, r2,
                getdNdx(isect), getdNdy(isect), dDdx, dDdy);

        varying const Vec3f &origin = Ray_getOrigin(rd);
        rd.mOriginX = origin + (rd.mOriginX - origin) * scale;
        rd.mOriginY = origin + (rd.mOriginY - origin) * scale;

        // Scaling dDdx is more accurate than scaling (mDirX - direction)
        // because of the normalization that happens when converting from
        // dDdx to mDirX
        dDdx = dDdx * scale;
        dDdy = dDdy * scale;

        rd.mDirX = normalize(wi + dDdx);
        rd.mDirY = normalize(wi + dDdy);
    }

#endif  // FORCE_SKIP_RAY_DIFFERENTIALS

    rd.dir = wi;
}


