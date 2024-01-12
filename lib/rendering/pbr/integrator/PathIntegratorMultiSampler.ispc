// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

/// @file PathIntegratorMultiSampler.ispc

// Integration implementation based on the BsdfSampler
// 
// When making changes to this file, you'll likely also need
// to update the scalar implementation and the one-sample
// integrator (both vector and scalar).
//   PathIntegratorMultiSampler.cc

#include "PathIntegrator.isph"

#include "BsdfSampler.isph"
#include "LightSetSampler.isph"
#include "PathIntegratorUtil.isph"

#include <moonray/rendering/pbr/Types.isph>
#include <moonray/rendering/pbr/core/Aov.isph>
#include <moonray/rendering/pbr/core/PbrTLState.isph>
#include <moonray/rendering/pbr/core/RayState.isph>
#include <moonray/rendering/pbr/light/LightSet.isph>

// Returns the number of BundledOcclRay entries filled and added to the index list.
inline uniform uint32_t
fillOcclRay(       uniform PbrTLState *      uniform pbrTls,
                   varying BundledOcclRay *  uniform dst,
                   uniform uint32_t *        uniform dstIndices,
                   varying uint32_t                  indices,
             const varying RayDifferential &         parentRay,
             const varying Vec3f &                   dir,
                   varying float                     minT,
                   varying float                     maxT,
             const varying Color &                   radiance,
             const varying RayState &                rayState,
                   varying uint32_t                  dataPtrHandle,
                   varying OcclTestType              occlTestType,
             const varying int32_t                   assignmentId)
{
    uniform uint32_t numNewEntries = 0;

    // This matches the behavior of the scalar codepath.
    if (!isBlack(radiance) && maxT > minT) {

        MNRY_ASSERT(isNormalized(dir));

        BundledOcclRay_init(dst,
                            parentRay.org,
                            dir,
                            minT,
                            maxT,
                            parentRay.time,
                            parentRay.ext.depth + 1,
                            radiance,
                            rayState.mSubpixel.mPixel,
                            rayState.mSubpixel.mSubpixelIndex,
                            rayState.mSequenceID,
                            rayState.mTilePass,
                            dataPtrHandle,
                            rayState.mDeepDataHandle,
                            rayState.mCryptomatteDataHandle,
                            rayState.mCryptoRefP,
                            rayState.mCryptoRefN,
                            rayState.mCryptoUV,
                            occlTestType,
                            assignmentId);
        PbrTLState_acquireDeepData(pbrTls, dst->mDeepDataHandle);
        PbrTLState_acquireCryptomatteData(pbrTls, dst->mCryptomatteDataHandle);

        // packed_store_active2() "may write one extra element to the output array"
        // according to ISPC documentation https://ispc.github.io/ispc.html.
        // Please make sure that 1 extra element was allocated for the output array.
        numNewEntries = packed_store_active2(dstIndices, indices);

        snapshotLaneUtilization(pbrTls->mStatistics, STATS_VEC_FILL_OCCL_RAY);

    } else {

        // Data
        // we are responsible for freeing the data memory of rays that are not
        // queued.  we also assume that a black radiance implies that all the
        // aov values are also 0.  I don't think we have negative radiance values
        // floating around our system
        if (dataPtrHandle != PBR_nullHandle) {
            PbrTLState_freeList(pbrTls, dataPtrHandle);
        }
        dst->mDataPtrHandle = PBR_nullHandle;
    }

    return numNewEntries;
}

// fill out occlusion or presence shadow ray based on light attribute
// (whether this light request presence shadow)
void
prepareOcclusionTestRays(
              uniform PbrTLState *      uniform pbrTls,
        const uniform Light *           varying light,
        const varying RayState &                rayState,
        const varying RayDifferential &         parentRay,
              varying float                     rayEpsilon,
        const varying float                     shadowRayEpsilon,
        const varying Vec3f &                   wi,
        const varying Color &                   tDirect,
              varying float                     distance,
              varying uint32_t &                bundledOcclRayData,
              varying uint32_t                  srcRayIndices,
              varying BundledOcclRay *  uniform dstOcclRays,
              uniform uint32_t *        uniform dstOcclRayIndices,
              uniform uint32_t &                numOcclRays,
              varying BundledOcclRay *  uniform dstPresenceShadowRays,
              uniform uint32_t *        uniform dstPresenceShadowsRayIndices,
              uniform uint32_t &                numPresenceShadowsRays,
              varying OcclTestType &            occlTestType,
              const varying Intersection &      isect)
{
    // Presence handling code for direct lighting
    // The shadow ray will be submitted to its own
    // presence shadow ray handling queue
    if (bundledOcclRayData == PBR_nullHandle) {
        // we don't have any bundledOcclRayData for this ray yet, create some
        bundledOcclRayData = PbrTLState_allocList(pbrTls,
            sizeof(uniform BundledOcclRayData), 1);
        MNRY_ASSERT(bundledOcclRayData != PBR_nullHandle);
        // if we allocate BundledOcclRayData in this late stage,
        // we are using it for non AOV purpose (otherwise
        // it would has been allocated in earlier stage)
        uniform BundledOcclRayData * varying b =
            (uniform BundledOcclRayData * varying)PbrTLState_getListItem(
            pbrTls, bundledOcclRayData, 0);
        b->mFlags = 0;
        b->mLpeStateId = 0;
    }
    uniform BundledOcclRayData * varying b =
        (uniform BundledOcclRayData * varying)PbrTLState_getListItem(
        pbrTls, bundledOcclRayData, 0);

    rayEpsilon = max(rayEpsilon, shadowRayEpsilon);
    float maxT;

    // if falloff enabled, don't shorten ray -- instead, use color interpolation (more expensive)
    float clearRadius = light->mClearRadiusFalloffDistance > 0.f ? 0.f : light->mClearRadius;

    // 2 * rayEpsilon ensures that the ray has non-zero length. 
    // Zero length shadow rays are not queued, which in this case we don't want.
    maxT = max((distance - clearRadius) * sHitEpsilonEnd, rayEpsilon * 2);
    if (light->mMaxShadowDistance > 0) {
        maxT = min(maxT, light->mMaxShadowDistance);
    }

    int32_t assignmentId = isect.mLayerAssignmentId;

    if (light->mPresenceShadows) {
        // These members are for presence shadows
        b->mRayEpsilon = rayEpsilon;
        // pass in light for shadow linking purpose
        b->mLight = light;
        numPresenceShadowsRays += fillOcclRay(pbrTls,
            dstPresenceShadowRays,
            &dstPresenceShadowsRayIndices[numPresenceShadowsRays],
            srcRayIndices,
            parentRay, wi,
            rayEpsilon, maxT,
            tDirect, rayState, bundledOcclRayData, occlTestType, assignmentId);
    } else {
        // pass in light for shadow linking purpose
        b->mLight = light;
        numOcclRays += fillOcclRay(pbrTls,
            dstOcclRays,
            &dstOcclRayIndices[numOcclRays],
            srcRayIndices,
            parentRay, wi,
            rayEpsilon, maxT,
            tDirect, rayState, bundledOcclRayData, occlTestType, assignmentId);
    }
}

//-----------------------------------------------------------------------------

void
addDirectVisibleBsdfLobeSampleContributionBundled( const uniform PathIntegrator *  uniform this,
                                                         uniform PbrTLState *      uniform pbrTls,
                                                   const varying BsdfSampler &             bSampler,
                                                         varying BsdfSample *      uniform currSamp,
                                                   const uniform int                       lobeIndex,
                                                   const varying bool                      doIndirect,
                                                   const varying RayDifferential &         parentRay,
                                                   const varying float                     rayEpsilon,
                                                   const varying float                     shadowRayEpsilon,
                                                   const varying RayState &                rayState,
                                                         varying uint32_t                  srcOcclRayIndices,
                                                         varying BundledOcclRay *  uniform dstOcclRays,
                                                         uniform uint32_t *        uniform dstOcclRayIndices,
                                                         uniform uint32_t &                numOcclRays,
                                                         varying BundledOcclRay *  uniform presenceShadowsRays,
                                                         uniform uint32_t *        uniform dstPresenceShadowsRayIndices,
                                                         uniform uint32_t &                numPresenceShadowsRays,
                                                         const varying Intersection &      isect)
{
    const uniform Light * varying light = (const uniform Light * varying) currSamp->lp.light;
    const varying Bsdf * uniform bsdf = BsdfSampler_getBsdf(&bSampler);
    const varying BsdfLobe * uniform lobe = BsdfSampler_getLobe(&bSampler, lobeIndex);

    // Ray termination lights are used in an attempt to cheaply fill in the zeros which result from
    // terminating ray paths too early, which is done by forcing the occlusion test to fail unconditionally.
    // A path can only be terminated when doIndrect is false, which we test here, and we don't apply a ray
    // termination light unless we have a (non-hair) transmission lobe.
    varying OcclTestType occlTestType = STANDARD;
    if (!doIndirect && Light_getIsRayTerminator(light)) {
        if (!BsdfLobe_matchesFlags(lobe, BSDF_LOBE_TYPE_ALL_TRANSMISSION) || BsdfLobe_getIsHair(lobe)) {
            return;
        } else {
            occlTestType = FORCE_NOT_OCCLUDED;
        }
    }

    snapshotLaneUtilization(pbrTls->mStatistics, STATS_VEC_ADD_DIRECT_VISIBLE_BSDF);

    // LPE
    varying uint32_t bundledOcclRayData = PBR_nullHandle;
    const uniform FrameState &fs = *pbrTls->mFs;
    const uniform AovSchema &aovSchema = *((const uniform AovSchema * uniform) fs.mAovSchema);
    if (!AovSchema_empty(aovSchema)) {
        int lpeStateId = rayState.mPathVertex.lpeStateId;
        if (lpeStateId >= 0) {
            const uniform LightAovs * uniform lightAovs =
                (const uniform LightAovs * uniform) pbrTls->mFs->mLightAovs;

            // transition
            lpeStateId = LightAovs_scatterEventTransition(lightAovs, lpeStateId, *bsdf, *lobe);
            if (lpeStateId >= 0) {
                // compute the state id as-if we hit the light,
                // the occlusionQueryHandler will need to actually compute
                // the occlusion
                lpeStateId = LightAovs_lightEventTransition(lightAovs, lpeStateId, light);
            }
            if (lpeStateId >= 0) {
                bundledOcclRayData = PbrTLState_allocList(pbrTls, sizeof(uniform BundledOcclRayData), 1);
                MNRY_ASSERT(bundledOcclRayData != PBR_nullHandle);
                if (bundledOcclRayData != PBR_nullHandle) {
                    uniform BundledOcclRayData * varying b =
                        (uniform BundledOcclRayData * varying) PbrTLState_getListItem(pbrTls, bundledOcclRayData, 0);
                    b->mLpeRadiance = currSamp->tDirect; // Scatter required
                    b->mLpeStateId = lpeStateId;
                    b->mFlags = LPE;
                }
            }
        }
    }

    prepareOcclusionTestRays(pbrTls, light, rayState, parentRay,
        rayEpsilon, shadowRayEpsilon, currSamp->wi, currSamp->tDirect, currSamp->distance, bundledOcclRayData,
        srcOcclRayIndices, dstOcclRays, dstOcclRayIndices, numOcclRays,
        presenceShadowsRays, dstPresenceShadowsRayIndices,
        numPresenceShadowsRays, occlTestType, isect);
}


void
addDirectVisibleBsdfSampleContributionsBundled( const uniform PathIntegrator * uniform this,
                                                      uniform PbrTLState *     uniform pbrTls,
                                                const varying BsdfSampler &            bSampler,
                                                const varying bool                     doIndirect,
                                                      varying BsdfSample *     uniform bsmp,
                                                const varying RayDifferential &        parentRay,
                                                const varying float                    rayEpsilon,
                                                const varying float                    shadowRayEpsilon,
                                                const varying RayState &               rayState,
                                                      varying uint32_t                 srcOcclRayIndices,
                                                      varying BundledOcclRay * uniform dstOcclRays,
                                                      uniform uint32_t *       uniform dstOcclRayIndices,
                                                      uniform uint32_t &               numOcclRays,
                                                      varying BundledOcclRay * uniform presenceShadowsRays,
                                                      uniform uint32_t *       uniform dstPresenceShadowsRayIndices,
                                                      uniform uint32_t &               numPresenceShadowsRays,
                                                      const varying Intersection &     isect)
{
    MNRY_ASSERT(CPP_isIntegratorAccumulatorRunning(pbrTls));

    uniform int sampleCount = BsdfSampler_getSampleCount(&bSampler);

    // Trace bsdf sample shadow rays
    const uniform int lobeCount = BsdfSampler_getLobeCount(&bSampler);
    for (uniform int lobeIndex = 0; lobeIndex < lobeCount; ++lobeIndex) {
        const int lobeSampleCount = BsdfSampler_getLobeSampleCount(&bSampler, lobeIndex);
        uniform int s = bSampler.mLobeToSampleIndex[lobeIndex];
        for (int i = 0; i < lobeSampleCount; ++i, ++s) {
            if (BsdfSample_isValid(&bsmp[s]) && BsdfSample_didHitLight(&bsmp[s])) {
                addDirectVisibleBsdfLobeSampleContributionBundled(this, pbrTls, bSampler, &bsmp[s], lobeIndex, doIndirect,
                    parentRay, rayEpsilon, shadowRayEpsilon, rayState,
                    srcOcclRayIndices + s * VLEN, &dstOcclRays[s], dstOcclRayIndices, numOcclRays,
                    &presenceShadowsRays[s], dstPresenceShadowsRayIndices, numPresenceShadowsRays, isect);
            }
        }
    }
}

void
addDirectVisibleLightSampleContributionsBundled( const uniform PathIntegrator * uniform this,
                                                       uniform PbrTLState *     uniform pbrTls,
                                                       uniform ShadingTLState * uniform shadingTls,
                                                 const varying LightSetSampler &        lSampler,
                                                 const varying BsdfSampler &            bSampler,
                                                       varying LightSample *    uniform lsmp,
                                                 const varying RayDifferential &        parentRay,
                                                 const varying Vec3f *          uniform cullingNormal,
                                                 const varying float                    rayEpsilon,
                                                 const varying float                    shadowRayEpsilon,
                                                 const varying Subpixel &               sp,
                                                 const varying PathVertex &             pv,
                                                 const varying RayState &               rayState,
                                                       varying uint32_t                 srcRayIndices,
                                                       varying BundledOcclRay * uniform dstOcclRays,
                                                       uniform uint32_t *       uniform dstOcclRayIndices,
                                                       uniform uint32_t &               numOcclRays,
                                                       varying BundledOcclRay * uniform presenceShadowsRays,
                                                       uniform uint32_t *       uniform dstPresenceShadowsRayIndices,
                                                       uniform uint32_t &               numPresenceShadowsRays,
                                                       const varying Intersection &     isect,
                                                       varying uint32_t &               sequenceID)
{
    MNRY_ASSERT(CPP_isIntegratorAccumulatorRunning(pbrTls));

    varying int lightSampleCount = LightSetSampler_getLightSampleCount(&lSampler);
    uniform int lightCount = LightSetSampler_getLightCount(&lSampler);
    const uniform int sampleCount = LightSetSampler_getSampleCount(&lSampler);

    SequenceID sid;
    SequenceID_init(sid, sp.mPixel, (uniform uint32_t)SequenceTypeRussianRouletteLight,
            (uint32_t)sp.mSubpixelIndex, sequenceID);
    IntegratorSample1D rrSamples;
    IntegratorSample1D_init(rrSamples, sid);
    rrSamples.mSampleNumber = pv.nonMirrorDepth * sampleCount;

    // Trace light sample shadow rays
    for (uniform int lightIndex = 0, s = 0; lightIndex < lightCount; lightIndex++) {

        const uniform Light * uniform light = LightSetSampler_getLight(&lSampler, lightIndex);

        // Draw light samples from the light and compute tentative contributions
        drawLightSetSamples(pbrTls, shadingTls, lSampler, bSampler, sp, pv, getP(isect),
                            *cullingNormal, Ray_getTime(parentRay), sequenceID, lsmp,
                            this->mSampleClampingDepth, sp.mSampleClampingValue, 
                            Ray_getDirFootprint(parentRay), &rayState, lightIndex);

        // Apply Russian Roulette to the light samples
        if (pv.nonMirrorDepth > 0  &&  this->mRussianRouletteThreshold > 0.0f) {
            applyRussianRoulette(pbrTls, lSampler, lsmp, sp, pv, sequenceID,
                                this->mRussianRouletteThreshold, 
                                this->mInvRussianRouletteThreshold,
                                rrSamples);
        }

        for (uniform int is = 0; is < lightSampleCount; s++, is++) {

            // Ray termination lights are used in an attempt to cheaply fill in the zeros which result from
            // terminating ray paths too early. To accomplish this, we force the occlusion test result to unoccluded.
            varying OcclTestType occlTestType = STANDARD;
            if (Light_getIsRayTerminator(light)) {
                occlTestType = FORCE_NOT_OCCLUDED;
            }

            varying LightSample *uniform currSamp = &lsmp[is];

            if (LightSample_isValid(currSamp)) {

                // Only trace up to the lsmp[s].distance. It is set to the distance to
                // the intersected light in this direction, if any.
                snapshotLaneUtilization(pbrTls->mStatistics, STATS_VEC_ADD_DIRECT_VISIBLE_LIGHTING);

                // LPE
                varying uint32_t data = PBR_nullHandle;
                const uniform FrameState &fs = *pbrTls->mFs;
                const uniform AovSchema &aovSchema = *((const uniform AovSchema * uniform) fs.mAovSchema);
                if (!AovSchema_empty(aovSchema)) {
                    const LightSampleLPE &lp = lsmp[is].lp;
                    const uniform LightAovs * uniform lightAovs =
                        (const uniform LightAovs * uniform) pbrTls->mFs->mLightAovs;

                    varying int lpeStateId[BSDF_MAX_LOBE];
                    varying int numItems = 0;

                    // LightSample contains up to BSDF_MAX_LOBE
                    for (uniform int i = 0; i < BSDF_MAX_LOBE; ++i) {
                        if (!isActive(lp.lobe[i])) {
                            lpeStateId[i] = -1;
                            continue;
                        }

                        const varying BsdfLobe &lobe = *lp.lobe[i];
                        const varying Bsdf &bsdf = *LightSetSampler_getBsdf(&lSampler);

                        // transition
                        lpeStateId[i] = rayState.mPathVertex.lpeStateId;
                        lpeStateId[i] = LightAovs_scatterEventTransition(lightAovs, lpeStateId[i], bsdf, lobe);
                        if (lpeStateId[i] >= 0) {
                            // compute the state id as-if we hit the light,
                            // the occlusionQueryHandler will need to actually compute
                            // the occlusion
                            lpeStateId[i] = LightAovs_lightEventTransition(lightAovs, lpeStateId[i], light);
                        }
                        if (lpeStateId[i] < 0) continue;

                        ++numItems;
                    }

                    if (numItems > 0) {
                        data = PbrTLState_allocList(pbrTls, sizeof(uniform BundledOcclRayData), numItems);
                    }

                    varying int next = 0;
                    for (uniform int i = 0; i < BSDF_MAX_LOBE; ++i) {
                        if (lpeStateId[i] < 0) continue;
                        MNRY_ASSERT(data != PBR_nullHandle);
                        MNRY_ASSERT(numItems > 0);
                        uniform BundledOcclRayData * varying b =
                            (uniform BundledOcclRayData * varying) PbrTLState_getListItem(pbrTls, data, next);
                        b->mFlags = LPE | LIGHT_SAMPLE;
                        b->mLpeRadiance = lp.t[i]; // Scatter required
                        b->mLpeStateId = lpeStateId[i];
                        ++next;
                        if (next < numItems) {
                            b = (uniform BundledOcclRayData * varying) PbrTLState_getListItem(pbrTls, data, next);
                            b->mFlags = LPE | LIGHT_SAMPLE;
                            b->mLpeStateId = 0;
                        }
                    }
                }

                prepareOcclusionTestRays(pbrTls, light, rayState, parentRay,
                    rayEpsilon, shadowRayEpsilon, currSamp->wi, currSamp->t, currSamp->distance,
                    data, srcRayIndices,
                    &dstOcclRays[s], dstOcclRayIndices,
                    numOcclRays,
                    &presenceShadowsRays[s], dstPresenceShadowsRayIndices,
                    numPresenceShadowsRays, occlTestType, isect);
            }

            // TODO: Ray debugging.

            unmasked {
                srcRayIndices += VLEN;
            }
        }
    }

    // tldr; Add inactive lights to the visibility aov
    // In order to encompass all of the cases where the point's normal faces away from the light, we have to
    // consider inactive lights, which are culled from the visible light set because the whole light faces away
    // from the intersection in question. This code sorts through all the lights in the accelerator and finds the ones 
    // that have been marked invalid. It then adds the appropriate number of "misses" to the visibility aov.
    /// NOTE: if we ever switch entirely to adaptive light sampling, we won't need the visible light 
    /// set, since the algorithm should ignore those lights anyway. That would make this portion of code unnecessary
    const uniform FrameState &fs = *pbrTls->mFs;
    const uniform AovSchema &aovSchema = *((const uniform AovSchema * uniform) fs.mAovSchema);
    const uniform LightAovs* uniform lightAovs = (const uniform LightAovs * uniform) fs.mLightAovs;

    if (!AovSchema_empty(aovSchema) && LightAovs_hasVisibilityEntries(lightAovs)) {
        uniform int totalLightCount = LightSetSampler_getAccLightCount(&lSampler); // active + inactive lights
        for (uniform int i = 0; i < totalLightCount; ++i) {
            if (!LightSetSampler_lightIsActive(&lSampler, i)) {
                const uniform Light * light = LightSetSampler_getLightFromAcc(&lSampler, i);
                accumVisibilityAovsOccluded(pbrTls, bSampler, lSampler, 
                                            pv, rayState.mSubpixel.mPixel, rayState.mDeepDataHandle, light, 
                                            PathIntegrator_getLightSampleCount(pbrTls->mFs->mIntegrator));
            }
        }
    }  
}


// This is a modified version of addIndirectOrDirectVisibleContributions,
// which is altered to handle bundling.
void
addIndirectOrDirectVisibleContributionsBundled( const uniform PathIntegrator * uniform this,
                                                      uniform PbrTLState *     uniform pbrTls,
                                                const varying Subpixel &               sp,
                                                const varying PathVertex &             parentPv,
                                                const varying BsdfSampler &            bSampler,
                                                      varying BsdfSample *     uniform bsmp,
                                                const varying RayDifferential &        parentRay,
                                                const varying float                    rayEpsilon,
                                                const varying float                    shadowRayEpsilon,
                                                const varying Intersection &           isect,
                                                      varying BsdfLobeType             indirectFlags,
                                                      varying uint32_t &               sequenceID,
                                                const varying RayState &               parentRayState )
{
    MNRY_ASSERT(CPP_isIntegratorAccumulatorRunning(pbrTls));

    varying Vec3f wo = parentRay.dir * -1.f;

    uniform Arena * uniform arena = pbrTls->mArena;

    const uniform FrameState &fs = *pbrTls->mFs;
    uniform Color dummyRadiance = sBlack;

    uniform int sampleCount = BsdfSampler_getSampleCount(&bSampler);

    uniform uint8_t *uniform memoryBookmark1 = Arena_getPtr(arena);
    varying RayState *uniform spawnedRays = (varying RayState *uniform)
        Arena_allocArray(arena, sampleCount, sizeof(varying RayState));
    uniform uint32_t *uniform spawnedRayIndices = (uniform uint32_t *uniform)
        Arena_allocArray(arena, sampleCount * VLEN + 1, sizeof(uniform uint32_t));
    uniform uint32_t numSpawnedRays = 0;

    uniform uint8_t *uniform memoryBookmark2 = Arena_getPtr(arena);
    varying BundledOcclRay *uniform spawnedOcclRays = (varying BundledOcclRay *uniform)
        Arena_allocArray(arena, sampleCount, sizeof(varying BundledOcclRay));
    uniform uint32_t *uniform spawnedOcclRayIndices = (uniform uint32_t *uniform)
        Arena_allocArray(arena, sampleCount * VLEN + 1, sizeof(uniform uint32_t));
    uniform uint32_t numSpawnedOcclRays = 0;

    varying BundledOcclRay *uniform spawnedPresenceRays = (varying BundledOcclRay *uniform)
        Arena_allocArray(arena, sampleCount, sizeof(varying BundledOcclRay));
    uniform uint32_t *uniform spawnedPresenceRayIndices = (uniform uint32_t *uniform)
        Arena_allocArray(arena, sampleCount * VLEN + 1, sizeof(uniform uint32_t));
    uniform uint32_t numSpawnedPresenceRays = 0;

    snapshotLaneUtilization(pbrTls->mStatistics, STATS_VEC_INDIRECT_A);

    // Trace bsdf sample continuation rays. We accumulate either direct or
    // indirect lighting contributions accordingly
    uniform int lobeCount = BsdfSampler_getLobeCount(&bSampler);
    for (uniform int lobeIndex = 0; lobeIndex < lobeCount; ++lobeIndex) {
        const varying BsdfLobe * uniform lobe = BsdfSampler_getLobe(&bSampler, lobeIndex);
        MNRY_ASSERT(CPP_Arena_isValidPtr(arena, lobe));
        if (!isActive(lobe)) {
            continue;
        }

        snapshotLaneUtilization(pbrTls->mStatistics, STATS_VEC_INDIRECT_B);

        // hair lobe type is a special case that often requires more bounces.
        const varying bool hairLobe = BsdfLobe_getIsHair(lobe);
        const varying bool doIndirect = BsdfLobe_matchesFlags(lobe, indirectFlags)
                                        || (hairLobe && (parentPv.hairDepth < this->mMaxHairDepth));
        const varying bool diffuseLobe = BsdfLobe_matchesFlags(lobe, BSDF_LOBE_TYPE_ALL_DIFFUSE);
        const varying bool glossyLobe = BsdfLobe_matchesFlags(lobe, BSDF_LOBE_TYPE_ALL_GLOSSY);
        const varying bool mirrorLobe = BsdfLobe_matchesFlags(lobe, BSDF_LOBE_TYPE_ALL_MIRROR);

        const varying Vec2f minRoughness = (doIndirect  ?  computeMinRoughness(lobe,
                this->mRoughnessClampingFactor, parentPv.minRoughness)  :  Vec2f_ctor(0.0f));

        // Loop over each lobe's samples
        varying int lobeSampleCount = BsdfSampler_getLobeSampleCount(&bSampler, lobeIndex);
        uniform int s = bSampler.mLobeToSampleIndex[lobeIndex];
        for (int i = 0; i < lobeSampleCount; ++i, ++s) {

            MNRY_ASSERT(s < sampleCount);

            const varying uint32_t rayIndices = s * VLEN + programIndex;

            if (BsdfSample_isInvalid(&bsmp[s])) {
                continue;
            }

            snapshotLaneUtilization(pbrTls->mStatistics, STATS_VEC_INDIRECT_C);

            if (BsdfSample_didHitLight(&bsmp[s])) {
                addDirectVisibleBsdfLobeSampleContributionBundled(this, pbrTls, bSampler, &bsmp[s], lobeIndex, doIndirect,
                    parentRay, rayEpsilon, shadowRayEpsilon, parentRayState,
                    rayIndices, &spawnedOcclRays[s], spawnedOcclRayIndices, numSpawnedOcclRays,
                    &spawnedPresenceRays[s], spawnedPresenceRayIndices,
                    numSpawnedPresenceRays, isect);
            }

            snapshotLaneUtilization(pbrTls->mStatistics, STATS_VEC_INDIRECT_D);
            if (!doIndirect) {
                continue;
            }

            // We have some self-intersections when rays leave at grazing
            // angle, so we adjust the rayEpsilon accordingly.
            // We only trace up to the currSamp->distance. It is set to the distance
            // to the intersected light in this direction, if any.
            varying BsdfSample *uniform currSamp = &bsmp[s];
            const varying float denom = abs(dot(getNg(isect), currSamp->wi));
            // isect.getNg() itself or the dot product above can be zero.
            const varying float start = isZero(denom) ? rayEpsilon : rayEpsilon / denom;
            float varying end = (currSamp->distance == sMaxValue  ?  sMaxValue  :
                                 currSamp->distance * sHitEpsilonEnd);

            if (end <= start) {
                continue;
            }

            // Check transparency threshold
            varying float newAccumOpacity;
            if (mirrorLobe && BsdfLobe_matchesFlags(lobe, BSDF_LOBE_TYPE_ALL_TRANSMISSION)) {
                varying float lobeTransparency = reduceTransparency(currSamp->f);
                newAccumOpacity = parentPv.accumOpacity + (1 - lobeTransparency) * (1 - parentPv.accumOpacity);
                if (newAccumOpacity > this->mTransparencyThreshold) {
                    continue;
                }
            } else {
                newAccumOpacity = parentPv.accumOpacity;
            }

            snapshotLaneUtilization(pbrTls->mStatistics, STATS_VEC_INDIRECT_E);

            varying RayState *uniform rayState = &spawnedRays[s];
            
            // packed_store_active2() "may write one extra element to the output array"
            // according to ISPC documentation https://ispc.github.io/ispc.html.
            // Please make sure that 1 extra element was allocated for the output array.
            numSpawnedRays += packed_store_active2(&spawnedRayIndices[numSpawnedRays], rayIndices);

            // Prepare a path vertex
            varying PathVertex &pv = rayState->mPathVertex;
            pv.pathThroughput = currSamp->tIndirect;
            // TODO: Don't set this to parentPv.pathPixelWeight here as this
            // messes up the weight accumulation in Film::addSampleBundleHandler
            pv.pathPixelWeight = 0.0f;
            // Use previous path pixel weight for aovPathPixelWeight as there's existing logic
            // in vector mode that sometimes assumes that pv.pathPixelWeight = 0.  Thus, we must seperately
            // keep track of the pathPixelWeight for aovs.  See comment in PathIntegratorMultiSampler.ispc::
            // addIndirectOrDirectVisibleContributionsBundled().
            pv.aovPathPixelWeight = parentPv.pathPixelWeight;
            pv.pathDistance = parentPv.pathDistance + Ray_getEnd(parentRay);
            pv.minRoughness = minRoughness;
            pv.diffuseDepth = parentPv.diffuseDepth + (diffuseLobe ? 1 : 0);
            pv.glossyDepth = parentPv.glossyDepth + (glossyLobe ? 1 : 0);
            pv.mirrorDepth = parentPv.mirrorDepth + (mirrorLobe ? 1 : 0);
            pv.nonMirrorDepth = parentPv.nonMirrorDepth + (mirrorLobe ? 0 : 1);
            pv.hairDepth = parentPv.hairDepth + (hairLobe ? 1 : 0);
            pv.volumeDepth = parentPv.volumeDepth + 1;
            pv.presenceDepth = parentPv.presenceDepth;
            pv.totalPresence = parentPv.totalPresence;
            pv.subsurfaceDepth = parentPv.subsurfaceDepth;
            pv.accumOpacity = newAccumOpacity;
            pv.lobeType = BsdfLobe_getType(lobe);

            // LPE
            const uniform AovSchema &aovSchema = *((const uniform AovSchema * uniform) fs.mAovSchema);
            if (!AovSchema_empty(aovSchema)) {
                const uniform LightAovs * uniform lightAovs =
                    (const uniform LightAovs * uniform) fs.mLightAovs;

                // transition
                // This is the complicated case.  We are creating a new ray that
                // will be handled in rayBundleHandler.  This ray may hit no
                // geometry.  In that case we'll want a "hit light" event in the
                // lpeStateId field correspodning to currSamp->lp.light.  Alternatively,
                // this ray may hit geometry.  In which case, we do not want the
                // hit light event.  We store both values in pv (lpeStateId,
                // lpeStateIdLight) and let rayHandler sort out the different cases.
                const varying Bsdf * uniform bsdf = BsdfSampler_getBsdf(&bSampler);
                pv.lpeStateId = LightAovs_scatterEventTransition(lightAovs, parentPv.lpeStateId, *bsdf, *lobe);
                pv.lpeStateIdLight = -1;
                if (pv.lpeStateId >= 0) {
                    const uniform Light * varying light =
                        (const uniform Light * varying) currSamp->lp.light;
                    pv.lpeStateIdLight = LightAovs_lightEventTransition(lightAovs, pv.lpeStateId, light);
                }
            }

            // Prepare a RayDifferential
            // Ray::depth is incremented inside of this call.
            RayDifferential &ray = rayState->mRay;
            Ray_init(&ray, parentRay, start, end);

            // Scatter and scale our next ray differential
            // We scatter the ray based on the sampled lobe
            scatterAndScale(isect, *lobe, wo, currSamp->wi,
                    ((BsdfLobe_getDifferentialFlags(lobe) & BSDF_LOBE_IGNORES_INCOMING_DIFFERENTIALS)
                                ? sp.mPrimaryRayDiffScale : 1.0f),
                    currSamp->sample.x, currSamp->sample.y,
                    ray);

            //
            // Finish filling in RayState data.
            //

            // We figure out the ray mask based on the lobe type.
            // For ray mask propagation, we simply merge with existing ray mask.
            // We should exclude the camera ray mask when propagating.
            int lobeMask = lobeTypeToRayMask(BsdfLobe_getType(lobe));
            ++sequenceID;

            // To mimic the scalar logic, we need to record all the values of interest
            // and feed them forward so that future logic can decide which ones are
            // relevant and which aren't.
            //
            // The radiance and transparency we've accumulated up to this point will
            // be added to the frame buffer via the radiance queue elsewhere, so
            // here we are just interested in accumulating the *additional* radiance
            // and transparency which comes from recursion.
            //
            // We allocate a new RayState, fill it in, and spawn a new
            // ray via the incoherent ray queue (since we know the newly spawned ray
            // can't be a primary ray).
            //
            // if ray hits a surface then
            //     add indirect radiance contribution from that surface to radiance
            //     add indirect transparency contribution to transparency
            //     This is the typical path. We mimic this behavior by simply spawning
            //     the new ray and letting it flow through the pipeline. The next 2
            //     cases are the edge cases.
            rayState->mRay.mask = fs.mPropagateVisibilityBounceType
                ? ((parentRayState.mRay.mask | lobeMask) & ~RTC_MASK_RAY_CAMERA)
                : lobeMask;

            rayState->mSequenceID = sequenceID;
            rayState->mSubpixel.mPixel = parentRayState.mSubpixel.mPixel;
            rayState->mSubpixel.mSubpixelIndex = sp.mSubpixelIndex;
            rayState->mSubpixel.mSubpixelX = sp.mSubpixelX;
            rayState->mSubpixel.mSubpixelY = sp.mSubpixelY;
            rayState->mSubpixel.mPixelSamples = sp.mPixelSamples;
            rayState->mSubpixel.mSampleClampingValue = sp.mSampleClampingValue;
            rayState->mSubpixel.mPrimaryRayDiffScale = sp.mPrimaryRayDiffScale;
            rayState->mSubpixel.mTextureDiffScale = sp.mTextureDiffScale;
            rayState->mTilePass = parentRayState.mTilePass;
            rayState->mDeepDataHandle = parentRayState.mDeepDataHandle;
            // Since cryptomatte records radiance information, its handle actually also needs to be acquired
            rayState->mCryptomatteDataHandle = parentRayState.mCryptomatteDataHandle;
            rayState->mCryptoRefP = parentRayState.mCryptoRefP;
            rayState->mCryptoRefN = parentRayState.mCryptoRefN;
            rayState->mCryptoUV = parentRayState.mCryptoUV;
            PbrTLState_acquireDeepData(pbrTls, rayState->mDeepDataHandle);
            PbrTLState_acquireCryptomatteData(pbrTls, rayState->mCryptomatteDataHandle);

            // Accumulate post scatter extra aovs
            if (!AovSchema_empty(aovSchema)) {
                const varying Bsdf * uniform bsdf = BsdfSampler_getBsdf(&bSampler);
                aovAccumPostScatterExtraAovs(pbrTls, fs, rayState, *bsdf);
            }

            snapshotLaneUtilization(pbrTls->mStatistics, STATS_VEC_INDIRECT_F);
        }
    }

    if (numSpawnedOcclRays) {
        MNRY_ASSERT(numSpawnedOcclRays <= sampleCount * VLEN);
        CPP_addOcclusionQueueEntries(pbrTls, spawnedOcclRays, numSpawnedOcclRays,
                                     spawnedOcclRayIndices);
    }

    if (numSpawnedPresenceRays) {
        MNRY_ASSERT(numSpawnedPresenceRays <= sampleCount * VLEN);
        CPP_addPresenceShadowsQueueEntries(pbrTls, spawnedPresenceRays, numSpawnedPresenceRays,
                                     spawnedPresenceRayIndices);
    }

    Arena_setPtr(arena, memoryBookmark2);

    if (numSpawnedRays) {
        MNRY_ASSERT(numSpawnedRays <= sampleCount * VLEN);
        CPP_addIncoherentRayQueueEntries(pbrTls, spawnedRays, numSpawnedRays,
                                         spawnedRayIndices);
    }

    Arena_setPtr(arena, memoryBookmark1);
}

void
computeRadianceBsdfMultiSampler(const uniform PathIntegrator * uniform this,
                                uniform PbrTLState * uniform pbrTls,
                                uniform ShadingTLState * uniform shadingTls,
                                const varying RayState * uniform rs,
                                const varying Intersection &isect,
                                const varying Bsdf &bsdf,
                                const varying BsdfSlice &slice,
                                varying bool doIndirect,
                                varying Flags indirectFlags,
                                const uniform LightSet &activeLightSet,
                                const varying Vec3f &cullingNormal,
                                varying float rayEpsilon,
                                varying float shadowRayEpsilon,
                                const varying Color &ssAov,
                                varying uint32_t &sequenceID)
{
    // We use a BsdfSampler object to keep track of sampling strategies and
    // sample budget per lobe.
    // We only want to split on the first scattering event seen from the
    // camera (either directly, either through one or many mirror bounces)
    uniform Arena * uniform arena = pbrTls->mArena;
    uniform uint8_t *uniform memoryBookmark = Arena_getPtr(arena);

    const varying PathVertex &pv = rs->mPathVertex;
    const varying int maxSamplesPerLobe = (pv.nonMirrorDepth == 0 ? this->mBsdfSamples :
                                           min(this->mBsdfSamples, 1));

    varying BsdfSampler bSampler;
    BsdfSampler_init(&bSampler, arena, bsdf, slice, maxSamplesPerLobe, doIndirect);

    const uniform int bsdfSampleCount = BsdfSampler_getSampleCount(&bSampler);

    varying BsdfSample *uniform bsmp = (varying BsdfSample *uniform)
        Arena_allocArray(arena, bsdfSampleCount, sizeof(varying BsdfSample));
    memset(bsmp, 0, sizeof(varying BsdfSample) * bsdfSampleCount);

    // Store max possible bsdf sample count, we'll refine this later by taking
    // the active lanes into account.
    addToCounter(pbrTls->mStatistics, STATS_BSDF_SAMPLE_LANE_MAX, bsdfSampleCount * VLEN);

    // We use a LightSetSampler object to keep track of sampling strategies and
    // sample budget per light.
    // We use the same splitting strategy as for lobes above.
    const varying int maxSamplesPerLight = (pv.nonMirrorDepth == 0 ?
                                            this->mLightSamples :
                                            min(this->mLightSamples, 1));

    varying LightSetSampler lSampler;
    LightSetSampler_init( &lSampler, arena, &activeLightSet, &bsdf, getP(isect), maxSamplesPerLight);

    const uniform int lightSampleCount = reduce_max(LightSetSampler_getLightSampleCount(&lSampler));
    const uniform int lightSetSampleCount = LightSetSampler_getSampleCount(&lSampler);

    varying LightSample *uniform lsmp = (varying LightSample *uniform)
        Arena_allocArray(arena, lightSampleCount, sizeof(varying LightSample));
    memset(lsmp, 0, sizeof(varying LightSample) * lightSampleCount);

    // Store max possible lightset sample count, we'll refine this later by taking
    // the active lanes into account.
    addToCounter(pbrTls->mStatistics, STATS_LIGHT_SAMPLE_LANE_MAX,
                 lightSampleCount * VLEN);

    // Draw Bsdf and LightSet samples and compute tentative contributions
    const varying Subpixel &sp = rs->mSubpixel;
    const varying RayDifferential &ray = rs->mRay;
    drawBsdfSamples(pbrTls, shadingTls, bSampler, lSampler, sp, pv, getP(isect), cullingNormal,
                    Ray_getTime(ray), sequenceID, bsmp, this->mSampleClampingDepth, 
                    sp.mSampleClampingValue, indirectFlags, Ray_getDirFootprint(ray));

    // now that we have drawn our bsdf samples, we can pack our material aovs
    const uniform FrameState &fs = *pbrTls->mFs;
    const uniform AovSchema &aovSchema = *((const uniform AovSchema * uniform) fs.mAovSchema);
    const uniform bool aovs = !AovSchema_empty(aovSchema);
    if (aovs) {
        const uniform LightAovs &lightAovs = *((const uniform LightAovs * uniform) fs.mLightAovs);
        const uniform MaterialAovs &materialAovs = *((const uniform MaterialAovs * uniform) fs.mMaterialAovs);
        aovAccumMaterialAovs(pbrTls, aovSchema, lightAovs, materialAovs,
                             isect, ray, fs.mScene, bsdf, ssAov,
                             &bSampler, bsmp, pv.aovPathPixelWeight, sp.mPixel,
                             rs->mDeepDataHandle, pv.lpeStateId);
    }

    snapshotLaneUtilization(pbrTls->mStatistics, STATS_VEC_COUNTER_B);

    //---------------------------------------------------------------------
    // Apply Russian Roulette (RR). Note we only do RR past a non-mirror
    // bounce, to avoid breaking the nice stratification of samples on the
    // first non-mirror hit.
    // TODO: We should find a way to use efficiency optimized RR. For this we
    // need an estimate of the final pixel color so our threshold can be
    // computed accordingly.
    // TODO: Compute RR culling rate
    if (pv.nonMirrorDepth > 0  &&  this->mRussianRouletteThreshold > 0.0f) {
        applyRussianRoulette(pbrTls, bSampler, bsmp, sp, pv, sequenceID,
                             this->mRussianRouletteThreshold, this->mInvRussianRouletteThreshold);
    }

    //---------------------------------------------------------------------
    // Let's trace some rays to resolve visibility / shadowing and sum up
    // contributions. We trace one ray per valid sample and affect all
    // contributions for that sample accordingly.

    const uniform int numOcclRaysToAlloc = bsdfSampleCount + lightSetSampleCount;
    varying BundledOcclRay *uniform occlRays = (varying BundledOcclRay *uniform)
        Arena_allocArray(arena, numOcclRaysToAlloc, sizeof(varying BundledOcclRay));
    uniform uint32_t *uniform occlRayIndices = (uniform uint32_t *uniform)
        Arena_allocArray(arena, numOcclRaysToAlloc * VLEN + 1, sizeof(uniform uint32_t));
    uniform uint32_t numOcclRaysFilled = 0;

    varying BundledOcclRay *uniform presenceShadowsRays = (varying BundledOcclRay *uniform)
        Arena_allocArray(arena, numOcclRaysToAlloc, sizeof(varying BundledOcclRay));
    uniform uint32_t *uniform presenceShadowsRayIndices = (uniform uint32_t *uniform)
        Arena_allocArray(arena, numOcclRaysToAlloc * VLEN + 1, sizeof(uniform uint32_t));
    uniform uint32_t numPresenceShadowsRaysFilled = 0;

    addDirectVisibleLightSampleContributionsBundled(this, pbrTls, shadingTls, lSampler, bSampler, lsmp, ray,
                                                    &cullingNormal, rayEpsilon, shadowRayEpsilon, sp, pv, *rs,
                                                    programIndex,
                                                    occlRays,
                                                    occlRayIndices,
                                                    numOcclRaysFilled,
                                                    presenceShadowsRays,
                                                    presenceShadowsRayIndices,
                                                    numPresenceShadowsRaysFilled,
                                                    isect, sequenceID);
    if (doIndirect) {
        addIndirectOrDirectVisibleContributionsBundled(this, pbrTls, sp, pv,
                                                       bSampler, bsmp, ray, rayEpsilon, shadowRayEpsilon, isect,
                                                       (varying BsdfLobeType) indirectFlags.mBits, sequenceID, *rs);
    } else {
        // TODO: Incorrect transparency if there is no indirect
        addDirectVisibleBsdfSampleContributionsBundled(this, pbrTls, bSampler, false, bsmp, ray,
                                                       rayEpsilon, shadowRayEpsilon, *rs,
                                                       lightSetSampleCount * VLEN + programIndex,
                                                       occlRays + lightSetSampleCount,
                                                       occlRayIndices,
                                                       numOcclRaysFilled,
                                                       presenceShadowsRays + lightSetSampleCount,
                                                       presenceShadowsRayIndices,
                                                       numPresenceShadowsRaysFilled,
                                                       isect);
    }

    MNRY_ASSERT(numOcclRaysFilled <= numOcclRaysToAlloc * VLEN);

    if (numOcclRaysFilled) {
        CPP_addOcclusionQueueEntries(pbrTls, occlRays, numOcclRaysFilled,
                                     occlRayIndices);
    }

    if (numPresenceShadowsRaysFilled) {
        CPP_addPresenceShadowsQueueEntries(pbrTls, presenceShadowsRays, numPresenceShadowsRaysFilled,
                                           presenceShadowsRayIndices);
    }

    Arena_setPtr(arena, memoryBookmark);
}


