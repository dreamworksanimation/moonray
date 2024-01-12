// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0


//
// Current issues which need attention
// -----------------------------------
// - SSS is still executed in a depth first fashion, it hasn't been converted
//   over to breadth first yet.
// - Sample ids passed around have different absolute values than in the
//   non-bundled case, and therefore result in a different noise pattern.
//   The question is if this bundled noise pattern is inferior, and if so,
//   what's the best solution to fixing it.
//
//
// Transparency
// ------------
// In non-bundled mode, we accumulate transparency as we recurse, and finally
// subtract it from the full alpha value when recursions ends, so the final frame
// buffer alpha looks like:
//
//     fb_alpha = full_alpha - (t0 + t1 + t2 + ... + tn)       // eqn1
//
// The bundled mode writes the full alpha into the frame buffer as it processes
// each primary ray, and from that point on, each new transparency value is simply
// subtracted from the alpha contained in the frame buffer. So, in effect, eqn1
// turns into the following:
//
//     fb_alpha = full_alpha - t0 - t1 - t2 - ... - tn
//
// Note that we fill the queued BundledRadiance structures with negative values
// in the alpha channel (or zero) for all non-primary ray entries to account for
// the fact we are subtracting transparency from alpha.
//
//
// Frame buffer path weight
// ------------------------
// - Our frame buffer contains 5 channels: red, green, blue, alpha and pixel weight.
// - rgba is divided through by the weight to retrive the final radiance value.
//   This is called normalization.
// - The russian roulette path throughput value is used to drive the initial
//   path throughput and path weight. Since these are equal, we still get white
//   after normalization.
// - For bundled mode, the full weight is added to the frame buffer when we write
//   out the radiance for the primary rays. All further spawned rays then write
//   out zero weight since the path weight has already been fully accounted for.
// - This scheme simplifies the accounting when accumulating path weights but
//   means that the radiance and path weights aren't always synchronized with
//   each other at any given point in time. This is visible as bright or dark
//   patches when the frame is normalized during snapshots.
//

#include "PathIntegrator.isph"

#include <moonray/rendering/pbr/core/Aov.isph>
#include <moonray/rendering/pbr/core/RayState.isph>
#include <moonray/rendering/pbr/light/LightSet.isph>
#include <moonray/rendering/bvh/shading/ispc/Intersection.isph>
#include <moonray/rendering/shading/ispc/bsdf/Bsdf.isph>
#include <scene_rdl2/render/util/Arena.isph>

extern "C" uniform bool
CPP_isIspcAccumulatorRunning( uniform PbrTLState *uniform pbrTls );

extern "C" void
CPP_computeRadianceSubsurface(  const uniform PathIntegrator * uniform pathIntegrator,
                                      uniform PbrTLState *     uniform pbrTls,
                                const uniform uint32_t *       uniform rayStateIndices,
                                const varying Bssrdf *         uniform bssrdf,
                                const varying VolumeSubsurface* uniform volumeSubsurface,
                                const uniform LightSet *       uniform lightSet,
                                const uniform int                      materialLabelIds,
                                const uniform int                      lpeMaterialLabelIds,
                                const uniform int *            uniform geomLabelIds,
                                const uniform int *            uniform doIndirect,
                                const uniform float *          uniform rayEpsilon,
                                const uniform float *          uniform shadowRayEpsilon,
                                      uniform uint32_t *       uniform sequenceID,
                                      uniform float *          uniform radiance,
                                      uniform float *          uniform ssAov,
                                      uniform int32_t                  laneMask );

extern "C" void
CPP_computeRadianceEmissiveRegionsBundled(const uniform PathIntegrator * uniform pathIntegrator,
    uniform PbrTLState * uniform pbrTls, const uniform uint32_t * uniform rayStateIndices,
    const uniform float * uniform rayEpsilons, const varying Bsdf * uniform bsdf,
    const varying BsdfSlice * uniform slice, uniform float * uniform results,
    uniform int32_t laneMask);

extern "C" void
CPP_applyVolumeTransmittance(const uniform PathIntegrator * uniform pathIntegrator,
    uniform PbrTLState * uniform pbrTls, const uniform uint32_t * uniform rayStateIndices,
    uniform int32_t laneMask);


extern "C" void
CPP_addRadianceQueueEntries(            uniform PbrTLState *     uniform pbrTls,
                                  const varying BundledRadiance *uniform radiances,
                                        uniform uint32_t                 numRadiances,
                                  const uniform uint32_t *       uniform indicies );


//-----------------------------------------------------------------------------

// Returns the number of BundledRadiance entries filled and added to the index list.
inline uniform uint32_t
fillBundledRadiance(       uniform PbrTLState *     uniform pbrTls,
                           varying BundledRadiance *uniform dst,
                           uniform uint32_t *       uniform dstIndices,
                           varying uint32_t                 indices,
                           varying int                      depth,
                     const varying Color &                  radiance,
                           varying float                    pathPixelWeight,
                           varying float                    transparency,
                     const varying RayState &               rayState )
{
    uniform uint32_t numNewEntries = 0;

    // For each primary ray which takes this code path, we want to initialize
    // the frame buffer alpha to the pathPixelWeight. Rays at higher depths only
    // subtract transparency from this value.
    const varying float alpha = (depth ? 0.f : pathPixelWeight) - transparency * pathPixelWeight;

    if (!isBlack(radiance) || !isEqual(pathPixelWeight, 0.f) || !isEqual(alpha, 0.f)) {

        // Negate transparency here since frame buffer is initialized to full opacity
        // for each primary ray.
        dst->mRadiance    = Vec4f_ctor(radiance.r, radiance.g, radiance.b, alpha);
        dst->mPathPixelWeight  = pathPixelWeight;
        dst->mPixel       = rayState.mSubpixel.mPixel;
        dst->mSubPixelIndex = rayState.mSubpixel.mSubpixelIndex;
        dst->mDeepDataHandle = rayState.mDeepDataHandle;
        dst->mCryptomatteDataHandle = rayState.mCryptomatteDataHandle;
        dst->mCryptoRefP = rayState.mCryptoRefP;
        dst->mCryptoRefN = rayState.mCryptoRefN;
        dst->mCryptoUV = rayState.mCryptoUV;
        PbrTLState_acquireDeepData(pbrTls, dst->mDeepDataHandle);
        PbrTLState_acquireCryptomatteData(pbrTls, dst->mCryptomatteDataHandle);
        dst->mTilePass = rayState.mTilePass;

#ifdef PBR_CHECK_FOR_NAN
        checkForNan(radiance, "BundledRadiance", rayState.mSubpixel,
                    rayState.mPathVertex, rayState.mRay, isect);
#endif
        // packed_store_active2() "may write one extra element to the output array"
        // according to ISPC documentation https://ispc.github.io/ispc.html.
        // Please make sure that 1 extra element was allocated for the output array.
        numNewEntries = packed_store_active2(dstIndices, indices);

        snapshotLaneUtilization(pbrTls->mStatistics, STATS_VEC_FILL_BUNDLED_RADIANCE);
    }

    return numNewEntries;
}

//-----------------------------------------------------------------------------

void
integrateBundled( const uniform PathIntegrator * uniform this,
                        uniform PbrTLState *     uniform pbrTls,
                        uniform ShadingTLState * uniform shadingTls,
                        uniform uint32_t                 numEntries,
                        varying RayState *       uniform rayStates,
                  const varying Intersection *   uniform isects,
                  const varying Bsdf *           uniform bsdfs,
                  const uniform LightSet *       uniform lightSet,
                  const varying float *          uniform presences)
{
    uniform Arena * uniform arena = pbrTls->mArena;
    uniform uint8_t *uniform memoryBookmark1 = Arena_getPtr(arena);

    const uniform FrameState &fs = *pbrTls->mFs;

    // Aovs
    const uniform AovSchema &aovSchema = *((const uniform AovSchema * uniform) fs.mAovSchema);
    const uniform LightAovs &lightAovs = *((const uniform LightAovs * uniform) fs.mLightAovs);
    const uniform MaterialAovs &materialAovs = *((const uniform MaterialAovs * uniform) fs.mMaterialAovs);
    const uniform bool aovs = !AovSchema_empty(aovSchema);

    // Each iteration of the main loop can cause another varying BundledRadiance
    // structure to be filled.
    uniform uint32_t numBlocks = (numEntries + (programCount - 1)) / programCount;

    varying BundledRadiance *uniform radiances = (varying BundledRadiance *uniform)
        Arena_allocArray(arena, numBlocks, sizeof(varying BundledRadiance));
    uniform uint32_t *uniform radianceIndices = (uniform uint32_t *uniform)
        Arena_allocArray(arena, numBlocks * VLEN + 1, sizeof(uniform uint32_t));
    uniform uint32_t numRadiances = 0;

    for (uniform uint32_t i = 0; i < numBlocks; ++i) {

        CHECK_CANCELLATION(pbrTls, return);

        varying RayState * uniform rs = &rayStates[i];
        const varying RayDifferential &ray = rs->mRay;
        const varying Intersection &isect = isects[i];
        const varying Bsdf * uniform bsdf = &bsdfs[i];

        const varying Subpixel &sp = rs->mSubpixel;
        const varying PathVertex &pv = rs->mPathVertex;
        varying uint32_t sequenceID = rs->mSequenceID;

        varying Color radiance = sBlack;
        varying float transparency = 0.f;
        const int rayDepth = ray.ext.depth;

        const varying uint32_t rayIndices = i * VLEN + programIndex;

        varying Color ssAov = sBlack;

        // Account for case where numEntries isn't a multiple of VLEN.
        if (rayIndices >= numEntries) {
            continue;
        }

        // Volumes
        if (rs->mVolHit) {
            radiance = radiance + rs->mVolRad;
            rs->mPathVertex.pathThroughput = rs->mPathVertex.pathThroughput * rs->mVolTr * rs->mVolTh;
            // and also on the SOA side for volumes and hybrid subsurface
            CPP_applyVolumeTransmittance(this, pbrTls, (const uniform uint32_t * uniform) &rs->mRayStateIdx,
                lanemask());
        }

        //---------------------------------------------------------------------
        // Termination (did the shader request termination of tracing?)

        if ((1 << programIndex) & bsdf->mEarlyTerminationMask) {
            if (rayDepth == 0) {
                transparency = reduceTransparency(rs->mVolTalpha);
            } else {
                transparency = reduceTransparency(pv.pathThroughput);
            }
            numRadiances += fillBundledRadiance(pbrTls,
                                                radiances + i,
                                                &radianceIndices[numRadiances],
                                                rayIndices,
                                                rayDepth,
                                                radiance,
                                                pv.pathPixelWeight,
                                                transparency,
                                                *rs);


            if (aovs) {
                aovAccumMaterialAovs(pbrTls, aovSchema, lightAovs, materialAovs, isect,
                                     ray, fs.mScene, *bsdf, ssAov,
                                     NULL, NULL, pv.aovPathPixelWeight, sp.mPixel,
                                     rs->mDeepDataHandle, pv.lpeStateId);
            }

            if (rs->mDeepDataHandle != PBR_nullHandle) {
                // If we have terminated, don't output anything to the deep buffer
                uniform DeepData * varying deepData =
                    (uniform DeepData * varying)PbrTLState_getListItem(
                    pbrTls, rs->mDeepDataHandle, 0);
                deepData->mHitDeep = 0;
            }

            if (rs->mCryptomatteDataHandle != PBR_nullHandle) {
                // If we have terminated, don't output anything to the cryptomatte buffer
                uniform CryptomatteData * varying cryptomatteData =
                    (uniform CryptomatteData * varying)PbrTLState_getListItem(
                    pbrTls, rs->mCryptomatteDataHandle, 0);
                cryptomatteData->mHit = 0;
            }

            continue;
        }

        // Only the presence continuation ray contributes to the radiance so we can early out here.
        // We must process cutouts (early termination) before this or the cutout alpha will be incorrect.
        if (isEqual(presences[i], 0.f)) {
            continue;
        }

        //---------------------------------------------------------------------
        // Self-emission
        Color selfEmission = pv.pathThroughput * Bsdf_getSelfEmission(bsdf);
        radiance = radiance + selfEmission;

        if (aovs) {
            if (!isBlack(selfEmission)) {
                // transition
                int lpeStateId = pv.lpeStateId;
                lpeStateId = LightAovs_emissionEventTransition(&lightAovs, lpeStateId, *bsdf);

                // accumulate matching aovs. No need to worry about pre-occlusion here because we are accumulating an
                // emission event, and emission is not supported by pre-occlusion aovs
                aovAccumLightAovs(pbrTls, aovSchema, lightAovs, selfEmission, lpeStateId,
                                  sp.mPixel, rs->mDeepDataHandle);
            }
        }
        //---------------------------------------------------------------------
        // Early out if we don't have any Bsdf lobes nor Bssrdf
        if (Bsdf_getLobeCount(bsdf) == 0  &&  Bsdf_hasSubsurface(bsdf) == false) {

            numRadiances += fillBundledRadiance(pbrTls,
                                                radiances + i,
                                                &radianceIndices[numRadiances],
                                                rayIndices,
                                                rayDepth,
                                                radiance,
                                                pv.pathPixelWeight,
                                                transparency,
                                                *rs);


            if (aovs) {
                aovAccumMaterialAovs(pbrTls, aovSchema, lightAovs, materialAovs,
                                     isect, ray, fs.mScene, *bsdf, ssAov,
                                     NULL, NULL, pv.aovPathPixelWeight, sp.mPixel,
                                     rs->mDeepDataHandle, pv.lpeStateId);
            }

            continue;
        }

        //---------------------------------------------------------------------
        // Have we reached the maximum number of bounces for each lobe types / overall ?
        // Note: hair lobes are also glossy lobes. So the max depth for hair lobes
        // would be max(mMaxGlossyDepth, mMaxHairDepth)
        varying Flags indirectFlags;
        Flags_setBits(&indirectFlags, BSDF_LOBE_TYPE_NONE);
        varying bool doIndirect = (this->mBsdfSamples > 0  &&  rayDepth < this->mMaxDepth);
        if (doIndirect) {
            Flags_set(&indirectFlags, (pv.diffuseDepth < this->mMaxDiffuseDepth  ?
                    BSDF_LOBE_TYPE_DIFFUSE :  BSDF_LOBE_TYPE_NONE));
            Flags_set(&indirectFlags, (pv.glossyDepth < this->mMaxGlossyDepth    ?
                    BSDF_LOBE_TYPE_GLOSSY   :  BSDF_LOBE_TYPE_NONE));
            Flags_set(&indirectFlags, (pv.mirrorDepth < this->mMaxMirrorDepth    ?
                    BSDF_LOBE_TYPE_MIRROR   :  BSDF_LOBE_TYPE_NONE));
            doIndirect = (Flags_getAll(indirectFlags) != BSDF_LOBE_TYPE_NONE) ||
                         (pv.hairDepth < this->mMaxHairDepth);
            // If doIndirect is true due to hairDepth only, then only side type bits
            // are set in indirectFlags.
            Flags_set(&indirectFlags, (doIndirect  ?
                    BSDF_LOBE_TYPE_ALL_SURFACE_SIDES :  BSDF_LOBE_TYPE_NONE));
        }

        //---------------------------------------------------------------------
        // For bssrdf or bsdfs which contain both reflection and transmission lobes
        // or is spherical, a single normal can't be used for culling so skip normal
        // culling.
        varying Vec3f cullingNormal;
        setInvalidCullingNormal(&cullingNormal);

        if (!Bsdf_hasSubsurface(bsdf)  &&  !Bsdf_getIsSpherical(bsdf) &&
            ((Bsdf_getType(bsdf) & BSDF_LOBE_TYPE_ALL_SURFACE_SIDES) != BSDF_LOBE_TYPE_ALL_SURFACE_SIDES)) {
            cullingNormal = getNg(isect);
            if (!(Bsdf_getType(bsdf) & BSDF_LOBE_TYPE_REFLECTION)) {
                cullingNormal = neg(cullingNormal);
            }
        }

        // Setup a slice, which handles selecting the lobe types and setup
        // evaluations to include the cosine term.
        // Note: Even though we may not be doing indirect for certain lobe types
        // (according to indirectFlags), we still want to draw samples according
        // to all lobes for direct lighting MIS.
        varying BsdfSlice slice;
        BsdfSlice_init(&slice, getNg(isect), ray.dir * -1.f, BSDF_LOBE_TYPE_ALL,
                       true, isEntering(isect), fs.mShadowTerminatorFix);

        // Compute automatic ray-tracing bias
        varying float pathDistance = pv.pathDistance + Ray_getEnd(ray);
        varying float rayEpsilon = getEpsilonHint(isect);
        if (rayEpsilon <= 0.0) {
            rayEpsilon = sHitEpsilonStart * max(pathDistance, 1.0f);
        }
        varying float shadowRayEpsilon = getShadowEpsilonHint(isect);

        // Gather up all lights that can affect the intersection point/normal.
        uniform LightSet activeLightSet;
        uniform bool hasRayTerminatorLights;
        computeActiveLights(arena, lightSet, getP(isect), cullingNormal, bsdf, ray.time,
                            &activeLightSet, hasRayTerminatorLights);

        //---------------------------------------------------------------------
        // Estimate sub-surface scattering
        // For now we get the Bssrdf from the Bsdf.
        // TODO: We'd want this to be from the shader so we can query a shader
        // seprately for Bsdf and for Bssrdf.

        // TODO: this is recursing and casting rays, change over to be
        //       breadth first and queued!
        const varying Bssrdf * uniform bssrdf = Bsdf_getBssrdf(bsdf);
        const varying VolumeSubsurface * uniform volumeSubsurface =
                Bsdf_getVolumeSubsurface(bsdf);
        if (isActive(bssrdf) || isActive(volumeSubsurface)) {

            // We assume that both are not active at the same time,
            // otherwise our depth counting will be messed up.
            MNRY_ASSERT(!(isActive(bssrdf) && isActive(volumeSubsurface)));

            // Don't pass bools through the ISPC/C++ interface, convert to int instead.
            varying int intDoIndirect = (int)doIndirect;

            // increment subsurface depth
            // Hybrid mode is tricky.  Incrementing the subsurface depth in rs->mPathVertex has
            // no impact on CPP_computeRadianceSubsurface.  That function is only
            // passed a set of ray state idx (pointers to RayStates).  Those were the
            // RayStates used to create the SOA version of rayState we (ispc) are working
            // with currently (See convertAOSRayStatesToSOA() in ShadeBundleHandler.cc).
            // So CPP_computeRadianceSubsurface will need to increment the subsurfaceDepth
            // as well when it retrieves the original path vertex from the RayState pool.
            // But we still need to increment our subsurfaceDepth to properly handle
            // secondary bounces from lobes handled in ispc.
            rs->mPathVertex.subsurfaceDepth += 1;

            CPP_computeRadianceSubsurface(this,
                                          pbrTls,
                                          (const uniform uint32_t *uniform)&rs->mRayStateIdx,
                                          bssrdf,
                                          volumeSubsurface,
                                          &activeLightSet,
                                          bsdf->mMaterialLabelId,
                                          bsdf->mLpeMaterialLabelId,
                                          (const uniform int *uniform)&bsdf->mGeomLabelId,
                                          (const uniform int *uniform)&intDoIndirect,
                                          (const uniform float *uniform)&rayEpsilon,
                                          (const uniform float *uniform)&shadowRayEpsilon,
                                          (uniform uint32_t *uniform)&sequenceID,
                                          (uniform float *uniform)&radiance,
                                          (uniform float *uniform)&ssAov,
                                          lanemask());
        }

#ifdef PBR_CHECK_FOR_NAN
        checkForNan(radiance, "Subsurface scattering", sp, pv, ray, isect);
#endif

        //---------------------------------------------------------------------
        // Estimate emissive volume region energy contribution
        if (Bsdf_getLobeCount(bsdf) > 0) {
            varying Color volumeEmission = sBlack;
            CPP_computeRadianceEmissiveRegionsBundled(this, pbrTls,
                (const uniform uint32_t * uniform) &rs->mRayStateIdx,
                (const uniform float *uniform) &rayEpsilon, bsdf, &slice,
                (uniform float * uniform) &volumeEmission, lanemask());
            radiance = radiance + volumeEmission;
        }

        float minTransparency = reduceTransparency(rs->mVolTm);
        transparency = transparency + (1 - transparency) * minTransparency;

        // Add radiance from emission and subsurface scattering. Since this queue
        // entry takes care of adding the current radiance and transparency
        // computed up until this point, we can ignore it for the rest of the
        // function.
        numRadiances += fillBundledRadiance(pbrTls,
                                            radiances + i,
                                            &radianceIndices[numRadiances],
                                            rayIndices,
                                            rayDepth,
                                            radiance,
                                            pv.pathPixelWeight,
                                            transparency,
                                            *rs);

        //---------------------------------------------------------------------
        // Early out if we don't have any Bsdf lobes
        if (Bsdf_getLobeCount(bsdf) == 0) {

            if (aovs) {
                aovAccumMaterialAovs(pbrTls, aovSchema, lightAovs, materialAovs,
                                     isect, ray, fs.mScene, *bsdf, ssAov,
                                     NULL, NULL, pv.aovPathPixelWeight, sp.mPixel,
                                     rs->mDeepDataHandle, pv.lpeStateId);
            }

            continue;
        }

        snapshotLaneUtilization(pbrTls->mStatistics, STATS_VEC_COUNTER_A);

        //---------------------------------------------------------------------
        // Setup bsdf and light samples
        computeRadianceBsdfMultiSampler(this,
                                        pbrTls,
                                        shadingTls,
                                        rs,
                                        isect,
                                        *bsdf,
                                        slice,
                                        doIndirect,
                                        indirectFlags,
                                        activeLightSet,
                                        cullingNormal,
                                        rayEpsilon,
                                        shadowRayEpsilon,
                                        ssAov,
                                        sequenceID);
    }

    MNRY_ASSERT(numRadiances <= numEntries);

    CHECK_CANCELLATION(pbrTls, return);

    if (numRadiances) {
        CPP_addRadianceQueueEntries(pbrTls, radiances, numRadiances,
                                    radianceIndices);
    }

    Arena_setPtr(arena, memoryBookmark1);
}

export uniform uint32_t
PathIntegrator_hudValidation(uniform bool verbose)
{
    PATH_INTEGRATOR_VALIDATION;
}

//-----------------------------------------------------------------------------

// Two copies of exported functions are generated, one with the trailing mask
// which can be called internally, and another without which can be called
// externally. Create a small version of the function which calls out to the real
// one.
export void
PathIntegrator_integrateBundled( const uniform PathIntegrator *uniform this,
                                       uniform PbrTLState *    uniform pbrTls,
                                       uniform ShadingTLState *uniform shadingTls,
                                       uniform uint32_t                numEntries,
#pragma ignore warning(all)
                                       varying RayState *      uniform rayStates,
#pragma ignore warning(all)
                                 const varying Intersection *  uniform isects,
#pragma ignore warning(all)
                                 const varying Bsdf *          uniform bsdfs,
                                 const uniform LightSet *      uniform lightList,
#pragma ignore warning(all)
                                 const varying float *         uniform presences)

{
    MNRY_ASSERT(CPP_isIntegratorAccumulatorRunning(pbrTls));
    MNRY_ASSERT(CPP_isIspcAccumulatorRunning(pbrTls));

    integrateBundled(this, pbrTls, shadingTls, numEntries, rayStates, isects, bsdfs, lightList, presences);
}

