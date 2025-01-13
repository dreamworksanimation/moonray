// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

#include "ShadeBundleHandler.h"

#include <moonray/rendering/pbr/core/Aov.h>
#include <moonray/rendering/pbr/core/Constants.h>

#include <moonray/rendering/bvh/shading/AttributeKey.h>
#include <moonray/rendering/bvh/shading/Intersection.h>
#include <moonray/rendering/bvh/shading/ShadingTLState.h>
#include <moonray/rendering/geom/IntersectionInit.h>
#include <moonray/rendering/mcrt_common/Clock.h>
#include <moonray/rendering/mcrt_common/SOAUtil.h>
#include <moonray/rendering/mcrt_common/ThreadLocalState.h>
#include <moonray/rendering/pbr/core/PbrTLState.h>
#include <moonray/rendering/pbr/core/RayState.h>
#include <moonray/rendering/pbr/core/Scene.h>
#include <moonray/rendering/pbr/core/Cryptomatte.h>
#include <moonray/rendering/pbr/integrator/PathIntegrator.h>
#include <moonray/rendering/pbr/integrator/PathIntegratorUtil.h>
#include <moonray/rendering/shading/bsdf/Bsdf.h>
#include <moonray/rendering/shading/EvalShader.h>
#include <moonray/rendering/shading/ispc/Shadingv.h>
#include <moonray/rendering/shading/Material.h>
#include <moonray/rendering/shading/Types.h>

#include <scene_rdl2/common/math/Color.h>
#include <scene_rdl2/common/math/Mat4.h>

namespace moonray {

namespace {

struct ALIGN(16) SortedEntry
{
    uint32_t    mSortKey;   // Material bundled id is stored in here.
    uint32_t    mRsIdx;     // Global ray state index.
    uint32_t    mIsectIdx;
    int         mLayerAssignmentId;
};

MNRY_STATIC_ASSERT(sizeof(SortedEntry) == 16);

inline void
convertAOSRayStatesToSOA(pbr::TLState *tls,
                         unsigned numEntries,
                         const pbr::RayState *rayStates,
                         pbr::RayStatev *rayStatesSOA,
                         const uint32_t *sortIndices)   // Stride of sizeof(SortedEntry) between indices!
{
    ACCUMULATOR_PROFILE(tls, ACCUM_AOS_TO_SOA_RAYSTATES);

#if (VLEN == 16u)
    mcrt_common::convertAOSToSOAIndexed_AVX512
#elif (VLEN == 8u)
    mcrt_common::convertAOSToSOAIndexed_AVX
#elif (VLEN == 4u) && defined(__ARM_NEON__)
    mcrt_common::convertAOSToSOAIndexed_NEON
#else
    #error Requires at least AVX to build.
#endif
    <sizeof(pbr::RayState), sizeof(pbr::RayState), sizeof(pbr::RayStatev), sizeof(SortedEntry), 0>
        (numEntries, (const uint32_t *)rayStates, (uint32_t *)rayStatesSOA, sortIndices);
}

inline void
convertAOSIntersectionsToSOA(pbr::TLState *tls,
                             unsigned numEntries,
                             const shading::Intersection *isects,
                             shading::Intersectionv *isectsSOA,
                             const uint32_t *sortIndices)   // Stride of sizeof(SortedEntry) between indices!
{
    ACCUMULATOR_PROFILE(tls, ACCUM_AOS_TO_SOA_INTERSECTIONS);

#if (VLEN == 16u)
    mcrt_common::convertAOSToSOAIndexed_AVX512
#elif (VLEN == 8u)
    mcrt_common::convertAOSToSOAIndexed_AVX
#elif (VLEN == 4u) && defined(__ARM_NEON__)
    mcrt_common::convertAOSToSOAIndexed_NEON
#else
    #error Requires at least AVX to build.
#endif
    <sizeof(shading::Intersection), sizeof(shading::Intersection),
     sizeof(shading::Intersectionv), sizeof(SortedEntry), 0>
        (numEntries, (const uint32_t *)isects, (uint32_t *)isectsSOA, sortIndices);
}

} // End of anonymous namespace.

namespace pbr {

void shadeBundleHandler(mcrt_common::ThreadLocalState *tls, unsigned numEntries,
                        shading::SortedRayState *entries, void *userData)
{
    pbr::TLState *pbrTls = tls->mPbrTls.get();

    EXCL_ACCUMULATOR_PROFILE(pbrTls, EXCL_ACCUM_SHADE_HANDLER);

    MNRY_ASSERT(numEntries && entries && userData);

    CHECK_CANCELLATION(pbrTls, return);

    shading::Material *ext = MNRY_VERIFY((shading::Material *)userData);

    // We should hit this case rarely, mainly in the case where russian roulette
    // isn't culling the number of rays we'd like it to.
    if (tls->checkForHandlerStackOverflowRisk()) {
        ext->deferEntriesForLaterProcessing(tls, numEntries, entries);
        return;
    }

    scene_rdl2::alloc::Arena *arena = pbrTls->mArena;
    SCOPED_MEM(arena);

    // Retrieve any shading entries which were previously deferred and process
    // them now (since we're no longer at risk for a handler stack overflow).
    // We potentially update the numEntries and entries variables with this call.
    ext->retrieveDeferredEntries(tls, arena, numEntries, entries);

    // Convert array of SortedRayState objects to RayState pointers in-place.
    RayState **rayStates = decodeRayStatesInPlace(numEntries, entries);
    MNRY_ASSERT(isRayStateListValid(pbrTls, numEntries, rayStates));

    const FrameState &fs = *pbrTls->mFs;

    //
    // Compute final sort order for shading points.
    //
    const scene_rdl2::rdl2::Layer *layer = fs.mLayer;
    const Scene *scene = fs.mScene;
    const scene_rdl2::rdl2::Material &material = *ext->getOwner().asA<scene_rdl2::rdl2::Material>();
    RayState *baseRayState = indexToRayState(0);
    shading::TLState *shadingTls = MNRY_VERIFY(tls->mShadingTls.get());

    // Allocate temp working memory buffers (automatically freed when we exit this function).
    shading::Intersection *isectMemory = arena->allocArray<shading::Intersection>(numEntries, CACHE_LINE_SIZE);
    SortedEntry *sortedEntries = arena->allocArray<SortedEntry>(numEntries);
    BundledRadiance *radiances = arena->allocArray<BundledRadiance>(numEntries, CACHE_LINE_SIZE);
    BundledRadiance *currRadiance = radiances;

    // Allocate memory to gather raystates so we can bulk free them later in the function.
    unsigned numRayStatesToFree = 0;
    RayState **rayStatesToFree = arena->allocArray<RayState *>(numEntries);

    // Aovs.
    float *aovs = nullptr;
    unsigned aovNumChannels = 0;
    if (!fs.mAovSchema->empty()) {
        // scratch space storage for per-pixel aov packing
        aovNumChannels = fs.mAovSchema->numChannels();
        aovs = arena->allocArray<float>(aovNumChannels);
        fs.mAovSchema->initFloatArray(aovs);
    }

    // If we have a ray that just came from a presence object, we need to handle the case where the ray now hits
    // a non presence object for the cryptomatte. If we need to handle that case, we set this variable:
    bool handlePresenceContinue = false;

    for (unsigned i = 0; i < numEntries; ++i) {

        CHECK_CANCELLATION_IN_LOOP(pbrTls, i, return);

        RayState *rs = rayStates[i];
        mcrt_common::RayDifferential *ray = &rs->mRay;
        PathVertex &pv = rs->mPathVertex;

        // Disable ray by default.
        // All terminated rays are put at the end of the sorted list.
        // Anything with a sortkey of 0xffffffff is counted as a terminated
        // ray and doesn't hit the shader evaluation code path.
        sortedEntries[i].mSortKey = 0xffffffff;

        const bool isSubsurfaceAllowed = fs.mIntegrator->isSubsurfaceAllowed(pv.subsurfaceDepth);

        // Initialize intersections.
        geom::initIntersectionFull(isectMemory[i],
                                   tls, *ray, layer,
                                   pv.mirrorDepth,
                                   pv.glossyDepth,
                                   pv.diffuseDepth,
                                   isSubsurfaceAllowed,
                                   pv.minRoughness,
                                   -ray->getDirection());

        shading::Intersection *isect = &isectMemory[i];

        // early termination if intersection doesn't provide all the
        // required primitive attributes shader request
        if (!isect->hasAllRequiredAttributes()) {
            scene_rdl2::math::Color radiance = pv.pathThroughput * fs.mFatalColor;
            currRadiance->mRadiance = RenderColor(
                radiance.r, radiance.g, radiance.b, pv.pathPixelWeight);
            currRadiance->mPathPixelWeight = pv.pathPixelWeight;
            currRadiance->mPixel = rs->mSubpixel.mPixel;
            currRadiance->mSubPixelIndex = rs->mSubpixel.mSubpixelIndex;
            currRadiance->mDeepDataHandle = pbrTls->acquireDeepData(rs->mDeepDataHandle);
            currRadiance->mCryptomatteDataHandle = pbrTls->acquireCryptomatteData(rs->mCryptomatteDataHandle);
            currRadiance->mCryptoRefP = rs->mCryptoRefP;
            currRadiance->mCryptoP0 = rs->mCryptoP0;
            currRadiance->mCryptoRefN = rs->mCryptoRefN;
            currRadiance->mCryptoUV = rs->mCryptoUV;
            currRadiance->mTilePass = rs->mTilePass;
            ++currRadiance;

            rayStatesToFree[numRayStatesToFree++] = rs;
        } else {
            // Populate deep data, first hit only for now
            if (ray->getDepth() == 0 && rs->mDeepDataHandle != pbr::nullHandle) {
                pbr::DeepData *deepData = static_cast<pbr::DeepData*>(pbrTls->getListItem(rs->mDeepDataHandle, 0));
                deepData->mHitDeep = 1;
                deepData->mSubpixelX = rs->mSubpixel.mSubpixelX;
                deepData->mSubpixelY = rs->mSubpixel.mSubpixelY;
                deepData->mRayZ = ray->dir.z;
                deepData->mDeepT = ray->tfar;
                deepData->mDeepNormal = ray->Ng;

                // retrieve all of the deep IDs for this intersection from the primitive attrs
                for (size_t i = 0; i < fs.mIntegrator->getDeepIDAttrIdxs().size(); i++) {
                    shading::TypedAttributeKey<float> deepIDAttrKey((fs.mIntegrator->getDeepIDAttrIdxs())[i]);
                    if (isect->isProvided(deepIDAttrKey)) {
                        deepData->mDeepIDs[i] = isect->getAttribute<float>(deepIDAttrKey);
                    } else {
                        deepData->mDeepIDs[i] = 0.f;
                    }
                }
            }

            // Populate cryptomatte data
            if (ray->getDepth() == 0 && rs->mCryptomatteDataHandle != pbr::nullHandle) {
                pbr::CryptomatteData *cryptomatteData =
                            static_cast<pbr::CryptomatteData*>(pbrTls->getListItem(rs->mCryptomatteDataHandle, 0));

                // Don't want to double count, so we set mHit to 0 only if it was originally a presence ray.
                if (cryptomatteData->mPrevPresence) {
                    cryptomatteData->mHit = 0;
                    handlePresenceContinue = true;
                } else {
                    cryptomatteData->mHit = 1;
                }

                // add position and normal data
                cryptomatteData->mPosition = isect->getP();
                cryptomatteData->mNormal = isect->getN();
                shading::State sstate(isect);
                sstate.getRefP(rs->mCryptoRefP);
                sstate.getRefN(rs->mCryptoRefN);

                if (isect->isProvided(shading::StandardAttributes::sP0)) {
                    rs->mCryptoP0 = scene_rdl2::math::transformPoint(
                        sstate.getGeometryObject()->getSceneClass().getSceneContext()->getRender2World()->inverse(),
                        isect->getAttribute(shading::StandardAttributes::sP0));
                }

                // Retrieve the first deep id (if present) for this intersection from the primitive attrs.
                // At the request of production, cryptomatte currently only supports one deep id associated
                // with each layer, rather than a generalized vector of ids. So here we are only interested
                // in the first id.
                if (fs.mIntegrator->getDeepIDAttrIdxs().size() != 0) {
                    shading::TypedAttributeKey<float> deepIDAttrKey(fs.mIntegrator->getDeepIDAttrIdxs()[0]);
                    if (isect->isProvided(deepIDAttrKey)) {
                        cryptomatteData->mId = isect->getAttribute<float>(deepIDAttrKey);
                    } else {
                        cryptomatteData->mId = 0.f;
                    }
                }

                shading::TypedAttributeKey<scene_rdl2::rdl2::Vec2f> cryptoUVAttrKey(fs.mIntegrator->getCryptoUVAttrIdx());
                if (isect->isProvided(cryptoUVAttrKey)) {
                    rs->mCryptoUV = isect->getAttribute<scene_rdl2::rdl2::Vec2f>(cryptoUVAttrKey);
                } else {
                    rs->mCryptoUV = isect->getSt();
                }
            }

            // Save the t value of the intersection point since we're about to relocate the ray origin there.
            // (Note that Embree uses tfar to signal the t value of the intersection point.)
            float tHit = ray->tfar;

            // TODO: Apply volume transmittance on the segment
            // ray origin --> ray isect

            // Transfer the ray to its intersection before we run shaders.
            // This is needed for texture filtering based on ray differentials.
            // Also scale the final differentials by a user factor.
            // This is left until the very end and not baked into the
            // ray differentials since the factor will typically be > 1,
            // and would cause the ray differentials to be larger
            // than necessary.
            isect->transferAndComputeDerivatives(tls, ray,
                rs->mSubpixel.mTextureDiffScale);

            sortedEntries[i].mSortKey = isect->computeShadingSortKey();

            sortedEntries[i].mLayerAssignmentId = isect->getLayerAssignmentId();

            // state and prim attr aovs
            const bool doAovs = ray->getDepth() == 0 && !fs.mAovSchema->empty();
            if (doAovs) {
                EXCL_ACCUMULATOR_PROFILE(pbrTls, EXCL_ACCUM_AOVS);
                fs.mAovSchema->initFloatArray(aovs);
                aovSetStateVars(pbrTls,
                                 *fs.mAovSchema,
                                 *isect,
                                 rs->mVolumeSurfaceT,
                                 *ray,
                                 *scene,
                                 pv.pathPixelWeight,
                                 aovs,
                                 tHit);
                aovSetPrimAttrs(pbrTls,
                                 *fs.mAovSchema,
                                 ext->getAovFlags(),
                                 *isect,
                                 pv.pathPixelWeight,
                                 aovs);
                aovAddToBundledQueue(pbrTls,
                                     *fs.mAovSchema,
                                     *isect,
                                     rs->mRay,
                                     AOV_TYPE_STATE_VAR | AOV_TYPE_PRIM_ATTR,
                                     aovs,
                                     rs->mSubpixel.mPixel,
                                     rs->mDeepDataHandle);
            }
        }

        sortedEntries[i].mRsIdx = rs - baseRayState;
        sortedEntries[i].mIsectIdx = i;
    }

    // Bulk free raystates.
    pbrTls->freeRayStates(numRayStatesToFree, rayStatesToFree);

    // Resort inputs based on mip selectors and uv coordinates.
    sortedEntries = scene_rdl2::util::smartSort32<SortedEntry, 0>(numEntries, sortedEntries, 0xffffffff, arena);

    //
    // Send terminated and primary ray radiance samples to radiance queue.
    //

    // Careful: pointer subtraction may overflow 32 bits if we have more than 2^32-1 radiances
    MNRY_ASSERT(static_cast<size_t>(currRadiance - radiances) < std::numeric_limits<int>::max());
    unsigned numTerminated = static_cast<unsigned>(currRadiance - radiances);

    if (numTerminated) {
        pbrTls->addRadianceQueueEntries(numTerminated, radiances);
        CHECK_CANCELLATION(pbrTls, return);
    }

    numEntries -= numTerminated;

    if (numEntries == 0) {
        return;
    }

    //
    // Shade all non-terminated entries and send them on to the integrator.
    //

    // Take fs.mShadingWorkloadChunkSize into account so that we're the blocks
    // of AOS data we need to convert to SOA are small enough that they stay
    // in our working caches. This works since we've already sorted the entire
    // entry list by this point.

    const unsigned shadingWorkloadChunkSize = fs.mShadingWorkloadChunkSize;
    const unsigned numWorkloadChunkBlocks = fs.mShadingWorkloadChunkSize / VLEN;

    MNRY_ASSERT(scene_rdl2::util::isPowerOfTwo(shadingWorkloadChunkSize));
    MNRY_ASSERT(shadingWorkloadChunkSize >= VLEN);

    shading::Intersectionv *isectsSOA =
        arena->allocArray<shading::Intersectionv>(numWorkloadChunkBlocks, CACHE_LINE_SIZE);

    shading::Bsdfv *bsdfv = arena->allocArray<shading::Bsdfv>(numWorkloadChunkBlocks, CACHE_LINE_SIZE);

    RayStatev *rayStatesSOA = arena->allocArray<RayStatev>(numWorkloadChunkBlocks, CACHE_LINE_SIZE);

    unsigned numRemaining = numEntries;
    const SortedEntry *endEntry = sortedEntries + numEntries;

    float *presences = arena->allocArray<float>(shadingWorkloadChunkSize);

    // Split total work load into workloads which can be kept within the
    // cache hierarchy.

    while (numRemaining) {

        CHECK_CANCELLATION(pbrTls, return);

        const uint32_t currLayerAssnId = sortedEntries->mLayerAssignmentId;
        const LightPtrList *lightList = scene->getLightPtrList(currLayerAssnId);
        const LightFilterLists *lightFilterLists = scene->getLightFilterLists(currLayerAssnId);
        const LightAccelerator *lightAcc = scene->getLightAccelerator(currLayerAssnId);

        // Figure out how many contiguous entries we have which use the
        // same lightset.
        SortedEntry *currEntry = sortedEntries + 1;
        while (currEntry != endEntry && currLayerAssnId == currEntry->mLayerAssignmentId) {
            ++currEntry;
        }

        // Note is likely that workLoadSize isn't a multiple of VLEN anymore.
        // We handle this by smearing the last valid entry across the remaining
        // lanes in the various convertAOS*ToSOA utility functions.
        //
        // For shading this suffices and we can ignore masking. It shouldn't
        // cause any slowdown since the extra lanes shouldn't fetch any memory
        // that we weren't fetching already.
        //
        // The vectorized integrator is passed in the workLoadSize value so
        // it knows how many "real" entries need processing.

        unsigned workLoadSize = scene_rdl2::math::min(unsigned(currEntry - sortedEntries),
                                          shadingWorkloadChunkSize);
        numRemaining -= workLoadSize;

        unsigned numBlocks = (workLoadSize + VLEN_MASK) / VLEN;

        // Resort ray state pointers based on sort results. This is necessary
        // so that we're updating the correct raystates objects after integration
        // is complete.
        // Also grab the presence values.
        for (unsigned i = 0; i < workLoadSize; ++i) {
            RayState *rs = baseRayState + sortedEntries[i].mRsIdx;
            rs->mAOSIsect = &isectMemory[sortedEntries[i].mIsectIdx];
            rs->mRayStateIdx = sortedEntries[i].mRsIdx;
            rayStates[i] = rs;

            presences[i] = shading::presence(&material, shadingTls,
                                             shading::State((shading::Intersection *)(rs->mAOSIsect)));

            // Check to see if we can use stochastic presence method
            if (luminance(rs->mPathVertex.pathThroughput) < (1.0f - fs.mPresenceQuality)) {
                SequenceIDIntegrator sidPresence(rs->mSubpixel.mPixel,
                                                 rs->mSubpixel.mSubpixelIndex,
                                                 SequenceType::Presence,
                                                 rs->mSequenceID,
                                                 rs->mPathVertex.presenceDepth);
                IntegratorSample1D prSamples(sidPresence);
                float prSample;
                prSamples.getSample(&prSample, rs->mPathVertex.presenceDepth);
                if (prSample < presences[i]) {
                    presences[i] = 1.f;
                } else {
                    presences[i] = 0.f;
                }
            }

            // Some NPR materials that want to allow for completely arbitrary shading normals
            // can request that the integrator does not perform any light culling based on the
            // normal. In those cases, we also want to prevent our call to adaptNormal() in the
            // Intersection when the material evaluates its normal map bindings.
            shading::Intersection * const isect = static_cast<shading::Intersection*>(rs->mAOSIsect);
            if (shading::preventLightCulling(&material, shading::State(isect))) {
               isect->setUseAdaptNormal(false);
            }
        }

        // TODO: overlapping dielectrics.  This is where most of it would be
        // implemented, similar to the scalar version in PathIntegrator.cc.

        // Do conversion to from AOS to AOSOA.
        convertAOSIntersectionsToSOA(pbrTls, workLoadSize, isectMemory, isectsSOA,
                                     &sortedEntries[0].mIsectIdx);

        // Reset bsdfv memory.
        memset(bsdfv, 0, sizeof(shading::Bsdfv) * numBlocks);

        {
            // Compute shading.
            int64_t ticks = 0;
            MCRT_COMMON_CLOCK_OPEN(fs.mRequiresHeatMap? &ticks : nullptr)

            shading::shadev(&material, shadingTls, numBlocks,
                            (const shading::Statev *)isectsSOA,
                            (scene_rdl2::rdl2::Bsdfv *)bsdfv);

            MCRT_COMMON_CLOCK_CLOSE();
            if (fs.mRequiresHeatMap) {
                heatMapBundledUpdate(pbrTls, ticks, rayStates, workLoadSize);
            }
        }

        // Cutout materials in scalar will return immediately when cutout
        // Need to allow presence continuation in this case, so flag
        bool cutout = bsdfv->mEarlyTerminationMask != 0;

        // Evaluate any extra aovs on this material
        if (!cutout && aovs) {
            aovAccumExtraAovsBundled(pbrTls, fs, rayStates, presences, isectsSOA, &material, workLoadSize);
        }

        {
            // ***** Presence handling code for regular rays.  See scalar code.

            // Count the number of presence rays we need
            unsigned numPresenceRays = 0;
            for (unsigned i = 0; i < workLoadSize; ++i) {
                if (presences[i] < 1.f - scene_rdl2::math::sEpsilon) {
                    numPresenceRays++;
                }
            }

            // Create the continuation rays if needed or handle the case of extra presence rays.
            if (numPresenceRays > 0 || handlePresenceContinue) {
                // If we just have to handle presence continue, then we don't want to allocate more ray states
                RayState **presenceRays = numPresenceRays > 0 ? pbrTls->allocRayStates(numPresenceRays) : nullptr;
                numPresenceRays = 0;

                for (unsigned i = 0; i < workLoadSize; ++i) {
                    shading::Intersection *isect = &isectMemory[i];

                    RayState *rs = rayStates[i];
                    mcrt_common::RayDifferential *ray = &rs->mRay;

                    unsigned px, py;
                    uint32ToPixelLocation(rs->mSubpixel.mPixel, &px, &py);

                    if (presences[i] < 1.f - scene_rdl2::math::sEpsilon) {
                        float totalPresence = (1.0f - rs->mPathVertex.totalPresence) * presences[i];
                        RayState *presenceRay = presenceRays[numPresenceRays++];

                        // Compute ray epsilon
                        float pathDistance = rs->mPathVertex.pathDistance + rs->mRay.getEnd();
                        float rayEpsilon = isect->getEpsilonHint();
                        if (rayEpsilon <= 0.0f) {
                            rayEpsilon = sHitEpsilonStart * std::max(pathDistance, 1.0f);
                        }

                        *presenceRay = *rs;

                        if (totalPresence >= fs.mPresenceThreshold ||
                            rs->mPathVertex.presenceDepth >= fs.mMaxPresenceDepth) {
                            // The cleanest way to terminate presence traversal is to make it impossible for the
                            // presence continuation ray to hit any more geometry.  This means we assume empty space
                            // past the last presence intersection, which will set the pixel's alpha to the
                            // total accumulated presence so far.  This is done by setting the ray's near and far
                            // distances to a large value.
                            // The other option is to assume a solid material past the last presence intersection.
                            // We don't want this because that would set the pixel alpha to 1 when we really want
                            // the alpha to be the total accumulated presence.
                            // Intuitively, you would just return here but that fails to set the path throughput
                            // and alpha correctly.  There is not a clean way to explicitly set the pixel alpha,
                            // especially in vector mode where it is not accessible at all from the presence code.
                            // It is also not possible to "return" from the vector code like in a traditional recursive
                            // ray tracer.
                            presenceRay->mRay.tnear = scene_rdl2::math::sMaxValue * 0.5f;
                            presenceRay->mRay.tfar = scene_rdl2::math::sMaxValue;
                        } else {
                            presenceRay->mRay.tnear = rayEpsilon;
                            presenceRay->mRay.tfar = rs->mRay.getOrigTfar() - rs->mRay.tfar;
                        }

                        presenceRay->mRay.setOrigTfar(presenceRay->mRay.tfar);
                        presenceRay->mPathVertex.pathDistance += rs->mRay.getEnd();
                        presenceRay->mPathVertex.pathPixelWeight *= (1.f - presences[i]);
                        presenceRay->mPathVertex.aovPathPixelWeight *= (1.f - presences[i]);
                        presenceRay->mPathVertex.pathThroughput *= (1.f - presences[i]);
                        presenceRay->mPathVertex.presenceDepth++;
                        presenceRay->mPathVertex.totalPresence = totalPresence;
                        presenceRay->mRay.geomID = -1;
                        presenceRay->mRay.primID = -1;
                        presenceRay->mRay.instID = -1;
                        presenceRay->mDeepDataHandle = nullHandle;

                        rs->mPathVertex.pathPixelWeight *= presences[i];
                        rs->mPathVertex.aovPathPixelWeight *= presences[i];
                        rs->mPathVertex.pathThroughput *= presences[i];
                        
                        // allocate memory for a new cryptomatte data object for the next presence ray
                        // this way, we can have consistent cryptomatte data for each path vertex
                        presenceRay->mCryptomatteDataHandle = pbrTls->allocList(sizeof(pbr::CryptomatteData), 1);
                        pbr::CryptomatteData *cryptomatteDataNext = static_cast<pbr::CryptomatteData*>
                                                        (pbrTls->getListItem(presenceRay->mCryptomatteDataHandle, 0));
                        cryptomatteDataNext->init(nullptr);

                        presenceRay->mCryptoRefP = scene_rdl2::math::Vec3f(0.f);
                        presenceRay->mCryptoP0 = scene_rdl2::math::Vec3f(0.f);
                        presenceRay->mCryptoRefN = scene_rdl2::math::Vec3f(0.f);
                        presenceRay->mCryptoUV = scene_rdl2::math::Vec2f(0.f);

                        // Add to the cryptomatte if we have a handle to it, and isn't a cutout:
                        if (!cutout && rs->mCryptomatteDataHandle != nullHandle && ray->getDepth() == 0) {
                            CryptomatteData *cryptomatteData =
                                static_cast<CryptomatteData*>(pbrTls->getListItem(rs->mCryptomatteDataHandle,
                                                                                  0));
                            // add presence data to cryptomatte -- the only data we don't have at this point is 
                            // radiance, which we will add to the cryptomatte in the radiance handler
                            if (cryptomatteData->mCryptomatteBuffer != nullptr && rs->mPathVertex.pathPixelWeight > 0.01f) {
                                scene_rdl2::math::Color4 beauty(0.f, 0.f, 0.f, presences[i]);
                                cryptomatteData->mCryptomatteBuffer->addSampleVector(px, py, cryptomatteData->mId, 
                                                                                    rs->mPathVertex.pathPixelWeight,
                                                                                    cryptomatteData->mPosition,
                                                                                    cryptomatteData->mNormal, 
                                                                                    beauty,
                                                                                    rs->mCryptoRefP,
                                                                                    rs->mCryptoP0,
                                                                                    rs->mCryptoRefN,
                                                                                    rs->mCryptoUV,
                                                                                    rs->mPathVertex.presenceDepth);
                            }
                            // update cryptomatte info for current presence ray
                            cryptomatteData->mPrevPresence = 1;
                            cryptomatteData->mHit = 0;
                            cryptomatteData->mPathPixelWeight = rs->mPathVertex.pathPixelWeight;
                            cryptomatteData->mPresenceDepth = rs->mPathVertex.presenceDepth;

                            // update cryptomatte info for spawned presence ray
                            cryptomatteDataNext->mPrevPresence = 1;
                            cryptomatteDataNext->mPathPixelWeight = rs->mPathVertex.pathPixelWeight;
                            cryptomatteDataNext->mCryptomatteBuffer = cryptomatteData->mCryptomatteBuffer;
                        }

                        // LPE
                        // Presence is a straight event
                        if (aovs) {
                            EXCL_ACCUMULATOR_PROFILE(pbrTls, EXCL_ACCUM_AOVS);
                            const FrameState &fs = *pbrTls->mFs;
                            const LightAovs &lightAovs = *fs.mLightAovs;
                            // transition
                            int lpeStateId = rs->mPathVertex.lpeStateId;
                            lpeStateId = lightAovs.straightEventTransition(pbrTls, lpeStateId);
                            presenceRay->mPathVertex.lpeStateId = lpeStateId;
                        }
                    } else if (!cutout && rs->mCryptomatteDataHandle != nullHandle && ray->getDepth() == 0) {
                        // Add to the cryptomatte if we came from a presence ray but we no longer hit such a case:
                        CryptomatteData *cryptomatteData =
                            static_cast<CryptomatteData*>(pbrTls->getListItem(rs->mCryptomatteDataHandle,
                                                                              0));
                        if (!cryptomatteData->mPrevPresence) {
                            continue;
                        }

                        // we will add beauty data in the radiance handler
                        if (cryptomatteData->mCryptomatteBuffer != nullptr && rs->mPathVertex.pathPixelWeight > 0.01f) {
                            scene_rdl2::math::Color4 beauty(0.f, 0.f, 0.f, presences[i]);
                            cryptomatteData->mCryptomatteBuffer->addSampleVector(px, py, cryptomatteData->mId, 
                                                                                rs->mPathVertex.pathPixelWeight,
                                                                                cryptomatteData->mPosition,
                                                                                cryptomatteData->mNormal, 
                                                                                beauty, 
                                                                                rs->mCryptoRefP,
                                                                                rs->mCryptoP0,
                                                                                rs->mCryptoRefN,
                                                                                rs->mCryptoUV,
                                                                                rs->mPathVertex.presenceDepth);
                        }
                        // update cryptomatte info for current ray
                        cryptomatteData->mPresenceDepth = rs->mPathVertex.presenceDepth;
                        cryptomatteData->mPrevPresence = 0;
                        cryptomatteData->mHit = 0;
                        cryptomatteData->mPathPixelWeight = rs->mPathVertex.pathPixelWeight;
                    }
                }

                // Trace presence continuation rays
                if (presenceRays) {
                    pbrTls->addRayQueueEntries(numPresenceRays, presenceRays);
                }
            }

            // ***** End Presence handling code for regular rays.
        }

        // Convert AOS RayState objects into SOA bundles.
        convertAOSRayStatesToSOA(pbrTls, workLoadSize, baseRayState,
                                 rayStatesSOA, &sortedEntries[0].mRsIdx);

        // Send results through to the integrator...
        fs.mIntegrator->integrateBundledv(pbrTls, shadingTls, workLoadSize, rayStatesSOA,
                                          isectsSOA, bsdfv, lightList, lightFilterLists, lightAcc, presences);

        // For hybrid scalar/vectorized rendering, the scalar ray states may still
        // be accessed during integration, so don't free them until after the
        // integration phase has completed.
        pbrTls->freeRayStates(workLoadSize, rayStates);

        // Move on to next block of inputs.
        sortedEntries += workLoadSize;
    }

    MNRY_ASSERT(sortedEntries == endEntry);

    shadingTls->clearAttributeOffsets();

    // Increment shader evaluation stats.
    Statistics &stats = pbrTls->mStatistics;
    stats.addToCounter(STATS_SHADER_EVALS, numEntries);
}

} // namespace pbr
} // namespace moonray


