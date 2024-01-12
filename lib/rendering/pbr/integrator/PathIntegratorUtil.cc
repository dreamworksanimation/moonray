// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0


#include "PathIntegratorUtil.h"

#include <moonray/rendering/bvh/shading/Intersection.h>
#include <moonray/rendering/geom/IntersectionInit.h>
#include <moonray/rendering/geom/prim/BVHUserData.h>
#include <moonray/rendering/geom/prim/NamedPrimitive.h>
#include <moonray/rendering/mcrt_common/Ray.h>
#include <moonray/rendering/mcrt_common/SOAUtil.h>
#include <moonray/rendering/pbr/core/Aov.h>
#include <moonray/rendering/pbr/core/PbrTLState.h>
#include <moonray/rendering/pbr/core/Util.h>
#include <moonray/rendering/pbr/handlers/RayHandlers.h>
#include <moonray/rendering/pbr/integrator/PathIntegrator.h>
#include <moonray/rendering/pbr/sampler/IntegratorSample.h>
#include <moonray/rendering/rt/rt.h>
#include <moonray/rendering/shading/bssrdf/VolumeSubsurface.h>
#include <moonray/rendering/shading/EvalShader.h>

#define DEBUG_BSDF_SAMPLES 0
#if DEBUG_BSDF_SAMPLES
//#include "/usr/home/kjeffery/tools/PrimiView.h"
#endif

/// Defining this avoid rays differential computations. Mainly useful for
/// profiling. TODO: put this on a runtime flag?
//#define FORCE_SKIP_RAY_DIFFERENTIALS

// using namespace scene_rdl2::math; // can't use this as it breaks openvdb in clang.
using namespace moonray::shading;

namespace moonray {
namespace pbr {

// ----------------------------------------------------------------------------

float updateMaterialPriorities(mcrt_common::RayDifferential& ray, const Scene* scene, 
        const scene_rdl2::rdl2::Camera* camera, shading::TLState* shadingTls, const shading::Intersection& isect, 
        const scene_rdl2::rdl2::Material* material, float* presence, int materialPriority, 
        const scene_rdl2::rdl2::Material** newPriorityList, int* newPriorityListCount, int presenceDepth)
{
    // find current high priority mat and get the mediumIor from that
    // if list is empty, mediumIor is 1.f
    float mediumIor = 1.f;

    // The material priority list can contain up to four materials that the
    // path is currently inside and each material has a counter.  This counter is
    // incremented each time the path enters the material and is decremented each time
    // the path exits.  This count lets us ignore self-overlapping geometry that's assigned
    // to the same material.
    getPriorityList(ray, newPriorityList, newPriorityListCount);

    // For the primary ray, get the medium_material from the camera, if there is one.
    // This is the material the camera is inside, so add it to the beginning of the priority list.
    // If/when the ray encounters a geometry with inverted normals and the same material applied, it will "exit" and 
    // remove the material from the priority list (we will use this method for air bubbles underwater). 
    if (ray.getDepth() == 0 && presenceDepth == 0) {
        const scene_rdl2::rdl2::Geometry* cameraMediumGeometry = camera->getMediumGeometry();
        const scene_rdl2::rdl2::Material* cameraMediumMaterial = camera->getMediumMaterial();

        bool hitCameraMedium = true;
        // Check for intersection with the medium_geometry, if it exists. Allows for partially submerged cameras. 
        // If there is no medium_geometry (and there IS a medium_material), apply the medium_material to all primary 
        // rays. We can think of the medium_geometry as limiting the medium_material domain. 
        if (cameraMediumGeometry) {
            hitCameraMedium = scene->intersectCameraMedium(ray);
        } 
        if (cameraMediumMaterial && hitCameraMedium) {
            addPriorityMaterial(cameraMediumMaterial, newPriorityList, newPriorityListCount);
            setPriorityList(ray, newPriorityList, newPriorityListCount);
            mediumIor = shading::ior(cameraMediumMaterial, shadingTls, shading::State(&isect));
        }
    }
    
    int highestPriority;
    const scene_rdl2::rdl2::Material* highestPriorityMaterial = getHighestPriorityMaterial(ray, highestPriority);
    // The medium IOR is the material the intersection ray is currently in, i.e. the current
    // highest priority material (or air if empty list.)
    // See Ior.h::ShaderIor()... the meaning of mediumIor is reversed
    // depending on whether we are entering or leaving the surface.
    if (highestPriorityMaterial) {
        mediumIor = shading::ior(highestPriorityMaterial, shadingTls, shading::State(&isect));
    }

    // Update the temporary priority list. Note that this list will not be set as the ray's new priority list UNLESS
    // we sample a transmissive lobe (see PathIntegratorMultiSampler::addIndirectOrDirectVisibleContributions)
    if (materialPriority > 0) {
        int materialCount = 0;

        if (isect.isEntering()) {
            // have we already entered this material?
            materialCount = getPriorityMaterialCount(material, newPriorityList, newPriorityListCount);

            // add material to new priority list
            addPriorityMaterial(material, newPriorityList, newPriorityListCount);
        } else {
            // remove material from new priority list
            removePriorityMaterial(material, newPriorityList, newPriorityListCount);

            // are we still inside this material?
            materialCount = getPriorityMaterialCount(material, newPriorityList, newPriorityListCount);

            // The medium IOR is the material we are going into, i.e. the new highest
            // priority material (or air if empty list.)
            int newHighestPriority;
            const scene_rdl2::rdl2::Material* newHighestPriorityMaterial = 
                                            getHighestPriorityMaterial(newPriorityList, newHighestPriority);
            if (newHighestPriorityMaterial) {
                mediumIor = shading::ior(newHighestPriorityMaterial, shadingTls, shading::State(&isect));
            } else {
                //priority list is empty, reset back to 1.0
                mediumIor = 1.f;
            }
        }

        if (highestPriorityMaterial) {
            if (materialPriority > highestPriority) {
                // false intersection, continue the ray without shading and the
                // medium ior isn't used
                *presence = 0.f;
            }
            if (materialCount > 0) {
                // For this to be a valid non-self-overlapping intersection,
                // the count must be zero.  I.e. we are entering for the first time
                // or exiting for the last time.
                *presence = 0.f;
            }
        }
    }
    return mediumIor;
}

//-----------------------------------------------------------------------------

finline void
integrateBsdfSample(const BsdfSampler &bSampler, int lobeIndex,
        const LightSetSampler &lSampler, const scene_rdl2::math::Color &pathThroughput,
        const LightContribution &lCo, BsdfSample &bsmp)
{
    // Mark the sample valid by setting the distance
    // TODO: Isn't it better to not add the sample in the first place ?
    bsmp.distance = (!lCo.isInvalid  ?  lCo.distance  :
            (bSampler.getDoIndirect()  ?  scene_rdl2::math::sMaxValue  :
                    BsdfSample::sInvalidDistance));
    if (bsmp.isInvalid()) {
        return;
    }

    const int ni = bSampler.getLobeSampleCount(lobeIndex);
    const float invNi = bSampler.getInvLobeSampleCount(lobeIndex);
    const scene_rdl2::math::Color pt = pathThroughput * invNi * bsmp.f * scene_rdl2::math::rcp(bsmp.pdf);

    // Compute direct tentative contribution (omit shadowing)
    const bool lobeIsMirror = bSampler.getLobe(lobeIndex)->matchesFlags(shading::BsdfLobe::ALL_MIRROR);
    const int nl = (lCo.isInvalid  ?  0  :
            lSampler.getLightSampleCount());
    bsmp.tDirect = (lCo.isInvalid  ?  scene_rdl2::math::sBlack  :  (lobeIsMirror  ?
            // Bsdf importance sampling
            lCo.Li * pt  :
            // Multiple importance sampling
            lCo.Li * pt * powerHeuristic(ni * bsmp.pdf, nl * lCo.pdf)
        ));

    // Compute indirect tentative contribution (omit Lindirect)
    bsmp.tIndirect = (bSampler.getDoIndirect()  ?  pt  :  scene_rdl2::math::sBlack);
}

void
drawBsdfSamples(pbr::TLState *pbrTls, const BsdfSampler &bSampler, const LightSetSampler &lSampler,
        const Subpixel &sp, const PathVertex &pv, const scene_rdl2::math::Vec3f& P, const scene_rdl2::math::Vec3f* N,
        float time, unsigned sequenceID, BsdfSample *bsmp, int clampingDepth,
        float clampingValue, shading::BsdfLobe::Type indirectFlags, float rayDirFootprint)
{
    IntegratorSample2D bsdfSamples;
    IntegratorSample2D lightFilterSamples;
    IntegratorSample3D lightFilterSamples3D;
    const int spIndex = (pv.nonMirrorDepth == 0) ? 0 : sp.mSubpixelIndex;

    LightContribution lCo;

    // Loop over active lobes
    int s = 0;
    const int lobeCount = bSampler.getLobeCount();
    for (int lobeIndex = 0; lobeIndex < lobeCount; ++lobeIndex) {
        const shading::BsdfLobe *lobe = bSampler.getLobe(lobeIndex);
        const int lobeSampleCount = bSampler.getLobeSampleCount(lobeIndex);

        // Setup sampler sequence
        // We want one shared sequence for depth 0
        SequenceIDIntegrator sid(   pv.nonMirrorDepth,     // TODO: Should be just depth ?
                                    sp.mPixel,
                                    lobeIndex,
                                    SequenceType::Bsdf,
                                    spIndex,
                                    sequenceID );
        int samplesSoFar = 0;
        if (pv.nonMirrorDepth == 0) {
            samplesSoFar = sp.mSubpixelIndex * lobeSampleCount;      // used here and below
            bsdfSamples.resume(sid, samplesSoFar);
        } else {
            bsdfSamples.restart(sid, lobeSampleCount);
        }

        const Scene *scene = MNRY_VERIFY(pbrTls->mFs->mScene);
        bool lightFilterNeedsSamples = scene->lightFilterNeedsSamples();

        if (lightFilterNeedsSamples) {
            SequenceIDIntegrator sidFilter(   pv.nonMirrorDepth,     // TODO: Should be just depth ?
                                              sp.mPixel,
                                              lobeIndex,
                                              SequenceType::LightFilter,
                                              spIndex,
                                              sequenceID );
            SequenceIDIntegrator sidFilter3D( pv.nonMirrorDepth,     // TODO: Should be just depth ?
                                              sp.mPixel,
                                              lobeIndex,
                                              SequenceType::LightFilter3D,
                                              spIndex,
                                              sequenceID );

            if (pv.nonMirrorDepth == 0) {
                lightFilterSamples.resume(sidFilter, samplesSoFar);
                lightFilterSamples3D.resume(sidFilter3D, samplesSoFar);
            } else {
                lightFilterSamples.restart(sidFilter, lobeSampleCount);
                lightFilterSamples3D.restart(sidFilter3D, lobeSampleCount);
            }
        }

        // Set up sampler sequence for stochastic choice of intersected light
        SequenceIDIntegrator lightChoiceSid(pv.nonMirrorDepth,
                                    sp.mPixel,
                                    lobeIndex,
                                    SequenceType::IndexSelection,
                                    sp.mSubpixelIndex,
                                    sequenceID);
        IntegratorSample1D lightChoiceSamples(lightChoiceSid);

        // lobeType is used for conditionally including ray termination lights.
        // Ray termination lights provide a way to cheaply fill in the zeros which result from terminating
        // ray paths too early. This is done by forcing the occlusion test to fail unconditionally.
        // A path can only be terminated when the relevant indirectFlags bit for this lobeType is set to false.
        shading::BsdfLobe::Type lobeType = lobe->getType();
        bool includeRayTerminationLights = (static_cast<uint32_t>(lobeType)      &
                                           ~static_cast<uint32_t>(indirectFlags) &
                                            static_cast<uint32_t>(shading::BsdfLobe::Type::ALL_LOBES))
                                                != 0;

        // lobeMask is used for comparison against a light's visibility flags.
        // This comparison is made in the LightSampler's intersectAndEval() function inside the loop below.
        int lobeMask = lobeTypeToRayMask(lobeType);

        // Loop over each lobe's samples
        for (int i = 0; i < lobeSampleCount; ++i, ++s) {

            // Draw the sample and test validity
            float bsdfSample[2];
            bsdfSamples.getSample(bsdfSample, pv.nonMirrorDepth);

            // Initialize to some value so you don't get NaNs if 
            // lightFilterNeedsSamples is false (see MOONRAY-4649)
            LightFilterRandomValues lightFilterSample = {
                            scene_rdl2::math::Vec2f(0.f, 0.f), 
                            scene_rdl2::math::Vec3f(0.f, 0.f, 0.f)};
            if (lightFilterNeedsSamples) {
                lightFilterSamples.getSample(&lightFilterSample.r2[0], pv.nonMirrorDepth);            
                lightFilterSamples3D.getSample(&lightFilterSample.r3[0], pv.nonMirrorDepth);
            }

#if DEBUG_BSDF_SAMPLES
            INIT_PRIMI_VIEW(pvg, PVPIPE);
            GroupStackGuard guard(pvg, "BSDF Samples");
            pvg.setColor(0,1,1);

            if (/*sp.mSubpixelIndex == 3 &&*/ bSampler.getCurrentLobeIndex() == 0 &&
                                              pv.nonMirrorDepth == 1) {
                pvg.point(bsdfSample[0], bsdfSample[1], 0);
                usleep(100000);
            }
#endif
            bool isValid = bSampler.sample(pbrTls, lobeIndex, P, bsdfSample[0], bsdfSample[1], bsmp[s]);
            if (!isValid) {
                continue;
            }

            // Compute lCo, which is the un-occluded light contribution for a
            // random physical light in the LightSet in the direction of the sample.
            // It should set Li and pdf to 0 if there is no light in that
            // direction, and return the distance to the chosen light if
            // there is. The distance is set to infinity if there is no light or
            // to sInfiniteLightDistance if there is an InfiniteAreaLight.
            lSampler.intersectAndEval(pbrTls->mTopLevelTls, P, N, bsmp[s].wi, lightFilterSample, time, false, includeRayTerminationLights,
                                      lightChoiceSamples, pv.nonMirrorDepth, lobeMask, lCo, rayDirFootprint);

            integrateBsdfSample(bSampler, lobeIndex, lSampler, pv.pathThroughput, lCo, bsmp[s]);
            // save the light, we'll need it (for its labels) when processing lpes
            bsmp[s].lp.light = lCo.isInvalid ? nullptr : lCo.light;

            // Selective clamp of tDirect with clampingValue
            if (pv.nonMirrorDepth >= clampingDepth) {
                bsmp[s].tDirect = smartClamp(bsmp[s].tDirect, clampingValue);
            }
        }
    }
}


//-----------------------------------------------------------------------------

finline void
integrateLightSetSample(const LightSetSampler &lSampler,
        int lightIndex, const BsdfSampler &bSampler,
        const PathVertex &pv, LightSample &lsmp,
        int clampingDepth, float clampingValue, const scene_rdl2::math::Vec3f &P)
{
    // Mark the sample valid only if we have valid lobe contributions and
    // initialize contribution for summing
    bool isInvalid = true;
    lsmp.t = scene_rdl2::math::sBlack;

    const int ni = lSampler.getLightSampleCount();
    const float invNi = lSampler.getInvLightSampleCount();

    const scene_rdl2::math::Color factor = pv.pathThroughput * invNi * lsmp.Li * scene_rdl2::math::rcp(lsmp.pdf);

    // Integrate with all the matching lobes
    const shading::BsdfSlice &slice = bSampler.getBsdfSlice();
    shading::BsdfLobe::Type flags = slice.getSurfaceFlags(bSampler.getBsdf(), lsmp.wi);
    const int lobeCount = bSampler.getLobeCount();
    MNRY_ASSERT(lobeCount > 0);

    // initialize lpe member, setting each lobe entry to a null value
    for (unsigned k = 0; k < shading::Bsdf::maxLobes; ++k) lsmp.lp.lobe[k] = nullptr;

    for (int k = 0; k < lobeCount; ++k) {
        const shading::BsdfLobe* const lobe = bSampler.getLobe(k);
        // TODO: Should we still go through MIS calculations if
        //  !lobe->matchesFlags(flags)
        if (!lobe->matchesFlags(flags)) {
            continue;
        }

        // skip lobe if light is marked as not visible from this lobe
        int lobeMask = lobeTypeToRayMask(lobe->getType());
        const Light* light = lSampler.getLight(lightIndex);
        if (!(lobeMask & light->getVisibilityMask())) {
            continue;
        }

        // Evaluate the lobe
        // TODO: Should we still go through MIS calculations if
        // isSampleInvalid() because of pdf = 0
        float pdf;
        scene_rdl2::math::Color f;
        // Pdf computation needs to be kept in sync with BsdfSampler::sample()
        // Skip path guiding on mirror lobes, because their
        // sample direction is already precisely determined.
        const PathGuide &pg = bSampler.getPathGuide();
        if (pg.canSample() && !(lobe->getType() & shading::BsdfLobe::MIRROR)) {
            const float u = pg.getPercentage();
            const float pgPdf = pg.getPdf(P, lsmp.wi);
            f = lobe->eval(slice, lsmp.wi, &pdf);
            // blending pdf values seems to work well enough in practice, and
            // allows for a potential user percentage control.
            pdf = u * pgPdf + (1.0 - u) * pdf;
        } else {
            f = lobe->eval(slice, lsmp.wi, &pdf);
        }
        if (isSampleInvalid(f, pdf)) {
            continue;
        }

        // Direct lighting tentative contribution (omits shadowing)
        // using multiple importance sampling:
        const int nk = bSampler.getLobeSampleCount(k);
        scene_rdl2::math::Color t = factor * f * powerHeuristic(ni * lsmp.pdf, nk * pdf);

        // Selective clamp of t with clampingValue
        if (pv.nonMirrorDepth >= clampingDepth) {
            t = smartClamp(t, clampingValue);
        }

        lsmp.t += t;

        // save off per-lobe t info, we'll need these for LPEs
        MNRY_ASSERT(k < static_cast<int>(shading::Bsdf::maxLobes));
        lsmp.lp.t[k] = t;
        lsmp.lp.lobe[k] = lobe;

        isInvalid = false;
    }

    if (isInvalid) {
        lsmp.setInvalid();
    }
}

/// This helper function adds "misses" to the visibility aov. We do this here because there are some light
/// samples that are thrown out early because they are facing away from the point, and they need to be 
/// added to the visibility aov before they are discarded.
void accumVisibilityAovsOccluded(float* aovs, pbr::TLState* pbrTls, const LightSetSampler& lSampler,
                                 const BsdfSampler& bSampler, const PathVertex& pv, const Light* const light,
                                 int missCount)
{
    const FrameState &fs = *pbrTls->mFs;
    const LightAovs &lightAovs = *fs.mLightAovs;
    
    // We only care about direct rays
    if (aovs && lightAovs.hasVisibilityEntries() && pv.nonMirrorDepth == 0) {
        const AovSchema &aovSchema = *fs.mAovSchema;

        bool addVisibility = true;
        for (int k = 0; k < bSampler.getLobeCount(); ++k) {
            const shading::BsdfLobe* const lobe = bSampler.getLobe(k);

            if (addVisibility) {       
                int lpeStateId = pv.lpeStateId;
                lpeStateId = lightAovs.scatterEventTransition(pbrTls, lpeStateId, lSampler.getBsdf(), *lobe);
                lpeStateId = lightAovs.lightEventTransition(pbrTls, lpeStateId, light);
                if (aovAccumVisibilityAovs(pbrTls, aovSchema, lightAovs,
                                           scene_rdl2::math::Vec2f(0.0f, missCount), 
                                           lpeStateId, aovs)) {
                    addVisibility = false;
                }
            }
        }
    }
}

void
drawLightSetSamples(pbr::TLState *pbrTls, const LightSetSampler &lSampler, const BsdfSampler &bSampler,
        const Subpixel &sp, const PathVertex &pv, const scene_rdl2::math::Vec3f &P, const scene_rdl2::math::Vec3f *N, 
        float time, unsigned sequenceID, LightSample *lsmp, int clampingDepth, float clampingValue, 
        float rayDirFootprint, float* aovs, int lightIndex)
{
    IntegratorSample3D lightSamples;
    IntegratorSample2D lightFilterSamples;
    IntegratorSample3D lightFilterSamples3D;

    const Scene *scene = MNRY_VERIFY(pbrTls->mFs->mScene);
    bool lightFilterNeedsSamples = scene->lightFilterNeedsSamples();

    const int spIndex = (pv.nonMirrorDepth == 0) ? 0 : sp.mSubpixelIndex;

    Statistics &stats = pbrTls->mStatistics;

    {
        const Light *light = lSampler.getLight(lightIndex);
        const LightFilterList *lightFilterList = lSampler.getLightFilterList(lightIndex);

        const int lightSampleCount = lSampler.getLightSampleCount();

        // Setup sampler sequence
        // We want one shared sequence for depth 0.
        const SequenceIDIntegrator sid(  pv.nonMirrorDepth,
                                         sp.mPixel,
                                         light->getHash(),
                                         SequenceType::Light,
                                         spIndex,
                                         sequenceID );
        int samplesSoFar = 0;
        if (pv.nonMirrorDepth == 0) {
            samplesSoFar = sp.mSubpixelIndex * lightSampleCount;       // used here and below
            lightSamples.resume(sid, samplesSoFar);
        } else {
            lightSamples.restart(sid, lightSampleCount);
        }

        if (lightFilterNeedsSamples) {
            const SequenceIDIntegrator sidFilter(  pv.nonMirrorDepth,
                                                   sp.mPixel,
                                                   light->getHash(),
                                                   SequenceType::LightFilter,
                                                   spIndex,
                                                   sequenceID );
            const SequenceIDIntegrator sidFilter3D(  pv.nonMirrorDepth,
                                                     sp.mPixel,
                                                     light->getHash(),
                                                     SequenceType::LightFilter3D,
                                                     spIndex,
                                                     sequenceID );

            if (pv.nonMirrorDepth == 0) {
                lightFilterSamples.resume(sidFilter, samplesSoFar);
                lightFilterSamples3D.resume(sidFilter3D, samplesSoFar);

            } else {
                lightFilterSamples.restart(sidFilter, lightSampleCount);
                lightFilterSamples3D.restart(sidFilter3D, lightSampleCount);
            }
        }

        // Loop over each light's samples
        for (int i = 0, s = 0; i < lightSampleCount; ++i, ++s) {
            // Draw the sample and test validity
            scene_rdl2::math::Vec3f lightSample;
            lightSamples.getSample(&lightSample[0], pv.nonMirrorDepth);
            
            // Initialize to some value so you don't get NaNs if 
            // lightFilterNeedsSamples is false (see MOONRAY-4649)
            LightFilterRandomValues lightFilterSample = {
                            scene_rdl2::math::Vec2f(0.f, 0.f), 
                            scene_rdl2::math::Vec3f(0.f, 0.f, 0.f)};
            if (lightFilterNeedsSamples) {
                lightFilterSamples.getSample(&lightFilterSample.r2[0], pv.nonMirrorDepth);
                lightFilterSamples3D.getSample(&lightFilterSample.r3[0], pv.nonMirrorDepth);
            }
            lSampler.sampleIntersectAndEval(pbrTls->mTopLevelTls,
                                            light, lightFilterList,
                                            P, N, lightFilterSample, time, lightSample,
                                            lsmp[s], rayDirFootprint);

            if (lsmp[s].isInvalid()) {
                // These samples occur on the shadow terminator -- they are invalid because they face
                // away from the point (dot(n, wi) < epsilon). They should count as "misses" in the visibility aov.
                accumVisibilityAovsOccluded(aovs, pbrTls, lSampler, bSampler, pv, light, /* miss count */ 1);
                continue;
            }

            integrateLightSetSample(lSampler, lightIndex, bSampler, pv, lsmp[s],
                clampingDepth, clampingValue, P);

            stats.incCounter(STATS_LIGHT_SAMPLES);
        }
    }
}


//-----------------------------------------------------------------------------

void
applyRussianRoulette(const BsdfSampler &bSampler, BsdfSample *bsmp,
        const Subpixel &sp, const PathVertex &pv, unsigned sequenceID,
        float threshold, float invThreshold)
{
    const int sampleCount = bSampler.getSampleCount();

    const SequenceIDRR sid(sp.mPixel,
            SequenceType::RussianRouletteBsdf,
            sp.mSubpixelIndex, sequenceID);
    IntegratorSample1D rrSamples;
    rrSamples.resume(sid, pv.nonMirrorDepth * sampleCount);

    // Cull rays from bsdf samples
    for (int s = 0; s < sampleCount; ++s) {
        if (bsmp[s].isInvalid()) {
            continue;
        }

        const float lumDirect = luminance(bsmp[s].tDirect);
        const float lumIndirect = luminance(bsmp[s].tIndirect);
        const float lum = scene_rdl2::math::max(lumDirect, lumIndirect);
        if (lum < threshold) {
            // This should always be < 1
            //
            // The rcp function (SSE version) produces a NaN when the value is
            // less than 0x1p-64f (the version I tested, anyway). FLT_EPSILON
            // is much greater than this, but still probably a good threshold
            // for our minimum probability.
            const float continueProbability = std::max(scene_rdl2::math::sEpsilon,
                                                       lum * invThreshold);
            float sample[1];
            rrSamples.getSample(sample, pv.nonMirrorDepth);
            if (sample[0] > continueProbability) {
                bsmp[s].setInvalid();
            } else {
                const float invContinueProbability = scene_rdl2::math::rcp(continueProbability);
                bsmp[s].tDirect *= invContinueProbability;
                bsmp[s].tIndirect *= invContinueProbability;
            }
        }
    }
}

void
applyRussianRoulette(const LightSetSampler &lSampler, LightSample *lsmp,
        const Subpixel &sp, const PathVertex &pv, unsigned sequenceID,
        float threshold, float invThreshold, IntegratorSample1D& rrSamples)
{
    const int lightSampleCount = lSampler.getLightSampleCount();

    // Cull shadow rays from the light samples
    for (int s = 0; s < lightSampleCount; ++s) {
        if (lsmp[s].isInvalid()) {
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
            const float continueProbability = std::max(scene_rdl2::math::sEpsilon,
                                                       lum * invThreshold);
            float sample[1];
            rrSamples.getSample(sample, pv.nonMirrorDepth);
            if (sample[0] > continueProbability) {
                lsmp[s].setInvalid();
            } else {
                const float continueProbabilityInv = scene_rdl2::math::rcp(continueProbability);
                lsmp[s].t *= continueProbabilityInv;

                // adjust per lobe values, if needed (see integrateLightSetSample())
                for (unsigned int k = 0; k < shading::Bsdf::maxLobes; ++k) {
                    if (lsmp[s].lp.lobe[k]) {
                        lsmp[s].lp.t[k] *= continueProbabilityInv;
                    }
                }
            }
        }
    }
}

void
accumulateRayPresence(pbr::TLState *pbrTls,
                      const Light* light,
                      const mcrt_common::Ray& shadowRay,
                      float rayEpsilon,
                      int maxDepth,
                      float& totalPresence)
{
    // Used for presence shadows in scalar and vectorized mode.
    // Presence primary visibility is handled elsewhere.

    mcrt_common::ThreadLocalState *topLevelTls = pbrTls->mTopLevelTls;
    shading::TLState *shadingTls = topLevelTls->mShadingTls.get();
    const Scene *scene = MNRY_VERIFY(pbrTls->mFs->mScene);
    const scene_rdl2::rdl2::Light* rdlLight = light->getRdlLight();

    mcrt_common::Ray currentShadowRay = shadowRay;
    shading::Intersection isect;

    while (scene->intersectPresenceRay(topLevelTls, currentShadowRay, isect)) {

        // handle shadow linking. This primitive may not cast shadow
        // for certain specified lights in occlusionSkipList
        geom::internal::BVHUserData* userData = static_cast<geom::internal::BVHUserData*>(currentShadowRay.ext.userData);
        const geom::internal::NamedPrimitive* primPtr =
            static_cast<const geom::internal::NamedPrimitive*>(userData->mPrimitive);
        const geom::internal::ShadowLinking* shadowLinking = primPtr->getShadowLinking(isect.getLayerAssignmentId());
        if (!shadowLinking || shadowLinking->canCastShadow(rdlLight)) {
            // Finish setting up the intersection with the minimum needed to evaluate the material's
            // presence
            geom::initIntersectionPhase2(isect,
                                         topLevelTls,
                                         0, // mirrordepth
                                         0, // glossydepth
                                         0, // diffuseDepth
                                         false, // subsurface allowed
                                         scene_rdl2::math::Vec2f(0.0f, 0.0f), // minRoughness
                                         -currentShadowRay.getDirection());
            // get the presence value from the material
            const scene_rdl2::rdl2::Material* material = isect.getMaterial()->asA<scene_rdl2::rdl2::Material>();
            MNRY_ASSERT(material != nullptr);
            float presence = shading::presence(material, shadingTls, shading::State(&isect));
            totalPresence += (1.0f - totalPresence) * presence;
        }

        if (currentShadowRay.getDepth() < maxDepth && totalPresence < pbrTls->mFs->mPresenceThreshold) {
            float tnear = rayEpsilon + currentShadowRay.tfar;
            if (tnear >= shadowRay.tfar) {
                return;
            }
            currentShadowRay = mcrt_common::Ray(currentShadowRay, tnear, shadowRay.tfar);
            // constructor increments the depth automatically
        } else {
            return;
        }
    }
}

void
scatterAndScale(const shading::Intersection &isect,
                const shading::BsdfLobe &lobe,
                const scene_rdl2::math::Vec3f &wo,
                const scene_rdl2::math::Vec3f &wi,
                float scale,
                float r1, float r2,
                mcrt_common::RayDifferential &rd)
{
    scatterAndScale(isect.getdNdx(),
                    isect.getdNdy(),
                    lobe,
                    wo,
                    wi,
                    scale,
                    r1,
                    r2,
                    rd);
}

void
scatterAndScale(const scene_rdl2::math::Vec3f &dNdx,
                const scene_rdl2::math::Vec3f &dNdy,
                const shading::BsdfLobe &lobe,
                const scene_rdl2::math::Vec3f &wo,
                const scene_rdl2::math::Vec3f &wi,
                float scale,
                float r1, float r2,
                mcrt_common::RayDifferential &rd)
{
    MNRY_ASSERT(isNormalized(wi));

#ifdef FORCE_SKIP_RAY_DIFFERENTIALS

    rd.clearDifferentials();

#else

    if (rd.hasDifferentials()) {

        MNRY_ASSERT(isNormalized(rd.getDirX()));
        MNRY_ASSERT(isNormalized(rd.getDirY()));

        scene_rdl2::math::Vec3f dDdx = rd.getdDdx();
        scene_rdl2::math::Vec3f dDdy = rd.getdDdy();

        lobe.differentials(wo, wi, r1, r2, dNdx, dNdy,
                           dDdx, dDdy);

        scene_rdl2::math::Vec3f const &origin = rd.getOrigin();
        rd.setOriginX(origin + (rd.getOriginX() - origin) * scale);
        rd.setOriginY(origin + (rd.getOriginY() - origin) * scale);

        // Scaling dDdx is more accurate than scaling (mDirX - direction)
        // because of the normalization that happens when converting from
        // dDdx to mDirX
        dDdx *= scale;
        dDdy *= scale;

        rd.setDirX(normalize(wi + dDdx));
        rd.setDirY(normalize(wi + dDdy));
    }

#endif  // FORCE_SKIP_RAY_DIFFERENTIALS

    rd.setDirection(wi);
}

// Subsurface Integrator Utility Functions
// Tri-Planar Projection Sampling Probabilities - T, B, N
// Note that [1] suggests a distribution of [0.5, 0.25, 0.25] whereas I find
// [0.75, 0.125, 0.125] works better
// 75% to N & the rest divided equally between T & B
static std::array<float, 3> sBssrdfAxisProbabilities = { 0.75f, 0.125f, 0.125f };

// Selects an axis for projection
int
bssrdfSelectAxisAndRemapSample(const scene_rdl2::math::ReferenceFrame &localF,
                               float &rnd,
                               scene_rdl2::math::Vec3f &directionProj)
{
    // We divide the T and B probabilities further between positive and negative options
    std::array<float, 5> axisProbabilities   = {sBssrdfAxisProbabilities[0],
                                                sBssrdfAxisProbabilities[1]/2.0f,
                                                sBssrdfAxisProbabilities[1]/2.0f,
                                                sBssrdfAxisProbabilities[2]/2.0f,
                                                sBssrdfAxisProbabilities[2]/2.0f};

    std::array<float, 5> axisCDF;
    axisCDF[0] = sBssrdfAxisProbabilities[0];
    for (int i = 1; i < 5; i++) {
        axisCDF[i] = axisProbabilities[i] + axisCDF[i-1];
    }
    MNRY_ASSERT(scene_rdl2::math::isOne(axisCDF[4]));

    int axisIndex;
    if (rnd < axisCDF[0]) {
        axisIndex = 0;
        directionProj = -localF.getN();
        rnd /= axisProbabilities[0];
    } else if (rnd < axisCDF[1]) {
        axisIndex = 1;
        directionProj = -localF.getX();
        rnd = (rnd - axisCDF[0]) / axisProbabilities[1];
    } else if (rnd < axisCDF[2]) {
        axisIndex = 2;
        directionProj = localF.getX();
        rnd = (rnd - axisCDF[1]) / axisProbabilities[2];
    } else if (rnd < axisCDF[3]) {
        axisIndex = 3;
        directionProj = -localF.getY();
        rnd = (rnd - axisCDF[2]) / axisProbabilities[3];
    } else {
        axisIndex = 4;
        directionProj = localF.getY();
        rnd = (rnd - axisCDF[3]) / axisProbabilities[4];
    }
    return axisIndex;
}


// Converts the *local* BSSRDF Offset into *render* space using the
// appropriate local frame based on the projection axis
scene_rdl2::math::Vec3f
bssrdfOffsetLocalToGlobal(const scene_rdl2::math::ReferenceFrame &localF,
                          int axisIndex,
                          const scene_rdl2::math::Vec3f &localOffset)
{
    scene_rdl2::math::Vec3f globalOffset;

    if (axisIndex == 0) {
        // Projecting along N
        globalOffset = localOffset[0] * localF.getX() + localOffset[1] * localF.getY();
    } else if (axisIndex == 1 || axisIndex == 2) {
        // Projecting along X
        globalOffset = localOffset[0] * localF.getY() + localOffset[1] * localF.getN();
    } else {
        // Projecting along Y
        globalOffset = localOffset[0] * localF.getN() + localOffset[1] * localF.getX();
    }
    return globalOffset;
}

// Calculates the MIS Weight Using Veachs' One-Sample Method
float
bssrdfGetMISAxisWeight(const scene_rdl2::math::ReferenceFrame &localF,
                       const scene_rdl2::math::Vec3f &sampleNormal,
                       const scene_rdl2::math::Vec3f &sampleOffset,
                       const shading::Bssrdf &bssrdf)
{
    // Convert the global offset into a Local offset
    scene_rdl2::math::Vec3f dpLocal = localF.globalToLocal(sampleOffset);
    // Distance along each projection axis
    float rN = scene_rdl2::math::sqrt(dpLocal[0] * dpLocal[0] + dpLocal[1] * dpLocal[1]);
    float rX = scene_rdl2::math::sqrt(dpLocal[1] * dpLocal[1] + dpLocal[2] * dpLocal[2]);
    float rY = scene_rdl2::math::sqrt(dpLocal[2] * dpLocal[2] + dpLocal[0] * dpLocal[0]);

    // Jacobian for Each Projection Axis
    float nN = scene_rdl2::math::abs(dot(localF.getN(), sampleNormal));
    float nX = scene_rdl2::math::abs(dot(localF.getX(), sampleNormal));
    float nY = scene_rdl2::math::abs(dot(localF.getY(), sampleNormal));

    // PDF for Each Projection Axis
    float pdfN = bssrdf.pdfLocal(rN);
    float pdfX = bssrdf.pdfLocal(rX);
    float pdfY = bssrdf.pdfLocal(rY);

    float misWeight  = sBssrdfAxisProbabilities[0] * pdfN * nN;
    misWeight       += sBssrdfAxisProbabilities[1] * pdfX * nX;
    misWeight       += sBssrdfAxisProbabilities[2] * pdfY * nY;

    if (!scene_rdl2::math::isZero(misWeight)) {
        return 1.0f/misWeight;
    }
    else {
        // throw an error?
        return 1.0f;
    }
}

// Calculates the Area-Compensation Term.
// We divide the analytically computed diffuse reflectance by the one
// computed via sampling. This gives us a measure of how much the underlying
// geometry deviates from the "semi-infinite, planar" assumption for diffusion based BSSRDFs.
// On a regular planar surface, this compensation term amount to 1.
scene_rdl2::math::Color
bssrdfAreaCompensation(const scene_rdl2::math::Color &measuredDiffuseReflectance,
                       const scene_rdl2::math::Color &analyticDiffuseReflectance)
{
    scene_rdl2::math::Color areaCompensation = scene_rdl2::math::sWhite;
    for (int i = 0; i < 3; i++) {
        // Checking if the measured reflectance was too low, indicating really bad sampling
        // This could happen at really thin areas or because of incorrect sampling, which could spike the
        // reflectance results. Hence the check below. 0.01f has been good in my tests so far.
        if ((measuredDiffuseReflectance[i]-0.01f) > 0) {
            areaCompensation[i] = analyticDiffuseReflectance[i] / measuredDiffuseReflectance[i];
        }
    }
    return areaCompensation;
}

// End of Subsurface Integrator Utility Functions

//----------------------------------------------------------------------------------------
// Shadow Falloff Utility Functions 
enum clearRadiusFalloffType {
    FALLOFF_LINEAR = 0,
    FALLOFF_EXPONENTIAL_UP,
    FALLOFF_EXPONENTIAL_DOWN,
    FALLOFF_SMOOTH
};

scene_rdl2::math::Color 
calculateShadowFalloff(const Light *light, float distToLight, const scene_rdl2::math::Color unoccludedColor) 
{
    float crFalloffMin = light->getClearRadius();
    float crFalloffMax = light->getClearRadiusFalloffDistance() + light->getClearRadius();
    float falloffDist = light->getClearRadiusFalloffDistance();
    int interpolationType = light->getClearRadiusInterpolationType(); 

    // if distance to light less than clear radius, no shadow
    if (distToLight <= crFalloffMin) {
        return unoccludedColor;
    } 
    
    if (distToLight <= crFalloffMax) {
        float t = (distToLight - crFalloffMin) / falloffDist;
        float weight;
        switch (interpolationType) {
            case (FALLOFF_LINEAR):
                weight = t;
                break;
            case (FALLOFF_EXPONENTIAL_UP):
                weight = t*t;
                break;
            case (FALLOFF_EXPONENTIAL_DOWN):
                weight = 1.f - t;
                weight = 1.f - (weight * weight);
                break;
            case (FALLOFF_SMOOTH):
                weight = t*t * (3 - 2*t);
                break;
            default:
                return scene_rdl2::math::Color(0.f);
        }
        scene_rdl2::math::Color leftCol = unoccludedColor;
        scene_rdl2::math::Color rightCol = scene_rdl2::math::Color(0.f);
        return leftCol + weight * (rightCol - leftCol);
    }  
    return scene_rdl2::math::Color(0.f);
}

//-----------------------------------------------------------------------------

bool
CPP_isIntegratorAccumulatorRunning(pbr::TLState *pbrTls)
{
    return pbrTls->isIntegratorAccumulatorRunning();
}

bool
CPP_isIspcAccumulatorRunning(pbr::TLState *pbrTls)
{
    return pbrTls->isIspcAccumulatorRunning();
}

void
CPP_computeRadianceSubsurface(const PathIntegrator * pathIntegrator,
                               pbr::PbrTLState *     pbrTls,
                              const uint32_t *       rayStateIndices,
                              const Bssrdfv *        bssrdfv,
                              const VolumeSubsurfacev * volumeSubsurfacev,
                              const LightSet *       lightSet,
                              const int              materialLabelIds,
                              const int              lpeMaterialLabelIds,
                              const int *            geomLabelIds,
                              const int *            doIndirect,
                              const float *          rayEpsilon,
                              const float *          shadowRayEpsilon,
                                    uint32_t *       sequenceID,
                                    float *          results,        // VLEN rgb colors in SOA format
                                    float *          ssAovResults,   // VLEN rgb colors in SOA format
                                    int32_t          lanemask)
{
    MNRY_ASSERT(pbrTls->isIntegratorAccumulatorRunning());
    MNRY_ASSERT(pbrTls->isIspcAccumulatorRunning());

    pbrTls->stopIspcAccumulator();

    scene_rdl2::alloc::Arena *arena = pbrTls->mArena;
    SCOPED_MEM(arena);

    RayState *baseRayState = indexToRayState(0);

    shading::Bsdf bsdf;

    // This is an interesting case for aovs.  We are now dropping into the
    // depth-first integrator which means that aovs are handled by
    // continually accumulating results into a single aov array.
    // Note that the bundled integrator does not handle aovs this way.
    // Instead it immediately queues results as they are matched.
    //
    // We will allocate an aov array, and then queue the results
    // when we exit back into the breadth-first integrator.  We
    // assume that we are called from the bundled integrator, which
    // means that all the non-light/visibility aovs have already been handled.
    // So this means we only need to allocate aovs if we have light
    // aovs in the scene.
    float *aovs = nullptr;

    unsigned int aovNumChannels = 0;
    const AovSchema &aovSchema = *pbrTls->mFs->mAovSchema;
    const LightAovs &lightAovs = *pbrTls->mFs->mLightAovs;
    if (!aovSchema.empty() && lightAovs.hasEntries()) {
        aovNumChannels = aovSchema.numChannels();
        aovs = arena->allocArray<float>(aovNumChannels);
    }

    for (unsigned i = 0; i < VLEN; ++i) {
        // Don't compute subsurface radiance for invalid lanes.
        if (!isActive(lanemask, i)) {
            continue;
        }
        // Convert from Bssrdf from SOA to AOS.
        shading::Bssrdf *bssrdf = shading::createBSSRDF(arena, bssrdfv, i);
        // Convert VolumeSubsurface from SOA to AOS.
        shading::VolumeSubsurface *volumeSubsurface =
                shading::createVolumeSubsurface(arena, volumeSubsurfacev, i);
        if (!bssrdf && !volumeSubsurface) {
            continue;
        }

        bsdf.setLabelIds(materialLabelIds, lpeMaterialLabelIds, geomLabelIds[i]);
        bsdf.setBssrdf(bssrdf);
        bsdf.setVolumeSubsurface(volumeSubsurface);

        RayState *rs = &baseRayState[rayStateIndices[i]];

        const Subpixel &sp = rs->mSubpixel;
        PathVertex &pv = rs->mPathVertex;
        const mcrt_common::RayDifferential &ray = rs->mRay;
        const shading::Intersection &isect = *rs->mAOSIsect;

        // Setup a slice, which handles selecting the lobe types and setup
        // evaluations to include the cosine term.
        // Note: Even though we may not be doing indirect for certain lobe types
        // (according to indirectFlags), we still want to draw samples according
        // to all lobes for direct lighting MIS.
        shading::BsdfSlice slice(isect.getNg(), -ray.getDirection(), true,
                isect.isEntering(), pbrTls->mFs->mShadowTerminatorFix, shading::BsdfLobe::ALL);

        if (aovs) {
            aovSchema.initFloatArray(aovs);
        }

        scene_rdl2::math::Color ssAov = scene_rdl2::math::Color(0);
        scene_rdl2::math::Color radiance = scene_rdl2::math::Color(0);
        if (bssrdf) {
            MNRY_ASSERT(!volumeSubsurface); // else our depth count will be messed up
            // increment subsurface depth
            // this is not a double increment.
            // see note in integrateBundled (PathIntegratorBundled.ispc)
            pv.subsurfaceDepth += 1;
            radiance += pathIntegrator->computeRadianceDiffusionSubsurface(
                    pbrTls, bsdf, sp, pv, ray, isect, slice, *bssrdf, *lightSet,
                    doIndirect[i], rayEpsilon[i], shadowRayEpsilon[i], sequenceID[i], ssAov, aovs);
        }
        if (volumeSubsurface) {
            MNRY_ASSERT(!bssrdf); // else our depth count will be messed up
            // increment subsurface depth
            // this is not a double increment.
            // see note in integrateBundled (PathIntegratorBundled.ispc)
            pv.subsurfaceDepth += 1;
            radiance += pathIntegrator->computeRadiancePathTraceSubsurface(
                    pbrTls, bsdf, sp, pv, ray, isect, *volumeSubsurface, *lightSet,
                    doIndirect[i], rayEpsilon[i], shadowRayEpsilon[i], sequenceID[i], ssAov, aovs);
        }

        if (aovs) {
            // This code only computes light and visibility AOVs, so we only want to pass along those
            // types of AOVs.  But, the aov buffer we created here contains ALL the AOVs.
            // If we pass all types of AOVs, this leads to problems with things
            // like depth, because we don't set the depth AOV value here but would then pass
            // the default zero depth value back to the bundled integrator.  This leads to erroneous zero
            // depth values in the final AOV output.
            // The light aovs had the stateids checked prior to being accumulated into aovs,
            // so no need to further check them.
            aovAddToBundledQueue(pbrTls, aovSchema, isect, ray,
                AOV_TYPE_LIGHT_AOV | AOV_TYPE_VISIBILITY_AOV, aovs, rs->mSubpixel.mPixel,
                rs->mDeepDataHandle);
        }

        results[i] += radiance.r;
        results[VLEN + i] += radiance.g;
        results[VLEN * 2 + i] += radiance.b;

        ssAovResults[i] += ssAov.r;
        ssAovResults[VLEN + i] += ssAov.g;
        ssAovResults[VLEN * 2 + i] += ssAov.b;
    }

    pbrTls->startIspcAccumulator();
}

void
CPP_addIncoherentRayQueueEntries(pbr::TLState *pbrTls, const RayStatev *rayStatesv,
                                 unsigned numRayStates, const unsigned *indices)
{
    MNRY_ASSERT(pbrTls->isIntegratorAccumulatorRunning());
    MNRY_ASSERT(pbrTls->isIspcAccumulatorRunning());
    MNRY_ASSERT(numRayStates);

    EXCL_ACCUMULATOR_PROFILE(pbrTls, EXCL_ACCUM_POST_INTEGRATION);

    pbrTls->stopIspcAccumulator();

    scene_rdl2::alloc::Arena *arena = pbrTls->mArena;
    SCOPED_MEM(arena);

    // Allocate AOS RayStates to copy data into.
    RayState **rayStates = pbrTls->allocRayStates(numRayStates);

    //
    // Convert SOA ray states to AOS.
    //
    {
        ACCUMULATOR_PROFILE(pbrTls, ACCUM_SOA_TO_AOS_RAYSTATES);

#if (VLEN == 16u)
        mcrt_common::convertSOAToAOSIndexed_AVX512
#elif (VLEN == 8u)
        mcrt_common::convertSOAToAOSIndexed_AVX
#else
#error Requires at least AVX to build.
#endif
        <sizeof(pbr::RayStatev), sizeof(pbr::RayStatev), sizeof(uint32_t), 0>
            (numRayStates, indices, (const uint32_t *)rayStatesv, (uint32_t **)rayStates);
    }

    for (unsigned i = 0; i < numRayStates; ++i) {
        MNRY_ASSERT(isValid(rayStates[i]));
    }

    pbrTls->addRayQueueEntries(numRayStates, rayStates);

    pbrTls->startIspcAccumulator();
}

void
CPP_addOcclusionQueueEntries(pbr::TLState *pbrTls, const BundledOcclRayv *occlRaysv,
                             unsigned numOcclRays, const unsigned *indices)
{
    MNRY_ASSERT(pbrTls->isIntegratorAccumulatorRunning());
    MNRY_ASSERT(pbrTls->isIspcAccumulatorRunning());
    MNRY_ASSERT(numOcclRays);

    EXCL_ACCUMULATOR_PROFILE(pbrTls, EXCL_ACCUM_POST_INTEGRATION);

    pbrTls->stopIspcAccumulator();

    scene_rdl2::alloc::Arena *arena = pbrTls->mArena;
    SCOPED_MEM(arena);

    // Allocate AOS BundledOcclRays to copy data into.
    BundledOcclRay *occlRayMemory = arena->allocArray<BundledOcclRay>
        (numOcclRays, CACHE_LINE_SIZE);
    BundledOcclRay **occlRays = arena->allocArray<BundledOcclRay *>
        (numOcclRays, CACHE_LINE_SIZE);

    for (unsigned i = 0; i < numOcclRays; ++i) {
        occlRays[i] = &occlRayMemory[i];
    }

    //
    // Convert SOA occl rays to AOS.
    //
    {
        ACCUMULATOR_PROFILE(pbrTls, ACCUM_SOA_TO_AOS_OCCL_RAYS);

#if (VLEN == 16u)
        mcrt_common::convertSOAToAOSIndexed_AVX512
#elif (VLEN == 8u)
        mcrt_common::convertSOAToAOSIndexed_AVX
#else
#error Requires at least AVX to build.
#endif
        <sizeof(pbr::BundledOcclRayv), sizeof(pbr::BundledOcclRayv), sizeof(uint32_t), 0>
            (numOcclRays, indices, (const uint32_t *)occlRaysv, (uint32_t **)occlRays);
    }

    for (unsigned i = 0; i < numOcclRays; ++i) {
        MNRY_ASSERT(occlRays[i]->isValid());
    }

    pbrTls->addOcclusionQueueEntries(numOcclRays, occlRayMemory);

    pbrTls->startIspcAccumulator();
}

void
CPP_addPresenceShadowsQueueEntries(pbr::TLState *pbrTls, const BundledOcclRayv *occlRaysv,
                                   unsigned numOcclRays, const unsigned *indices)
{
    MNRY_ASSERT(pbrTls->isIntegratorAccumulatorRunning());
    MNRY_ASSERT(pbrTls->isIspcAccumulatorRunning());
    MNRY_ASSERT(numOcclRays);

    EXCL_ACCUMULATOR_PROFILE(pbrTls, EXCL_ACCUM_POST_INTEGRATION);

    pbrTls->stopIspcAccumulator();

    scene_rdl2::alloc::Arena *arena = pbrTls->mArena;
    SCOPED_MEM(arena);

    // Allocate AOS BundledOcclRays to copy data into.
    BundledOcclRay *occlRayMemory = arena->allocArray<BundledOcclRay>
        (numOcclRays, CACHE_LINE_SIZE);
    BundledOcclRay **occlRays = arena->allocArray<BundledOcclRay *>
        (numOcclRays, CACHE_LINE_SIZE);

    for (unsigned i = 0; i < numOcclRays; ++i) {
        occlRays[i] = &occlRayMemory[i];
    }

    //
    // Convert SOA occl rays to AOS.
    //
    {
        ACCUMULATOR_PROFILE(pbrTls, ACCUM_SOA_TO_AOS_OCCL_RAYS);

#if (VLEN == 16u)
        mcrt_common::convertSOAToAOSIndexed_AVX512
#elif (VLEN == 8u)
        mcrt_common::convertSOAToAOSIndexed_AVX
#else
#error Requires at least AVX to build.
#endif
        <sizeof(pbr::BundledOcclRayv), sizeof(pbr::BundledOcclRayv), sizeof(uint32_t), 0>
            (numOcclRays, indices, (const uint32_t *)occlRaysv, (uint32_t **)occlRays);
    }

    for (unsigned i = 0; i < numOcclRays; ++i) {
        MNRY_ASSERT(occlRays[i]->isValid());
    }

    pbrTls->addPresenceShadowsQueueEntries(numOcclRays, occlRayMemory);

    pbrTls->startIspcAccumulator();
}

void
CPP_addRadianceQueueEntries(pbr::TLState *pbrTls, const BundledRadiancev *radiancesv,
                            unsigned numRadiances, const unsigned *indices)
{
    MNRY_ASSERT(pbrTls->isIntegratorAccumulatorRunning());
    MNRY_ASSERT(pbrTls->isIspcAccumulatorRunning());
    MNRY_ASSERT(numRadiances);

    EXCL_ACCUMULATOR_PROFILE(pbrTls, EXCL_ACCUM_POST_INTEGRATION);

    pbrTls->stopIspcAccumulator();

    scene_rdl2::alloc::Arena *arena = pbrTls->mArena;
    SCOPED_MEM(arena);

    // Allocate AOS BundledRadiance to copy data into.
    BundledRadiance *radianceMemory = arena->allocArray<BundledRadiance>
        (numRadiances, CACHE_LINE_SIZE);
    BundledRadiance **radiances = arena->allocArray<BundledRadiance *>
        (numRadiances, CACHE_LINE_SIZE);

    for (unsigned i = 0; i < numRadiances; ++i) {
        radiances[i] = &radianceMemory[i];
    }

    //
    // Convert SOA bundled radiances to AOS.
    //
    {
        ACCUMULATOR_PROFILE(pbrTls, ACCUM_SOA_TO_AOS_RADIANCES);

#if (VLEN == 16u)
        mcrt_common::convertSOAToAOSIndexed_AVX512
#elif (VLEN == 8u)
        mcrt_common::convertSOAToAOSIndexed_AVX
#else
#error Requires at least AVX to build.
#endif
        <sizeof(pbr::BundledRadiancev), sizeof(pbr::BundledRadiancev), sizeof(uint32_t), 0>
            (numRadiances, indices, (const uint32_t *)radiancesv, (uint32_t **)radiances);
    }

    pbrTls->addRadianceQueueEntries(numRadiances, radianceMemory);

    pbrTls->startIspcAccumulator();
}

//-----------------------------------------------------------------------------

void
CPP_computeRadianceEmissiveRegionsBundled(const PathIntegrator *pathIntegrator,
    PbrTLState *pbrTls, const uint32_t *rayStateIndices, const float *rayEpsilons,
    const shading::Bsdfv *bsdfv, const shading::BsdfSlicev *slicev,
    float *results, int32_t lanemask)
{
    MNRY_ASSERT(pbrTls->isIntegratorAccumulatorRunning());
    MNRY_ASSERT(pbrTls->isIspcAccumulatorRunning());

    pbrTls->stopIspcAccumulator();

    RayState *baseRayState = indexToRayState(0);

    for (unsigned i = 0; i < VLEN; ++i) {
        // Don't compute emissive region radiance for invalid lanes
        if (!isActive(lanemask, i)) {
            continue;
        }

        RayState *rs = &baseRayState[rayStateIndices[i]];
        const float rayEpsilon = rayEpsilons[i];

        scene_rdl2::math::Color radiance = pathIntegrator->computeRadianceEmissiveRegionsBundled(pbrTls,
            *rs, *bsdfv, *slicev, rayEpsilon, i);

        results[i] += radiance.r;
        results[VLEN + i] += radiance.g;
        results[VLEN * 2 + i] += radiance.b;
    }

    pbrTls->startIspcAccumulator();
}

void
CPP_applyVolumeTransmittance(const PathIntegrator *pathIntegrator,
    PbrTLState *pbrTls, const uint32_t *rayStateIndices, int32_t lanemask)

{
    MNRY_ASSERT(pbrTls->isIntegratorAccumulatorRunning());
    MNRY_ASSERT(pbrTls->isIspcAccumulatorRunning());

    pbrTls->stopIspcAccumulator();

    RayState *baseRayState = indexToRayState(0);

    for (unsigned i = 0; i < VLEN; ++i) {
        // Don't apply volume transmittance for invalid lanes
        if (!isActive(lanemask, i)) {
            continue;
        }

        RayState *rs = &baseRayState[rayStateIndices[i]];
        PathVertex &pv = rs->mPathVertex;

        MNRY_ASSERT(rs->mVolHit); // should have been checked on the ispc side and inclued in the lanemask
        pv.pathThroughput *= rs->mVolTr * rs->mVolTh;
    }


    pbrTls->startIspcAccumulator();
}

// -----------------------------------------------------------------

} // namespace pbr
} // namespace moonray

