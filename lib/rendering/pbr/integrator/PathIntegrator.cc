// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

///
/// @file PathIntegrator.cc
/// $Id$
///

#include "PathIntegrator.h"
#include "PathIntegratorUtil.h"

#include "BsdfSampler.h"
#include "LightSetSampler.h"
#include "VolumeTransmittance.h"
#include <moonray/rendering/pbr/camera/Camera.h>
#include <moonray/rendering/pbr/core/Aov.h>
#include <moonray/rendering/pbr/core/Constants.h>
#include <moonray/rendering/pbr/core/Cryptomatte.h>
#include <moonray/rendering/pbr/core/DebugRay.h>
#include <moonray/rendering/pbr/core/DeepBuffer.h>
#include <moonray/rendering/pbr/core/RayState.h>
#include <moonray/rendering/pbr/core/Scene.h>
#include <moonray/rendering/pbr/core/Util.h>
#include <moonray/rendering/pbr/light/Light.h>

#include <moonray/rendering/geom/IntersectionInit.h>
#include <moonray/rendering/geom/prim/BVHUserData.h>
#include <moonray/rendering/mcrt_common/Ray.h>
#include <moonray/rendering/mcrt_common/ThreadLocalState.h>
#include <moonray/rendering/rt/rt.h>
#include <moonray/rendering/rt/EmbreeAccelerator.h>
#include <moonray/rendering/bvh/shading/AttributeKey.h>
#include <moonray/rendering/bvh/shading/ShadingTLState.h>
#include <moonray/rendering/bvh/shading/ThreadLocalObjectState.h>
#include <moonray/rendering/shading/BsdfBuilder.h>
#include <moonray/rendering/shading/Material.h>
#include <moonray/rendering/shading/AovLabels.h>
#include <moonray/rendering/shading/bsdf/Bsdf.h>
#include <moonray/rendering/shading/bsdf/BsdfSlice.h>
#include <moonray/rendering/shading/bssrdf/Bssrdf.h>
#include <moonray/rendering/shading/EvalShader.h>
#include <moonray/rendering/shading/Geometry.h>
#include <moonray/rendering/shading/ispc/Shadingv.h>
#include <moonray/rendering/shading/Material.h>

#include <moonray/common/time/Ticker.h>

#include <scene_rdl2/common/math/Constants.h>
#include <scene_rdl2/scene/rdl2/SceneContext.h>
#include <scene_rdl2/scene/rdl2/Material.h>
#include <scene_rdl2/scene/rdl2/VisibilityFlags.h>

#include <limits.h>

namespace ispc {
extern "C"
uint32_t PathIntegrator_hudValidation(bool);

extern "C"
void PathIntegrator_integrateBundled(const moonray::pbr::PathIntegrator *integrator,
                                     moonray::pbr::TLState *pbrTls,
                                     moonray::shading::TLState *shadingTls,
                                     unsigned numEntries,
                                     moonray::pbr::RayStatev *rayStates,
                                     const moonray::shading::Intersectionv *isects,
                                     const moonray::shading::Bsdfv *bsdfs,
                                     const moonray::pbr::LightSet *lightList,
                                     const float *presences);
}

// using namespace scene_rdl2::math; // can't use this as it breaks openvdb in clang.
using scene_rdl2::logging::Logger;


namespace moonray {

namespace pbr {


namespace {

/// Output intersection geometric normal as color    
scene_rdl2::math::Color
generateNormalColor(shading::Intersection const &isect)
{
    scene_rdl2::math::Vec3f ng = isect.getNg();
    return scene_rdl2::math::Color(0.5f * ng[0] + 0.5f, 0.5f * ng[1] + 0.5f, 0.5f * ng[2] + 0.5f);
}
/// Output intersection shading normal as color
scene_rdl2::math::Color
generateShadingNormalColor(shading::Intersection const &isect)
{
    scene_rdl2::math::Vec3f n = isect.getN();
    return scene_rdl2::math::Color(0.5f * n[0] + 0.5f, 0.5f * n[1] + 0.5f, 0.5f * n[2] + 0.5f);
}
/// Output intersection facing ratio as color
scene_rdl2::math::Color
generateFacingRatioColor(shading::Intersection const &isect,
                                         mcrt_common::RayDifferential const &ray)
{
    scene_rdl2::math::Vec3f viewDirection = - ray.getDirection();
    viewDirection.normalize();
    scene_rdl2::math::Vec3f n = isect.getN();
    float dot = scene_rdl2::math::dot(viewDirection, n);
    return scene_rdl2::math::Color(scene_rdl2::math::abs(dot), scene_rdl2::math::abs(dot), scene_rdl2::math::abs(dot));
}
/// Output intersection inverse facing ratio as color
scene_rdl2::math::Color
generateInverseFacingRatioColor(shading::Intersection const &isect,
                                                mcrt_common::RayDifferential const &ray)
{
    scene_rdl2::math::Vec3f viewDirection = - ray.getDirection();
    viewDirection.normalize();
    scene_rdl2::math::Vec3f n = isect.getN();
    float dot = scene_rdl2::math::dot(viewDirection, n);
    return scene_rdl2::math::Color(1.f - scene_rdl2::math::abs(dot), 1.f - scene_rdl2::math::abs(dot), 1.f -
                                   scene_rdl2::math::abs(dot));
}
/// Output intersection UVs as color
scene_rdl2::math::Color
generateUVColor(shading::Intersection const &isect)
{
    scene_rdl2::math::Vec2f st = isect.getSt();
    return scene_rdl2::math::Color(st[0], st[1], 0);
}

}


//----------------------------------------------------------------------------

PathIntegrator::PathIntegrator() :
    mLightSamples(0),
    mBsdfSamplesSqrt(0),
    mBsdfSamples(0),
    mBssrdfSamples(0),
    mMaxDepth(0),
    mMaxDiffuseDepth(0),
    mMaxGlossyDepth(0),
    mMaxMirrorDepth(0),
    mMaxVolumeDepth(0),
    mMaxPresenceDepth(0),
    mMaxHairDepth(0),
    mMaxSubsurfacePerPath(0),
    mTransparencyThreshold(1.f),
    mPresenceThreshold(0.999f),
    mRussianRouletteThreshold(0.f),
    mInvRussianRouletteThreshold(0.f),
    mSampleClampingValue(0.0f),
    mSampleClampingDepth(1),
    mRoughnessClampingFactor(0.0f),
    mInvVolumeQuality(1.0f),
    mInvVolumeShadowQuality(1.0f),
    mVolumeIlluminationSamples(0),
    mVolumeTransmittanceThreshold(0.0f),
    mVolumeAttenuationFactor(1.0f),
    mVolumeContributionFactor(1.0f),
    mVolumePhaseAttenuationFactor(1.0f),
    mVolumeOverlapMode(VolumeOverlapMode::SUM),
    mEnableSSS(true),
    mEnableShadowing(true)
{
}


PathIntegrator::~PathIntegrator()
{
}

void
PathIntegrator::update(const FrameState &fs, const PathIntegratorParams& params)
{
    mLightSamples = params.mIntegratorLightSamplesSqrt * params.mIntegratorLightSamplesSqrt;
    mBsdfSamplesSqrt = params.mIntegratorBsdfSamplesSqrt;
    mBsdfSamples = mBsdfSamplesSqrt * mBsdfSamplesSqrt;
    // Currently we are using triplanar projection sampling for subsurface scattering.
    // This requires at least 2 samples.
    mBssrdfSamples = scene_rdl2::math::max(2, params.mIntegratorBssrdfSamplesSqrt * params.mIntegratorBssrdfSamplesSqrt);
    mMaxDepth = params.mIntegratorMaxDepth;
    mMaxDiffuseDepth = params.mIntegratorMaxDiffuseDepth;
    mMaxGlossyDepth = params.mIntegratorMaxGlossyDepth;
    mMaxMirrorDepth = params.mIntegratorMaxMirrorDepth;
    mMaxVolumeDepth = params.mIntegratorMaxVolumeDepth;
    mMaxPresenceDepth = params.mIntegratorMaxPresenceDepth;
    mMaxHairDepth = params.mIntegratorMaxHairDepth;
    mMaxSubsurfacePerPath = params.mIntegratorMaxSubsurfacePerPath;
    mTransparencyThreshold = params.mIntegratorTransparencyThreshold;
    mPresenceThreshold = params.mIntegratorPresenceThreshold;
    mRussianRouletteThreshold = params.mIntegratorRussianRouletteThreshold;
    mInvRussianRouletteThreshold = 1.f / mRussianRouletteThreshold;
    mSampleClampingValue = params.mSampleClampingValue;
    // volume related params
    mInvVolumeQuality = 1.0f / scene_rdl2::math::max(1e-5f, params.mIntegratorVolumeQuality);
    mInvVolumeShadowQuality = 1.0f / scene_rdl2::math::max(1e-5f, params.mIntegratorVolumeShadowQuality);
    mVolumeIlluminationSamples = params.mIntegratorVolumeIlluminationSamples;
    mVolumeTransmittanceThreshold =
        params.mIntegratorVolumeTransmittanceThreshold;
    mVolumeAttenuationFactor = params.mIntegratorVolumeAttenuationFactor;
    mVolumeContributionFactor = params.mIntegratorVolumeContributionFactor;
    mVolumePhaseAttenuationFactor =
        params.mIntegratorVolumePhaseAttenuationFactor;
    mVolumeOverlapMode = params.mIntegratorVolumeOverlapMode;

    mSampleClampingDepth = params.mSampleClampingDepth;
    mRoughnessClampingFactor = params.mRoughnessClampingFactor;

    const Scene *scene = MNRY_VERIFY(fs.mScene);
    const scene_rdl2::rdl2::SceneContext *sc = scene->getRdlSceneContext();
    const scene_rdl2::rdl2::SceneVariables &vars = sc->getSceneVariables();

    mResolution           = vars.get(scene_rdl2::rdl2::SceneVariables::sResKey);
    mEnableSSS            = vars.get(scene_rdl2::rdl2::SceneVariables::sEnableSSS);
    mEnableShadowing      = vars.get(scene_rdl2::rdl2::SceneVariables::sEnableShadowing);

    mDeepMaxLayers = vars.get(scene_rdl2::rdl2::SceneVariables::sDeepMaxLayers);
    mDeepLayerBias = vars.get(scene_rdl2::rdl2::SceneVariables::sDeepLayerBias);

    // Create attr keys for the deep IDs so we can retrieve the IDs from the
    //  primitive attributes during integration.
    mDeepIDAttrIdxs.clear();
    scene_rdl2::rdl2::StringVector attrNames = vars.get(scene_rdl2::rdl2::SceneVariables::sDeepIDAttributeNames);
    for (size_t i = 0; i < attrNames.size(); i++) {
        shading::TypedAttributeKey<float> deepIDAttrKey(attrNames[i]);
        mDeepIDAttrIdxs.push_back(deepIDAttrKey.getIndex());
    }

    scene_rdl2::rdl2::String cryptoUVAttributeName = vars.get(scene_rdl2::rdl2::SceneVariables::sCryptoUVAttributeName);
    if (!cryptoUVAttributeName.empty()) {
        shading::TypedAttributeKey<scene_rdl2::rdl2::Vec2f> cryptoUVAttrKey(cryptoUVAttributeName);
        mCryptoUVAttrIdx = cryptoUVAttrKey.getIndex();
    } else {
        mCryptoUVAttrIdx = shading::StandardAttributes::sSurfaceST;
    }

    // initialize path guiding
    mPathGuide.startFrame(fs.mEmbreeAccel->getBounds(), vars);
}

void
PathIntegrator::passReset()
{
    if (getEnablePathGuide()) {
        mPathGuide.passReset();
    }
}

//-----------------------------------------------------------------------------

// TODO: MOONRAY-3174
// The introduction of the scalar shading API caused a significant regression
// in shading performance (ie. the "[Total shading time]" category.  This was
// largely the result of moving lots of code out of inlined header functions
// and into .cc files to hide non-public types.  One such case is the
// shading::shade() function which is called here.
static finline void
shadeMaterial(mcrt_common::ThreadLocalState *tls, const scene_rdl2::rdl2::Material *mat,
              const shading::Intersection &intersection, shading::Bsdf *bsdf)
{
    EXCL_ACCUMULATOR_PROFILE(tls, EXCL_ACCUM_SHADING);

    // TODO: don't call this due to performance concerns ( MOONRAY-3174 )
    // ----------------------------------------------------------------
    // shading::shade(mat, tls->mShadingTls.get(), shading::State(&intersection), bsdf);
    // ----------------------------------------------------------------
    // ...instead copy/paste shading::shade() contents inline, starting here:



#define SHADING_BRACKET_TIMING_ENABLED
#ifdef SHADING_BRACKET_TIMING_ENABLED
    auto threadIndex = tls->mThreadIdx;
    time::RAIITicker<util::InclusiveExclusiveAverage<int64> > ticker(
            MNRY_VERIFY(mat->getThreadLocalObjectState())[threadIndex].mShaderCallStat);
#endif

    const shading::State state(&intersection);
    shading::BsdfBuilder bsdfBuilder(*bsdf, tls->mShadingTls.get(), state);
    mat->shade(tls->mShadingTls.get(), state, bsdfBuilder);

    // Evaluate and store the post scatter extra aovs on the bsdf object.
    // They will be accumulated after ray scattering.
    MNRY_ASSERT(mat->hasExtension());
    if (mat->hasExtension()) {
        EXCL_ACCUMULATOR_PROFILE(tls, EXCL_ACCUM_AOVS);
        const auto &ext = mat->get<shading::Material>();
        const std::vector<shading::Material::ExtraAov> &extraAovs = ext.getPostScatterExtraAovs();
        if (!extraAovs.empty()) {
            shading::TLState *shadingTls = tls->mShadingTls.get();
            scene_rdl2::math::Color *colors = shadingTls->mArena->allocArray<scene_rdl2::math::Color>(extraAovs.size(),
                                                                                                      CACHE_LINE_SIZE);
            int *labelIds = shadingTls->mArena->allocArray<int>(extraAovs.size(), CACHE_LINE_SIZE);
            for (size_t i = 0; i < extraAovs.size(); ++i) {
                const shading::Material::ExtraAov &ea = extraAovs[i];
                labelIds[i] = ea.mLabelId;
                shading::sample(mat, ea.mMap, shadingTls, state, colors + i);
            }
            bsdf->setPostScatterExtraAovs(extraAovs.size(), labelIds, colors);
        }
    }

    // force setBsdfLabels inline ( JIRA )
    // setBsdfLabels(*mat, state, bsdf);

    MNRY_ASSERT(mat->hasExtension());
    if (mat->hasExtension()) {
        const auto &ext = mat->get<shading::Material>();
        const scene_rdl2::rdl2::Geometry *geometry = state.getGeometryObject();
        int geomLabelId = -1;
        MNRY_ASSERT(geometry);
        MNRY_ASSERT(geometry->hasExtension());
        if (geometry && geometry->hasExtension()) {
            geomLabelId = geometry->get<shading::Geometry>().getGeomLabelId();
        }
        bsdf->setLabelIds(ext.getMaterialLabelId(), ext.getLpeMaterialLabelId(),
                          geomLabelId);
    }

    // force xformLabels inline ( JIRA )
    // xformLobeLabels(mat, bsdf);

    MNRY_ASSERT(mat->hasExtension());
    if (mat->hasExtension()) {
        const auto &ext = mat->get<shading::Material>();
        const int  materialLabelId    = ext.getMaterialLabelId();    // material aovs
        const auto &lobeLabelIds      = ext.getLobeLabelIds();       // material aovs
        const int  lpeMaterialLabelId = ext.getLpeMaterialLabelId(); // lpe aovs
        const auto &lpeLobeLabelIds   = ext.getLpeLobeLabelIds();    // lpe aovs

        for (int i = 0; i < bsdf->getLobeCount(); ++i) {
            shading::BsdfLobe *lobe = bsdf->getLobe(i);
            lobe->setLabel(shading::aovEncodeLabels(lobe->getLabel(),
                                           materialLabelId, lpeMaterialLabelId,
                                           lobeLabelIds, lpeLobeLabelIds));
        }

        shading::Bssrdf *bssrdf = bsdf->getBssrdf();
        if (bssrdf) {
            bssrdf->setLabel(shading::aovEncodeLabels(bssrdf->getLabel(),
                                             materialLabelId, lpeMaterialLabelId,
                                             lobeLabelIds, lpeLobeLabelIds));
        }

        shading::VolumeSubsurface *vs = bsdf->getVolumeSubsurface();
        if (vs) {
            vs->setLabel(shading::aovEncodeLabels(vs->getLabel(),
                                         materialLabelId, lpeMaterialLabelId,
                                         lobeLabelIds, lpeLobeLabelIds));
        }
    }

#ifdef SHADING_PRINT_DEBUG_BSDF_INFO_ENABLED
    bsdf->show(mat->getSceneClass().getName(), mat->getName(), std::cout);
#endif
}

// BsdfLobe is passed in so that we can track the lobe type which generated
// the ray for ray debugging purposes. Other than that, it's not needed.
PathIntegrator::IndirectRadianceType
PathIntegrator::computeRadianceRecurse(pbr::TLState *pbrTls, mcrt_common::RayDifferential &ray,
        const Subpixel &sp, const PathVertex &prevPv, const shading::BsdfLobe *lobe,
        scene_rdl2::math::Color &radiance, float &transparency, VolumeTransmittance& vt,
        unsigned &sequenceID, float *aovs, float *depth,
        DeepParams* deepParams, CryptomatteParams *cryptomatteParamsPtr,
        CryptomatteParams *refractCryptomatteParamsPtr,
        bool ignoreVolumes, bool &hitVolume) const
{
    CHECK_CANCELLATION(pbrTls, return NONE);

    // Turn off profiling integrator profiling for the first part of this
    // function. It's turned back on later.
    MNRY_ASSERT(pbrTls->isIntegratorAccumulatorRunning());

    const FrameState &fs = *pbrTls->mFs;
    const Scene *scene = MNRY_VERIFY(pbrTls->mFs->mScene);

    Statistics &stats = pbrTls->mStatistics;

    const AovSchema &aovSchema = *fs.mAovSchema;

    scene_rdl2::alloc::Arena *arena = pbrTls->mArena;
    SCOPED_MEM(arena);

    radiance = scene_rdl2::math::sBlack;
    transparency = 0.0f;

    vt.reset();

    scene_rdl2::math::Color ssAov = scene_rdl2::math::sBlack;
    IndirectRadianceType indirectRadianceType = NONE;
    //---------------------------------------------------------------------
    // Trace continuation ray

    // Find next vertex of the path. The intersectRay call doesn't intersect
    // with any lights, only geometry.
    // TODO: scale differentials if we want to use for geometry LOD
    shading::Intersection isect;

    int lobeType = (lobe == nullptr) ? 0 : lobe->getType();
    bool hitGeom = scene->intersectRay(pbrTls->mTopLevelTls, ray, isect, lobeType);
    if (hitGeom && deepParams) {
        deepParams->mHitDeep = true;
        for (size_t i = 0; i < mDeepIDAttrIdxs.size(); i++) {
            shading::TypedAttributeKey<float> deepIDAttrKey(mDeepIDAttrIdxs[i]);
            if (isect.isProvided(deepIDAttrKey)) {
                deepParams->mDeepIDs[i] = isect.getAttribute<float>(deepIDAttrKey);
            } else {
                deepParams->mDeepIDs[i] = 0.f;
            }
        }
    }

    // Prevent aliasing in the visibility aov by accounting for 
    // primary rays that don't hit anything
    if (ray.getDepth() == 0 && !hitGeom) {
        
        // If we're on the edge of the geometry, some rays should count as "hits", some as "misses". Here, 
        // we're adding light_sample_count * lights number of "misses" to the visibility aov to account for 
        // the light samples that couldn't be taken because the primary ray doesn't hit anything. 
        // This improves aliasing on the edges.
        if (aovs) {
            const LightAovs &lightAovs = *fs.mLightAovs;
            const AovSchema &aovSchema = *fs.mAovSchema;
            // predict the number of light samples that would have been taken if the ray hit geom
            int totalLightSamples = mLightSamples * scene->getLightCount();

            // Doesn't matter what the lpe is -- if there are subpixels that hit a surface that isn't included
            // in the lpe, this would be black anyway. If there are subpixels that DO hit a surface that is
            // included in the lpe, this addition prevents aliasing. 
            aovAccumVisibilityAttempts(pbrTls, aovSchema, lightAovs, totalLightSamples, aovs);
        }
    }

    // Set the cryptomatte information
    float cryptoId = 0.f;
    scene_rdl2::math::Vec2f cryptoUV = isect.getSt();
    if (hitGeom && (cryptomatteParamsPtr || refractCryptomatteParamsPtr)) {
        if (mDeepIDAttrIdxs.size() != 0) {
            // Get the cryptomatte ID if we need it and it's specified
            shading::TypedAttributeKey<float> deepIDAttrKey(mDeepIDAttrIdxs[0]);
            if (isect.isProvided(deepIDAttrKey)) {
                cryptoId = isect.getAttribute<float>(deepIDAttrKey);
            }
        }
        shading::TypedAttributeKey<scene_rdl2::rdl2::Vec2f> cryptoUVAttrKey(mCryptoUVAttrIdx);
        if (isect.isProvided(cryptoUVAttrKey)) {
            cryptoUV = isect.getAttribute<scene_rdl2::rdl2::Vec2f>(cryptoUVAttrKey);
        }
    }
    if (hitGeom && cryptomatteParamsPtr) {
        cryptomatteParamsPtr->mHit = true;
        cryptomatteParamsPtr->mPosition = isect.getP();
        cryptomatteParamsPtr->mNormal = isect.getN();
        shading::State sstate(&isect);
        sstate.getRefP(cryptomatteParamsPtr->mRefP);
        sstate.getRefN(cryptomatteParamsPtr->mRefN);
        cryptomatteParamsPtr->mUV = cryptoUV;
        cryptomatteParamsPtr->mId = cryptoId;
    }

    PathVertex pv(prevPv);

    //--------------------------------------------------------------------
    // Volumes
    hitVolume = false;

    // computeRadianceVolume() increases the pv.volumeDepth.  We want the presence
    // continuation ray's volume depth to be unchanged, so we restore it.
    // Not doing this can cause problems with presence objects inside volumes
    // if the max volume depth is low (default = 1).  They will appear to 'hold out'
    // the volume behind them as the max volume depth is reached.
    int currentVolumeDepth = pv.volumeDepth;
    float volumeSurfaceT = scene_rdl2::math::sMaxValue;

    if (!ignoreVolumes) {
        hitVolume = computeRadianceVolume(pbrTls, ray, sp, pv, lobeType,
            radiance, sequenceID, vt, aovs, deepParams, nullptr, &volumeSurfaceT);
        if (hitVolume) {
            indirectRadianceType = IndirectRadianceType(indirectRadianceType | VOLUME);
        }
        pv.pathThroughput *= vt.transmittance();
    }

    if (!hitGeom && aovs) {
        // accumuate background aovs
        aovAccumBackgroundExtraAovs(pbrTls, fs, pv, aovs);
    }

    //---------------------------------------------------------------------
    // Code for rendering lights, only executed for primary rays since lights
    // appear in deeper passes already.
    if ((ray.getDepth() == 0)) {

        LightIntersection hitLightIsect;
        int numHits = 0;
        const Light *hitLight;
        SequenceIDIntegrator sid(0, sp.mPixel, sp.mSubpixelIndex,
            SequenceType::IndexSelection, sequenceID);
        IntegratorSample1D lightChoiceSamples(sid);
        hitLight = scene->intersectVisibleLight(ray,
            hitGeom ? ray.getEnd() : sInfiniteLightDistance,
            lightChoiceSamples, hitLightIsect, numHits);

        if (hitLight) {
            // Evaluate the radiance on the selected light in camera.
            // Note: we multiply the radiance contribution by the number of
            // lights hit. This is because we want to compute the sum of all
            // contributing lights, but we're stochastically sampling just one.
            LightFilterRandomValues lightFilterR = {
                scene_rdl2::math::Vec2f(0.f, 0.f), 
                scene_rdl2::math::Vec3f(0.f, 0.f, 0.f)}; // light filters don't apply to camera rays
            scene_rdl2::math::Color lightContribution = pv.pathThroughput * numHits *
                hitLight->eval(pbrTls->mTopLevelTls,
                    ray.getDirection(), ray.getOrigin(), lightFilterR,
                    ray.getTime(), hitLightIsect, true, nullptr, ray.getDirFootprint());
            radiance += lightContribution;
            // If we hit a light that's opaque, set the transparency to the
            // minimum volume transmittance: if the volume is not a cutout,
            // volumeTransmittanceM == 0 and it is fully opaque.  If the volume is
            // a cutout, volumeTransmittanceMin > 0 and we are not fully opaque.
            // If we hit a light that's not opaque, then we just use the volume
            // alpha transmittance.
            transparency = (hitLight->getIsOpaqueInAlpha() ?
                reduceTransparency(vt.mTransmittanceMin) :
                reduceTransparency(vt.mTransmittanceAlpha));

            checkForNan(radiance, "Camera visible lights", sp, pv, ray, isect);

            // LPE
            if (aovs) {
                EXCL_ACCUMULATOR_PROFILE(pbrTls, EXCL_ACCUM_AOVS);
                const LightAovs &lightAovs = *fs.mLightAovs;
                // transition
                int lpeStateId = pv.lpeStateId;
                lpeStateId = lightAovs.lightEventTransition(pbrTls, lpeStateId, hitLight);
                // accumulate matching aovs
                aovAccumLightAovs(pbrTls, *fs.mAovSchema, *fs.mLightAovs,
                    lightContribution, nullptr, AovSchema::sLpePrefixNone, lpeStateId, aovs);
            }

            return indirectRadianceType;
        }
    }

    if (!hitGeom) {
        transparency = reduceTransparency(vt.mTransmittanceAlpha);

        // Did we hit a volume and do we have volume depth/position AOVs?
        if (ray.getDepth() == 0 && hitVolume && volumeSurfaceT < scene_rdl2::math::sMaxValue) {
            aovSetStateVarsVolumeOnly(pbrTls, aovSchema, volumeSurfaceT, ray,
                                      *scene, pv.pathPixelWeight, aovs);
        }

        return indirectRadianceType;
    }
    indirectRadianceType = IndirectRadianceType(indirectRadianceType | SURFACE);

    CHECK_CANCELLATION(pbrTls, return NONE);

    // Early return with error color if isect doesn't provide all the
    // required attributes shader request
    if (!isect.hasAllRequiredAttributes()) {
        radiance += pv.pathThroughput * fs.mFatalColor;
        return indirectRadianceType;
    }

    // Finalize Intersection setup.
    geom::initIntersectionPhase2(isect,
                                 pbrTls->mTopLevelTls,
                                 pv.mirrorDepth,
                                 pv.glossyDepth,
                                 pv.diffuseDepth,
                                 isSubsurfaceAllowed(pv.subsurfaceDepth),
                                 pv.minRoughness,
                                 -ray.getDirection());

    //---------------------------------------------------------------------
    // Run the material shader at the ray intersection point to get the Bsdf

    // Transfer the ray to its intersection before we run shaders. This is
    // needed for texture filtering based on ray differentials.
    // Also scale the final differentials by a user factor. This is left until
    // the very end and not baked into the ray differentials since the factor
    // will typically be > 1, and would cause the ray differentials to be larger
    // than necessary. The mip selector is computed in this call also.
    isect.transferAndComputeDerivatives(pbrTls->mTopLevelTls, &ray,
            sp.mTextureDiffScale);

    float rayEpsilon = isect.getEpsilonHint();
    if (rayEpsilon <= 0.0f) {
        // Compute automatic ray-tracing bias
        float pathDistance = pv.pathDistance + ray.getEnd();
        rayEpsilon = sHitEpsilonStart * scene_rdl2::math::max(pathDistance, 1.0f);
    }
    float shadowRayEpsilon = isect.getShadowEpsilonHint();

    // Must be called post-transfer.
    RAYDB_EXTEND_RAY(pbrTls, ray, isect);
    RAYDB_SET_TAGS(pbrTls, lobe ? lobe->getType() : TAG_PRIMARY);

    const scene_rdl2::rdl2::Material* material = isect.getMaterial()->asA<scene_rdl2::rdl2::Material>();
    MNRY_ASSERT(material != NULL);

    // perform material substitution if needed
    scene_rdl2::rdl2::RaySwitchContext switchCtx;
    switchCtx.mRayType = lobeTypeToRayType(pv.lobeType);
    material = material->raySwitch(switchCtx);

    shading::TLState* shadingTls = pbrTls->mTopLevelTls->mShadingTls.get();

    // Presence handling code for regular rays
    // Need to continue the current ray through the partially present geometry.
    float presence = shading::presence(material, shadingTls, shading::State(&isect));

    // Some NPR materials that want to allow for completely arbitrary shading normals
    // can request that the integrator does not perform any light culling based on the
    // normal. In those cases, we also want to prevent our call to adaptNormal() in the
    // Intersection when the material evaluates its normal map bindings.
    if (shading::preventLightCulling(material, shading::State(&isect))) {
       isect.setUseAdaptNormal(false);
    }

    // Nested dielectric handling.  Uses presence code for skipping false intersections.
    // See "Simple Nested Dielectrics in Ray Traced Images". 
    // This also enables automatic removal of self-overlapping geometry that's assigned to the
    // same material.
    int materialPriority = material->priority();
    const scene_rdl2::rdl2::Camera* camera = scene->getCamera()->getRdlCamera();

    const scene_rdl2::rdl2::Material* newPriorityList[4];
    int newPriorityListCount[4]; 
    float mediumIor = updateMaterialPriorities(ray, scene, camera, shadingTls, isect, material, &presence,
                                               materialPriority, newPriorityList, newPriorityListCount, 
                                               pv.presenceDepth);
    isect.setMediumIor(mediumIor);

    // If we terminate early, we do not want the contribution of the presence
    // value in the pathThroughput or the pathPixelWeight.
    scene_rdl2::math::Color earlyTerminatorPathThroughput = pv.pathThroughput;
    float earlyTerminatorPathPixelWeight = pv.pathPixelWeight;

    scene_rdl2::math::Color presenceRadiance = scene_rdl2::math::sBlack;
    float presenceTransparency = 0.f;
    if (presence < 1.f - scene_rdl2::math::sEpsilon) {

        // We will record cryptomatte information in this if block if necessary already.
        // Don't double count by stating that the hit was false:
        if (cryptomatteParamsPtr) {
            cryptomatteParamsPtr->mHit = false;
        }
        float totalPresence = (1.0f - prevPv.totalPresence) * presence;

        float rayNear = rayEpsilon;
        float rayFar = ray.getOrigTfar() - ray.tfar;
        if (totalPresence >= mPresenceThreshold || prevPv.presenceDepth >= mMaxPresenceDepth) {
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
            rayNear = scene_rdl2::math::sMaxValue * 0.5f;
            rayFar = scene_rdl2::math::sMaxValue;
        }

        // The origin and tfar has been moved to the geometry intersection
        // point so the new tnear is just rayEpsilon.  We also need to shorten the tfar
        // appropriately so we don't overshoot the original ray length.
        mcrt_common::RayDifferential presenceRay(ray, rayNear, rayFar);
        setPriorityList(presenceRay, newPriorityList, newPriorityListCount);

        // The above constructor increments the ray depth.  We want to keep the parent ray's
        //  depth as the presence ray is a continuation of the same parent ray.  Also, if the
        //  ray depth is incremented, lights are not visible through presence as the depth for
        //  continued camera rays is no longer 0.
        presenceRay.setDepth(presenceRay.getDepth() - 1);

        PathVertex newPv = pv;                  // new path vertex for continued ray
        newPv.pathDistance += ray.getEnd();
        newPv.pathPixelWeight *= (1-presence);  // weight of continued ray
        newPv.aovPathPixelWeight *= (1-presence); // weight of continued ray for aov use
        newPv.pathThroughput *= (1-presence);
        newPv.presenceDepth++;
        newPv.totalPresence = totalPresence;
        newPv.volumeDepth = currentVolumeDepth;

        // LPE
        // presence is a straight event
        if (aovs) {
            EXCL_ACCUMULATOR_PROFILE(pbrTls, EXCL_ACCUM_AOVS);
            const FrameState &fs = *pbrTls->mFs;
            const LightAovs &lightAovs = *fs.mLightAovs;
            // transition
            newPv.lpeStateId = lightAovs.straightEventTransition(pbrTls, pv.lpeStateId);
        }

        // We need to weight the continued ray and the regular shading appropriately.
        // Scaling the throughput and pixel weight for the path handles this properly.
        // Note that we don't need to scale the bsdf lobes of the material if we do this.
        pv.pathPixelWeight *= presence;  // weight of regular shading
        pv.aovPathPixelWeight *= presence; // weight of regular shading for aov use
        pv.pathThroughput *= presence;

        // Construct new cryptomatte parameters to handle stacked presence objects.
        CryptomatteParams newCryptomatteParams;
        if (cryptomatteParamsPtr) newCryptomatteParams.init(cryptomatteParamsPtr->mCryptomatteBuffer);
        CryptomatteParams *newCryptomatteParamsPtr = cryptomatteParamsPtr ? &newCryptomatteParams : nullptr;

        // Fire continued ray and add in its radiance
        VolumeTransmittance vtPresence;
        unsigned presenceSequenceID = sequenceID;
        bool presenceHitVolume;
        computeRadianceRecurse(pbrTls, presenceRay, sp, newPv, lobe,
            presenceRadiance, presenceTransparency, vtPresence,
            presenceSequenceID, aovs, nullptr, nullptr, newCryptomatteParamsPtr, nullptr, 
            false, presenceHitVolume);

        radiance += presenceRadiance;
        vt.mTransmittanceE *= vtPresence.mTransmittanceE;
        transparency = (1 - presence) * presenceTransparency;

        // We need to handle the case where we hit something with a presence of 1.0f (or without a presence) 
        // as we continue. To handle this case, we just check if the new cryptomatte params have the hit member set.
        if (newCryptomatteParamsPtr && newCryptomatteParamsPtr->mHit) {
            unsigned px, py;
            uint32ToPixelLocation(sp.mPixel, &px, &py);
            newCryptomatteParamsPtr->mCryptomatteBuffer->addSampleScalar(px, py, newCryptomatteParamsPtr->mId, 
                                                                                 newPv.pathPixelWeight,
                                                                                 newCryptomatteParamsPtr->mPosition,
                                                                                 newCryptomatteParamsPtr->mNormal,
                                                                                 newCryptomatteParamsPtr->mBeauty,
                                                                                 newCryptomatteParamsPtr->mRefP,
                                                                                 newCryptomatteParamsPtr->mRefN,
                                                                                 newCryptomatteParamsPtr->mUV,
                                                                                 newPv.presenceDepth,
                                                                                 moonray::pbr::CRYPTOMATTE_TYPE_REGULAR);
        }
    }

    // refractive cryptomatte PART A
    if (hitGeom && refractCryptomatteParamsPtr) {
        // refractCryptomatteParamsPtr is only non-null if we are on a primary ray path
        // that is passing through refractive materials, or is a camera ray.
        // See the code in PathIntegrator::addIndirectOrDirectVisibleContributions() that
        // sets this pointer if we have passed through a refractive material
        // (refractive cryptomatte PART B)
        if (!material->invisibleRefractiveCryptomatte()) {
            // If we have now hit a material that is NOT invisible in refractive cryptomatte,
            //   we end the refractive cryptomatte path by setting the intersection data.
            refractCryptomatteParamsPtr->mHit = true;
            refractCryptomatteParamsPtr->mPosition = isect.getP();
            refractCryptomatteParamsPtr->mNormal = isect.getN();
            shading::State sstate(&isect);
            sstate.getRefP(refractCryptomatteParamsPtr->mRefP);
            sstate.getRefN(refractCryptomatteParamsPtr->mRefN);
            refractCryptomatteParamsPtr->mUV = cryptoUV;
            refractCryptomatteParamsPtr->mId = cryptoId;
        }
    }

    auto bsdf = arena->allocWithCtor<shading::Bsdf>();
    shadeMaterial(pbrTls->mTopLevelTls, material, isect, bsdf);
    stats.incCounter(STATS_SHADER_EVALS);

    // Evaluate any extra aovs on this material
    if (aovs) {
        aovAccumExtraAovs(pbrTls, fs, pv, isect,  material, aovs);
    }

    CHECK_CANCELLATION(pbrTls, return NONE);

    //---------------------------------------------------------------------
    // Termination (did the shader request termination of tracing?)
    if (bsdf->getEarlyTermination()) {
        if (ray.getDepth() == 0) {
            // We override the previous transparency value when we encounter a cutout.
            // If there is presence, we need to combine the transparency encountered
            // during presence continuation/traversal above with the current presence
            // value.
            // This is a bit confusing because the presence is for the *cutout*.
            // e.g. if the cutout has a presence of 1, the transparency value should
            // be forced to 1 because the cutout is 100% there and completely cuts out.
            // If the cutout has a presence of 0, the cutout is 0% there and thus has no
            // effect on the transparency.
            // Also, the alpha in the render output is 1 - transparency.
            // We also multiply by the volume's transmittanceAlpha, but that's independent
            // of presence.
            transparency = reduceTransparency(vt.mTransmittanceAlpha) *
                           (presenceTransparency + (1 - presenceTransparency) * presence);
        } else {
            transparency = reduceTransparency(earlyTerminatorPathThroughput);
        }

        // if this is a primary ray, fill out the aovs
        if (aovs) {
            aovSetMaterialAovs(pbrTls, aovSchema, *fs.mLightAovs, *fs.mMaterialAovs,
                               isect, ray, *scene, *bsdf, ssAov,
                               nullptr, nullptr, earlyTerminatorPathPixelWeight, pv.lpeStateId, aovs);
        }
        if (aovs && ray.getDepth() == 0) {
            aovSetStateVars(pbrTls, aovSchema, isect, volumeSurfaceT, ray, *scene, earlyTerminatorPathPixelWeight, aovs);
            aovSetPrimAttrs(pbrTls, aovSchema, material->get<shading::Material>().getAovFlags(),
                            isect, earlyTerminatorPathPixelWeight, aovs);
        }

        if (depth && ray.getDepth() == 0) {
            *depth = scene->getCamera()->computeZDistance(isect.getP(), ray.getOrigin(), ray.getTime());
        }

        if (deepParams) {
            // If we have terminated, don't output anything to the deep buffer
            deepParams->mHitDeep = false;
        }

        if (cryptomatteParamsPtr) {
            // If we have terminated, don't output anything to the cryptomatte buffer
            cryptomatteParamsPtr->mHit = false;
        }

        if (refractCryptomatteParamsPtr) {
            // If we have terminated, don't output anything to the refract cryptomatte buffer
            refractCryptomatteParamsPtr->mHit = false;
        }

        return indirectRadianceType;
    }

    if (scene_rdl2::math::isEqual(presence, 0.f)) {
        // Only the presence continuation ray contributes to the radiance so we can early out here.
        // We must process cutouts (early termination) before this or the cutout alpha will be incorrect.
        return indirectRadianceType;
    }

    // if this is a primary ray, fill out the intersection and primitive attribute aovs
    if (aovs && ray.getDepth() == 0) {
        aovSetStateVars(pbrTls, aovSchema, isect, volumeSurfaceT, ray, *scene, pv.pathPixelWeight, aovs);
        aovSetPrimAttrs(pbrTls, aovSchema, material->get<shading::Material>().getAovFlags(),
                         isect, pv.pathPixelWeight, aovs);
    }

    if (depth && ray.getDepth() == 0) {
        *depth = scene->getCamera()->computeZDistance(isect.getP(), ray.getOrigin(), ray.getTime());
    }

    //---------------------------------------------------------------------
    // Self-emission
    scene_rdl2::math::Color selfEmission = pv.pathThroughput * bsdf->getSelfEmission();
    radiance += selfEmission;
    // LPE
    if (aovs) {
        if (!isBlack(selfEmission)) {
            EXCL_ACCUMULATOR_PROFILE(pbrTls, EXCL_ACCUM_AOVS);
            const LightAovs &lightAovs = *fs.mLightAovs;

            // transition
            int lpeStateId = pv.lpeStateId;
            lpeStateId = lightAovs.emissionEventTransition(pbrTls, lpeStateId, *bsdf);

            // accumulate matching aovs
            aovAccumLightAovs(pbrTls, *fs.mAovSchema, *fs.mLightAovs, selfEmission, 
                              nullptr, AovSchema::sLpePrefixNone, lpeStateId, aovs);
        }
    }

    //---------------------------------------------------------------------
    // Early out if we don't have any Bsdf lobes nor Bssrdf, VolumeSubsurface
    if (bsdf->getLobeCount() == 0 && !bsdf->hasSubsurface()) {
        RAYDB_SET_CONTRIBUTION(pbrTls, radiance / pv.pathThroughput);

        if (aovs) {
            aovSetMaterialAovs(pbrTls, aovSchema, *fs.mLightAovs, *fs.mMaterialAovs,
                               isect, ray, *scene, *bsdf, ssAov,
                               nullptr, nullptr, pv.aovPathPixelWeight, pv.lpeStateId, aovs);
        }

        return indirectRadianceType;
    }

    // Starting the integrator accumulator here is roughly equivalent of what
    // the bundled code does.

    //---------------------------------------------------------------------
    // Have we reached the maximum number of bounces for each lobe types / overall
    // Note: hair lobes are also glossy lobes. So the max depth for hair lobes
    // would be max(mMaxGlossyDepth, mMaxHairDepth)
    shading::BsdfLobe::Type indirectFlags = shading::BsdfLobe::NONE;
    bool doIndirect = (mBsdfSamples > 0 && ray.getDepth() < mMaxDepth);
    if (doIndirect) {
        setFlag(indirectFlags, (pv.diffuseDepth < mMaxDiffuseDepth  ?
                shading::BsdfLobe::DIFFUSE  :  shading::BsdfLobe::NONE));
        setFlag(indirectFlags, (pv.glossyDepth < mMaxGlossyDepth    ?
                shading::BsdfLobe::GLOSSY   :  shading::BsdfLobe::NONE));
        setFlag(indirectFlags, (pv.mirrorDepth < mMaxMirrorDepth    ?
                shading::BsdfLobe::MIRROR   :  shading::BsdfLobe::NONE));
        doIndirect = (indirectFlags != shading::BsdfLobe::NONE) ||
                     (pv.hairDepth < mMaxHairDepth);
        // If doIndirect is true due to hairDepth only, then only side type bits
        // are set in indirectFlags.
        setFlag(indirectFlags, (doIndirect  ?
                shading::BsdfLobe::ALL_SURFACE_SIDES  :  shading::BsdfLobe::NONE));
    }

    CHECK_CANCELLATION(pbrTls, return NONE);

    //---------------------------------------------------------------------
    // For bssrdf/VolumeSubsurface or bsdfs which contain both
    // reflection and transmission lobes or is spherical,
    // a single normal can't be used for culling so skip normal culling.
    scene_rdl2::math::Vec3f normal(scene_rdl2::math::zero);
    scene_rdl2::math::Vec3f *normalPtr = nullptr;
    if (!bsdf->hasSubsurface() && !bsdf->getIsSpherical() &&
        ((bsdf->getType() & shading::BsdfLobe::ALL_SURFACE_SIDES) != shading::BsdfLobe::ALL_SURFACE_SIDES)) {
        normal = (bsdf->getType() & shading::BsdfLobe::REFLECTION) ? isect.getNg() : -isect.getNg();
        normalPtr = &normal;
    }

    // Gather up all lights which can affect the intersection point/normal.
    LightSet activeLightSet;
    bool hasRayTerminatorLights;
    computeActiveLights(arena, scene, isect, normalPtr, *bsdf, ray.getTime(),
        activeLightSet, hasRayTerminatorLights);

    // Setup a slice, which handles selecting the lobe types and setup
    // evaluations to include the cosine term.
    // Note: Even though we may not be doing indirect for certain lobe types
    // (according to indirectFlags), we still want to draw samples according
    // to all lobes for direct lighting MIS.
    shading::BsdfSlice slice(isect.getNg(), -ray.getDirection(), true,
                    isect.isEntering(), fs.mShadowTerminatorFix, shading::BsdfLobe::ALL);

    //---------------------------------------------------------------------
    // Estimate subsurface scattering
    // Option 1: diffusion profile (bssrdf) approach
    const shading::Bssrdf *bssrdf = bsdf->getBssrdf();
    if (bssrdf != nullptr) {
        // increment subsurface depth
        pv.subsurfaceDepth += 1;
        // Stop the accumulator here since the subsurface accumulator will be
        // started up inside of computeRadianceSubsurface as needed.
        radiance += computeRadianceDiffusionSubsurface(pbrTls, *bsdf, sp, pv, ray,
            isect, slice, *bssrdf, activeLightSet, doIndirect, rayEpsilon, shadowRayEpsilon,
            sequenceID, ssAov, aovs);
    }
    // Option 2: path trace volumetric approach
    const shading::VolumeSubsurface *volumeSubsurface = bsdf->getVolumeSubsurface();
    if (volumeSubsurface != nullptr) {
        // increment subsurface depth
        pv.subsurfaceDepth += 1;
        radiance += computeRadiancePathTraceSubsurface(pbrTls, *bsdf, sp, pv, ray,
            isect, *volumeSubsurface, activeLightSet, doIndirect, rayEpsilon, shadowRayEpsilon,
            sequenceID, ssAov, aovs);
    }

    checkForNan(radiance, "Subsurface scattering", sp, pv, ray, isect);


    //---------------------------------------------------------------------
    // Early out if we don't have any Bsdf lobes
    if (bsdf->getLobeCount() == 0) {
        RAYDB_SET_CONTRIBUTION(pbrTls, radiance / pv.pathThroughput);

        if (aovs) {
            aovSetMaterialAovs(pbrTls, aovSchema, *fs.mLightAovs, *fs.mMaterialAovs,
                               isect, ray, *scene, *bsdf, ssAov,
                               nullptr, nullptr, pv.aovPathPixelWeight, pv.lpeStateId, aovs);
        }

        return indirectRadianceType;
    }

    CHECK_CANCELLATION(pbrTls, return NONE );

    //---------------------------------------------------------------------
    // Estimate emissive volume region energy contribution
    radiance += computeRadianceEmissiveRegionsScalar(pbrTls, sp, pv, ray, isect,
        *bsdf, slice, rayEpsilon, sequenceID, aovs);

    //---------------------------------------------------------------------
    // Setup bsdf and light samples
    radiance += computeRadianceBsdfMultiSampler(pbrTls, sp, pv, ray, isect, *bsdf, slice,
        doIndirect, indirectFlags, newPriorityList, newPriorityListCount, activeLightSet, normalPtr,
        rayEpsilon, shadowRayEpsilon, ssAov, sequenceID, aovs, refractCryptomatteParamsPtr);

    // -------------------------------------------------------------------
    // Add presence fragment to cryptomatte or update beauty for other recursions 
    if (cryptomatteParamsPtr && (presence < 1.f - scene_rdl2::math::sEpsilon)) {
        unsigned px, py;
        uint32ToPixelLocation(sp.mPixel, &px, &py);

    if (pv.pathPixelWeight > 0.01f) {
        // We divide by pathPixelWeight to compute Cryptomatte beauty.  This can cause fireflies if
        // the value is small, so we clamp at 0.01.

        float presenceInv = pv.pathPixelWeight == 0.f ? 0.f : (1.f / pv.pathPixelWeight);
        scene_rdl2::math::Color cryptoBeauty = radiance - presenceRadiance;
        cryptoBeauty *= presenceInv;

        cryptomatteParamsPtr->mCryptomatteBuffer->addSampleScalar(px, py, cryptomatteParamsPtr->mId, 
                                                                          pv.pathPixelWeight, 
                                                                          cryptomatteParamsPtr->mPosition, 
                                                                          cryptomatteParamsPtr->mNormal,
                                                                          scene_rdl2::math::Color4(cryptoBeauty),
                                                                          cryptomatteParamsPtr->mRefP,
                                                                          cryptomatteParamsPtr->mRefN,
                                                                          cryptomatteParamsPtr->mUV,
                                                                          pv.presenceDepth,
                                                                          moonray::pbr::CRYPTOMATTE_TYPE_REGULAR);
        }
    } else if (cryptomatteParamsPtr) {
        // if non-presence path, we still need to record radiance

        if (pv.pathPixelWeight > 0.01f) {
            cryptomatteParamsPtr->mBeauty = scene_rdl2::math::Color4(radiance / pv.pathPixelWeight);
        } else {
            cryptomatteParamsPtr->mBeauty = scene_rdl2::math::Color4(0, 0, 0, 0);
        }
    }

    RAYDB_SET_CONTRIBUTION(pbrTls, radiance / pv.pathThroughput);

    float minTransparency = reduceTransparency(vt.mTransmittanceMin);
    transparency = transparency + (1 - transparency) * minTransparency;

    return indirectRadianceType;
}


bool
PathIntegrator::initPrimaryRay(pbr::TLState *pbrTls,
                               const Camera *camera,
                               int pixelX,
                               int pixelY,
                               int subpixelIndex,
                               int pixelSamples,
                               const Sample& sample,
                               mcrt_common::RayDifferential &ray,
                               Subpixel &sp,
                               PathVertex &pv) const
{
    EXCL_ACCUMULATOR_PROFILE(pbrTls, EXCL_ACCUM_PRIMARY_RAY_GEN);

    MNRY_ASSERT(camera);

    camera->createRay(&ray, pixelX + sample.pixelX, pixelY + sample.pixelY,
                      sample.time, sample.lensU, sample.lensV, true);

     // check that the ray is valid
     if (ray.getStart() == scene_rdl2::math::sMaxValue && ray.getEnd() == scene_rdl2::math::sMaxValue) {
         // ray is invalid
         return false;
     }

    // Create a sub-pixel structure
    sp.mPixel = pixelLocationToUint32(unsigned(pixelX), unsigned(pixelY));
    sp.mSubpixelIndex = subpixelIndex;
    sp.mSubpixelX = sample.pixelX;
    sp.mSubpixelY = sample.pixelY;
    sp.mPixelSamples = pixelSamples;

    // Make sure our clamping is look-invariant with changes in bsdf sample counts
    sp.mSampleClampingValue = mSampleClampingValue / mBsdfSamples;

    // We cap the scaling factor of ray-differentials based on pixel samples,
    // so that we band-limit the texture filtering (over the resulting ray
    // footprint) to twice the frequency of the pixels. The nyquist theorem
    // guarantees us that this will be in-distinguishable compared to band-limiting
    // to a higher frequency.
    static const float sMaxFrequency = 2.0;
    sp.mPrimaryRayDiffScale = 1.0f / scene_rdl2::math::min(scene_rdl2::math::sqrt(float(pixelSamples)), sMaxFrequency);

    // Compute an adjusted texture differentials scale, based on input
    // "texture blur", so the blurring amount scales linearly, independently
    // of the effect of primaryRayDiffScale. Also compensate for image
    // resolution so the amount of blur is constant in screen space across
    // image resolutions. Finally, we ensure that the texture diff scale never
    // shrinks the texture differentials.
    float scale = pbrTls->mFs->mTextureBlur + 1.0f;
    sp.mTextureDiffScale = (sp.mPrimaryRayDiffScale < 1  ?  2.0f * scale - 1.0f  :  scale)
                        / mResolution;
    sp.mTextureDiffScale = scene_rdl2::math::max(sp.mTextureDiffScale, 1.0f);

    // Setup an initial PathVertex
    pv.pathThroughput = scene_rdl2::math::Color(1.f);
    pv.pathPixelWeight = 1.f;
    pv.aovPathPixelWeight = 1.f;
    pv.pathDistance = 0.f;
    pv.minRoughness = scene_rdl2::math::Vec2f(0.0f);
    pv.diffuseDepth = 0;
    pv.subsurfaceDepth = 0;
    pv.glossyDepth = 0;
    pv.mirrorDepth = 0;
    pv.nonMirrorDepth = 0;
    pv.presenceDepth = 0;
    pv.totalPresence = 0;
    pv.hairDepth = 0;
    pv.volumeDepth = 0;
    pv.accumOpacity = 0;
    pv.lpeStateId = 0;
    pv.lobeType = 0;

    //
    // Apply scaling factor for ray differentials, according to planned
    // ray count per pixel.
    //
    // This works like so... The aim is to keep the ray differential footprint
    // as small as possible so that the surface derivative computations don't
    // get too inaccurate. To achieve this, we try and bake in scaling factors
    // which are less than 1 directly into the ray differential itself.
    // These factors are contained in the ray differential as it traverses the
    // scene.
    //
    // A problem occurs when we hit a lobe which ignores the existing ray differential
    // information when performing its scatter operation. If this happens, all
    // our accumulated scales implicit in the ray differential will be lost. To
    // get around this, lobes which throw out existing ray differential information
    // when computing differentials should set the IGNORES_INCOMING_DIFFERENTIALS
    // flag.
    //
    if (sp.mPrimaryRayDiffScale < 1.0f) {
        ray.scaleDifferentials(sp.mPrimaryRayDiffScale);
    }

    // Increment stats
    Statistics &stats = pbrTls->mStatistics;
    stats.incCounter(STATS_PIXEL_SAMPLES);

    return true;
}

bool
PathIntegrator::isRayOccluded(pbr::TLState *pbrTls, const Light* light, mcrt_common::Ray& shadowRay, float rayEpsilon,
                              float shadowRayEpsilon, float& presence, int receiverId, bool isVolume) const
{
    presence = 0.0f;
    if (!mEnableShadowing) {
        return false;
    }
    // offset shadow ray from surface
    shadowRay.tnear = scene_rdl2::math::max(shadowRay.tnear, shadowRayEpsilon);

    // if falloff enabled, don't shorten ray -- instead, use color interpolation (more expensive)
    const float clearRadius = light->getClearRadiusFalloffDistance() > 0.f ? 0.f : light->getClearRadius();
    // shorten shadowRay length by clear radius
    shadowRay.tfar = scene_rdl2::math::max(shadowRay.tfar - clearRadius, shadowRay.tnear);
    // clip shadowRay length to maxShadowDistance if necessary
    float maxShadowDistance = light->getMaxShadowDistance();
    if (maxShadowDistance > 0.f) {
        shadowRay.tfar = scene_rdl2::math::min(shadowRay.tfar, maxShadowDistance);
    }
    if (light->getPresenceShadows()) {
        // The presence is accumulated as the shadow ray
        // pierces surfaces on its way to the light.
        accumulateRayPresence(pbrTls,
                              light,
                              shadowRay,
                              rayEpsilon,
                              mMaxPresenceDepth,
                              presence);
        return presence > mPresenceThreshold;
    } else {
        // a simpler occlusion ray is sufficient
        shadowRay.ext.instance0OrLight = light->getRdlLight();
        shadowRay.ext.shadowReceiverId = receiverId;
        shadowRay.ext.volumeInstanceState = isVolume; // Reuse this member (otherwise unused in occlusion tests)
        const Scene *scene = MNRY_VERIFY(pbrTls->mFs->mScene);
        return scene->isRayOccluded(pbrTls->mTopLevelTls, shadowRay);
    }
}

scene_rdl2::math::Color
PathIntegrator::computeRadiance(pbr::TLState *pbrTls, int pixelX, int pixelY,
        int subpixelIndex, int pixelSamples, const Sample& sample,
        ComputeRadianceAovParams &aovParams, DeepBuffer *deepBuffer,
        CryptomatteBuffer *cryptomatteBuffer) const
{
    EXCL_ACCUMULATOR_PROFILE(pbrTls, EXCL_ACCUM_INTEGRATION);

    // Aov results
    float &alpha    = aovParams.mAlpha;
    float *depth    = aovParams.mDepth;
    float *aovs     = aovParams.mAovs;

    // Create a sub-pixel structure
    mcrt_common::RayDifferential ray;
    Subpixel sp;
    PathVertex pv;

    const FrameState &fs = *pbrTls->mFs;

    // Create primary ray.
    const Scene *scene = MNRY_VERIFY(fs.mScene);
    const bool validRay = initPrimaryRay(pbrTls, scene->getCamera(), pixelX, pixelY, subpixelIndex,
                                         pixelSamples, sample, ray, sp, pv);

    if (!validRay) {
        alpha = 0.f;
        return scene_rdl2::math::sBlack;
    }

    // LPE
    if (aovs) {
        EXCL_ACCUMULATOR_PROFILE(pbrTls, EXCL_ACCUM_AOVS);

        const LightAovs &lightAovs = *fs.mLightAovs;

        // transition
        pv.lpeStateId = lightAovs.cameraEventTransition(pbrTls);
    }

#if 0
    INIT_PRIMI_VIEW(pvg, PVPIPE);
    GroupStackGuard guard(pvg, "Lens Samples");
    pvg.setColor(0,1,0);

    pvg.point(sample.lensU, sample.lensV, 0);
    //usleep(200000);
#endif

    RAYDB_START_NEW_RAY(pbrTls, ray.org, pixelX, pixelY);

    scene_rdl2::math::Color radiance;
    float pathPixelWeight;
    unsigned sequenceID = fs.mInitialSeed;
    unsigned hsSequenceID = sequenceID;

    // the volume attenuation along this ray to the first hit (or infinity)
    VolumeTransmittance vt;

    CryptomatteParams cryptomatteParams;
    cryptomatteParams.init(cryptomatteBuffer);
    CryptomatteParams *cryptomatteParamsPtr = cryptomatteBuffer ? &cryptomatteParams : nullptr;

    // We maintain a second set of cryptomatte data for refractive cryptomatte that's completely
    //  independent of the regular cryptomatte data.
    CryptomatteParams refractCryptomatteParams;
    refractCryptomatteParams.init(cryptomatteBuffer);
    CryptomatteParams *refractCryptomatteParamsPtr = cryptomatteBuffer ? &refractCryptomatteParams : nullptr;

    if (deepBuffer) {

        float *deepVolumeAovs = aovParams.mDeepVolumeAovs;

        for (int layer = 0; layer < mDeepMaxLayers; layer++) {

            int samplesDivision = 1 << (layer * 2);  // 1, 4, 16, 64 ...
            if ((subpixelIndex % samplesDivision) > 0) {
                continue;
            }

            if (layer > 0) {
                float tfar = ray.tfar;
                initPrimaryRay(pbrTls, scene->getCamera(), pixelX, pixelY, subpixelIndex,
                               pixelSamples, sample, ray, sp, pv);
                ray.tnear = tfar + mDeepLayerBias;

                // LPE
                if (aovs) {
                    fs.mAovSchema->initFloatArray(aovParams.mDeepAovs);
                    const LightAovs &lightAovs = *fs.mLightAovs;
                    pv.lpeStateId = lightAovs.cameraEventTransition(pbrTls);
                }
            }

            DeepParams deepParams;
            deepParams.mDeepBuffer = deepBuffer;
            deepParams.mPixelX = pixelX;
            deepParams.mPixelY = pixelY;
            deepParams.mSampleX = sample.pixelX;
            deepParams.mSampleY = sample.pixelY;
            deepParams.mPixelSamples = pixelSamples;
            deepParams.mHitDeep = false;
            deepParams.mVolumeAovs = deepVolumeAovs;

            // This is the only place we pass a non-null deepParams as we only want to
            // populate the deep buffer for the primary ray.

            // First render normally, capturing the deep volume segments (if any) and
            // the flat radiance + aovs.  Check if we hit any volumes with the primary ray.
            bool hitVolume = false;
            scene_rdl2::math::Color deepRadiance;
            float deepTransparency;
            float *deepAovs = (layer == 0) ? aovs : aovParams.mDeepAovs;
            computeRadianceRecurse(pbrTls, ray, sp, pv, nullptr, deepRadiance,
                                   deepTransparency, vt, sequenceID, deepAovs, depth,
                                   &deepParams, cryptomatteParamsPtr, refractCryptomatteParamsPtr, 
                                   false, hitVolume);

            float deepAlpha = 1.f - deepTransparency;

            if (layer == 0) {
                radiance = deepRadiance;
                alpha = deepAlpha;
                pathPixelWeight = pv.pathPixelWeight;
            }

            if (hitVolume) {
                // We hit a volume.  Render again with the volume disabled to get the
                // correct hard surface radiance without volume attenuation applied.
                // Note that we can't alter the radiance or aovs computed above.  We need
                // independent hs* data structures for this.
                mcrt_common::RayDifferential hsRay;
                Subpixel hsSp;
                PathVertex hsPv;
                initPrimaryRay(pbrTls, scene->getCamera(), pixelX, pixelY, subpixelIndex,
                               pixelSamples, sample, hsRay, hsSp, hsPv);

                // LPE
                if (aovs) {
                    EXCL_ACCUMULATOR_PROFILE(pbrTls, EXCL_ACCUM_AOVS);

                    const LightAovs &lightAovs = *fs.mLightAovs;

                    // transition
                    hsPv.lpeStateId = lightAovs.cameraEventTransition(pbrTls);
                }

                scene_rdl2::math::Color hsRadiance;
                float hsTransparency;
                float *hsAovs = aovParams.mDeepAovs;
                computeRadianceRecurse(pbrTls, hsRay, hsSp, hsPv, nullptr, hsRadiance,
                                       hsTransparency, vt, hsSequenceID, hsAovs, depth,
                                       &deepParams, cryptomatteParamsPtr, refractCryptomatteParamsPtr, 
                                       true, hitVolume);

                float hsAlpha = 1.f;

                if (deepParams.mHitDeep) {
                    // put the hard surface deep intersection into the deep buffer
                    scene_rdl2::math::Vec3f hsNormal = hsRay.getNg();
                    deepBuffer->addSample(pbrTls, deepParams.mPixelX, deepParams.mPixelY,
                                          deepParams.mSampleX, deepParams.mSampleY, layer,
                                          deepParams.mDeepIDs, hsRay.tfar, hsRay.dir.z, hsNormal,
                                          hsAlpha, hsRadiance, hsAovs, 1.f, 1.f);
                }

            } else {
                // We didn't hit a volume.  Fill in the hard surface data with the existing
                // radiance+aov values.
                if (deepParams.mHitDeep) {
                    // put the hard surface deep intersection into the deep buffer
                    scene_rdl2::math::Vec3f deepNormal = ray.getNg();
                    deepBuffer->addSample(pbrTls, deepParams.mPixelX, deepParams.mPixelY,
                                          deepParams.mSampleX, deepParams.mSampleY, layer,
                                          deepParams.mDeepIDs, ray.tfar, ray.dir.z, deepNormal,
                                          deepAlpha, deepRadiance, deepAovs, 1.f, 1.f);
                } else {
                    break; // out of layer loop
                }
            }

            if (checkForNanSimple(radiance, "Path integrator", sp)) {
                // Put a negative value in alpha to denote an invalid sample.
               radiance = scene_rdl2::math::sBlack;
               alpha = -1.0f;
            }

        } // layer

    } else {

        // not deep render
        float transparency;
        bool hitVolume;
        computeRadianceRecurse(pbrTls, ray, sp, pv, nullptr, radiance,
            transparency, vt, sequenceID, aovs, depth, nullptr, cryptomatteParamsPtr,
            refractCryptomatteParamsPtr, false, hitVolume);

        alpha = 1.f - transparency;
        pathPixelWeight = pv.pathPixelWeight;

        if (checkForNanSimple(radiance, "Path integrator", sp)) {
            // Put a negative value in alpha to denote an invalid sample.
            radiance = scene_rdl2::math::sBlack;
            alpha = -1.0f;
        }
    }

    if (cryptomatteBuffer && cryptomatteParams.mHit) {
        cryptomatteBuffer->addSampleScalar(pixelX, pixelY, cryptomatteParams.mId, 
                                                           1.0f, 
                                                           cryptomatteParams.mPosition, 
                                                           cryptomatteParams.mNormal,
                                                           scene_rdl2::math::Color4(radiance),
                                                           cryptomatteParams.mRefP,
                                                           cryptomatteParams.mRefN,
                                                           cryptomatteParams.mUV,
                                                           pv.presenceDepth,
                                                           moonray::pbr::CRYPTOMATTE_TYPE_REGULAR);
    }

    if (cryptomatteBuffer && refractCryptomatteParams.mHit) {
        cryptomatteBuffer->addSampleScalar(pixelX, pixelY, refractCryptomatteParams.mId, 
                                                           1.0f, 
                                                           refractCryptomatteParams.mPosition, 
                                                           refractCryptomatteParams.mNormal,
                                                           scene_rdl2::math::Color4(radiance),
                                                           refractCryptomatteParams.mRefP,
                                                           refractCryptomatteParams.mRefN,
                                                           refractCryptomatteParams.mUV,
                                                           pv.presenceDepth,
                                                           moonray::pbr::CRYPTOMATTE_TYPE_REFRACTED);
    }

#ifdef DO_AOV_RADIANCE_CLAMPING
    // Clamp negative values in final results.
    radiance[0] = std::max(radiance[0], 0.f);
    radiance[1] = std::max(radiance[1], 0.f);
    radiance[2] = std::max(radiance[2], 0.f);
    alpha = std::max(alpha, 0.f);
#endif

    aovSetBeautyAndAlpha(pbrTls, *fs.mAovSchema, radiance, alpha, pathPixelWeight, aovs);

    return radiance;
}

scene_rdl2::math::Color
PathIntegrator::computeColorFromIntersection(pbr::TLState *pbrTls, int pixelX, int pixelY,
        int subpixelIndex, int pixelSamples, const Sample& sample,
        ComputeRadianceAovParams &aovParams, rndr::FastRenderMode fastMode) const
{
    EXCL_ACCUMULATOR_PROFILE(pbrTls, EXCL_ACCUM_INTEGRATION);

    // Create a sub-pixel structure
    mcrt_common::RayDifferential ray;
    Subpixel sp;
    PathVertex pv;

    const FrameState &fs = *pbrTls->mFs;

    // Create primary ray.
    const Scene *scene = MNRY_VERIFY(fs.mScene);
    const bool validRay = initPrimaryRay(pbrTls, scene->getCamera(), pixelX, pixelY, subpixelIndex,
                                         pixelSamples, sample, ray, sp, pv);

    if (!validRay) {
        aovParams.mAlpha = 0.f;
        return scene_rdl2::math::sBlack;
    }

    // LPE
    if (aovParams.mAovs) {
        EXCL_ACCUMULATOR_PROFILE(pbrTls, EXCL_ACCUM_AOVS);

        const LightAovs &lightAovs = *fs.mLightAovs;

        // transition
        pv.lpeStateId = lightAovs.cameraEventTransition(pbrTls);
    }

    RAYDB_START_NEW_RAY(pbrTls, ray.org, pixelX, pixelY);

    shading::Intersection isect;
    int lobeType = 0;
    bool hitGeom = pbrTls->mFs->mScene->intersectRay(pbrTls->mTopLevelTls, ray, isect, lobeType);
    if (!hitGeom) {
        return scene_rdl2::math::sBlack;
    }
    switch(fastMode) {
        case rndr::FastRenderMode::NORMALS:
            return generateNormalColor(isect);
        case rndr::FastRenderMode::NORMALS_SHADING:
            return generateShadingNormalColor(isect);
        case rndr::FastRenderMode::FACING_RATIO:
            return generateFacingRatioColor(isect, ray);
        case rndr::FastRenderMode::FACING_RATIO_INVERSE:
            return generateInverseFacingRatioColor(isect, ray);
        case rndr::FastRenderMode::UVS:
            return generateUVColor(isect);
        default:
            MNRY_ASSERT(!"Should not get here");
            aovParams.mAlpha = 0.f;
            return scene_rdl2::math::sBlack;
    }
}

bool
PathIntegrator::queuePrimaryRay(pbr::TLState *pbrTls,
                                int pixelX,
                                int pixelY,
                                int subpixelIndex,
                                int pixelSamples,
                                const Sample& sample,
                                RayState *rs) const
{
    // Create primary ray.
    const Scene *scene = MNRY_VERIFY(pbrTls->mFs->mScene);
    bool validRay = initPrimaryRay(pbrTls, scene->getCamera(), pixelX, pixelY,
                                   subpixelIndex, pixelSamples, sample, rs->mRay,
                                   rs->mSubpixel, rs->mPathVertex);
    if (!validRay) {
        return false;
    }

    // Fill in remaining RayState members.
    rs->mRay.mask = scene_rdl2::rdl2::CAMERA;
    rs->mSequenceID = pbrTls->mFs->mInitialSeed;

    // LPE
    if (!pbrTls->mFs->mAovSchema->empty()) {
        const LightAovs &lightAovs = *pbrTls->mFs->mLightAovs;
        PathVertex &pv = rs->mPathVertex;

        // transition
        pv.lpeStateId = lightAovs.cameraEventTransition(pbrTls);
    }

    // Queue up ray.
    pbrTls->addRayQueueEntries(1, &rs);

    return true;
}

void
PathIntegrator::integrateBundledv(pbr::TLState *pbrTls,
                                  shading::TLState *shadingTls,
                                  unsigned numEntries,
                                  RayStatev *rayStates,
                                  const shading::Intersectionv *isects,
                                  const shading::Bsdfv *bsdfs,
                                  const LightPtrList *lightList,
                                  const LightFilterLists *lightFilterLists,
                                  const LightAccelerator *lightAcc,
                                  const float *presences) const
{
    EXCL_ACCUMULATOR_PROFILE(pbrTls, EXCL_ACCUM_INTEGRATION);

    MNRY_ASSERT(pbrTls && numEntries && rayStates && isects && bsdfs && lightList);

    // Convert from a LightPtrList into a LightSet before calling into ISPC.
    LightSet lightSet;
    lightSet.init(lightList->empty() ? nullptr : &lightList->front(),
                  int(lightList->size()),
                  &lightFilterLists->front());

    // Augment light set with Embree-based acceleration structure
    // We don't need the light id map yet, this will be filled in during computeActiveLights.
    lightSet.setAccelerator(lightAcc, nullptr);

    pbrTls->startIspcAccumulator();
    ispc::PathIntegrator_integrateBundled(this,
                                          pbrTls,
                                          shadingTls,
                                          numEntries,
                                          rayStates,
                                          isects,
                                          bsdfs,
                                          &lightSet,
                                          presences);
    pbrTls->stopIspcAccumulator();
}

bool
PathIntegrator::getEnablePathGuide() const
{
    return mPathGuide.isEnabled();
}

HUD_VALIDATOR(PathIntegrator);

//----------------------------------------------------------------------------

} // namespace pbr
} // namespace moonray

