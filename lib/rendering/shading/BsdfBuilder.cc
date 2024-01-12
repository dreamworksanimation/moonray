// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

#include "BsdfBuilder.h"
#include "BsdfComponent.h"

#include "bsdf/Bsdf.h"
#include "bsdf/BsdfEyeCaustic.h"
#include "bsdf/BsdfLambert.h"
#include "bsdf/BsdfOrenNayar.h"
#include "bsdf/BsdfMirror.h"
#include "bsdf/cook_torrance/BsdfCookTorrance.h"
#include "bsdf/BsdfStochasticFlakes.h"
#include "bsdf/BsdfIridescence.h"
#include "bsdf/Fresnel.h"
#include "bsdf/hair/BsdfHairDiffuse.h"
#include "bsdf/hair/BsdfHairLobes.h"
#include "bsdf/hair/HairUtil.h"
#include "bsdf/hair/BsdfHairOneSampler.h"
#include "bsdf/under/BsdfUnderClearcoat.h"
#include "bsdf/under/BsdfUnderClearcoatTransmission.h"
#include "bsdf/fabric/BsdfFabric.h"
#include "bsdf/npr/BsdfToon.h"
#include "bsdf/npr/BsdfFlatDiffuse.h"
#include "bssrdf/Bssrdf.h"
#include "bssrdf/VolumeSubsurface.h"

#include <moonray/rendering/shading/Ior.h>
#include <moonray/rendering/shading/Iridescence.h>
#include <moonray/rendering/shading/LobeAttenuator.h>
#include <moonray/rendering/shading/Shading.h>

#include <moonray/rendering/bvh/shading/ShadingTLState.h>
#include <moonray/rendering/bvh/shading/State.h>
#include <moonray/rendering/shading/ispc/BsdfComponent_ispc_stubs.h>

#include <scene_rdl2/common/math/Math.h>
#include <scene_rdl2/render/util/Arena.h>

// this limit is tied to the maximum number of lobes
// supported in Moonray
#define BSDF_BUILDER_MAX_ATTENUATORS 16

namespace {

// Hair seems unstable below 0.01
// but glints benefit from lower roughness
static const float sHairRoughnessMin = 0.01f;
static const float sHairGlintRoughnessMin = 0.001f;

finline float
clampHairRoughness(const float minAttrRoughness, const float minRoughness, const float r)
{
    const float rMin = scene_rdl2::math::max(minAttrRoughness, minRoughness);
    const float rMax = 1.0f - scene_rdl2::math::sEpsilon;
    return scene_rdl2::math::clamp(r, rMin, rMax);
}

} // end anonymous namespace

namespace moonray {
namespace shading {


class BsdfBuilder::Impl
{
public:
    Impl(Bsdf& bsdf,
         shading::TLState *tls,
         const State& state) :
        mBsdf(bsdf),
        mTls(tls),
        mState(state),
        mActiveAttenuators(), // fills array with nulls
        mStagedAttenuators(), // fills array with nulls
        mActiveAttenuatorCount(0),
        mStagedAttenuatorCount(0),
        mActiveHairFresnelAttenuatorChain(nullptr),
        mStagedHairFresnelAttenuatorChain(nullptr),
        mCurrentMediumIor(mState.getMediumIor()),
        mCurrentTransmittance(1.f),
        mWeightAccum(0.f),
        mIsThinGeo(false),
        mPreventLightCulling(false),
        mInAdjacentBlock(false)
    {}

    Impl(const Impl& other) =delete;
    Impl& operator= (const Impl& other) =delete;

    ~Impl() {}

    finline float
    getMinRoughness() const
    {
        const scene_rdl2::math::Vec2f minRoughnessAniso = mState.getMinRoughness();
        return scene_rdl2::math::min(minRoughnessAniso.x, minRoughnessAniso.y);
    }

    finline bool getInAdjacentBlock() const { return mInAdjacentBlock; }
    finline void setInAdjacentBlock(bool state) { mInAdjacentBlock = state; }

    finline void
    addEmission(const scene_rdl2::math::Color& emission)
    {
        mBsdf.setSelfEmission(mBsdf.getSelfEmission() + emission);
    }

    finline void
    setEarlyTermination()
    {
        mBsdf.setEarlyTermination();
    }

    finline Fresnel*
    applyFresnelBehavior(const scene_rdl2::math::Color& eta,
                         const scene_rdl2::math::Color& k,
                         const bool isConductor,
                         const float weight,
                         BsdfLobe * lobe)
    {
        MNRY_ASSERT(lobe);
        Fresnel * fresnel = createFresnel(eta, k, isConductor);
        fresnel->setWeight(weight);

        lobe->setFresnel(fresnel);

        return fresnel;
    }

    finline void
    stageAttenuator(const LobeAttenuator * attenuator)
    {
        MNRY_ASSERT(attenuator);
        MNRY_ASSERT(mStagedAttenuatorCount < (BSDF_BUILDER_MAX_ATTENUATORS - 1));

        mStagedAttenuators[mStagedAttenuatorCount] = attenuator;
        ++mStagedAttenuatorCount;
    }

    // This function is called after adding an individual component,
    // or a set of adjacent components. This allows any component that
    // will somehow affect subsequent components to avoid affecting the
    // adjacent components.  This is achieved by 'staging' these effects
    // as the adjacent components are added, and then finally moving these
    // effects to an 'active' state, after which subsequent components will
    // be affected.
    finline void
    accumulateAttenuation()
    {
        MNRY_ASSERT((mActiveAttenuatorCount + mStagedAttenuatorCount) <= BSDF_BUILDER_MAX_ATTENUATORS);

        // Append any staged attenuators to the active list for subsequent BsdfLobes
        for (size_t i = 0; i < mStagedAttenuatorCount; ++i) {
            mActiveAttenuators[mActiveAttenuatorCount] = mStagedAttenuators[i];
            ++mActiveAttenuatorCount;
        }

        // "Reset" the staging area (so to speak)
        mStagedAttenuatorCount = 0;

        // apply the recently accumulated weight to transmittance for
        // attenuating subsequent lobes
        mCurrentTransmittance *= (1.0f - mWeightAccum);
        mWeightAccum = 0.f;
    }

    finline Fresnel*
    createFresnel(const scene_rdl2::math::Color& eta,
                  const scene_rdl2::math::Color& k,
                  const bool isConductor)
    {
        if (isConductor) {
            // Currently, Moonray does not respect the "media IOR" for conductors.
            // Eventually we should compute *relative* IOR, using mCurrentMediumIor
            return mTls->mArena->allocWithArgs<ConductorFresnel>(eta, k);
        } else {
            shading::ShaderIor ior(mState,
                                   eta.r,
                                   mIsThinGeo);
            return mTls->mArena->allocWithArgs<DielectricFresnel>(ior.getIncident(),
                                                                  ior.getTransmitted());
        }
    }

    inline bool
    isUnder(ispc::BsdfBuilderBehavior combineBehavior)
    {
        return (combineBehavior & ispc::BSDFBUILDER_UNDER_PREVIOUS);
    }

    inline bool
    isOver(ispc::BsdfBuilderBehavior combineBehavior)
    {
        return (combineBehavior & ispc::BSDFBUILDER_OVER_SUBSEQUENT);
    }

    // check to see if a new lobe would potentially be visible
    finline bool
    testForVisibility(const float weight,
                      ispc::BsdfBuilderBehavior combineBehavior)
    {
        if (weight < scene_rdl2::math::sEpsilon) { return false; }
        return (!isUnder(combineBehavior) || mCurrentTransmittance > 0.f);
    }

    finline BsdfLobe *
    placeUnderPreviousLobes(BsdfLobe * lobe)
    {
        MNRY_ASSERT(lobe);

        if (mActiveAttenuatorCount > 0) {
            int idx = mActiveAttenuatorCount - 1;
            while (idx >= 0) {
                const LobeAttenuator& attenuate = *(mActiveAttenuators[idx--]);
                lobe = attenuate(mTls->mArena, lobe);
            }
        }

        return lobe;
    }

    finline void
    placeUnderPreviousLobes(Bssrdf * bssrdf)
    {
        MNRY_ASSERT(bssrdf);

        if (mActiveAttenuatorCount > 0) {
            int idx = mActiveAttenuatorCount - 1;
            while (idx >= 0) {
                const LobeAttenuator& attenuate = *(mActiveAttenuators[idx--]);
                attenuate(mTls->mArena, bssrdf);
            }
        }
    }

    finline void
    placeUnderPreviousLobes(VolumeSubsurface* vs)
    {
        MNRY_ASSERT(vs);

        if (mActiveAttenuatorCount > 0) {
            int idx = mActiveAttenuatorCount - 1;
            while (idx >= 0) {
                const LobeAttenuator& attenuate = *(mActiveAttenuators[idx--]);
                attenuate(mTls->mArena, vs);
            }
        }
    }

    finline BsdfLobe *
    handleIridescence(BsdfLobe * lobe,
                      const Iridescence * const iridescence)
    {
        if (!iridescence) { return lobe; }

        lobe = mTls->mArena->allocWithArgs<IridescenceBsdfLobe>(
                lobe,
                iridescence->getN(),
                iridescence->getStrength(),
                iridescence->getColorControl(),
                iridescence->getPrimary(),
                iridescence->getSecondary(),
                iridescence->getFlipHue(),
                iridescence->getRampInterpolationMode(),
                iridescence->getRampNumPoints(),
                iridescence->getRampPositions(),
                iridescence->getRampInterpolators(),
                iridescence->getRampColors(),
                iridescence->getThickness(),
                iridescence->getExponent(),
                iridescence->getIridescenceAt0(),
                iridescence->getIridescenceAt90());

        return lobe;
    }

    finline void
    addComponent(const MicrofacetAnisotropicClearcoat& component,
                 float weight,
                 ispc::BsdfBuilderBehavior combineBehavior,
                 const int label)
    {
        const float roughnessU = component.getRoughnessU();
        const float roughnessV = component.getRoughnessV();
        const float averageRoughness = (roughnessU + roughnessV) * 0.5f;

        // Adapt normal to prevent reflection ray from self-intersecting this geometry
        const scene_rdl2::math::Vec3f adaptedNormal = mState.adaptNormal(component.getN());

        // We currently only support anisotropy with Beckmann
        BsdfLobe * lobe =
            mTls->mArena->allocWithArgs<AnisoCookTorranceBsdfLobe>(adaptedNormal,
                                                                   component.getShadingTangent(),
                                                                   roughnessU,
                                                                   roughnessV);

        Fresnel * fresnel = applyFresnelBehavior(scene_rdl2::math::Color(component.getEta()),
                                                 scene_rdl2::math::sBlack, // k
                                                 false,        // isConductor
                                                 weight,
                                                 lobe);

        lobe = handleIridescence(lobe, component.getIridescence());

        scene_rdl2::math::Color scale = scene_rdl2::math::sWhite;

        if (isUnder(combineBehavior)) {
            // account for the non-dielectric lobes above
            scale *= mCurrentTransmittance;
            lobe->setScale(scale);
            // account for dielectric/clearcoat lobes above
            lobe = placeUnderPreviousLobes(lobe);
        } else {
            lobe->setScale(scale);
        }

        if (isOver(combineBehavior)) {
            ClearcoatAttenuator * attenuator =
                mTls->mArena->allocWithArgs<ClearcoatAttenuator>(
                        mTls->mArena,
                        adaptedNormal,
                        averageRoughness,
                        mCurrentMediumIor,
                        component.getEta(),
                        component.getRefracts(),
                        component.getThickness(),
                        component.getAttenuationColor(),
                        weight, // The attenuation effect diminishes with weight
                        fresnel);

            stageAttenuator(attenuator);
        }

        lobe->setLabel(label);
        mBsdf.addLobe(lobe);
    }

    finline void
    addComponent(const MicrofacetIsotropicClearcoat& component,
                 float weight,
                 ispc::BsdfBuilderBehavior combineBehavior,
                 const int label)
    {
        // Disable layering clearcoat/spec IORs
        // The results looks visually better when we don't layer clearcoat/spec IORs
        // Checkout MOONSHINE-601 for detailed renders
        // updateMediumIor(component.getEta(), weight);

        // Adapt normal to prevent reflection ray from self-intersecting this geometry
        const scene_rdl2::math::Vec3f adaptedNormal = mState.adaptNormal(component.getN());

        BsdfLobe * lobe;
        if (component.getMicrofacetDistribution() == ispc::MICROFACET_DISTRIBUTION_BECKMANN) {
            lobe = mTls->mArena->allocWithArgs<CookTorranceBsdfLobe>(adaptedNormal,
                                                                     component.getRoughness());
        } else {
            lobe = mTls->mArena->allocWithArgs<GGXCookTorranceBsdfLobe>(adaptedNormal,
                                                                        component.getRoughness());
        }

        Fresnel * fresnel = applyFresnelBehavior(scene_rdl2::math::Color(component.getEta()),
                                                 scene_rdl2::math::sBlack, // k
                                                 false,        // isConductor
                                                 weight,
                                                 lobe);

        lobe = handleIridescence(lobe, component.getIridescence());

        scene_rdl2::math::Color scale = scene_rdl2::math::sWhite;

        if (isUnder(combineBehavior)) {
            // account for the non-dielectric lobes above
            scale *= mCurrentTransmittance;
            lobe->setScale(scale);
            // account for dielectric/clearcoat lobes above
            lobe = placeUnderPreviousLobes(lobe);
        } else {
            lobe->setScale(scale);
        }


        if (isOver(combineBehavior)) {
            ClearcoatAttenuator * attenuator =
                mTls->mArena->allocWithArgs<ClearcoatAttenuator>(
                        mTls->mArena,
                        adaptedNormal,
                        component.getRoughness(),
                        mCurrentMediumIor,
                        component.getEta(),
                        component.getRefracts(),
                        component.getThickness(),
                        component.getAttenuationColor(),
                        weight, // The attenuation effect diminishes with weight
                        fresnel);

            stageAttenuator(attenuator);
        }

        lobe->setLabel(label);
        mBsdf.addLobe(lobe);
    }

    finline void
    addComponent(const MirrorClearcoat& component,
                 float weight,
                 ispc::BsdfBuilderBehavior combineBehavior,
                 const int label)
    {
        // Disable layering clearcoat/spec IORs
        // The results looks visually better when we don't layer clearcoat/spec IORs
        // Checkout MOONSHINE-601 for detailed renders
        // updateMediumIor(component.getEta(), weight);

        // Adapt normal to prevent reflection ray from self-intersecting this geometry
        const scene_rdl2::math::Vec3f adaptedNormal = mState.adaptNormal(component.getN());

        BsdfLobe * lobe =
            mTls->mArena->allocWithArgs<MirrorReflectionBsdfLobe>(
                    adaptedNormal);

        Fresnel * fresnel = applyFresnelBehavior(scene_rdl2::math::Color(component.getEta()),
                                                 scene_rdl2::math::sBlack, // k
                                                 false,        // isConductor
                                                 weight,
                                                 lobe);

        lobe = handleIridescence(lobe, component.getIridescence());

        scene_rdl2::math::Color scale = scene_rdl2::math::sWhite;

        if (isUnder(combineBehavior)) {
            // account for the non-dielectric lobes above
            scale *= mCurrentTransmittance;
            lobe->setScale(scale);
            // account for dielectric/clearcoat lobes above
            lobe = placeUnderPreviousLobes(lobe);
        } else {
            lobe->setScale(scale);
        }

        if (isOver(combineBehavior)) {
            ClearcoatAttenuator * attenuator =
                mTls->mArena->allocWithArgs<ClearcoatAttenuator>(
                        mTls->mArena,
                        adaptedNormal,
                        0.f,
                        mCurrentMediumIor,
                        component.getEta(),
                        component.getRefracts(),
                        component.getThickness(),
                        component.getAttenuationColor(),
                        weight, // The attenuation effect diminishes with weight
                        fresnel);

            stageAttenuator(attenuator);
        }

        lobe->setLabel(label);
        mBsdf.addLobe(lobe);
    }

    finline void
    addComponent(const EyeCausticBRDF& component,
                 float weight,
                 ispc::BsdfBuilderBehavior combineBehavior,
                 const int label)
    {
        BsdfLobe* lobe = mTls->mArena->allocWithArgs<EyeCausticBsdfLobe>(
                        mState.adaptNormal(component.getN()),
                        component.getIrisNormal(),
                        component.getCausticColor(),
                        component.getExponent());
        scene_rdl2::math::Color scale(weight);

        if (isUnder(combineBehavior)) {
            // account for the non-dielectric lobes above
            scale *= mCurrentTransmittance;
            lobe->setScale(scale);
            // account for dielectric/clearcoat lobes above
            lobe = placeUnderPreviousLobes(lobe);
        } else {
            lobe->setScale(scale);
        }

        if (isOver(combineBehavior)) {
            // account for this lobe's energy allocation
            mWeightAccum += weight;
        }

        lobe->setLabel(label);
        mBsdf.addLobe(lobe);
    }

    finline void
    addComponent(const LambertianBRDF& component,
                 float weight,
                 ispc::BsdfBuilderBehavior combineBehavior,
                 const int label)
    {
        BsdfLobe* lobe = mTls->mArena->allocWithArgs<LambertBsdfLobe>(
                component.getN(),
                component.getAlbedo(),
                true); // true for reflection, false for transmission

        if (mPreventLightCulling) {
            // Force lobe to be "spherical" to prevent light culling 
            // when light sample is on backside
            const shading::BsdfLobe::Type lobeType = shading::BsdfLobe::ALL_DIFFUSE;
            lobe->setType(lobeType);
            lobe->setIsSpherical(true);
        }

        scene_rdl2::math::Color scale(weight);

        if (isUnder(combineBehavior)) {
            // account for the non-dielectric lobes above
            scale *= mCurrentTransmittance;
            lobe->setScale(scale);
            // account for the dielectric/clearcoat lobes above
            lobe = placeUnderPreviousLobes(lobe);
        } else {
            lobe->setScale(scale);
        }

        if (isOver(combineBehavior)) {
            // account for this lobe's energy allocation
            mWeightAccum += weight;
        }

        lobe->setLabel(label);
        mBsdf.addLobe(lobe);
    }

    finline void
    addComponent(const LambertianBTDF& component,
                 float weight,
                 ispc::BsdfBuilderBehavior combineBehavior,
                 const int label)
    {
        BsdfLobe* lobe = mTls->mArena->allocWithArgs<LambertBsdfLobe>(
                component.getN(),
                component.getTint(),
                false); // true for reflection, false for transmission

        scene_rdl2::math::Color scale(weight);

        if (isUnder(combineBehavior)) {
            // account for the non-dielectric lobes above
            scale *= mCurrentTransmittance;
            lobe->setScale(scale);
            // account for the dielectric/clearcoat lobes above
            lobe = placeUnderPreviousLobes(lobe);
        } else {
            lobe->setScale(scale);
        }

        if (isOver(combineBehavior)) {
            // account for this lobe's energy allocation
            mWeightAccum += weight;
        }

        lobe->setLabel(label);
        mBsdf.addLobe(lobe);
    }

    finline void
    addComponent(const OrenNayarBRDF& component,
                 float weight,
                 ispc::BsdfBuilderBehavior combineBehavior,
                 const int label)
    {
        BsdfLobe* lobe = mTls->mArena->allocWithArgs<OrenNayarBsdfLobe>(
                component.getN(),
                component.getAlbedo(),
                component.getRoughness(),
                true); // true for reflection, false for transmission

        if (mPreventLightCulling) {
            // Force lobe to be "spherical" to prevent culling
            // when light sample is on backside
            const shading::BsdfLobe::Type lobeType = shading::BsdfLobe::ALL_DIFFUSE;
            lobe->setType(lobeType);
            lobe->setIsSpherical(true);
        }

        scene_rdl2::math::Color scale(weight);

        if (isUnder(combineBehavior)) {
            // account for the non-dielectric lobes above
            scale *= mCurrentTransmittance;
            lobe->setScale(scale);
            // account for the dielectric/clearcoat lobes above
            lobe = placeUnderPreviousLobes(lobe);
        } else {
            lobe->setScale(scale);
        }

        if (isOver(combineBehavior)) {
            // account for this lobe's energy allocation
            mWeightAccum += weight;
        }

        lobe->setLabel(label);
        mBsdf.addLobe(lobe);
    }

    finline void
    addComponent(const FlatDiffuseBRDF& component,
                 float weight,
                 ispc::BsdfBuilderBehavior combineBehavior,
                 const int label)
    {
        BsdfLobe* lobe = mTls->mArena->allocWithArgs<FlatDiffuseBsdfLobe>(
                component.getN(),
                component.getAlbedo(),
                component.getRoughness(),
                component.getTerminatorShift(),
                component.getFlatness(),
                component.getFlatnessFalloff(),
                true); // true for reflection, false for transmission

        if (mPreventLightCulling) {
            // Force lobe to be "spherical" to prevent culling
            // when light sample is on backside
            const shading::BsdfLobe::Type lobeType = shading::BsdfLobe::ALL_DIFFUSE;
            lobe->setType(lobeType);
            lobe->setIsSpherical(true);
        }

        scene_rdl2::math::Color scale(weight);

        if (isUnder(combineBehavior)) {
            // account for the non-dielectric lobes above
            scale *= mCurrentTransmittance;
            lobe->setScale(scale);
            // account for the dielectric/clearcoat lobes above
            lobe = placeUnderPreviousLobes(lobe);
        } else {
            lobe->setScale(scale);
        }

        if (isOver(combineBehavior)) {
            // account for this lobe's energy allocation
            mWeightAccum += weight;
        }

        lobe->setLabel(label);
        mBsdf.addLobe(lobe);
    }

    finline void
    addComponent(const DipoleDiffusion& component,
                 float weight,
                 ispc::BsdfBuilderBehavior combineBehavior,
                 const int label)
    {
        // Switch to Lambertian diffuse if max_subsurface_per_path has been reached
        if (!mState.isSubsurfaceAllowed()) {
            LambertianBRDF diffuse(component.getN(),
                                   component.getAlbedo());
            addComponent(diffuse,
                         weight,
                         combineBehavior,
                         label);
            return;
        }

        auto bssrdf = createBSSRDF(ispc::SUBSURFACE_DIPOLE_DIFFUSION,
                                   mTls->mArena,
                                   component.getN(),
                                   component.getAlbedo(),
                                   component.getRadius(),
                                   component.getMaterial(),
                                   component.getEvalNormalFn());

        scene_rdl2::math::Color scale(weight);

        if (isUnder(combineBehavior)) {
            // account for the non-dielectric lobes above
            scale *= mCurrentTransmittance;
            bssrdf->setScale(scale);

            // setup a MultipleTransmissionFresnel so this bssrdf can
            // be attenuated by previous lobes
            MultipleTransmissionFresnel * fresnel =
                mTls->mArena->allocWithCtor<MultipleTransmissionFresnel>();
            bssrdf->setTransmissionFresnel(fresnel);

            // account for dielectric lobes above (clearcoat not supported for bssrdfs)
            placeUnderPreviousLobes(bssrdf);
        } else {
            bssrdf->setScale(scale);
        }

        if (isOver(combineBehavior)) {
            // account for this lobe's energy allocation
            mWeightAccum += weight;
        }

        bssrdf->setTraceSet(component.getTraceSet());
        bssrdf->setLabel(label);
        mBsdf.setBssrdf(bssrdf);
    }

    finline void
    addComponent(const NormalizedDiffusion& component,
                 float weight,
                 ispc::BsdfBuilderBehavior combineBehavior,
                 const int label)
    {
        // Switch to Lambertian diffuse if max_subsurface_per_path has been reached
        if (!mState.isSubsurfaceAllowed()) {
            LambertianBRDF diffuse(component.getN(),
                                   component.getAlbedo());
            addComponent(diffuse,
                         weight,
                         combineBehavior,
                         label);
            return;
        }

        auto bssrdf = createBSSRDF(ispc::SUBSURFACE_NORMALIZED_DIFFUSION,
                                   mTls->mArena,
                                   component.getN(),
                                   component.getAlbedo(),
                                   component.getRadius(),
                                   component.getMaterial(),
                                   component.getEvalNormalFn());

        scene_rdl2::math::Color scale(weight);

        if (isUnder(combineBehavior)) {
            // account for the non-dielectric lobes above
            scale *= mCurrentTransmittance;
            bssrdf->setScale(scale);

            // setup a MultipleTransmissionFresnel so this bssrdf can
            // be attenuated by previous lobes
            MultipleTransmissionFresnel * fresnel =
                mTls->mArena->allocWithCtor<MultipleTransmissionFresnel>();
            bssrdf->setTransmissionFresnel(fresnel);

            // account for dielectric lobes above (clearcoat not supported for bssrdfs)
            placeUnderPreviousLobes(bssrdf);
        } else {
            bssrdf->setScale(scale);
        }

        if (isOver(combineBehavior)) {
            // account for this lobe's energy allocation
            mWeightAccum += weight;
        }

        bssrdf->setTraceSet(component.getTraceSet());
        bssrdf->setLabel(label);
        mBsdf.setBssrdf(bssrdf);
    }

    finline void
    addComponent(const RandomWalkSubsurface& component,
                 float weight,
                 ispc::BsdfBuilderBehavior combineBehavior,
                 const int label)
    {
        // Switch to Lambertian diffuse if max_subsurface_per_path has been reached
        if (!mState.isSubsurfaceAllowed()) {
            LambertianBRDF diffuse(component.getN(),
                                   component.getAlbedo());
            addComponent(diffuse,
                         weight,
                         combineBehavior,
                         label);
            return;
        }

        auto bssrdf = createVolumeSubsurface(mTls->mArena,
                                             component.getAlbedo(),
                                             component.getRadius(),
                                             component.getMaterial(),
                                             component.getEvalNormalFn(),
                                             component.getN(),
                                             component.getResolveSelfIntersections());

        scene_rdl2::math::Color scale(weight);

        if (isUnder(combineBehavior)) {
            // account for the non-dielectric lobes above
            scale *= mCurrentTransmittance;
            bssrdf->setScale(scale);

            // setup a MultipleTransmissionFresnel so this bssrdf can
            // be attenuated by previous lobes
            MultipleTransmissionFresnel * fresnel =
                mTls->mArena->allocWithCtor<MultipleTransmissionFresnel>();
            bssrdf->setTransmissionFresnel(fresnel);

            // account for dielectric lobes above (clearcoat not supported for bssrdfs)
            placeUnderPreviousLobes(bssrdf);
        } else {
            bssrdf->setScale(scale);
        }

        if (isOver(combineBehavior)) {
            // account for this lobe's energy allocation
            mWeightAccum += weight;
        }

        bssrdf->setTraceSet(component.getTraceSet());
        bssrdf->setLabel(label);
        mBsdf.setVolumeSubsurface(bssrdf);
    }

    finline void
    addComponent(const MicrofacetAnisotropicBRDF& component,
                 float weight,
                 ispc::BsdfBuilderBehavior combineBehavior,
                 const int label)
    {
        // Adapt normal to prevent reflection ray from self-intersecting this geometry
        const scene_rdl2::math::Vec3f adaptedNormal = mState.adaptNormal(component.getN());

        BsdfLobe * lobe = mTls->mArena->allocWithArgs<AnisoCookTorranceBsdfLobe>(
                adaptedNormal,
                component.getShadingTangent(),
                component.getRoughnessU(),
                component.getRoughnessV());

        Fresnel * fresnel = applyFresnelBehavior(component.getEta(),
                                                 component.getK(),
                                                 component.isConductor(),
                                                 weight,
                                                 lobe);

        lobe = handleIridescence(lobe, component.getIridescence());

        scene_rdl2::math::Color scale = scene_rdl2::math::sWhite;

        if (isUnder(combineBehavior)) {
            // account for the non-dielectric lobes above
            scale *= mCurrentTransmittance;
            lobe->setScale(scale);
            // account for the dielectric/clearcoat lobes above
            lobe = placeUnderPreviousLobes(lobe);
        } else {
            lobe->setScale(scale);
        }

        if (isOver(combineBehavior)) {
            if (component.isConductor()) {
                // conductors don't transmit the energy, they reflect it or absorb it
                mWeightAccum += weight;
            } else {
                // dielectrics transmit any unreflected energy
                const float avgRoughness = (component.getRoughnessU() + component.getRoughnessV()) * 0.5f;
                SimpleAttenuator * atten = mTls->mArena->allocWithArgs<SimpleAttenuator>(
                        mTls->mArena,
                        adaptedNormal,
                        avgRoughness,
                        fresnel);

                stageAttenuator(atten);
            }
        }

        lobe->setLabel(label);
        mBsdf.addLobe(lobe);
    }

    finline void
    addComponent(const MicrofacetAnisotropicBTDF& component,
                 float weight,
                 ispc::BsdfBuilderBehavior combineBehavior,
                 const int label)
    {
        shading::ShaderIor ior(mState,
                               component.getEta(),
                               mIsThinGeo);
        // TODO::
        // Add support for anisotropic BTDF energy preservation
        float favg, favgInv;
        shading::averageFresnelReflectance(
                ior.getTransmitted()/ior.getIncident(),
                favg, favgInv);
        // TODO: MOONSHINE-1035
        // Relative IOR == 1 is not handled and neither is thinGeometry

        // only isotropic microfacet transmission supported currently
        const float avgRoughness = (component.getRoughnessU() + component.getRoughnessV()) * 0.5f;
        BsdfLobe * lobe = mTls->mArena->allocWithArgs<TransmissionCookTorranceBsdfLobe>(
                component.getN(),
                avgRoughness,
                ior.getIncident(),
                ior.getTransmitted(),
                component.getTint(),
                favg, favgInv,
                component.getAbbeNumber());

        scene_rdl2::math::Color scale = scene_rdl2::math::Color(weight);

        if (isUnder(combineBehavior)) {
            // account for the non-dielectric lobes above
            scale *= mCurrentTransmittance;
            lobe->setScale(scale);
            // account for the dielectric/clearcoat lobes above
            lobe = placeUnderPreviousLobes(lobe);
        } else {
            lobe->setScale(scale);
        }

        if (isOver(combineBehavior)) {
            // account for this lobe's energy allocation
            mWeightAccum += weight;
        }

        lobe->setLabel(label);
        mBsdf.addLobe(lobe);
    }

    finline void
    addComponent(const MicrofacetAnisotropicBSDF& component,
                 float weight,
                 ispc::BsdfBuilderBehavior combineBehavior,
                 const int reflectionLabel,
                 const int transmissionLabel)
    {
        const float reflWeight  = weight * component.getReflectionWeight();
        const float transWeight = weight * component.getTransmissionWeight();

        if (scene_rdl2::math::isZero(reflWeight + transWeight)) { return; }

        shading::ShaderIor ior(mState,
                               component.getRefractionEta(),
                               mIsThinGeo);

        BsdfLobe * refl = nullptr;
        BsdfLobe * trans = nullptr;

        // Adapt normal to prevent reflection ray from self-intersecting this geometry
        const scene_rdl2::math::Vec3f adaptedNormal = mState.adaptNormal(component.getN());

        if (reflWeight > 0.f) {
            refl = mTls->mArena->allocWithArgs<AnisoCookTorranceBsdfLobe>(
                    adaptedNormal,
                    component.getShadingTangent(),
                    component.getRoughnessU(),
                    component.getRoughnessV());
        }

        const float avgRoughness = (component.getRoughnessU() + component.getRoughnessV()) * 0.5f;
        if (transWeight > 0.f) {
            // only isotropic microfacet transmission supported currently
            float favg, favgInv;
            shading::averageFresnelReflectance(
                    ior.getTransmitted()/ior.getIncident(),
                    favg, favgInv);
            // TODO: MOONSHINE-1035
            // Relative IOR == 1 is not handled and neither is thinGeometry
            trans = mTls->mArena->allocWithArgs<TransmissionCookTorranceBsdfLobe>(
                    component.getN(),
                    avgRoughness,
                    ior.getIncident(),
                    ior.getTransmitted(),
                    component.getTint(),
                    favg, favgInv,
                    component.getAbbeNumber());
        }

        Fresnel * fresnel = nullptr;
        if (refl) {
            fresnel = applyFresnelBehavior(scene_rdl2::math::Color(component.getEta()),
                                           scene_rdl2::math::sBlack,    // k
                                           false,           // isConductor
                                           reflWeight,
                                           refl);

            refl = handleIridescence(refl, component.getIridescence());

            if (trans) {
                Fresnel * oneMinusFresnel =
                    mTls->mArena->allocWithArgs<OneMinusFresnel>(fresnel);
                trans->setFresnel(oneMinusFresnel);
            }
        }

        scene_rdl2::math::Color scale = scene_rdl2::math::sWhite;

        if (isUnder(combineBehavior)) {
            // account for the non-dielectric lobes above
            scale *= mCurrentTransmittance;
            // account for the dielectric/clearcoat lobes above
            if (refl) { refl = placeUnderPreviousLobes(refl); }
            if (trans) { trans = placeUnderPreviousLobes(trans); }
        }

        if (isOver(combineBehavior)) {
            if (fresnel && transWeight < 1.f) {
                // the remaining (unreflected) energy normally given to the
                // transmission lobe is to be divided up and some given to
                // subsequent lobes.  When this happens, we still need to
                // account for the light being reflected, therefore we need
                // a SimpleAttenuator
                SimpleAttenuator * atten = mTls->mArena->allocWithArgs<SimpleAttenuator>(
                        mTls->mArena,
                        adaptedNormal,
                        avgRoughness,
                        fresnel);

                stageAttenuator(atten);
            }

            // account for portion of energy given to transmission lobe
            mWeightAccum += transWeight;
        }

        if (refl) {
            refl->setScale(scale);
            refl->setLabel(reflectionLabel);
            mBsdf.addLobe(refl);
        }

        if (trans) {
            trans->setScale(scale * transWeight);
            trans->setLabel(transmissionLabel);
            mBsdf.addLobe(trans);
        }
    }

    finline void
    addComponent(const MicrofacetIsotropicBRDF& component,
                 float weight,
                 ispc::BsdfBuilderBehavior combineBehavior,
                 const int label)
    {
        // Adapt normal to prevent reflection ray from self-intersecting this geometry
        const scene_rdl2::math::Vec3f adaptedNormal = mState.adaptNormal(component.getN());

        BsdfLobe * lobe;
        if (component.getMicrofacetDistribution() == ispc::MICROFACET_DISTRIBUTION_BECKMANN) {
            lobe = mTls->mArena->allocWithArgs<CookTorranceBsdfLobe>(
                    adaptedNormal,
                    component.getRoughness(),
                    component.getFavg());
        } else {
            lobe = mTls->mArena->allocWithArgs<GGXCookTorranceBsdfLobe>(
                    adaptedNormal,
                    component.getRoughness(),
                    component.getFavg());
        }

        Fresnel * fresnel = applyFresnelBehavior(component.getEta(),
                                                 component.getK(),
                                                 component.isConductor(),
                                                 weight,
                                                 lobe);

        lobe = handleIridescence(lobe, component.getIridescence());

        scene_rdl2::math::Color scale = scene_rdl2::math::sWhite;

        if (isUnder(combineBehavior)) {
            // account for the non-dielectric lobes above
            scale *= mCurrentTransmittance;
            lobe->setScale(scale);
            // account for the dielectric/clearcoat lobes above
            lobe = placeUnderPreviousLobes(lobe);
        } else {
            lobe->setScale(scale);
        }

        if (isOver(combineBehavior)) {
            if (component.isConductor()) {
                // conductors don't transmit the energy, they reflect it or absorb it
                mWeightAccum += weight;
            } else {
                // dielectrics transmit any unreflected energy
                SimpleAttenuator * atten = mTls->mArena->allocWithArgs<SimpleAttenuator>(
                        mTls->mArena,
                        adaptedNormal,
                        component.getRoughness(),
                        fresnel);

                stageAttenuator(atten);
            }
        }

        lobe->setLabel(label);
        mBsdf.addLobe(lobe);
    }

    finline void
    addComponent(const MicrofacetIsotropicBTDF& component,
                 float weight,
                 ispc::BsdfBuilderBehavior combineBehavior,
                 const int label)
    {
        shading::ShaderIor ior(mState,
                               component.getEta(),
                               mIsThinGeo);
        float favg, favgInv;
        shading::averageFresnelReflectance(
                ior.getTransmitted()/ior.getIncident(),
                favg, favgInv);
        // TODO: MOONSHINE-1035
        // Relative IOR == 1 is not handled and neither is thinGeometry
        BsdfLobe * lobe = mTls->mArena->allocWithArgs<TransmissionCookTorranceBsdfLobe>(
                component.getN(),
                component.getRoughness(),
                ior.getIncident(),
                ior.getTransmitted(),
                component.getTint(),
                favg, favgInv,
                component.getAbbeNumber());

        scene_rdl2::math::Color scale(weight);

        if (isUnder(combineBehavior)) {
            // account for the non-dielectric lobes above
            scale *= mCurrentTransmittance;
            lobe->setScale(scale);
            // account for the dielectric/clearcoat lobes above
            lobe = placeUnderPreviousLobes(lobe);
        } else {
            lobe->setScale(scale);
        }

        if (isOver(combineBehavior)) {
            // account for this lobe's energy allocation
            mWeightAccum += weight;
        }

        lobe->setLabel(label);
        mBsdf.addLobe(lobe);
    }

    finline void
    addComponent(const MicrofacetIsotropicBSDF& component,
                 float weight,
                 ispc::BsdfBuilderBehavior combineBehavior,
                 const int reflectionLabel,
                 const int transmissionLabel)
    {
        const float reflWeight  = weight * component.getReflectionWeight();
        const float transWeight = weight * component.getTransmissionWeight();

        const bool isCoupledWithTransmission = !scene_rdl2::math::isZero(transWeight);

        if (scene_rdl2::math::isZero(reflWeight + transWeight)) { return; }

        BsdfLobe * refl = nullptr;
        BsdfLobe * trans = nullptr;

        // Adapt normal to prevent reflection ray from self-intersecting this geometry
        const scene_rdl2::math::Vec3f adaptedNormal = mState.adaptNormal(component.getN());

        shading::ShaderIor ior(mState,
                               component.getRefractionEta(),
                               mIsThinGeo);
        float favg, favgInv;
        shading::averageFresnelReflectance(
                ior.getTransmitted()/ior.getIncident(),
                favg, favgInv);

        if (reflWeight > 0.f) {
            if (component.getMicrofacetDistribution() == ispc::MICROFACET_DISTRIBUTION_BECKMANN) {
                refl = mTls->mArena->allocWithArgs<CookTorranceBsdfLobe>(
                       adaptedNormal,
                       component.getRoughness(),
                       scene_rdl2::math::sWhite*favg, scene_rdl2::math::sWhite*favgInv,
                       ior.getIncident(),
                       ior.getTransmitted(),
                       isCoupledWithTransmission); //coupledWithTransmission
            } else {
                refl = mTls->mArena->allocWithArgs<GGXCookTorranceBsdfLobe>(
                       adaptedNormal,
                       component.getRoughness(),
                       scene_rdl2::math::sWhite*favg, scene_rdl2::math::sWhite*favgInv,
                       ior.getIncident(),
                       ior.getTransmitted(),
                       isCoupledWithTransmission); //coupledWithTransmission
            }
        }

        if (transWeight > 0.f) {
            // Because the TransmissionCookTorranceBsdfLobe becomes unstable as the
            // relative IOR approaches 1.0 we create a mirror transmission lobe here.
            // see MOONSHINE-1035
            // I'd like to remove this check and just do something simple, like detect
            // when the ratio is 1.0 and adjust one of the IORs accordingly so it is not.
            if (scene_rdl2::math::isOne(ior.getRatio()) || mIsThinGeo) {
                trans = mTls->mArena->allocWithArgs<MirrorTransmissionBsdfLobe>(
                        component.getN(),
                        ior.getIncident(),
                        mIsThinGeo ? ior.getIncident() : ior.getTransmitted(),
                        component.getTint(),
                        component.getAbbeNumber());
            } else {
                // only isotropic microfacet transmission supported currently
                trans = mTls->mArena->allocWithArgs<TransmissionCookTorranceBsdfLobe>(
                        component.getN(),
                        component.getRoughness(),
                        ior.getIncident(),
                        ior.getTransmitted(),
                        component.getTint(),
                        favg, favgInv,
                        component.getAbbeNumber());
            }
        }

        Fresnel* fresnel = nullptr;
        if (refl) {
            fresnel = applyFresnelBehavior(scene_rdl2::math::Color(component.getEta()),
                                           scene_rdl2::math::sBlack,    // k
                                           false,           // isConductor
                                           reflWeight,
                                           refl);

            refl = handleIridescence(refl, component.getIridescence());

            if (trans) {
                Fresnel* oneMinusFresnel =
                    mTls->mArena->allocWithArgs<OneMinusFresnel>(fresnel);
                trans->setFresnel(oneMinusFresnel);
            }
        }

        scene_rdl2::math::Color scale = scene_rdl2::math::sWhite;

        if (isUnder(combineBehavior)) {
            // account for the non-dielectric lobes above
            scale *= mCurrentTransmittance;
            // account for the dielectric/clearcoat lobes above
            if (refl) {
                refl->setScale(scale);
                refl = placeUnderPreviousLobes(refl);
            }
            if (trans) {
                trans->setScale(scale * transWeight);
                trans = placeUnderPreviousLobes(trans);
            }
        } else {
            if (refl) { refl->setScale(scale); }
            if (trans) { trans->setScale(scale * transWeight); }
        }

        if (isOver(combineBehavior)) {
            if (fresnel && transWeight < 1.f) {
                // the remaining (unreflected) energy normally given to the
                // transmission lobe is to be divided up and some given to
                // subsequent lobes.  When this happens, we still need to
                // account for the light being reflected, therefore we need
                // a SimpleAttenuator
                SimpleAttenuator * atten = mTls->mArena->allocWithArgs<SimpleAttenuator>(
                        mTls->mArena,
                        adaptedNormal,
                        component.getRoughness(),
                        fresnel);

                stageAttenuator(atten);
            }

            // account for portion of energy given to transmission lobe
            mWeightAccum += transWeight;
        }

        if (refl) {
            refl->setLabel(reflectionLabel);
            mBsdf.addLobe(refl);
        }

        if (trans) {
            trans->setLabel(transmissionLabel);
            mBsdf.addLobe(trans);
        }
    }

    finline void
    addComponent(const MirrorBRDF& component,
                 float weight,
                 ispc::BsdfBuilderBehavior combineBehavior,
                 const int label)
    {
        // Adapt normal to prevent reflection ray from self-intersecting this geometry
        const scene_rdl2::math::Vec3f adaptedNormal = mState.adaptNormal(component.getN());

        BsdfLobe * lobe =
            mTls->mArena->allocWithArgs<MirrorReflectionBsdfLobe>(
                adaptedNormal);
        Fresnel * fresnel = applyFresnelBehavior(component.getEta(),
                                                 component.getK(),
                                                 component.isConductor(),
                                                 weight,
                                                 lobe);
        lobe = handleIridescence(lobe, component.getIridescence());

        scene_rdl2::math::Color scale = scene_rdl2::math::sWhite;

        if (isUnder(combineBehavior)) {
            // account for the non-dielectric lobes above
            scale *= mCurrentTransmittance;
            lobe->setScale(scale);
            // account for the dielectric/clearcoat lobes above
            lobe = placeUnderPreviousLobes(lobe);
        } else {
            lobe->setScale(scale);
        }

        if (isOver(combineBehavior)) {
            if (component.isConductor()) {
                // conductors don't transmit the energy, they reflect it or absorb it
                mWeightAccum += weight;
            } else {
                // dielectrics transmit any unreflected energy
                SimpleAttenuator * atten = mTls->mArena->allocWithArgs<SimpleAttenuator>(
                        mTls->mArena,
                        adaptedNormal,
                        0.f, // roughness
                        fresnel);

                stageAttenuator(atten);
            }
        }

        lobe->setLabel(label);
        mBsdf.addLobe(lobe);
    }

    finline void
    addComponent(const MirrorBTDF& component,
                 float weight,
                 ispc::BsdfBuilderBehavior combineBehavior,
                 const int label)
    {
        shading::ShaderIor ior(mState,
                               component.getEta(),
                               mIsThinGeo);
        BsdfLobe* lobe = mTls->mArena->allocWithArgs<MirrorTransmissionBsdfLobe>(
                component.getN(),
                ior.getIncident(),
                mIsThinGeo ? ior.getIncident() : ior.getTransmitted(),
                component.getTint(),
                component.getAbbeNumber());

        scene_rdl2::math::Color scale(weight);

        if (isUnder(combineBehavior)) {
            // account for the non-dielectric lobes above
            scale *= mCurrentTransmittance;
            lobe->setScale(scale);
            // account for the dielectric/clearcoat lobes above
            lobe = placeUnderPreviousLobes(lobe);
        } else {
            lobe->setScale(scale);
        }

        if (isOver(combineBehavior)) {
            // account for this lobe's energy allocation
            mWeightAccum += weight;
        }

        lobe->setLabel(label);
        mBsdf.addLobe(lobe);
    }

    finline void
    addComponent(const MirrorBSDF& component,
                 float weight,
                 ispc::BsdfBuilderBehavior combineBehavior,
                 const int reflectionLabel,
                 const int transmissionLabel)
    {
        const float reflWeight  = weight * component.getReflectionWeight();
        const float transWeight = weight * component.getTransmissionWeight();

        shading::ShaderIor ior(mState,
                               component.getRefractionEta(),
                               mIsThinGeo);

        BsdfLobe * refl = nullptr;
        BsdfLobe * trans = nullptr;

        // Adapt normal to prevent reflection ray from self-intersecting this geometry
        const scene_rdl2::math::Vec3f adaptedNormal = mState.adaptNormal(component.getN());

        if (reflWeight > 0.f) {
            refl = mTls->mArena->allocWithArgs<MirrorReflectionBsdfLobe>(
                    adaptedNormal);
        }

        if (transWeight > 0.f) {
            trans = mTls->mArena->allocWithArgs<MirrorTransmissionBsdfLobe>(
                    component.getN(),
                    ior.getIncident(),
                    mIsThinGeo ? ior.getIncident() : ior.getTransmitted(),
                    component.getTint(),
                    component.getAbbeNumber());
        }

        Fresnel * fresnel = nullptr;
        if (refl) {
            fresnel = applyFresnelBehavior(scene_rdl2::math::Color(component.getEta()),
                                           scene_rdl2::math::sBlack,    // k
                                           false,           // isConductor
                                           reflWeight,
                                           refl);

            refl = handleIridescence(refl, component.getIridescence());

            if (trans) {
                Fresnel * oneMinusFresnel =
                    mTls->mArena->allocWithArgs<OneMinusFresnel>(fresnel);
                trans->setFresnel(oneMinusFresnel);
            }
        }

        scene_rdl2::math::Color scale = scene_rdl2::math::sWhite;

        if (isUnder(combineBehavior)) {
            // account for the non-dielectric lobes above
            scale *= mCurrentTransmittance;
            // account for the dielectric/clearcoat lobes above
            if (refl) {
                refl->setScale(scale);
                refl = placeUnderPreviousLobes(refl);
            }
            if (trans) {
                trans->setScale(scale * transWeight);
                trans = placeUnderPreviousLobes(trans);
            }
        } else {
            if (refl) { refl->setScale(scale); }
            if (trans) { trans->setScale(scale * transWeight); }
        }

        if (isOver(combineBehavior)) {
            if (fresnel && transWeight < 1.f) {
                // the remaining (unreflected) energy normally given to the
                // transmission lobe is to be divided up and some given to
                // subsequent lobes.  When this happens, we still need to
                // account for the light being reflected, therefore we need
                // a SimpleAttenuator
                SimpleAttenuator * atten = mTls->mArena->allocWithArgs<SimpleAttenuator>(
                        mTls->mArena,
                        adaptedNormal,
                        0.f, // roughness
                        fresnel);

                stageAttenuator(atten);
            }

            // account for portion of energy given to transmission lobe
            mWeightAccum += transWeight;
        }

        if (refl) {
            refl->setLabel(reflectionLabel);
            mBsdf.addLobe(refl);
        }

        if (trans) {
            trans->setLabel(transmissionLabel);
            mBsdf.addLobe(trans);
        }
    }

    finline void
    addComponent(const GlitterFlakeBRDF& component,
                 float weight,
                 ispc::BsdfBuilderBehavior combineBehavior,
                 const int label)
    {
        // no cook-torrance compensation for glitter
        scene_rdl2::math::Color favg = scene_rdl2::math::sBlack;
        BsdfLobe* lobe = mTls->mArena->allocWithArgs<GlitterGGXCookTorranceBsdfLobe>(
                mState.adaptNormal(component.getN()),
                component.getFlakeN(),
                component.getRoughness(),
                favg);

        applyFresnelBehavior(component.getEta(),
                             component.getK(),
                             true, // isConductor
                             weight,
                             lobe);

        lobe = handleIridescence(lobe, component.getIridescence());

        scene_rdl2::math::Color scale = scene_rdl2::math::sWhite;

        if (isUnder(combineBehavior)) {
            // account for the non-dielectric lobes above
            scale *= mCurrentTransmittance;
            lobe->setScale(scale);
            // account for the dielectric/clearcoat lobes above
            lobe = placeUnderPreviousLobes(lobe);
        } else {
            lobe->setScale(scale);
        }

        if (isOver(combineBehavior)) {
            // conductors don't transmit the energy, they reflect it or absorb it
            mWeightAccum += weight;
        }

        lobe->setLabel(label);
        mBsdf.addLobe(lobe);
    }

    finline void
    addComponent(const HairBSDF& component,
                 float weight,
                 ispc::BsdfBuilderBehavior combineBehavior,
                 const int label)
    {
        float minRoughness = getMinRoughness();

        // Computing SigmaA in Builder for now
        // TODO - Move this inside the BSDF lobe.
        // Right now it's inefficient to compute inside the lobe since it'll
        // get computed 3x the way the lobes are currently designed.
        // This work can be a part of extending hair shading model to include a
        // near-field shading solution.
        const scene_rdl2::math::Color hairSigmaA =
            HairUtil::computeAbsorptionCoefficients(component.getHairColor(),
                                                    component.getAziRoughnessTT());

        BsdfLobe* lobe = mTls->mArena->allocWithArgs<HairOneSampleLobe>(
                component.getHairDir(),
                component.getHairUV(),
                mState.getMediumIor(),
                component.getIOR(),
                (ispc::HairFresnelType)component.getFresnelType(),
                component.getCuticleLayerThickness(),
                component.getShowR(),
                component.getShiftR(),
                clampHairRoughness(sHairRoughnessMin, minRoughness, component.getRoughnessR()),
                component.getTintR(),
                component.getShowTT(),
                component.getShiftTT(),
                clampHairRoughness(sHairRoughnessMin, minRoughness, component.getRoughnessTT()),
                clampHairRoughness(sHairRoughnessMin, minRoughness, component.getAziRoughnessTT()),
                component.getTintTT(),
                component.getSaturationTT(),
                component.getShowTRT(),
                component.getShiftTRT(),
                clampHairRoughness(sHairRoughnessMin, minRoughness, component.getRoughnessTRT()),
                component.getTintTRT(),
                component.getShowGlint(),
                clampHairRoughness(sHairGlintRoughnessMin, minRoughness, component.getRoughnessGlint()),
                component.getEccentricityGlint(),
                component.getSaturationGlint(),
                component.getHairRotation(),
                component.getHairNormal(),
                component.getShowTRRT(),
                component.getHairColor(),
                hairSigmaA);

        applyFresnelBehavior(scene_rdl2::math::Color(component.getIOR()),
                             scene_rdl2::math::sBlack,     // k
                             false,            // isConductor
                             weight,
                             lobe);

        scene_rdl2::math::Color scale = scene_rdl2::math::sWhite;

        if (isUnder(combineBehavior)) {
            // account for the non-dielectric lobes above
            scale *= mCurrentTransmittance;
            lobe->setScale(scale);
            // account for the dielectric/clearcoat lobes above
            lobe = placeUnderPreviousLobes(lobe);
        } else {
            lobe->setScale(scale);
        }

        if (isOver(combineBehavior)) {
            mWeightAccum += weight;
        }

        lobe->setLabel(label);
        mBsdf.addLobe(lobe);
    }

    finline void
    addComponent(const HairRBRDF& component,
                 float weight,
                 ispc::BsdfBuilderBehavior combineBehavior,
                 const int label)
    {
        BsdfLobe* lobe = mTls->mArena->allocWithArgs<HairRLobe>(
                component.getHairDir(),
                component.getHairUV(),
                mState.getMediumIor(),
                component.getIOR(),
                (ispc::HairFresnelType)component.getFresnelType(),
                component.getCuticleLayerThickness(),
                component.getShift(),
                clampHairRoughness(sHairRoughnessMin, getMinRoughness(), component.getRoughness()),
                component.getTint());

        Fresnel * fresnel = applyFresnelBehavior(scene_rdl2::math::Color(component.getIOR()),
                                                 scene_rdl2::math::sBlack,    // k
                                                 false,           // isConductor
                                                 weight,
                                                 lobe);

        scene_rdl2::math::Color scale(fresnel ? scene_rdl2::math::sWhite : scene_rdl2::math::Color(weight));

        if (isUnder(combineBehavior)) {
            // account for the non-dielectric lobes above
            scale *= mCurrentTransmittance;
            lobe->setScale(scale);
            // account for the dielectric/clearcoat lobes above
            lobe = placeUnderPreviousLobes(lobe);
        } else {
            lobe->setScale(scale);
        }

        if (isOver(combineBehavior)) {
            HairAttenuator * atten = mTls->mArena->allocWithArgs<HairAttenuator>(mTls->mArena,
                    fresnel);

            stageAttenuator(atten);
        }

        lobe->setLabel(label);
        mBsdf.addLobe(lobe);
    }

    finline void
    addComponent(const HairTRTBRDF& component,
                 float weight,
                 ispc::BsdfBuilderBehavior combineBehavior,
                 const int label)
    {
        const scene_rdl2::math::Color hairSigmaA =
            HairUtil::computeAbsorptionCoefficients(component.getHairColor(),
                                                    component.getAziRoughness());

        BsdfLobe* lobe = mTls->mArena->allocWithArgs<HairTRTLobe>(
                component.getHairDir(),
                component.getHairUV(),
                mState.getMediumIor(),
                component.getIOR(),
                (ispc::HairFresnelType)component.getFresnelType(),
                component.getCuticleLayerThickness(),
                component.getShift(),
                clampHairRoughness(sHairRoughnessMin, getMinRoughness(), component.getRoughness()),
                component.getHairColor(),
                hairSigmaA,
                component.getTint(),
                component.getShowGlint(),
                clampHairRoughness(sHairGlintRoughnessMin, getMinRoughness(), component.getRoughnessGlint()),
                component.getEccentricityGlint(),
                component.getSaturationGlint(),
                component.getHairRotation(),
                component.getHairNormal());

        scene_rdl2::math::Color scale(weight);

        if (isUnder(combineBehavior)) {
            // account for the non-dielectric lobes above
            scale *= mCurrentTransmittance;
            lobe->setScale(scale);
            // account for the dielectric/clearcoat lobes above
            lobe = placeUnderPreviousLobes(lobe);
        } else {
            lobe->setScale(scale);
        }

        if (isOver(combineBehavior)) {
            mWeightAccum += weight;
        }

        lobe->setLabel(label);
        mBsdf.addLobe(lobe);
    }

    finline void
    addComponent(const HairTTBTDF& component,
                 float weight,
                 ispc::BsdfBuilderBehavior combineBehavior,
                 const int label)
    {
        const scene_rdl2::math::Color hairSigmaA =
            HairUtil::computeAbsorptionCoefficients(component.getHairColor(),
                                                    component.getAziRoughness());

        float minRoughness = getMinRoughness();
        BsdfLobe* lobe = mTls->mArena->allocWithArgs<HairTTLobe>(
                component.getHairDir(),
                component.getHairUV(),
                mState.getMediumIor(),
                component.getIOR(),
                (ispc::HairFresnelType)component.getFresnelType(),
                component.getCuticleLayerThickness(),
                component.getShift(),
                clampHairRoughness(sHairRoughnessMin, minRoughness, component.getRoughness()),
                clampHairRoughness(sHairRoughnessMin, minRoughness, component.getAziRoughness()),
                component.getHairColor(),
                hairSigmaA,
                component.getTint(),
                component.getSaturation());

        scene_rdl2::math::Color scale(weight);

        if (isUnder(combineBehavior)) {
            // account for the non-dielectric lobes above
            scale *= mCurrentTransmittance;
            lobe->setScale(scale);
            // account for the dielectric/clearcoat lobes above
            lobe = placeUnderPreviousLobes(lobe);
        } else {
            lobe->setScale(scale);
        }

        if (isOver(combineBehavior)) {
            mWeightAccum += weight;
        }

        lobe->setLabel(label);
        mBsdf.addLobe(lobe);
    }

    finline void
    addComponent(const HairTRRTBRDF& component,
                 float weight,
                 ispc::BsdfBuilderBehavior combineBehavior,
                 const int label)
    {
        const scene_rdl2::math::Color hairSigmaA =
            HairUtil::computeAbsorptionCoefficients(component.getHairColor(),
                                                    component.getAziRoughness());

        BsdfLobe* lobe = mTls->mArena->allocWithArgs<HairTRRTLobe>(
                component.getHairDir(),
                component.getHairUV(),
                mState.getMediumIor(),
                component.getIOR(),
                (ispc::HairFresnelType)component.getFresnelType(),
                component.getCuticleLayerThickness(),
                clampHairRoughness(sHairRoughnessMin, getMinRoughness(), component.getRoughness()),
                component.getHairColor(),
                hairSigmaA,
                component.getTint());

        scene_rdl2::math::Color scale(weight);

        if (isUnder(combineBehavior)) {
            // account for the non-dielectric lobes above
            scale *= mCurrentTransmittance;
            lobe->setScale(scale);
            // account for the dielectric/clearcoat lobes above
            lobe = placeUnderPreviousLobes(lobe);
        } else {
            lobe->setScale(scale);
        }

        if (isOver(combineBehavior)) {
            mWeightAccum += weight;
        }

        lobe->setLabel(label);
        mBsdf.addLobe(lobe);
    }

    finline void
    addComponent(const HairDiffuseBSDF& component,
                 float weight,
                 ispc::BsdfBuilderBehavior combineBehavior,
                 const int label)
    {
        BsdfLobe* lobe = mTls->mArena->allocWithArgs<HairDiffuseLobe>(
                component.getHairDir(),
                component.getReflectionColor(),
                component.getTransmissionColor());

        scene_rdl2::math::Color scale(weight);

        if (isUnder(combineBehavior)) {
            // account for the non-dielectric lobes above
            scale *= mCurrentTransmittance;
            lobe->setScale(scale);
            // account for the dielectric/clearcoat lobes above
            lobe = placeUnderPreviousLobes(lobe);
        } else {
            lobe->setScale(scale);
        }

        if (isOver(combineBehavior)) {
            mWeightAccum += weight;
        }

        lobe->setLabel(label);
        mBsdf.addLobe(lobe);
    }

    finline void
    addComponent(const ToonBRDF& component,
                 float weight,
                 ispc::BsdfBuilderBehavior combineBehavior,
                 const int label)
    {
        BsdfLobe* lobe = mTls->mArena->allocWithArgs<ToonBsdfLobe>(
            component.getN(),
            component.getAlbedo(),
            component.getRampNumPoints(),
            component.getRampPositions(),
            component.getRampInterpolators(),
            component.getRampColors(),
            component.getExtendRamp());

        if (mPreventLightCulling) {
            // Force lobe to be "spherical" to prevent culling
            // when light sample is on backside
            const shading::BsdfLobe::Type lobeType = shading::BsdfLobe::ALL_DIFFUSE;
            lobe->setType(lobeType);
            lobe->setIsSpherical(true);
        }

        if (isUnder(combineBehavior)) {
            // account for the non-dielectric lobes above
            scene_rdl2::math::Color scale(weight * mCurrentTransmittance);
            lobe->setScale(scale);
            // account for the dielectric/clearcoat lobes above
            lobe = placeUnderPreviousLobes(lobe);
        }

        if (isOver(combineBehavior)) {
            // account for this lobe's energy allocation
            mWeightAccum += weight;
        }

        lobe->setLabel(label);
        mBsdf.addLobe(lobe);
    }

    finline void
    addComponent(const ToonSpecularBRDF& component,
                 float weight,
                 ispc::BsdfBuilderBehavior combineBehavior,
                 const int label)
    {
        const float cosNO = dot(component.getN(), mState.getWo());

        // It is common in practice for production to use ToonSpecularBRDF with
        // explicit normals on the geometry for highly-stylized looks, and sometimes
        // those normals can be backfacing to the observer (for example, when rendering
        // normal-oriented curves). To prevent shading artifacts we'll attempt to apply a
        // correction to bend the normal towards the observer similar to how we attempt
        // to prevent self-instersections with reflections when normal mapping is used.
        // In extreme cases where both the normal and geometric normal are facing nearly
        // 180 degrees away from the observer, the approach used below to correct the normal
        // is not adequate, and so we instead fade the specular lobe out and eventually
        // skip it altogether.

        scene_rdl2::math::Vec3f adaptedNormal(component.getN());
        float fadeWeight = 1.0f;

        if (cosNO < 0.0f) {
            // Fade off as normal approaches 180 degrees from the observer direction, and
            // don't add the lobe at all beyond some limit of stability.
            const float minCosNO = -0.9f;  // limit of stability, determined emperically
            fadeWeight = 1.0f - cosNO/minCosNO;

            if (fadeWeight < 0.0f) return; // skip adding lobe altogether

            const float epsilon = 0.001f;
            adaptedNormal = normalize(component.getN() + (epsilon - cosNO) * mState.getWo());
            MNRY_ASSERT(isNormalized(adaptedNormal));
        }

        // Create the fresnel early to set on GGX child lobe of ToonSpecularGGXBsdfLobe
        Fresnel * fresnel = createFresnel(scene_rdl2::math::Color(1.5f), // eta
                                          scene_rdl2::math::sBlack, // k
                                          false); // is conductor
        fresnel->setWeight(weight * fadeWeight);

        BsdfLobe * lobe = mTls->mArena->allocWithArgs<ToonSpecularBsdfLobe>(
                adaptedNormal,
                component.getIntensity() * fadeWeight * weight,
                component.getTint(),
                component.getRampInputScale(),
                component.getRampNumPoints(),
                component.getRampPositions(),
                component.getRampInterpolators(),
                component.getRampValues(),
                component.getStretchU(),
                component.getStretchV(),
                component.getdPds(),
                component.getdPdt(),
                component.getEnableIndirectReflections(),
                component.getIndirectReflectionsRoughness(),
                component.getIndirectReflectionsIntensity(),
                fresnel);

        if (isUnder(combineBehavior)) {
            // account for the non-dielectric lobes above
            const scene_rdl2::math::Color scale = scene_rdl2::math::sWhite * mCurrentTransmittance;
            lobe->setScale(scale);
            // account for the dielectric/clearcoat lobes above
            lobe = placeUnderPreviousLobes(lobe);
        }

        if (isOver(combineBehavior)) {
            // dielectrics transmit any unreflected energy
            SimpleAttenuator * atten = mTls->mArena->allocWithArgs<SimpleAttenuator>(
                    mTls->mArena,
                    adaptedNormal,
                    component.getIndirectReflectionsRoughness(),
                    fresnel);

            stageAttenuator(atten);
        }

        lobe->setLabel(label);
        mBsdf.addLobe(lobe);
    }

    finline void
    addComponent(const HairToonSpecularBRDF& component,
                 float weight,
                 ispc::BsdfBuilderBehavior combineBehavior,
                 const int label)
    {
        // Adapt normal to prevent reflection ray from self-intersecting this geometry
        const scene_rdl2::math::Vec3f adaptedNormal = mState.adaptNormal(component.getN());

        BsdfLobe * lobe = mTls->mArena->allocWithArgs<HairToonSpecularBsdfLobe>(
                    adaptedNormal,
                    component.getIntensity(),
                    component.getTint(),
                    component.getRampNumPoints(),
                    component.getRampPositions(),
                    component.getRampInterpolators(),
                    component.getRampValues(),
                    component.getEnableIndirectReflections(),
                    clampHairRoughness(sHairRoughnessMin,
                                       getMinRoughness(),
                                       component.getIndirectReflectionsRoughness()),
                    component.getIndirectReflectionsIntensity(),
                    component.getHairDir(),
                    component.getHairUV(),
                    mState.getMediumIor(),
                    component.getIOR(),
                    (ispc::HairFresnelType)component.getFresnelType(),
                    component.getCuticleLayerThickness(),
                    component.getShift(),
                    clampHairRoughness(sHairRoughnessMin,
                                       getMinRoughness(),
                                       component.getRoughness()));

        Fresnel * fresnel = applyFresnelBehavior(scene_rdl2::math::Color(1.5f), // eta
                                                 scene_rdl2::math::sBlack, // k
                                                 false, // is conductor
                                                 weight,
                                                 lobe);

        if (isUnder(combineBehavior)) {
            // account for the non-dielectric lobes above
            const scene_rdl2::math::Color scale = scene_rdl2::math::sWhite * mCurrentTransmittance;
            lobe->setScale(scale);
            // account for the dielectric/clearcoat lobes above
            lobe = placeUnderPreviousLobes(lobe);
        }

        if (isOver(combineBehavior)) {
            // dielectrics transmit any unreflected energy
            SimpleAttenuator * atten = mTls->mArena->allocWithArgs<SimpleAttenuator>(
                    mTls->mArena,
                    adaptedNormal,
                    component.getRoughness(),
                    fresnel);

            stageAttenuator(atten);
        }

        lobe->setLabel(label);
        mBsdf.addLobe(lobe);
    }

    finline void
    addComponent(const StochasticFlakesBRDF& component,
                 float weight,
                 ispc::BsdfBuilderBehavior combineBehavior,
                 const int label)
    {
        BsdfLobe* lobe =
            mTls->mArena->allocWithArgs<StochasticFlakesBsdfLobe>(
                    mState.adaptNormal(component.getN()),
                    component.getFlakeNormals(),
                    component.getFlakeColors(),
                    component.getFlakeCount(),
                    component.getFlakeRoughness(),
                    component.getFlakeRandomness());

        lobe = handleIridescence(lobe, component.getIridescence());

        scene_rdl2::math::Color scale(weight);

        if (isUnder(combineBehavior)) {
            // account for the non-dielectric lobes above
            scale *= mCurrentTransmittance;
            lobe->setScale(scale);
            // account for the dielectric/clearcoat lobes above
            lobe = placeUnderPreviousLobes(lobe);
        } else {
            lobe->setScale(scale);
        }

        if (isOver(combineBehavior)) {
            // account for this lobe's energy allocation
            mWeightAccum += weight;
        }

        lobe->setLabel(label);
        mBsdf.addLobe(lobe);
    }

    finline void
    addComponent(const VelvetBRDF& component,
                 float weight,
                 ispc::BsdfBuilderBehavior combineBehavior,
                 const int label)
    {
        BsdfLobe* lobe =
            mTls->mArena->allocWithArgs<FabricVelvetBsdfLobe>(
                    component.getN(),
                    component.getRoughness(),
                    component.getColor());

        lobe = handleIridescence(lobe, component.getIridescence());

        scene_rdl2::math::Color scale(weight);

        if (isUnder(combineBehavior)) {
            // account for the non-dielectric lobes above
            scale *= mCurrentTransmittance;
            lobe->setScale(scale);
            // account for the dielectric/clearcoat lobes above
            lobe = placeUnderPreviousLobes(lobe);
        } else {
            lobe->setScale(scale);
        }

        if (isOver(combineBehavior)) {
            VelvetAttenuator * atten = mTls->mArena->allocWithArgs<VelvetAttenuator>(
                    mTls->mArena,
                    component,
                    weight);
            stageAttenuator(atten);
        }

        lobe->setLabel(label);
        mBsdf.addLobe(lobe);
    }

    finline void
    addComponent(const FabricBRDF& component,
                 float weight,
                 ispc::BsdfBuilderBehavior combineBehavior,
                 const int label)
    {
        BsdfLobe* lobe = mTls->mArena->allocWithArgs<DwaFabricBsdfLobe>(
                component.getN(),
                component.getT(),
                component.getThreadDirection(),
                component.getThreadElevation(),
                component.getRoughness(),
                component.getFabricColor());

        lobe = handleIridescence(lobe, component.getIridescence());

        scene_rdl2::math::Color scale(weight);

        if (isUnder(combineBehavior)) {
            // account for the non-dielectric lobes above
            scale *= mCurrentTransmittance;
            lobe->setScale(scale);
            // account for the dielectric/clearcoat lobes above
            lobe = placeUnderPreviousLobes(lobe);
        } else {
            lobe->setScale(scale);
        }

        if (isOver(combineBehavior)) {
            // account for this lobe's energy allocation
            mWeightAccum += weight;
        }

        lobe->setLabel(label);
        mBsdf.addLobe(lobe);
    }

    void setThinGeo() { mIsThinGeo = true; }

    void setPreventLightCulling(bool isPrevented) { mPreventLightCulling = isPrevented; }

    const Bsdf*
    getBsdf() const
    {
        return &mBsdf;
    }

private:
    Bsdf&                  mBsdf;
    shading::TLState*      mTls;
    const State&           mState;

    const LobeAttenuator * mActiveAttenuators[BSDF_BUILDER_MAX_ATTENUATORS];
    const LobeAttenuator * mStagedAttenuators[BSDF_BUILDER_MAX_ATTENUATORS];
    size_t mActiveAttenuatorCount;
    size_t mStagedAttenuatorCount;

    Fresnel * mActiveHairFresnelAttenuatorChain;
    Fresnel * mStagedHairFresnelAttenuatorChain;

    float mCurrentMediumIor;
    float mCurrentTransmittance;
    float mWeightAccum;

    bool mIsThinGeo;
    bool mPreventLightCulling;
    bool mInAdjacentBlock;
};

BsdfBuilder::BsdfBuilder(Bsdf& bsdf,
                         shading::TLState *tls,
                         const State& state) :
    mImpl(tls->mArena->allocWithArgs<Impl>(bsdf, tls, state))
{}

BsdfBuilder::~BsdfBuilder()
{
    // unfortunately nothing to do, cannot free mImpl from the Arena
}

#define DEFINE_ADDCOMPONENT_1_LABEL(Type)                                   \
void                                                                        \
BsdfBuilder::add##Type(                                                     \
        const Type &component,                                              \
        float weight,                                                       \
        ispc::BsdfBuilderBehavior combineBehavior,                          \
        int label)                                                          \
{                                                                           \
    if (!mImpl->testForVisibility(weight, combineBehavior)) { return; }     \
    mImpl->addComponent(component, weight, combineBehavior, label);         \
    if (!mImpl->getInAdjacentBlock()) {                                     \
        mImpl->accumulateAttenuation();                                     \
    }                                                                       \
}

#define DEFINE_ADDCOMPONENT_2_LABELS(Type)                                  \
void                                                                        \
BsdfBuilder::add##Type(                                                     \
        const Type &component,                                              \
              float weight,                                                 \
              ispc::BsdfBuilderBehavior combineBehavior,                    \
              int label1,                                                   \
              int label2)                                                   \
{                                                                           \
    if (!mImpl->testForVisibility(weight, combineBehavior)) { return; }     \
    mImpl->addComponent(component, weight, combineBehavior, label1, label2);\
    if (!mImpl->getInAdjacentBlock()) {                                     \
        mImpl->accumulateAttenuation();                                     \
    }                                                                       \
}

DEFINE_ADDCOMPONENT_1_LABEL(MicrofacetAnisotropicClearcoat)
DEFINE_ADDCOMPONENT_1_LABEL(MicrofacetIsotropicClearcoat)
DEFINE_ADDCOMPONENT_1_LABEL(MirrorClearcoat)
DEFINE_ADDCOMPONENT_1_LABEL(DipoleDiffusion)
DEFINE_ADDCOMPONENT_1_LABEL(EyeCausticBRDF)
DEFINE_ADDCOMPONENT_1_LABEL(GlitterFlakeBRDF)
DEFINE_ADDCOMPONENT_1_LABEL(HairBSDF)
DEFINE_ADDCOMPONENT_1_LABEL(HairRBRDF)
DEFINE_ADDCOMPONENT_1_LABEL(HairTRTBRDF)
DEFINE_ADDCOMPONENT_1_LABEL(HairTTBTDF)
DEFINE_ADDCOMPONENT_1_LABEL(HairTRRTBRDF)
DEFINE_ADDCOMPONENT_1_LABEL(HairDiffuseBSDF)
DEFINE_ADDCOMPONENT_1_LABEL(LambertianBRDF)
DEFINE_ADDCOMPONENT_1_LABEL(LambertianBTDF)
DEFINE_ADDCOMPONENT_1_LABEL(FlatDiffuseBRDF)
DEFINE_ADDCOMPONENT_1_LABEL(OrenNayarBRDF)
DEFINE_ADDCOMPONENT_1_LABEL(NormalizedDiffusion)
DEFINE_ADDCOMPONENT_1_LABEL(RandomWalkSubsurface)
DEFINE_ADDCOMPONENT_1_LABEL(MicrofacetAnisotropicBRDF)
DEFINE_ADDCOMPONENT_1_LABEL(MicrofacetAnisotropicBTDF)
DEFINE_ADDCOMPONENT_2_LABELS(MicrofacetAnisotropicBSDF)
DEFINE_ADDCOMPONENT_1_LABEL(MicrofacetIsotropicBRDF)
DEFINE_ADDCOMPONENT_1_LABEL(MicrofacetIsotropicBTDF)
DEFINE_ADDCOMPONENT_2_LABELS(MicrofacetIsotropicBSDF)
DEFINE_ADDCOMPONENT_1_LABEL(MirrorBRDF)
DEFINE_ADDCOMPONENT_1_LABEL(MirrorBTDF)
DEFINE_ADDCOMPONENT_2_LABELS(MirrorBSDF)
DEFINE_ADDCOMPONENT_1_LABEL(StochasticFlakesBRDF)
DEFINE_ADDCOMPONENT_1_LABEL(VelvetBRDF)
DEFINE_ADDCOMPONENT_1_LABEL(FabricBRDF)
DEFINE_ADDCOMPONENT_1_LABEL(ToonBRDF)
DEFINE_ADDCOMPONENT_1_LABEL(ToonSpecularBRDF)
DEFINE_ADDCOMPONENT_1_LABEL(HairToonSpecularBRDF)

void
BsdfBuilder::startAdjacentComponents()
{
    mImpl->setInAdjacentBlock(true);
}

void
BsdfBuilder::endAdjacentComponents()
{
    mImpl->accumulateAttenuation();
    mImpl->setInAdjacentBlock(false);
}

void
BsdfBuilder::addEmission(const scene_rdl2::math::Color& emission)
{
    mImpl->addEmission(emission);
}

void
BsdfBuilder::setEarlyTermination()
{
    mImpl->setEarlyTermination();
}

void
BsdfBuilder::setThinGeo()
{
    mImpl->setThinGeo();
}

void
BsdfBuilder::setPreventLightCulling(bool isPrevented)
{
    mImpl->setPreventLightCulling(isPrevented);
}


const Bsdf*
BsdfBuilder::getBsdf() const
{
    return mImpl->getBsdf();
}

} // end namespace shading
} // end namespace moonray

