// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

/// @file BsdfBuilder.ispc

#include "BsdfBuilder.isph"
#include "BsdfComponent.isph"

#include "bsdf/Bsdf.isph"
#include "bsdf/BsdfEyeCaustic.isph"
#include "bsdf/BsdfIridescence.isph"
#include "bsdf/BsdfLambert.isph"
#include "bsdf/BsdfMirror.isph"
#include "bsdf/BsdfOrenNayar.isph"
#include "bsdf/cook_torrance/BsdfAnisoCookTorrance.isph"
#include "bsdf/cook_torrance/BsdfCookTorrance.isph"
#include "bsdf/cook_torrance/BsdfGGXCookTorrance.isph"
#include "bsdf/cook_torrance/BsdfGlitterGGXCookTorrance.isph"
#include "bsdf/cook_torrance/BsdfTransmissionCookTorrance.isph"
#include "bsdf/fabric/BsdfFabric.isph"
#include "bsdf/npr/BsdfToon.isph"
#include "bsdf/npr/BsdfFlatDiffuse.isph"
#include "bsdf/fabric/BsdfFabricVelvet.isph"
#include "bsdf/Fresnel.isph"
#include "bsdf/hair/BsdfHairDiffuse.isph"
#include "bsdf/hair/BsdfHairOneSampler.isph"
#include "bsdf/hair/HairUtil.isph"
#include "bsdf/BsdfStochasticFlakes.isph"
#include "bssrdf/Dipole.isph"
#include "bssrdf/NormalizedDiffusion.isph"
#include "bssrdf/VolumeSubsurface.isph"

#include <moonray/rendering/shading/ispc/Closure.isph>
#include <moonray/rendering/shading/ispc/Ior.isph>
#include <moonray/rendering/shading/ispc/LobeAttenuator.isph>

#include <moonray/rendering/bvh/shading/ispc/Intersection.isph>
#include <moonray/rendering/bvh/shading/ispc/ShadingTLState.isph>
#include <moonray/rendering/bvh/shading/ispc/State.isph>

// this limit is tied to the maximum number of lobes
// supported in Moonray. Keep in sync w/ scalar BsdfBuilder
#define BSDF_BUILDER_MAX_ATTENUATORS 16

#pragma ignore warning(all)
ISPC_UTIL_EXPORT_STRUCT_TO_HEADER(BsdfBuilder);
ISPC_UTIL_EXPORT_ENUM_TO_HEADER(BsdfBuilderBehavior);

// Hair seems unstable below 0.01
// but glints benefit from lower roughness
static const uniform float sHairRoughnessMin = 0.01f;
static const uniform float sHairGlintRoughnessMin = 0.001f;

struct BsdfBuilderImpl
{
    varying Bsdf            * uniform mBsdf;
    uniform ShadingTLState  * uniform mTls;
    const varying State     * uniform mState;

    varying const LobeAttenuator * uniform mActiveAttenuators[BSDF_BUILDER_MAX_ATTENUATORS];
    varying const LobeAttenuator * uniform mStagedAttenuators[BSDF_BUILDER_MAX_ATTENUATORS];
    uniform uint32_t mActiveAttenuatorCount;
    uniform uint32_t mStagedAttenuatorCount;

    varying float mCurrentMediumIor;
    varying float mCurrentTransmittance;
    varying float mWeightAccum;

    uniform bool mIsThinGeo;
    uniform bool mPreventLightCulling;
    uniform bool mInAdjacentBlock;
};

void
BsdfBuilderImpl_init(
        varying BsdfBuilderImpl& me,
        varying Bsdf& bsdf,
        uniform ShadingTLState& tls,
        const varying State &state)
{
    Bsdf_init(&bsdf);

    me.mBsdf    = &bsdf;
    me.mTls     = &tls;
    me.mState   = &state;

    memset(me.mActiveAttenuators, 0, BSDF_BUILDER_MAX_ATTENUATORS * sizeof(LobeAttenuator* uniform));
    memset(me.mStagedAttenuators, 0, BSDF_BUILDER_MAX_ATTENUATORS * sizeof(LobeAttenuator* uniform));
    me.mActiveAttenuatorCount = 0;
    me.mStagedAttenuatorCount = 0;

    me.mCurrentMediumIor = 1.0f;
    me.mCurrentTransmittance = 1.0f;
    me.mWeightAccum = 0.0f;

    me.mIsThinGeo = false;
    me.mPreventLightCulling = false;
    me.mInAdjacentBlock = false;
}

varying float
getMinRoughness(varying BsdfBuilderImpl& me)
{
    const varying Intersection& isect = asAnIntersection(*(me.mState));
    const varying Vec2f minRoughnessAniso = getMinRoughness(isect);
    return min(minRoughnessAniso.x, minRoughnessAniso.y);
}

varying float
clampHairRoughness(const uniform float minAttrRoughness,
                   const varying float minRoughness,
                   const varying float r)
{
    const varying float rMin = max(minAttrRoughness, minRoughness);
    const varying float rMax = 1.0f - sEpsilon;
    return clamp(r, rMin, rMax);
}

varying Fresnel * uniform
createFresnel(varying BsdfBuilderImpl& me,
              const varying Color& eta,
              const varying Color& k,
              const uniform bool isConductor,
              const varying float weight)
{
    if (isConductor) {
        // Currently, Moonray does not respect the "media IOR" for conductors.
        // Eventually we should compute *relative* IOR, using mCurrentMediumIor
        varying ConductorFresnel * uniform fresnel = (varying ConductorFresnel * uniform)
            Arena_alloc(me.mTls->mArena, sizeof(varying ConductorFresnel));
        ConductorFresnel_init(fresnel, eta, k, weight);

        return (varying Fresnel * uniform) fresnel;
    } else {
        varying ShaderIor ior;
        ShaderIor_init(*(me.mState), eta.r, &ior, me.mIsThinGeo);

        varying DielectricFresnel * uniform fresnel = (varying DielectricFresnel * uniform)
            Arena_alloc(me.mTls->mArena, sizeof(varying DielectricFresnel));
        DielectricFresnel_init(fresnel, ior.mIncident, ior.mTransmitted, weight);

        return (varying Fresnel * uniform) fresnel;
    }
}

varying Fresnel * uniform
applyFresnelBehavior(varying BsdfBuilderImpl& me,
                     const varying Color& eta,
                     const varying Color& k,
                     const uniform bool isConductor,
                     const varying float weight,
                     varying BsdfLobe * uniform lobe)
{
    MNRY_ASSERT(lobe);
    varying Fresnel * uniform fresnel =
        createFresnel(me, eta, k, isConductor, weight);

    BsdfLobe_setFresnel(lobe, fresnel);

    return fresnel;
}

varying BsdfLobe * uniform
handleIridescence(varying BsdfBuilderImpl& me,
                  varying BsdfLobe * uniform lobe,
                  const varying Iridescence * const uniform iridescence)
{
    MNRY_ASSERT(lobe);
    if (!iridescence) { return lobe; }

    varying IridescenceBsdfLobe * uniform ilobe = (varying IridescenceBsdfLobe * uniform)
        Arena_alloc(me.mTls->mArena, sizeof(varying IridescenceBsdfLobe));
    IridescenceBsdfLobe_init(ilobe,
                             me.mTls->mArena,
                             iridescence->mN,
                             iridescence->mStrength,
                             iridescence->mColorControl,
                             iridescence->mPrimary,
                             iridescence->mSecondary,
                             iridescence->mFlipHue,
                             iridescence->mRampInterpolationMode,
                             iridescence->mRampNumPoints,
                             iridescence->mRampPositions,
                             iridescence->mRampInterpolators,
                             iridescence->mRampColors,
                             iridescence->mThickness,
                             iridescence->mExponent,
                             iridescence->mIridescenceAt0,
                             iridescence->mIridescenceAt90);

    IridescenceBsdfLobe_setChildLobe(ilobe, lobe);

    return (varying BsdfLobe * uniform) ilobe;
}

void
stageAttenuator(varying BsdfBuilderImpl& me,
                const varying LobeAttenuator * uniform attenuator)
{
    MNRY_ASSERT(attenuator);
    MNRY_ASSERT(me.mStagedAttenuatorCount < (BSDF_BUILDER_MAX_ATTENUATORS - 1));

    me.mStagedAttenuators[me.mStagedAttenuatorCount] = attenuator;
    ++me.mStagedAttenuatorCount;
}

void
accumulateAttenuation(varying BsdfBuilderImpl& me)
{
    if (me.mInAdjacentBlock) {
        // exit early
        return;
    }

    MNRY_ASSERT((me.mActiveAttenuatorCount + me.mStagedAttenuatorCount) <= BSDF_BUILDER_MAX_ATTENUATORS);

    // Append any staged attenuators to the active list for subsequent BsdfLobes
    for (uniform uint32_t i = 0; i < me.mStagedAttenuatorCount; ++i) {
        me.mActiveAttenuators[me.mActiveAttenuatorCount] = me.mStagedAttenuators[i];
        ++me.mActiveAttenuatorCount;
    }

    // "Reset" the staging area (so to speak)
    me.mStagedAttenuatorCount = 0;

    // apply the recently accumulated weight to transmittance for
    // attenuating subsequent lobes
    me.mCurrentTransmittance *= (1.0f - me.mWeightAccum);
    me.mWeightAccum = 0.f;
}

inline varying bool
isUnder(varying BsdfBuilderBehavior combineBehavior)
{
    return (combineBehavior & BSDFBUILDER_UNDER_PREVIOUS);
}

inline varying bool
isOver(varying BsdfBuilderBehavior combineBehavior)
{
    return (combineBehavior & BSDFBUILDER_OVER_SUBSEQUENT);
}

// check to see if a new lobe would potentially be visible
inline bool
testForVisibility(const varying BsdfBuilderImpl& me,
                  const varying float weight,
                  const varying BsdfBuilderBehavior combineBehavior)
{
    if (weight < sEpsilon) { return false; }
    return (!isUnder(combineBehavior) || me.mCurrentTransmittance > 0.f);
}

varying BsdfLobe * uniform
placeUnderPreviousLobes(varying BsdfBuilderImpl& me,
                        varying BsdfLobe * uniform lobe)
{
    MNRY_ASSERT(lobe);

    if (me.mActiveAttenuatorCount > 0) {
        uniform int idx = me.mActiveAttenuatorCount - 1;
        while (idx >= 0) {
            const varying LobeAttenuator * uniform attenuator = me.mActiveAttenuators[idx--];
            lobe = LobeAttenuator_attenuate(attenuator, me.mTls->mArena, lobe);
        }
    }

    return lobe;
}

void
placeUnderPreviousLobes(varying BsdfBuilderImpl& me,
                        varying Bssrdf * uniform bssrdf)
{
    MNRY_ASSERT(bssrdf);

    if (me.mActiveAttenuatorCount > 0) {
        uniform int idx = me.mActiveAttenuatorCount - 1;
        while (idx >= 0) {
            const varying LobeAttenuator * uniform attenuator = me.mActiveAttenuators[idx--];
            LobeAttenuator_attenuate(attenuator, me.mTls->mArena, bssrdf);
        }
    }
}

void
placeUnderPreviousLobes(varying BsdfBuilderImpl& me,
                        varying VolumeSubsurface * uniform volume)
{
    MNRY_ASSERT(volume);

    if (me.mActiveAttenuatorCount > 0) {
        uniform int idx = me.mActiveAttenuatorCount - 1;
        while (idx >= 0) {
            const varying LobeAttenuator * uniform attenuator = me.mActiveAttenuators[idx--];
            LobeAttenuator_attenuate(attenuator, me.mTls->mArena, volume);
        }
    }
}

void
BsdfBuilder_init(varying BsdfBuilder& me,
                 varying Bsdf& bsdf,
                 uniform ShadingTLState& tls,
                 const varying State &state)
{
    me.mImpl = (varying BsdfBuilderImpl * uniform)
        Arena_alloc(tls.mArena, sizeof(varying BsdfBuilderImpl));

    BsdfBuilderImpl_init(*me.mImpl, bsdf, tls, state);
}

void
BsdfBuilder_addMicrofacetAnisotropicClearcoat(
        varying BsdfBuilder& builder,
        const varying MicrofacetAnisotropicClearcoat& brdf,
        const varying float weight,
        const varying BsdfBuilderBehavior combineBehavior,
        const uniform int label)
{
    varying BsdfBuilderImpl& me = *builder.mImpl;

    cif (!testForVisibility(me, weight, combineBehavior)) { return; }

    // Adapt normal to prevent reflection ray from self-intersecting this geometry
    const varying Vec3f adaptedNormal =
        Intersection_adaptNormal(asAnIntersection(*(me.mState)), brdf.mN);

    // We currently only support anisotropy with Beckmann
    varying BsdfLobe * uniform lobe = (varying BsdfLobe * uniform)
        Arena_alloc(me.mTls->mArena, sizeof(varying AnisoCookTorranceBsdfLobe));
    AnisoCookTorranceBsdfLobe_init(
            (varying AnisoCookTorranceBsdfLobe * uniform) lobe,
            adaptedNormal,
            brdf.mShadingTangent,
            brdf.mRoughnessU,
            brdf.mRoughnessV);

    varying Fresnel * uniform fresnel =
        applyFresnelBehavior(me,
                             Color_ctor(brdf.mEta),
                             (const varying Color) sBlack, // k
                             false, // isConductor
                             weight,
                             lobe);

    lobe = handleIridescence(me, lobe, brdf.mIridescence);

    cif (isUnder(combineBehavior)) {
        // account for the non-dielectric lobes above
        const varying Color scale = sWhite * me.mCurrentTransmittance;
        BsdfLobe_setScale(lobe, scale);
        // account for dielectric/clearcoat lobes above
        lobe = placeUnderPreviousLobes(me, lobe);
    }

    cif (isOver(combineBehavior)) {
        varying ClearcoatAttenuator * uniform atten = (varying ClearcoatAttenuator * uniform)
            Arena_alloc(me.mTls->mArena, sizeof(varying ClearcoatAttenuator));
        ClearcoatAttenuator_init(atten,
                                 me.mTls->mArena,
                                 adaptedNormal,
                                 (brdf.mRoughnessU + brdf.mRoughnessV) * 0.5f, // average
                                 me.mCurrentMediumIor,
                                 brdf.mEta,
                                 brdf.mRefracts,
                                 brdf.mThickness,
                                 brdf.mAttenuationColor,
                                 weight, // The attenuation effect diminishes with weight
                                 fresnel);
        stageAttenuator(me, (varying LobeAttenuator * uniform) atten);
    }

    BsdfLobe_setLabel(lobe, label);
    Bsdf_addLobe(me.mBsdf, lobe);
    accumulateAttenuation(me);
}

void
BsdfBuilder_addMicrofacetIsotropicClearcoat(
        varying BsdfBuilder& builder,
        const varying MicrofacetIsotropicClearcoat& brdf,
        const varying float weight,
        const varying BsdfBuilderBehavior combineBehavior,
        const uniform int label)
{
    varying BsdfBuilderImpl& me = *builder.mImpl;

    cif (!testForVisibility(me, weight, combineBehavior)) { return; }

    // Adapt normal to prevent reflection ray from self-intersecting this geometry
    const varying Vec3f adaptedNormal =
        Intersection_adaptNormal(asAnIntersection(*(me.mState)), brdf.mN);

    varying BsdfLobe * uniform lobe;

    if (brdf.mMicrofacetDistribution == MICROFACET_DISTRIBUTION_BECKMANN) {
        lobe = (varying BsdfLobe * uniform)
            Arena_alloc(me.mTls->mArena, sizeof(varying CookTorranceBsdfLobe));
        CookTorranceBsdfLobe_init(
                (varying CookTorranceBsdfLobe * uniform) lobe,
                adaptedNormal,
                brdf.mRoughness);
    } else {    // MICROFACET_DISTRIBUTION_GGX
        lobe = (varying BsdfLobe * uniform)
            Arena_alloc(me.mTls->mArena, sizeof(varying GGXCookTorranceBsdfLobe));
        GGXCookTorranceBsdfLobe_init(
                (varying GGXCookTorranceBsdfLobe * uniform) lobe,
                adaptedNormal,
                brdf.mRoughness);
    }

    varying Fresnel * uniform fresnel =
        applyFresnelBehavior(me,
                             Color_ctor(brdf.mEta),
                             (const varying Color) sBlack, // k
                             false, // isConductor
                             weight,
                             lobe);

    lobe = handleIridescence(me, lobe, brdf.mIridescence);

    cif (isUnder(combineBehavior)) {
        // account for the non-dielectric lobes above
        const varying Color scale = sWhite * me.mCurrentTransmittance;
        BsdfLobe_setScale(lobe, scale);
        // account for dielectric/clearcoat lobes above
        lobe = placeUnderPreviousLobes(me, lobe);
    }

    cif (isOver(combineBehavior)) {
        varying ClearcoatAttenuator * uniform atten = (varying ClearcoatAttenuator * uniform)
            Arena_alloc(me.mTls->mArena, sizeof(varying ClearcoatAttenuator));
        ClearcoatAttenuator_init(atten,
                                 me.mTls->mArena,
                                 adaptedNormal,
                                 brdf.mRoughness,
                                 me.mCurrentMediumIor,
                                 brdf.mEta,
                                 brdf.mRefracts,
                                 brdf.mThickness,
                                 brdf.mAttenuationColor,
                                 weight, // The attenuation effect diminishes with weight
                                 fresnel);
        stageAttenuator(me, (varying LobeAttenuator * uniform) atten);
    }

    BsdfLobe_setLabel(lobe, label);
    Bsdf_addLobe(me.mBsdf, lobe);
    accumulateAttenuation(me);
}

void
BsdfBuilder_addMirrorClearcoat(
        varying BsdfBuilder& builder,
        const varying MirrorClearcoat& brdf,
        const varying float weight,
        const varying BsdfBuilderBehavior combineBehavior,
        const uniform int label)
{
    varying BsdfBuilderImpl& me = *builder.mImpl;

    cif (!testForVisibility(me, weight, combineBehavior)) { return; }

    // Adapt normal to prevent reflection ray from self-intersecting this geometry
    const varying Vec3f adaptedNormal =
        Intersection_adaptNormal(asAnIntersection(*(me.mState)), brdf.mN);

    varying BsdfLobe * uniform lobe = (varying BsdfLobe * uniform)
        Arena_alloc(me.mTls->mArena, sizeof(varying MirrorReflectionBsdfLobe));
    MirrorReflectionBsdfLobe_init((varying MirrorReflectionBsdfLobe * uniform) lobe, adaptedNormal);

    varying Fresnel * uniform fresnel =
        applyFresnelBehavior(me,
                             Color_ctor(brdf.mEta),
                             (const varying Color) sBlack, // k
                             false, // isConductor
                             weight,
                             lobe);

    lobe = handleIridescence(me, lobe, brdf.mIridescence);

    cif (isUnder(combineBehavior)) {
        // account for the non-dielectric lobes above
        const varying Color scale = sWhite * me.mCurrentTransmittance;
        BsdfLobe_setScale(lobe, scale);
        // account for dielectric/clearcoat lobes above
        lobe = placeUnderPreviousLobes(me, lobe);
    }

    cif (isOver(combineBehavior)) {
        varying ClearcoatAttenuator * uniform atten = (varying ClearcoatAttenuator * uniform)
            Arena_alloc(me.mTls->mArena, sizeof(varying ClearcoatAttenuator));
        ClearcoatAttenuator_init(atten,
                                 me.mTls->mArena,
                                 adaptedNormal,
                                 0.0f, // roughness
                                 me.mCurrentMediumIor,
                                 brdf.mEta,
                                 brdf.mRefracts,
                                 brdf.mThickness,
                                 brdf.mAttenuationColor,
                                 weight, // The attenuation effect diminishes with weight
                                 fresnel);
        stageAttenuator(me, (varying LobeAttenuator * uniform) atten);
    }

    BsdfLobe_setLabel(lobe, label);
    Bsdf_addLobe(me.mBsdf, lobe);
    accumulateAttenuation(me);
}

void
BsdfBuilder_addMirrorBRDF(
        varying BsdfBuilder& builder,
        const varying MirrorBRDF& brdf,
        const varying float weight,
        const varying BsdfBuilderBehavior combineBehavior,
        const uniform int label)
{
    varying BsdfBuilderImpl& me = *builder.mImpl;

    cif (!testForVisibility(me, weight, combineBehavior)) { return; }

    // Adapt normal to prevent reflection ray from self-intersecting this geometry
    const varying Vec3f adaptedNormal =
        Intersection_adaptNormal(asAnIntersection(*(me.mState)), brdf.mN);

    varying BsdfLobe * uniform lobe = (varying BsdfLobe * uniform)
        Arena_alloc(me.mTls->mArena, sizeof(varying MirrorReflectionBsdfLobe));
    MirrorReflectionBsdfLobe_init((varying MirrorReflectionBsdfLobe * uniform) lobe, adaptedNormal);

    varying Fresnel * uniform fresnel =
        applyFresnelBehavior(me,
                             brdf.mEta,
                             brdf.mK,
                             brdf.mIsConductor,
                             weight,
                             lobe);

    lobe = handleIridescence(me, lobe, brdf.mIridescence);

    cif (isUnder(combineBehavior)) {
        // account for the non-dielectric lobes above
        const varying Color scale = sWhite * me.mCurrentTransmittance;
        BsdfLobe_setScale(lobe, scale);
        // account for the dielectric/clearcoat lobes above
        lobe = placeUnderPreviousLobes(me, lobe);
    }

    cif (isOver(combineBehavior)) {
        if (brdf.mIsConductor) {
            // conductors don't transmit the energy, they reflect it or absorb it
            me.mWeightAccum += weight;
        } else {
            // dielectrics transmit any unreflected energy
            varying SimpleAttenuator * uniform atten = (varying SimpleAttenuator * uniform)
                Arena_alloc(me.mTls->mArena, sizeof(varying SimpleAttenuator));
            SimpleAttenuator_init(atten,
                                  me.mTls->mArena,
                                  adaptedNormal,
                                  0.0f, // roughness
                                  fresnel);

            stageAttenuator(me, (varying LobeAttenuator * uniform) atten);
        }
    }

    BsdfLobe_setLabel(lobe, label);
    Bsdf_addLobe(me.mBsdf, lobe);
    accumulateAttenuation(me);
}

void
BsdfBuilder_addMirrorBTDF(
        varying BsdfBuilder& builder,
        const varying MirrorBTDF& btdf,
        const varying float weight,
        const varying BsdfBuilderBehavior combineBehavior,
        const uniform int label)
{
    varying BsdfBuilderImpl& me = *builder.mImpl;

    cif (!testForVisibility(me, weight, combineBehavior)) { return; }

    varying ShaderIor ior;
    ShaderIor_init(*(me.mState), btdf.mEta, &ior, me.mIsThinGeo);

    varying BsdfLobe * uniform lobe = (varying BsdfLobe * uniform)
        Arena_alloc(me.mTls->mArena, sizeof(varying MirrorTransmissionBsdfLobe));
    MirrorTransmissionBsdfLobe_init((varying MirrorTransmissionBsdfLobe * uniform) lobe,
                                    btdf.mN,
                                    ior.mIncident,
                                    me.mIsThinGeo ? ior.mIncident : ior.mTransmitted,
                                    btdf.mTint,
                                    btdf.mAbbeNumber);

    varying Color scale = Color_ctor(weight);

    cif (isUnder(combineBehavior)) {
        // account for the non-dielectric lobes above
        scale = scale * me.mCurrentTransmittance;
        BsdfLobe_setScale(lobe, scale);
        // account for the dielectric/clearcoat lobes above
        lobe = placeUnderPreviousLobes(me, lobe);
    } else {
        BsdfLobe_setScale(lobe, scale);
    }

    cif (isOver(combineBehavior)) {
        // account for this lobe's energy allocation
        me.mWeightAccum += weight;
    }

    BsdfLobe_setLabel(lobe, label);
    Bsdf_addLobe(me.mBsdf, lobe);
    accumulateAttenuation(me);
}

void
BsdfBuilder_addMirrorBSDF(
        varying BsdfBuilder& builder,
        const varying MirrorBSDF& bsdf,
        const varying float weight,
        const varying BsdfBuilderBehavior combineBehavior,
        const uniform int reflectionLabel,
        const uniform int transmissionLabel)
{
    varying BsdfBuilderImpl& me = *builder.mImpl;

    cif (!testForVisibility(me, weight, combineBehavior)) { return; }

    const varying float reflWeight  = weight * bsdf.mReflectionWeight;
    const varying float transWeight = weight * bsdf.mTransmissionWeight;

    varying ShaderIor ior;
    ShaderIor_init(*(me.mState), bsdf.mRefractionEta, &ior, me.mIsThinGeo);

    varying BsdfLobe * uniform refl = nullptr;
    varying BsdfLobe * uniform trans = nullptr;

    // Adapt normal to prevent reflection ray from self-intersecting this geometry
    const varying Vec3f adaptedNormal =
        Intersection_adaptNormal(asAnIntersection(*(me.mState)), bsdf.mN);

    if (reflWeight > 0.f) {
        refl = (varying BsdfLobe * uniform)
            Arena_alloc(me.mTls->mArena, sizeof(varying MirrorReflectionBsdfLobe));
        MirrorReflectionBsdfLobe_init(
                (varying MirrorReflectionBsdfLobe * uniform) refl,
                adaptedNormal);
    }

    if (transWeight > 0.f) {
        trans = (varying BsdfLobe * uniform)
            Arena_alloc(me.mTls->mArena, sizeof(varying MirrorTransmissionBsdfLobe));
        MirrorTransmissionBsdfLobe_init(
                (varying MirrorTransmissionBsdfLobe * uniform) trans,
                bsdf.mN,
                ior.mIncident,
                me.mIsThinGeo ? ior.mIncident : ior.mTransmitted,
                bsdf.mTint,
                bsdf.mAbbeNumber);
    }

    varying Fresnel * uniform fresnel = nullptr;
    if (refl) {
        fresnel = applyFresnelBehavior(
                me,
                Color_ctor(bsdf.mEta),
                (varying Color) sBlack, // k
                false,                  // isConductor
                reflWeight,
                refl);

        refl = handleIridescence(me, refl, bsdf.mIridescence);

        if (trans) {
            varying Fresnel * uniform oneMinusFresnel = (varying Fresnel * uniform)
                Arena_alloc(me.mTls->mArena, sizeof(varying OneMinusFresnel));
            OneMinusFresnel_init((varying OneMinusFresnel * uniform) oneMinusFresnel, fresnel);

            BsdfLobe_setFresnel(trans, oneMinusFresnel);
        }
    }

    cif (isUnder(combineBehavior)) {
        // account for the non-dielectric lobes above
        const varying Color scale = sWhite * me.mCurrentTransmittance;
        // account for the dielectric/clearcoat lobes above
        if (refl) {
            BsdfLobe_setScale(refl, scale);
            refl = placeUnderPreviousLobes(me, refl);
        }
        if (trans) {
            BsdfLobe_setScale(trans, scale * transWeight);
            trans = placeUnderPreviousLobes(me, trans);
        }
    }

    cif (isOver(combineBehavior)) {

        if (fresnel && transWeight < 1.f) {
            // the remaining (unreflected) energy normally given to the
            // transmission lobe is to be divided up and some given to
            // subsequent lobes.  When this happens, we still need to
            // account for the light being reflected, therefore we need
            // a SimpleAttenuator
            varying SimpleAttenuator * uniform atten = (varying SimpleAttenuator * uniform)
                Arena_alloc(me.mTls->mArena, sizeof(varying SimpleAttenuator));
            SimpleAttenuator_init(atten,
                                  me.mTls->mArena,
                                  adaptedNormal,
                                  0.0f, // roughness,
                                  fresnel);

            stageAttenuator(me, (varying LobeAttenuator * uniform) atten);
        }

        // account for portion of energy given to transmission lobe
        me.mWeightAccum += transWeight;
    }

    if (refl) {
        BsdfLobe_setLabel(refl, reflectionLabel);
        Bsdf_addLobe(me.mBsdf, refl);
    }

    if (trans) {
        BsdfLobe_setLabel(trans, transmissionLabel);
        Bsdf_addLobe(me.mBsdf, trans);
    }

    accumulateAttenuation(me);
}

void
BsdfBuilder_addMicrofacetAnisotropicBRDF(
        varying BsdfBuilder& builder,
        const varying MicrofacetAnisotropicBRDF& brdf,
        const varying float weight,
        const varying BsdfBuilderBehavior combineBehavior,
        const uniform int label)
{
    varying BsdfBuilderImpl& me = *builder.mImpl;

    cif (!testForVisibility(me, weight, combineBehavior)) { return; }

    // Adapt normal to prevent reflection ray from self-intersecting this geometry
    const varying Vec3f adaptedNormal =
        Intersection_adaptNormal(asAnIntersection(*(me.mState)), brdf.mN);

    varying BsdfLobe * uniform lobe = (varying BsdfLobe * uniform)
        Arena_alloc(me.mTls->mArena, sizeof(varying AnisoCookTorranceBsdfLobe));

    AnisoCookTorranceBsdfLobe_init(
            (varying AnisoCookTorranceBsdfLobe * uniform) lobe,
            adaptedNormal,
            brdf.mShadingTangent,
            brdf.mRoughnessU,
            brdf.mRoughnessV);

    varying Fresnel * uniform fresnel =
        applyFresnelBehavior(me,
                             brdf.mEta,
                             brdf.mK,
                             brdf.mIsConductor,
                             weight,
                             lobe);

    lobe = handleIridescence(me, lobe, brdf.mIridescence);

    cif (isUnder(combineBehavior)) {
        // account for the non-dielectric lobes above
        varying Color scale = sWhite * me.mCurrentTransmittance;
        BsdfLobe_setScale(lobe, scale);
        // account for the dielectric/clearcoat lobes above
        lobe = placeUnderPreviousLobes(me, lobe);
    }

    cif (isOver(combineBehavior)) {
        if (brdf.mIsConductor) {
            // conductors don't transmit the energy, they reflect it or absorb it
            me.mWeightAccum += weight;
        } else {
            // dielectrics transmit any unreflected energy
            varying SimpleAttenuator * uniform atten = (varying SimpleAttenuator * uniform)
                Arena_alloc(me.mTls->mArena, sizeof(varying SimpleAttenuator));
            SimpleAttenuator_init(atten,
                                  me.mTls->mArena,
                                  adaptedNormal,
                                  (brdf.mRoughnessU + brdf.mRoughnessV) * 0.5f, // average
                                  fresnel);

            stageAttenuator(me, (varying LobeAttenuator * uniform) atten);
        }
    }

    BsdfLobe_setLabel(lobe, label);
    Bsdf_addLobe(me.mBsdf, lobe);
    accumulateAttenuation(me);
}

void
BsdfBuilder_addMicrofacetIsotropicBRDF(
        varying BsdfBuilder& builder,
        const varying MicrofacetIsotropicBRDF& brdf,
        const varying float weight,
        const varying BsdfBuilderBehavior combineBehavior,
        const uniform int label)
{
    varying BsdfBuilderImpl& me = *builder.mImpl;

    cif (!testForVisibility(me, weight, combineBehavior)) { return; }

    // Adapt normal to prevent reflection ray from self-intersecting this geometry
    const varying Vec3f adaptedNormal =
        Intersection_adaptNormal(asAnIntersection(*(me.mState)), brdf.mN);

    varying BsdfLobe * uniform lobe;
    Color favgInv = sBlack;
    if (brdf.mMicrofacetDistribution == MICROFACET_DISTRIBUTION_BECKMANN) {
        varying CookTorranceBsdfLobe * uniform beckmann = (varying CookTorranceBsdfLobe * uniform)
            Arena_alloc(me.mTls->mArena, sizeof(varying CookTorranceBsdfLobe));
        CookTorranceBsdfLobe_init(beckmann,
                                  adaptedNormal,
                                  brdf.mRoughness,
                                  brdf.mFavg, favgInv);
        lobe = (varying BsdfLobe * uniform) beckmann;
    } else {    // MICROFACET_DISTRIBUTION_GGX
        varying GGXCookTorranceBsdfLobe * uniform ggx = (varying GGXCookTorranceBsdfLobe * uniform)
            Arena_alloc(me.mTls->mArena, sizeof(varying GGXCookTorranceBsdfLobe));
        GGXCookTorranceBsdfLobe_init(ggx,
                                     adaptedNormal,
                                     brdf.mRoughness,
                                     brdf.mFavg, favgInv);
        lobe = (varying BsdfLobe * uniform) ggx;
    }

    varying Fresnel * uniform fresnel =
        applyFresnelBehavior(me,
                             brdf.mEta,
                             brdf.mK,
                             brdf.mIsConductor,
                             weight,
                             lobe);

    lobe = handleIridescence(me, lobe, brdf.mIridescence);

    cif (isUnder(combineBehavior)) {
        // account for the non-dielectric lobes above
        const varying Color scale = sWhite * me.mCurrentTransmittance;
        BsdfLobe_setScale(lobe, scale);
        // account for the dielectric/clearcoat lobes above
        lobe = placeUnderPreviousLobes(me, lobe);
    }

    cif (isOver(combineBehavior)) {
        if (brdf.mIsConductor) {
            // conductors don't transmit the energy, they reflect it or absorb it
            me.mWeightAccum += weight;
        } else {
            // dielectrics transmit any unreflected energy
            varying SimpleAttenuator * uniform atten = (varying SimpleAttenuator * uniform)
                Arena_alloc(me.mTls->mArena, sizeof(varying SimpleAttenuator));
            SimpleAttenuator_init(atten,
                                  me.mTls->mArena,
                                  adaptedNormal,
                                  brdf.mRoughness,
                                  fresnel);

            stageAttenuator(me, (varying LobeAttenuator * uniform) atten);
        }
    }

    BsdfLobe_setLabel(lobe, label);
    Bsdf_addLobe(me.mBsdf, lobe);
    accumulateAttenuation(me);
}

void
BsdfBuilder_addMicrofacetAnisotropicBTDF(
        varying BsdfBuilder& builder,
        const varying MicrofacetAnisotropicBTDF& btdf,
        const varying float weight,
        const varying BsdfBuilderBehavior combineBehavior,
        const uniform int label)
{
    varying BsdfBuilderImpl& me = *builder.mImpl;

    cif (!testForVisibility(me, weight, combineBehavior)) { return; }

    varying ShaderIor ior;
    ShaderIor_init(*(me.mState), btdf.mEta, &ior, me.mIsThinGeo);
    float favg, favgInv;
    averageFresnelReflectance(ior.mTransmitted/ior.mIncident,
                              favg, favgInv);
    // TODO: MOONSHINE-1035
    // Relative IOR == 1 is not handled and neither is thinGeometry

    // only isotropic microfacet transmission supported currently
    varying BsdfLobe * uniform lobe = (varying BsdfLobe * uniform)
        Arena_alloc(me.mTls->mArena, sizeof(varying TransmissionCookTorranceBsdfLobe));
    TransmissionCookTorranceBsdfLobe_init((varying TransmissionCookTorranceBsdfLobe * uniform) lobe,
                                          btdf.mN,
                                          (btdf.mRoughnessU + btdf.mRoughnessV) * 0.5f, // average
                                          ior.mIncident,
                                          ior.mTransmitted,
                                          btdf.mTint,
                                          favg, favgInv,
                                          btdf.mAbbeNumber);

    varying Color scale = Color_ctor(weight);

    cif (isUnder(combineBehavior)) {
        // account for the non-dielectric lobes above
        scale = scale * me.mCurrentTransmittance;
        BsdfLobe_setScale(lobe, scale);
        // account for the dielectric/clearcoat lobes above
        lobe = placeUnderPreviousLobes(me, lobe);
    } else {
        BsdfLobe_setScale(lobe, scale);
    }

    cif (isOver(combineBehavior)) {
        // account for this lobe's energy allocation
        me.mWeightAccum += weight;
    }

    BsdfLobe_setLabel(lobe, label);
    Bsdf_addLobe(me.mBsdf, lobe);
    accumulateAttenuation(me);
}

void
BsdfBuilder_addMicrofacetIsotropicBTDF(
        varying BsdfBuilder& builder,
        const varying MicrofacetIsotropicBTDF& btdf,
        const varying float weight,
        const varying BsdfBuilderBehavior combineBehavior,
        const uniform int label)
{
    varying BsdfBuilderImpl& me = *builder.mImpl;

    cif (!testForVisibility(me, weight, combineBehavior)) { return; }

    varying ShaderIor ior;
    ShaderIor_init(*(me.mState), btdf.mEta, &ior, me.mIsThinGeo);
    float favg, favgInv;
    averageFresnelReflectance(ior.mTransmitted/ior.mIncident,
                              favg, favgInv);
    // TODO: MOONSHINE-1035
    // Relative IOR == 1 is not handled and neither is thinGeometry

    varying BsdfLobe * uniform lobe = (varying BsdfLobe * uniform)
        Arena_alloc(me.mTls->mArena, sizeof(varying TransmissionCookTorranceBsdfLobe));
    TransmissionCookTorranceBsdfLobe_init((varying TransmissionCookTorranceBsdfLobe * uniform) lobe,
                                          btdf.mN,
                                          btdf.mRoughness,
                                          ior.mIncident,
                                          ior.mTransmitted,
                                          btdf.mTint,
                                          favg, favgInv,
                                          btdf.mAbbeNumber);

    varying Color scale = Color_ctor(weight);

    cif (isUnder(combineBehavior)) {
        // account for the non-dielectric lobes above
        scale = scale * me.mCurrentTransmittance;
        BsdfLobe_setScale(lobe, scale);
        // account for the dielectric/clearcoat lobes above
        lobe = placeUnderPreviousLobes(me, lobe);
    } else {
        BsdfLobe_setScale(lobe, scale);
    }

    cif (isOver(combineBehavior)) {
        // account for this lobe's energy allocation
        me.mWeightAccum += weight;
    }

    BsdfLobe_setLabel(lobe, label);
    Bsdf_addLobe(me.mBsdf, lobe);
    accumulateAttenuation(me);
}

void
BsdfBuilder_addMicrofacetAnisotropicBSDF(
        varying BsdfBuilder& builder,
        const varying MicrofacetAnisotropicBSDF& bsdf,
        const varying float weight,
        const varying BsdfBuilderBehavior combineBehavior,
        const uniform int reflectionLabel,
        const uniform int transmissionLabel)
{
    varying BsdfBuilderImpl& me = *builder.mImpl;

    cif (!testForVisibility(me, weight, combineBehavior)) { return; }

    const varying float reflWeight  = weight * bsdf.mReflectionWeight;
    const varying float transWeight = weight * bsdf.mTransmissionWeight;

    varying ShaderIor ior;
    ShaderIor_init(*(me.mState), bsdf.mRefractionEta, &ior, me.mIsThinGeo);

    varying BsdfLobe * uniform refl = nullptr;
    varying BsdfLobe * uniform trans = nullptr;

    // Adapt normal to prevent reflection ray from self-intersecting this geometry
    const varying Vec3f adaptedNormal =
        Intersection_adaptNormal(asAnIntersection(*(me.mState)), bsdf.mN);

    if (reflWeight > 0.f) {
        refl = (varying BsdfLobe * uniform)
            Arena_alloc(me.mTls->mArena, sizeof(varying AnisoCookTorranceBsdfLobe));
        AnisoCookTorranceBsdfLobe_init(
                (varying AnisoCookTorranceBsdfLobe * uniform) refl,
                adaptedNormal,
                bsdf.mShadingTangent,
                bsdf.mRoughnessU,
                bsdf.mRoughnessV);
    }

    const varying float avgRoughness = (bsdf.mRoughnessU + bsdf.mRoughnessV) * 0.5f;
    if (transWeight > 0.0f) {
        trans = (varying BsdfLobe * uniform)
            Arena_alloc(me.mTls->mArena, sizeof(varying TransmissionCookTorranceBsdfLobe));
        float favg, favgInv;
        averageFresnelReflectance(ior.mTransmitted/ior.mIncident,
                                  favg, favgInv);
        // TODO: MOONSHINE-1035
        // Relative IOR == 1 is not handled and neither is thinGeometry

        TransmissionCookTorranceBsdfLobe_init(
                (varying TransmissionCookTorranceBsdfLobe * uniform) trans,
                bsdf.mN,
                avgRoughness,
                ior.mIncident,
                ior.mTransmitted,
                bsdf.mTint,
                favg, favgInv,
                bsdf.mAbbeNumber);
    }

    varying Fresnel * uniform fresnel = nullptr;
    if (refl) {
        fresnel = applyFresnelBehavior(
                me,
                Color_ctor(bsdf.mEta),
                (varying Color) sBlack, // k
                false,                  // isConductor
                reflWeight,
                refl);

        refl = handleIridescence(me, refl, bsdf.mIridescence);

        if (trans) {
            varying Fresnel * uniform oneMinusFresnel = (varying Fresnel * uniform)
                Arena_alloc(me.mTls->mArena, sizeof(varying OneMinusFresnel));
            OneMinusFresnel_init((varying OneMinusFresnel * uniform) oneMinusFresnel,
                                 fresnel);

            BsdfLobe_setFresnel(trans, oneMinusFresnel);
        }
    }

    cif (isUnder(combineBehavior)) {
        // account for the non-dielectric lobes above
        const varying Color scale = sWhite * me.mCurrentTransmittance;
        // account for the dielectric/clearcoat lobes above
        if (refl) {
            BsdfLobe_setScale(refl, scale);
            refl = placeUnderPreviousLobes(me, refl);
        }
        if (trans) {
            BsdfLobe_setScale(trans, scale * transWeight);
            trans = placeUnderPreviousLobes(me, trans);
        }
    }

    cif (isOver(combineBehavior)) {
        if (fresnel && transWeight < 1.f) {
            // the remaining (unreflected) energy normally given to the
            // transmission lobe is to be divided up and some given to
            // subsequent lobes.  When this happens, we still need to
            // account for the light being reflected, therefore we need
            // a SimpleAttenuator
            varying SimpleAttenuator * uniform atten = (varying SimpleAttenuator * uniform)
                Arena_alloc(me.mTls->mArena, sizeof(varying SimpleAttenuator));
            SimpleAttenuator_init(atten,
                                  me.mTls->mArena,
                                  adaptedNormal,
                                  avgRoughness,
                                  fresnel);

            stageAttenuator(me, (varying LobeAttenuator * uniform) atten);
        }

        // account for portion of energy given to transmission lobe
        me.mWeightAccum += transWeight;
    }

    if (refl) {
        BsdfLobe_setLabel(refl, reflectionLabel);
        Bsdf_addLobe(me.mBsdf, refl);
    }

    if (trans) {
        BsdfLobe_setLabel(trans, transmissionLabel);
        Bsdf_addLobe(me.mBsdf, trans);
    }

    accumulateAttenuation(me);
}

void
BsdfBuilder_addMicrofacetIsotropicBSDF(
        varying BsdfBuilder& builder,
        const varying MicrofacetIsotropicBSDF& bsdf,
        const varying float weight,
        const varying BsdfBuilderBehavior combineBehavior,
        const uniform int reflectionLabel,
        const uniform int transmissionLabel)
{
    varying BsdfBuilderImpl& me = *builder.mImpl;

    cif (!testForVisibility(me, weight, combineBehavior)) { return; }

    const varying float reflWeight  = weight * bsdf.mReflectionWeight;
    const varying float transWeight = weight * bsdf.mTransmissionWeight;

    const bool isCoupledWithTransmission = !isZero(transWeight);

    varying ShaderIor ior;
    ShaderIor_init(*(me.mState), bsdf.mRefractionEta, &ior, me.mIsThinGeo);

    float favg, favgInv;
    averageFresnelReflectance(ior.mTransmitted/ior.mIncident,
                              favg, favgInv);

    varying BsdfLobe * uniform refl = nullptr;
    varying BsdfLobe * uniform trans = nullptr;

    // Adapt normal to prevent reflection ray from self-intersecting this geometry
    const varying Vec3f adaptedNormal =
        Intersection_adaptNormal(asAnIntersection(*(me.mState)), bsdf.mN);

    if (reflWeight > 0.f) {
        if (bsdf.mMicrofacetDistribution == MICROFACET_DISTRIBUTION_BECKMANN) {
            refl = (varying BsdfLobe * uniform)
                Arena_alloc(me.mTls->mArena, sizeof(varying CookTorranceBsdfLobe));
            CookTorranceBsdfLobe_init(
                    (varying CookTorranceBsdfLobe * uniform) refl,
                    adaptedNormal,
                    bsdf.mRoughness,
                    sWhite*favg,
                    sWhite*favgInv,
                    ior.mIncident,
                    ior.mTransmitted,
                    isCoupledWithTransmission);
        } else {    // MICROFACET_DISTRIBUTION_GGX
            refl = (varying BsdfLobe * uniform)
                Arena_alloc(me.mTls->mArena, sizeof(varying GGXCookTorranceBsdfLobe));
            GGXCookTorranceBsdfLobe_init(
                    (varying GGXCookTorranceBsdfLobe * uniform) refl,
                    adaptedNormal,
                    bsdf.mRoughness,
                    sWhite*favg,
                    sWhite*favgInv,
                    ior.mIncident,
                    ior.mTransmitted,
                    isCoupledWithTransmission);
        }
    }

    if (transWeight > 0.f) {
        // Because the TransmissionCookTorranceBsdfLobe becomes unstable as the
        // relative IOR approaches 1.0 we create a mirror transmission lobe here.
        // Using any() here will likely introduce non-determinism and potentially
        // visual artifacts, but we can't set a uniform pointer to two different
        // values based on a varying conditional.
        // see MOONSHINE-1035
        // I'd like to remove this check and just do something simple, like detect
        // when the ratio is 1.0 and adjust one of the IORs accordingly so it is not.
        if (any(isOne(ior.mRatio)) || me.mIsThinGeo) {
            trans = (varying BsdfLobe * uniform)
                Arena_alloc(me.mTls->mArena, sizeof(varying MirrorTransmissionBsdfLobe));
            MirrorTransmissionBsdfLobe_init(
                    (varying MirrorTransmissionBsdfLobe * uniform) trans,
                    bsdf.mN,
                    ior.mIncident,
                    me.mIsThinGeo ? ior.mIncident : ior.mTransmitted,
                    bsdf.mTint,
                    bsdf.mAbbeNumber);
        } else {
            trans = (varying BsdfLobe * uniform)
                Arena_alloc(me.mTls->mArena, sizeof(varying TransmissionCookTorranceBsdfLobe));
            TransmissionCookTorranceBsdfLobe_init(
                    (varying TransmissionCookTorranceBsdfLobe * uniform) trans,
                    bsdf.mN,
                    bsdf.mRoughness,
                    ior.mIncident,
                    ior.mTransmitted,
                    bsdf.mTint,
                    favg, favgInv,
                    bsdf.mAbbeNumber);
        }
    }

    varying Fresnel * uniform fresnel = nullptr;
    if (refl) {
        fresnel = applyFresnelBehavior(
                me,
                Color_ctor(bsdf.mEta),
                (varying Color) sBlack, // k
                false,                  // isConductor
                reflWeight,
                refl);

        refl = handleIridescence(me, refl, bsdf.mIridescence);

        if (trans) {
            varying Fresnel * uniform oneMinusFresnel = (varying Fresnel * uniform)
                Arena_alloc(me.mTls->mArena, sizeof(varying OneMinusFresnel));
            OneMinusFresnel_init((varying OneMinusFresnel * uniform) oneMinusFresnel,
                                 fresnel);

            BsdfLobe_setFresnel(trans, oneMinusFresnel);
        }
    }

    cif (isUnder(combineBehavior)) {
        // account for the non-dielectric lobes above
        const varying Color scale = sWhite * me.mCurrentTransmittance;
        // account for the dielectric/clearcoat lobes above
        if (refl) {
            BsdfLobe_setScale(refl, scale);
            refl = placeUnderPreviousLobes(me, refl);
        }
        if (trans) {
            BsdfLobe_setScale(trans, scale * transWeight);
            trans = placeUnderPreviousLobes(me, trans);
        }
    }

    cif (isOver(combineBehavior)) {
        if (fresnel && transWeight < 1.f) {
            // the remaining (unreflected) energy normally given to the
            // transmission lobe is to be divided up and some given to
            // subsequent lobes.  When this happens, we still need to
            // account for the light being reflected, therefore we need
            // a SimpleAttenuator
            varying SimpleAttenuator * uniform atten = (varying SimpleAttenuator * uniform)
                Arena_alloc(me.mTls->mArena, sizeof(varying SimpleAttenuator));
            SimpleAttenuator_init(atten,
                                  me.mTls->mArena,
                                  adaptedNormal,
                                  bsdf.mRoughness,
                                  fresnel);

            stageAttenuator(me, (varying LobeAttenuator * uniform) atten);
        }

        // account for portion of energy given to transmission lobe
        me.mWeightAccum += transWeight;
    }

    if (refl) {
        BsdfLobe_setLabel(refl, reflectionLabel);
        Bsdf_addLobe(me.mBsdf, refl);
    }

    if (trans) {
        BsdfLobe_setLabel(trans, transmissionLabel);
        Bsdf_addLobe(me.mBsdf, trans);
    }

    accumulateAttenuation(me);
}

void
BsdfBuilder_addToonBRDF(
        varying BsdfBuilder& builder,
        const varying ToonBRDF& brdf,
        const varying float weight,
        const varying BsdfBuilderBehavior combineBehavior,
        const uniform int label)
{
    varying BsdfBuilderImpl& me = *builder.mImpl;

    cif (!testForVisibility(me, weight, combineBehavior)) { return; }

    varying BsdfLobe * uniform lobe = (varying BsdfLobe * uniform)
        Arena_alloc(me.mTls->mArena, sizeof(varying ToonBsdfLobe));

    ToonBsdfLobe_init((varying ToonBsdfLobe * uniform) lobe,
                      me.mTls->mArena,
                      brdf.mN,
                      brdf.mAlbedo,
                      brdf.mRampNumPoints,
                      brdf.mRampPositions,
                      brdf.mRampInterpolators,
                      brdf.mRampColors,
                      brdf.mExtendRamp);

    if (me.mPreventLightCulling) {
        // Force lobe to be "spherical" to prevent culling
        // when light sample is on backside
        const uniform BsdfLobeType lobeType = BSDF_LOBE_TYPE_ALL_DIFFUSE;
        BsdfLobe_setType(lobe, lobeType);
        BsdfLobe_setIsSpherical(lobe, true);
    }

    varying Color scale = Color_ctor(weight);

    cif (isUnder(combineBehavior)) {
        // account for the non-dielectric lobes above
        scale = scale * me.mCurrentTransmittance;
        BsdfLobe_setScale(lobe, scale);
        // account for the dielectric/clearcoat lobes above
        lobe = placeUnderPreviousLobes(me, lobe);
    } else {
        BsdfLobe_setScale(lobe, scale);
    }

    cif (isOver(combineBehavior)) {
        // account for this lobe's energy allocation
        me.mWeightAccum += weight;
    }

    BsdfLobe_setLabel(lobe, label);
    Bsdf_addLobe(me.mBsdf, lobe);
    accumulateAttenuation(me);
}

void
BsdfBuilder_addHairToonSpecularBRDF(
        varying BsdfBuilder& builder,
        const varying HairToonSpecularBRDF& brdf,
        const varying float weight,
        const varying BsdfBuilderBehavior combineBehavior,
        const uniform int label)
{
    varying BsdfBuilderImpl& me = *builder.mImpl;

    cif (!testForVisibility(me, weight, combineBehavior)) { return; }

    // Adapt normal to prevent reflection ray from self-intersecting this geometry
    const varying Vec3f adaptedNormal =
        Intersection_adaptNormal(asAnIntersection(*(me.mState)), brdf.mN);

    varying BsdfLobe * uniform lobe = (varying BsdfLobe * uniform)
        Arena_alloc(me.mTls->mArena, sizeof(varying HairToonSpecularBsdfLobe));

    HairToonSpecularBsdfLobe_init((varying HairToonSpecularBsdfLobe * uniform)lobe,
                                  me.mTls->mArena,
                                  brdf.mN,
                                  brdf.mIntensity,
                                  brdf.mTint,
                                  brdf.mRampNumPoints,
                                  brdf.mRampPositions,
                                  brdf.mRampInterpolators,
                                  brdf.mRampValues,
                                  brdf.mEnableIndirectReflections,
                                  clampHairRoughness(sHairRoughnessMin, getMinRoughness(me), brdf.mIndirectReflectionsRoughness),
                                  brdf.mIndirectReflectionsIntensity,
                                  brdf.mHairDir,
                                  brdf.mHairUV,
                                  getMediumIor(*me.mState),
                                  brdf.mIOR,
                                  brdf.mFresnelType,
                                  brdf.mCuticleLayerThickness,
                                  brdf.mShift,
                                  clampHairRoughness(sHairRoughnessMin, getMinRoughness(me), brdf.mRoughness));

    varying Fresnel * uniform fresnel =
        applyFresnelBehavior(me,
                             (varying Color)Color_ctor(1.5f), // eta
                             (varying Color) sBlack, // k
                             false,                  // isConductor
                             weight,
                             lobe);

    cif (isUnder(combineBehavior)) {
        // account for the non-dielectric lobes above
        const varying Color scale = sWhite * me.mCurrentTransmittance;
        BsdfLobe_setScale(lobe, scale);
        // account for the dielectric/clearcoat lobes above
        lobe = placeUnderPreviousLobes(me, lobe);
    }

    cif (isOver(combineBehavior)) {
        // dielectrics transmit any unreflected energy
        varying SimpleAttenuator * uniform atten = (varying SimpleAttenuator * uniform)
            Arena_alloc(me.mTls->mArena, sizeof(varying SimpleAttenuator));
        SimpleAttenuator_init(atten,
                              me.mTls->mArena,
                              adaptedNormal,
                              brdf.mRoughness,
                              fresnel);

        stageAttenuator(me, (varying LobeAttenuator * uniform) atten);
    }

    BsdfLobe_setLabel(lobe, label);
    Bsdf_addLobe(me.mBsdf, lobe);
    accumulateAttenuation(me);
}

void
BsdfBuilder_addToonSpecularBRDF(
        varying BsdfBuilder& builder,
        const varying ToonSpecularBRDF& brdf,
        const varying float weight,
        const varying BsdfBuilderBehavior combineBehavior,
        const uniform int label)
{
    varying BsdfBuilderImpl& me = *builder.mImpl;
    const varying State& state = *(me.mState);

    cif (!testForVisibility(me, weight, combineBehavior)) { return; }

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

    const varying float cosNO = dot(brdf.mN, state.mWo);

    varying Vec3f adaptedNormal = brdf.mN;
    varying float fadeWeight = 1.0f;

    // Adapt normal to prevent reflection ray from self-intersecting this geometry.
    if (cosNO < 0.0f) {
        // Fade off as normal approaches 180 degrees from the observer direction, and
        // don't add the lobe at all beyond the empirical limit of stability.
        const varying float minCosNsO = -0.9f;  // limit of stability
        fadeWeight = 1.0f - cosNO/minCosNsO;

        if (fadeWeight < 0.0f) return;

        const varying float epsilon = 0.001f;
        adaptedNormal = normalize(brdf.mN + (epsilon - cosNO) * state.mWo);
        MNRY_ASSERT(isNormalized(adaptedNormal));
    }

    varying Fresnel * uniform fresnel =
        createFresnel(me,
                     (varying Color) Color_ctor(1.5f), // eta
                     (varying Color) sBlack, // k
                     false, // isConductor
                     weight * fadeWeight);

    varying BsdfLobe * uniform lobe = (varying BsdfLobe * uniform)
        Arena_alloc(me.mTls->mArena, sizeof(varying ToonSpecularBsdfLobe));

    ToonSpecularBsdfLobe_init((varying ToonSpecularBsdfLobe * uniform) lobe,
                              me.mTls->mArena,
                              adaptedNormal,
                              brdf.mIntensity * fadeWeight * weight,
                              brdf.mTint,
                              brdf.mRampInputScale,
                              brdf.mRampNumPoints,
                              brdf.mRampPositions,
                              brdf.mRampInterpolators,
                              brdf.mRampValues,
                              brdf.mStretchU,
                              brdf.mStretchV,
                              brdf.mdPds,
                              brdf.mdPdt,
                              brdf.mEnableIndirectReflections,
                              brdf.mIndirectReflectionsRoughness,
                              brdf.mIndirectReflectionsIntensity,
                              fresnel);

    cif (isUnder(combineBehavior)) {
        // account for the non-dielectric lobes above
        const varying Color scale = sWhite * me.mCurrentTransmittance;
        BsdfLobe_setScale(lobe, scale);
        // account for the dielectric/clearcoat lobes above
        lobe = placeUnderPreviousLobes(me, lobe);
    }

    cif (isOver(combineBehavior)) {
        // dielectrics transmit any unreflected energy
        varying SimpleAttenuator * uniform atten = (varying SimpleAttenuator * uniform)
            Arena_alloc(me.mTls->mArena, sizeof(varying SimpleAttenuator));
        SimpleAttenuator_init(atten,
                              me.mTls->mArena,
                              adaptedNormal,
                              brdf.mIndirectReflectionsRoughness,
                              fresnel);

        stageAttenuator(me, (varying LobeAttenuator * uniform) atten);
    }

    BsdfLobe_setLabel(lobe, label);
    Bsdf_addLobe(me.mBsdf, lobe);
    accumulateAttenuation(me);
}

void
BsdfBuilder_addLambertianBRDF(
        varying BsdfBuilder& builder,
        const varying LambertianBRDF& brdf,
        const varying float weight,
        const varying BsdfBuilderBehavior combineBehavior,
        const uniform int label)
{
    varying BsdfBuilderImpl& me = *builder.mImpl;

    cif (!testForVisibility(me, weight, combineBehavior)) { return; }

    varying BsdfLobe * uniform lobe = (varying BsdfLobe * uniform)
        Arena_alloc(me.mTls->mArena, sizeof(varying LambertBsdfLobe));

    LambertBsdfLobe_init((varying LambertBsdfLobe * uniform) lobe,
                         brdf.mN,
                         brdf.mAlbedo,
                         true); // isReflection

    if (me.mPreventLightCulling) {
        // Force lobe to be "spherical" to prevent culling
        // when light sample is on backside
        const uniform BsdfLobeType lobeType = BSDF_LOBE_TYPE_ALL_DIFFUSE;
        BsdfLobe_setType(lobe, lobeType);
        BsdfLobe_setIsSpherical(lobe, true);
    }

    varying Color scale = Color_ctor(weight);

    cif (isUnder(combineBehavior)) {
        // account for the non-dielectric lobes above
        scale = scale * me.mCurrentTransmittance;
        BsdfLobe_setScale(lobe, scale);
        // account for the dielectric/clearcoat lobes above
        lobe = placeUnderPreviousLobes(me, lobe);
    } else {
        BsdfLobe_setScale(lobe, scale);
    }

    cif (isOver(combineBehavior)) {
        // account for this lobe's energy allocation
        me.mWeightAccum += weight;
    }

    BsdfLobe_setLabel(lobe, label);
    Bsdf_addLobe(me.mBsdf, lobe);
    accumulateAttenuation(me);
}

void
BsdfBuilder_addLambertianBTDF(
        varying BsdfBuilder& builder,
        const varying LambertianBTDF& btdf,
        const varying float weight,
        const varying BsdfBuilderBehavior combineBehavior,
        const uniform int label)
{
    varying BsdfBuilderImpl& me = *builder.mImpl;

    cif (!testForVisibility(me, weight, combineBehavior)) { return; }

    varying BsdfLobe * uniform lobe = (varying BsdfLobe * uniform)
        Arena_alloc(me.mTls->mArena, sizeof(varying LambertBsdfLobe));

    LambertBsdfLobe_init((varying LambertBsdfLobe * uniform) lobe,
                         btdf.mN,
                         btdf.mTint,
                         false); // true for reflection, false for transmission

    varying Color scale = Color_ctor(weight);

    cif (isUnder(combineBehavior)) {
        // account for the non-dielectric lobes above
        scale = scale * me.mCurrentTransmittance;
        BsdfLobe_setScale(lobe, scale);
        // account for the dielectric/clearcoat lobes above
        lobe = placeUnderPreviousLobes(me, lobe);
    } else {
        BsdfLobe_setScale(lobe, scale);
    }

    cif (isOver(combineBehavior)) {
        // account for this lobe's energy allocation
        me.mWeightAccum += weight;
    }

    BsdfLobe_setLabel(lobe, label);
    Bsdf_addLobe(me.mBsdf, lobe);
    accumulateAttenuation(me);
}

void
BsdfBuilder_addFlatDiffuseBRDF(
        varying BsdfBuilder& builder,
        const varying FlatDiffuseBRDF& brdf,
        const varying float weight,
        const varying BsdfBuilderBehavior combineBehavior,
        const uniform int label)
{
    varying BsdfBuilderImpl& me = *builder.mImpl;

    cif (!testForVisibility(me, weight, combineBehavior)) { return; }

    varying BsdfLobe * uniform lobe = (varying BsdfLobe * uniform)
        Arena_alloc(me.mTls->mArena, sizeof(varying FlatDiffuseBsdfLobe));

    FlatDiffuseBsdfLobe_init((varying FlatDiffuseBsdfLobe * uniform) lobe,
                             brdf.mN,
                             brdf.mAlbedo,
                             brdf.mRoughness,
                             brdf.mTerminatorShift,
                             brdf.mFlatness,
                             brdf.mFlatnessFalloff,
                             true); // true for reflection, false for transmission

    if (me.mPreventLightCulling) {
        // Force lobe to be "spherical" to prevent culling
        // when light sample is on backside
        const uniform BsdfLobeType lobeType = BSDF_LOBE_TYPE_ALL_DIFFUSE;
        BsdfLobe_setType(lobe, lobeType);
        BsdfLobe_setIsSpherical(lobe, true);
    }

    varying Color scale = Color_ctor(weight);

    cif (isUnder(combineBehavior)) {
        // account for the non-dielectric lobes above
        scale = scale * me.mCurrentTransmittance;
        BsdfLobe_setScale(lobe, scale);
        // account for the dielectric/clearcoat lobes above
        lobe = placeUnderPreviousLobes(me, lobe);
    } else {
        BsdfLobe_setScale(lobe, scale);
    }

    cif (isOver(combineBehavior)) {
        // account for this lobe's energy allocation
        me.mWeightAccum += weight;
    }

    BsdfLobe_setLabel(lobe, label);
    Bsdf_addLobe(me.mBsdf, lobe);
    accumulateAttenuation(me);
}

void
BsdfBuilder_addOrenNayarBRDF(
        varying BsdfBuilder& builder,
        const varying OrenNayarBRDF& brdf,
        const varying float weight,
        const varying BsdfBuilderBehavior combineBehavior,
        const uniform int label)
{
    varying BsdfBuilderImpl& me = *builder.mImpl;

    cif (!testForVisibility(me, weight, combineBehavior)) { return; }

    varying BsdfLobe * uniform lobe = (varying BsdfLobe * uniform)
        Arena_alloc(me.mTls->mArena, sizeof(varying OrenNayarBsdfLobe));

    OrenNayarBsdfLobe_init((varying OrenNayarBsdfLobe * uniform) lobe,
                           brdf.mN,
                           brdf.mAlbedo,
                           brdf.mRoughness,
                           true); // isReflection

    if (me.mPreventLightCulling) {
        // Force lobe to be "spherical" to prevent culling
        // when light sample is on backside
        const uniform BsdfLobeType lobeType = BSDF_LOBE_TYPE_ALL_DIFFUSE;
        BsdfLobe_setType(lobe, lobeType);
        BsdfLobe_setIsSpherical(lobe, true);
    }

    varying Color scale = Color_ctor(weight);

    cif (isUnder(combineBehavior)) {
        // account for the non-dielectric lobes above
        scale = scale * me.mCurrentTransmittance;
        BsdfLobe_setScale(lobe, scale);
        // account for the dielectric/clearcoat lobes above
        lobe = placeUnderPreviousLobes(me, lobe);
    } else {
        BsdfLobe_setScale(lobe, scale);
    }

    cif (isOver(combineBehavior)) {
        // account for this lobe's energy allocation
        me.mWeightAccum += weight;
    }

    BsdfLobe_setLabel(lobe, label);
    Bsdf_addLobe(me.mBsdf, lobe);
    accumulateAttenuation(me);
}

void
BsdfBuilder_addDipoleDiffusion(
        varying BsdfBuilder& builder,
        const varying DipoleDiffusion& profile,
        const varying float weight,
        const varying BsdfBuilderBehavior combineBehavior,
        const uniform int label)
{
    varying BsdfBuilderImpl& me = *builder.mImpl;

    cif (!testForVisibility(me, weight, combineBehavior)) { return; }

    if (!isSubsurfaceAllowed(*me.mState)) {
        LambertianBRDF diffuse;
        LambertianBRDF_init(diffuse, profile.mN, profile.mAlbedo);

        BsdfBuilder_addLambertianBRDF(builder,
                                      diffuse,
                                      weight,
                                      combineBehavior,
                                      label);
        return;
    }

    varying Bssrdf * uniform bssrdf = (varying Bssrdf * uniform)
        Arena_alloc(me.mTls->mArena, sizeof(varying DipoleBssrdf));

    DipoleBssrdf_init((varying DipoleBssrdf * uniform) bssrdf,
                      profile.mN,
                      profile.mAlbedo,
                      profile.mRadius,
                      profile.mMaterial,
                      profile.mEvalNormalFn);

    varying Color scale = Color_ctor(weight);

    cif (isUnder(combineBehavior)) {
        // account for the non-dielectric lobes above
        scale = scale * me.mCurrentTransmittance;
        Bssrdf_setScale(bssrdf, scale);

        // setup a MultipleTransmissionFresnel so this bssrdf can
        // be attenuated by previous lobes
        varying Fresnel * uniform fresnel = (varying Fresnel * uniform)
            Arena_alloc(me.mTls->mArena, sizeof(varying MultipleTransmissionFresnel));
        MultipleTransmissionFresnel_init((varying MultipleTransmissionFresnel * uniform) fresnel);
        Bssrdf_setTransmissionFresnel(bssrdf, fresnel);

        // account for dielectric lobes above (clearcoat not supported for bssrdfs)
        placeUnderPreviousLobes(me, bssrdf);
    } else {
        Bssrdf_setScale(bssrdf, scale);
    }

    cif (isOver(combineBehavior)) {
        // account for this lobe's energy allocation
        me.mWeightAccum += weight;
    }

    Bssrdf_setTraceSet(bssrdf, profile.mTraceSet);
    Bssrdf_setLabel(bssrdf, label);
    Bsdf_setBssrdf(me.mBsdf, bssrdf);
    accumulateAttenuation(me);
}

void
BsdfBuilder_addNormalizedDiffusion(
        varying BsdfBuilder& builder,
        const varying NormalizedDiffusion& profile,
        const varying float weight,
        const varying BsdfBuilderBehavior combineBehavior,
        const uniform int label)
{
    varying BsdfBuilderImpl& me = *builder.mImpl;

    cif (!testForVisibility(me, weight, combineBehavior)) { return; }

    if (!isSubsurfaceAllowed(*me.mState)) {
        LambertianBRDF diffuse;
        LambertianBRDF_init(diffuse, profile.mN, profile.mAlbedo);

        BsdfBuilder_addLambertianBRDF(builder,
                                      diffuse,
                                      weight,
                                      combineBehavior,
                                      label);
        return;
    }

    varying Bssrdf * uniform bssrdf = (varying Bssrdf * uniform)
        Arena_alloc(me.mTls->mArena, sizeof(varying NormalizedDiffusionBssrdf));

    NormalizedDiffusionBssrdf_init((varying NormalizedDiffusionBssrdf * uniform) bssrdf,
                                   profile.mN,
                                   profile.mAlbedo,
                                   profile.mRadius,
                                   profile.mMaterial,
                                   profile.mEvalNormalFn);

    varying Color scale = Color_ctor(weight);

    cif (isUnder(combineBehavior)) {
        // account for the non-dielectric lobes above
        scale = scale * me.mCurrentTransmittance;
        Bssrdf_setScale(bssrdf, scale);

        // setup a MultipleTransmissionFresnel so this bssrdf can
        // be attenuated by previous lobes
        varying Fresnel * uniform fresnel = (varying Fresnel * uniform)
            Arena_alloc(me.mTls->mArena, sizeof(varying MultipleTransmissionFresnel));
        MultipleTransmissionFresnel_init((varying MultipleTransmissionFresnel * uniform) fresnel);
        Bssrdf_setTransmissionFresnel(bssrdf, fresnel);

        // account for dielectric lobes above (clearcoat not supported for bssrdfs)
        placeUnderPreviousLobes(me, bssrdf);
    } else {
        Bssrdf_setScale(bssrdf, scale);
    }

    cif (isOver(combineBehavior)) {
        // account for this lobe's energy allocation
        me.mWeightAccum += weight;
    }

    Bssrdf_setTraceSet(bssrdf, profile.mTraceSet);
    Bssrdf_setLabel(bssrdf, label);
    Bsdf_setBssrdf(me.mBsdf, bssrdf);
    accumulateAttenuation(me);
}

void
BsdfBuilder_addRandomWalkSubsurface(
        varying BsdfBuilder& builder,
        const varying RandomWalkSubsurface& sss,
        const varying float weight,
        const varying BsdfBuilderBehavior combineBehavior,
        const uniform int label)
{
    varying BsdfBuilderImpl& me = *builder.mImpl;

    cif (!testForVisibility(me, weight, combineBehavior)) { return; }

    if (!isSubsurfaceAllowed(*me.mState)) {
        LambertianBRDF diffuse;
        LambertianBRDF_init(diffuse, sss.mN, sss.mAlbedo);

        BsdfBuilder_addLambertianBRDF(builder,
                                      diffuse,
                                      weight,
                                      combineBehavior,
                                      label);
        return;
    }

    varying VolumeSubsurface * uniform volume = (varying VolumeSubsurface * uniform)
        Arena_alloc(me.mTls->mArena, sizeof(varying VolumeSubsurface));

    VolumeSubsurface_init((varying VolumeSubsurface * uniform) volume,
                                   sss.mN,
                                   sss.mAlbedo,
                                   sss.mRadius,
                                   sss.mResolveSelfIntersections,
                                   sss.mMaterial,
                                   sss.mEvalNormalFn);

    varying Color scale = Color_ctor(weight);

    cif (isUnder(combineBehavior)) {
        // account for the non-dielectric lobes above
        scale = scale * me.mCurrentTransmittance;

        // setup a MultipleTransmissionFresnel so this bssrdf can
        // be attenuated by previous lobes
        varying Fresnel * uniform fresnel = (varying Fresnel * uniform)
            Arena_alloc(me.mTls->mArena, sizeof(varying MultipleTransmissionFresnel));
        MultipleTransmissionFresnel_init((varying MultipleTransmissionFresnel * uniform) fresnel);
        volume->mFresnel = fresnel;

        // account for dielectric lobes above (clearcoat not supported for bssrdfs)
        placeUnderPreviousLobes(me, volume);
    }

    volume->mScale = scale;

    cif (isOver(combineBehavior)) {
        // account for this lobe's energy allocation
        me.mWeightAccum += weight;
    }

    volume->mTraceSet = sss.mTraceSet;
    volume->mLabel = label;
    Bsdf_setVolumeSubsurface(me.mBsdf, volume);
    accumulateAttenuation(me);
}

void
BsdfBuilder_addFabricBRDF(
        varying BsdfBuilder& builder,
        const varying FabricBRDF& brdf,
        const varying float weight,
        const varying BsdfBuilderBehavior combineBehavior,
        const uniform int label)
{
    varying BsdfBuilderImpl& me = *builder.mImpl;

    cif (!testForVisibility(me, weight, combineBehavior)) { return; }

    varying BsdfLobe * uniform lobe = (varying BsdfLobe * uniform)
        Arena_alloc(me.mTls->mArena, sizeof(varying FabricBsdfLobe));

    FabricBsdfLobe_init((varying FabricBsdfLobe * uniform) lobe,
                           brdf.mN,
                           brdf.mShadingTangent,
                           brdf.mThreadDirection,
                           brdf.mThreadElevation,
                           brdf.mRoughness,
                           brdf.mThreadColor,
                           true); // use DwaFabric

    lobe = handleIridescence(me, lobe, brdf.mIridescence);

    varying Color scale = Color_ctor(weight);

    cif (isUnder(combineBehavior)) {
        // account for the non-dielectric lobes above
        scale = scale * me.mCurrentTransmittance;
        BsdfLobe_setScale(lobe, scale);
        // account for the dielectric/clearcoat lobes above
        lobe = placeUnderPreviousLobes(me, lobe);
    } else {
        BsdfLobe_setScale(lobe, scale);
    }

    cif (isOver(combineBehavior)) {
        // account for this lobe's energy allocation
        me.mWeightAccum += weight;
    }

    BsdfLobe_setLabel(lobe, label);
    Bsdf_addLobe(me.mBsdf, lobe);
    accumulateAttenuation(me);
}

void
BsdfBuilder_addVelvetBRDF(
        varying BsdfBuilder& builder,
        const varying VelvetBRDF& brdf,
        const varying float weight,
        const varying BsdfBuilderBehavior combineBehavior,
        const uniform int label)
{
    varying BsdfBuilderImpl& me = *builder.mImpl;

    cif (!testForVisibility(me, weight, combineBehavior)) { return; }

    varying BsdfLobe * uniform lobe = (varying BsdfLobe * uniform)
        Arena_alloc(me.mTls->mArena, sizeof(varying FabricVelvetBsdfLobe));

    FabricVelvetBsdfLobe_init((varying FabricVelvetBsdfLobe * uniform) lobe,
                              brdf.mN,
                              brdf.mRoughness,
                              brdf.mColor);

    lobe = handleIridescence(me, lobe, brdf.mIridescence);

    varying Color scale = Color_ctor(weight);

    cif (isUnder(combineBehavior)) {
        // account for the non-dielectric lobes above
        scale = scale * me.mCurrentTransmittance;
        BsdfLobe_setScale(lobe, scale);
        // account for the dielectric/clearcoat lobes above
        lobe = placeUnderPreviousLobes(me, lobe);
    } else {
        BsdfLobe_setScale(lobe, scale);
    }

    cif (isOver(combineBehavior)) {
        varying VelvetAttenuator * uniform atten = (varying VelvetAttenuator * uniform)
            Arena_alloc(me.mTls->mArena, sizeof(varying VelvetAttenuator));
        VelvetAttenuator_init(atten,
                              me.mTls->mArena,
                              brdf,
                              weight);

        stageAttenuator(me, atten);
    }

    BsdfLobe_setLabel(lobe, label);
    Bsdf_addLobe(me.mBsdf, lobe);
    accumulateAttenuation(me);
}

void
BsdfBuilder_addEyeCausticBRDF(
        varying BsdfBuilder& builder,
        const varying EyeCausticBRDF& brdf,
        const varying float weight,
        const varying BsdfBuilderBehavior combineBehavior,
        const uniform int label)
{
    varying BsdfBuilderImpl& me = *builder.mImpl;

    cif (!testForVisibility(me, weight, combineBehavior)) { return; }

    varying BsdfLobe * uniform lobe = (varying BsdfLobe * uniform)
        Arena_alloc(me.mTls->mArena, sizeof(varying EyeCausticBsdfLobe));

    const varying Vec3f adaptedNormal =
        Intersection_adaptNormal(asAnIntersection(*(me.mState)), brdf.mN);

    EyeCausticBsdfLobe_init((varying EyeCausticBsdfLobe * uniform) lobe,
                         adaptedNormal,
                         brdf.mIrisNormal,
                         brdf.mExponent,
                         brdf.mCausticColor);

    varying Color scale = Color_ctor(weight);

    cif (isUnder(combineBehavior)) {
        // account for the non-dielectric lobes above
        scale = scale * me.mCurrentTransmittance;
        BsdfLobe_setScale(lobe, scale);
        // account for the dielectric/clearcoat lobes above
        lobe = placeUnderPreviousLobes(me, lobe);
    } else {
        BsdfLobe_setScale(lobe, scale);
    }

    cif (isOver(combineBehavior)) {
        // account for this lobe's energy allocation
        me.mWeightAccum += weight;
    }

    BsdfLobe_setLabel(lobe, label);
    Bsdf_addLobe(me.mBsdf, lobe);
    accumulateAttenuation(me);
}

void
BsdfBuilder_addHairDiffuseBSDF(
        varying BsdfBuilder& builder,
        const varying HairDiffuseBSDF& bsdf,
        const varying float weight,
        const varying BsdfBuilderBehavior combineBehavior,
        const uniform int label)
{
    varying BsdfBuilderImpl& me = *builder.mImpl;

    cif (!testForVisibility(me, weight, combineBehavior)) { return; }

    varying BsdfLobe * uniform lobe = (varying BsdfLobe * uniform)
        Arena_alloc(me.mTls->mArena, sizeof(varying HairDiffuseBsdfLobe));

    HairDiffuseBsdfLobe_init((varying HairDiffuseBsdfLobe * uniform) lobe,
                             bsdf.mHairDir,
                             bsdf.mReflectionColor,
                             bsdf.mTransmissionColor);

    varying Color scale = Color_ctor(weight);

    cif (isUnder(combineBehavior)) {
        // account for the non-dielectric lobes above
        scale = scale * me.mCurrentTransmittance;
        BsdfLobe_setScale(lobe, scale);
        // account for the dielectric/clearcoat lobes above
        lobe = placeUnderPreviousLobes(me, lobe);
    } else {
        BsdfLobe_setScale(lobe, scale);
    }

    cif (isOver(combineBehavior)) {
        // account for this lobe's energy allocation
        me.mWeightAccum += weight;
    }

    BsdfLobe_setLabel(lobe, label);
    Bsdf_addLobe(me.mBsdf, lobe);
    accumulateAttenuation(me);
}

void
BsdfBuilder_addHairBSDF(
        varying BsdfBuilder& builder,
        const varying HairBSDF& bsdf,
        const varying float weight,
        const varying BsdfBuilderBehavior combineBehavior,
        const uniform int label)
{
    varying BsdfBuilderImpl& me = *builder.mImpl;

    cif (!testForVisibility(me, weight, combineBehavior)) { return; }

    const varying float minRoughness = getMinRoughness(me);

    // Computing SigmaA in Builder for now
    // TODO - Move this inside the BSDF lobe.
    // Right now it's inefficient to compute inside the lobe since it'll
    // get computed 3x the way the lobes are currently designed.
    // This work can be a part of extending hair shading model to include a
    // near-field shading solution.
    const varying Color hairSigmaA =
        HairUtil_computeAbsorptionCoefficients(
                bsdf.mHairColor,
                bsdf.mAziRoughnessTT);

    varying BsdfLobe * uniform lobe = (varying BsdfLobe * uniform)
        Arena_alloc(me.mTls->mArena, sizeof(varying HairOneSamplerBsdfLobe));

    HairOneSamplerBsdfLobe_init(
            (varying HairOneSamplerBsdfLobe * uniform) lobe,
            bsdf.mHairDir,
            bsdf.mHairUV,
            getMediumIor(*me.mState),
            bsdf.mIOR,
            bsdf.mFresnelType,
            bsdf.mCuticleLayerThickness,
            bsdf.mShowR,
            bsdf.mShiftR,
            clampHairRoughness(sHairRoughnessMin, minRoughness, bsdf.mRoughnessR),
            bsdf.mTintR,
            bsdf.mShowTT,
            bsdf.mShiftTT,
            clampHairRoughness(sHairRoughnessMin, minRoughness, bsdf.mRoughnessTT),
            clampHairRoughness(sHairRoughnessMin, minRoughness, bsdf.mAziRoughnessTT),
            bsdf.mTintTT,
            bsdf.mSaturationTT,
            bsdf.mShowTRT,
            bsdf.mShiftTRT,
            clampHairRoughness(sHairRoughnessMin, minRoughness, bsdf.mRoughnessTRT),
            bsdf.mTintTRT,
            bsdf.mShowGlint,
            clampHairRoughness(sHairGlintRoughnessMin, minRoughness, bsdf.mRoughnessGlint),
            bsdf.mEccentricityGlint,
            bsdf.mSaturationGlint,
            bsdf.mHairRotation,
            bsdf.mHairNormal,
            bsdf.mShowTRRT,
            clampHairRoughness(sHairRoughnessMin, minRoughness, 1.0f), // TRRTlongRoughness (hard-coded in scalar HairOneSampleLobe ctor)
            bsdf.mHairColor,
            hairSigmaA);

    varying Fresnel * uniform fresnel =
        applyFresnelBehavior(me,
                             Color_ctor(bsdf.mIOR),
                             (varying Color) sBlack, // k
                             false,                  // isConductor
                             weight,
                             lobe);

    cif (isUnder(combineBehavior)) {
        // account for the non-dielectric lobes above
        const varying Color scale = sWhite * me.mCurrentTransmittance;
        BsdfLobe_setScale(lobe, scale);
        // account for the dielectric/clearcoat lobes above
        lobe = placeUnderPreviousLobes(me, lobe);
    }

    cif (isOver(combineBehavior)) {
        // account for this lobe's energy allocation
        me.mWeightAccum += weight;
    }

    BsdfLobe_setLabel(lobe, label);
    Bsdf_addLobe(me.mBsdf, lobe);
    accumulateAttenuation(me);
}

void
BsdfBuilder_addHairRBRDF(
        varying BsdfBuilder& builder,
        const varying HairRBRDF& brdf,
        const varying float weight,
        const varying BsdfBuilderBehavior combineBehavior,
        const uniform int label)
{
    varying BsdfBuilderImpl& me = *builder.mImpl;

    cif (!testForVisibility(me, weight, combineBehavior)) { return; }

    varying BsdfLobe * uniform lobe = (varying BsdfLobe * uniform)
        Arena_alloc(me.mTls->mArena, sizeof(varying HairBsdfLobe));

    HairRBsdfLobe_init((varying HairBsdfLobe * uniform) lobe,
                       brdf.mHairDir,
                       brdf.mHairUV,
                       getMediumIor(*me.mState),
                       brdf.mIOR,
                       brdf.mFresnelType,
                       brdf.mCuticleLayerThickness,
                       brdf.mShift,
                       clampHairRoughness(sHairRoughnessMin, getMinRoughness(me), brdf.mRoughness),
                       brdf.mTint);

    varying Fresnel * uniform fresnel = applyFresnelBehavior(
            me,
            Color_ctor(brdf.mIOR),
            (varying Color) sBlack, // k
            false,                  // isConductor
            weight,
            lobe);

    cif (isUnder(combineBehavior)) {
        // account for the non-dielectric lobes above
        const varying Color scale = sWhite * me.mCurrentTransmittance;
        BsdfLobe_setScale(lobe, scale);
        // account for the dielectric/clearcoat lobes above
        lobe = placeUnderPreviousLobes(me, lobe);
    }

    cif (isOver(combineBehavior)) {
        varying LobeAttenuator * uniform atten = (varying LobeAttenuator * uniform)
            Arena_alloc(me.mTls->mArena, sizeof(varying HairAttenuator));
        HairAttenuator_init((varying HairAttenuator * uniform) atten,
                            me.mTls->mArena,
                            fresnel);

        stageAttenuator(me, atten);
    }

    BsdfLobe_setLabel(lobe, label);
    Bsdf_addLobe(me.mBsdf, lobe);
    accumulateAttenuation(me);
}

void
BsdfBuilder_addHairTRTBRDF(
        varying BsdfBuilder& builder,
        const varying HairTRTBRDF& brdf,
        const varying float weight,
        const varying BsdfBuilderBehavior combineBehavior,
        const uniform int label)
{
    varying BsdfBuilderImpl& me = *builder.mImpl;

    cif (!testForVisibility(me, weight, combineBehavior)) { return; }

    const varying Color hairSigmaA =
            HairUtil_computeAbsorptionCoefficients(brdf.mHairColor,
                                                   brdf.mAziRoughness);

    varying BsdfLobe * uniform lobe = (varying BsdfLobe * uniform)
        Arena_alloc(me.mTls->mArena, sizeof(varying HairBsdfLobe));

    HairTRTBsdfLobe_init((varying HairBsdfLobe * uniform) lobe,
                         brdf.mHairDir,
                         brdf.mHairUV,
                         getMediumIor(*me.mState),
                         brdf.mIOR,
                         brdf.mFresnelType,
                         brdf.mCuticleLayerThickness,
                         brdf.mShift,
                         clampHairRoughness(sHairRoughnessMin, getMinRoughness(me), brdf.mRoughness),
                         brdf.mHairColor,
                         hairSigmaA,
                         brdf.mTint,
                         brdf.mShowGlint,
                         clampHairRoughness(sHairGlintRoughnessMin, getMinRoughness(me), brdf.mRoughnessGlint),
                         brdf.mEccentricityGlint,
                         brdf.mSaturationGlint,
                         brdf.mHairRotation,
                         brdf.mHairNormal);

    varying Color scale = Color_ctor(weight);

    cif (isUnder(combineBehavior)) {
        // account for the non-dielectric lobes above
        scale = scale * me.mCurrentTransmittance;
        BsdfLobe_setScale(lobe, scale);
        // account for the dielectric/clearcoat lobes above
        lobe = placeUnderPreviousLobes(me, lobe);
    } else {
        BsdfLobe_setScale(lobe, scale);
    }

    cif (isOver(combineBehavior)) {
        // account for this lobe's energy allocation
        me.mWeightAccum += weight;
    }

    BsdfLobe_setLabel(lobe, label);
    Bsdf_addLobe(me.mBsdf, lobe);
    accumulateAttenuation(me);
}

void
BsdfBuilder_addHairTTBTDF(
        varying BsdfBuilder& builder,
        const varying HairTTBTDF& btdf,
        const varying float weight,
        const varying BsdfBuilderBehavior combineBehavior,
        const uniform int label)
{
    varying BsdfBuilderImpl& me = *builder.mImpl;

    cif (!testForVisibility(me, weight, combineBehavior)) { return; }

    const varying Color hairSigmaA =
            HairUtil_computeAbsorptionCoefficients(btdf.mHairColor,
                                                   btdf.mAziRoughness);

    varying BsdfLobe * uniform lobe = (varying BsdfLobe * uniform)
        Arena_alloc(me.mTls->mArena, sizeof(varying HairBsdfLobe));

    const varying float minRoughness = getMinRoughness(me);

    HairTTBsdfLobe_init((varying HairBsdfLobe * uniform) lobe,
                        btdf.mHairDir,
                        btdf.mHairUV,
                        getMediumIor(*me.mState),
                        btdf.mIOR,
                        btdf.mFresnelType,
                        btdf.mCuticleLayerThickness,
                        btdf.mShift,
                        clampHairRoughness(sHairRoughnessMin, minRoughness, btdf.mRoughness),
                        clampHairRoughness(sHairRoughnessMin, minRoughness, btdf.mAziRoughness),
                        btdf.mHairColor,
                        hairSigmaA,
                        btdf.mTint,
                        btdf.mSaturation);

    varying Color scale = Color_ctor(weight);

    cif (isUnder(combineBehavior)) {
        // account for the non-dielectric lobes above
        scale = scale * me.mCurrentTransmittance;
        BsdfLobe_setScale(lobe, scale);
        // account for the dielectric/clearcoat lobes above
        lobe = placeUnderPreviousLobes(me, lobe);
    } else {
        BsdfLobe_setScale(lobe, scale);
    }

    cif (isOver(combineBehavior)) {
        // account for this lobe's energy allocation
        me.mWeightAccum += weight;
    }

    BsdfLobe_setLabel(lobe, label);
    Bsdf_addLobe(me.mBsdf, lobe);
    accumulateAttenuation(me);
}

void
BsdfBuilder_addHairTRRTBRDF(
        varying BsdfBuilder& builder,
        const varying HairTRRTBRDF& brdf,
        const varying float weight,
        const varying BsdfBuilderBehavior combineBehavior,
        const uniform int label)
{
    varying BsdfBuilderImpl& me = *builder.mImpl;

    cif (!testForVisibility(me, weight, combineBehavior)) { return; }

    const varying Color hairSigmaA =
            HairUtil_computeAbsorptionCoefficients(brdf.mHairColor,
                                                   brdf.mAziRoughness);

    varying BsdfLobe * uniform lobe = (varying BsdfLobe * uniform)
        Arena_alloc(me.mTls->mArena, sizeof(varying HairBsdfLobe));

    HairTRRTBsdfLobe_init((varying HairBsdfLobe * uniform) lobe,
                          brdf.mHairDir,
                          brdf.mHairUV,
                          getMediumIor(*me.mState),
                          brdf.mIOR,
                          brdf.mFresnelType,
                          brdf.mCuticleLayerThickness,
                          0.0f, // longShift
                          clampHairRoughness(sHairRoughnessMin, getMinRoughness(me), brdf.mRoughness),
                          brdf.mHairColor,
                          hairSigmaA,
                          brdf.mTint);

    varying Color scale = Color_ctor(weight);

    cif (isUnder(combineBehavior)) {
        // account for the non-dielectric lobes above
        scale = scale * me.mCurrentTransmittance;
        BsdfLobe_setScale(lobe, scale);
        // account for the dielectric/clearcoat lobes above
        lobe = placeUnderPreviousLobes(me, lobe);
    } else {
        BsdfLobe_setScale(lobe, scale);
    }

    cif (isOver(combineBehavior)) {
        // account for this lobe's energy allocation
        me.mWeightAccum += weight;
    }

    BsdfLobe_setLabel(lobe, label);
    Bsdf_addLobe(me.mBsdf, lobe);
    accumulateAttenuation(me);
}

void
BsdfBuilder_addGlitterFlakeBRDF(
        varying BsdfBuilder& builder,
        const varying GlitterFlakeBRDF& brdf,
        const varying float weight,
        const varying BsdfBuilderBehavior combineBehavior,
        const uniform int label)
{
    varying BsdfBuilderImpl& me = *builder.mImpl;

    cif (!testForVisibility(me, weight, combineBehavior)) { return; }

    varying BsdfLobe * uniform lobe = (varying BsdfLobe * uniform)
        Arena_alloc(me.mTls->mArena, sizeof(varying GlitterGGXCookTorranceBsdfLobe));

    // Adapt normal to prevent reflection ray from self-intersecting this geometry
    const varying Vec3f adaptedNormal =
        Intersection_adaptNormal(asAnIntersection(*(me.mState)), brdf.mN);

    // no cook-torrance compensation for glitter
    Color favg = sBlack;
    GlitterGGXCookTorranceBsdfLobe_init((varying GlitterGGXCookTorranceBsdfLobe * uniform) lobe,
                                        adaptedNormal,
                                        brdf.mFlakeN,
                                        brdf.mRoughness,
                                        favg);

    applyFresnelBehavior(me,
                         brdf.mEta, brdf.mK,
                         true, // isConductor
                         weight,
                         lobe);

    lobe = handleIridescence(me, lobe, brdf.mIridescence);

    varying Color scale = sWhite;

    cif (isUnder(combineBehavior)) {
        // account for the non-dielectric lobes above
        scale = scale * me.mCurrentTransmittance;
        BsdfLobe_setScale(lobe, scale);
        // account for the dielectric/clearcoat lobes above
        lobe = placeUnderPreviousLobes(me, lobe);
    } else {
        BsdfLobe_setScale(lobe, scale);
    }

    cif (isOver(combineBehavior)) {
        // account for this lobe's energy allocation
        me.mWeightAccum += weight;
    }

    BsdfLobe_setLabel(lobe, label);
    Bsdf_addLobe(me.mBsdf, lobe);
    accumulateAttenuation(me);
}

void
BsdfBuilder_addStochasticFlakesBRDF(
        varying BsdfBuilder& builder,
        const varying StochasticFlakesBRDF& brdf,
        const varying float weight,
        const varying BsdfBuilderBehavior combineBehavior,
        const uniform int label)
{
    varying BsdfBuilderImpl& me = *builder.mImpl;

    cif (!testForVisibility(me, weight, combineBehavior)) { return; }

    varying BsdfLobe * uniform lobe = (varying BsdfLobe * uniform)
        Arena_alloc(me.mTls->mArena, sizeof(varying StochasticFlakesBsdfLobe));

    StochasticFlakesBsdfLobe_init((varying StochasticFlakesBsdfLobe * uniform) lobe,
                                  brdf.mN,
                                  brdf.mFlakeNormals,
                                  brdf.mFlakeColors,
                                  brdf.mFlakeCount,
                                  brdf.mFlakeRoughness,
                                  brdf.mFlakeRandomness);

    varying Color scale = Color_ctor(weight);

    cif (isUnder(combineBehavior)) {
        // account for the non-dielectric lobes above
        scale = scale * me.mCurrentTransmittance;
        BsdfLobe_setScale(lobe, scale);
        // account for the dielectric/clearcoat lobes above
        lobe = placeUnderPreviousLobes(me, lobe);
    } else {
        BsdfLobe_setScale(lobe, scale);
    }

    cif (isOver(combineBehavior)) {
        // account for this lobe's energy allocation
        me.mWeightAccum += weight;
    }

    BsdfLobe_setLabel(lobe, label);
    Bsdf_addLobe(me.mBsdf, lobe);
    accumulateAttenuation(me);
}

void
BsdfBuilder_addEmission(
        varying BsdfBuilder& builder,
        const varying Color& emission)
{
    varying BsdfBuilderImpl& me = *builder.mImpl;

    Bsdf_setSelfEmission(me.mBsdf, Bsdf_getSelfEmission(me.mBsdf) + emission);
}

void
BsdfBuilder_setEarlyTermination(varying BsdfBuilder& builder)
{
    varying BsdfBuilderImpl& me = *builder.mImpl;

    Bsdf_setEarlyTermination(me.mBsdf, lanemask());
}

void
BsdfBuilder_setThinGeo(varying BsdfBuilder& builder)
{
    varying BsdfBuilderImpl& me = *builder.mImpl;

    me.mIsThinGeo = true;
}

void
BsdfBuilder_setPreventLightCulling(varying BsdfBuilder& builder,
                                   uniform bool isPrevented)
{
    varying BsdfBuilderImpl& me = *builder.mImpl;

    me.mPreventLightCulling = isPrevented;
}


// ---------------------------------------------------------------------------
void
BsdfBuilder_startAdjacentComponents(varying BsdfBuilder& builder)
{
    varying BsdfBuilderImpl& me = *builder.mImpl;

    me.mInAdjacentBlock = true;
}

void
BsdfBuilder_endAdjacentComponents(varying BsdfBuilder& builder)
{
    varying BsdfBuilderImpl& me = *builder.mImpl;

    me.mInAdjacentBlock = false;
    accumulateAttenuation(me);
}

varying Closure * uniform
BsdfBuilder_getClosure(const varying BsdfBuilder& builder)
{
    varying BsdfBuilderImpl& me = *builder.mImpl;

    return asClosure(me.mBsdf);
}

void
BsdfBuilder_show(varying BsdfBuilder& builder,
                 const uniform Material * const uniform material)
{
    varying BsdfBuilderImpl& me = *builder.mImpl;
    Bsdf_show(*me.mBsdf, material);
}

// ----------------------------------------------------------------------------------
// functions called by ShadeBundleHandler.cc
// ----------------------------------------------------------------------------------

// Initializes an array of BsdfBuilders so they are ready to shade.
// This function is called from Shadingv.cc's shadev() function.
// Notice that this function is not exported to C++, but is instead
// invoked via Function pointer returned by getBsdfBuilderInitFunc().
// This allows implicit casting of c++ to corresponding (size matching)
// ispc types:
// * shading::TLState -> ShadingTLState
// * shading::State   -> State
void
initBsdfBuilderArray(uniform ShadingTLState * uniform tls,
                     uniform int n,
                     const varying State * const uniform state,
                     varying Bsdf * uniform bsdf,
                     varying BsdfBuilder * uniform builder)
{
    for (uniform int i = 0; i < n; ++i) {
        BsdfBuilder_init(builder[i], bsdf[i], *tls, state[i]);
    }
}
export void * uniform getBsdfBuilderInitFunc() { return initBsdfBuilderArray; }


