// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0


/// @file LobeAttenuator.ispc

#include "LobeAttenuator.isph"

#include <moonray/rendering/shading/ispc/BsdfComponent.isph>
#include <moonray/rendering/shading/ispc/bsdf/Fresnel.isph>
#include <moonray/rendering/shading/ispc/bsdf/under/BsdfUnder.isph>
#include <moonray/rendering/shading/ispc/bsdf/under/BsdfUnderClearcoat.isph>
#include <moonray/rendering/shading/ispc/bsdf/under/BsdfUnderClearcoatTransmission.isph>
#include <moonray/rendering/shading/ispc/bssrdf/Bssrdf.isph>
#include <moonray/rendering/shading/ispc/bssrdf/VolumeSubsurface.isph>

void
LobeAttenuator_init(
        varying LobeAttenuator * uniform me,
        const varying Vec3f& N,
        const varying Fresnel * uniform fresnel,
        const uniform intptr_t attenuateLobeFn,
        const uniform intptr_t attenuateBssrdfFn,
        const uniform intptr_t attenuateVolumeFn)
{
    MNRY_ASSERT(attenuateLobeFn);
    MNRY_ASSERT(attenuateBssrdfFn);
    MNRY_ASSERT(attenuateVolumeFn);

    me->mNormal = N;
    me->mOneMinusFresnel = fresnel;

    me->mAttenuateLobeFn    = attenuateLobeFn;
    me->mAttenuateBssrdfFn  = attenuateBssrdfFn;
    me->mAttenuateVolumeFn  = attenuateVolumeFn;
    me->mMask = lanemask();
}

varying BsdfLobe * uniform
LobeAttenuator_attenuate(
        const varying LobeAttenuator * uniform me,
        uniform Arena * uniform arena,
        varying BsdfLobe * const uniform lobe)
{
    varying BsdfLobe * uniform result = lobe;
    if (isActive(me)) {
        result = ((AttenuateLobeFnType)(me->mAttenuateLobeFn))(me, arena, lobe);
    }
    return result;
}

void
LobeAttenuator_attenuate(
        const varying LobeAttenuator * uniform me,
        uniform Arena * uniform arena,
        varying Bssrdf * const uniform bssrdf)
{
    if (isActive(me)) {
        ((AttenuateBssrdfFunType)(me->mAttenuateBssrdfFn))(me, arena, bssrdf);
    }
}

void
LobeAttenuator_attenuate(
        const varying LobeAttenuator * uniform me,
        uniform Arena * uniform arena,
        varying VolumeSubsurface * const uniform vs)
{
    if (isActive(me)) {
        ((AttenuateVolumeFnType)(me->mAttenuateVolumeFn))(me, arena, vs);
    }
}

// --------------------------------------------

// Default BSDF Attenuator
varying BsdfLobe * uniform
attenuateLobeSimple(
        const varying LobeAttenuator * uniform me,
        uniform Arena * uniform arena,
        varying BsdfLobe * const uniform lobe)
{
    if (me->mOneMinusFresnel == nullptr)
        return lobe;

    varying BsdfLobe * uniform under = (varying BsdfLobe * uniform)
        Arena_alloc(arena, sizeof(varying UnderBsdfLobe));

    UnderBsdfLobe_init((varying UnderBsdfLobe * uniform) under, lobe, me->mNormal, 0.0f, sWhite, 1.0f);
    BsdfLobe_setFresnel(under, me->mOneMinusFresnel);
    return under;
}

// Default BSSRDF Attenuator
void
attenuateBssrdfSimple(
        const varying LobeAttenuator * uniform me,
        uniform Arena * uniform arena,
        varying Bssrdf * const uniform bssrdf)
{
    if (me->mOneMinusFresnel == nullptr)
        return;

    varying MultipleTransmissionFresnel * uniform fresnel =
        (varying MultipleTransmissionFresnel * uniform) Bssrdf_getTransmissionFresnel(bssrdf);
    // Assumes all BSSRDF's have a MultipleTransmissionFresnel attached.
    MNRY_ASSERT(fresnel);
    MultipleTransmissionFresnel_add(fresnel, me->mOneMinusFresnel);
}

// Default VolumeSubsurface Attenuator
void
attenuateVolumeSimple(
        const varying LobeAttenuator * uniform me,
        uniform Arena * uniform arena,
        varying VolumeSubsurface * const uniform vs)
{
    if (me->mOneMinusFresnel == nullptr)
        return;

    varying MultipleTransmissionFresnel * uniform fresnel =
        (varying MultipleTransmissionFresnel * uniform) VolumeSubsurface_getTransmissionFresnel(vs);
    // Assumes all VS's have a MultipleTransmissionFresnel attached.
    MNRY_ASSERT(fresnel);
    MultipleTransmissionFresnel_add(fresnel, me->mOneMinusFresnel);
}

// --------------------------------------------

// Attenuate lobe under hair
varying BsdfLobe * uniform
attenuateLobeHair(
        const varying LobeAttenuator * uniform me,
        uniform Arena * uniform arena,
        varying BsdfLobe * const uniform lobe)
{
    BsdfLobe_setFresnel(lobe, me->mOneMinusFresnel);
    return lobe;
}

// Attenuate lobe under clearcoat
varying BsdfLobe * uniform
attenuateLobeClearcoat(
        const varying LobeAttenuator * uniform attenuator,
        uniform Arena * uniform arena,
        varying BsdfLobe * const uniform lobe)
{
    const varying ClearcoatAttenuator * uniform me =
        (const varying ClearcoatAttenuator * uniform) attenuator;

    varying BsdfLobe * uniform result;

    // TODO: this conditional may cause non-determinism: !isEqual(me->mEtaI, me->mEtaT))
    // Generally we should never decide a uniform's type based on varying conditional, as
    // both code paths will execute with the latter one overwriting the uniform  and
    // potentially leading to a memory leak or worse. It may be better to ignore the case
    // where (etaI = etaT) here and always create the UnderClearcoat* lobe when (me->mRefracts = true).
    // We need to check to see how robust UnderClearcoatBsdfLobe and UnderClearcoatTransmissionBsdfLobe
    // to make sure they can handle the case of (etaI = etaT).
    //
    // NOTE: Using 'cif' here because we can assume that in the vast majority of cases
    // isEqual(me->mEtaI, me->mEtaT) will be coherent across lanes.
    cif (me->mRefracts && !isEqual(me->mEtaI, me->mEtaT)) {
        const varying bool isReflectionLobe = BsdfLobe_matchesFlag(lobe, BSDF_LOBE_TYPE_REFLECTION);
        // NOTE: using 'cif' here since we expect this varying conditional to *always* be
        // conherent (the same varying BsdfLobe should never have some lanes that are
        // only BSDF_LOBE_TYPE_REFLECTION while others are only BSDF_LOBE_TYPE_TRANSMISSION)
        cif (isReflectionLobe) {
            result = (varying BsdfLobe * uniform)
                Arena_alloc(arena, sizeof(varying UnderClearcoatBsdfLobe));
            UnderClearcoatBsdfLobe_init(
                    (varying UnderClearcoatBsdfLobe * uniform) result,
                    lobe,
                    me->mNormal,
                    me->mExitingFresnel,
                    me->mEtaI, me->mEtaT,
                    me->mThickness, me->mAttenuationColor, me->mAttenuationWeight);
        } else { // we are wrapping a transmission lobe
            result = (varying BsdfLobe * uniform)
                Arena_alloc(arena, sizeof(varying UnderClearcoatTransmissionBsdfLobe));
            UnderClearcoatTransmissionBsdfLobe_init(
                    (varying UnderClearcoatTransmissionBsdfLobe * uniform) result,
                    lobe,
                    me->mNormal,
                    me->mExitingFresnel,
                    me->mEtaI, me->mEtaT,
                    me->mThickness, me->mAttenuationColor, me->mAttenuationWeight);
        }
    } else {
        // no refraction
        result = (varying BsdfLobe * uniform)
            Arena_alloc(arena, sizeof(varying UnderBsdfLobe));
        UnderBsdfLobe_init(
                (varying UnderBsdfLobe * uniform) result,
                lobe,
                me->mNormal,
                me->mThickness, me->mAttenuationColor, me->mAttenuationWeight);
    }

    BsdfLobe_setFresnel(result, me->mOneMinusFresnel);
    return result;
}

// --------------------------------------------

void
SimpleAttenuator_init(
        varying SimpleAttenuator * uniform me,
        uniform Arena * uniform arena,
        const varying Vec3f& N,
        const varying float roughness,
        const varying Fresnel * uniform fresnel)
{
    // in the vector version of this attenuator we cannot choose
    // a uniform Fresnel based on a varying conditional (roughness),
    // so we rely on OneMinusRoughFresnel matching OneMinusFresnel
    // when roughness == 0
    varying OneMinusRoughFresnel * uniform omFresnel = (varying OneMinusRoughFresnel * uniform)
        Arena_alloc(arena, sizeof(varying OneMinusRoughFresnel));
    OneMinusRoughFresnel_init(omFresnel, fresnel, roughness);

    LobeAttenuator_init((varying LobeAttenuator * uniform) me, N,
                        (const varying Fresnel * uniform) omFresnel,
                        (uniform intptr_t) (void*) attenuateLobeSimple,     // |
                        (uniform intptr_t) (void*) attenuateBssrdfSimple,   // | standard atten fns
                        (uniform intptr_t) (void*) attenuateVolumeSimple);  // |
}

void
HairAttenuator_init(
        varying HairAttenuator * uniform me,
        uniform Arena * uniform arena,
        const varying Fresnel * uniform fresnel)
{
    varying OneMinusFresnel * uniform omFresnel = 0;

    if (fresnel) {
        omFresnel = (varying OneMinusFresnel * uniform)
            Arena_alloc(arena, sizeof(varying OneMinusFresnel));
        OneMinusFresnel_init(omFresnel, fresnel);
    }

    const varying Vec3f unusedVec = Vec3f_ctor(0.f, 0.f, 1.f);
    LobeAttenuator_init((varying LobeAttenuator * uniform) me,
                        unusedVec,
                        (const varying Fresnel * uniform) omFresnel,
                        (uniform intptr_t) (void*) attenuateLobeHair,       // custom atten fn
                        (uniform intptr_t) (void*) attenuateBssrdfSimple,   // standard atten fn
                        (uniform intptr_t) (void*) attenuateVolumeSimple);  // standard atten fn
}

void
VelvetAttenuator_init(
        varying VelvetAttenuator * uniform me,
        uniform Arena * uniform arena,
        const varying VelvetBRDF& velvet,
        varying float weight)
{
    if (velvet.mUseAbsorbingFibers == false) {
        // Lets more energy through based on the velvet color
        const varying Color& c = velvet.mColor;
        weight *= max(c.r,
                  max(c.g,
                      c.b));
    }

    varying OneMinusVelvetFresnel * uniform fresnel = (varying OneMinusVelvetFresnel * uniform)
        Arena_alloc(arena, sizeof(varying OneMinusVelvetFresnel));
    OneMinusVelvetFresnel_init(fresnel, velvet.mRoughness, weight);

    LobeAttenuator_init((varying LobeAttenuator * uniform) me,
                        velvet.mN,
                        (const varying Fresnel * uniform) fresnel,
                        (uniform intptr_t) (void*) attenuateLobeSimple,     // |
                        (uniform intptr_t) (void*) attenuateBssrdfSimple,   // | standard atten fns
                        (uniform intptr_t) (void*) attenuateVolumeSimple);  // |
}

void
ClearcoatAttenuator_init(
        varying ClearcoatAttenuator * uniform me,
        uniform Arena * uniform arena,
        const varying Vec3f& N,
        varying float roughness,
        varying float etaI,
        varying float etaT,
        uniform bool refracts,
        varying float thickness,
        const varying Color& attenuationColor,
        varying float attenuationWeight,
        varying Fresnel * uniform fresnel)
{
    me->mEtaI = etaI;
    me->mEtaT = etaT;
    me->mRefracts = refracts;
    me->mThickness = thickness;
    me->mAttenuationColor = attenuationColor;
    me->mAttenuationWeight = attenuationWeight;
    me->mExitingFresnel = (varying Fresnel * uniform)
        Arena_alloc(arena, sizeof(varying DielectricFresnel));
    DielectricFresnel_init((varying DielectricFresnel * uniform) me->mExitingFresnel,
                           etaT, etaI,  // flip etaI, etaT for exiting
                           1.0f);       // weight

    // in the vector version of this attenuator we cannot choose
    // a uniform Fresnel based on a varying conditional (roughness),
    // so we rely on OneMinusRoughFresnel matching OneMinusFresnel
    // when roughness == 0
    varying OneMinusRoughFresnel * uniform omFresnel = (varying OneMinusRoughFresnel * uniform)
        Arena_alloc(arena, sizeof(varying OneMinusRoughFresnel));
    OneMinusRoughFresnel_init(omFresnel, fresnel, roughness);

    LobeAttenuator_init((varying LobeAttenuator * uniform) me, N,
                        (const varying Fresnel * uniform) omFresnel,
                        (uniform intptr_t) (void*) attenuateLobeClearcoat,  // custom atten fn
                        (uniform intptr_t) (void*) attenuateBssrdfSimple,   // standard atten fn
                        (uniform intptr_t) (void*) attenuateVolumeSimple);  // standard atten fn
}

