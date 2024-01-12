// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

/// @file Closure.ispc

#include "Closure.isph"

#include "bsdf/Bsdf.isph"
#include "bsdf/cook_torrance/BsdfAnisoCookTorrance.isph"
#include "bsdf/cook_torrance/BsdfCookTorrance.isph"
#include "bsdf/cook_torrance/BsdfGGXCookTorrance.isph"
#include "bsdf/cook_torrance/BsdfGlitterGGXCookTorrance.isph"
#include "bsdf/cook_torrance/BsdfTransmissionCookTorrance.isph"
#include "bsdf/BsdfLambert.isph"
#include "bsdf/BsdfRetroreflection.isph"
#include "bsdf/BsdfEyeCaustic.isph"
#include "bsdf/BsdfMirror.isph"
#include "bsdf/under/BsdfUnder.isph"
#include "bsdf/under/BsdfUnderClearcoatTransmission.isph"
#include "bsdf/BsdfIridescence.isph"
#include "bsdf/fabric/BsdfFabric.isph"
#include "bsdf/fabric/BsdfFabricVelvet.isph"
#include "bsdf/ward/BsdfWardDuer.isph"
#include "bsdf/BsdfStochasticFlakes.isph"
#include "bsdf/Fresnel.isph"
#include "bssrdf/Dipole.isph"
#include "bssrdf/NormalizedDiffusion.isph"

#include <moonray/rendering/shading/ispc/EvalShader.isph>

#include <moonray/rendering/bvh/shading/ispc/ShadingTLState.isph>
#include <moonray/rendering/bvh/shading/ispc/State.isph>
#include <scene_rdl2/common/math/ispc/Color.isph>
#include <scene_rdl2/common/math/ispc/ReferenceFrame.isph>
#include <scene_rdl2/common/math/ispc/Vec3.isph>
#include <scene_rdl2/render/util/Arena.isph>

// ---------------------------------------------------------------------
// Closure Creation
// ---------------------------------------------------------------------

// initialize a Closure
void
Closure_init(varying Closure * uniform closure)
{
    Bsdf_init(asBsdf(closure));
}

// ---------------------------------------------------------------------
// Bssrdf Creation
// ---------------------------------------------------------------------

// add a dipole bssrdf to the closure.
void
Closure_addDipoleBssrdf(varying Closure * uniform closure,
                        uniform ShadingTLState * uniform tls,
                        const varying State &state,
                        const varying Color &scale,
                        varying Fresnel * uniform fresnel,
                        const varying Vec3f &normal,
                        const varying Color &translucentColor,
                        const varying Color &radius,
                        const uniform TraceSet * uniform traceSet,
                        const uniform Material * uniform material,
                        const uniform intptr_t evalNormalFn,
                        const uniform int label)
{
    varying Bsdf * uniform bsdf = asBsdf(closure);

    // lib/shading only supports a single bssrdf at the moment.
    // so adding a bssrdf to a closure that already has one is
    // most likely a shader error.
    assert(bsdf->mBssrdf == NULL);

    uniform Arena * uniform arena = tls->mArena;

    varying DipoleBssrdf * uniform bssrdf =
        (varying DipoleBssrdf * uniform)
        Arena_alloc(arena, sizeof(varying DipoleBssrdf));
    varying Bssrdf * uniform bss = (varying Bssrdf * uniform)bssrdf;

    DipoleBssrdf_init(bssrdf, normal, translucentColor,
                      radius, material, evalNormalFn);
    Bssrdf_setTransmissionFresnel(bss, fresnel);
    Bssrdf_setScale(bss, scale);
    Bssrdf_setTraceSet(bss, traceSet);
    Bssrdf_setLabel(bss, label);
    Bsdf_setBssrdf(bsdf, bss);
}

void
Closure_addNormalizedDiffusionBssrdf(varying Closure * uniform closure,
                                     uniform ShadingTLState * uniform tls,
                                     const varying State &state,
                                     const varying Color &scale,
                                     varying Fresnel * uniform fresnel,
                                     const varying Vec3f &normal,
                                     const varying Color &translucentColor,
                                     const varying Color &radius,
                                     const uniform TraceSet * uniform traceSet,
                                     const uniform Material * uniform material,
                                     const uniform intptr_t evalNormalFn,
                                     const uniform int label)
{
    varying Bsdf * uniform bsdf = asBsdf(closure);

    // lib/shading only supports a single bssrdf at the moment.
    // so adding a bssrdf to a closure that already has one is
    // most likely a shader error.
    assert(bsdf->mBssrdf == NULL);

    uniform Arena * uniform arena = tls->mArena;

    varying NormalizedDiffusionBssrdf * uniform nd =
        (varying NormalizedDiffusionBssrdf * uniform)
        Arena_alloc(arena, sizeof(varying NormalizedDiffusionBssrdf));
    varying Bssrdf * uniform bssrdf = (varying Bssrdf * uniform)nd;
    NormalizedDiffusionBssrdf_init(nd, normal, translucentColor, radius, 
                                   material, evalNormalFn);
    Bssrdf_setTransmissionFresnel(bssrdf, fresnel);
    Bssrdf_setScale(bssrdf, scale);
    Bssrdf_setTraceSet(bssrdf, traceSet);
    Bssrdf_setLabel(bssrdf, label);
    Bsdf_setBssrdf(bsdf, bssrdf);
}

void
Closure_addVolumeSubsurface(varying Closure * uniform closure,
                            uniform ShadingTLState * uniform tls,
                            const varying State &state,
                            const varying Color &scale,
                            varying Fresnel * uniform fresnel,
                            const varying Vec3f &normal,
                            const varying Color &scatteringColor,
                            const varying Color &scatteringRadius,
                            uniform bool resolveSelfIntersections,
                            const uniform TraceSet * uniform traceSet,
                            const uniform Material * uniform material,
                            const uniform intptr_t evalNormalFn,
                            const uniform int label)
{
    varying Bsdf * uniform bsdf = asBsdf(closure);

    // lib/pbr only supports a single volume subsurface at the moment.
    // so adding a volume subsurface to a closure that already has one is
    // most likely a shader error.
    assert(bsdf->mVolumeSubsurface == NULL);

    uniform Arena * uniform arena = tls->mArena;

    varying VolumeSubsurface * uniform vs =
        (varying VolumeSubsurface * uniform)
        Arena_alloc(arena, sizeof(varying VolumeSubsurface));
    VolumeSubsurface_init(vs,
                          normal,
                          scatteringColor,
                          scatteringRadius,
                          resolveSelfIntersections,
                          material,
                          evalNormalFn,
                          scale,
                          fresnel,
                          traceSet);
    VolumeSubsurface_setLabel(vs, label);
    Bsdf_setVolumeSubsurface(bsdf, vs);
}

// ---------------------------------------------------------------------
// Fresnel Creation
// ---------------------------------------------------------------------

// create a SchlickFresnel with a memory arena
varying Fresnel * uniform
Fresnel_createSchlick(uniform ShadingTLState * uniform tls,
                      const varying State &state,
                      const varying Color &spec,
                      const varying float factor,
                      const varying float neta,
                      const varying float weight)
{
    uniform Arena * uniform arena = tls->mArena;

    varying SchlickFresnel * uniform fresnel =
        (varying SchlickFresnel * uniform)
        Arena_alloc(arena, sizeof(varying SchlickFresnel));

    SchlickFresnel_init(fresnel, spec, factor, neta, weight);
    return (varying Fresnel * uniform) fresnel;
}

// create a ConductorFresnel with a memory arena
varying Fresnel * uniform
Fresnel_createConductor(uniform ShadingTLState * uniform tls,
                        const varying State &state,
                        const varying Color &eta,
                        const varying Color &absorption,
                        const varying float weight)
                                   
{
    uniform Arena * uniform arena = tls->mArena;

    varying ConductorFresnel * uniform fresnel =
        (varying ConductorFresnel * uniform)
        Arena_alloc(arena, sizeof(varying ConductorFresnel));

    ConductorFresnel_init(fresnel, eta, absorption, weight);

    return (varying Fresnel * uniform) fresnel;
}

// create a SchlickFresnel with a memory arena
varying Fresnel * uniform
Fresnel_createDielectric(uniform ShadingTLState * uniform tls,
                         const varying State &state,
                         const varying float etaI,
                         const varying float etaT,
                         const varying float weight)
{
    uniform Arena * uniform arena = tls->mArena;

    varying DielectricFresnel * uniform fresnel =
        (varying DielectricFresnel * uniform)
        Arena_alloc(arena, sizeof(varying DielectricFresnel));

    DielectricFresnel_init(fresnel, etaI, etaT, weight);
    return (varying Fresnel * uniform) fresnel;
}

// create a OneMinusRoughSchlickFresnel with a memory arena
varying Fresnel * uniform
Fresnel_createOneMinusRoughSchlick(uniform ShadingTLState * uniform tls,
                                   const varying State &state,
                                   varying Fresnel * uniform schlick,
                                   varying float specRoughness)
                                   
{
    uniform Arena * uniform arena = tls->mArena;

    varying OneMinusRoughSchlickFresnel * uniform fresnel =
        (varying OneMinusRoughSchlickFresnel * uniform)
        Arena_alloc(arena, sizeof(varying OneMinusRoughSchlickFresnel));

    OneMinusRoughSchlickFresnel_init(fresnel, schlick, specRoughness);

    return (varying Fresnel * uniform) fresnel;
}

// create a OneMinusFresnel with a memory arena
varying Fresnel * uniform
Fresnel_createOneMinusFresnel(uniform ShadingTLState * uniform tls,
                              const varying State &state,
                              const varying Fresnel * uniform topFresnel)
{
    uniform Arena * uniform arena = tls->mArena;

    varying OneMinusFresnel * uniform fresnel =
        (varying OneMinusFresnel * uniform)
        Arena_alloc(arena, sizeof(varying OneMinusFresnel));

    OneMinusFresnel_init(fresnel, topFresnel);

    return (varying Fresnel * uniform) fresnel;
}

// create a OneMinusRoughFresnel with a memory arena
varying Fresnel * uniform
Fresnel_createOneMinusRoughFresnel(uniform ShadingTLState * uniform tls,
                                   const varying State &state,
                                   const varying Fresnel * uniform topFresnel,
                                   varying float specRoughness)

{
    uniform Arena * uniform arena = tls->mArena;

    varying OneMinusRoughFresnel * uniform fresnel =
        (varying OneMinusRoughFresnel * uniform)
        Arena_alloc(arena, sizeof(varying OneMinusRoughFresnel));

    OneMinusRoughFresnel_init(fresnel,
                              topFresnel,
                              specRoughness);

    return (varying Fresnel * uniform) fresnel;
}

// create a OneMinusVelvetFresnel with a memory arena
varying Fresnel * uniform
Fresnel_createOneMinusVelvetFresnel(uniform ShadingTLState * uniform tls,
                                    const varying State &state,
                                    const varying float roughness,
                                    const varying float weight)
{
    uniform Arena * uniform arena = tls->mArena;

    varying OneMinusVelvetFresnel * uniform fresnel =
        (varying OneMinusVelvetFresnel * uniform)
        Arena_alloc(arena, sizeof(varying OneMinusVelvetFresnel));

    OneMinusVelvetFresnel_init(fresnel,
                               roughness,
                               weight);

    return (varying Fresnel * uniform) fresnel;
}


varying Fresnel * uniform
Fresnel_createMultipleTransmissionFresnel(uniform ShadingTLState * uniform tls,
                                          const varying State &state,
                                          const varying Fresnel * uniform fresnel1,
                                          const varying Fresnel * uniform fresnel2,
                                          const varying Fresnel * uniform fresnel3,
                                          const varying float etaI,
                                          const varying float etaT,
                                          const varying float thickness,
                                          const varying float attnColorR,
                                          const varying float attnColorG,
                                          const varying float attnColorB,
                                          const uniform bool  useBending)
{
    uniform Arena * uniform arena = tls->mArena;

    varying MultipleTransmissionFresnel * uniform fresnel =
        (varying MultipleTransmissionFresnel * uniform)
        Arena_alloc(arena, sizeof(varying MultipleTransmissionFresnel));

    MultipleTransmissionFresnel_init(fresnel,
                                     etaI, etaT,
                                     thickness,
                                     attnColorR, attnColorG, attnColorB,
                                     useBending);

    if (isActive(fresnel1)) {
        MultipleTransmissionFresnel_add(fresnel, fresnel1);
    }
    if (isActive(fresnel2)) {
        MultipleTransmissionFresnel_add(fresnel, fresnel2);
    }
    if (isActive(fresnel3)) {
        MultipleTransmissionFresnel_add(fresnel, fresnel3);
    }
    return (varying Fresnel * uniform) fresnel;

}

// ---------------------------------------------------------------------
// BsdfLobe Creation
// ---------------------------------------------------------------------

// if we have an active underFresnel, then parent our lobe to
// an under lobe, set the under lobe's fresnel and add it,
// otherwise, just add the lobe.
inline void
addUnderLobe(uniform ShadingTLState * uniform tls,
             const varying State &state,
             uniform Arena * uniform arena,
             varying Bsdf * uniform bsdf,
             varying BsdfLobe * uniform lobe,
             varying Fresnel * uniform underFresnel, // fresnel of underLobe
             const varying Vec3f &normal,
             uniform int label)
{
    varying BsdfLobe * uniform underLobe;

    if (underFresnel->mUseBending) {

        varying Fresnel* uniform exitingFresnel = Fresnel_createDielectric(tls,
                                                                           state,
                                                                           underFresnel->mEtaT,
                                                                           underFresnel->mEtaI,
                                                                           underFresnel->mWeight);
        if (BsdfLobe_matchesFlag(lobe,
                                 BSDF_LOBE_TYPE_TRANSMISSION)) {
            underLobe = (varying BsdfLobe * uniform)
                    Arena_alloc(arena, sizeof(varying UnderClearcoatTransmissionBsdfLobe));
            UnderClearcoatTransmissionBsdfLobe_init((varying UnderClearcoatTransmissionBsdfLobe * uniform) underLobe,
                                                    lobe,
                                                    normal,
                                                    exitingFresnel,
                                                    underFresnel->mEtaI,
                                                    underFresnel->mEtaT,
                                                    underFresnel->mThickness,
                                                    underFresnel->mAttenuationColor,
                                                    1.0f);  // attenuationWeight: Only BaseMaterial
                                                            // uses the closure API and does not have clearcoat,
                                                            // nor a weighted clearcoat so no point in adding support
        } else {
            underLobe = (varying BsdfLobe * uniform)
                    Arena_alloc(arena, sizeof(varying UnderClearcoatBsdfLobe));
            UnderClearcoatBsdfLobe_init((varying UnderClearcoatBsdfLobe * uniform) underLobe,
                                    lobe,
                                    normal,
                                    exitingFresnel,
                                    underFresnel->mEtaI,
                                    underFresnel->mEtaT,
                                    underFresnel->mThickness,
                                    underFresnel->mAttenuationColor,
                                    1.0f);  // attenuationWeight: Only BaseMaterial
                                            // uses the closure API and does not have clearcoat,
                                            // nor a weighted clearcoat so no point in adding support
        }
    }
    else {
        underLobe = (varying BsdfLobe * uniform)
                Arena_alloc(arena, sizeof(varying UnderBsdfLobe));

        UnderBsdfLobe_init((varying UnderBsdfLobe * uniform) underLobe,
                           lobe,
                           normal,
                           underFresnel->mThickness,
                           underFresnel->mAttenuationColor,
                           1.0f);   // attenuationWeight: Only BaseMaterial
                                    // uses the closure API and does not have clearcoat,
                                    // nor a weighted clearcoat so no point in adding support
    }

    BsdfLobe_setFresnel(underLobe, underFresnel);
    BsdfLobe_setLabel(underLobe, label);
    Bsdf_addLobe(bsdf, underLobe);
}

/// A lobe with a well defined normal can be added directly to the
/// Closure with an assigned scale and fresnel.  Optionally, if an "underFresnel"
/// is also active, the lobe can be made a child of an UnderLobe, which
/// is assigned the under fresnel.  This UnderLobe is then added to the Bsdf.
inline void
addNormalBsdfLobeToClosure(uniform ShadingTLState * uniform tls,
                           const varying State &state,
                           uniform Arena * uniform arena,
                           varying Closure * uniform closure,
                           varying BsdfLobe * uniform lobe,
                           const varying Color &scale,
                           varying Fresnel * uniform fresnel,       // fresnel of lobe
                           varying Fresnel * uniform underFresnel,  // fresnel of underLobe
                           const varying Vec3f &normal,
                           uniform int label)
{
    varying Bsdf * uniform bsdf = asBsdf(closure);

    if (isActive(fresnel)) {
        BsdfLobe_setFresnel(lobe, fresnel);
    }
    BsdfLobe_setScale(lobe, scale);

    // if we have one active underFresnel, then parent our lobe to
    // an under lobe, set the under lobe's fresnel and add it,
    // otherwise, just add the lobe.
    //
    // if we are using an underLobe, then the label must be placed
    // on the underLobe.  it wouldn't hurt to place them on the child
    // lobe, but shouldn't be needed.
    //
    // WARNING:  Bsdf_addLobe() changes the uniform mask of the lobe
    // to match the lanemask() at the time of the addition.  This means
    // that the mask on the underlobe may not match the lobe in the case where
    // lobe is both added to the underlobe and to the bsdf directly.  We
    // should verify that this is safe.  This should work fine, but the
    // underlobe needs to not pay attention to the mask on the owned lobe.
    if (isActive(underFresnel)) {
        addUnderLobe(tls,
                     state,
                     arena,
                     bsdf,
                     lobe,
                     underFresnel,
                     normal,
                     label);
    } else {
        Bsdf_addLobe(bsdf, lobe);
        BsdfLobe_setLabel(lobe, label);
    }
}



// allocate and init an ansiotropic cook-torrance lobe
void
Closure_addAnisoCookTorranceBsdfLobe(varying Closure * uniform closure,
                                     uniform ShadingTLState * uniform tls,
                                     const varying State &state,
                                     const varying Color &scale,
                                     varying Fresnel * uniform fresnel,
                                     varying Fresnel * uniform underFresnel,
                                     const varying Vec3f &normal,
                                     const varying Vec3f &anisotropicDirection,
                                     const varying float uRoughness,
                                     const varying float vRoughness,
                                     uniform int label)
{
    uniform Arena * uniform arena = tls->mArena;

    varying BsdfLobe * uniform lobe = (varying BsdfLobe * uniform)
        Arena_alloc(arena, sizeof(varying AnisoCookTorranceBsdfLobe));

    // Adapt normal to prevent reflection ray from self-intersecting this geometry
    const Vec3f adaptedNormal = Intersection_adaptNormal((const varying Intersection &) state, normal);

    AnisoCookTorranceBsdfLobe_init((varying AnisoCookTorranceBsdfLobe * uniform) lobe, adaptedNormal,
                                   anisotropicDirection, uRoughness, vRoughness);
    addNormalBsdfLobeToClosure(tls, state, arena, closure, lobe, scale, fresnel, underFresnel, adaptedNormal, label);
}

// allocate and init a Cook-Torrance lobe
void
Closure_addCookTorranceBsdfLobe(varying Closure * uniform closure,
                                uniform ShadingTLState * uniform tls,
                                const varying State &state,
                                const varying Color &scale,
                                varying Fresnel * uniform fresnel,
                                varying Fresnel * uniform underFresnel,
                                const varying Vec3f &normal,
                                const varying float roughness,
                                const varying Color favg,
                                const varying Color favgInv,
                                uniform int label)
{
    uniform Arena * uniform arena = tls->mArena;

    varying BsdfLobe * uniform lobe = (varying BsdfLobe * uniform)
        Arena_alloc(arena, sizeof(varying CookTorranceBsdfLobe));

    // Adapt normal to prevent reflection ray from self-intersecting this geometry
    const Vec3f adaptedNormal = Intersection_adaptNormal((const varying Intersection &) state, normal);

    CookTorranceBsdfLobe_init((varying CookTorranceBsdfLobe * uniform) lobe, adaptedNormal, roughness,
                              favg, favgInv);
    addNormalBsdfLobeToClosure(tls, state, arena, closure, lobe, scale, fresnel, underFresnel, adaptedNormal, label);
}

// allocate and init a GGX Cook-Torrance lobe
void
Closure_addGGXCookTorranceBsdfLobe(varying Closure * uniform closure,
                                   uniform ShadingTLState * uniform tls,
                                   const varying State &state,
                                   const varying Color &scale,
                                   varying Fresnel * uniform fresnel,
                                   varying Fresnel * uniform underFresnel,
                                   const varying Vec3f &normal,
                                   const varying float roughness,
                                   uniform int label)
{
    uniform Arena * uniform arena = tls->mArena;

    varying BsdfLobe * uniform lobe = (varying BsdfLobe * uniform)
        Arena_alloc(arena, sizeof(varying GGXCookTorranceBsdfLobe));

    // Adapt normal to prevent reflection ray from self-intersecting this geometry
    const Vec3f adaptedNormal = Intersection_adaptNormal((const varying Intersection &) state, normal);

    GGXCookTorranceBsdfLobe_init((varying GGXCookTorranceBsdfLobe * uniform) lobe, adaptedNormal, roughness);
    addNormalBsdfLobeToClosure(tls, state, arena, closure, lobe, scale, fresnel, underFresnel , adaptedNormal, label);
}

// allocate and init a Glitter GGX Cook-Torrance lobe
void
Closure_addGlitterGGXCookTorranceBsdfLobe(varying Closure * uniform closure,
                                          uniform ShadingTLState * uniform tls,
                                          const varying State &state,
                                          const varying Color &scale,
                                          varying Fresnel * uniform fresnel,
                                          varying Fresnel * uniform underFresnel,
                                          const varying Vec3f &normal,
                                          const varying Vec3f &flakeNormal,
                                          const varying float roughness,
                                          const varying Color& favg,
                                          uniform int label)
{
    uniform Arena * uniform arena = tls->mArena;

    varying BsdfLobe * uniform lobe = (varying BsdfLobe * uniform)
        Arena_alloc(arena, sizeof(varying GlitterGGXCookTorranceBsdfLobe));

    // Adapt normal to prevent reflection ray from self-intersecting this geometry
    const Vec3f adaptedNormal = Intersection_adaptNormal((const varying Intersection &) state, normal);

    GlitterGGXCookTorranceBsdfLobe_init((varying GlitterGGXCookTorranceBsdfLobe * uniform) lobe, adaptedNormal,
        flakeNormal, roughness, favg);
    addNormalBsdfLobeToClosure(tls, state, arena, closure, lobe, scale, fresnel, underFresnel, adaptedNormal, label);
}

void Closure_addTransmissionCookTorranceBsdfLobe(varying Closure * uniform closure,
                                     uniform ShadingTLState * uniform tls,
                                     const varying State &state,
                                     const varying Color &scale,
                                     varying Fresnel * uniform fresnel,
                                     varying Fresnel * uniform underFresnel,
                                     const varying Vec3f &normal,
                                     const varying float roughness,
                                     const varying float iorIncident,
                                     const varying float iorTransmitted,
                                     const varying float favg,
                                     const varying float favgInv,
                                     const varying float abbeNumber,
                                     uniform int label)
{
    uniform Arena * uniform arena = tls->mArena;

    varying BsdfLobe * uniform lobe = (varying BsdfLobe * uniform)
            Arena_alloc(arena, sizeof(varying TransmissionCookTorranceBsdfLobe));

    TransmissionCookTorranceBsdfLobe_init(
            (varying TransmissionCookTorranceBsdfLobe * uniform) lobe, normal,
            roughness, iorIncident, iorTransmitted, sWhite, favg, favgInv, abbeNumber);
    addNormalBsdfLobeToClosure(tls, state, arena, closure, lobe, scale, fresnel, underFresnel,
                               normal, label);
}

// allocate and init an iridescence Cook-Torrance lobe
void
Closure_addIridescenceBsdfLobe(varying Closure * uniform closure,
                               uniform ShadingTLState * uniform tls,
                               const varying State &state,
                               const varying Color &scale,
                               varying Fresnel * uniform fresnel,
                               varying Fresnel * uniform underFresnel,
                               const varying Vec3f &normal,
                               const varying float roughness,
                               const varying float anisotropy,
                               const varying Vec3f &anisotropicDirection,
                               const varying float uRoughness,
                               const varying float vRoughness,
                               const varying float iridescence,
                               const varying Color &primary,
                               const varying Color &secondary,
                               const varying bool flipHue,
                               const varying float thickness,
                               const varying float exponent,
                               const varying float iridescenceAt0,
                               const varying float iridescenceAt90,
                               const varying int specularModel,
                               uniform int label)
{
    uniform Arena * uniform arena = tls->mArena;

    varying BsdfLobe * uniform lobe = (varying BsdfLobe * uniform)
            Arena_alloc(arena, sizeof(varying IridescenceBsdfLobe));

    // Adapt normal to prevent reflection ray from self-intersecting this geometry
    const Vec3f adaptedNormal = Intersection_adaptNormal((const varying Intersection &) state, normal);

    IridescenceBsdfLobe_init(
            (varying IridescenceBsdfLobe * uniform) lobe, arena, normal, iridescence,
            (uniform SHADING_IridescenceColorMode)0 /* color control */, primary, secondary, flipHue,
            (uniform ColorRampControlSpace)0 /* ramp interpolation mode */, 0 /* num points*/,
            NULL /* positions */, NULL /* interpolators */, NULL /* colors */,
            thickness, exponent, iridescenceAt0, iridescenceAt90);

    // *Warning* : Setting a uniform pointer based on varying conditional
    // We only have one BsdfLobe* pointer for all lanes. To ensure only one type of
    // child specular lobe gets created, we're handling this as a special case

    if (all(roughness < sEpsilon)) { // cheapest - all lanes get mirror lobe
        varying BsdfLobe * uniform mirror = (varying BsdfLobe * uniform)
            Arena_alloc(arena, sizeof(varying MirrorReflectionBsdfLobe));
        MirrorReflectionBsdfLobe_init((varying MirrorReflectionBsdfLobe * uniform) mirror, adaptedNormal);

        IridescenceBsdfLobe_setChildLobe((varying IridescenceBsdfLobe * uniform)lobe, mirror);
    } else {
        if (any(!isZero(anisotropy))) { // at least one lane is anisotropic - create all anisotropic
            varying BsdfLobe * uniform aniso = (varying BsdfLobe * uniform)
                Arena_alloc(arena, sizeof(varying AnisoCookTorranceBsdfLobe));
            AnisoCookTorranceBsdfLobe_init((varying AnisoCookTorranceBsdfLobe * uniform) aniso, adaptedNormal,
                                           anisotropicDirection, uRoughness, vRoughness);

            IridescenceBsdfLobe_setChildLobe((varying IridescenceBsdfLobe * uniform)lobe, aniso);
        } else { // at least one lane is cook torrance, others mirror - create all cook torrance


            varying BsdfLobe * uniform cookTorrance = (varying BsdfLobe * uniform)
                Arena_alloc(arena, sizeof(varying CookTorranceBsdfLobe));

            if (specularModel == GGX) {
                GGXCookTorranceBsdfLobe_init((varying GGXCookTorranceBsdfLobe * uniform) cookTorrance, adaptedNormal, roughness);
            } else {
                CookTorranceBsdfLobe_init((varying CookTorranceBsdfLobe * uniform) cookTorrance, adaptedNormal, roughness);
            }

            IridescenceBsdfLobe_setChildLobe((varying IridescenceBsdfLobe * uniform)lobe, cookTorrance);
        }
    }

    varying BsdfLobe * uniform spec = ((varying IridescenceBsdfLobe * uniform) lobe)->mChildLobe;
    // Set fresnel and scale on the child specular lobe
    if (isActive(fresnel)) {
        BsdfLobe_setFresnel(spec, fresnel);
    }
    BsdfLobe_setScale(spec, scale);

    const varying Color iridescenceScale = sWhite;
    addNormalBsdfLobeToClosure(tls, state, arena, closure, lobe, iridescenceScale, nullptr, underFresnel, normal, label);
}

// allocate and init a lambert lobe
void
Closure_addLambertBsdfLobe(varying Closure * uniform closure,
                           uniform ShadingTLState * uniform tls,
                           const varying State &state,
                           const varying Color &scale,
                           varying Fresnel * uniform fresnel,
                           varying Fresnel * uniform underFresnel,
                           const varying Vec3f &normal,
                           const varying Color &albedo,
                           uniform bool isReflection,
                           uniform int label)
{
    uniform Arena * uniform arena = tls->mArena;

    varying BsdfLobe * uniform lobe = (varying BsdfLobe * uniform)
        Arena_alloc(arena, sizeof(varying LambertBsdfLobe));

    LambertBsdfLobe_init((varying LambertBsdfLobe * uniform) lobe, normal, albedo, isReflection);
    addNormalBsdfLobeToClosure(tls, state, arena, closure, lobe, scale, fresnel, underFresnel, normal, label);
}

// allocate and init a Retroreflection lobe
void
Closure_addRetroreflectionBsdfLobe(varying Closure * uniform closure,
                                   uniform ShadingTLState * uniform tls,
                                   const varying State &state,
                                   const varying Color &scale,
                                   varying Fresnel * uniform fresnel,
                                   varying Fresnel * uniform underFresnel,
                                   const varying Vec3f &normal,
                                   const varying float roughness,
                                   uniform int label)
{
    uniform Arena * uniform arena = tls->mArena;

    varying BsdfLobe * uniform lobe = (varying BsdfLobe * uniform)
        Arena_alloc(arena, sizeof(varying RetroreflectionBsdfLobe));

    RetroreflectionBsdfLobe_init((varying RetroreflectionBsdfLobe * uniform) lobe, normal, roughness);
    addNormalBsdfLobeToClosure(tls, state, arena, closure, lobe, scale, fresnel, underFresnel, normal, label);
}

// allocate and init a EyeCaustic lobe
void
Closure_addEyeCausticBsdfLobe(varying Closure * uniform closure,
                              uniform ShadingTLState * uniform tls,
                              const varying State &state,
                              const varying Color &scale,
                              varying Fresnel * uniform fresnel,
                              varying Fresnel * uniform underFresnel,
                              const varying Vec3f &normal,
                              const varying Vec3f &irisN,
                              const varying float exponent,
                              uniform int labels)
{
    uniform Arena * uniform arena = tls->mArena;

    varying BsdfLobe * uniform lobe = (varying BsdfLobe * uniform)
        Arena_alloc(arena, sizeof(varying EyeCausticBsdfLobe));

    // Adapt normal to prevent reflection ray from self-intersecting this geometry
    const Vec3f adaptedNormal = Intersection_adaptNormal((const varying Intersection &) state, normal);

    EyeCausticBsdfLobe_init((varying EyeCausticBsdfLobe * uniform) lobe, adaptedNormal, irisN, exponent, (varying Color) sWhite);
    addNormalBsdfLobeToClosure(tls, state, arena, closure, lobe, scale, fresnel, underFresnel, adaptedNormal, labels);
}

// allocate and init a mirror reflection lobe
void
Closure_addMirrorReflectionBsdfLobe(varying Closure * uniform closure,
                                    uniform ShadingTLState * uniform tls,
                                    const varying State &state,
                                    const varying Color &scale,
                                    varying Fresnel * uniform fresnel,
                                    varying Fresnel * uniform underFresnel,
                                    const varying Vec3f &normal,
                                    uniform int label)

{
    uniform Arena * uniform arena = tls->mArena;

    varying BsdfLobe * uniform lobe = (varying BsdfLobe * uniform)
        Arena_alloc(arena, sizeof(varying MirrorReflectionBsdfLobe));

    // Adapt normal to prevent reflection ray from self-intersecting this geometry
    const Vec3f adaptedNormal = Intersection_adaptNormal((const varying Intersection &) state, normal);

    MirrorReflectionBsdfLobe_init((varying MirrorReflectionBsdfLobe * uniform) lobe, adaptedNormal);
    addNormalBsdfLobeToClosure(tls, state, arena, closure, lobe, scale, fresnel,
                               underFresnel, adaptedNormal, label);
}

// allocate and init a mirror transmission lobe
void
Closure_addMirrorTransmissionBsdfLobe(varying Closure * uniform closure,
                                      uniform ShadingTLState * uniform tls,
                                      const varying State &state,
                                      const varying Color &scale,
                                      varying Fresnel * uniform fresnel,
                                      varying Fresnel * uniform underFresnel,
                                      const varying Vec3f &normal,
                                      const varying float etaI,
                                      const varying float etaT,
                                      const varying float abbeNumber,
                                      uniform int label)
{
    uniform Arena * uniform arena = tls->mArena;

    varying BsdfLobe * uniform lobe = (varying BsdfLobe * uniform)
        Arena_alloc(arena, sizeof(varying MirrorTransmissionBsdfLobe));

    MirrorTransmissionBsdfLobe_init((varying MirrorTransmissionBsdfLobe * uniform) lobe,
                                    normal,
                                    etaI, etaT,
                                    (varying Color) sWhite,
                                    abbeNumber);
    addNormalBsdfLobeToClosure(tls, state, arena, closure, lobe, scale, fresnel, underFresnel,
                               normal, label);
}

// allocate and init a mirror retroreflection lobe
void
Closure_addMirrorRetroreflectionBsdfLobe(varying Closure * uniform closure,
                                         uniform ShadingTLState * uniform tls,
                                         const varying State &state,
                                         const varying Color &scale,
                                         varying Fresnel * uniform fresnel,
                                         varying Fresnel * uniform underFresnel,
                                         const varying Vec3f &normal,
                                         uniform int label)

{
    uniform Arena * uniform arena = tls->mArena;

    varying BsdfLobe * uniform lobe = (varying BsdfLobe * uniform)
        Arena_alloc(arena, sizeof(varying MirrorRetroreflectionBsdfLobe));

    MirrorRetroreflectionBsdfLobe_init((varying MirrorRetroreflectionBsdfLobe * uniform) lobe, normal);
    addNormalBsdfLobeToClosure(tls, state, arena, closure, lobe, scale, fresnel, underFresnel, normal, label);
}

// allocate and init a fabric lobe
void
Closure_addFabricBsdfLobe(varying Closure * uniform closure,
                          uniform ShadingTLState * uniform tls,
                          const varying State &state,
                          const varying Color &scale,
                          varying Fresnel * uniform fresnel,
                          varying Fresnel * uniform underFresnel,
                          const varying Vec3f &normal,
                          const varying Vec3f &tangent,
                          const varying Vec3f &threadDirection,
                          const varying float threadRotation,
                          const varying float roughness,
                          const uniform int fabricModel,
                          const uniform int label)
{
    uniform Arena * uniform arena = tls->mArena;

    varying BsdfLobe * uniform lobe = (varying BsdfLobe * uniform)
        Arena_alloc(arena, sizeof(varying FabricBsdfLobe));

    FabricBsdfLobe_init((varying FabricBsdfLobe * uniform) lobe,
                         normal, tangent, threadDirection, threadRotation, roughness, (varying Color) sWhite, fabricModel);

    addNormalBsdfLobeToClosure(tls, state, arena, closure, lobe, scale, fresnel, underFresnel, normal, label);
}

// allocate and init a velvet fabric lobe
void
Closure_addFabricVelvetBsdfLobe(varying Closure * uniform closure,
                                uniform ShadingTLState * uniform tls,
                                const varying State &state,
                                const varying Color &scale,
                                varying Fresnel * uniform fresnel,
                                varying Fresnel * uniform underFresnel,
                                const varying Vec3f &normal,
                                const varying float roughness,
                                const uniform int label)
{
    uniform Arena * uniform arena = tls->mArena;

    varying BsdfLobe * uniform lobe = (varying BsdfLobe * uniform)
        Arena_alloc(arena, sizeof(varying FabricVelvetBsdfLobe));

    FabricVelvetBsdfLobe_init((varying FabricVelvetBsdfLobe * uniform) lobe,
                              normal,
                              roughness,
                              (varying Color) sWhite);

    addNormalBsdfLobeToClosure(tls, state, arena, closure, lobe, scale, fresnel, underFresnel, normal, label);
}

// allocate and init a fabric lobe
void
Closure_addFabricBsdfLobe(varying Closure * uniform closure,
                          uniform ShadingTLState * uniform tls,
                          const varying State &state,
                          const varying Color &scale,
                          varying Fresnel * uniform fresnel,
                          varying Fresnel * uniform underFresnel,
                          const varying Vec3f &normal,
                          const varying Vec3f &tangent,
                          const varying Vec3f &threadDirection,
                          const uniform float threadElevation,
                          const varying float specularExponent,
                          const uniform bool  useSineLobe,
                          const uniform int label)
{
    uniform Arena * uniform arena = tls->mArena;

    varying BsdfLobe * uniform lobe = (varying BsdfLobe * uniform)
        Arena_alloc(arena, sizeof(varying FabricBsdfLobe));

    FabricBsdfLobe_init((varying FabricBsdfLobe * uniform) lobe,
                        normal,
                        tangent,
                        threadDirection,
                        threadElevation,
                        specularExponent,
                        (varying Color) sWhite,
                        useSineLobe);

    addNormalBsdfLobeToClosure(tls, state, arena, closure,
                               lobe, scale,
                               fresnel, underFresnel,
                               normal, label);
}

// allocate and init a ward duer lobe
void 
Closure_addWardDuerBsdfLobe(varying Closure * uniform closure,
                            uniform ShadingTLState * uniform tls,
                            const varying State &state,
                            const varying Color &scale,
                            varying Fresnel * uniform fresnel,
                            varying Fresnel * uniform underFresnel,
                            const varying Vec3f &normal,
                            const varying Vec3f &anisoDirection,
                            const varying float roughnessU,
                            const varying float roughnessV,
                            const uniform bool squareRoughness,
                            uniform int label)
{
    uniform Arena * uniform arena = tls->mArena;
    
    varying BsdfLobe * uniform lobe = (varying BsdfLobe * uniform)
        Arena_alloc(arena, sizeof(varying WardDuerBsdfLobe));

    // Adapt normal to prevent reflection ray from self-intersecting this geometry
    const Vec3f adaptedNormal = Intersection_adaptNormal((const varying Intersection &) state, normal);

    WardDuerBsdfLobe_init((varying WardDuerBsdfLobe * uniform) lobe,
                          adaptedNormal, anisoDirection, roughnessU, roughnessV, squareRoughness);
    addNormalBsdfLobeToClosure(tls, state, arena, closure, lobe, scale, fresnel, underFresnel, adaptedNormal, label);
}

// allocate and init a stochastic flakes lobe
void
Closure_addStochasticFlakesBsdfLobe(varying Closure * uniform closure,
                                    uniform ShadingTLState * uniform tls,
                                    const varying State &state,
                                    const varying Color &scale,
                                    const varying Vec3f& normal,
                                    const varying Vec3f * varying flakeNormals,
                                    const varying Color * varying flakeColors,
                                    const varying size_t flakeCount,
                                    const varying float roughness,
                                    const varying float inputFlakeRandomness,
                                    const uniform int label = -1)
{
    uniform Arena * uniform arena = tls->mArena;

    varying BsdfLobe * uniform lobe = (varying BsdfLobe * uniform)
        Arena_alloc(arena, sizeof(varying StochasticFlakesBsdfLobe));

    // Adapt normal to prevent reflection ray from self-intersecting this geometry
    const Vec3f adaptedNormal = Intersection_adaptNormal((const varying Intersection &) state, normal);

    StochasticFlakesBsdfLobe_init((varying StochasticFlakesBsdfLobe * uniform) lobe, adaptedNormal, flakeNormals,
                                  flakeColors, flakeCount, roughness, inputFlakeRandomness);
    addNormalBsdfLobeToClosure(tls, state, arena, closure, lobe, scale, nullptr, nullptr, adaptedNormal, label);
}

// ---------------------------------------------------------------------
// Misc
// ---------------------------------------------------------------------

/// @brief take the lobes from @param bsdf and add them the @param closure,
/// applying a scale if necessary
void
Closure_addBsdfLobes(varying Closure * uniform closure,
                     varying Bsdf * uniform bsdf,
                     const varying float scale,
                     const uniform bool overrideBssrdfAndEmission)
{
    varying Bsdf * uniform finalBsdf = asBsdf(closure);
    uniform int numLobes = bsdf->mNumLobes;

    // Re-weight sub material's lobes with scale and add them
    // to the final material's bsdf
    for (uniform int i = 0; i < numLobes; ++i) {
        if (finalBsdf->mNumLobes < BSDF_MAX_LOBE) {
            varying BsdfLobe * uniform lobe = bsdf->mLobes[i];
            const Color finalScale = lobe->mScale * scale;
            BsdfLobe_setScale(lobe, finalScale);
            if (isActive(lobe)) {
                Bsdf_addLobe(finalBsdf, lobe);
            }
        }
    }

    if (overrideBssrdfAndEmission) {
        // Set closure's bssrdf and emission to that of bsdf
        Bsdf_setBssrdf(finalBsdf, bsdf->mBssrdf);
        Bsdf_setSelfEmission(finalBsdf, Bsdf_getSelfEmission(bsdf));
    }
}

/// @brief set early termination on a closure
void
Closure_setEarlyTermination(varying Closure * uniform closure,
                            uniform ShadingTLState * uniform tls,
                            const varying State &state)
{
    uniform Arena * uniform arena = tls->mArena;

    Bsdf_setEarlyTermination(asBsdf(closure), lanemask());
}

/// @brief set self emission on a closure
void
Closure_setSelfEmission(varying Closure * uniform closure, const varying Color &emissionColor)
{
    Bsdf_setSelfEmission(asBsdf(closure), emissionColor);
}

uniform int
Closure_getSize()
{
    return sizeof(varying Bsdf);
}
