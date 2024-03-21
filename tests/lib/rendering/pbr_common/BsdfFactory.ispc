// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

/// @file BsdfFactory.ispc

#include <moonray/rendering/shading/ispc/bsdf/Bsdf.isph>
#include <moonray/rendering/shading/ispc/bsdf/ashikhmin_shirley/BsdfAshikhminShirleyDiffuse.isph>
#include <moonray/rendering/shading/ispc/bsdf/ashikhmin_shirley/BsdfAshikhminShirleyGlossy.isph>
#include <moonray/rendering/shading/ispc/bsdf/BsdfLambert.isph>
#include <moonray/rendering/shading/ispc/bsdf/cook_torrance/BsdfAnisoCookTorrance.isph>
#include <moonray/rendering/shading/ispc/bsdf/BsdfIridescence.isph>
#include <moonray/rendering/shading/ispc/bsdf/cook_torrance/BsdfCookTorrance.isph>
#include <moonray/rendering/shading/ispc/bsdf/cook_torrance/BsdfGGXCookTorrance.isph>
#include <moonray/rendering/shading/ispc/bsdf/cook_torrance/BsdfTransmissionCookTorrance.isph>
#include <moonray/rendering/shading/ispc/bsdf/BsdfRetroreflection.isph>
#include <moonray/rendering/shading/ispc/bsdf/BsdfEyeCaustic.isph>
#include <moonray/rendering/shading/ispc/bsdf/fabric/BsdfFabric.isph>
#include <moonray/rendering/shading/ispc/bsdf/fabric/BsdfFabricVelvet.isph>
#include <moonray/rendering/shading/ispc/bsdf/hair/BsdfHairDiffuse.isph>
#include <moonray/rendering/shading/ispc/bsdf/hair/BsdfHairLobes.isph>
#include <moonray/rendering/shading/ispc/bsdf/ward/BsdfWardCorrected.isph>
#include <moonray/rendering/shading/ispc/bsdf/ward/BsdfWardDuer.isph>
#include <moonray/rendering/shading/ispc/bsdf/BsdfStochasticFlakes.isph>
#include <moonray/rendering/shading/ispc/bsdf/Fresnel.isph>
#include <moonray/rendering/shading/ispc/Util.isph>

#include <scene_rdl2/common/math/ispc/ReferenceFrame.isph>
#include <scene_rdl2/render/util/Arena.isph>

static varying Bsdf * uniform
createBsdf(uniform Arena * uniform arena)
{
    varying Bsdf * uniform bsdf =
        (varying Bsdf * uniform) Arena_alloc(arena, sizeof(Bsdf));
    Bsdf_init(bsdf);

    return bsdf;
}

static varying Fresnel * uniform
createSchlickFresnel(uniform Arena * uniform arena, varying Color scale)
{
    varying Fresnel * uniform f = (varying Fresnel * uniform)
        Arena_alloc(arena, sizeof(SchlickFresnel));
    SchlickFresnel_init((varying SchlickFresnel * uniform) f, scale, 1.f);

    return f;
}

static varying BsdfLobe * uniform
allocLobe(uniform Arena * uniform arena, uniform size_t size)
{
    return (varying BsdfLobe * uniform) Arena_alloc(arena, size);
}

#pragma ignore warning(all)
export varying Bsdf * uniform
LambertBsdfFactory(uniform Arena * uniform arena,
                   const uniform ReferenceFrame * uniform frame)
{
    varying Bsdf * uniform bsdf = createBsdf(arena);

    const varying Vec3f N = getN(*frame);

    varying BsdfLobe * uniform lobe = allocLobe(arena, sizeof(LambertBsdfLobe));
    LambertBsdfLobe_init((varying LambertBsdfLobe * uniform) lobe, N, (varying Color)sWhite, true);

    Bsdf_addLobe(bsdf, lobe);

    return bsdf;
}

#pragma ignore warning(all)
export varying Bsdf * uniform
CookTorranceBsdfFactory(uniform Arena * uniform arena,
                        const uniform ReferenceFrame * uniform frame,
                        uniform float roughness)
{
    varying Bsdf * uniform bsdf = createBsdf(arena);

    const varying Vec3f N = getN(*frame);

    float favg, favgInv;
    averageFresnelReflectance(1.5f,
                              favg, favgInv);

    varying BsdfLobe * uniform lobe = allocLobe(arena, sizeof(CookTorranceBsdfLobe));
    CookTorranceBsdfLobe_init((varying CookTorranceBsdfLobe * uniform) lobe, N, roughness,
                              sWhite*favg, sWhite*favgInv);

    varying Fresnel * uniform fresnel = createSchlickFresnel(arena, Color_ctor(1.f));
    BsdfLobe_setFresnel(lobe, fresnel);

    Bsdf_addLobe(bsdf, lobe);

    return bsdf;
}

#pragma ignore warning(all)
export varying Bsdf * uniform
GGXCookTorranceBsdfFactory(uniform Arena * uniform arena,
                           const uniform ReferenceFrame * uniform frame,
                           uniform float roughness)
{
    varying Bsdf * uniform bsdf = createBsdf(arena);

    const varying Vec3f N = getN(*frame);

    varying BsdfLobe * uniform lobe = allocLobe(arena, sizeof(GGXCookTorranceBsdfLobe));
    GGXCookTorranceBsdfLobe_init((varying GGXCookTorranceBsdfLobe * uniform) lobe, N, roughness);

    Bsdf_addLobe(bsdf, lobe);

    return bsdf;
}

#pragma ignore warning(all)
export varying Bsdf * uniform
AnisoCookTorranceBsdfFactory(uniform Arena * uniform arena,
                             const uniform ReferenceFrame * uniform frame,
                             uniform float roughnessU,
                             uniform float roughnessV)
{
    varying Bsdf * uniform bsdf = createBsdf(arena);

    const varying Vec3f N = getN(*frame);
    const varying Vec3f T = frame->mX;

    varying BsdfLobe * uniform lobe = allocLobe(arena, sizeof(AnisoCookTorranceBsdfLobe));
    AnisoCookTorranceBsdfLobe_init((varying AnisoCookTorranceBsdfLobe * uniform) lobe,
                                   N, T, roughnessU, roughnessV);

    Bsdf_addLobe(bsdf, lobe);

    return bsdf;
}

#pragma ignore warning(all)
export varying Bsdf * uniform
TransmissionCookTorranceBsdfFactory(uniform Arena * uniform arena,
                             const uniform ReferenceFrame * uniform frame,
                             const uniform float roughness,
                             const uniform float iorIncident,
                             const uniform float iorTransmitted)
{
    varying Bsdf * uniform bsdf = createBsdf(arena);

    const varying Vec3f N = getN(*frame);

    float favg, favgInv;
    averageFresnelReflectance(iorTransmitted/iorIncident,
                              favg, favgInv);

    varying BsdfLobe * uniform lobe = allocLobe(arena,
            sizeof(TransmissionCookTorranceBsdfLobe));
    TransmissionCookTorranceBsdfLobe_init((varying TransmissionCookTorranceBsdfLobe * uniform) lobe,
                                          N,
                                          roughness,
                                          iorIncident,
                                          iorTransmitted,
                                          (varying Color) sWhite /*tint*/,
                                          favg, favgInv,
                                          0.0f /*abbeNumber*/);

    Bsdf_addLobe(bsdf, lobe);

    return bsdf;
}

#pragma ignore warning(all)
export varying Bsdf * uniform
IridescenceBsdfFactory(uniform Arena * uniform arena,
                       const uniform ReferenceFrame * uniform frame,
                       const uniform float roughness,
                       const uniform float iridescence,
                       const uniform SHADING_IridescenceColorMode colorControl,
                       const uniform Color &primary,
                       const uniform Color &secondary,
                       const uniform bool flipHue,
                       const uniform ColorRampControlSpace rampInterpolationMode,
                       const uniform size_t numPoints,
                       const uniform float * uniform positions,
                       const uniform RampInterpolatorMode * uniform interpolators,
                       const uniform Color * uniform colors,
                       const uniform float thickness,
                       const uniform float exponent,
                       const uniform float iridescenceAt0,
                       const uniform float iridescenceAt90)
{
    varying Bsdf * uniform bsdf = createBsdf(arena);

    const varying Vec3f N = getN(*frame);

    const varying Color prim = primary;
    const varying Color sec = secondary;

    // Testing only CookTorrance as child lobe
    varying BsdfLobe * uniform cLobe = allocLobe(arena, sizeof(CookTorranceBsdfLobe));
    CookTorranceBsdfLobe_init((varying CookTorranceBsdfLobe * uniform) cLobe, N, roughness);

    // allocate varying pointers locally to copy from uniform data. type-casting from uniform to varying
    // is incorrect
    varying float* uniform positionsV = (varying float* uniform)
        Arena_allocArray(arena, numPoints, sizeof(float));
    varying Color* uniform colorsV = (varying Color* uniform)
        Arena_allocArray(arena, numPoints, sizeof(Color));
    varying RampInterpolatorMode* uniform interpolatorsV = (varying RampInterpolatorMode* uniform)
        Arena_allocArray(arena, numPoints, sizeof(int));
    for(int i=0; i<numPoints; i++) {
        positionsV[i] = positions[i];
        colorsV[i] = colors[i];
        interpolatorsV[i] = (varying RampInterpolatorMode)interpolators[i];
    }

    varying BsdfLobe * uniform lobe = allocLobe(arena, sizeof(IridescenceBsdfLobe));
    IridescenceBsdfLobe_init((varying IridescenceBsdfLobe * uniform) lobe, arena, N, iridescence,
            colorControl, prim, sec, flipHue,
            rampInterpolationMode, numPoints, positionsV, interpolatorsV, colorsV,
            thickness, exponent, iridescenceAt0, iridescenceAt90);

    IridescenceBsdfLobe_setChildLobe((varying IridescenceBsdfLobe * uniform)lobe, cLobe);

    Bsdf_addLobe(bsdf, lobe);

    return bsdf;
}

#pragma ignore warning(all)
export varying Bsdf * uniform
RetroreflectionBsdfFactory(uniform Arena * uniform arena,
                           const uniform ReferenceFrame * uniform frame,
                           uniform float roughness)
{
    varying Bsdf * uniform bsdf = createBsdf(arena);

    const varying Vec3f N = getN(*frame);

    varying BsdfLobe * uniform lobe = allocLobe(arena, sizeof(RetroreflectionBsdfLobe));
    RetroreflectionBsdfLobe_init((varying RetroreflectionBsdfLobe * uniform) lobe, N, roughness);

    Bsdf_addLobe(bsdf, lobe);

    return bsdf;
}

#pragma ignore warning(all)
export varying Bsdf * uniform
EyeCausticBsdfFactory(uniform Arena * uniform arena,
                      const uniform ReferenceFrame * uniform frame,
                      uniform float roughness)
{
    varying Bsdf * uniform bsdf = createBsdf(arena);

    const varying Vec3f N = getN(*frame);
    //convert roughness to phong exponent
    const float exponent = 2.0f * rcp(roughness * roughness);

    varying BsdfLobe * uniform lobe = allocLobe(arena, sizeof(EyeCausticBsdfLobe));
    EyeCausticBsdfLobe_init((varying EyeCausticBsdfLobe * uniform) lobe,
                            N, N, exponent, (varying Color) sWhite);

    Bsdf_addLobe(bsdf, lobe);

    return bsdf;
}

#pragma ignore warning(all)
export varying Bsdf * uniform
FabricBsdfFactory(uniform Arena * uniform arena,
                  const uniform ReferenceFrame * uniform frame,
                  uniform float roughness,
                  uniform bool useDwaFabric)
{
    varying Bsdf * uniform bsdf = createBsdf(arena);

    const varying Vec3f N = getN(*frame);
    const varying Vec3f T = (*frame).mX;
    const varying Vec3f threadDir = Vec3f_ctor(1,0,0);

    varying BsdfLobe * uniform lobe = allocLobe(arena, sizeof(FabricBsdfLobe));
    FabricBsdfLobe_init((varying FabricBsdfLobe * uniform) lobe,
                        N, T,
                        threadDir,
                        0.0,
                        roughness,
                        (varying Color) sWhite,
                        useDwaFabric);

    Bsdf_addLobe(bsdf, lobe);

    return bsdf;
}

#pragma ignore warning(all)
export varying Bsdf * uniform
AshikminhShirleyBsdfFactory(uniform Arena * uniform arena,
                            const uniform ReferenceFrame * uniform frame,
                            uniform float roughness)
{
    varying Bsdf * uniform bsdf = createBsdf(arena);

    const varying Vec3f N = getN(*frame);
    const varying Vec3f T = frame->mX;

    varying BsdfLobe * uniform lobe = allocLobe(arena, sizeof(AshikhminShirleyGlossyBsdfLobe));
    AshikhminShirleyGlossyBsdfLobe_init((varying AshikhminShirleyGlossyBsdfLobe * uniform) lobe,
                                        N, T, roughness, roughness);

    varying Fresnel * uniform fresnel = createSchlickFresnel(arena, Color_ctor(1.f));
    BsdfLobe_setFresnel(lobe, fresnel);

    Bsdf_addLobe(bsdf, lobe);

    return bsdf;
}

#pragma ignore warning(all)
export varying Bsdf * uniform
AshikminhShirleyFullBsdfFactory(uniform Arena * uniform arena,
                                const uniform ReferenceFrame * uniform frame,
                                uniform float roughness)
{
    varying Bsdf * uniform bsdf = createBsdf(arena);

    const varying Vec3f N = getN(*frame);
    const varying Vec3f T = frame->mX;

    varying BsdfLobe * uniform asDiffuse = allocLobe(arena, sizeof(AshikhminShirleyDiffuseBsdfLobe));
    AshikhminShirleyDiffuseBsdfLobe_init((varying AshikhminShirleyDiffuseBsdfLobe * uniform) asDiffuse,
                                         N);

    varying BsdfLobe * uniform asGlossy = allocLobe(arena, sizeof(AshikhminShirleyGlossyBsdfLobe));
    AshikhminShirleyGlossyBsdfLobe_init((varying AshikhminShirleyGlossyBsdfLobe * uniform) asGlossy,
                                        N, T, roughness, roughness);

    varying Color scale = Color_ctor(.5f);
    BsdfLobe_setScale(asDiffuse, scale);

    varying Fresnel * uniform fresnel = createSchlickFresnel(arena, scale);
    BsdfLobe_setFresnel(asGlossy, fresnel);

    Bsdf_addLobe(bsdf, asDiffuse);
    Bsdf_addLobe(bsdf, asGlossy);

    return bsdf;
}

#pragma ignore warning(all)
export varying Bsdf * uniform
WardCorrectedBsdfFactory(uniform Arena * uniform arena,
                         const uniform ReferenceFrame * uniform frame,
                         uniform float roughness)
{
    varying Bsdf * uniform bsdf = createBsdf(arena);

    const varying Vec3f N = getN(*frame);
    const varying Vec3f T = frame->mX;

    varying BsdfLobe * uniform lobe = allocLobe(arena, sizeof(WardCorrectedBsdfLobe));
    WardCorrectedBsdfLobe_init((varying WardCorrectedBsdfLobe * uniform) lobe,
                               N, T, roughness, roughness, true);

    varying Fresnel * uniform fresnel = createSchlickFresnel(arena, Color_ctor(1.f));
    BsdfLobe_setFresnel(lobe, fresnel);

    Bsdf_addLobe(bsdf, lobe);

    return bsdf;
}

#pragma ignore warning(all)
export varying Bsdf * uniform
WardDuerBsdfFactory(uniform Arena * uniform arena,
                    const uniform ReferenceFrame * uniform frame,
                    uniform float roughness)
{
    varying Bsdf * uniform bsdf = createBsdf(arena);

    const varying Vec3f N = getN(*frame);
    const varying Vec3f T = frame->mX;

    varying BsdfLobe * uniform lobe = allocLobe(arena, sizeof(WardDuerBsdfLobe));
    WardDuerBsdfLobe_init((varying WardDuerBsdfLobe * uniform) lobe,
                          N, T, roughness, roughness, false);

    varying Fresnel * uniform fresnel = createSchlickFresnel(arena, Color_ctor(1.f));
    BsdfLobe_setFresnel(lobe, fresnel);

    Bsdf_addLobe(bsdf, lobe);

    return bsdf;
}

#pragma ignore warning(all)
export varying Bsdf * uniform
StochasticFlakesBsdfFactory(uniform Arena * uniform arena,
                            const uniform ReferenceFrame * uniform frame,
                            const uniform Vec3f * uniform normals,
                            const uniform Color * uniform colors,
                            const uniform size_t flakeCount,
                            const uniform float roughness,
                            const uniform float inputFlakeRandomness)
{
    varying Bsdf * uniform bsdf = createBsdf(arena);

    const varying Vec3f N = getN(*frame);

    intptr_t flakeNormalsPtr;
    intptr_t flakeColorsPtr;

    foreach_active(lane) {
        flakeNormalsPtr = insert(flakeNormalsPtr, lane, (uniform intptr_t)Arena_allocArray(arena, flakeCount, sizeof(Vec3f)));
        flakeColorsPtr = insert(flakeColorsPtr, lane, (uniform intptr_t)Arena_allocArray(arena, flakeCount, sizeof(Color)));
    }

    varying Vec3f * varying flakeNormals = (varying Vec3f * varying)flakeNormalsPtr;
    varying Color * varying flakeColors = (varying Color * varying)flakeColorsPtr;
    for (int i = 0; i < flakeCount; ++i) {
        flakeColors[i] = colors[i];
        flakeNormals[i] = normals[i];
    }

    varying BsdfLobe * uniform lobe = allocLobe(arena, sizeof(StochasticFlakesBsdfLobe));
    StochasticFlakesBsdfLobe_init((varying StochasticFlakesBsdfLobe * uniform)lobe, N, flakeNormals,
                                  flakeColors, flakeCount, roughness, inputFlakeRandomness);

    Bsdf_addLobe(bsdf, lobe);

    return bsdf;
}

#pragma ignore warning(all)
export varying Bsdf * uniform
HairDiffuseBsdfFactory(uniform Arena * uniform arena,
                       const uniform ReferenceFrame * uniform frame)
{
    varying Bsdf * uniform bsdf = createBsdf(arena);

    const varying Vec3f T = frame->mX;
    varying BsdfLobe * uniform lobe = allocLobe(arena, sizeof(HairDiffuseBsdfLobe));
    Color reflColor = Color_ctor(1.0f, 1.0f, 1.0f);
    Color refrColor = Color_ctor(1.0f, 1.0f, 1.0f);
    HairDiffuseBsdfLobe_init((varying HairDiffuseBsdfLobe * uniform) lobe,
                             T,
                             reflColor,
                             refrColor);

    Bsdf_addLobe(bsdf, lobe);

    return bsdf;
}

#pragma ignore warning(all)
export varying Bsdf * uniform
HairRBsdfFactory(uniform Arena * uniform arena,
                 const uniform ReferenceFrame * uniform frame,
                 uniform float roughness,
                 uniform float offset)

{
    varying Bsdf * uniform bsdf = createBsdf(arena);

    const varying Vec3f hairDir = frame->mX;
    const Vec2f hairUV = Vec2f_ctor(0.25f, 0.1f);
    const uniform float eta = 1.55f;
    const uniform float shift = deg2rad(offset);

    varying BsdfLobe * uniform lobe = allocLobe(arena, sizeof(HairBsdfLobe));
    HairRBsdfLobe_init((varying HairBsdfLobe * uniform)lobe,
                       hairDir,
                       hairUV,
                       1.0f, //medium IOR
                       eta,
                       HAIR_FRESNEL_DIELECTRIC_CYLINDER,
                       0, //cuticle layer thickness
                       shift,
                       roughness,
                       (varying Color) sWhite);

    Bsdf_addLobe(bsdf, lobe);

    return bsdf;
}

#pragma ignore warning(all)
export varying Bsdf * uniform
HairTTBsdfFactory(uniform Arena * uniform arena,
                  const uniform ReferenceFrame * uniform frame,
                  uniform float roughness,
                  uniform float offsetR,
                  uniform float aziWidth)
{
    varying Bsdf * uniform bsdf = createBsdf(arena);

    const varying Vec3f hairDir = frame->mX;
    const Color hairColor = Color_ctor(1.0f, 1.0f, 1.0f);
    const uniform float aziRoughness = deg2rad(aziWidth);
    const Color hairSigmaA = HairUtil_computeAbsorptionCoefficients(hairColor,
                                                                    aziRoughness);
    const Vec2f hairUV = Vec2f_ctor(0.25f, 0.1f);
    const uniform float eta = 1.55f;

    const uniform float shift = deg2rad(-0.5 * offsetR);

    varying BsdfLobe * uniform lobe = allocLobe(arena, sizeof(HairBsdfLobe));
    HairTTBsdfLobe_init((varying HairBsdfLobe * uniform) lobe,
                        hairDir,
                        hairUV,
                        1.0f, //medium IOR
                        eta,
                        HAIR_FRESNEL_DIELECTRIC_CYLINDER,
                        0, //cuticle layer thickness
                        shift,
                        roughness,
                        aziRoughness,
                        hairColor,
                        hairSigmaA,
                        (varying Color) sWhite);

    Bsdf_addLobe(bsdf, lobe);

    return bsdf;
}

#pragma ignore warning(all)
export varying Bsdf * uniform
TwoLobeBsdfFactory(uniform Arena * uniform arena,
                   const uniform ReferenceFrame * uniform frame,
                   uniform float roughness)
{
    varying Bsdf * uniform bsdf = createBsdf(arena);

    const varying Vec3f N = getN(*frame);

    varying BsdfLobe * uniform lobe1 = allocLobe(arena, sizeof(LambertBsdfLobe));
    LambertBsdfLobe_init((varying LambertBsdfLobe * uniform) lobe1, N, (varying Color)sWhite, true);

    varying BsdfLobe * uniform lobe2 = allocLobe(arena, sizeof(CookTorranceBsdfLobe));
    CookTorranceBsdfLobe_init((varying CookTorranceBsdfLobe * uniform) lobe2, N, roughness);

    varying Color scale = Color_ctor(.3f);
    BsdfLobe_setScale(lobe1, scale);

    varying Fresnel * uniform fresnel = createSchlickFresnel(arena, Color_ctor(.7f));
    BsdfLobe_setFresnel(lobe2, fresnel);

    Bsdf_addLobe(bsdf, lobe1);
    Bsdf_addLobe(bsdf, lobe2);

    return bsdf;
}

#pragma ignore warning(all)
export varying Bsdf * uniform
ThreeLobeBsdfFactory(uniform Arena * uniform arena,
                     const uniform ReferenceFrame * uniform frame,
                     uniform float roughness1,
                     uniform float roughness2)
{
    // TODO: This bsdf is not energy preserving
    varying Bsdf * uniform bsdf = createBsdf(arena);

    const varying Vec3f N = getN(*frame);

    varying BsdfLobe * uniform lobe1 = allocLobe(arena, sizeof(LambertBsdfLobe));
    LambertBsdfLobe_init((varying LambertBsdfLobe * uniform) lobe1, N, (varying Color)sWhite, true);

    varying BsdfLobe * uniform lobe2 = allocLobe(arena, sizeof(CookTorranceBsdfLobe));
    CookTorranceBsdfLobe_init((varying CookTorranceBsdfLobe * uniform) lobe2, N, roughness2);

    varying BsdfLobe * uniform lobe3 = allocLobe(arena, sizeof(CookTorranceBsdfLobe));
    CookTorranceBsdfLobe_init((varying CookTorranceBsdfLobe * uniform) lobe3, N, roughness1);

    const varying Color scale1 = Color_ctor(.1f);
    BsdfLobe_setScale(lobe1, scale1);

    varying Fresnel * uniform f1 = createSchlickFresnel(arena, Color_ctor(.5f));
    BsdfLobe_setFresnel(lobe2, f1);

    varying Fresnel * uniform f2 = createSchlickFresnel(arena, Color_ctor(.3f));
    BsdfLobe_setFresnel(lobe3, f2);

    Bsdf_addLobe(bsdf, lobe1);
    Bsdf_addLobe(bsdf, lobe2);
    Bsdf_addLobe(bsdf, lobe3);

    return bsdf;
}

