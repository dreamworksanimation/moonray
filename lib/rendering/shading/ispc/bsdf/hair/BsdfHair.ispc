// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

/// @file HairBsdf.ispc

#include "BsdfHair.isph"
#include <moonray/rendering/shading/ispc/bsdf/BsdfUtil.isph>
#include <moonray/rendering/shading/ispc/ShaderMacros.isph>
#include <moonray/rendering/shading/ispc/Util.isph>
#include <scene_rdl2/common/platform/IspcUtil.isph>

// Utility Func to evalHairFresnel Types
Color
HairBsdfLobe_evalHairFresnel(const varying HairBsdfLobe * uniform lobe,
                             const varying HairState& hairState,
                             const varying float cosTheta)
{

    Color fresnel = sWhite;
    switch(lobe->mFresnelType) {
    case HAIR_FRESNEL_SIMPLE_LONGITUDINAL:
    default:
        // Simple longitudinal cosine term
        fresnel = Fresnel_eval(lobe->mFresnel,
                               cosTheta);
        break;
    case HAIR_FRESNEL_DIELECTRIC_CYLINDER:
        // From "An Energy-Conserving Hair Reflectance Model"
        // This curve is very similar to the Marschner Fresnel equations
        // and slightly cheaper to evaluate without etaP & etaPP in Marschner:
        // https://www.desmos.com/calculator/fmxsatvxi3
        fresnel = Fresnel_eval(lobe->mFresnel,
                               cosTheta*hairState.mCosGammaO);
        break;
    case HAIR_FRESNEL_LAYERED_CUTICLES:
        // From "Physically-Accurate Fur Reflectance: Modeling, Measurement and Rendering"
        // https://www.desmos.com/calculator/fmxsatvxi3
        fresnel = LayeredDielectricFresnel_evalFresnel(lobe->mMediumIOR,
                                                       lobe->mIOR,
                                                       hairState.mEtaP,
                                                       hairState.mEtaPP,
                                                       hairState.mCosGammaO,
                                                       lobe->mCuticleLayerThickness);
        break;
    }
    return fresnel;
}

varying Color
HairBsdfLobe_evalBsdf(const varying HairBsdfLobe * uniform lobe,
                      const varying HairState& hairState,
                      const varying HairBsdfLobeGlintAttrs * uniform glintAttrs,
                      const varying bool include)
{
    return HairBsdfLobe_evalFresnel(lobe, hairState) *
            HairBsdfLobe_evalMTerm(lobe, hairState) *
            HairBsdfLobe_evalNTermWithAbsorption(lobe, hairState, glintAttrs) *
            HairBsdfLobe_evalCosineTerm(lobe, hairState) *
            lobe->mTint;
}


varying Color
HairBsdfLobe_eval(const varying BsdfLobe * uniform lobe,
                  const varying BsdfSlice &slice,
                  const varying Vec3f &wi,
                  varying float * uniform pdf)
{
    const varying HairBsdfLobe * uniform hairLobe =
            (const varying HairBsdfLobe * uniform) lobe;

    if (!HairBsdfLobe_clampTest(hairLobe->mHairDir,
                                wi)) {
        if (pdf) *pdf = 0.0f;
        return Color_ctor(0.0f, 0.0f, 0.0f);
    }

    // calculate all the relevant hair bsdf parameters
    HairState hairState;
    HairState_init(&hairState,
                   slice.mWo,
                   wi,
                   hairLobe->mHairDir,
                   hairLobe->mH,
                   hairLobe->mIOR,
                   hairLobe->mSigmaA,
                   hairLobe->mHairRotation * hairLobe->mHairUV.x, // use hair s coord to vary rotation from base to tip
                   hairLobe->mHairNormal);

    HairBsdfLobeGlintAttrs glintAttrs;
    glintAttrs.mShowGlint = hairLobe->mShowGlint;
    glintAttrs.mGlintRoughness = hairLobe->mGlintRoughness;
    glintAttrs.mGlintEccentricity = hairLobe->mGlintEccentricity;
    glintAttrs.mGlintSaturation = hairLobe->mGlintSaturation;

    Color bsdf = HairBsdfLobe_evalBsdf(hairLobe,
                                       hairState,
                                       &glintAttrs,
                                       slice.mIncludeCosineTerm);

    if (pdf) *pdf = HairBsdfLobe_evalPdf(hairLobe,
                                         hairState);

    return bsdf;
}

varying Color
HairBsdfLobe_sample(const varying BsdfLobe * uniform lobe,
                    const varying BsdfSlice &slice,
                    varying float r1,
                    varying float r2,
                    varying Vec3f &wi,
                    varying float &pdf)
{
    const varying HairBsdfLobe * uniform hairLobe =
            (const varying HairBsdfLobe * uniform) lobe;

    // calculate all the relevant hair bsdf parameters
    HairState hairState;
    HairState_init(&hairState,
                   slice.mWo,
                   hairLobe->mHairDir,
                   hairLobe->mH,
                   hairLobe->mIOR,
                   hairLobe->mSigmaA,
                   hairLobe->mHairRotation * hairLobe->mHairUV.x, // use hair s coord to vary rotation from base to tip
                   hairLobe->mHairNormal);

    HairBsdfLobeGlintAttrs glintAttrs;
    glintAttrs.mShowGlint = hairLobe->mShowGlint;
    glintAttrs.mGlintRoughness = hairLobe->mGlintRoughness;
    glintAttrs.mGlintEccentricity = hairLobe->mGlintEccentricity;
    glintAttrs.mGlintSaturation = hairLobe->mGlintSaturation;

#ifdef PBR_HAIR_USE_UNIFORM_SAMPLING
    const varying Vec3f dir = sampleSphereUniform(r1, r2);
    wi = HairState_localToGlobal(hairState, dir);
#else
    float phiI = 0.0f;
    float phiPdf = HairBsdfLobe_samplePhi(hairLobe,
                                          r2,
                                          hairState.mPhiO,
                                          phiI); // phiI is in (-pi, pi]
    MNRY_ASSERT(phiI >= -sPi  &&  phiI <= sPi);

    float sinPhi, cosPhi;
    sincos(phiI, &sinPhi, &cosPhi);

    float thetaI;
    const float thetaPdf = HairBsdfLobe_sampleTheta(hairLobe,
                                                    r1,
                                                    hairState.mThetaO,
                                                    thetaI);

    float sinTheta, cosTheta;
    sincos(thetaI, &sinTheta, &cosTheta);

    if (!HairBsdfLobe_clampTest(cosTheta)) {
        pdf = 0.0f;
        return Color_ctor(0.0f, 0.0f, 0.0f);
    }

    pdf = max(0.0f, phiPdf * thetaPdf);

    // Compute the light direction vector for shading.
    // TODO: use ReferenceFrame local2global
    float uWgt = sinTheta;
    float vWgt = cosTheta * cosPhi;
    float wWgt = cosTheta * sinPhi;
    wi = normalize(HairState_localToGlobal(&hairState,
                                           uWgt, vWgt, wWgt));

#endif

    HairState_updateAngles(&hairState,
                           wi,
                           phiI,
                           thetaI);

    return HairBsdfLobe_evalBsdf(hairLobe,
                                 hairState,
                                 &glintAttrs,
                                 slice.mIncludeCosineTerm);
}

varying float
HairBaseBsdfLobe_evalMTerm(const varying HairState& hairState,
                           const varying float sinAlpha,
                           const varying float cosAlpha,
                           const varying float variance)
{
    float sinThetaICone = hairState.mSinThetaI * cosAlpha + hairState.mCosThetaI * sinAlpha;
    float cosThetaICone = hairState.mCosThetaI * cosAlpha - hairState.mSinThetaI * sinAlpha;

    // Handle out-of-range thetaI values because of the shift
    cosThetaICone = abs(cosThetaICone);

    // Perfect Longitudinal Function From [1]
    return HairUtil_deonLongitudinalM(variance,
                                      sinThetaICone,
                                      cosThetaICone,
                                      hairState.mSinThetaO,
                                      hairState.mCosThetaO);
}

varying float
HairBaseBsdfLobe_evalNTerm(const varying HairBsdfLobe * uniform lobe,
                           const varying HairState& hairState,
                           const varying float v,
                           const varying HairBsdfLobeGlintAttrs * uniform glintAttrs)
{
    // Normalized Azimuthal Lobe - 1/4 * cos(phiD/2)
    const float cosPhiDOver2 = max(0.0f, 0.25f * hairState.mCosPhiDOverTwo);
    return cosPhiDOver2;
}

varying Color
HairBaseBsdfLobe_evalAbsorption(const varying HairState& hairState,
                                const varying Color& scale,
                                const varying float saturation)
{
    return sWhite;
}

varying Color
HairBaseBsdfLobe_evalNTermWithAbsorption(const varying HairBsdfLobe * uniform lobe,
                                         const varying HairState& hairState,
                                         const varying Color& scale,
                                         const varying HairBsdfLobeGlintAttrs * uniform glintAttrs,
                                         const varying float v,
                                         const varying float saturation)
{
    return HairBaseBsdfLobe_evalNTerm(lobe, hairState, v, glintAttrs) * HairBaseBsdfLobe_evalAbsorption(hairState, scale, saturation);
}

varying float
HairBaseBsdfLobe_evalPhiPdf(const varying HairBsdfLobe * uniform lobe,
                            const varying HairState& hairState,
                            const varying float variance)
{
    return HairBaseBsdfLobe_evalNTerm(lobe,
                                      hairState,
                                      variance,
                                      NULL);
}

varying float
HairBaseBsdfLobe_evalThetaPdf(const varying HairState& hairState,
                              const varying float sinAlpha,
                              const varying float cosAlpha,
                              const varying float variance)
{
    // Perfect Sampling = PDF is the same as M Term
    const float thetaPDF = HairBaseBsdfLobe_evalMTerm(hairState,
                                                      sinAlpha,
                                                      cosAlpha,
                                                      variance);
    return thetaPDF;
}

varying float
HairBsdfLobe_computeCosineTerm(varying HairState * uniform hair,
                               const varying HairState& hairState,
                               const varying bool includeCosineTerm)
{
    // The Cosine terms cancel out since the Longitudinal Function includes
    // a oneOverCosThetaI
    if (includeCosineTerm || isZero(hairState.mCosThetaI)) {
        return 1.0f;
    } else {
        return 1.0f / hairState.mCosThetaI;
    }
}

varying float
HairBaseBsdfLobe_sampleTheta(const varying float sinAlpha,
                             const varying float cosAlpha,
                             const varying float variance,
                             const varying float r,
                             const varying float  thetaO,
                             varying float& thetaI)
{
    // Create Two Unique Random Numbers using a Morton Curve
    // https://fgiesen.wordpress.com/2009/12/13/decoding-morton-codes/
    // PBRT Section 4.3.3
    varying float * uniform eps = asArray(HairUtil_demuxFloat(r));

    float sinThetaO, cosThetaO;
    sincos(thetaO,
           &sinThetaO,
           &cosThetaO);

    eps[0] = max(eps[0], 1e-5f);

    // Eugene's Derivation - [3] Section 3.2
    // float cosTheta = mLongitudinalVariance *
    //     log( exp(rcp(mLongitudinalVariance)) - 2 * eps[0] * HairUtil_sinh(rcp(mLongitudinalVariance)) );

    // PBRT Derivation (More Stable at low variance values)
    float cosTheta = 1.0f + variance *
        log(eps[0] + (1.0f - eps[0]) * exp(-2.0f / variance));

    float sinTheta = HairUtil_safeSqrt(1.0f - HairUtil_sqr(cosTheta));
    float cosPhi = cos(sTwoPi * eps[1]);
    float sinThetaI = -cosTheta * sinThetaO + sinTheta * cosPhi * cosThetaO;
    float cosThetaI = HairUtil_safeSqrt(1.0f - HairUtil_sqr(sinThetaI));

    // Update sampled $\sin \thetai$ and $\cos \thetai$ to account for scales
    // Note - Shift by -negative Alpha, because in eval() you shift by positive
    float sinThetaICone = sinThetaI * cosAlpha - cosThetaI * sinAlpha;
    float cosThetaICone = cosThetaI * cosAlpha + sinThetaI * sinAlpha;
    // Handle out-of-range thetaI values because of the shift
    cosThetaICone = abs(cosThetaICone);

    // This is our sample direction. We clamp sinThetaI to [-1,1] to remove
    // small floating point math errors, because asin() is extremely sensitive
    // and will throw NaNs if the sine value is even slightly outside this range.
    sinThetaI = clamp(sinThetaICone, -1.0f, 1.0f);
    cosThetaI = cosThetaICone;

    thetaI = asin(sinThetaI);

    // Shift by +alpha to calculate the PDF, same as will happen in eval()
    sinThetaICone = sinThetaI * cosAlpha + cosThetaI * sinAlpha;
    cosThetaICone = cosThetaI * cosAlpha - sinThetaI * sinAlpha;

    const float thetaPDF = HairUtil_deonLongitudinalM(variance,
                                                      sinThetaICone, cosThetaICone,
                                                      sinThetaO, cosThetaO);

    return thetaPDF;
}

varying float
HairBaseBsdfLobe_samplePhi(const varying float variance,
                           const float r2,
                           const varying float phiO,
                           varying float &phiI)
{
    float sinPhi_2 = 2.0f * r2 - 1.0f;
    sinPhi_2 = clamp(sinPhi_2, -1.0f, 1.0f);
    const float phi = 2.0f * asin(sinPhi_2);
    phiI = rangeAngle(phiO + phi);

    float phiPdf = cos(phi * 0.5f) * 0.25f;
    return phiPdf;
}

void
HairBsdfLobe_init(varying HairBsdfLobe * uniform lobe,
                  uniform BsdfLobeName name,
                  uniform BsdfLobeType type,
                  const varying Vec3f& hairDir,
                  const varying Vec2f& hairUV,
                  const varying float mediumIOR,
                  const varying float ior,
                  const varying HairFresnelType fresnelType,
                  const varying float cuticleLayerThickness,
                  const varying float longShift,
                  const varying float longRoughness,
                  const varying float azimRoughness,
                  const varying Color& hairColor,
                  const varying Color& hairSigmaA,
                  const varying Color& hairTint,
                  const varying float hairRotation,
                  const varying Vec3f& hairNormal,
                  const varying float hairSaturation,
                  HairBsdfLobeEvalFresnelFn evalFresnelFn,
                  HairBsdfLobeEvalAbsorptionFn evalAbsorptionFn,
                  HairBsdfLobeEvalMTermFn evalMTermFn,
                  HairBsdfLobeEvalNTermWithAbsorptionFn evalNTermWithAbsorptionFn,
                  HairBsdfLobeEvalPhiPdfFn evalPhiPdfFn,
                  HairBsdfLobeEvalThetaPdfFn evalThetaPdfFn,
                  HairBsdfLobeSamplePhiFn samplePhiFn,
                  HairBsdfLobeSampleThetaFn sampleThetaFn,
                  uniform BsdfLobeGetPropertyFn getPropertyFn,
                  uniform BsdfLobeShowFn showFn)
{
    MNRY_ASSERT(longRoughness > 0.0f && longRoughness < 1.0f);
    BsdfLobe_init((varying BsdfLobe * uniform) lobe,
                  /* name = */ name,
                  /* type = */ type,
                  /* differentialFlags = */ BSDF_LOBE_IGNORES_INCOMING_DIFFERENTIALS,
                  /* isSpherical = */ true,
                  /* propertyFlags = */ BSDF_LOBE_PROPERTY_ROUGHNESS |
                      BSDF_LOBE_PROPERTY_COLOR | BSDF_LOBE_PROPERTY_NORMAL |
                      BSDF_LOBE_PROPERTY_PBR_VALIDITY,
                  /* evalFn = */ HairBsdfLobe_eval,
                  /* sampleFn = */ HairBsdfLobe_sample,
                  /* albedoFn = */ HairBsdfLobe_albedo,
                  /* differentialsFn = */ HairBsdfLobe_differentials,
                  /* getPropertyFn = */ getPropertyFn,
                  /* showFn = */ showFn);

    BsdfLobe_setIsHair((varying BsdfLobe * uniform) lobe, true);

    lobe->mHairDir  = hairDir;
    lobe->mHairUV   = hairUV;
    // For Moonray, Texture 't' varies across the width of hair curve
    // and goes from [0,1].   We remap this to [-1,1].
    // Note - this is not true of all curve types
    // Please take a look at MOONRAY-3116 and make sure that is fixed.
    lobe->mH = hairUV.y * 2.0f - 1.0f;

    lobe->mHairRotation = hairRotation;
    lobe->mHairNormal = hairNormal;

    lobe->mMediumIOR = mediumIOR;
    lobe->mIOR = ior;

    lobe->mHairColor    = hairColor;
    lobe->mSigmaA       = hairSigmaA;
    lobe->mTint         = hairTint;
    lobe->mSaturation   = hairSaturation;

    lobe->mLongitudinalShift = longShift;
    sincos(longShift, &lobe->mSinAlpha, &lobe->mCosAlpha);

    lobe->mLongitudinalRoughness    = longRoughness;
    lobe->mAzimuthalRoughness       = azimRoughness;

    // calculate variance
    lobe->mLongitudinalVariance     = HairUtil_longitudinalVar(longRoughness);;
    lobe->mAzimuthalVariance        = HairUtil_azimuthalVar(azimRoughness);;

    // Derive a directional differential scale that varies according to the
    // smallest of longitudinal and azimuthal width
    const float minWidth = min(roughness2Width(longRoughness),
                               roughness2Width(azimRoughness));

    // Derive a directional differential scale that varies according to width
    lobe->mdDFactor = HairBsdfLobe_sdDFactorMin + minWidth * HairBsdfLobe_sdDFactorSlope;

    lobe->mFresnelType   = fresnelType;
    // Cuticle-Layer has been parameterized to be between 0 and 1
    // Layers vary in between [0.5, 1.5]
    // See Table 2
    /// From "Physically-Accurate Fur Reflectance: Modeling, Measurement and Rendering"
    lobe->mCuticleLayerThickness = lerp(0.5f, 1.5f,
                                        clamp(cuticleLayerThickness, 0.0f, 1.0f));

    // all of these funcs are required
    MNRY_ASSERT(evalFresnelFn != NULL);
    MNRY_ASSERT(evalAbsorptionFn != NULL);
    MNRY_ASSERT(evalMTermFn != NULL);
    MNRY_ASSERT(evalNTermWithAbsorptionFn != NULL);
    MNRY_ASSERT(evalPhiPdfFn != NULL);
    MNRY_ASSERT(evalThetaPdfFn != NULL);
    MNRY_ASSERT(samplePhiFn != NULL);
    MNRY_ASSERT(sampleThetaFn != NULL);
    MNRY_ASSERT(getPropertyFn != NULL);

    lobe->mEvalFresnelFn = (intptr_t) evalFresnelFn;
    lobe->mEvalAbsorptionFn = (intptr_t) evalAbsorptionFn;
    lobe->mEvalMTermFn = (intptr_t) evalMTermFn;
    lobe->mEvalNTermWithAbsorptionFn = (intptr_t) evalNTermWithAbsorptionFn;
    lobe->mEvalPhiPdfFn = (intptr_t) evalPhiPdfFn;
    lobe->mEvalThetaPdfFn = (intptr_t) evalThetaPdfFn;
    lobe->mSamplePhiFn = (intptr_t) samplePhiFn;
    lobe->mSampleThetaFn = (intptr_t) sampleThetaFn;
}

#pragma ignore warning(all)
ISPC_UTIL_EXPORT_STRUCT_TO_HEADER(HairBsdfLobe);

