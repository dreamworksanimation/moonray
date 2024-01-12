// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

/// @file BsdfHairEnergyConserving.ispc

#include "BsdfHairLobes.isph"

#include <moonray/rendering/shading/ispc/ColorCorrect.isph>
#include <scene_rdl2/common/platform/IspcUtil.isph>

varying Color
HairRBsdfLobe_evalFresnelTerm(const varying HairBsdfLobe * uniform lobe,
                              const varying HairState& hairState)
{
    if (!lobe->mFresnel) return sWhite;

    // [1] Section 6
    const Color fresnel = HairBsdfLobe_evalHairFresnel(lobe,
                                                       hairState,
                                                       hairState.mCosThetaD);
    return fresnel;
}

varying Color
HairRBsdfLobe_evalAbsorptionTerm(const varying HairState& hairState,
                                 const varying Color& scale,
                                 const varying float saturation)
{
    return scale;
}

varying Color
HairRBsdfLobe_evalNTermWithAbsorption(const varying HairBsdfLobe * uniform lobe,
                                      const varying HairState& hairState,
                                      const varying Color& scale,
                                      const varying HairBsdfLobeGlintAttrs * uniform glintAttrs,
                                      const varying float variance,
                                      const varying float saturation)
{
    return HairBaseBsdfLobe_evalNTerm(lobe, hairState, variance) *
            HairRBsdfLobe_evalAbsorptionTerm(hairState, scale, saturation);
}

varying Color
HairTRTBsdfLobe_evalFresnelTerm(const varying HairBsdfLobe * uniform lobe,
                                const varying HairState& hairState)
{
    Color fresnel = sWhite;
    if (lobe->mFresnel != nullptr) {
        const Color F = HairBsdfLobe_evalHairFresnel(lobe,
                                                     hairState,
                                                     hairState.mCosThetaD);
        Color oneMinusF;
        if (lobe->mFresnelType == HAIR_FRESNEL_LAYERED_CUTICLES) {
            // Note for LayerFresnel, we call a static function to evaluate
            // so we substract here instead
            oneMinusF = sWhite - F;
        } else {
            // We assume that this lobe has been set up using oneMinusFresnel
            oneMinusF = F;
        }
        fresnel = oneMinusF * oneMinusF * (sWhite - oneMinusF);
    }
    return fresnel;
}

varying Color
HairTRTBsdfLobe_evalAbsorptionTerm(const varying HairState& hairState,
                                   const varying Color& scale,
                                   const varying float saturation)
{
    const Color T = hairState.mAbsorptionTerm;
    return (scale * T * T);
}

varying Color
HairTRTBsdfLobe_evalGlintAbsorptionTerm(const varying HairBsdfLobe * uniform lobe,
                                        const varying HairState& hairState,
                                        const varying HairBsdfLobeGlintAttrs * uniform glintAttrs,
                                        const varying float etaStar,
                                        const varying float etaP)
{
    if (!glintAttrs || !glintAttrs->mShowGlint) {
        return sBlack;
    } else {
        Color glintAbsorption = HairState_calculateAbsorptionTerm(hairState.mSinGammaO,
                                                                  etaStar,
                                                                  hairState.mSigmaA,
                                                                  etaP,
                                                                  hairState.mCosThetaO,
                                                                  hairState.mSinThetaO);
        applySaturationWithoutPreservingLuminance(glintAttrs->mGlintSaturation,
                                                  glintAbsorption.r,
                                                  glintAbsorption.g,
                                                  glintAbsorption.b);
        return glintAbsorption * glintAbsorption;
    }
}

varying float
HairTRTBsdfLobe_evalNTermTRT(const varying HairBsdfLobe * uniform lobe,
                             const varying HairState& hairState,
                             const varying float variance)
{
    // Normalized Azimuthal Lobe - 1/4 * cos(phiD/2)
    const float cosPhiDOver2 = max(0.0f, 0.25f * hairState.mCosPhiDOverTwo);
    return cosPhiDOver2;
}

varying float
HairTRTBsdfLobe_evalNTermGlint(const varying HairBsdfLobe * uniform lobe,
                               const varying HairState& hairState,
                               const varying float variance,
                               const varying HairBsdfLobeGlintAttrs * uniform glintAttrs,
                               varying float& etaStar,
                               varying float& etaP,
                               varying float& trtWeight)
{
    if (!glintAttrs || !glintAttrs->mShowGlint) {
        etaStar = 0.f;
        etaP = 0.f;
        trtWeight = 1.f;
        return 0.f;
    } else {
        // Glint calculation taken from "Light Scattering from Human Hair Fibers" Marschner, et al.
        // Eccentricity is approximated by adjusting the index of refraction (Marschner 5.2.3)
        const float eta = hairState.mEta;
        const float etaStar1 = 2.f * (eta - 1.f) * glintAttrs->mGlintEccentricity * glintAttrs->mGlintEccentricity - eta + 2.f;
        const float etaStar2 = 2.f * (eta - 1.f) * rcp(glintAttrs->mGlintEccentricity * glintAttrs->mGlintEccentricity) - eta + 2.f;
        etaStar = 0.5f * ((etaStar1 + etaStar2) + cos(2.f * hairState.mPhiH) * (etaStar1 - etaStar2));
        etaP = HairUtil_safeSqrt(etaStar*etaStar - hairState.mSinThetaD*hairState.mSinThetaD) / hairState.mCosThetaD;

        // hc is the offset at which the glint caustic occurs (Marschner 4.2 equation 4)
        const float hc = HairUtil_safeSqrt((4.f - etaP*etaP)/3.f);
        const float gammai = asin(hc);
        const float gammat = asin(hc/etaP);

        // phiC is the angle offset from phiD at which the glint occurs
        // phiC is undefined for incidence angles past the caustic merge, when etaP equals 2
        // Unlike Marschner, we do not bother fading out the caustic glint
        // (Marschner 5.2.2)
        float phiC;
        if (etaP < 2) {
            phiC = rangeAngle(2.f * 2.f * gammat - 2.f * gammai + 2.f * sPi);
        } else {
            phiC = 0;
        }

        // Unlike Marschner, we use trimmed logisitic functions instead of gaussians
        // to model the glint distributions

        // g0 is the normalization term used to calculate the weights below
        const float g0 = HairUtil_trimmedLogisticFunction(0.f,
                                                          glintAttrs->mGlintRoughness,
                                                          -sPi,
                                                           sPi);
        float g1 = HairUtil_trimmedLogisticFunction(rangeAngle(hairState.mPhiD - phiC),
                                                    glintAttrs->mGlintRoughness,
                                                    -sPi,
                                                     sPi);
        float g2 = HairUtil_trimmedLogisticFunction(rangeAngle(hairState.mPhiD + phiC),
                                                    glintAttrs->mGlintRoughness,
                                                    -sPi,
                                                     sPi);

        // The following graph shows the glint distributions as well as the trt distribution.
        // As well as the weights for each distribution and how all three pdfs sum to 1.
        // https://www.desmos.com/calculator/hyx1kavyk9

        // w1 and w2 are glint weights.
        const float w1 = g1/g0;
        const float w2 = g2/g0;
        trtWeight = 1.f - w1 - w2;

        return w1 * g1 + w2 * g2;
    }
}

varying Color
HairTRTBsdfLobe_evalNTermWithAbsorption(const varying HairBsdfLobe * uniform lobe,
                                        const varying HairState& hairState,
                                        const varying Color& scale,
                                        const varying HairBsdfLobeGlintAttrs * uniform glintAttrs,
                                        const varying float variance,
                                        const varying float saturation)
{
    // evalNTermGlint and evalNTermTRT were separated so that their respective
    // absorptions could be applied to each individually.
    float etaStar, etaP, trtWeight;
    // evalNTermGlint evaluates the glint component of the TRT lobe
    // and returns a weight for the remaining TRT response which
    // is dependent on the glint response
    const float glintN = HairTRTBsdfLobe_evalNTermGlint(lobe, hairState, variance, glintAttrs, etaStar, etaP, trtWeight);
    return trtWeight * HairTRTBsdfLobe_evalNTermTRT(lobe, hairState, variance)
                     * HairTRTBsdfLobe_evalAbsorptionTerm(hairState, scale, saturation)
            + glintN * HairTRTBsdfLobe_evalGlintAbsorptionTerm(lobe, hairState, glintAttrs, etaStar, etaP);
}

varying Color
HairTTBsdfLobe_evalFresnelTerm(const varying HairBsdfLobe * uniform lobe,
                               const varying HairState& hairState)
{
    Color fresnel = sWhite;
    if (lobe->mFresnel != nullptr) {
        const Color F = HairBsdfLobe_evalHairFresnel(lobe,
                                                     hairState,
                                                     hairState.mCosThetaD);
        Color oneMinusF;
        if (lobe->mFresnelType == HAIR_FRESNEL_LAYERED_CUTICLES) {
            // Note for LayerFresnel, we call a static function to evaluate
            // so we substract here instead
            oneMinusF = sWhite - F;
        } else {
            // We assume that this lobe has been set up using oneMinusFresnel
            oneMinusF = F;
        }
        fresnel = fresnel * oneMinusF * oneMinusF;
    }
    return fresnel;
}

// MOONSHINE-1238
// Shader Trick to further saturate/desaturate Transmission
void
applySaturation(const varying float saturation,
                varying Color& c,
                const varying HairState& hairState)
{
    if (all(isOne(saturation))) {
        return;
    }

    // phiD goes from [-pi, pi] so normalizing it to [0, 1]
    const float x = abs(hairState.mPhiD*sOneOverPi);
    // isolate a narrow band of direct transmission
    // step [0.8f, 1.0f]
    // This range *could be exposed to the artists but we are  trying
    // to limit confusing user-controls here.
    float t = clamp((x - 0.8f) / (0.2f), 0.0f, 1.0f);

    // smoothstep
    t = t*t*(3.0f - 2*t);

    const float s = lerp(1.0f, saturation, t);
    applySaturation(s, s, s, c.r, c.g, c.b);
    c = clampColor(c);
}

varying Color
HairTTBsdfLobe_evalAbsorptionTerm(const varying HairState& hairState,
                                  const varying Color& scale,
                                  const varying float saturation)
{
    Color c  = hairState.mAbsorptionTerm;
    // MOONSHINE-1238
    // Shader Trick to further saturate/desaturate Transmission
    applySaturation(saturation, c,
                    hairState);

    return (c*scale);
}

varying float
HairTTBsdfLobe_evalNTerm(const varying HairBsdfLobe * uniform lobe,
                         const varying HairState& hairState,
                         const varying float variance,
                         const varying HairBsdfLobeGlintAttrs * uniform glintAttrs)
{
    const float Np = HairUtil_trimmedLogisticFunction(rangeAngle(hairState.mPhiD - sPi),
                                                       variance,
                                                      -sPi,
                                                       sPi);

    return Np;
}

varying Color
HairTTBsdfLobe_evalNTermWithAbsorption(const varying HairBsdfLobe * uniform lobe,
                                       const varying HairState& hairState,
                                       const varying Color& scale,
                                       const varying HairBsdfLobeGlintAttrs * uniform glintAttrs,
                                       const varying float variance,
                                       const varying float saturation)
{
    return HairTTBsdfLobe_evalNTerm(lobe, hairState, variance, glintAttrs) *
            HairTTBsdfLobe_evalAbsorptionTerm(hairState, scale, saturation);
}

varying float
HairTTBsdfLobe_evalPhiPdf(const varying HairBsdfLobe * uniform lobe,
                          const varying HairState& hairState,
                          const varying float variance)
{
    return HairTTBsdfLobe_evalNTerm(lobe,
                                    hairState,
                                    variance,
                                    NULL);
}

varying float
HairTTBsdfLobe_samplePhi(const varying float variance,
                         const varying float r2,
                         const varying float phiO,
                         varying float& phiI)
{
    // Importance sample phi
    const float phiD = HairUtil_sampleTrimmedLogistic(r2,
                                                       variance,
                                                      -sPi,
                                                       sPi);

    phiI = rangeAngle(phiO + phiD + sPi);
    const float phiPdf = HairUtil_trimmedLogisticFunction(phiD, variance, -sPi, sPi);
    return phiPdf;
}

inline void
HairTRTBsdfLobe_show(const varying BsdfLobe * const uniform lobe,
                     const uniform uint8_t ntab)
{
    const varying HairBsdfLobe * uniform me =
        (const varying HairBsdfLobe * uniform) lobe;

    BsdfUtil_indent(ntab);   print("[HairTRTLobe] mask: %\n", lobe->mMask);
    HairBsdfLobe_show(me, ntab);
}

inline void
HairTTBsdfLobe_show(const varying BsdfLobe * const uniform lobe,
                    const uniform uint8_t ntab)
{
    const varying HairBsdfLobe * uniform me =
        (const varying HairBsdfLobe * uniform) lobe;

    BsdfUtil_indent(ntab);   print("[HairTTLobe] mask: %\n", lobe->mMask);
    HairBsdfLobe_show(me, ntab);
}

inline void
HairTRRTBsdfLobe_show(const varying BsdfLobe * const uniform lobe,
                      const uniform uint8_t ntab)
{
    const varying HairBsdfLobe * uniform me =
        (const varying HairBsdfLobe * uniform) lobe;

    BsdfUtil_indent(ntab);   print("[HairTRRTLobe] mask: %\n", lobe->mMask);
    HairBsdfLobe_show(me, ntab);
}

inline uniform bool
HairTTBsdfLobe_getProperty(const varying BsdfLobe * uniform hairLobe,
                           uniform BsdfLobeProperty property,
                           varying float * uniform dest)
{
    const varying HairBsdfLobe * uniform lobe =
            (const varying HairBsdfLobe * uniform) hairLobe;

    uniform bool handled = true;

    switch (property)
    {
    case BSDF_LOBE_PROPERTY_COLOR:
        *dest       = lobe->mHairColor.r;
        *(dest + 1) = lobe->mHairColor.g;
        *(dest + 2) = lobe->mHairColor.b;
        break;
    case BSDF_LOBE_PROPERTY_ROUGHNESS:
        *dest       = lobe->mLongitudinalRoughness;
        *(dest + 1) = lobe->mAzimuthalRoughness;
        break;
    case BSDF_LOBE_PROPERTY_NORMAL:
        *dest       = lobe->mHairDir.x;
        *(dest + 1) = lobe->mHairDir.y;
        *(dest + 2) = lobe->mHairDir.z;
        break;
    case BSDF_LOBE_PROPERTY_PBR_VALIDITY:
        {
            // Forward call to fresnel
            varying Color res = sBlack;
            if (isActive(lobe->mFresnel)) {
                res = Fresnel_computePbrValidity(lobe->mFresnel);
            }
            *dest       = res.r;
            *(dest + 1) = res.g;
            *(dest + 2) = res.b;
        }
        break;
    default:
        handled = false;
        break;
    }
    return handled;
}

varying Color
HairTRRTBsdfLobe_evalFresnelTerm(const varying HairBsdfLobe * uniform lobe,
                                 const varying HairState& hairState)
{
    // We assume that this lobe has been set up using oneMinusFresnel
    // It is conceivable that the developer wrote the shader in such a way that
    // the above MNRY_ASSERT passes during testing, but the Fresnel could still be
    // inactive on one or more lanes for certain scenes.  In this case, not much
    // we can do here but exit early with black.
    if (!isActive(lobe->mFresnel)) {
        return sBlack;
    }

    // We assume that this lobe has been set up using oneMinusFresnel
    // We use thetaO to approximate the energy passed on by the lobe above
    // [1] Section 6
    const Color oneMinusF = HairBsdfLobe_evalHairFresnel(lobe,
                                                         hairState,
                                                         hairState.mCosThetaD);
    const Color f = sWhite - oneMinusF;
    return HairTRRTBsdfLobe_evalCompensationTerm(f,
                                                 oneMinusF,
                                                 hairState.mAbsorptionTerm);

}

varying Color
HairTRRTBsdfLobe_evalCompensationTerm(const varying Color& f,
                                      const varying Color& oneMinusF,
                                      const varying Color& T)
{
    const Color result = oneMinusF*oneMinusF*f*f;
    const Color oneMinusFT = sWhite - f * T;
    return (result*T*T*T*rcp(oneMinusFT));
}

varying Color
HairTRRTBsdfLobe_evalAbsorptionTerm(const varying HairState& hairState,
                                    const varying Color& scale,
                                    const varying float saturation)
{
    return scale;
}

varying float
HairTRRTBsdfLobe_evalNTerm(const varying HairBsdfLobe * uniform lobe,
                           const varying HairState& hairState,
                           const varying  float variance,
                           const varying HairBsdfLobeGlintAttrs * uniform glintAttrs)
{
    return sOneOverTwoPi;
}

varying Color
HairTRRTBsdfLobe_evalNTermWithAbsorption(const varying HairBsdfLobe * uniform lobe,
                                         const varying HairState& hairState,
                                         const varying Color& scale,
                                         const varying HairBsdfLobeGlintAttrs * uniform glintAttrs,
                                         const varying float variance,
                                         const varying float saturation)
{
    return HairTRRTBsdfLobe_evalNTerm(lobe, hairState, variance, glintAttrs) *
            HairTRRTBsdfLobe_evalAbsorptionTerm(hairState, scale, saturation);
}

varying float
HairTRRTBsdfLobe_evalPhiPdf(const varying HairBsdfLobe * uniform lobe,
                            const varying HairState& hairState,
                            const varying float variance)
{
    // Uniform sampling TRRT
    return HairTRRTBsdfLobe_evalNTerm(lobe,
                                      hairState,
                                      variance,
                                      NULL);
}

varying float
HairTRRTBsdfLobe_samplePhi(const varying float variance,
                           const varying float r2,
                           const varying float phiO,
                           varying float& phiI)
{
    // dPhi is in [-pi,pi]
    phiI =  (sPi * (2.0f*r2 - 1.0f));
    // Uniform sampling TRRT
    return sOneOverTwoPi;
}

void
HairRBsdfLobe_init(varying HairBsdfLobe * uniform lobe,
                   const varying Vec3f& hairDir,
                   const varying Vec2f& hairUV,
                   const varying float mediumIOR,
                   const varying float ior,
                   const varying HairFresnelType fType,
                   const varying float cuticleLayers,
                   const varying float longShift,
                   const varying float longRoughness,
                   const varying Color& hairTint)
{
    varying float azimRoughness = 0.75f;
    varying Color hairColor, hairSigmaA;
    hairColor = hairSigmaA = sWhite;
    const varying float hairRotation = 0.f;
    const varying Vec3f hairNormal = Vec3f_ctor(0.0f);

    // Note RLobe uses unique Fresnel and Absorption Terms
    HairBsdfLobe_init((varying HairBsdfLobe * uniform)lobe,
                      BSDF_LOBE_HAIR_R,
                      BSDF_LOBE_TYPE_REFLECTION | BSDF_LOBE_TYPE_GLOSSY,
                      hairDir, hairUV,
                      mediumIOR, ior,
                      fType, cuticleLayers,
                      longShift, longRoughness,
                      azimRoughness,
                      hairColor, hairSigmaA, hairTint,
                      hairRotation, hairNormal,
                      1.0f, // saturation
                      HairRBsdfLobe_evalFresnelTerm,
                      HairRBsdfLobe_evalAbsorptionTerm,
                      HairBaseBsdfLobe_evalMTerm,
                      HairRBsdfLobe_evalNTermWithAbsorption,
                      HairBaseBsdfLobe_evalPhiPdf,
                      HairBaseBsdfLobe_evalThetaPdf,
                      HairBaseBsdfLobe_samplePhi,
                      HairBaseBsdfLobe_sampleTheta,
                      HairBsdfLobe_getProperty,
                      HairRBsdfLobe_show);
}

void
HairTTBsdfLobe_init(varying HairBsdfLobe * uniform lobe,
                    const varying Vec3f& hairDir,
                    const varying Vec2f& hairUV,
                    const varying float mediumIOR,
                    const varying float ior,
                    const varying HairFresnelType fType,
                    const varying float cuticleLayers,
                    const varying float longShift,
                    const varying float longRoughness,
                    const varying float azimRoughness,
                    const varying Color &hairColor,
                    const varying Color &hairSigmaA,
                    const varying Color &hairTint,
                    const varying float hairSaturation)
{
    const varying float hairRotation = 0.f;
    const varying Vec3f hairNormal = Vec3f_ctor(0.0f);

    // Note TTLobe uses unique Fresnel, Absorption & N Terms
    HairBsdfLobe_init((varying HairBsdfLobe * uniform)lobe,
                      BSDF_LOBE_HAIR_TT,
                      BSDF_LOBE_TYPE_TRANSMISSION | BSDF_LOBE_TYPE_GLOSSY,
                      hairDir, hairUV,
                      mediumIOR, ior,
                      fType, cuticleLayers,
                      longShift, longRoughness,
                      azimRoughness,
                      hairColor, hairSigmaA, hairTint,
                      hairRotation, hairNormal,
                      hairSaturation,
                      HairTTBsdfLobe_evalFresnelTerm,
                      HairTTBsdfLobe_evalAbsorptionTerm,
                      HairBaseBsdfLobe_evalMTerm,
                      HairTTBsdfLobe_evalNTermWithAbsorption,
                      HairTTBsdfLobe_evalPhiPdf,
                      HairBaseBsdfLobe_evalThetaPdf,
                      HairTTBsdfLobe_samplePhi,
                      HairBaseBsdfLobe_sampleTheta,
                      HairTTBsdfLobe_getProperty,
                      HairTTBsdfLobe_show);
}

void
HairTRTBsdfLobe_init(varying HairBsdfLobe * uniform lobe,
                    const varying Vec3f& hairDir,
                    const varying Vec2f& hairUV,
                    const varying float mediumIOR,
                    const varying float ior,
                    const varying HairFresnelType fType,
                    const varying float cuticleLayers,
                    const varying float longShift,
                    const varying float longRoughness,
                    const varying Color &hairColor,
                    const varying Color &hairSigmaA,
                    const varying Color &hairTint,
                    const varying bool showGlint,
                    const varying float glintRoughness,
                    const varying float glintEccentricity,
                    const varying float glintSaturation,
                    const varying float hairRotation,
                    const varying Vec3f& hairNormal)
{
    varying float azimRoughness = 0.75f;

    // Note TRTLobe uses unique Fresnel & Absorption Terms
    HairBsdfLobe_init((varying HairBsdfLobe * uniform)lobe,
                      BSDF_LOBE_HAIR_TRT,
                      BSDF_LOBE_TYPE_REFLECTION | BSDF_LOBE_TYPE_GLOSSY,
                      hairDir, hairUV,
                      mediumIOR, ior,
                      fType, cuticleLayers,
                      longShift, longRoughness,
                      azimRoughness,
                      hairColor, hairSigmaA, hairTint,
                      hairRotation, hairNormal,
                      1.0f, // saturation
                      HairTRTBsdfLobe_evalFresnelTerm,
                      HairTRTBsdfLobe_evalAbsorptionTerm,
                      HairBaseBsdfLobe_evalMTerm,
                      HairTRTBsdfLobe_evalNTermWithAbsorption,
                      HairBaseBsdfLobe_evalPhiPdf,
                      HairBaseBsdfLobe_evalThetaPdf,
                      HairBaseBsdfLobe_samplePhi,
                      HairBaseBsdfLobe_sampleTheta,
                      HairBsdfLobe_getProperty,
                      HairTRTBsdfLobe_show);

    lobe->mShowGlint = showGlint;
    lobe->mGlintRoughness = glintRoughness;
    lobe->mGlintEccentricity = glintEccentricity;
    lobe->mGlintSaturation = glintSaturation;
}

void
HairTRRTBsdfLobe_init(varying HairBsdfLobe * uniform lobe,
                      const varying Vec3f& hairDir,
                      const varying Vec2f& hairUV,
                      const varying float mediumIOR,
                      const varying float ior,
                      const varying HairFresnelType fType,
                      const varying float cuticleLayers,
                      const varying float longShift,
                      const varying float longRoughness,
                      const varying Color &hairColor,
                      const varying Color &hairSigmaA,
                      const varying Color &hairTint)
{
    const varying float hairRotation = 0.f;
    const varying Vec3f hairNormal = Vec3f_ctor(0.0f);

    varying float azimRoughness = 0.75f;
    HairBsdfLobe_init((varying HairBsdfLobe * uniform)lobe,
                      BSDF_LOBE_HAIR_TRRT,
                      BSDF_LOBE_TYPE_REFLECTION | BSDF_LOBE_TYPE_GLOSSY,
                      hairDir, hairUV,
                      mediumIOR, ior,
                      fType, cuticleLayers,
                      longShift,
                      longRoughness,
                      azimRoughness,
                      hairColor, hairSigmaA, hairTint,
                      hairRotation, hairNormal,
                      1.0f, // saturation
                      HairTRRTBsdfLobe_evalFresnelTerm,
                      HairTRRTBsdfLobe_evalAbsorptionTerm,
                      HairBaseBsdfLobe_evalMTerm,
                      HairTRRTBsdfLobe_evalNTermWithAbsorption,
                      HairTRRTBsdfLobe_evalPhiPdf,
                      HairBaseBsdfLobe_evalThetaPdf,
                      HairTRRTBsdfLobe_samplePhi,
                      HairBaseBsdfLobe_sampleTheta,
                      HairBsdfLobe_getProperty,
                      HairTRRTBsdfLobe_show);
}

void
HairBsdfLobe_show(const varying HairBsdfLobe * const uniform me,
                   const uniform uint8_t ntab)
{
    const varying Fresnel * const uniform fresnel = me->mFresnel;

    BsdfUtil_indent(ntab+1); print("scale: % % %\n", BSDF_UTIL_EXTRACT(me->mScale.r),
                                                     BSDF_UTIL_EXTRACT(me->mScale.g),
                                                     BSDF_UTIL_EXTRACT(me->mScale.b));
    BsdfUtil_indent(ntab+1); print("hair dir: % % %\n", BSDF_UTIL_EXTRACT(me->mHairDir.x),
                                                        BSDF_UTIL_EXTRACT(me->mHairDir.y),
                                                        BSDF_UTIL_EXTRACT(me->mHairDir.z));
    BsdfUtil_indent(ntab+1); print("hair uv: % %\n", BSDF_UTIL_EXTRACT(me->mHairUV.x),
                                                     BSDF_UTIL_EXTRACT(me->mHairUV.y));
    BsdfUtil_indent(ntab+1); print("medium ior: %\n", BSDF_UTIL_EXTRACT(me->mMediumIOR));
    BsdfUtil_indent(ntab+1); print("eta: %\n", BSDF_UTIL_EXTRACT(me->mIOR));
    BsdfUtil_indent(ntab+1); print("fresnel type: %\n", (int)me->mFresnelType);
    BsdfUtil_indent(ntab+1); print("cuticle layer thickness: %\n", BSDF_UTIL_EXTRACT(me->mCuticleLayerThickness));
    BsdfUtil_indent(ntab+1); print("longitudinal roughness: %\n", BSDF_UTIL_EXTRACT(me->mLongitudinalRoughness));
    BsdfUtil_indent(ntab+1); print("longitudinal variance: %\n", BSDF_UTIL_EXTRACT(me->mLongitudinalVariance));
    BsdfUtil_indent(ntab+1); print("longitudinal shift: %\n", BSDF_UTIL_EXTRACT(me->mLongitudinalShift));
    BsdfUtil_indent(ntab+1); print("azimuthal roughness: %\n", BSDF_UTIL_EXTRACT(me->mAzimuthalRoughness));
    BsdfUtil_indent(ntab+1); print("azimuthal variance: %\n", BSDF_UTIL_EXTRACT(me->mAzimuthalVariance));
    BsdfUtil_indent(ntab+1); print("hair color: % % %\n", BSDF_UTIL_EXTRACT(me->mHairColor.r),
                                                          BSDF_UTIL_EXTRACT(me->mHairColor.g),
                                                          BSDF_UTIL_EXTRACT(me->mHairColor.b));
    BsdfUtil_indent(ntab+1); print("sigmaA: % % %\n", BSDF_UTIL_EXTRACT(me->mSigmaA.r),
                                                      BSDF_UTIL_EXTRACT(me->mSigmaA.g),
                                                      BSDF_UTIL_EXTRACT(me->mSigmaA.b));
    BsdfUtil_indent(ntab+1); print("sin alpha: %\n", BSDF_UTIL_EXTRACT(me->mSinAlpha));
    BsdfUtil_indent(ntab+1); print("cos alpha: %\n", BSDF_UTIL_EXTRACT(me->mCosAlpha));
    BsdfUtil_indent(ntab+1); print("H: %\n", BSDF_UTIL_EXTRACT(me->mH));
    BsdfUtil_indent(ntab+1); print("dD factor: %\n", BSDF_UTIL_EXTRACT(me->mdDFactor));
    BsdfUtil_indent(ntab+1); print("tint: % % %\n", BSDF_UTIL_EXTRACT(me->mTint.r),
                                                    BSDF_UTIL_EXTRACT(me->mTint.g),
                                                    BSDF_UTIL_EXTRACT(me->mTint.b));
    BsdfUtil_indent(ntab+1); print("transmission saturation: %\n", BSDF_UTIL_EXTRACT(me->mSaturation));
    BsdfUtil_indent(ntab+1); print("hair rotation: %\n", BSDF_UTIL_EXTRACT(me->mHairRotation));
    BsdfUtil_indent(ntab+1); print("hair normal: % % %\n", BSDF_UTIL_EXTRACT(me->mHairNormal.x),
                                                           BSDF_UTIL_EXTRACT(me->mHairNormal.y),
                                                           BSDF_UTIL_EXTRACT(me->mHairNormal.z));
    BsdfUtil_indent(ntab+1); print("show glint: %\n", (bool)me->mShowGlint);
    BsdfUtil_indent(ntab+1); print("glint roughness: %\n", BSDF_UTIL_EXTRACT(me->mGlintRoughness));
    BsdfUtil_indent(ntab+1); print("glint eccentricity: %\n", BSDF_UTIL_EXTRACT(me->mGlintEccentricity));
    BsdfUtil_indent(ntab+1); print("glint saturation: %\n", BSDF_UTIL_EXTRACT(me->mGlintSaturation));
    if (fresnel) {
        Fresnel_show(fresnel, ntab+1);
    }
}

void
HairRBsdfLobe_show(const varying BsdfLobe * const uniform lobe,
                   const uniform uint8_t ntab)
{
    const varying HairBsdfLobe * uniform me =
        (const varying HairBsdfLobe * uniform) lobe;

    BsdfUtil_indent(ntab);   print("[HairRLobe] mask: %\n", lobe->mMask);
    HairBsdfLobe_show(me, ntab);
}

