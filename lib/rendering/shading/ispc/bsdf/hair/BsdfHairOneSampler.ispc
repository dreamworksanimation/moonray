// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

/// @file HairBsdfOneSampler.ispc

#include "BsdfHairOneSampler.isph"
#include <moonray/rendering/shading/ispc/bsdf/BsdfUtil.isph>
#include <moonray/rendering/shading/ispc/ShaderMacros.isph>
#include <scene_rdl2/common/platform/IspcUtil.isph>

#define PBR_HAIR_USE_UNIFORM_SAMPLING 0

void
HairOneSamplerBsdfLobe_calculateSamplingWeightsAndCDF(const varying HairOneSamplerBsdfLobe * uniform hairLobe,
                                                      const varying HairState& hairState,
                                                      varying float* varying weights,
                                                      varying float* varying cdf)
{
    const Color fresnel = HairBsdfLobe_evalHairFresnel((varying HairBsdfLobe * uniform)hairLobe,
                                                       hairState,
                                                       hairState.mCosThetaO);
    const Color oneMinusFresnel = max(sBlack, sWhite - fresnel);

    float sum = 0.0f;
    weights[0] = 0.0f;
    weights[1] = 0.0f;
    weights[2] = 0.0f;
    if (hairLobe->mShowR) {
        weights[0] = max(0.2f, luminance(fresnel * HairRBsdfLobe_evalAbsorptionTerm(hairState,
                                                                                    hairLobe->mRTint)));
        sum += weights[0];
    }
    if (hairLobe->mShowTRT) {
        weights[1] = luminance(fresnel * oneMinusFresnel * oneMinusFresnel * HairTRTBsdfLobe_evalAbsorptionTerm(hairState,
                                                                                                                hairLobe->mTRTTint));
        sum += weights[1];
    }
    if (hairLobe->mShowTT) {
        weights[2] = luminance(oneMinusFresnel* oneMinusFresnel * HairTTBsdfLobe_evalAbsorptionTerm(hairState,
                                                                                                    hairLobe->mTTTint));
        sum += weights[2];
    }
    if (!isZero(sum)) {
        for (int i = 0; i < 3; ++i) {
            weights[i] /= sum;
        }
    }
    cdf[0] = weights[0];
    cdf[1] = cdf[0] + weights[1];
    cdf[2] = cdf[1] + weights[2];
}

float
HairOneSamplerBsdfLobe_evalPdf(const varying HairOneSamplerBsdfLobe * uniform hairLobe,
                               const varying HairState& hairState,
                               const varying float* const varying weights)
{
#if PBR_HAIR_USE_UNIFORM_SAMPLING
    return  1.0f / sFourPi;
#else
    float pdf = 0.0f;
    if (hairLobe->mShowR) {
        const float rProb   = weights[0] *
                              HairBaseBsdfLobe_evalThetaPdf(hairState,
                                                            hairLobe->mRSinAlpha,
                                                            hairLobe->mRCosAlpha,
                                                            hairLobe->mRLongitudinalVariance) *
                              HairBaseBsdfLobe_evalPhiPdf((varying HairBsdfLobe * uniform)hairLobe,
                                                          hairState);
        pdf = pdf + rProb;
    }
    if (hairLobe->mShowTRT) {
        const float trtProb   = weights[1] *
                                HairBaseBsdfLobe_evalThetaPdf(hairState,
                                                              hairLobe->mTRTSinAlpha,
                                                              hairLobe->mTRTCosAlpha,
                                                              hairLobe->mTRTLongitudinalVariance) *
                               HairBaseBsdfLobe_evalPhiPdf((varying HairBsdfLobe * uniform)hairLobe,
                                                           hairState);
        pdf = pdf + trtProb;
    }
    if (hairLobe->mShowTT) {
        const float ttProb   = weights[2] *
                               HairBaseBsdfLobe_evalThetaPdf(hairState,
                                                             hairLobe->mTTSinAlpha,
                                                             hairLobe->mTTCosAlpha,
                                                             hairLobe->mTTLongitudinalVariance) *
                               HairTTBsdfLobe_evalPhiPdf((varying HairBsdfLobe * uniform)hairLobe,
                                                         hairState,
                                                         hairLobe->mTTAzimuthalVariance);
        pdf = pdf + ttProb;
    }
    return max(0.0f, pdf);
#endif
}

float
HairOneSamplerBsdfLobe_evalPdf(const varying HairOneSamplerBsdfLobe * uniform lobe,
                               const HairState& hairState)
{
    float weights[3];
    float cdf[3];
    HairOneSamplerBsdfLobe_calculateSamplingWeightsAndCDF(lobe,
                                                          hairState,
                                                          weights,
                                                          cdf);
    return HairOneSamplerBsdfLobe_evalPdf(lobe, hairState, weights);
}

varying Color
HairOneSamplerBsdfLobe_evalBsdf(const varying HairOneSamplerBsdfLobe * uniform lobe,
                                const varying HairState& hairState,
                                const varying HairBsdfLobeGlintAttrs * uniform glintAttrs,
                                const varying bool includeCosineTerm)
{
    Color bsdf = sBlack;

    const Color fresnel = HairBsdfLobe_evalHairFresnel((varying HairBsdfLobe * uniform)lobe,
                                                       hairState,
                                                       hairState.mCosThetaD);

    const Color oneMinusFresnel = sWhite - fresnel;

    if (lobe->mShowR) {
        bsdf = bsdf + fresnel *
                        HairBaseBsdfLobe_evalMTerm(hairState,
                                                   lobe->mRSinAlpha,
                                                   lobe->mRCosAlpha,
                                                   lobe->mRLongitudinalVariance) *
                        HairRBsdfLobe_evalNTermWithAbsorption((varying HairBsdfLobe * uniform)lobe,
                                                              hairState,
                                                              lobe->mRTint);
    }

    if (lobe->mShowTRT) {
        bsdf = bsdf + fresnel * oneMinusFresnel * oneMinusFresnel *
                HairBaseBsdfLobe_evalMTerm(hairState,
                                           lobe->mTRTSinAlpha,
                                           lobe->mTRTCosAlpha,
                                           lobe->mTRTLongitudinalVariance) *
                HairTRTBsdfLobe_evalNTermWithAbsorption((varying HairBsdfLobe * uniform)lobe,
                                                        hairState,
                                                        lobe->mTRTTint,
                                                        glintAttrs);

    }

    if (lobe->mShowTT) {
        bsdf = bsdf + oneMinusFresnel * oneMinusFresnel *
                HairBaseBsdfLobe_evalMTerm(hairState,
                                           lobe->mTTSinAlpha,
                                           lobe->mTTCosAlpha,
                                           lobe->mTTLongitudinalVariance) *
                HairTTBsdfLobe_evalNTermWithAbsorption((varying HairBsdfLobe * uniform)lobe,
                                                       hairState,
                                                       lobe->mTTTint,
                                                       NULL,
                                                       lobe->mTTAzimuthalVariance,
                                                       lobe->mTTSaturation);
    }
    if (lobe->mShowTRRT) {
        const Color trrt = HairTRRTBsdfLobe_evalCompensationTerm(fresnel,
                                                                 oneMinusFresnel,
                                                                 hairState.mAbsorptionTerm);
        bsdf = bsdf + trrt *
        HairBaseBsdfLobe_evalMTerm(hairState,
                                   0.0f,
                                   1.0f,
                                   lobe->mTRRTLongitudinalVariance) *
        HairTRRTBsdfLobe_evalNTermWithAbsorption((varying HairBsdfLobe * uniform)lobe,
                                                 hairState,
                                                 (varying Color)sWhite);
    }

    return bsdf;
}

varying Color
HairOneSamplerBsdfLobe_eval(const varying BsdfLobe * uniform lobe,
                            const varying BsdfSlice &slice,
                            const varying Vec3f &wi,
                            varying float * uniform pdf)
{
    const varying HairOneSamplerBsdfLobe * uniform hairLobe =
            (const varying HairOneSamplerBsdfLobe * uniform) lobe;

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

    Color bsdf = lobe->mScale * HairOneSamplerBsdfLobe_evalBsdf(hairLobe,
                                                                hairState,
                                                                &glintAttrs,
                                                                slice.mIncludeCosineTerm);

    if (pdf) *pdf = HairOneSamplerBsdfLobe_evalPdf(hairLobe,
                                                   hairState);

    return bsdf;
}

//-----------------------------------------------------------------------------------//
varying Color
HairOneSamplerBsdfLobe_sample(const varying BsdfLobe * uniform lobe,
                              const varying BsdfSlice &slice,
                              varying float r1,
                              varying float r2,
                              varying Vec3f &wi,
                              varying float &pdf)
{
    const varying HairOneSamplerBsdfLobe * uniform hairLobe =
            (const varying HairOneSamplerBsdfLobe * uniform) lobe;

    if (!hairLobe->mShowR && !hairLobe->mShowTT && !hairLobe->mShowTRT) {
        pdf = 0.0f;
        return sBlack;
    }
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


#if PBR_HAIR_USE_UNIFORM_SAMPLING
    const varying Vec3f dir = sampleSphereUniform(r1, r2);
    wi = HairState_localToGlobal(&hairState, dir);
    return HairOneSamplerBsdfLobe_eval(lobe, slice, wi, &pdf);
#else
    float phiI, thetaI;

    const HairOneSamplerBsdfLobe* sampleLobe;

    float weights[3];
    float cdf[3];
    HairOneSamplerBsdfLobe_calculateSamplingWeightsAndCDF(hairLobe,
                                                          hairState,
                                                          weights,
                                                          cdf);

    float sinAlpha, cosAlpha, longVariance, azimVariance  = 0.0f;
    if (r1 < cdf[0]) {
        r1 = r1 / weights[0];
        sinAlpha = hairLobe->mRSinAlpha;
        cosAlpha = hairLobe->mRCosAlpha;
        longVariance = hairLobe->mRLongitudinalVariance;
        HairBaseBsdfLobe_samplePhi(azimVariance,
                                   r2,
                                   hairState.mPhiO,
                                   phiI);
    } else if (r1 < cdf[1]) {
        r1 = (r1 - cdf[0]) / weights[1];
        sinAlpha = hairLobe->mTRTSinAlpha;
        cosAlpha = hairLobe->mTRTCosAlpha;
        longVariance = hairLobe->mTRTLongitudinalVariance;
        HairBaseBsdfLobe_samplePhi(azimVariance,
                                   r2,
                                   hairState.mPhiO,
                                   phiI);
    } else {
        r1 = clamp((r1 - cdf[1]) / weights[2], 0.0f, 0.99f);
        sinAlpha = hairLobe->mTTSinAlpha;
        cosAlpha = hairLobe->mTTCosAlpha;
        longVariance = hairLobe->mTTLongitudinalVariance;
        azimVariance = hairLobe->mTTAzimuthalVariance;
        HairTTBsdfLobe_samplePhi(azimVariance,
                                 r2,
                                 hairState.mPhiO,
                                 phiI);
    }

    HairBaseBsdfLobe_sampleTheta(sinAlpha,
                                 cosAlpha,
                                 longVariance,
                                 r1,
                                 hairState.mThetaO,
                                 thetaI);

    MNRY_ASSERT(phiI >= -sPi  &&  phiI <= sPi);

    float sinPhi, cosPhi;
    sincos(phiI, &sinPhi, &cosPhi);
    float sinTheta, cosTheta;
    sincos(thetaI, &sinTheta, &cosTheta);
    if (abs(cosTheta) > (1.0f - sEpsilon)) {
        pdf = 0.0f;
        return sBlack;
    }

    // Compute the light direction vector for shading.
    // TODO: use ReferenceFrame local2global
    const float uWgt = sinTheta;
    const float vWgt = cosTheta  * cosPhi;
    const float wWgt = cosTheta * sinPhi;

    wi = HairState_localToGlobal(&hairState,
                                 uWgt, vWgt, wWgt);

    HairState_updateAngles(&hairState,
                           wi,
                           phiI,
                           thetaI);

    pdf = HairOneSamplerBsdfLobe_evalPdf(hairLobe,
                                         hairState,
                                         weights);

    return lobe->mScale * HairOneSamplerBsdfLobe_evalBsdf(hairLobe,
                                                          hairState,
                                                          &glintAttrs,
                                                          slice.mIncludeCosineTerm);
#endif
}

inline void
showLobeInfo(const uniform uint8_t ntab,
             const varying Color& scale,
             const varying Vec3f& hairDir,
             const varying Vec2f& hairUV,
             const varying float mediumIOR,
             const varying float ior,
             const varying HairFresnelType fresnelType,
             const varying float cuticleLayerThickness,
             const varying float longRoughness,
             const varying float longVariance,
             const varying float longShift,
             const varying float aziRoughness,
             const varying float aziVariance,
             const varying float sinAlpha,
             const varying float cosAlpha,
             const varying float H,
             const varying float dDFactor,
             const varying Color& tint,
             const varying Color& hairColor,
             const varying Color& hairSigmaA,
             const varying float saturation,
             const varying float hairRotation,
             const varying Vec3f& hairNormal,
             const varying bool showGlint,
             const varying float glintRoughness,
             const varying float glintEccentricity,
             const varying float glintSaturation)
{
    BsdfUtil_indent(ntab+1); print("scale: % % %\n", BSDF_UTIL_EXTRACT(scale.r),
                                                     BSDF_UTIL_EXTRACT(scale.g),
                                                     BSDF_UTIL_EXTRACT(scale.b));
    BsdfUtil_indent(ntab+1); print("hair dir: % % %\n", BSDF_UTIL_EXTRACT(hairDir.x),
                                                        BSDF_UTIL_EXTRACT(hairDir.y),
                                                        BSDF_UTIL_EXTRACT(hairDir.z));
    BsdfUtil_indent(ntab+1); print("hair uv: % %\n", BSDF_UTIL_EXTRACT(hairUV.x),
                                                     BSDF_UTIL_EXTRACT(hairUV.y));
    BsdfUtil_indent(ntab+1); print("medium ior: %\n", BSDF_UTIL_EXTRACT(mediumIOR));
    BsdfUtil_indent(ntab+1); print("eta: %\n", BSDF_UTIL_EXTRACT(ior));
    BsdfUtil_indent(ntab+1); print("fresnel type: %\n", (int)fresnelType);
    BsdfUtil_indent(ntab+1); print("cuticle layer thickness: %\n", BSDF_UTIL_EXTRACT(cuticleLayerThickness));
    BsdfUtil_indent(ntab+1); print("longitudinal roughness: %\n", BSDF_UTIL_EXTRACT(longRoughness));
    BsdfUtil_indent(ntab+1); print("longitudinal variance: %\n", BSDF_UTIL_EXTRACT(longVariance));
    BsdfUtil_indent(ntab+1); print("longitudinal shift: %\n", BSDF_UTIL_EXTRACT(longShift));
    BsdfUtil_indent(ntab+1); print("azimuthal roughness: %\n", BSDF_UTIL_EXTRACT(aziRoughness));
    BsdfUtil_indent(ntab+1); print("azimuthal variance: %\n", BSDF_UTIL_EXTRACT(aziVariance));
    BsdfUtil_indent(ntab+1); print("hair color: % % %\n", BSDF_UTIL_EXTRACT(hairColor.r),
                                                          BSDF_UTIL_EXTRACT(hairColor.g),
                                                          BSDF_UTIL_EXTRACT(hairColor.b));
    BsdfUtil_indent(ntab+1); print("sigmaA: % % %\n", BSDF_UTIL_EXTRACT(hairSigmaA.r),
                                                      BSDF_UTIL_EXTRACT(hairSigmaA.g),
                                                      BSDF_UTIL_EXTRACT(hairSigmaA.b));
    BsdfUtil_indent(ntab+1); print("sin alpha: %\n", BSDF_UTIL_EXTRACT(sinAlpha));
    BsdfUtil_indent(ntab+1); print("cos alpha: %\n", BSDF_UTIL_EXTRACT(cosAlpha));
    BsdfUtil_indent(ntab+1); print("H: %\n", BSDF_UTIL_EXTRACT(H));
    BsdfUtil_indent(ntab+1); print("dD factor: %\n", BSDF_UTIL_EXTRACT(dDFactor));
    BsdfUtil_indent(ntab+1); print("tint: % % %\n", BSDF_UTIL_EXTRACT(tint.r),
                                                    BSDF_UTIL_EXTRACT(tint.g),
                                                    BSDF_UTIL_EXTRACT(tint.b));
    BsdfUtil_indent(ntab+1); print("transmission saturation: %\n", BSDF_UTIL_EXTRACT(saturation));
    BsdfUtil_indent(ntab+1); print("hair rotation: %\n", BSDF_UTIL_EXTRACT(hairRotation));
    BsdfUtil_indent(ntab+1); print("hair normal: % % %\n", BSDF_UTIL_EXTRACT(hairNormal.x),
                                                           BSDF_UTIL_EXTRACT(hairNormal.y),
                                                           BSDF_UTIL_EXTRACT(hairNormal.z));
    BsdfUtil_indent(ntab+1); print("show glint: %\n", (bool)showGlint);
    BsdfUtil_indent(ntab+1); print("glint roughness: %\n", BSDF_UTIL_EXTRACT(glintRoughness));
    BsdfUtil_indent(ntab+1); print("glint eccentricity: %\n", BSDF_UTIL_EXTRACT(glintEccentricity));
    BsdfUtil_indent(ntab+1); print("glint saturation: %\n", BSDF_UTIL_EXTRACT(glintSaturation));

}

inline void
HairOneSamplerBsdfLobe_show(const varying BsdfLobe * const uniform lobe,
                            const uniform uint8_t ntab)
{
    const varying HairOneSamplerBsdfLobe * uniform me =
        (const varying HairOneSamplerBsdfLobe * uniform) lobe;

    const varying Fresnel * const uniform fresnel = me->mFresnel;

    BsdfUtil_indent(ntab);   print("[HairOneSampleLobe] mask: %\n", lobe->mMask);

    BsdfUtil_indent(ntab+1); print("scale: % % %\n", BSDF_UTIL_EXTRACT(me->mScale.r),
                                                     BSDF_UTIL_EXTRACT(me->mScale.g),
                                                     BSDF_UTIL_EXTRACT(me->mScale.b));
    if (fresnel) {
        Fresnel_show(fresnel, ntab+1);
    }

    const varying Color varyWhite = Color_ctor(1.0f, 1.0f, 1.0f);
    const varying Vec3f zeroVec3 = Vec3f_ctor(0.0f, 0.0f, 0.0f);

    // we'll print some "fake" lobes here to match the formatting
    // of the scalar version, to make comparing the two easier.
    if (me->mShowR) {
        BsdfUtil_indent(ntab+1); print("[HairRLobe] mask: %\n", lobe->mMask);
        showLobeInfo(ntab+1,
                     varyWhite, // scale
                     me->mHairDir,
                     me->mHairUV,
                     me->mMediumIOR,
                     me->mIOR,
                     me->mFresnelType,
                     me->mCuticleLayerThickness,
                     me->mRLongitudinalRoughness,
                     me->mRLongitudinalVariance,
                     me->mRLongitudinalShift,
                     0.75f, // aziRoughness, hard-coded in scalar
                     0.0f,  // aziVariance
                     me->mRSinAlpha,
                     me->mRCosAlpha,
                     me->mH,
                     me->mdDFactor,
                     me->mRTint,
                     varyWhite,  // hairColor
                     varyWhite,  // sigmaA
                     0.0f,       // transmission saturation
                     0.0f,       // hair rotation
                     zeroVec3,   // hair normal
                     false,      // show glint
                     0.0f,       // glint roughness
                     0.0f,       // glint eccentricity
                     0.0f);      // glint saturation
    }

    if (me->mShowTT) {
        BsdfUtil_indent(ntab+1); print("[HairTTLobe] mask: %\n", lobe->mMask);
        showLobeInfo(ntab+1,
                     varyWhite, // scale
                     me->mHairDir,
                     me->mHairUV,
                     me->mMediumIOR,
                     me->mIOR,
                     me->mFresnelType,
                     me->mCuticleLayerThickness,
                     me->mTTLongitudinalRoughness,
                     me->mTTLongitudinalVariance,
                     me->mTTLongitudinalShift,
                     me->mTTAzimuthalRoughness,
                     me->mTTAzimuthalVariance,
                     me->mTTSinAlpha,
                     me->mTTCosAlpha,
                     me->mH,
                     me->mdDFactor,
                     me->mTTTint,
                     me->mHairColor,
                     me->mSigmaA,
                     me->mSaturation,
                     0.0f,       // hair rotation
                     zeroVec3,   // hair normal
                     false,      // show glint
                     0.0f,       // glint roughness
                     0.0f,       // glint eccentricity
                     0.0f);      // glint saturation
    }

    if (me->mShowTRT) {
        BsdfUtil_indent(ntab+1); print("[HairTRTLobe] mask: %\n", lobe->mMask);
        showLobeInfo(ntab+1,
                     varyWhite, // scale
                     me->mHairDir,
                     me->mHairUV,
                     me->mMediumIOR,
                     me->mIOR,
                     me->mFresnelType,
                     me->mCuticleLayerThickness,
                     me->mTRTLongitudinalRoughness,
                     me->mTRTLongitudinalVariance,
                     me->mTRTLongitudinalShift,
                     0.75f, // aziRoughness, hard-coded in scalar
                     0.0f,  // aziVariance
                     me->mTRTSinAlpha,
                     me->mTRTCosAlpha,
                     me->mH,
                     me->mdDFactor,
                     me->mTRTTint,
                     me->mHairColor,
                     me->mSigmaA,
                     0.0f,       // transmission saturation
                     me->mHairRotation,
                     me->mHairNormal,
                     me->mShowGlint,
                     me->mGlintRoughness,
                     me->mGlintEccentricity,
                     me->mGlintSaturation);
    }

    if (me->mShowTRRT) {
        BsdfUtil_indent(ntab+1); print("[HairTRRTLobe] mask: %\n", lobe->mMask);
        showLobeInfo(ntab+1,
                     varyWhite, // scale
                     me->mHairDir,
                     me->mHairUV,
                     me->mMediumIOR,
                     me->mIOR,
                     me->mFresnelType,
                     me->mCuticleLayerThickness,
                     me->mTRRTLongitudinalRoughness,
                     me->mTRRTLongitudinalVariance,
                     0.0f,  // longShift, hard-coded in scalar
                     0.75f, // aziRoughness, hard-coded in scalar
                     0.0f,  // aziVariance
                     0.0f,  // sin(longShift)
                     1.0f, //  cos(longShift)
                     me->mH,
                     me->mdDFactor,
                     me->mTRTTint,
                     me->mHairColor,
                     me->mSigmaA,
                     0.0f,       // transmission saturation
                     0.0f,       // hair rotation
                     zeroVec3,   // hair normal
                     false,      // show glint
                     0.0f,       // glint roughness
                     0.0f,       // glint eccentricity
                     0.0f);      // glint saturation
    }
}

void
HairOneSamplerBsdfLobe_init(varying HairOneSamplerBsdfLobe * uniform lobe,
                            const varying Vec3f& hairDir,
                            const varying Vec2f& hairUV,
                            const varying float mediumIOR,
                            const varying float ior,
                            const varying HairFresnelType fresnelType,
                            const varying float cuticleLayerThickness,
                            const varying bool  showR,
                            const varying float rShift,
                            const varying float rLongRoughness,
                            const varying Color& rTint,
                            const varying bool  showTT,
                            const varying float ttShift,
                            const varying float ttLongRoughness,
                            const varying float ttAzimRoughness,
                            const varying Color& ttTint,
                            const varying float ttSaturation,
                            const varying bool  showTRT,
                            const varying float trtShift,
                            const varying float trtLongRoughness,
                            const varying Color& trtTint,
                            const varying bool showGlint,
                            const varying float glintRoughness,
                            const varying float glintEccentricity,
                            const varying float glintSaturation,
                            const varying float hairRotation,
                            const varying Vec3f& hairNormal,
                            const varying bool  showTRRT,
                            const varying float trrtLongRoughness,
                            const varying Color& hairColor,
                            const varying Color& hairSigmaA)
{
    MNRY_ASSERT(rLongRoughness > 0.0f && rLongRoughness < 1.0f);
    MNRY_ASSERT(ttLongRoughness > 0.0f && ttLongRoughness < 1.0f);
    MNRY_ASSERT(trtLongRoughness > 0.0f && trtLongRoughness < 1.0f);

    BsdfLobe_init((varying BsdfLobe * uniform) lobe,
                  BSDF_LOBE_HAIR_ONE_SAMPLER,
                  BSDF_LOBE_TYPE_REFLECTION | BSDF_LOBE_TYPE_TRANSMISSION | BSDF_LOBE_TYPE_GLOSSY,
                  /* differentialFlags = */ BSDF_LOBE_IGNORES_INCOMING_DIFFERENTIALS,
                  /* isSpherical = */ true,
                  /* propertyFlags = */ BSDF_LOBE_PROPERTY_ROUGHNESS |
                      BSDF_LOBE_PROPERTY_COLOR | BSDF_LOBE_PROPERTY_NORMAL,
                  /* evalFn = */ HairOneSamplerBsdfLobe_eval,
                  /* sampleFn = */ HairOneSamplerBsdfLobe_sample,
                  /* albedoFn = */ HairBsdfLobe_albedo,
                  /* differentialsFn = */ HairBsdfLobe_differentials,
                  /* getPropertyFn = */ HairOneSamplerBsdfLobe_getProperty,
                  /* showFn = */ HairOneSamplerBsdfLobe_show);

    BsdfLobe_setIsHair((varying BsdfLobe * uniform) lobe, true);

    lobe->mHairDir  = hairDir;
    lobe->mHairUV   = hairUV;
    // For Moonray, Texture 't' varies across the width of hair curve
    // and goes from [0,1].   We remap this to [-1,1].
    // Note - this is not true of all curve types
    // Please take a look at MOONRAY-3116 and make sure that is fixed.
    lobe->mH = hairUV.y * 2.0f - 1.0f;

    lobe->mMediumIOR = mediumIOR;
    lobe->mIOR = ior;

    lobe->mHairColor    = hairColor;
    lobe->mSigmaA       = hairSigmaA;

    lobe->mHairRotation = hairRotation;
    lobe->mHairNormal = hairNormal;

    lobe->mShowR    = showR;
    lobe->mShowTT   = showTT;
    lobe->mShowTRT  = showTRT;
    lobe->mShowTRRT = showTRRT;

    // Derive a directional differential scale that varies according to the
    // smallest of longitudinal and azimuthal width
    // TODO - Find how this impacts renders
    float minWidth = 0.0f;
    if (lobe->mShowR) {
        sincos(rShift,   &lobe->mRSinAlpha,   &lobe->mRCosAlpha);
        lobe->mRLongitudinalRoughness     = rLongRoughness;
        lobe->mRLongitudinalVariance      = HairUtil_longitudinalVar(rLongRoughness);
        lobe->mRLongitudinalShift         = rShift;
        lobe->mRTint   = rTint;

        minWidth = min(roughness2Width(rLongRoughness), minWidth);
    }

    if (lobe->mShowTT) {
        sincos(ttShift,  &lobe->mTTSinAlpha,  &lobe->mTTCosAlpha);
        lobe->mTTLongitudinalRoughness    = ttLongRoughness;
        lobe->mTTAzimuthalRoughness       = ttAzimRoughness;
        lobe->mTTLongitudinalVariance     = HairUtil_longitudinalVar(ttLongRoughness);
        lobe->mTTLongitudinalShift        = ttShift;
        lobe->mTTAzimuthalVariance        = HairUtil_azimuthalVar(ttAzimRoughness);;
        lobe->mTTTint  = ttTint;
        lobe->mTTSaturation               = ttSaturation;

        minWidth = min(roughness2Width(ttLongRoughness), minWidth);
    }

    if (lobe->mShowTRT) {
        sincos(trtShift, &lobe->mTRTSinAlpha, &lobe->mTRTCosAlpha);
        lobe->mTRTLongitudinalRoughness   = trtLongRoughness;
        lobe->mTRTLongitudinalVariance    = HairUtil_longitudinalVar(trtLongRoughness);
        lobe->mTRTLongitudinalShift       = trtShift;
        lobe->mTRTTint = trtTint;

        lobe->mShowGlint = showGlint;
        lobe->mGlintRoughness = glintRoughness;
        lobe->mGlintEccentricity = glintEccentricity;
        lobe->mGlintSaturation = glintSaturation;

        minWidth = min(roughness2Width(trtLongRoughness), minWidth);
    }

    if (lobe->mShowTRRT) {
        lobe->mTRRTLongitudinalRoughness   = trrtLongRoughness;
        // calculate variance
        lobe->mTRRTLongitudinalVariance    = HairUtil_longitudinalVar(trrtLongRoughness);
    }

    // Derive a directional differential scale that varies according to width
    lobe->mdDFactor = HairBsdfLobe_sdDFactorMin + minWidth * HairBsdfLobe_sdDFactorSlope;

    lobe->mFresnelType = (HairFresnelType) fresnelType;
    // Cuticle-Layer has been parameterized to be between 0 and 1
    // Layers vary in between [0.5, 1.5]
    // See Table 2
    /// From "Physically-Accurate Fur Reflectance: Modeling, Measurement and Rendering"
    lobe->mCuticleLayerThickness = lerp(0.5f, 1.5f,
                                        clamp(cuticleLayerThickness, 0.0f, 1.0f));
}

#pragma ignore warning(all)
ISPC_UTIL_EXPORT_STRUCT_TO_HEADER(HairOneSamplerBsdfLobe);

