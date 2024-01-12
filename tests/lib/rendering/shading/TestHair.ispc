// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

#include <moonray/rendering/shading/ispc/bsdf/BsdfSlice.isph>
#include <moonray/rendering/shading/ispc/bsdf/hair/BsdfHairLobes.isph>
#include <scene_rdl2/common/platform/Platform.isph>

#define AS_HAIRSTATE(A, B) \
varying HairState * uniform A = (varying HairState * uniform) B

#define AS_HAIRBSDFLOBE(A, B) \
varying HairBsdfLobe * uniform A = (varying HairBsdfLobe * uniform) B

uniform Color
extractColor(const varying Color &c, uniform int i)
{
    uniform Color result = Color_ctor(
            extract(c.r, i),
            extract(c.g, i),
            extract(c.b, i));
    return result;
}

uniform Vec2f
extractVec2f(const varying Vec2f &v, uniform int i)
{
    uniform Vec2f result = Vec2f_ctor(
            extract(v.x, i),
            extract(v.y, i));
    return result;
}

uniform Vec3f
extractVec3f(const varying Vec3f &v, uniform int i)
{
    uniform Vec3f result = Vec3f_ctor(
            extract(v.x, i),
            extract(v.y, i),
            extract(v.z, i));
    return result;
}

export uniform float
testPow2(const uniform float f)
{
    const varying float result = HairUtil_Pow2(f);
    return extract(result, 0);
}

export uniform float
testPow3(const uniform float f)
{
    const varying float result = HairUtil_Pow3(f);
    return extract(result, 0);
}

export uniform float
testPow4(const uniform float f)
{
    const varying float result = HairUtil_Pow4(f);
    return extract(result, 0);
}

export uniform float
testPow20(const uniform float f)
{
    const varying float result = HairUtil_Pow20(f);
    return extract(result, 0);
}

export uniform float
testPow22(const uniform float f)
{
    const varying float result = HairUtil_Pow22(f);
    return extract(result, 0);
}

export uniform float
testSafeSqrt(const uniform float f)
{
    const varying float result = HairUtil_safeSqrt(f);
    return extract(result, 0);
}

export uniform float
testSqr(const uniform float f)
{
    const varying float result = HairUtil_sqr(f);
    return extract(result, 0);
}

export uniform float
testSinh(const uniform float f)
{
    const varying float result = HairUtil_sinh(f);
    return extract(result, 0);
}

export uniform float
testAzimuthalVar(const uniform float f)
{
    const varying float result = HairUtil_azimuthalVar(f);
    return extract(result, 0);
}

export uniform float
testLongitudinalVar(const uniform float f)
{
    const varying float result = HairUtil_longitudinalVar(f);
    return extract(result, 0);
}

export void
testComputeAbsorptionCoefficients(const uniform Color &c,
                                  const uniform float f,
                                  uniform Color &result)
{
    const varying Color vc = c;
    const varying Color alpha = HairUtil_computeAbsorptionCoefficients(vc, f);
    result = extractColor(alpha, 0);
}

export uniform float
testBesselIO(const uniform float f)
{
    const varying float result = HairUtil_besselIO(f);
    return extract(result, 0);
}

export uniform float
testLogBesselIO(const uniform float f)
{
    const varying float result = HairUtil_logBesselIO(f);
    return extract(result, 0);
}

export uniform float
testDeonLongitudinalM(const uniform float variance,
                      const uniform float sinThetaI, const uniform float cosThetaI,
                      const uniform float sinThetaO, const uniform float cosThetaO)
{
    const varying float result =
        HairUtil_deonLongitudinalM(variance, sinThetaI, cosThetaI, sinThetaO, cosThetaO);
    return extract(result, 0);
}

export uniform float
testLogisticFunction(const uniform float x, const uniform float s)
{
    const varying float result = HairUtil_logisticFunction(x, s);
    return extract(result, 0);
}

export uniform float
testLogisticCDF(const uniform float x, const uniform float s)
{
    const varying float result = HairUtil_logisticCDF(x, s);
    return extract(result, 0);
}

export uniform float
testTrimmedLogisticFunction(const uniform float x, const uniform float s,
                            const uniform float a, const uniform float b)
{
    const varying float result = HairUtil_trimmedLogisticFunction(x, s, a, b);
    return extract(result, 0);
}

export uniform float
testSampleTrimmedLogistic(const uniform float x, const uniform float s,
                          const uniform float a, const uniform float b)
{
    const varying float result = HairUtil_sampleTrimmedLogistic(x, s, a, b);
    return extract(result, 0);
}

export uniform float
testUnitGaussianForShade(const uniform float stddev, const uniform float x)
{
    const varying float result = HairUtil_unitGaussianForShade(stddev, x);
    return extract(result, 0);
}

export uniform float
testUnitCauchyForShade(const uniform float stddev, const uniform float x)
{
    const varying float result = HairUtil_unitCauchyForShade(stddev, x);
    return extract(result, 0);
}

export uniform float
testUnitCauchyForSample(const uniform float stddev, const uniform float x)
{
    const varying float result = HairUtil_unitCauchyForSample(stddev, x);
    return extract(result, 0);
}

export uniform uint32_t
testCompact1By1(const uniform uint32_t x)
{
    const varying uint32_t result = HairUtil_compact1By1(x);
    return extract(result, 0);
}

export void
testDemuxFloat(const uniform float f, uniform Vec2f &result)
{
    const varying Vec2f v = HairUtil_demuxFloat(f);
    result = extractVec2f(v, 0);
}

export void * uniform
createHairState(const uniform Vec3f& wo,
                const uniform Vec3f& wi,
                const uniform Vec3f &hairDir,
                const uniform float &H,
                const uniform float eta,
                const uniform Color &hairSigmaA,
                const uniform float &hairRotation,
                const uniform Vec3f& hairNormal)
{
    const Vec3f vWo = Vec3f_ctor(wo.x, wo.y, wo.z);
    const Vec3f vWi = Vec3f_ctor(wi.x, wi.y, wi.z);
    const Vec3f vhairDir = Vec3f_ctor(hairDir.x, hairDir.y, hairDir.z);
    const Color vhairSigmaA = Color_ctor(hairSigmaA.r, hairSigmaA.g, hairSigmaA.b);
    const float vH = H;
    const float vhairRotation = hairRotation;
    const Vec3f vhairNormal = hairNormal;

    varying HairState * uniform hs = uniform new varying HairState;
    HairState_init(hs,
                   vWo,
                   vWi,
                   vhairDir,
                   vH,
                   eta,
                   vhairSigmaA,
                   vhairRotation,
                   vhairNormal);
    return (void * uniform) hs;
}

export void
deleteHairState(void * uniform hairState)
{
    AS_HAIRSTATE(hs, hairState);
    delete hs;
}

export void
getAbsorptionTerm(void * uniform hairState, uniform Color &result)
{
    AS_HAIRSTATE(hs, hairState);
    result = extractColor(hs->mAbsorptionTerm, 0);
}

export void
getHairNorm(void * uniform hairState, uniform Vec3f &result)
{
    AS_HAIRSTATE(hs, hairState);
    result = extractVec3f(hs->mHairNorm, 0);
}

export void
getHairBiNorm(void * uniform hairState, uniform Vec3f &result)
{
    AS_HAIRSTATE(hs, hairState);
    result = extractVec3f(hs->mHairBiNorm, 0);
}

export uniform float
getThetaO(void * uniform hairState)
{
    AS_HAIRSTATE(hs, hairState);
    return extract(hs->mThetaO, 0);
}

export uniform float
getPhiI(void * uniform hairState)
{
    AS_HAIRSTATE(hs, hairState);
    return extract(hs->mPhiI, 0);
}

export uniform float
getThetaI(void * uniform hairState)
{
    AS_HAIRSTATE(hs, hairState);
    return extract(hs->mThetaI, 0);
}

export uniform float
getSinThetaO(void * uniform hairState)
{
    AS_HAIRSTATE(hs, hairState);
    return extract(hs->mSinThetaO, 0);
}

export uniform float
getCosThetaO(void * uniform hairState)
{
    AS_HAIRSTATE(hs, hairState);
    return extract(hs->mCosThetaO, 0);
}

export uniform float
getSinThetaI(void * uniform hairState)
{
    AS_HAIRSTATE(hs, hairState);
    return extract(hs->mSinThetaI, 0);
}

export uniform float
getCosThetaI(void * uniform hairState)
{
    AS_HAIRSTATE(hs, hairState);
    return extract(hs->mCosThetaI, 0);
}

export uniform float
getPhiD(void * uniform hairState)
{
    AS_HAIRSTATE(hs, hairState);
    return extract(hs->mPhiD, 0);
}

export uniform float
getThetaD(void * uniform hairState)
{
    AS_HAIRSTATE(hs, hairState);
    return extract(hs->mThetaD, 0);
}

export void
updateAngles(void * uniform hairState,
             const uniform Vec3f &wi,
             const uniform float phiI,
             const uniform float thetaI)
{
    AS_HAIRSTATE(hs, hairState);
    const varying Vec3f vwi = Vec3f_ctor(wi.x, wi.y, wi.z);
    HairState_updateAngles(hs,
                           vwi,
                           phiI,
                           thetaI);

}

export void * uniform
createFresnel(const uniform float etaI, const uniform float etaT)
{
    const varying float vetaI = etaI;
    const varying float vetaT = etaT;
    varying DielectricFresnel * uniform fresnel = uniform new varying DielectricFresnel;
    DielectricFresnel_init(fresnel, vetaI, vetaT);
    return (void * uniform) fresnel;
}

export void
deleteFresnel(void * uniform fresnel)
{
    varying DielectricFresnel * uniform p = (varying DielectricFresnel * uniform) fresnel;
    delete p;
}

export void * uniform
createBsdfSlice(const uniform Vec3f &Ng, const uniform Vec3f &wo,
                const uniform bool includeCosineTerm, const uniform bool entering)
{
    varying BsdfSlice * uniform slice = uniform new varying BsdfSlice;
    const varying Vec3f vNg = Vec3f_ctor(Ng.x, Ng.y, Ng.z);
    const varying Vec3f vwo = Vec3f_ctor(wo.x, wo.y, wo.z);
    const varying bool ict = includeCosineTerm;
    const varying bool ent = entering;
    BsdfSlice_init(slice, vNg, vwo,
                   BSDF_LOBE_TYPE_REFLECTION | BSDF_LOBE_TYPE_GLOSSY,
                   ict, ent, SHADOW_TERMINATOR_FIX_OFF);
    return (void * uniform) slice;
}

export void
deleteBsdfSlice(void * uniform lobe)
{
    varying BsdfSlice * uniform p = (varying BsdfSlice * uniform) lobe;
    delete p;
}

export void * uniform
createRLobe(const uniform Vec3f &hairDir,
            const uniform Vec2f &hairUV,
            const uniform float refractiveIndex,
            const uniform float longShift,
            const uniform float longRoughness,
            void * uniform fresnel)
{
    const Vec3f vhairDir = Vec3f_ctor(hairDir.x, hairDir.y, hairDir.z);
    const Vec2f vhairUV  = Vec2f_ctor(hairUV.x, hairUV.y);
    const float vlongShift = longShift;
    const float vlongRoughness = longRoughness;

    varying HairBsdfLobe * uniform lobe = uniform new varying HairBsdfLobe;

    HairRBsdfLobe_init(lobe,
                       vhairDir,
                       vhairUV,
                       1.0f, //medium IOR
                       refractiveIndex,
                       HAIR_FRESNEL_DIELECTRIC_CYLINDER,
                       0, //cuticle layer thickness
                       vlongShift,
                       vlongRoughness,
                       (varying Color) sWhite);

    BsdfLobe_setFresnel((varying BsdfLobe * uniform) lobe,
                        (varying Fresnel * uniform) fresnel);
    return (void * uniform) lobe;
}

export void * uniform
createTRTLobe(const uniform Vec3f &hairDir,
              const uniform Vec2f &hairUV,
              const uniform float refractiveIndex,
              const uniform float longShift,
              const uniform float longRoughness,
              const uniform Color &hairColor,
              const uniform Color &hairSigmaA,
              const uniform bool &showGlint,
              const uniform float &glintRoughness,
              const uniform float &glintEccentricity,
              const uniform float &glintSaturation,
              const uniform float &hairRotation,
              const uniform Vec3f &hairNormal,
              void * uniform fresnel)
{
    const Vec3f vhairDir = Vec3f_ctor(hairDir.x, hairDir.y, hairDir.z);
    const Vec2f vhairUV  = Vec2f_ctor(hairUV.x, hairUV.y);
    const Color vhairColor = Color_ctor(hairColor.r, hairColor.g, hairColor.b);
    const Color vhairSigmaA = Color_ctor(hairSigmaA.r, hairSigmaA.g, hairSigmaA.b);
    const float vlongShift = longShift;
    const float vlongRoughness = longRoughness;
    const float vglintRoughness = glintRoughness;
    const float vglintEccentricity = glintEccentricity;
    const float vglintSaturation = glintSaturation;
    const float vhairRotation = hairRotation;
    const Vec3f vhairNormal = Vec3f_ctor(hairNormal.x, hairNormal.y, hairNormal.z);

    varying HairBsdfLobe * uniform lobe = uniform new varying HairBsdfLobe;

    HairTRTBsdfLobe_init(lobe,
                         vhairDir,
                         vhairUV,
                         1.0f, //medium IOR
                         refractiveIndex,
                         HAIR_FRESNEL_DIELECTRIC_CYLINDER,
                         0, //cuticle layer thickness
                         vlongShift,
                         vlongRoughness,
                         vhairColor,
                         vhairSigmaA,
                         (varying Color) sWhite,
                         showGlint,
                         vglintRoughness,
                         vglintEccentricity,
                         vglintSaturation,
                         vhairRotation,
                         vhairNormal);

    BsdfLobe_setFresnel((varying BsdfLobe * uniform) lobe,
                        (varying Fresnel * uniform) fresnel);
    return (void * uniform) lobe;
}

export void * uniform
createTTLobe(const uniform Vec3f& hairDir,
             const uniform Vec2f& hairUV,
             const uniform float refractiveIndex,
             const uniform float longShift,
             const uniform float longRoughness,
             const uniform float azimRoughness,
             const uniform Color &hairColor,
             const uniform Color &hairSigmaA,
             void * uniform fresnel)
{
    const Vec3f vhairDir = Vec3f_ctor(hairDir.x, hairDir.y, hairDir.z);
    const Vec2f vhairUV  = Vec2f_ctor(hairUV.x, hairUV.y);
    const Color vhairColor = Color_ctor(hairColor.r, hairColor.g, hairColor.b);
    const Color vhairSigmaA = Color_ctor(hairSigmaA.r, hairSigmaA.g, hairSigmaA.b);
    const float vlongShift = longShift;
    const float vlongRoughness = longRoughness;
    const float vazimRoughness = azimRoughness;

    varying HairBsdfLobe * uniform lobe = uniform new varying HairBsdfLobe;

    HairTTBsdfLobe_init(lobe,
                        vhairDir,
                        vhairUV,
                        1.0f, //medium IOR
                        refractiveIndex,
                        HAIR_FRESNEL_DIELECTRIC_CYLINDER,
                        0, //cuticle layer thickness
                        vlongShift,
                        vlongRoughness,
                        vazimRoughness,
                        vhairColor,
                        vhairSigmaA,
                        (varying Color) sWhite);

    BsdfLobe_setFresnel((varying BsdfLobe * uniform) lobe,
                        (varying Fresnel * uniform) fresnel);
    return (void * uniform) lobe;
}

export void * uniform
createTRRTLobe(const uniform Vec3f& hairDir,
               const uniform Vec2f& hairUV,
               const uniform float refractiveIndex,
               const uniform float longShift,
               const uniform float longRoughness,
               const uniform Color &hairColor,
               const uniform Color &hairSigmaA,
               void * uniform fresnel)
{
    const Vec3f vhairDir = Vec3f_ctor(hairDir.x, hairDir.y, hairDir.z);
    const Vec2f vhairUV  = Vec2f_ctor(hairUV.x, hairUV.y);
    const Color vhairColor = Color_ctor(hairColor.r, hairColor.g, hairColor.b);
    const Color vhairSigmaA = Color_ctor(hairSigmaA.r, hairSigmaA.g, hairSigmaA.b);
    const float vlongShift = longShift;
    const float vlongRoughness = longRoughness;
    varying HairBsdfLobe * uniform lobe = uniform new varying HairBsdfLobe;

    HairTRRTBsdfLobe_init(lobe,
                          vhairDir,
                          vhairUV,
                          1.0f, //medium IOR
                          refractiveIndex,
                          HAIR_FRESNEL_DIELECTRIC_CYLINDER,
                          0, //cuticle layer thickness
                          vlongShift,
                          vlongRoughness,
                          vhairColor,
                          vhairSigmaA,
                          (varying Color) sWhite);

    BsdfLobe_setFresnel((varying BsdfLobe * uniform) lobe,
                        (varying Fresnel * uniform) fresnel);
    return (void * uniform) lobe;
}

export void
deleteLobe(void * uniform lobe)
{
    varying HairBsdfLobe * uniform p =
        (varying HairBsdfLobe * uniform) lobe;
    delete p;
}

export uniform float
getSinAlpha(void * uniform hairLobe)
{
    AS_HAIRBSDFLOBE(lobe, hairLobe);
    varying float result = lobe->mSinAlpha;
    return extract(result, 0);
}

export uniform float
getCosAlpha(void * uniform hairLobe)
{
    AS_HAIRBSDFLOBE(lobe, hairLobe);
    varying float result = lobe->mCosAlpha;
    return extract(result, 0);
}

export uniform float
getLongitudinalRoughness(void * uniform hairLobe)
{
    AS_HAIRBSDFLOBE(lobe, hairLobe);
    varying float result = lobe->mLongitudinalRoughness;
    return extract(result, 0);
}

export uniform float
getLongitudinalVariance(void * uniform hairLobe)
{
    AS_HAIRBSDFLOBE(lobe, hairLobe);
    varying float result = lobe->mLongitudinalVariance;
    return extract(result, 0);
}

export uniform float
evalMTerm(void * uniform hairLobe,
          void * uniform hairState)
{
    AS_HAIRBSDFLOBE(lobe, hairLobe);
    AS_HAIRSTATE(hs, hairState);
    varying float result = HairBsdfLobe_evalMTerm(lobe, *hs);
    return extract(result, 0);
}

export void
evalNTermWithAbsorption(void * uniform hairLobe,
                        void * uniform hairState,
                        uniform Color &result)
{
    AS_HAIRBSDFLOBE(lobe, hairLobe);
    AS_HAIRSTATE(hs, hairState);
    HairBsdfLobeGlintAttrs glintAttrs;
    glintAttrs.mShowGlint = lobe->mShowGlint;
    glintAttrs.mGlintRoughness = lobe->mGlintRoughness;
    glintAttrs.mGlintEccentricity = lobe->mGlintEccentricity;
    glintAttrs.mGlintSaturation = lobe->mGlintSaturation;

    Color vresult = HairBsdfLobe_evalNTermWithAbsorption(lobe, *hs, &glintAttrs);
    result = extractColor(vresult, 0);
}

export uniform float
evalPhiPdf(void * uniform hairLobe,
           void * uniform hairState)
{
    AS_HAIRBSDFLOBE(lobe, hairLobe);
    AS_HAIRSTATE(hs, hairState);
    varying float result = HairBsdfLobe_evalPhiPdf(lobe, *hs);
    return extract(result, 0);
}

export uniform float
evalThetaPdf(void * uniform hairLobe,
             void * uniform hairState)
{
    AS_HAIRBSDFLOBE(lobe, hairLobe);
    AS_HAIRSTATE(hs, hairState);
    varying float result = HairBsdfLobe_evalThetaPdf(lobe, *hs);
    return extract(result, 0);
}

export uniform float
sampleTheta(void * uniform hairLobe,
            const uniform float r1,
            const uniform float thetaO,
            uniform float &thetaI)
{
    AS_HAIRBSDFLOBE(lobe, hairLobe);
    float vthetaI;
    const float pdf = HairBsdfLobe_sampleTheta(lobe, (uniform float) r1, (uniform float) thetaO, vthetaI);
    thetaI = extract(vthetaI, 0);
    return extract(pdf, 0);
}

export uniform float
samplePhi(void * uniform hairLobe,
          const uniform float r2,
          const uniform float phiO,
          uniform float &phiI)
{
    AS_HAIRBSDFLOBE(lobe, hairLobe);
    float vphiI;
    float vpdf = HairBsdfLobe_samplePhi(lobe,
                                         (uniform float) r2,
                                         (uniform float) phiO,
                                         vphiI);
    phiI = extract(vphiI, 0);
    return extract(vpdf, 0);
}

export uniform float
evalPdf(void * uniform hairLobe,
        void * uniform hairState)
{
    AS_HAIRBSDFLOBE(lobe, hairLobe);
    AS_HAIRSTATE(hs, hairState);
    float pdf = HairBsdfLobe_evalPdf(lobe, *hs);
    return extract(pdf, 0);
}

export void
evalLobe(void * uniform lobe,
         void * uniform slice,
         const uniform Vec3f &wi,
         uniform Color &result,
         uniform float &pdf)
{
    const varying BsdfLobe * uniform vlobe = (const varying BsdfLobe * uniform) lobe;
    const varying BsdfSlice * uniform vslice = (const varying BsdfSlice * uniform) slice;
    const Vec3f vwi = Vec3f_ctor(wi.x, wi.y, wi.z);
    float vpdf;
    Color vresult = BsdfLobe_eval(vlobe, *vslice, vwi, &vpdf);
    pdf = extract(vpdf, 0);
    result = extractColor(vresult, 0);
}

export void
sampleLobe(void * uniform lobe,
           void * uniform slice,
           const uniform float r1,
           const uniform float r2,
           uniform Vec3f &wi,
           uniform float &pdf,
           uniform Color &result)
{
    const varying BsdfLobe * uniform vlobe = (const varying BsdfLobe * uniform) lobe;
    const varying BsdfSlice * uniform vslice = (const varying BsdfSlice * uniform) slice;
    varying Vec3f vwi;
    varying float vpdf;
    varying Color vresult = BsdfLobe_sample(vlobe, *vslice, (uniform float) r1, (uniform float) r2, vwi, vpdf);
    wi = extractVec3f(vwi, 0);
    pdf = extract(vpdf, 0);
    result = extractColor(vresult, 0);
}

