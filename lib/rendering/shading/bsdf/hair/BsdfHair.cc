// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

///
/// @file BsdfHairLobe.cc
/// $Id$
///

#include "BsdfHair.h"
#include <moonray/rendering/shading/bsdf/BsdfSlice.h>
#include <moonray/rendering/shading/Util.h>

// Uncomment this to use uniform sampling
//#define PBR_HAIR_USE_UNIFORM_SAMPLING 1

using namespace scene_rdl2::math;

namespace moonray {
namespace shading {

HairBsdfLobe::HairBsdfLobe(Type type,
                           const Vec3f& hairDir,
                           const Vec2f& hairUV,
                           const float mediumIOR,
                           const float ior,
                           ispc::HairFresnelType fresnelType,
                           const float cuticleLayerThickness,
                           const float longShift,
                           const float longRoughness,
                           const float azimRoughness,
                           const Color& hairColor,
                           const Color& hairSigmaA,
                           const Color& tint,
                           const float hairRotation,
                           const Vec3f& hairNormal):
                 BsdfLobe(type,
                          DifferentialFlags(IGNORES_INCOMING_DIFFERENTIALS),
                          true,
                          PROPERTY_ROUGHNESS | PROPERTY_COLOR |
                          PROPERTY_NORMAL | PROPERTY_PBR_VALIDITY),
                 mHairDir(hairDir),
                 mHairUV(hairUV),
                 mMediumIOR(mediumIOR),
                 mIOR(ior),
                 mFresnelType(fresnelType),
                 mLongitudinalRoughness(longRoughness),
                 mAzimuthalRoughness(azimRoughness),
                 mHairColor(hairColor),
                 mSigmaA(hairSigmaA),
                 mLongitudinalShift(longShift),
                 mTint(tint),
                 mHairRotation(hairRotation),
                 mHairNormal(hairNormal)
{
    setIsHair(true);
    scene_rdl2::math::sincos(longShift, &mSinAlpha, &mCosAlpha);

    // Derive the longitudinal & azimuthal variance using Disney's approximation
    mLongitudinalVariance  = HairUtil::longitudinalVar(mLongitudinalRoughness);
    mAzimuthalVariance     = HairUtil::azimuthalVar(mAzimuthalRoughness);

    // For Moonray, Texture 't' varies across the width of hair curve
    // and goes from [0,1].   We remap this to [-1,1].
    // Note - this is not true of all curve types
    // Please take a look at MOONRAY-3116 and make sure that is fixed.
    mH = mHairUV[1] * 2.0f - 1.0f;

    // Cuticle-Layer has been parameterized to be between 0 and 1
    // Layers vary in between [0.5, 1.5]
    // See Table 2
    /// From "Physically-Accurate Fur Reflectance: Modeling, Measurement and Rendering"
    mCuticleLayerThickness = scene_rdl2::math::lerp(0.5f, 1.5f,
                                                    scene_rdl2::math::clamp(
                                                        cuticleLayerThickness));

    // Derive a directional differential scale that varies according to the
    // smallest of longitudinal and azimuthal width
    float minWidth = scene_rdl2::math::min(shading::roughness2Width(longRoughness),
                               shading::roughness2Width(azimRoughness));

    // Derive a directional differential scale that varies according to width
    mdDFactor = sdDFactorMin + minWidth * sdDFactorSlope;
}

//----------------------------------------------------------------------------
Color
HairBsdfLobe::eval(const BsdfSlice &slice,
                   const Vec3f &wi,
                   float *pdf) const
{
    if (!HairUtil::clampTest(mHairDir,
                             wi)) {
        if (pdf) *pdf = 0.0f;
        return sBlack;
    }

    // calculate all the relevant hair bsdf parameters
    HairState hairState(slice.getWo(),
                        wi,
                        mHairDir,
                        mH,
                        mIOR,
                        mSigmaA,
                        mHairRotation * mHairUV.x, // use hair s coord to vary rotation from base to tip
                        mHairNormal);

    Color bsdf = getScale() * mTint * evalBsdf(hairState, slice.getIncludeCosineTerm());

    if (pdf) *pdf = evalPdf(hairState);

    return bsdf;
}


// BSDF = Fresnel * Absorption * M * N * CosineTerm
Color
HairBsdfLobe::evalBsdf(const HairState& hairState,
                       bool includeCosineTerm) const
{
    // The Cosine terms cancel out since the Longitudinal Function includes
    // a oneOverCosThetaI
    return  fresnel(hairState, hairState.cosThetaD()) *
            evalMTerm(hairState) *
            evalNTermWithAbsorption(hairState);
}

//----------------------------------------------------------------------------
// Default Longitudinal Function
float
HairBsdfLobe::evalMTerm(const HairState& hairState) const
{
    float sinThetaICone = hairState.sinThetaI() * mCosAlpha + hairState.cosThetaI() * mSinAlpha;
    float cosThetaICone = hairState.cosThetaI() * mCosAlpha - hairState.sinThetaI() * mSinAlpha;

    // Handle out-of-range thetaI values because of the shift
    cosThetaICone = scene_rdl2::math::abs(cosThetaICone);

    // Perfect Longitudinal Function From [1]
    return HairUtil::deonLongitudinalM(mLongitudinalVariance,
                                       sinThetaICone,
                                       cosThetaICone,
                                       hairState.sinThetaO(),
                                       hairState.cosThetaO());
}

// Default Azimuthal Function
float
HairBsdfLobe::evalNTerm(const HairState& hairState) const
{
    // Normalized Azimuthal Lobe - 1/4 * cos(phiD/2)
    const float cosPhiDOver2 = max(0.0f, 0.25f * hairState.cosPhiDOverTwo());
    return cosPhiDOver2;
}

Color
HairBsdfLobe::evalNTermWithAbsorption(const HairState& hairState) const
{
    return evalNTerm(hairState) * absorption(hairState);
}


//-----------------------------------------------------------------------------------//
Color
HairBsdfLobe::sample(const BsdfSlice &slice,
                     float r1, float r2,
                     Vec3f &wi,
                     float &pdf) const
{
    // calculate all the relevant hair bsdf parameters based on omegaO
    HairState hairState(slice.getWo(),
                        mHairDir,
                        mH,
                        mIOR,
                        mSigmaA,
                        mHairRotation * mHairUV.x,  // use hair s coord to vary rotation from base to tip
                        mHairNormal);

#ifdef PBR_HAIR_USE_UNIFORM_SAMPLING
    wi = hairState.localToGlobal(
            sampleSphereUniform(r1, r2));
    return eval(slice, wi, &pdf);

#else

    // Sample Theta and Phi
    float thetaI;
    const float thetaPdf = sampleTheta(r1,
                                       hairState.thetaO(),
                                       thetaI);

    float phiI = 0.0f;
    const float phiPdf = samplePhi(r2,
                                   hairState.phiO(),
                                   phiI); // phiI is in (-pi, pi]
    MNRY_ASSERT(phiI >= -sPi  &&  phiI <= sPi);


    float sinPhi, cosPhi;
    sincos(phiI, &sinPhi, &cosPhi);

    float sinTheta, cosTheta;
    sincos(thetaI, &sinTheta, &cosTheta);

    if (!HairUtil::clampTest(cosTheta)) {
        pdf = 0.0f;
        return sBlack;
    }

    pdf = max(0.0f, phiPdf * thetaPdf);

    // Compute the light direction vector for shading.
    // TODO: use ReferenceFrame local2global
    const float uWgt = sinTheta;
    const float vWgt = cosTheta * cosPhi;
    const float wWgt = cosTheta * sinPhi;
    wi = normalize(hairState.localToGlobal(uWgt, vWgt, wWgt));

    // Update HairState based on omegaI
    hairState.updateAngles(wi, phiI, thetaI);

    // Return the shaded color.
    return getScale() * mTint * evalBsdf(hairState, slice.getIncludeCosineTerm());
#endif
}

float
HairBsdfLobe::evalPdf(const HairState& hairState) const
{
#ifdef PBR_HAIR_USE_UNIFORM_SAMPLING
    return  1.0f / sFourPi;
#else

    // This much is the same for all lobes --
    // we need to have a parent Bsdf object, and we generally
    // return zero probability when the light direction is pretty
    // just about parallel to the hair direction.
    // clamptest done at beginning of eval, done before pdf() is called

    const float phiPdf = evalPhiPdf(hairState);
    const float thetaPdf = evalThetaPdf(hairState);

    const float pdf = max(0.0f, phiPdf * thetaPdf);
    return pdf;
#endif
}

float
HairBsdfLobe::sampleTheta(float  r,
                          float  thetaO,
                          float& thetaI) const
{
    // Create Two Unique Random Numbers using a Morton Curve
    // https://fgiesen.wordpress.com/2009/12/13/decoding-morton-codes/
    // PBRT Section 4.3.3
    Vec2f eps = HairUtil::demuxFloat(r);

    float sinThetaO, cosThetaO;
    sincos(thetaO,
           &sinThetaO,
           &cosThetaO);

     eps[0] = max(eps[0], 1e-5f);

     // Eugene's Derivation - [3] Section 3.2
     // float cosTheta = mLongitudinalVariance *
     //     log( exp(rcp(mLongitudinalVariance)) - 2 * eps[0] * sinh(rcp(mLongitudinalVariance)) );

     // PBRT Derivation (More Stable at low variance values)
     float cosTheta =
         1.0f + mLongitudinalVariance * scene_rdl2::math::log(
             eps[0] + (1.0f - eps[0]) * scene_rdl2::math::exp(-2.0f / mLongitudinalVariance));

     float sinTheta = HairUtil::safeSqrt(1.0f - sqr(cosTheta));
     float cosPhi = scene_rdl2::math::cos(sTwoPi * eps[1]);
     float sinThetaI = -cosTheta * sinThetaO + sinTheta * cosPhi * cosThetaO;
     float cosThetaI = HairUtil::safeSqrt(1.0f - sqr(sinThetaI));

     // Update sampled $\sin \thetai$ and $\cos \thetai$ to account for scales
     // Note - Shift by -negative Alpha, because in eval() you shift by positive
     float sinThetaICone = sinThetaI * mCosAlpha - cosThetaI * mSinAlpha;
     float cosThetaICone = cosThetaI * mCosAlpha + sinThetaI * mSinAlpha;
     // Handle out-of-range thetaI values because of the shift
     cosThetaICone = scene_rdl2::math::abs(cosThetaICone);

     // This is our sample direction. We clamp sinThetaI to [-1,1] to remove
     // small floating point math errors, because asin() is extremely sensitive
     // and will throw NaNs if the sine value is even slightly outside this range.
     sinThetaI = clamp(sinThetaICone, -1.0f, 1.0f);
     cosThetaI = cosThetaICone;

     thetaI = scene_rdl2::math::asin(sinThetaI);

     // Shift by +alpha to calculate the PDF, same as will happen in eval()
     sinThetaICone = sinThetaI * mCosAlpha + cosThetaI * mSinAlpha;
     cosThetaICone = cosThetaI * mCosAlpha - sinThetaI * mSinAlpha;

     // Handle out-of-range thetaI values because of the shift
     cosThetaICone = scene_rdl2::math::abs(cosThetaICone);

     const float thetaPDF = HairUtil::deonLongitudinalM(mLongitudinalVariance,
                                                        sinThetaICone, cosThetaICone,
                                                        sinThetaO, cosThetaO);
     return thetaPDF;
}

// Default Azimuthal Lobe Sampling
float
HairBsdfLobe::samplePhi(float  randomV,
                        float  phiO,
                        float& phiI) const
{
    float sinPhi_2 = 2.0f * randomV - 1.0f;
    sinPhi_2 = clamp(sinPhi_2, -1.0f, 1.0f);
    const float phi = 2.0f * scene_rdl2::math::asin(sinPhi_2);

    phiI = rangeAngle(phiO + phi);

    float phiPdf = scene_rdl2::math::cos(phi * 0.5f) * 0.25f;
    return phiPdf;
}

float
HairBsdfLobe::evalThetaPdf(const HairState& hairState) const
{
    // Perfect Sampling = PDF is the same as M Term
    return evalMTerm(hairState);
}

Color
HairBsdfLobe::evalHairFresnel(const HairState& hairState,
                              float cosTheta) const
{
    Color fresnel = scene_rdl2::math::sWhite;
    switch(mFresnelType) {
    case ispc::HAIR_FRESNEL_SIMPLE_LONGITUDINAL:
    default:
        // Simple longitudinal cosine term
        fresnel = computeFresnel(cosTheta);
        break;
    case ispc::HAIR_FRESNEL_DIELECTRIC_CYLINDER:
        // From "An Energy-Conserving Hair Reflectance Model"
        // This curve is very similar to the Marschner Fresnel equations
        // and slightly cheaper to evaluate without etaP & etaPP in Marschner:
        // https://www.desmos.com/calculator/fmxsatvxi3
        fresnel = computeFresnel(cosTheta*hairState.cosGammaO());
        break;
    case ispc::HAIR_FRESNEL_LAYERED_CUTICLES:
        // From "Physically-Accurate Fur Reflectance: Modeling, Measurement and Rendering"
        // https://www.desmos.com/calculator/fmxsatvxi3
        fresnel = LayeredDielectricFresnel::evalFresnel(mMediumIOR,
                                                        hairState.eta(),
                                                        hairState.etaP(),
                                                        hairState.etaPP(),
                                                        hairState.cosGammaO(),
                                                        mCuticleLayerThickness);
        break;
    }
    return fresnel;
}


float
HairBsdfLobe::evalPhiPdf(const HairState& hairState) const
{
    return evalNTerm(hairState);
}

//----------------------------------------------------------------------------
void
HairBsdfLobe::differentials(const Vec3f &wo,
                            const Vec3f &wi,
                            float r1, float r2,
                            const Vec3f &dNdx,
                            const Vec3f &dNdy,
                            Vec3f &dDdx, Vec3f &dDdy) const
{
    Vec3f H;
    if (!computeNormalizedHalfVector(wo, wi, H)) {
        H = mHairDir;
    }
    computeReflectionDirectionDifferentialNoCurvature(wo, wi, H, dDdx, dDdy);
    dDdx = normalize(dDdx) * mdDFactor;
    dDdy = normalize(dDdy) * mdDFactor;
}

//----------------------------------------------------------------------------
bool
HairBsdfLobe::getProperty(Property property,
                          float *dest) const
{
    bool handled = true;

    switch (property)
    {
    case PROPERTY_ROUGHNESS:
        {
            *dest       = mLongitudinalRoughness;
            *(dest + 1) = mAzimuthalRoughness;
        }
        break;
    case PROPERTY_COLOR:
        {
            const Color c = getScale() * mHairColor;
            *dest       = c.r;
            *(dest + 1) = c.g;
            *(dest + 2) = c.b;
        }
        break;
    case PROPERTY_NORMAL:
        {
            *dest       = mHairDir.x;
            *(dest + 1) = mHairDir.y;
            *(dest + 2) = mHairDir.z;
        }
        break;
    case PROPERTY_PBR_VALIDITY:
        {
            const Fresnel* fresnel = getFresnel();
            // Get the property from the fresnel object
            Color res = scene_rdl2::math::sBlack;
            if (fresnel) {
                res = fresnel->computePbrValidity();
            }
            *dest       = res.r;
            *(dest + 1) = res.g;
            *(dest + 2) = res.b;
        }
        break;
    default:
        handled = BsdfLobe::getProperty(property, dest);
        break;
    }
    return handled;
}

//----------------------------------------------------------------------------
} // namespace shading
} // namespace moonray

