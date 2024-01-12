// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

///
/// @file BsdfHairOneSampler.cc
/// $Id$
///


#include "BsdfHairOneSampler.h"
#include <moonray/rendering/shading/bsdf/BsdfSlice.h>
#include <moonray/rendering/shading/Util.h>

#define PBR_HAIR_USE_UNIFORM_SAMPLING 0

namespace moonray {
namespace shading {

using namespace scene_rdl2::math;

///----------------------------------------------------------------------------
Color
HairOneSampleLobe::eval(const BsdfSlice &slice,
                        const Vec3f &wi,
                        float *pdf) const
{
    // calculate all the relevant hair bsdf parameters
    HairState hairState(slice.getWo(),
                        wi,
                        mHairDir,
                        mH,
                        mIOR,
                        mSigmaA,
                        mHairRotation * mHairUV.x,  // use hair s coord to vary rotation from base to tip
                        mHairNormal);

    if (pdf) *pdf = evalPdf(hairState);

    return getScale() * evalBsdf(hairState);
}

Color
HairOneSampleLobe::evalBsdf(const HairState& hairState,
                            bool includeCosineTerm) const
{
    Color bsdf = scene_rdl2::math::sBlack;

    const Color T = hairState.absorptionTerm();

    const Color fresnel = evalHairFresnel(hairState,
                                          hairState.cosThetaD());

    const Color oneMinusFresnel = scene_rdl2::math::sWhite - fresnel;

    if (mShowR) {
        const Color rFresnel = fresnel;
        bsdf += rFresnel *
                rLobe.getTint() *
                rLobe.evalMTerm(hairState) *
                rLobe.evalNTermWithAbsorption(hairState);
    }

    if (mShowTT) {
        const Color ttFresnel = oneMinusFresnel * oneMinusFresnel;
        bsdf += ttFresnel *
                ttLobe.getTint() *
                ttLobe.evalMTerm(hairState) *
                ttLobe.evalNTermWithAbsorption(hairState);
    }

    if (mShowTRT) {
        const Color trtFresnel = fresnel * oneMinusFresnel * oneMinusFresnel;
        bsdf += trtFresnel *
                trtLobe.getTint() *
                trtLobe.evalMTerm(hairState) *
                trtLobe.evalNTermWithAbsorption(hairState);
    }

    if (mShowTRRT) {
        const Color trrt = trrtLobe.compensationFactor(fresnel, oneMinusFresnel, T);
        bsdf += trrt *
                trrtLobe.getTint() *
                trrtLobe.evalMTerm(hairState) *
                trrtLobe.evalNTermWithAbsorption(hairState);
    }

    return bsdf;
}

//-----------------evaluate phi pdf ------------------------------------
float
HairOneSampleLobe::evalPdf(const HairState& hairState) const
{
#if PBR_HAIR_USE_UNIFORM_SAMPLING
    return  1.0f / sFourPi;
#else
    // weights and cdfs for R, TRT, TT lobes respectivel
    float weights[3];
    float cdf[3];
    calculateSamplingWeightsAndCDF(hairState,
                                   weights,
                                   cdf);
    if (scene_rdl2::math::isZero(cdf[2])) {
        // uniform sampling
        // oneOverFourPi
        return (0.5f * sOneOverTwoPi);
    }

    return evalPdf(hairState, weights);
#endif
}

float
HairOneSampleLobe::evalPdf(const HairState& hairState,
                           const float (&weights)[3]) const
{
#if PBR_HAIR_USE_UNIFORM_SAMPLING
    return  1.0f / sFourPi;
#else
    float rProb = 0.0f, ttProb = 0.0f, trtProb = 0.0f;
    if (mShowR)
        rProb   = weights[0] * rLobe.evalThetaPdf(hairState)   * rLobe.evalPhiPdf(hairState);
    if (mShowTRT)
        trtProb = weights[1] * trtLobe.evalThetaPdf(hairState) * trtLobe.evalPhiPdf(hairState);
    if (mShowTT)
        ttProb  = weights[2] * ttLobe.evalThetaPdf(hairState)  * ttLobe.evalPhiPdf(hairState);

    return max(0.0f, (rProb + trtProb + ttProb));
#endif
}

//-----------------------------------------------------------------------------------//
Color
HairOneSampleLobe::sample(const BsdfSlice &slice,
                          float r1, float r2,
                          Vec3f &wi,
                          float &pdf) const
{
    // If none of the lobes are ON
    if (!mShowR && !mShowTT && !mShowTRT && !mShowTRRT) {
        pdf = 0.0f;
        return sBlack;
    }

    // calculate all the relevant hair bsdf parameters based on omegaO
    HairState hairState(slice.getWo(),
                        mHairDir,
                        mH,
                        mIOR,
                        mSigmaA,
                        mHairRotation * mHairUV.x,  // use hair s coord to vary rotation from base to tip
                        mHairNormal);
#if PBR_HAIR_USE_UNIFORM_SAMPLING
    wi = hairState.localToGlobal(sampleSphereUniform(r1, r2));
    return eval(slice, wi, &pdf);
#else
    float phiI, thetaI;

    const HairBsdfLobe* sampleLobe;

    // weights and cdfs for R, TRT, TT lobes respectivel
    float weights[3];
    float cdf[3];
    calculateSamplingWeightsAndCDF(hairState,
                                   weights,
                                   cdf);

    if (scene_rdl2::math::isZero(cdf[2])) {
        // Uniform Sample TRRT
        wi = sampleSphereUniform(r1, r2);
        wi = hairState.localToGlobal(wi);
        return eval(slice, wi, &pdf);
    }

    // Pick a Lobe to Sample From
    if (r1 < cdf[0]) {
        r1 = r1 / weights[0];
        sampleLobe = &rLobe;
    } else if (r1 < cdf[1]) {
        r1 = (r1 - cdf[0]) / weights[1];
        sampleLobe = &trtLobe;
    } else {
        // Clamp to ensure [0,1)
        r1 = clamp((r1 - cdf[1]) / weights[2], 0.0f, 0.99f);
        sampleLobe = &ttLobe;
    }

    sampleLobe->sampleTheta(r1,
                            hairState.thetaO(),
                            thetaI);
    sampleLobe->samplePhi(r2,
                          hairState.phiO(),
                          phiI); // phiI is in (-pi, pi]
    MNRY_ASSERT(phiI >= -sPi  &&  phiI <= sPi);

    float sinPhi, cosPhi;
    sincos(phiI, &sinPhi, &cosPhi);
    float sinTheta, cosTheta;
    sincos(thetaI, &sinTheta, &cosTheta);

    // Compute the light direction vector for shading.
    const float uWgt = sinTheta;
    const float vWgt = cosTheta * cosPhi;
    const float wWgt = cosTheta * sinPhi;
    wi = hairState.localToGlobal(uWgt, vWgt, wWgt);

    hairState.updateAngles(wi,
                           phiI,
                           thetaI);

    pdf = evalPdf(hairState, weights);

    return getScale() * evalBsdf(hairState);
#endif
}

void
HairOneSampleLobe::calculateSamplingWeightsAndCDF(const HairState& hairState,
                                                  float (&weights)[3],
                                                  float (&cdf)[3]) const
{
    // We use cosThetaO as an approximation to calculate Fresnel here since
    // in the sampling stage, we only have access to omega_o. Even though the
    // fresnel in eval() is computed correctly using cosThetaD (which is equivivalent to cosThetaH in
    // hair frame), cosThetaO serves as a decent approximation to select a lobe to sample.
    const Color fresnel = evalHairFresnel(hairState,
                                          hairState.cosThetaO());

    const Color oneMinusFresnel = scene_rdl2::math::sWhite - fresnel;

    float sum = 0.0f;
    weights[0] = weights[1] = weights[2] = 0.0f;
    if (mShowR) {
        // clamping the R-weight here reduces fireflies for lighter hair
        // as r term doesn't get unfairly voted high for the cosThetaO approx above.
        weights[0] = max(0.2f, scene_rdl2::math::luminance(fresnel * rLobe.absorption(hairState)));
        sum += weights[0];
    }
    if (mShowTRT) {
        weights[1] = scene_rdl2::math::luminance(fresnel * oneMinusFresnel * oneMinusFresnel * trtLobe.absorption(hairState));
        sum += weights[1];
    }
    if (mShowTT) {
        weights[2] = scene_rdl2::math::luminance(oneMinusFresnel* oneMinusFresnel * ttLobe.absorption(hairState));
        sum += weights[2];
    }

    // Normalize the Weights
    if (!scene_rdl2::math::isZero(sum)) {
        for (int i = 0; i < 3; ++i) weights[i] /= sum;
    }
    cdf[0] = weights[0];
    cdf[1] = cdf[0] + weights[1];
    cdf[2] = cdf[1] + weights[2];
}

//----------------------------------------------------------------------------
bool
HairOneSampleLobe::getProperty(Property property,
                               float *dest) const
{
    bool handled = true;

    switch (property)
    {
    case PROPERTY_ROUGHNESS:
        {
            *dest       = rLobe.mLongitudinalRoughness;
            *(dest + 1) = ttLobe.mAzimuthalRoughness;
        }
        break;
    default:
        handled = HairBsdfLobe::getProperty(property, dest);
        break;
    }
    return handled;
}

//----------------------------------------------------------------------------
} // namespace shading
} // namespace moonray

