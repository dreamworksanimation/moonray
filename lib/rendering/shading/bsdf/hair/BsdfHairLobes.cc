// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

///
/// @file BsdfHairLobes.cc
/// $Id$
///


#include <moonray/rendering/shading/ColorCorrect.h>
#include "BsdfHairLobes.h"
#include <moonray/rendering/shading/bsdf/BsdfSlice.h>
#include <moonray/rendering/shading/Util.h>
#include <moonray/rendering/shading/ColorCorrect.h>

namespace moonray {
namespace shading {

using namespace scene_rdl2::math;

Color
HairRLobe::fresnel(const HairState& hairState,
                   const float cosTheta) const
{
    // [1] Section 6
    return evalHairFresnel(hairState,
                           cosTheta);
}

Color
HairTRTLobe::fresnel(const HairState& hairState,
                     const float cosTheta) const
{
    Color fresnel = scene_rdl2::math::sWhite;
    if (getFresnel() != nullptr) {
        const Color F = evalHairFresnel(hairState,
                                        cosTheta);
        Color oneMinusF;
        if (isLayeredFresnel()) {
            // Note for LayerFresnel, we call a static function to evaluate
            // so we substract here instead
            oneMinusF = scene_rdl2::math::sWhite - F;
        } else {
            // We assume that this lobe has been set up using oneMinusFresnel
            oneMinusF = F;
        }
        fresnel = oneMinusF * oneMinusF * (sWhite - oneMinusF);
    }
    return fresnel;
}

Color
HairTRTLobe::absorption(const HairState& hairState) const
{
    Color T = hairState.absorptionTerm();
    return T * T;
}

finline Color
HairTRTLobe::glintAbsorption(const HairState& hairState,
                             const float etaStar,
                             const float etaP) const
{
    if (!mShowGlint) {
        return scene_rdl2::math::sBlack;
    } else {
        Color glintAbsorption = HairState::calculateAbsorptionTerm(hairState.sinGammaO(),
                                                                   etaStar,
                                                                   hairState.sigmaA(),
                                                                   etaP,
                                                                   hairState.cosThetaO(),
                                                                   hairState.sinThetaO());
        shading::applySaturationWithoutPreservingLuminance(mGlintSaturation, glintAbsorption);
        return glintAbsorption * glintAbsorption;
    }
}

finline float
HairTRTLobe::evalNTermTRT(const HairState& hairState) const
{
    // Normalized Azimuthal Lobe - 1/4 * cos(phiD/2)
    const float cosPhiDOver2 = max(0.0f, 0.25f * hairState.cosPhiDOverTwo());
    return cosPhiDOver2;
}

finline float
HairTRTLobe::evalNTermGlint(const HairState& hairState,
                            float& etaStar,
                            float& etaP,
                            float& trtWeight) const
{
    if (!mShowGlint) {
        etaStar = 0.f;
        etaP = 0.f;
        trtWeight = 1.f;
        return 0.f;
    } else {
        // Glint calculation taken from "Light Scattering from Human Hair Fibers" Marschner, et al.
        // Eccentricity is approximated by adjusting the index of refraction (Marschner 5.2.3)
        const float eta = hairState.eta();
        const float etaStar1 = 2.f * (eta - 1.f) * mGlintEccentricity * mGlintEccentricity - eta + 2.f;
        const float etaStar2 = 2.f * (eta - 1.f) * scene_rdl2::math::rcp(mGlintEccentricity * mGlintEccentricity) - eta + 2.f;
        etaStar = 0.5f * ((etaStar1 + etaStar2) + scene_rdl2::math::cos(2.f * hairState.phiH()) * (etaStar1 - etaStar2));
        etaP = HairUtil::safeSqrt(etaStar*etaStar - hairState.sinThetaD()*hairState.sinThetaD()) / hairState.cosThetaD();

        // hc is the offset at which the glint caustic occurs (Marschner 4.2 equation 4)
        const float hc = HairUtil::safeSqrt((4.f - etaP*etaP)/3.f);
        const float gammai = scene_rdl2::math::asin(hc);
        const float gammat = scene_rdl2::math::asin(hc/etaP);

        // phiC is the angle offset from phiD at which the glint occurs
        // phiC is undefined for incidence angles past the caustic merge, when etaP equals 2
        // Unlike Marschner, we do not bother fading out the caustic glint
        // (Marschner 5.2.2)
        float phiC;
        if (etaP < 2) {
            phiC = rangeAngle(2.f * 2.f * gammat - 2.f * gammai + 2.f * scene_rdl2::math::sPi);
        } else {
            phiC = 0;
        }

        // Unlike Marschner, we use trimmed logisitic functions instead of gaussians
        // to model the glint distributions

        // g0 is the normalization term used to calculate the weights below
        const float g0 = HairUtil::trimmedLogisticFunction(0.f,
                                                           mGlintRoughness,
                                                          -sPi,
                                                           sPi);
        const float g1 = HairUtil::trimmedLogisticFunction(rangeAngle(hairState.phiD() - phiC),
                                                           mGlintRoughness,
                                                          -sPi,
                                                           sPi);
        const float g2 = HairUtil::trimmedLogisticFunction(rangeAngle(hairState.phiD() + phiC),
                                                           mGlintRoughness,
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

Color
HairTRTLobe::evalNTermWithAbsorption(const HairState& hairState) const
{
    // evalNTermGlint and evalNTermTRT were separated so that their respective
    // absorptions could be applied to each individually.
    float etaStar, etaP, trtWeight;
    const float glintN = evalNTermGlint(hairState, etaStar, etaP, trtWeight);
    return trtWeight * evalNTermTRT(hairState) * absorption(hairState)
            + glintN * glintAbsorption(hairState, etaStar, etaP);
}

Color
HairTTLobe::fresnel(const HairState& hairState,
                    const float cosTheta) const
{
    Color fresnel = scene_rdl2::math::sWhite;
    if (getFresnel() != nullptr) {
        const Color F = evalHairFresnel(hairState,
                                        cosTheta);
        Color oneMinusF;
        if (isLayeredFresnel()) {
            // Note for LayerFresnel, we call a static function to evaluate fresnel
            // so we substract here instead
            oneMinusF = scene_rdl2::math::sWhite - F;
        } else {
            // We assume that this lobe has been set up using oneMinusFresnel
            oneMinusF = F;
        }
        fresnel *= oneMinusF * oneMinusF;
    }
    return fresnel;
}

Color
HairTTLobe::absorption(const HairState& hairState) const
{
    Color c = hairState.absorptionTerm();

    // MOONSHINE-1238
    // Shader trick to further saturate/desaturate Transmission
    applySaturation(c, hairState);

    return c;
}

void
HairTTLobe::applySaturation(scene_rdl2::math::Color& c,
                            const HairState& hairState) const
{
    if (scene_rdl2::math::isOne(mSaturation))
        return;

    // phiD goes from [-pi, pi] so normalizing it to [0, 1]
    const float x = scene_rdl2::math::abs(hairState.phiD()*sOneOverPi);
    // isolate a narrow band of direct transmission
    // step [0.8f, 1.0f]
    // This range *could be exposed to the artists but we are  trying
    // to limit confusing user-controls here.
    float t = scene_rdl2::math::clamp((x - 0.8f) / (0.2f));

    if (!scene_rdl2::math::isZero(t)) {
        // smoothstep
        t = t*t*(3.0f - 2*t);

        float saturation = lerp(1.0f, mSaturation, t);
        shading::applySaturation(saturation, c);
        c = shading::clampColor(c);
    }
}

float
HairTTLobe::evalNTerm(const HairState& hairState) const
{
    const float Np = HairUtil::trimmedLogisticFunction(rangeAngle(hairState.phiD() - sPi),
                                                       azimuthalVariance(),
                                                      -sPi,
                                                       sPi);

    return Np;
}

Color
HairTTLobe::evalNTermWithAbsorption(const HairState& hairState) const
{
    return evalNTerm(hairState) * absorption(hairState);
}

float
HairTTLobe::evalPhiPdf(const HairState& hairState) const
{
    return evalNTerm(hairState);
}

float
HairTTLobe::samplePhi(float  randomV,
                      float  phiO,
                      float& phiI) const
{
    // Importance sample phi
    const float phiD = HairUtil::sampleTrimmedLogistic(randomV,
                                                       azimuthalVariance(),
                                                      -sPi,
                                                       sPi);

    phiI = rangeAngle(phiO + phiD + sPi);

    const float phiPdf = HairUtil::trimmedLogisticFunction(phiD,
                                                           azimuthalVariance(),
                                                          -sPi,
                                                           sPi);
    return phiPdf;
}

// Special TT Lobe Returns AziRoughness in the Green Channel
bool
HairTTLobe::getProperty(Property property,
                       float *dest) const
{
    bool handled = true;

    switch (property)
    {
    case PROPERTY_ROUGHNESS:
        {
            *dest       = getLongitudinalRoughness();
            *(dest + 1) = getAzimuthalRoughness();
        }
        break;
    default:
        handled = HairBsdfLobe::getProperty(property, dest);
        break;
    }
    return handled;
}

Color
HairTRRTLobe::fresnel(const HairState& hairState,
                      const float cosTheta) const
{
    if (getFresnel() != nullptr) {
        const Color oneMinusF = evalHairFresnel(hairState,
                                                cosTheta);
        const Color f = sWhite - oneMinusF;
        return compensationFactor(f,
                                  oneMinusF,
                                  hairState.absorptionTerm());
    }
    else
        return scene_rdl2::math::sWhite;
}

// TRRT Compensation Factor
// [2], Section 3.4
Color
HairTRRTLobe::compensationFactor(const Color& fresnel,
                                 const Color& oneMinusF,
                                 const Color& T) const
{
    Color result = oneMinusF*oneMinusF*fresnel*fresnel;
    const Color oneMinusFT = scene_rdl2::math::clamp(sWhite - fresnel * T,
                                         Color(scene_rdl2::math::sEpsilon),
                                         scene_rdl2::math::sWhite);
    result *= T*T*T / oneMinusFT;

    return result;
}

float
HairTRRTLobe::evalNTerm(const HairState& hairState) const
{
    // Normalized Hemisphere
    return sOneOverTwoPi;
}

Color
HairTRRTLobe::evalNTermWithAbsorption(const HairState& hairState) const
{
    return evalNTerm(hairState) * absorption(hairState);
}


float
HairTRRTLobe::evalPhiPdf(const HairState& hairState) const
{
    // Uniform sampling TRRT
    return evalNTerm(hairState);
}

float
HairTRRTLobe::samplePhi(float  randomV,
                        float  phiO,
                        float& phiI) const
{
    // dPhi is in [-pi,pi]
    phiI = (sPi * (2.0f * randomV - 1.0f));

    // Uniform sampling TRRT
    return sOneOverTwoPi;
}

//----------------------------------------------------------------------------
} // namespace shading
} // namespace moonray

