// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

///
/// @file FabricBsdfLobe.cc
/// $Id$
///

#include "BsdfFabric.h"


namespace moonray {
namespace shading {

using namespace scene_rdl2::math;

///
/// @class KajiyaKayFabricBsdfLobe BsdfFabric.h <pbr/fabric/BsdfFabric.h>
/// @brief The KajiyaKay Cosine Specular Lobe  - cos(thetaD)^N
/// thetaD = thetaO - thetaIPrime, theta wrt hair tangent
///

// Based on the paper - "Rendering fur with three dimensional textures" (Kajiya et al)
// https://www.cs.drexel.edu/~david/Classes/CS586/Papers/p271-kajiya.pdf
// Note - we include a custom normalization factor that we've calculated for this model.

//----------------------------------------------------------------------------
// Constructor / Destructor
KajiyaKayFabricBsdfLobe::KajiyaKayFabricBsdfLobe(const Vec3f &N,
                               const Vec3f &T,
                               const Vec3f &threadDirection,
                               const float threadElevation,
                               const float roughness) :
                                       FabricBsdfLobe(N, T,
                                                      threadDirection, threadElevation,
                                                      roughness)
{
    // Convert the Roughness->Exponent
    // Square the roughness for a visually linear effect
    const float oneOverR2 = 1.0 / (mRoughness*mRoughness);

    // Using the Conversion from:
    // Walter et al, EGSR 2007, "Microfacet Models for Refraction through Rough Surfaces"
    mSpecularExponent = 2.0*oneOverR2*oneOverR2;

    calculateNormalizationFactor();
}

// Initialize Specular Exponent and NormalizationFactor
void
KajiyaKayFabricBsdfLobe::calculateNormalizationFactor()
{
    // Normalization factor = gamma((n+3)/2) / gamma((n+2)/2) / pi^1.5
    mNormalizationFactor = scene_rdl2::math::exp(lgammaf(mSpecularExponent*0.5f+1.5f) - lgammaf(mSpecularExponent*0.5+1.0f));
    // 1/powf(pi,1.5) = 0.1795871;
    mNormalizationFactor *= 0.1795871f;
}

// KajiyaKay Specular Lobe - cos(thetaD)^N, thetaD = abs(thetaO - thetaIPrime)
// Regular cosine lobe specular, looks similar to DWAFabric but softer peaks
Color
KajiyaKayFabricBsdfLobe::eval(const BsdfSlice &slice,
                              const Vec3f &wi,
                              float *pdf) const
{
    // Prepare for early exit
    if (pdf != nullptr) {
        *pdf = 0.0f;
    }
    const float cosThetaWi = dot(mFrame.getN(), wi);
    if (cosThetaWi <= sEpsilon) return sBlack;
    const float cosThetaWo = dot(mFrame.getN(), slice.getWo());
    if (cosThetaWo <= sEpsilon) return sBlack;

    // BRDF = cos^N(thetaD)
    // BRDF = cos^N(thetaIPrime - thetaO)
    // BRDF = (cos(thetaIPrime)*cos(thetaO) + sin(thetaIPrime)*sin(thetaO))^N
    const float cosThetaO = clamp(dot(mThreadDirection, slice.getWo()), -0.99f, 0.99f);
    const float sinThetaO = scene_rdl2::math::sqrt(1 - cosThetaO*cosThetaO);

    // IPrime is I reflected across the normal plane
    const float cosThetaIPrime = (-1) * clamp(dot(mThreadDirection, wi), -0.99f, 0.99f);
    const float sinThetaIPrime = scene_rdl2::math::sqrt(1 - cosThetaIPrime*cosThetaIPrime);

    // cos(thetaD) = cos(thetaIPrime - thetaO);
    const float cosThetaD = cosThetaO*cosThetaIPrime + sinThetaO*sinThetaIPrime;

    if (cosThetaD < sEpsilon || cosThetaD > 1.f) {
        return sBlack;
    }

    const float cosThetaDPow = powf(cosThetaD, mSpecularExponent);
    float fabricBRDF = mNormalizationFactor * cosThetaDPow;

    // PDF
    if (pdf != nullptr && sinThetaIPrime > sEpsilon) {
        // We sample theta_d, so to convert the PDF to sample theta_i we divide by
        // the Jacobian (dOmegaI/dOmegaD) = sin(thetaI) / sin(thetaD)
        // PDF for sampling theta_d is (n+1)/2pi * powf(costhetaD, n)
        *pdf = (mSpecularExponent+1)*sOneOverTwoPi*cosThetaDPow;

        // Converting the PDF from sampling theta_d to sampling theta_i
        // Divide by jacobian (dOmegaI/dOmegaD)
        const float sinThetaD = scene_rdl2::math::sqrt(clamp(1 - cosThetaD*cosThetaD, 0.0f, 1.0f));
        *pdf *= sinThetaD;
        *pdf /= sinThetaIPrime; // sinThetaI = sinThetaIPrime
        // Control Variance Spikes - www.pbrt.org/hair.pdf (Section 1.8)
        // If the ratio of pdf in theta_d with the BRDF gets too high, clamp it
        // to control variance spikes. A ratio more than 4 seems satisfactory in tests.
        if (fabricBRDF/(*pdf) > 4) {
            *pdf = fabricBRDF/4;
        }
    }

    const Fresnel *F = getFresnel();
    const Color fresnel = (F != nullptr  ?  F->eval(cosThetaWo)  :  Color(1.0f));
    const Color f =  fresnel * getScale() * fabricBRDF;

    // Soften hard shadow terminator due to shading normals
    const float Gs = slice.computeShadowTerminatorFix(mFrame.getN(), wi);

    return Gs * f;
}

Color
KajiyaKayFabricBsdfLobe::sample(const BsdfSlice &slice,
                                float r1, float r2,
                                Vec3f &wi,
                                float &pdf) const
{
    const float cosNO = dot(mFrame.getN(), slice.getWo());
    if (cosNO <= sEpsilon) {
        pdf = 0.0f;
        return sBlack;
    }

    Vec3f omegaI;
    sampleVector(slice.getWo(), r1, r2, omegaI);

    // Convert back in Global Space
    wi = normalize(mFrame.localToGlobal(omegaI));

    return eval(slice, wi, &pdf);
}

void
KajiyaKayFabricBsdfLobe::sampleVector(const Vec3f& wo,
                                      float r1, float r2,
                                      Vec3f& wi) const
{
    // Theta is wrt the hair tangent
    const float thetaO = safeACos(dot(mThreadDirection, wo));

    // We sample theta_d and then convert it into the sample vector omega_i
    const float thetaD = scene_rdl2::math::acos(powf(r1, 1/(mSpecularExponent+1)));

    // theta_d = fabs(theta_o - theta_i_prime)
    // so, theta_i_prime = theta_o - theta_d || theta_i_prime = theta_o + theta_d
    float thetaIPrime = thetaO + thetaD;
    if (thetaIPrime >= sPi) {
        thetaIPrime = thetaO - thetaD;
    }
    if ((thetaD < thetaO) && (r2 < 0.5)) {
        thetaIPrime = thetaO - thetaD;
    }

    float cosThetaI, sinThetaI;
    sincos(thetaIPrime, &sinThetaI, &cosThetaI);

    // Phi Sampling
    const float phiM = sPi*r2;
    float cosPhi, sinPhi;
    sincos(phiM, &sinPhi, &cosPhi);

    // Theta wrt tangent
    wi.x = -cosThetaI;
    wi.y =  sinThetaI * cosPhi;
    wi.z =  sinThetaI * sinPhi;
}

} // namespace shading
} // namespace moonray

