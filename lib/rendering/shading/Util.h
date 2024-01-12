// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0


#pragma once

#include <moonray/rendering/shading/ispc/Util_ispc_stubs.h>

#include <scene_rdl2/common/math/Color.h>
#include <scene_rdl2/common/math/ColorSpace.h>
#include <scene_rdl2/common/math/Constants.h>
#include <scene_rdl2/common/math/Vec2.h>
#include <scene_rdl2/common/math/Mat4.h>
#include <scene_rdl2/common/math/Math.h>
#include <scene_rdl2/common/math/MathUtil.h>
#include <moonray/rendering/shading/ispc/Util_ispc_stubs.h>
#include <moonray/rendering/shading/ispc/RampControl_ispc_stubs.h>

namespace moonray {
namespace shading {

// Returns angle in range in [-pi,pi]
static finline float rangeAngle(float angle)
{
    while (angle > scene_rdl2::math::sPi) angle -= scene_rdl2::math::sTwoPi;
    while (angle < -scene_rdl2::math::sPi) angle += scene_rdl2::math::sTwoPi;
    return angle;
}

//----------------------------------------------------------------------------

// Conversion from roughness to exponent metric
finline float
roughness2Exponent(float roughness)
{
    return 2.0f * scene_rdl2::math::rcp(roughness * roughness);
}

finline scene_rdl2::math::Vec2f
roughness2Exponent(const scene_rdl2::math::Vec2f &roughness)
{
    return 2.0f * scene_rdl2::math::rcp(roughness * roughness);
}


// Conversion from exponent to roughness metric
finline float
exponent2Roughness(float exponent)
{
    return scene_rdl2::math::sSqrtTwo * scene_rdl2::math::rsqrt(exponent);
}

finline scene_rdl2::math::Vec2f
exponent2Roughness(const scene_rdl2::math::Vec2f &exponent)
{
    return scene_rdl2::math::Vec2f(scene_rdl2::math::sSqrtTwo) * scene_rdl2::math::rsqrt(exponent);
}


// Conversion from roughness to width (in radians)
finline float
roughness2Width(float roughness)
{
    // See rats test rats/material/base/roughness_comparison
    static const float factor = 55.0f * scene_rdl2::math::sPi / 180.0f;

    const float exponent = roughness2Exponent(roughness);
    const float width = factor * scene_rdl2::math::pow(exponent, -0.8f);

    return width;
}

finline scene_rdl2::math::Vec2f
roughness2Width(const scene_rdl2::math::Vec2f &roughness)
{
    // See rats test rats/material/base/roughness_comparison
    static const float factor = 55.0f * scene_rdl2::math::sPi / 180.0f;

    const scene_rdl2::math::Vec2f exponent = roughness2Exponent(roughness);
    scene_rdl2::math::Vec2f width;
    width.x = factor * scene_rdl2::math::pow(exponent.x, -0.8f);
    width.y = factor * scene_rdl2::math::pow(exponent.y, -0.8f);

    return width;
}


// Conversion from width (in radians) to roughness
finline float
width2Roughness(float width)
{
    // See inverse conversion above
    static const float factor = 180.0f / (scene_rdl2::math::sPi * 55.0f);
    static const float e = 1.0f / (2.0f * 0.8f);

    const float roughness = scene_rdl2::math::sSqrtTwo * scene_rdl2::math::pow(factor * width, e);

    return roughness;
}

finline scene_rdl2::math::Vec2f
width2Roughness(const scene_rdl2::math::Vec2f &width)
{
    // See inverse conversion above
    static const float factor = 180.0f / (scene_rdl2::math::sPi * 55.0f);
    static const float e = 1.0f / (2.0f * 0.8f);

    scene_rdl2::math::Vec2f roughness;
    roughness.x = scene_rdl2::math::sSqrtTwo * scene_rdl2::math::pow(factor * width.x, e);
    roughness.y = scene_rdl2::math::sSqrtTwo * scene_rdl2::math::pow(factor * width.y, e);

    return roughness;
}

// Dispersion Support Via Abbe Numbers

// The following function computes three different IORs for R, G and B wavelengths based on a concept called
// the "abbe number" as defined in https://en.wikipedia.org/wiki/Abbe_number.
// I've used the following three equations to calculate the three variables
// AbbeNumber = (etaG - 1) / (etaB - etaR)          (1)
// Assuming etaGreen as the average eta specified by the user:
// etaG = etaT                                      (2)
// avgEta = etaT = (etaR + etaG + etaB) / 3.0f      (3)
// Substituing for etaG in equation (3), we get:
// etaB + etaR = 2 * etaG                           (4)
// Eqn (1) can be written as:
// etaB - etaR = (etaG - 1) / AbbeNumber            (5)
// Eqns (4) and (5) can be solved as:
// etaB = (etaG - 1) / (2 * AbbeNumber) + etaG
// etaR = 2 * etaG - etaB
finline void
computeSpectralIOR(float eta,
                   float abbeNumber,
                   float *etaR,
                   float *etaG,
                   float *etaB)
{
    MNRY_ASSERT(!scene_rdl2::math::isZero(abbeNumber));
    *etaG = eta;
    *etaB = (*etaG - 1.0f) / (2 * abbeNumber) + *etaG;
    *etaR = 2 * *etaG - *etaB;
}

// Sampling R, G and B wavelengths with the weights (0.35f, 0.35f, 0.30f).
// Choosing this specific combination instead of (0.33, 0.33, 0.33) since it quite evenly sums to 1.0f
// This function also renormalizes the random number used to select a wavelength and returns
// the appropriate dispersionColor and PDF.
finline void
sampleSpectralIOR(float& r,
                  float etaR,
                  float etaG,
                  float etaB,
                  float *sampleEta,
                  float *pdf,
                  scene_rdl2::math::Color& dispersionColor)
{
    if (r < 0.35f) {
        r /= 0.35f;
        // Red
        dispersionColor = scene_rdl2::math::Color(1.0f, 0.0f, 0.0f);
        *sampleEta = etaR;
        *pdf = 0.35f;
    } else if (r < 0.7f) {
        r = (r - 0.35f) / 0.35f;
        // Green
        dispersionColor = scene_rdl2::math::Color(0.0f, 1.0f, 0.0f);
        *sampleEta = etaG;
        *pdf = 0.35f;
    } else {
        r = (r - 0.7f) / 0.3f;
        // Blue
        dispersionColor = scene_rdl2::math::Color(0.0f, 0.0f, 1.0f);
        *sampleEta = etaB;
        *pdf = 0.3f;
    }
}

//----------------------------------------------------------------------------

finline scene_rdl2::math::Vec3f
sampleLocalHemisphereCosine(float r1, float r2)
{
    // Sample polar coordinates
    float cosTheta = (r1 > 1.0f  ?  1.0f  :  scene_rdl2::math::sqrt(r1));
    float sinTheta = scene_rdl2::math::sqrt(1.0f - r1);

    float phi = r2 * scene_rdl2::math::sTwoPi;
    float cosPhi;
    float sinPhi;
    scene_rdl2::math::sincos(phi, &sinPhi, &cosPhi);

    // Convert to direction in the reference frame
    scene_rdl2::math::Vec3f result;
    result[0] = sinTheta * cosPhi;
    result[1] = sinTheta * sinPhi;
    result[2] = cosTheta;

    return result;
}

finline void
squarifyRectangle(scene_rdl2::math::Vec3f &edge1, scene_rdl2::math::Vec3f &edge2)
{
    float length1 = length(edge1);
    float length2 = length(edge2);
    float newLength = scene_rdl2::math::sqrt(length1 * length2);

    edge1 *= newLength / length1;
    edge2 *= newLength / length2;
}

/// See ReferenceFrame for local sphere parameterization (Z-up)
finline scene_rdl2::math::Vec3f
computeLocalSphericalDirection(float cosTheta, float sinTheta, float phi)
{
    scene_rdl2::math::Vec3f result;

    float cosPhi;
    float sinPhi;
    scene_rdl2::math::sincos(phi, &sinPhi, &cosPhi);

    result[0] = sinTheta * cosPhi;
    result[1] = sinTheta * sinPhi;
    result[2] = cosTheta;
    MNRY_ASSERT(isNormalized(result));

    return result;
}

finline scene_rdl2::math::Vec3f
computeLocalSphericalDirection(float cosTheta, float sinTheta,
                               float cosPhi, float sinPhi)
{
    scene_rdl2::math::Vec3f result;

    result[0] = sinTheta * cosPhi;
    result[1] = sinTheta * sinPhi;
    result[2] = cosTheta;
    MNRY_ASSERT(isNormalized(result));

    return result;
}

/// Y-up sphere parameterization: theta=0 --> y-axis, phi=0..pi/2 --> x..z axes
finline scene_rdl2::math::Vec3f
computeYupSphericalDirection(float cosTheta, float sinTheta, float phi)
{
    scene_rdl2::math::Vec3f result;

    float cosPhi;
    float sinPhi;
    scene_rdl2::math::sincos(phi, &sinPhi, &cosPhi);

    result[0] = sinTheta * cosPhi;
    result[1] = cosTheta;
    result[2] = sinTheta * sinPhi;
    MNRY_ASSERT(isNormalized(result));

    return result;
}


/// Y-up sphere parameterization: theta=0 --> y-axis, phi=0..pi/2 --> x..z axes
finline void
computeInverseYupSphericalDirection(const scene_rdl2::math::Vec3f &wi, float &theta, float &phi)
{
    // Inverse the transform from computeYupSphericalDirection()
    float cosTheta = wi[1];
    cosTheta = scene_rdl2::math::max(cosTheta, -1.0f);
    cosTheta = scene_rdl2::math::min(cosTheta, 1.0f);
    theta = scene_rdl2::math::acos(cosTheta);

    phi = scene_rdl2::math::atan2(wi[2], wi[0]);
    phi = (phi < 0.0f  ?  phi + scene_rdl2::math::sTwoPi  :  phi);
}


finline scene_rdl2::math::Vec3f
computeLocalReflectionDirection(const scene_rdl2::math::Vec3f &woL)
{
    scene_rdl2::math::Vec3f wiL;

    wiL[0] = -woL[0];
    wiL[1] = -woL[1];
    wiL[2] = woL[2];

    return wiL;
}


/// Assumes all vectors are normalized, and N points on the side of Ng, not Ngf
/// Returns N.dot(wo) (which is also N.dot(wi))
finline float
computeReflectionDirection(const scene_rdl2::math::Vec3f &N, const scene_rdl2::math::Vec3f &wo, scene_rdl2::math::Vec3f &wi)
{
    float NDotWo = dot(N, wo);

    wi = -wo + 2.0f * NDotWo * N;

    // There is no need to normalize R here
    MNRY_ASSERT(isNormalized(wi));

    return NDotWo;
}


/// For details see:
///     "Tracing ray differentials - Homan Igehy,
///      Computer Graphics #33 (Annual Conference Series 1999)"
finline void
computeReflectionDirectionDifferential(
        const scene_rdl2::math::Vec3f &wo, const scene_rdl2::math::Vec3f &wi, const scene_rdl2::math::Vec3f &H,
        const scene_rdl2::math::Vec3f &dNdx, const scene_rdl2::math::Vec3f &dNdy, scene_rdl2::math::Vec3f &dDdx, scene_rdl2::math::Vec3f &dDdy)
{
    // Note: This is slightly incorrect, since we use the shading normal
    // derivatives, not the H vector derivatives. But it is only important
    // for this to be correct for mirror reflections (when H == N).
    float Cx = dot(dDdx, H) - dot(wo, dNdx);
    float Cy = dot(dDdy, H) - dot(wo, dNdy);

    float dDotN = -dot(wo, H);
    dDdx -= 2.0f * (dDotN * dNdx + Cx * H);
    dDdy -= 2.0f * (dDotN * dNdy + Cy * H);
}


/// Same as computeReflectionDirectionDifferential(), but ignoring the effect
/// of surface curvature
finline void
computeReflectionDirectionDifferentialNoCurvature(
        const scene_rdl2::math::Vec3f &wo, const scene_rdl2::math::Vec3f &wi, const scene_rdl2::math::Vec3f &H,
        scene_rdl2::math::Vec3f &dDdx, scene_rdl2::math::Vec3f &dDdy)
{
    dDdx -= 2.0f * (dot(dDdx, H) * H);
    dDdy -= 2.0f * (dot(dDdy, H) * H);
}


/// Assumes all vectors are normalized, and N points on the side of Ng (it has
/// not been flipped towards wo).
/// Returns false in case of total internal reflection, and true otherwise.
/// In the latter case, it also returns scene_rdl2::math::abs(N.dot(wo)) and
/// scene_rdl2::math::abs(N.dot(wi)).
finline bool
computeRefractionDirection(const scene_rdl2::math::Vec3f &N,
                           const scene_rdl2::math::Vec3f &wo,
                           float neta,
                           scene_rdl2::math::Vec3f &wi,
                           float &NdotWo,
                           float &NdotWi)
{
    // Compute refraction vector
    NdotWo = dot(N, wo);
    scene_rdl2::math::Vec3f Nf = (NdotWo > 0.0f  ?  N  :  -N);
    NdotWo = scene_rdl2::math::abs(NdotWo);

    NdotWi = 1.0f - neta * neta * (1.0f - (NdotWo * NdotWo));
    if (NdotWi < 0.0f) {
        // Total internal reflection
        return false;
    }
    NdotWi = scene_rdl2::math::sqrt(NdotWi);

    float negMu = neta * NdotWo - NdotWi;
    wi = negMu * Nf - neta * wo;

    // There is no need to normalize R here
    MNRY_ASSERT(isNormalized(wi));

    return true;
}


/// For details see:
///     "Tracing ray differentials - Homan Igehy,
///      Computer Graphics #33 (Annual Conference Series 1999)"
finline void
computeRefractionDirectionDifferential(const float neta,
                                       const scene_rdl2::math::Vec3f &wo,
                                       const scene_rdl2::math::Vec3f &wi,
                                       const scene_rdl2::math::Vec3f &H,
                                       const scene_rdl2::math::Vec3f &dNdx,
                                       const scene_rdl2::math::Vec3f &dNdy,
                                       scene_rdl2::math::Vec3f &dDdx,
                                       scene_rdl2::math::Vec3f &dDdy)
{
    // Note: This is slightly incorrect, since we use the shading normal
    // derivatives, not the H vector derivatives. But it is only important
    // for this to be correct for mirror reflections (when H == N).
    float HdotWo = dot(H, wo);

    // TODO: Check if this test is necessary
    scene_rdl2::math::Vec3f Hf = (HdotWo > 0.0f  ?  H  :  -H);
    HdotWo = scene_rdl2::math::abs(HdotWo);

    float Cx = dot(dDdx, Hf) - dot(wo, dNdx);
    float Cy = dot(dDdy, Hf) - dot(wo, dNdy);

    float HdotWi = dot(Hf, wi);
    HdotWi = scene_rdl2::math::min(HdotWi, -scene_rdl2::math::sEpsilon);

    float dmu = (neta + (neta * neta * HdotWo / HdotWi));
    float dmudx = dmu * Cx;
    float dmudy = dmu * Cy;

    float mu = - neta * HdotWo - HdotWi;
    dDdx = neta * dDdx - (mu * dNdx + dmudx * Hf);
    dDdy = neta * dDdy - (mu * dNdy + dmudy * Hf);
}


/// H can be very small if wo and wi are almost the opposite vectors. This
/// happens in the unfortunate event when the sampled H vector was almost
/// at 90 degrees from wo. In that case we can't re-construct H
/// and return false. We return a normalized H and true otherwise.
finline bool
computeNormalizedHalfVector(const scene_rdl2::math::Vec3f &wo, const scene_rdl2::math::Vec3f &wi, scene_rdl2::math::Vec3f &H)
{
    static const float halfVectorLengthMinSqr = 0.001f * 0.001f;

    H = wo + wi;

    float lengthHSqr = lengthSqr(H);
    if (lengthHSqr > halfVectorLengthMinSqr) {
        H *= scene_rdl2::math::rsqrt(lengthHSqr);
        return true;
    } else {
        return false;
    }
}


// H is always pointing towards medium with lowest IOR, but only if wo and wi
// are pointing on opposite side of the surface (wrt. N).
finline bool
computeNormalizedRefractionHalfVector(const float iorWo, const scene_rdl2::math::Vec3f &wo,
        const float iorWi, const scene_rdl2::math::Vec3f &wi, scene_rdl2::math::Vec3f &H)
{
    static const float halfVectorLengthMinSqr = 0.001f * 0.001f;

    H = -(iorWo * wo + iorWi * wi);

    float lengthHSqr = lengthSqr(H);
    if (lengthHSqr > halfVectorLengthMinSqr) {
        H *= scene_rdl2::math::rsqrt(lengthHSqr);
        return true;
    } else {
        return false;
    }
}


finline void
localHemisphereCosineDifferentials(float r1, float r2, scene_rdl2::math::Vec3f &dDdr1, scene_rdl2::math::Vec3f &dDdr2)
{
    // See lib/rendering/pbr/doc/diffuse_scatter_derivatives.mw for Maple
    // source code on how we got to this result

    // The derivative gets very big close to the equator and the pole
    // (because of the division by cosTheta and sinTheta below)
    // Cap r1 to make sure we stay an epsilon away from both
    static const float minR1 = scene_rdl2::math::pow(scene_rdl2::math::cos(scene_rdl2::math::deg2rad(85.0f)), 2.0f);
    static const float maxR1 = 1.0f - minR1;
    r1 = scene_rdl2::math::clamp(r1, minR1, maxR1);

    float cosTheta = (r1 > 1.0f  ?  1.0f  :  scene_rdl2::math::sqrt(r1));
    float sinTheta = scene_rdl2::math::sqrt(1.0f - r1);

    float phi = r2 * scene_rdl2::math::sTwoPi;
    float cosPhi;
    float sinPhi;
    scene_rdl2::math::sincos(phi, &sinPhi, &cosPhi);

    // We scale our derivatives down to account for multiple samples taken over
    // the hemisphere
    static const float scale = 1.0f / 8.0f;
    static const float halfScale = 0.5f * scale;
    static const float twoPiScale = scene_rdl2::math::sTwoPi * scale;

    dDdr1 = scene_rdl2::math::Vec3f(-halfScale * cosPhi / sinTheta,
                  -halfScale * sinPhi / sinTheta,
                   halfScale / cosTheta);
    dDdr2 = scene_rdl2::math::Vec3f(-twoPiScale * sinPhi * sinTheta,
                   twoPiScale * cosPhi * sinTheta,
                   0.0f);
}


/// pdf = 1.0 / (2.0 * PI)
finline scene_rdl2::math::Vec3f
sampleLocalHemisphereUniform(float r1, float r2)
{
    // Sample polar coordinates
    float cosTheta = (r1 > 1.0f  ?  1.0f  :  r1);
    float sinTheta = scene_rdl2::math::sqrt(scene_rdl2::math::max(0.0f, 1.0f - cosTheta * cosTheta));

    float phi = r2 * scene_rdl2::math::sTwoPi;
    float cosPhi;
    float sinPhi;
    scene_rdl2::math::sincos(phi, &sinPhi, &cosPhi);

    // Convert to direction in the reference frame
    scene_rdl2::math::Vec3f result;
    result[0] = sinTheta * cosPhi;
    result[1] = sinTheta * sinPhi;
    result[2] = cosTheta;

    return result;
}


/// pdf = 1.0f / (2.0f * PI * (1.0f - cosThetaMax))
/// See next function for a more accurate version.
finline scene_rdl2::math::Vec3f
sampleLocalSphericalCapUniform(float r1, float r2, float cosThetaMax)
{
    float cosTheta = (1.0f - r1 * (1.0f - cosThetaMax));
    cosTheta = (cosTheta > 1.0f  ?  1.0f  :  cosTheta);
    float sinTheta = scene_rdl2::math::sqrt(scene_rdl2::math::max(0.0f, 1.0f - cosTheta * cosTheta));

    float phi = r2 * scene_rdl2::math::sTwoPi;
    float cosPhi;
    float sinPhi;
    scene_rdl2::math::sincos(phi, &sinPhi, &cosPhi);

    // Convert to direction in the reference frame
    scene_rdl2::math::Vec3f result;
    result[0] = sinTheta * cosPhi;
    result[1] = sinTheta * sinPhi;
    result[2] = cosTheta;

    return result;
}


/// A second version of the function above. This version produces much more accurate results when the cap
/// becomes very small and cosThetaMax approaches 1.
finline scene_rdl2::math::Vec3f
sampleLocalSphericalCapUniform2(float r1, float r2, float versineThetaMax)
{
    float versineTheta = scene_rdl2::math::max(r1 * versineThetaMax, 0.0f);
    float cosTheta = 1.0f - versineTheta;
    float sineSquaredTheta = versineTheta * (2.0f - versineTheta);  // more accurate way to compute 1-cosTheta^2
    float sinTheta = scene_rdl2::math::sqrt(scene_rdl2::math::max(0.0f, sineSquaredTheta));

    float phi = r2 * scene_rdl2::math::sTwoPi;
    float cosPhi;
    float sinPhi;
    scene_rdl2::math::sincos(phi, &sinPhi, &cosPhi);

    // Convert to direction in the reference frame
    scene_rdl2::math::Vec3f result;
    result[0] = sinTheta * cosPhi;
    result[1] = sinTheta * sinPhi;
    result[2] = cosTheta;

    return result;
}


/// pdf = 1.0 / (4.0 * PI)
finline scene_rdl2::math::Vec3f
sampleSphereUniform(float r1, float r2)
{
    // Sample polar coordinates
    float cosTheta = 1.0f - 2.0f * r1;
    MNRY_ASSERT(cosTheta >= -1.0f  &&  cosTheta <= 1.0f);
    float sinTheta = scene_rdl2::math::sqrt(scene_rdl2::math::max(0.0f, 1.0f - cosTheta * cosTheta));

    float phi = r2 * scene_rdl2::math::sTwoPi;
    float cosPhi;
    float sinPhi;
    scene_rdl2::math::sincos(phi, &sinPhi, &cosPhi);

    // Convert to direction in the reference frame
    scene_rdl2::math::Vec3f result;
    result[0] = sinTheta * cosPhi;
    result[1] = sinTheta * sinPhi;
    result[2] = cosTheta;

    return result;
}

// Converts a continuous variable in the space [0,1] (t)
// to the nearest of n (numBins) discrete values in [0,1].
finline static float
discretize(const float t, const size_t numBins)
{
    // optimization
    if (numBins == 1) { return 1.0f; }

    return (scene_rdl2::math::ceil(scene_rdl2::math::saturate(t) * static_cast<float>(numBins)) /
            static_cast<float>(numBins));
}

// Determines whether or not to apply gamma correction based on the gamma mode and number of channels
finline bool
getApplyGamma(ispc::TEXTURE_GammaMode gammaMode,
              int nChannels)
{
        switch(gammaMode) {
            case ispc::TEXTURE_GAMMA_OFF:
                return false;
            case ispc::TEXTURE_GAMMA_ON:
                return true;
            case ispc::TEXTURE_GAMMA_AUTO:
                return true;
            case ispc::TEXTURE_GAMMA_USD:
                return nChannels != 1;
            default:
                return true;
        }
}

} // namespace shading
} // namespace moonray


