// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

///
/// @file HairUtil.h
/// $Id$
///

#pragma once

#include <scene_rdl2/common/math/Color.h>
#include <scene_rdl2/common/math/Vec2.h>
#include <scene_rdl2/common/math/Vec3.h>


namespace moonray {
namespace shading {

//----------------------------------------------------------------------------

// Utility Function For Converting std::pow() into a set of multiplications
// From PBRTv3: pow() is usually implemented as : exp(log(a)*power)
// These transcendental functions -exp and log - are, usually, much slower than multiplications.
// This provides the motivation to convert r^k where k is known at compile time into a set of multiplications.
// In the hair code, we calculate r^20 and r^22 to convert roughness into variance for linear perception.
template<int n>
static float Pow(float v)
{
    float n2 = Pow<n/2>(v);
    return n2 * n2 * Pow<n & 1>(v);
}
template<> float Pow<0>(float v) { return 1; }
template<> float Pow<1>(float v) { return v; }

static float sSqrtPiOver8 = 0.626657069f;

/// Implementing Common Functions from:
/// [1] An Energy-conserving Hair Reflectance Model - Sig'11
/// [2] A Practical and Controllable Hair and Fur Model for Production Path Tracing - EGSR'16
class HairUtil {
public:

    // The paper suggests using a threshold of 1e-5. We use scene_rdl2::math::sEpsilon (1e-6)
    static bool
    clampTest(const scene_rdl2::math::Vec3f& hairDir,
              const scene_rdl2::math::Vec3f& wi)
    {
        static const float threshold = 1.0f - scene_rdl2::math::sEpsilon;
        if (scene_rdl2::math::abs(dot(wi, hairDir)) < threshold) {
            return true;
        }
        return false;
    }

    static bool
    clampTest(float cosThetaWi)
    {
        static const float threshold = 1.0f - scene_rdl2::math::sEpsilon;
        if (scene_rdl2::math::abs(cosThetaWi) < threshold) {
            return true;
        }
        return false;
    }

    static float safeSqrt(float x) { return (x > 0) ? scene_rdl2::math::sqrt(x) : 0.0f;  }

    // From [2] Section 4.1
    // Converts roughness (0,1) into longitudinal variance to linearize
    // the visual response for the *Energy Conserving Longitudinal*.
    static float
    longitudinalVar(float r)
    {
//        MNRY_ASSERT(r > 0.0f && r < 1.0f);
        float v = scene_rdl2::math::sqr(0.726f * r + 0.812f * r * r + 3.7f * Pow<20>(r));
        return v;
    }

    // From [2] Section 4.1
    // Convert roughness (0,1) into azimuthal variance to linearize
    // the visual response for the *Trimmed Logistic Function*.
    static float
    azimuthalVar(float r)
    {
//        MNRY_ASSERT(r > 0.0f && r < 1.0f);
        float v = sSqrtPiOver8*(0.265f * r + 1.194f * r * r +
                                5.372f * Pow<22>(r));
        return scene_rdl2::math::max(v, 0.05f);
    }

    // From [2], Section 4.2
    // Convert HairColor to Absorption Coeffs
    static scene_rdl2::math::Color
    computeAbsorptionCoefficients(const scene_rdl2::math::Color &hairColor,
                                  const float aziRoughness)
    {
        scene_rdl2::math::Color sigmaA = scene_rdl2::math::sBlack;
        const float denom = 5.969f - 0.215f * aziRoughness + 2.532f * scene_rdl2::math::sqr(aziRoughness) -
                            10.73f * Pow<3>(aziRoughness) + 5.574f * Pow<4>(aziRoughness) +
                            0.245f * Pow<5>(aziRoughness);
        if (!scene_rdl2::math::isZero(denom)) {
            for (int i = 0; i < 3; ++i) {
                sigmaA[i] = scene_rdl2::math::sqr(scene_rdl2::math::log(scene_rdl2::math::max(hairColor[i], scene_rdl2::math::sEpsilon)) / denom);
            }
        }
        return sigmaA;
    }

    // From [1], Section 6
    static float
    logBesselIO(float x)
    {
        if (x > 12.0f) {
            const float oneOverX = 1.0f / x;
            return x + 0.5f * (-scene_rdl2::math::log(scene_rdl2::math::sTwoPi) + scene_rdl2::math::log(oneOverX + 0.125f*oneOverX));
        }
        else {
            return scene_rdl2::math::log(besselIO(x));
        }
    }

    // From [1], Section 6
    static float
    besselIO(float x)
    {
        float sum = 0.0f;
        float x2i = 1.0f;
        int ifact = 1;
        int i4 = 1;
        for (int i = 0; i < 10; ++i) {
            if (i > 1) ifact *= i;
            sum += x2i / i4 / ifact / ifact;
            x2i *= x * x;
            i4 *= 4;
        }
        return sum;
    }


    // From [1], Section 4.2
    // Graph Comparison of Gaussian Vs This Function:
    // https://www.desmos.com/calculator/xuok8nijyn
    // Proof that this integrates to one:
    // WolframAlpha Link - https://tinyurl.com/m7bceaj
    static float
    deonLongitudinalM(float variance,
                      float sinThetaI, float cosThetaI,
                      float sinThetaO, float cosThetaO)
    {
        if (variance < scene_rdl2::math::sEpsilon)
            return 0.0f;

        float oneOverV = 1.0f / variance;
        float b = sinThetaI*sinThetaO*oneOverV;
        float x = cosThetaI*cosThetaO*oneOverV;

        float result = 1.0f;
        if (variance <= 0.1) {
            result = scene_rdl2::math::exp(logBesselIO(x) - b - oneOverV + 0.6931f + scene_rdl2::math::log(0.5f*oneOverV));
        } else {
            result = 0.5f*oneOverV*scene_rdl2::math::exp(-b)*besselIO(x) / scene_rdl2::math::sinh(oneOverV);
        }
        return result;
    }

    // From [2], Appendix A
    // https://en.wikipedia.org/wiki/Logistic_distribution
    static float
    logisticFunction(float x,
                     float s)
    {
        x = scene_rdl2::math::abs(x);
        const float r = -x/s;
        return scene_rdl2::math::exp(r) / (s * scene_rdl2::math::sqr(1 + scene_rdl2::math::exp(r)));
    }

    // From [2], Appendix A
    // https://en.wikipedia.org/wiki/Logistic_distribution
    static float
    logisticCDF(float x,
                float s)
    {
        return 1.0f / (1.0f + scene_rdl2::math::exp(-x/s));
    }

    // From [2], Appendix A
    // Note - Appendix A has a mistake in the normalization factor
    // which we've corrected here
    static float
    trimmedLogisticFunction(float x,
                            float s,
                            float a,
                            float b)
    {
        return logisticFunction(x, s) / (logisticCDF(b,s) - logisticCDF(a,s));
    }

    // This follows from the CDF derivation, not explained in Appendix A
    // Comments for the Derivation Inline
    static float
    sampleTrimmedLogistic(float r,
                          float s,
                          float a,
                          float b)
    {
        // Normalization Factor
        const float k = logisticCDF(b, s) - logisticCDF(a, s);
        // logisticCdf(x, s, a, b) = epsilon
        // Therefore, 1/k * (logisticCdf(x, s) - logisticCdf(a,s)) = epsilon
        // Therefore, logisticCdf(x,s) = epsilon * k + logisticCdf(a,s)
        const float logisticCdfX = (r * k + logisticCDF(a,s));
        float x = 0.0;
        if (!scene_rdl2::math::isZero(logisticCdfX)) {
            // Therefore, 1/(1+exp(-x/s)) = epsilon * k + logisticCdf(a,s) = logisticCdfX
            // Therefore, exp(-x/s) = 1/logisticCdfX - 1
            // Therefore, x = -s * log(1/logisticCdfX - 1)
            x = -s * scene_rdl2::math::log(1.0f/logisticCdfX - 1.0f);
        }
        return scene_rdl2::math::clamp(x, a, b);
    }

    static float
    unitGaussianForShade(float stddev, float x)
    {
        if (stddev < scene_rdl2::math::sEpsilon)
            return 0.0f;

        // Normalized Gaussian lobe used at shade time
        // The denominator normalizes the lobe (otherwise would give a peak value
        // of 1.0)
        return scene_rdl2::math::exp(-(x * x) / (2.0f * stddev * stddev)) / (stddev * scene_rdl2::math::sSqrtTwoPi);
    }

    // NOTE : added for future use in case we ever use Cauchy lobe shading.
    static float
    unitCauchyForShade(float stddev, float x)
    {
        if (stddev < scene_rdl2::math::sEpsilon)
            return 0.0f;

        // An unnormalized Cauchy lobe with sigma^2 in the
        // numerator gives you always a peak value of 1.0
        float retval = stddev * stddev;
        return retval / (x*x + retval);
    }

    static float
    unitCauchyForSample(float stddev, float x)
    {
        if (stddev < scene_rdl2::math::sEpsilon)
            return 0.0f;

        // A normalized Cauchy used for samples with sigma in
        // the numerator gives a total integral of 1.0.
        float retval = stddev * stddev;
        return stddev / (x*x + retval);
    }

    // Code for creating two new random numbers using one
    // From PBRT Section 4.3.3
    // https://fgiesen.wordpress.com/2009/12/13/decoding-morton-codes/
    static uint32_t
    compact1By1(uint32_t x) {
        // TODO: as of Haswell, the PEXT instruction could do all this in a
        // single instruction.
        // x = -f-e -d-c -b-a -9-8 -7-6 -5-4 -3-2 -1-0
        x &= 0x55555555;
        // x = --fe --dc --ba --98 --76 --54 --32 --10
        x = (x ^ (x >> 1)) & 0x33333333;
        // x = ---- fedc ---- ba98 ---- 7654 ---- 3210
        x = (x ^ (x >> 2)) & 0x0f0f0f0f;
        // x = ---- ---- fedc ba98 ---- ---- 7654 3210
        x = (x ^ (x >> 4)) & 0x00ff00ff;
        // x = ---- ---- ---- ---- fedc ba98 7654 3210
        x = (x ^ (x >> 8)) & 0x0000ffff;
        return x;
    }

    // Create Two Unique Random Numbers using a Morton Curve
    // https://fgiesen.wordpress.com/2009/12/13/decoding-morton-codes/
    static scene_rdl2::math::Vec2f
    demuxFloat(float f) {
        assert(f >= 0 && f < 1);
        uint64_t v = f * (1ull << 32);
        uint32_t bits[2] = {compact1By1(v), compact1By1(v >> 1)};
        return scene_rdl2::math::Vec2f(bits[0] / float(1 << 16), bits[1] / float(1 << 16));
    }
};

//----------------------------------------------------------------------------
} // namespace shading
} // namespace moonray


