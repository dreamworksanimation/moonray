// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

#include "TestDielectricFresnel.h"

#include <moonray/rendering/shading/bsdf/Fresnel.h>
#include <moonray/rendering/shading/bsdf/BsdfSlice.h>
#include <moonray/rendering/shading/bsdf/cook_torrance/BsdfCookTorrance.h>

#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>

namespace moonray {
namespace shading {

namespace {

using scene_rdl2::math::Color;
using scene_rdl2::math::Vec3f;

float
radicalInverseBase2(unsigned bits)
{
    bits = (bits << 16u) | (bits >> 16u);
    bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xaaaaaaaau) >> 1u);
    bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xccccccccu) >> 2u);
    bits = ((bits & 0x0f0f0f0fu) << 4u) | ((bits & 0xf0f0f0f0u) >> 4u);
    bits = ((bits & 0x00ff00ffu) << 8u) | ((bits & 0xff00ff00u) >> 8u);
    return static_cast<float>(bits) * 2.3283064365386963e-10f;
}

float
integrateDirectionalReflectance(const GGXCookTorranceBsdfLobe& lobe,
                                float cosThetaO,
                                unsigned sampleCount)
{
    const float sinThetaO = std::sqrt(std::max(0.0f, 1.0f - cosThetaO * cosThetaO));
    const Vec3f wo(sinThetaO, 0.0f, cosThetaO);
    const Vec3f n(0.0f, 0.0f, 1.0f);
    const BsdfSlice slice(n, wo, true, true, ispc::SHADOW_TERMINATOR_FIX_OFF);

    double sum = 0.0;
    for (unsigned i = 0; i < sampleCount; ++i) {
        Vec3f wi;
        float pdf = 0.0f;
        const Color f = lobe.sample(slice,
                                    (static_cast<float>(i) + 0.5f) / sampleCount,
                                    radicalInverseBase2(i),
                                    wi,
                                    pdf);
        if (pdf > 0.0f) {
            sum += f.r / pdf;
        }
    }
    return static_cast<float>(sum / sampleCount);
}

struct RoughMetrics
{
    float directionalReflectance;
    float hemisphericalReflectance;
    float directionalReflectanceSingleScatter;
    float hemisphericalReflectanceSingleScatter;
    float peakBrdf;
    float halfWidthDegrees;
    float totalDirectionalReflectance;
    float totalHemisphericalReflectance;
};

RoughMetrics
measureRoughDielectric(float roughness, float ior)
{
    const Vec3f n(0.0f, 0.0f, 1.0f);
    const Color favg(averageFresnelReflectance(ior));
    GGXCookTorranceBsdfLobe lobe(n, roughness, favg, Color(0.0f), 1.0f, ior, false);
    GGXCookTorranceBsdfLobe singleScatterLobe(n, roughness);
    DielectricFresnel fresnel(1.0f, ior);
    lobe.setFresnel(&fresnel);
    singleScatterLobe.setFresnel(&fresnel);

    RoughMetrics result;
    result.directionalReflectance = integrateDirectionalReflectance(lobe, 1.0f, 262144u);
    result.directionalReflectanceSingleScatter =
        integrateDirectionalReflectance(singleScatterLobe, 1.0f, 262144u);

    constexpr unsigned outgoingSteps = 128u;
    constexpr unsigned incomingSamples = 16384u;
    double specHemispherical = 0.0;
    double specHemisphericalSingleScatter = 0.0;
    double diffuseHemispherical = 0.0;
    const OneMinusRoughFresnel diffuseAttenuation(&fresnel, roughness);
    for (unsigned o = 0; o < outgoingSteps; ++o) {
        const float cosThetaO = (static_cast<float>(o) + 0.5f) / outgoingSteps;
        const float directional = integrateDirectionalReflectance(lobe, cosThetaO, incomingSamples);
        const float directionalSingleScatter =
            integrateDirectionalReflectance(singleScatterLobe, cosThetaO, incomingSamples);
        specHemispherical += 2.0 * directional * cosThetaO / outgoingSteps;
        specHemisphericalSingleScatter +=
            2.0 * directionalSingleScatter * cosThetaO / outgoingSteps;
        diffuseHemispherical += 2.0 * diffuseAttenuation.eval(cosThetaO).r * cosThetaO / outgoingSteps;
    }
    result.hemisphericalReflectance = static_cast<float>(specHemispherical);
    result.hemisphericalReflectanceSingleScatter =
        static_cast<float>(specHemisphericalSingleScatter);
    result.totalHemisphericalReflectance =
        static_cast<float>(specHemispherical + diffuseHemispherical);

    const BsdfSlice evalSlice(n, n, false, true, ispc::SHADOW_TERMINATOR_FIX_OFF);
    constexpr unsigned angularSteps = 131072u;
    result.peakBrdf = 0.0f;
    result.halfWidthDegrees = 90.0f;
    bool foundHalfWidth = false;
    for (unsigned i = 0; i <= angularSteps; ++i) {
        const float theta = scene_rdl2::math::sHalfPi * i / angularSteps;
        const Vec3f wi(std::sin(theta), 0.0f, std::cos(theta));
        const float value = lobe.eval(evalSlice, wi).r;
        result.peakBrdf = std::max(result.peakBrdf, value);
    }
    const float halfPeak = 0.5f * result.peakBrdf;
    for (unsigned i = 0; i <= angularSteps; ++i) {
        const float theta = scene_rdl2::math::sHalfPi * i / angularSteps;
        const Vec3f wi(std::sin(theta), 0.0f, std::cos(theta));
        if (lobe.eval(evalSlice, wi).r <= halfPeak) {
            result.halfWidthDegrees = 90.0f * i / angularSteps;
            foundHalfWidth = true;
            break;
        }
    }
    CPPUNIT_ASSERT(foundHalfWidth || result.peakBrdf == 0.0f);

    const float normalDiffuse = diffuseAttenuation.eval(1.0f).r;
    result.totalDirectionalReflectance = result.directionalReflectance + normalDiffuse;
    return result;
}

} // namespace

void
TestDielectricFresnel::testNormalIncidenceFromAir()
{
    struct TestCase
    {
        float ior;
        float expectedF0;
    };

    constexpr std::array<TestCase, 5> testCases = {{
        {1.0f, 0.0f},
        {1.1f, 0.0022675737f},
        {1.3f, 0.0170132325f},
        {1.5f, 0.04f},
        {2.0f, 0.1111111111f},
    }};

    for (const TestCase& testCase : testCases) {
        const scene_rdl2::math::Color result =
            DielectricFresnel::eval(1.0f, 1.0f, testCase.ior, 1.0f);

        CPPUNIT_ASSERT_DOUBLES_EQUAL(testCase.expectedF0, result.r, 1.0e-7f);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(testCase.expectedF0, result.g, 1.0e-7f);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(testCase.expectedF0, result.b, 1.0e-7f);
    }
}

void
TestDielectricFresnel::testRoughDielectricEnergy()
{
    constexpr std::array<float, 4> roughnessValues = {{0.1f, 0.25f, 0.5f, 0.8f}};
    constexpr std::array<float, 4> iorValues = {{1.0f, 1.3f, 1.5f, 2.0f}};

    std::cout << "\nroughness,ior,R_directional_normal,R_hemispherical,"
                 "R_directional_single_scatter,R_hemispherical_single_scatter,peak_brdf,"
                 "half_width_degrees,total_directional_normal,total_hemispherical\n";
    std::cout << std::fixed << std::setprecision(8);
    for (float roughness : roughnessValues) {
        for (float ior : iorValues) {
            const RoughMetrics result = measureRoughDielectric(roughness, ior);
            std::cout << roughness << ',' << ior << ','
                      << result.directionalReflectance << ','
                      << result.hemisphericalReflectance << ','
                      << result.directionalReflectanceSingleScatter << ','
                      << result.hemisphericalReflectanceSingleScatter << ','
                      << result.peakBrdf << ','
                      << result.halfWidthDegrees << ','
                      << result.totalDirectionalReflectance << ','
                      << result.totalHemisphericalReflectance << '\n';

            CPPUNIT_ASSERT(std::isfinite(result.directionalReflectance));
            CPPUNIT_ASSERT(std::isfinite(result.hemisphericalReflectance));
            CPPUNIT_ASSERT(result.directionalReflectance >= 0.0f);
            CPPUNIT_ASSERT(result.hemisphericalReflectance >= 0.0f);
            CPPUNIT_ASSERT(result.directionalReflectance <= 1.001f);
            CPPUNIT_ASSERT(result.hemisphericalReflectance <= 1.001f);
            CPPUNIT_ASSERT(result.totalDirectionalReflectance <= 1.005f);
            CPPUNIT_ASSERT(result.totalHemisphericalReflectance <= 1.005f);
        }
    }
}

CPPUNIT_TEST_SUITE_REGISTRATION(moonray::shading::TestDielectricFresnel);

} // namespace shading
} // namespace moonray
