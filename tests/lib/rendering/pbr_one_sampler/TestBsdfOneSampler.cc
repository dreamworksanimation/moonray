// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

///
/// @file TestBsdfOneSampler.cc
/// $Id$
///

#include "BsdfFactory.h"
#include "TestBsdfCommon.h"
#include "TestBsdfOneSampler.h"
#include "TestUtil.h"

#include <scene_rdl2/common/math/Color.h>
#include <scene_rdl2/render/util/Random.h>
#include <scene_rdl2/common/math/ReferenceFrame.h>


namespace moonray {
namespace pbr {


//----------------------------------------------------------------------------

// Change the line below to make tests simpler (for debugging)
#if 0
static const int sViewAnglesTheta = 4;
static const int sRoughnessCount = 1;
static const float sRoughness[sRoughnessCount] =
    { 0.4f };
#else
static const int sViewAnglesTheta = 4;
static const int sRoughnessCount = 6;
static const float sRoughness[sRoughnessCount] =
    { 0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.7f };
#endif


//----------------------------------------------------------------------------

static void
runTest(TestBsdfSettings &test, int viewAnglesTheta,
        int viewAnglesPhy, int sampleCount)
{
    // Passthrough
    runBsdfTest(test, viewAnglesTheta, viewAnglesPhy, sampleCount);
}

//----------------------------------------------------------------------------

void TestBsdfOneSampler::setUp()
{
    setupThreadLocalData();
}

void TestBsdfOneSampler::tearDown()
{
    cleanupThreadLocalData();
}

TestBsdfOneSampler::TestBsdfOneSampler()
{
}

TestBsdfOneSampler::~TestBsdfOneSampler()
{
}


//----------------------------------------------------------------------------

void
TestBsdfOneSampler::testLambert()
{
    scene_rdl2::math::ReferenceFrame frame;

    printInfo("##### TestBsdfOneSampler::testLambert() ####################");
    LambertBsdfFactory factory;
    TestBsdfSettings test(factory, frame, true, 0.002, 0.01, true, true,
            TestBsdfSettings::BSDF_ONE_SAMPLER);
    runTest(test, sViewAnglesTheta, 1, getSampleCount(sRoughness[sRoughnessCount - 1]));
}


//----------------------------------------------------------------------------

void
TestBsdfOneSampler::testCookTorrance()
{
    scene_rdl2::math::ReferenceFrame frame;

    printInfo("##### TestBsdfOneSampler::testCookTorrance() ####################");
    for (int i=0; i < sRoughnessCount; i++) {
        printInfo("===== Roughness: %f ====================", sRoughness[i]);

        CookTorranceBsdfFactory factory(sRoughness[i]);
        TestBsdfSettings test(factory, frame, true, 0.002, 0.03,
                (sRoughness[i] > 0.1f  &&  sRoughness[i] < 0.7f),
                (sRoughness[i] > 0.1f), TestBsdfSettings::BSDF_ONE_SAMPLER);
        runTest(test, sViewAnglesTheta, 1, getSampleCount(sRoughness[i]));
    }
}


void
TestBsdfOneSampler::testGGXCookTorrance()
{
    scene_rdl2::math::ReferenceFrame frame;

    printInfo("##### TestBsdfOneSampler::testGGXCookTorrance() ####################");
    for (int i=0; i < sRoughnessCount; i++) {
        printInfo("===== Roughness: %f ====================", sRoughness[i]);

        GGXCookTorranceBsdfFactory factory(sRoughness[i]);
        TestBsdfSettings test(factory, frame, true, 0.002, 0.03,
                (sRoughness[i] > 0.1f  &&  sRoughness[i] < 0.4f),
                (sRoughness[i] > 0.1f  &&  sRoughness[i] < 0.4f),
                TestBsdfSettings::BSDF_ONE_SAMPLER);
        runTest(test, sViewAnglesTheta, 1, getSampleCount(sRoughness[i]));
    }
}


void
TestBsdfOneSampler::testAnisoCookTorrance()
{
    scene_rdl2::math::ReferenceFrame frame;

    printInfo("##### TestBsdfOneSampler::testAnisoCookTorrance() ####################");
    for (int i=0; i < sRoughnessCount; i++) {
        int j = (i + 4) % sRoughnessCount;
        printInfo("===== Roughness: %f and %f ====================",
                sRoughness[i], sRoughness[j]);

        int sampleCount = getSampleCount(sRoughness[i]) +
                          getSampleCount(sRoughness[j]);

        AnisoCookTorranceBsdfFactory factory(sRoughness[i], sRoughness[j]);
        TestBsdfSettings test(factory, frame, true, 0.002, 0.03,
                (sRoughness[i] > 0.1f  &&  sRoughness[j] > 0.1f  &&
                 sRoughness[i] < 0.7f  &&  sRoughness[j] < 0.7f),
                sRoughness[i] > 0.1f  &&  sRoughness[j] > 0.1f,
                TestBsdfSettings::BSDF_ONE_SAMPLER);
        runTest(test, sViewAnglesTheta, 1, sampleCount);
    }
}


void
TestBsdfOneSampler::testTransmissionCookTorrance()
{
    scene_rdl2::math::ReferenceFrame frame;

    printInfo("##### TestBsdfOneSampler::testTransmissionCookTorrance() ####################");
    for (int i=0; i < sRoughnessCount; i++) {
        printInfo("===== Roughness: %f ====================", sRoughness[i]);

        TransmissionCookTorranceBsdfFactory factory(sRoughness[i]);
        TestBsdfSettings test(factory, frame, false, 0.002, 0.05,
                (sRoughness[i] > 0.2f  &&  sRoughness[i] < 0.7f),
                (sRoughness[i] > 0.2f), TestBsdfSettings::BSDF_ONE_SAMPLER);
        runTest(test, sViewAnglesTheta, 1, 4 * getSampleCount(sRoughness[i]));
    }
}

//----------------------------------------------------------------------------

void
TestBsdfOneSampler::testIridescence()
{
    scene_rdl2::math::ReferenceFrame frame;

    static const int size = 2;
    const float positions[size] = {0.f, 1.f};
    const ispc::RampInterpolatorMode interpolators[size] = {
        ispc::RAMP_INTERPOLATOR_MODE_SMOOTH, 
        ispc::RAMP_INTERPOLATOR_MODE_SMOOTH };
    const scene_rdl2::math::Color colors[size] = {
        scene_rdl2::math::Color(1.f, 0.f, 0.f), scene_rdl2::math::Color(1.f, 0.f, 1.f)};

    printInfo("##### TestBsdfOneSampler::testIridescence() ####################");
    for (int i=0; i < sRoughnessCount; i++) {
        printInfo("===== Roughness: %f ====================", sRoughness[i]);

        IridescenceBsdfFactory factory(sRoughness[i], 1.0f, ispc::SHADING_IRIDESCENCE_COLOR_USE_HUE_INTERPOLATION,
                scene_rdl2::math::Color(1.f, 0.f, 0.f), scene_rdl2::math::Color(1.f, 0.f, 1.f), false,
                ispc::COLOR_RAMP_CONTROL_SPACE_RGB, size, positions, interpolators, colors,
                2.f, 1.f, 1.f, 1.f);
        TestBsdfSettings test(factory, frame, true, 0.002, 0.04,
                (sRoughness[i] > 0.1  &&  sRoughness[i] < 0.5),
                (sRoughness[i] > 0.1), TestBsdfSettings::BSDF_ONE_SAMPLER);
        runTest(test, sViewAnglesTheta, 1, getSampleCount(sRoughness[i]));
    }
}


//----------------------------------------------------------------------------

void
TestBsdfOneSampler::testRetroreflection()
{
    scene_rdl2::math::ReferenceFrame frame;

    printInfo("##### TestBsdfOneSampler::testRetroreflection() ####################");
    for (int i=0; i < sRoughnessCount; i++) {
        printInfo("===== Roughness: %f ====================", sRoughness[i]);

        RetroreflectionBsdfFactory factory(sRoughness[i]);
        // The integrals are evaluated using uniform sampling which makes it an
        // error-prone calculation. Ideally, we'd have higher tolerance for sharper lobes.
        // Right now we set it up to 'fail' only for higher roughness values.
        TestBsdfSettings test(factory, frame, true, 0.002, 0.04,
                sRoughness[i] > 0.3f, sRoughness[i] > 0.3f,
                TestBsdfSettings::BSDF_ONE_SAMPLER);
        runTest(test, sViewAnglesTheta, 1, getSampleCount(sRoughness[i]));
    }
}

//----------------------------------------------------------------------------

void
TestBsdfOneSampler::testEyeCaustic()
{
    scene_rdl2::math::ReferenceFrame frame;

    printInfo("##### TestBsdfOneSampler::testEyeCaustic() ####################");
    for (int i=0; i < sRoughnessCount; i++) {
        printInfo("===== Roughness: %f ====================", sRoughness[i]);

        EyeCausticBsdfFactory factory(sRoughness[i]);
        // The integrals are evaluated using uniform sampling which makes it an
        // error-prone calculation. Ideally, we'd have higher tolerance for sharper lobes.
        // Right now we set it up to 'fail' only for higher roughness values.
        TestBsdfSettings test(factory, frame, false , 0.002, 0.04,
                sRoughness[i] > 0.1f, sRoughness[i] > 0.1f,
                TestBsdfSettings::BSDF_ONE_SAMPLER);
        runTest(test, sViewAnglesTheta, 1, getSampleCount(sRoughness[i]));
    }
}

//----------------------------------------------------------------------------

void
TestBsdfOneSampler::testDwaFabric()
{
    scene_rdl2::math::ReferenceFrame frame;

    printInfo("##### TestBsdfOneSampler::testDwaFabric() ####################");
    for (int i=0; i < sRoughnessCount; i++) {
        printInfo("===== SpecRoughness: %f ====================", sRoughness[i]);
        DwaFabricBsdfFactory factory(sRoughness[i]);
        // The integrals are evaluated using uniform sampling which makes it an
        // error-prone calculation. Ideally, we'd have higher tolerance for sharper lobes.
        // Right now we set it up to 'fail' only for higher roughness values.
        TestBsdfSettings test(factory, frame, true, 0.002, 0.03,
                              (sRoughness[i] > 0.1f  &&  sRoughness[i] < 0.7f),
                              (sRoughness[i] > 0.1f), TestBsdfSettings::BSDF_ONE_SAMPLER);
        runTest(test, sViewAnglesTheta, 1, getSampleCount(sRoughness[i]));
    }
}

//----------------------------------------------------------------------------

void
TestBsdfOneSampler::testKajiyaKayFabric()
{
    scene_rdl2::math::ReferenceFrame frame;

    printInfo("##### TestBsdfOneSampler::testKajiyaKayFabric() ####################");
    for (int i=0; i < sRoughnessCount; i++) {
        printInfo("===== SpecRoughness: %f ====================", sRoughness[i]);
        // Kajiya Kay Model
        KajiyaKayFabricBsdfFactory factory(sRoughness[i]);

        // The integrals are evaluated using uniform sampling which makes it an
        // error-prone calculation. Ideally, we'd have higher tolerance for sharper lobes.
        // Right now we set it up to 'fail' only for higher roughness values.
        TestBsdfSettings test(factory, frame, true, 0.002, 0.03,
                              (sRoughness[i] > 0.1f  &&  sRoughness[i] < 0.7f),
                              (sRoughness[i] > 0.1f), TestBsdfSettings::BSDF_ONE_SAMPLER);
        runTest(test, sViewAnglesTheta, 1, getSampleCount(sRoughness[i]));
    }
}

//----------------------------------------------------------------------------

void
TestBsdfOneSampler::testAshikminhShirley()
{
    scene_rdl2::math::ReferenceFrame frame;

    printInfo("##### TestBsdfOneSampler::testAshikminhShirley() ####################");
    for (int i=0; i < sRoughnessCount; i++) {
        printInfo("===== Roughness: %f ====================", sRoughness[i]);

        AshikminhShirleyBsdfFactory factory(sRoughness[i]);
        TestBsdfSettings test(factory, frame, true, 0.008, 0.03,
                (sRoughness[i] > 0.1f), (sRoughness[i] > 0.1f),
                TestBsdfSettings::BSDF_ONE_SAMPLER);
        runTest(test, sViewAnglesTheta, 1, getSampleCount(sRoughness[i]));
    }
}


void
TestBsdfOneSampler::testAshikminhShirleyFull()
{
    scene_rdl2::math::ReferenceFrame frame;

    printInfo("##### TestBsdfOneSampler::testAshikminhShirleyFull() ####################");
    for (int i=0; i < sRoughnessCount; i++) {

        // TODO: There is a heisenbug in here where the consistency check fails
        // with small roughnesses. It stops failing when trying to debug (?!?)
        // Since we don't really use AshikminhShirley I am skipping this
        // here for now
        if (sRoughness[i] < 0.3f) {
            continue;
        }

        printInfo("===== Roughness: %f ====================", sRoughness[i]);

        int sampleCount = getSampleCount(sRoughness[i]) +
                          getSampleCount(sRoughness[sRoughnessCount - 1]);

        AshikminhShirleyFullBsdfFactory factory(sRoughness[i]);
        TestBsdfSettings test(factory, frame, true, 0.008, 0.02,
                (sRoughness[i] > 0.1f),
                (sRoughness[i] > 0.1f), TestBsdfSettings::BSDF_ONE_SAMPLER);
        runTest(test, sViewAnglesTheta, 1, sampleCount);
    }
}


void
TestBsdfOneSampler::testWardCorrected()
{
    scene_rdl2::math::ReferenceFrame frame;

    printInfo("##### TestBsdfOneSampler::testWardCorrected() ####################");
    for (int i=0; i < sRoughnessCount; i++) {
        printInfo("===== Roughness: %f ====================", sRoughness[i]);

        WardCorrectedBsdfFactory factory(sRoughness[i]);
        TestBsdfSettings test(factory, frame, true, 0.008, 0.03,
                (sRoughness[i] > 0.1f  &&  sRoughness[i] < 0.7f),
                (sRoughness[i] > 0.1f), TestBsdfSettings::BSDF_ONE_SAMPLER);
        runTest(test, sViewAnglesTheta, 1, getSampleCount(sRoughness[i]));
    }
}


void
TestBsdfOneSampler::testWardDuer()
{
    scene_rdl2::math::ReferenceFrame frame;

    printInfo("##### TestBsdfOneSampler::testWardDuer() ####################");
    for (int i=0; i < sRoughnessCount; i++) {
        printInfo("===== Roughness: %f ====================", sRoughness[i]);

        WardDuerBsdfFactory factory(sRoughness[i]);
        TestBsdfSettings test(factory, frame, true, 0.002, 0.03,
                // TODO: Why does the pdf integrate to values > 1 ?!?!
                false, (sRoughness[i] > 0.1f), TestBsdfSettings::BSDF_ONE_SAMPLER);
        runTest(test, sViewAnglesTheta, 1, getSampleCount(sRoughness[i]));
    }
}


//----------------------------------------------------------------------------

void
TestBsdfOneSampler::testHairDiffuse()
{
    scene_rdl2::math::ReferenceFrame frame;

    printInfo("##### TestBsdfOneSampler::testHairDiffuse() ####################");
    HairDiffuseBsdfFactory factory;
    TestBsdfSettings test(factory, frame, true, 0.002, 0.01, true, true,
            TestBsdfSettings::BSDF_ONE_SAMPLER);
    runTest(test, sViewAnglesTheta, 1, getSampleCount(sRoughness[sRoughnessCount - 1]));
}

void
TestBsdfOneSampler::testHairR()
{
    scene_rdl2::math::ReferenceFrame frame;

    // running into some numerical issues where
    // consistency tests fail intermittently so increasing the
    // threshold to 0.08
    static const float toleranceConsistency = 0.08f;
    static const float toleranceIntegrals   = 0.06f;
    printInfo("##### TestBsdfOneSampler::testHairR() ####################");
    for (int i=0; i < sRoughnessCount; i++) {
        printInfo("===== Roughness: %f ====================", sRoughness[i]);

        HairRBsdfFactory factory(sRoughness[i], 0.0f);
        TestBsdfSettings test(factory,
                              frame,
                              false,
                              toleranceConsistency,
                              toleranceIntegrals,
                              (sRoughness[i] > 0.1f),
                              true,
                              TestBsdfSettings::BSDF_ONE_SAMPLER);
        runTest(test, sViewAnglesTheta, 1, getSampleCount(sRoughness[i]));
    }
}

void
TestBsdfOneSampler::testHairTT()
{
    scene_rdl2::math::ReferenceFrame frame;

    // running into some numerical issues where
    // consistency tests fail intermittently so increasing the
    // threshold to 0.08
    static const float toleranceConsistency = 0.08f;
    static const float toleranceIntegrals   = 0.06f;
    printInfo("##### TestBsdfOneSampler::testHairTT() ####################");
    for (int i=0; i < sRoughnessCount; i++) {
        printInfo("===== Roughness: %f ====================", sRoughness[i]);

        HairTTBsdfFactory factory(sRoughness[i],
                                  0.5f,
                                  0.0f);
        TestBsdfSettings test(factory,
                              frame,
                              false,
                              toleranceConsistency,
                              toleranceIntegrals,
                              (sRoughness[i] > 0.1f),
                              true,
                              TestBsdfSettings::BSDF_ONE_SAMPLER);
        runTest(test, sViewAnglesTheta, 1, getSampleCount(sRoughness[i]));
    }
}

//----------------------------------------------------------------------------

void
TestBsdfOneSampler::testTwoLobes()
{
    scene_rdl2::math::ReferenceFrame frame;

    printInfo("##### TestBsdfOneSampler::testTwoLobes() ####################");
    for (int i=0; i < sRoughnessCount; i++) {
        printInfo("===== Roughness: %f ====================", sRoughness[i]);

        int sampleCount = getSampleCount(sRoughness[i]) +
                          getSampleCount(sRoughness[sRoughnessCount - 1]);

        TwoLobeBsdfFactory factory(sRoughness[i]);
        TestBsdfSettings test(factory, frame, true, 0.002, 0.02,
                (sRoughness[i] > 0.1f  &&  sRoughness[i] < 0.7f),
                (sRoughness[i] > 0.1f), TestBsdfSettings::BSDF_ONE_SAMPLER);
        runTest(test, sViewAnglesTheta, 1, sampleCount);
    }
}


void
TestBsdfOneSampler::testThreeLobes()
{
    scene_rdl2::math::ReferenceFrame frame;

    printInfo("##### TestBsdfOneSampler::testThreeLobes() ####################");
    for (int i=0; i < sRoughnessCount; i++) {
        int j = (i + 4) % sRoughnessCount;
        printInfo("===== Roughness: %f and %f ====================",
                sRoughness[i], sRoughness[j]);

        int sampleCount = getSampleCount(sRoughness[i]) +
                          getSampleCount(sRoughness[j]) +
                          getSampleCount(sRoughness[sRoughnessCount - 1]);

        ThreeLobeBsdfFactory factory(sRoughness[i], sRoughness[j]);
        TestBsdfSettings test(factory, frame, true, 0.002, 0.02,
                (sRoughness[i] > 0.1f  &&  sRoughness[j] > 0.1f  &&
                 sRoughness[i] < 0.7f  &&  sRoughness[j] < 0.7f),
                true, TestBsdfSettings::BSDF_ONE_SAMPLER);
        runTest(test, sViewAnglesTheta, 1, sampleCount);
    }
}

//----------------------------------------------------------------------------

void
TestBsdfOneSampler::testStochasticFlakes()
{
    scene_rdl2::math::ReferenceFrame frame;

    static const size_t flakesDataSize = 100;

    printInfo("##### TestBsdfOneSampler::testStochasticFlakes() ####################");
    for (int i = 0; i < sRoughnessCount; i++) {
        printInfo("===== Roughness: %f ====================", sRoughness[i]);

        Vec3Array normals;
        ColorArray colors;
        generateGGXNormals(sRoughness[i] * sRoughness[i], flakesDataSize, normals);
        generateWeightedFlakeColors(flakesDataSize, colors);
        for (size_t j = 0; j < flakesDataSize; ++j) {
            normals[j] = frame.localToGlobal(normals[j]);
        }

        StochasticFlakesBsdfFactory factory(&normals[0], &colors[0], flakesDataSize, 0.1f, sRoughness[i]);
        TestBsdfSettings test(factory, frame, true, 0.008, 0.03,
                              (sRoughness[i] > 0.1f  &&  sRoughness[i] < 0.4f),
                              (sRoughness[i] > 0.1f  &&  sRoughness[i] < 0.4f),
                              TestBsdfSettings::BSDF_ONE_SAMPLER);
        runTest(test, sViewAnglesTheta, 1, getSampleCount(sRoughness[i]));

        normals.clear();
        colors.clear();
    }
}


//----------------------------------------------------------------------------

} // namespace pbr
} // namespace moonray


CPPUNIT_TEST_SUITE_REGISTRATION(moonray::pbr::TestBsdfOneSampler);


