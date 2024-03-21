// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

///
/// @file TestBsdfSampler.cc
/// $Id$
///


#include "BsdfFactory.h"
#include "TestBsdfCommon.h"
#include "TestBsdfSampler.h"
#include "TestUtil.h"
#include <moonray/rendering/pbr/core/PbrTLState.h>
#include <moonray/rendering/pbr/integrator/BsdfSampler.h>
#include <moonray/rendering/pbr/integrator/PathGuide.h>

#include <moonray/rendering/pbr/core/PbrTLState.h>
#include <moonray/rendering/shading/bsdf/Bsdf.h>
#include <moonray/rendering/shading/bsdf/BsdfSlice.h>

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
//    printInfo("Input sample count: %d", sampleCount);

    mcrt_common::ThreadLocalState *tls = mcrt_common::getFrameUpdateTLS();
    pbr::TLState *pbrTls = MNRY_VERIFY(tls->mPbrTls.get());
    scene_rdl2::alloc::Arena *arena = &tls->mArena;

    {
        SCOPED_MEM(arena);
        shading::Bsdf *bsdf = test.bsdfFactory(*arena, test.frame);
        shading::BsdfSlice slice(test.frame.getN(), test.frame.getN(), true, true,
            ispc::SHADOW_TERMINATOR_FIX_OFF);
        const PathGuide pg;
        BsdfSampler sampler(pbrTls->mArena, *bsdf, slice, test.sMaxSamplesPerLobe,
                    true, pg);
        sampleCount /= sampler.getSampleCount();
    }

//    printInfo("Sampler sample count: %d", sampler.getSampleCount());
//    printInfo("Adjusted sample count: %d", sampleCount);

    runBsdfTest(test, viewAnglesTheta, viewAnglesPhy, sampleCount);
}

//----------------------------------------------------------------------------

void TestBsdfSampler::setUp()
{
    setupThreadLocalData();
}

void TestBsdfSampler::tearDown()
{
    cleanupThreadLocalData();
}

TestBsdfSampler::TestBsdfSampler()
{
}

TestBsdfSampler::~TestBsdfSampler()
{
}


//----------------------------------------------------------------------------

void
TestBsdfSampler::testLambert()
{
    scene_rdl2::math::ReferenceFrame frame;

    printInfo("##### TestBsdfSampler::testLambert() ####################");
    LambertBsdfFactory factory;
    TestBsdfSettings test(factory, frame, true, 0.002, 0.01, true, true,
                          TestBsdfSettings::BSDF_SAMPLER);
    runTest(test, sViewAnglesTheta, 1, getSampleCount(sRoughness[sRoughnessCount - 1]));
}


//----------------------------------------------------------------------------

void
TestBsdfSampler::testCookTorrance()
{
    scene_rdl2::math::ReferenceFrame frame;

    printInfo("##### TestBsdfSampler::testCookTorrance() ####################");
    for (int i=0; i < sRoughnessCount; i++) {
        printInfo("===== Roughness: %f ====================", sRoughness[i]);

        CookTorranceBsdfFactory factory(sRoughness[i]);
        TestBsdfSettings test(factory, frame, true, 0.002, 0.03,
                (sRoughness[i] > 0.1f  &&  sRoughness[i] < 0.7f),
                (sRoughness[i] > 0.1f), TestBsdfSettings::BSDF_SAMPLER);
        runTest(test, sViewAnglesTheta, 1, getSampleCount(sRoughness[i]));
    }
}


void
TestBsdfSampler::testGGXCookTorrance()
{
    scene_rdl2::math::ReferenceFrame frame;

    printInfo("##### TestBsdfSampler::testGGXCookTorrance() ####################");
    for (int i=0; i < sRoughnessCount; i++) {
        printInfo("===== Roughness: %f ====================", sRoughness[i]);

        GGXCookTorranceBsdfFactory factory(sRoughness[i]);
        TestBsdfSettings test(factory, frame, true, 0.002, 0.03,
                (sRoughness[i] > 0.1f  &&  sRoughness[i] < 0.4f),
                (sRoughness[i] > 0.1f  &&  sRoughness[i] < 0.4f),
                TestBsdfSettings::BSDF_SAMPLER);
        runTest(test, sViewAnglesTheta, 1, getSampleCount(sRoughness[i]));
    }
}


void
TestBsdfSampler::testAnisoCookTorrance()
{
    scene_rdl2::math::ReferenceFrame frame;

    printInfo("##### TestBsdfSampler::testAnisoCookTorrance() ####################");
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
                TestBsdfSettings::BSDF_SAMPLER);
        runTest(test, sViewAnglesTheta, 1, sampleCount);
    }
}


void
TestBsdfSampler::testTransmissionCookTorrance()
{
    scene_rdl2::math::ReferenceFrame frame;

    printInfo("##### TestBsdfSampler::testTransmissionCookTorrance() ####################");
    for (int i=0; i < sRoughnessCount; i++) {
        printInfo("===== Roughness: %f ====================", sRoughness[i]);

        TransmissionCookTorranceBsdfFactory factory(sRoughness[i]);
        TestBsdfSettings test(factory, frame, false, 0.002, 0.05,
                (sRoughness[i] > 0.2f  &&  sRoughness[i] < 0.7f),
                (sRoughness[i] > 0.2f), TestBsdfSettings::BSDF_SAMPLER);
        runTest(test, sViewAnglesTheta, 1, 4 * getSampleCount(sRoughness[i]));
    }
}


void
TestBsdfSampler::testIridescence()
{
    scene_rdl2::math::ReferenceFrame frame;

    printInfo("##### TestBsdfSampler::testIridescence() ####################");
    for (int i=0; i < sRoughnessCount; i++) {
        printInfo("===== Roughness: %f ====================", sRoughness[i]);

        IridescenceBsdfFactory factory(sRoughness[i], 1.0f, ispc::SHADING_IRIDESCENCE_COLOR_USE_HUE_INTERPOLATION,
                scene_rdl2::math::Color(1.f, 0.f, 0.f), scene_rdl2::math::Color(1.f, 0.f, 1.f), false,
                ispc::COLOR_RAMP_CONTROL_SPACE_RGB, 0, nullptr, nullptr, nullptr,
                2.f, 1.f, 1.f, 1.f);
        TestBsdfSettings test(factory, frame, true, 0.002, 0.04,
                (sRoughness[i] > 0.1  &&  sRoughness[i] < 0.5),
                (sRoughness[i] > 0.1), TestBsdfSettings::BSDF_SAMPLER);
        runTest(test, sViewAnglesTheta, 1, getSampleCount(sRoughness[i]));
    }
}


//----------------------------------------------------------------------------

void
TestBsdfSampler::testRetroreflection()
{
    scene_rdl2::math::ReferenceFrame frame;

    printInfo("##### TestBsdfSampler::testRetroreflection() ####################");
    for (int i=0; i < sRoughnessCount; i++) {
        printInfo("===== Roughness: %f ====================", sRoughness[i]);

        RetroreflectionBsdfFactory factory(sRoughness[i]);
        // The integrals are evaluated using uniform sampling which makes it an
        // error-prone calculation. Ideally, we'd have higher tolerance for sharper lobes.
        // Right now we set it up to 'fail' only for higher roughness values.
        TestBsdfSettings test(factory, frame, true, 0.008, 0.08,
                              sRoughness[i] > 0.1f, sRoughness[i] > 0.1f,
                              TestBsdfSettings::BSDF_SAMPLER);
        runTest(test, sViewAnglesTheta, 1, getSampleCount(sRoughness[i]));
    }
}

//----------------------------------------------------------------------------

void
TestBsdfSampler::testEyeCaustic()
{
    scene_rdl2::math::ReferenceFrame frame;

    printInfo("##### TestBsdfSampler::testEyeCaustic() ####################");
    for (int i=0; i < sRoughnessCount; i++) {
        printInfo("===== Roughness: %f ====================", sRoughness[i]);

        EyeCausticBsdfFactory factory(sRoughness[i]);
        // The integrals are evaluated using uniform sampling which makes it an
        // error-prone calculation. Ideally, we'd have higher tolerance for sharper lobes.
        // Right now we set it up to 'fail' only for higher roughness values.
        TestBsdfSettings test(factory, frame, false, 0.002, 0.04,
                              sRoughness[i] > 0.1f, sRoughness[i] > 0.1f,
                              TestBsdfSettings::BSDF_SAMPLER);
        runTest(test, sViewAnglesTheta, 1, getSampleCount(sRoughness[i]));
    }
}

//----------------------------------------------------------------------------

void
TestBsdfSampler::testDwaFabric()
{
    scene_rdl2::math::ReferenceFrame frame;

    printInfo("##### TestBsdfSampler::testDwaFabric() ####################");
    for (int i=0; i < sRoughnessCount; i++) {
        printInfo("===== Roughness: %f ====================", sRoughness[i]);

        DwaFabricBsdfFactory factory(sRoughness[i]);
        // The integrals are evaluated using uniform sampling which makes it an
        // error-prone calculation. Ideally, we'd have higher tolerance for sharper lobes.
        // Right now we set it up to 'fail' only for higher roughness values.
        TestBsdfSettings test(factory, frame, true, 0.002, 0.04,
                (sRoughness[i] > 0.1  &&  sRoughness[i] < 0.5),
                (sRoughness[i] > 0.1), TestBsdfSettings::BSDF_SAMPLER);
        runTest(test, sViewAnglesTheta, 1, getSampleCount(sRoughness[i]));
    }
}

//----------------------------------------------------------------------------

void
TestBsdfSampler::testKajiyaKayFabric()
{
    scene_rdl2::math::ReferenceFrame frame;

    printInfo("##### TestBsdfSampler::testKajiyaKayFabric() ####################");
    for (int i=0; i < sRoughnessCount; i++) {
        printInfo("===== Roughness: %f ====================", sRoughness[i]);

        KajiyaKayFabricBsdfFactory factory(sRoughness[i]);
        // The integrals are evaluated using uniform sampling which makes it an
        // error-prone calculation. Ideally, we'd have higher tolerance for sharper lobes.
        // Right now we set it up to 'fail' only for higher roughness values.
        TestBsdfSettings test(factory, frame, true, 0.002, 0.04,
                (sRoughness[i] > 0.1  &&  sRoughness[i] < 0.5),
                (sRoughness[i] > 0.1), TestBsdfSettings::BSDF_SAMPLER);
        runTest(test, sViewAnglesTheta, 1, getSampleCount(sRoughness[i]));
    }
}

//----------------------------------------------------------------------------

void
TestBsdfSampler::testAshikminhShirley()
{
    scene_rdl2::math::ReferenceFrame frame;

    printInfo("##### TestBsdfSampler::testAshikminhShirley() ####################");
    for (int i=0; i < sRoughnessCount; i++) {
        printInfo("===== Roughness: %f ====================", sRoughness[i]);

        AshikminhShirleyBsdfFactory factory(sRoughness[i]);
        TestBsdfSettings test(factory, frame, true, 0.008, 0.02,
                (sRoughness[i] > 0.1f), (sRoughness[i] > 0.1f), TestBsdfSettings::BSDF_SAMPLER);
        runTest(test, sViewAnglesTheta, 1, getSampleCount(sRoughness[i]));
    }
}


void
TestBsdfSampler::testAshikminhShirleyFull()
{
    scene_rdl2::math::ReferenceFrame frame;

    printInfo("##### TestBsdfSampler::testAshikminhShirleyFull() ####################");
    for (int i=0; i < sRoughnessCount; i++) {
        printInfo("===== Roughness: %f ====================", sRoughness[i]);

        int sampleCount = getSampleCount(sRoughness[i]) +
                          getSampleCount(sRoughness[sRoughnessCount - 1]);

        AshikminhShirleyFullBsdfFactory factory(sRoughness[i]);
        TestBsdfSettings test(factory, frame, true, 0.008, 0.02,
                (sRoughness[i] > 0.1f),
                (sRoughness[i] > 0.1f), TestBsdfSettings::BSDF_SAMPLER);
        runTest(test, sViewAnglesTheta, 1, sampleCount);
    }
}


//----------------------------------------------------------------------------

void
TestBsdfSampler::testWardCorrected()
{
    scene_rdl2::math::ReferenceFrame frame;

    printInfo("##### TestBsdfSampler::testWardCorrected() ####################");
    for (int i=0; i < sRoughnessCount; i++) {
        printInfo("===== Roughness: %f ====================", sRoughness[i]);

        WardCorrectedBsdfFactory factory(sRoughness[i]);
        TestBsdfSettings test(factory, frame, true, 0.008, 0.02,
                (sRoughness[i] > 0.1f  &&  sRoughness[i] < 0.7f),
                (sRoughness[i] > 0.1f), TestBsdfSettings::BSDF_SAMPLER);
        runTest(test, sViewAnglesTheta, 1, getSampleCount(sRoughness[i]));
    }
}


void
TestBsdfSampler::testWardDuer()
{
    scene_rdl2::math::ReferenceFrame frame;

    printInfo("##### TestBsdfSampler::testWardDuer() ####################");
    for (int i=0; i < sRoughnessCount; i++) {
        printInfo("===== Roughness: %f ====================", sRoughness[i]);

        WardDuerBsdfFactory factory(sRoughness[i]);
        TestBsdfSettings test(factory, frame, true, 0.002, 0.03,
                // TODO: Why does the pdf integrate to values > 1 ?!?!
                false, (sRoughness[i] > 0.1f), TestBsdfSettings::BSDF_SAMPLER);
        runTest(test, sViewAnglesTheta, 1, getSampleCount(sRoughness[i]));
    }
}


//----------------------------------------------------------------------------

void
TestBsdfSampler::testHairDiffuse()
{
    scene_rdl2::math::ReferenceFrame frame;

    printInfo("##### TestBsdfSampler::testHairDiffuse() ####################");
    HairDiffuseBsdfFactory factory;
    TestBsdfSettings test(factory, frame, true, 0.002, 0.01, true, true,
            TestBsdfSettings::BSDF_SAMPLER);
    runTest(test, sViewAnglesTheta, 1, getSampleCount(sRoughness[sRoughnessCount - 1]));
}

void
TestBsdfSampler::testHairR()
{
    scene_rdl2::math::ReferenceFrame frame;

    // running into some numerical issues where
    // consistency tests fail intermittently so increasing the
    // threshold to 0.08
    static const float toleranceConsistency = 0.08f;
    static const float toleranceIntegrals   = 0.06f;
    printInfo("##### TestBsdfSampler::testHairR() ####################");
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
                              TestBsdfSettings::BSDF_SAMPLER);
        runTest(test, sViewAnglesTheta, 1, getSampleCount(sRoughness[i]));
    }
}

void
TestBsdfSampler::testHairTT()
{
    scene_rdl2::math::ReferenceFrame frame;

    // running into some numerical issues where
    // consistency tests fail intermittently so increasing the
    // threshold to 0.08
    static const float toleranceConsistency = 0.08f;
    static const float toleranceIntegrals   = 0.06f;
    printInfo("##### TestBsdfSampler::testHairTT() ####################");
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
                              TestBsdfSettings::BSDF_SAMPLER);
        runTest(test, sViewAnglesTheta, 1, getSampleCount(sRoughness[i]));
    }
}

//----------------------------------------------------------------------------

void
TestBsdfSampler::testTwoLobes()
{
    scene_rdl2::math::ReferenceFrame frame;

    printInfo("##### TestBsdfSampler::testTwoLobes() ####################");
    for (int i=0; i < sRoughnessCount; i++) {
        printInfo("===== Roughness: %f ====================", sRoughness[i]);

        int sampleCount = getSampleCount(sRoughness[i]) +
                          getSampleCount(sRoughness[sRoughnessCount - 1]);

        TwoLobeBsdfFactory factory(sRoughness[i]);
        TestBsdfSettings test(factory, frame, true, 0.002, 0.02,
                (sRoughness[i] > 0.1f  &&  sRoughness[i] < 0.7f),
                (sRoughness[i] > 0.1f), TestBsdfSettings::BSDF_SAMPLER);
        runTest(test, sViewAnglesTheta, 1, sampleCount);
    }
}


void
TestBsdfSampler::testThreeLobes()
{
    scene_rdl2::math::ReferenceFrame frame;

    printInfo("##### TestBsdfSampler::testThreeLobes() ####################");
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
                 true, TestBsdfSettings::BSDF_SAMPLER);
        runTest(test, sViewAnglesTheta, 1, sampleCount);
    }
}

//----------------------------------------------------------------------------

void
TestBsdfSampler::testStochasticFlakes()
{
    scene_rdl2::math::ReferenceFrame frame;

    static const size_t flakesDataSize = 100;

    printInfo("##### TestBsdfSampler::testStochasticFlakes() ####################");
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
                              TestBsdfSettings::BSDF_SAMPLER);
        runTest(test, sViewAnglesTheta, 1, getSampleCount(sRoughness[i]));

        normals.clear();
        colors.clear();
    }
}

//----------------------------------------------------------------------------

void
TestBsdfSampler::testUnderClearcoatLambert()
{
    scene_rdl2::math::ReferenceFrame frame;
    const std::vector<float> refrIndices= {1.1f, 1.2f, 1.5f, 1.9f};

    printInfo("##### TestBsdfSampler::testUnderClearcoatLambert() ####################");
    for (const auto ior : refrIndices) {
        printInfo("===== IOR: %f ====================", ior);
        UnderClearcoatBsdfFactory factory(frame.getN(), 1.0f, ior,
                                          0.0f, scene_rdl2::math::sWhite, 1.0f,
                                          UnderClearcoatBsdfFactory::UnderlobeType::Lambert);
        TestBsdfSettings test(factory,
                              frame,
                              false, // test reciprocity
                              0.02f, // tolerance consistency
                              0.08f, // tolerance integral
                              true, // assert pdf integral
                              true, // assert eval integral
                              TestBsdfSettings::BSDF_SAMPLER);
        runTest(test, sViewAnglesTheta, 1, getSampleCount(sRoughness[sRoughnessCount - 1]));
    }
}

void
TestBsdfSampler::testUnderClearcoatCookTorrance()
{
    scene_rdl2::math::ReferenceFrame frame;
    const std::vector<float> refrIndices= {1.1f, 1.2f, 1.5f, 1.9f};

    printInfo("##### TestBsdfSampler::testUnderClearcoatCookTorrance() ####################");
    for (const auto ior : refrIndices) {
        printInfo("===== IOR: %f ====================", ior);
        UnderClearcoatBsdfFactory factory(frame.getN(), 1.0f, ior,
                                          0.0f, scene_rdl2::math::sWhite, 1.0f,
                                          UnderClearcoatBsdfFactory::UnderlobeType::CookTorrance);

        for (int i=0; i < sRoughnessCount; i++) {
            printInfo("===== Roughness: %f ====================", sRoughness[i]);
            factory.setRoughness(sRoughness[i]);
            TestBsdfSettings test(factory,
                                  frame,
                                  false, // test reciprocity
                                  0.05f, // tolerance consistency
                                  0.08f, // tolerance integral
                                  (sRoughness[i] > 0.1f  &&  sRoughness[i] < 0.4f),
                                  (sRoughness[i] > 0.1f),
                                  TestBsdfSettings::BSDF_SAMPLER);
            runTest(test, sViewAnglesTheta, 1, getSampleCount(sRoughness[i]));
        }
    }
}
//----------------------------------------------------------------------------

} // namespace pbr
} // namespace moonray


CPPUNIT_TEST_SUITE_REGISTRATION(moonray::pbr::TestBsdfSampler);


