// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

/// @file TestBsdfv.cc

#include "BsdfFactory.h"
#include "TestBsdfCommon.h"
#include "TestBsdfv.h"
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
    //printInfo("Sample count: %d", sampleCount);

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

void TestBsdfv::setUp()
{
    setupThreadLocalData();
}

void TestBsdfv::tearDown()
{
    cleanupThreadLocalData();
}

TestBsdfv::TestBsdfv()
{
}

TestBsdfv::~TestBsdfv()
{
}


//----------------------------------------------------------------------------

void
TestBsdfv::testLambert()
{
    scene_rdl2::math::ReferenceFrame frame;

    printInfo("##### TestBsdfv::testLambert() ####################");
    LambertBsdfFactory factory;
    TestBsdfSettings test(factory, frame, true, 0.002, 0.01, true, true,
            TestBsdfSettings::BSDFV);
    runTest(test, sViewAnglesTheta, 1, getSampleCount(sRoughness[sRoughnessCount - 1]));
}


//----------------------------------------------------------------------------

void
TestBsdfv::testCookTorrance()
{
    scene_rdl2::math::ReferenceFrame frame;

    printInfo("##### TestBsdfv::testCookTorrance() ####################");
    for (int i=0; i < sRoughnessCount; i++) {
        printInfo("===== Roughness: %f ====================", sRoughness[i]);

        CookTorranceBsdfFactory factory(sRoughness[i]);
        TestBsdfSettings test(factory, frame, true, 0.002, 0.03,
                (sRoughness[i] > 0.1f  &&  sRoughness[i] < 0.7f),
                (sRoughness[i] > 0.1f), TestBsdfSettings::BSDFV);
        runTest(test, sViewAnglesTheta, 1, getSampleCount(sRoughness[i]));
    }
}


void
TestBsdfv::testGGXCookTorrance()
{
    scene_rdl2::math::ReferenceFrame frame;

    printInfo("##### TestBsdfv::testGGXCookTorrance() ####################");
    for (int i=0; i < sRoughnessCount; i++) {
        printInfo("===== Roughness: %f ====================", sRoughness[i]);

        GGXCookTorranceBsdfFactory factory(sRoughness[i]);
        TestBsdfSettings test(factory, frame, true, 0.002, 0.03,
                (sRoughness[i] > 0.1f  &&  sRoughness[i] < 0.4f),
                (sRoughness[i] > 0.1f  &&  sRoughness[i] < 0.4f), TestBsdfSettings::BSDFV);
        runTest(test, sViewAnglesTheta, 1, getSampleCount(sRoughness[i]));
    }
}


void
TestBsdfv::testAnisoCookTorrance()
{
    scene_rdl2::math::ReferenceFrame frame;

    printInfo("##### TestBsdfv::testAnisoCookTorrance() ####################");
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
                TestBsdfSettings::BSDFV);
        runTest(test, sViewAnglesTheta, 1, sampleCount);
    }
}

void
TestBsdfv::testTransmissionCookTorrance()
{
    scene_rdl2::math::ReferenceFrame frame;

    printInfo("##### TestBsdfv::testTransmissionCookTorrance() ####################");
    for (int i=0; i < sRoughnessCount; i++) {
        printInfo("===== Roughness: %f ====================", sRoughness[i]);

        TransmissionCookTorranceBsdfFactory factory(sRoughness[i]);
        TestBsdfSettings test(factory, frame, false, 0.002, 0.05,
                (sRoughness[i] > 0.2f  &&  sRoughness[i] < 0.7f),
                (sRoughness[i] > 0.2f), TestBsdfSettings::BSDFV);
        runTest(test, sViewAnglesTheta, 1, 4 * getSampleCount(sRoughness[i]));
    }
}

void
TestBsdfv::testIridescence()
{
    scene_rdl2::math::ReferenceFrame frame;

    static const int size = 2;
    const float positions[size] = {0.f, 1.f};
    const ispc::RampInterpolatorMode interpolators[size] = {
        ispc::RAMP_INTERPOLATOR_MODE_SMOOTH, 
        ispc::RAMP_INTERPOLATOR_MODE_SMOOTH };
    const scene_rdl2::math::Color colors[size] = {scene_rdl2::math::Color(1.f, 0.f, 0.f), scene_rdl2::math::Color(1.f, 0.f, 1.f)};

    printInfo("##### TestBsdfv::testIridescence() ####################");
    for (int i=0; i < sRoughnessCount; i++) {
        printInfo("===== Roughness: %f ====================", sRoughness[i]);

        IridescenceBsdfFactory factory(sRoughness[i], 1.0f, ispc::SHADING_IRIDESCENCE_COLOR_USE_HUE_INTERPOLATION,
                scene_rdl2::math::Color(1.f, 0.f, 0.f), scene_rdl2::math::Color(1.f, 0.f, 1.f), false,
                ispc::COLOR_RAMP_CONTROL_SPACE_RGB, size, positions, interpolators, colors,
                2.f, 1.f, 1.f, 1.f);
        TestBsdfSettings test(factory, frame, true, 0.002, 0.06,
                (sRoughness[i] > 0.1  &&  sRoughness[i] < 0.5),
                (sRoughness[i] > 0.1), TestBsdfSettings::BSDFV);
        runTest(test, sViewAnglesTheta, 1, getSampleCount(sRoughness[i]));
    }
}


//----------------------------------------------------------------------------

void
TestBsdfv::testRetroreflection()
{
    scene_rdl2::math::ReferenceFrame frame;

    printInfo("##### TestBsdfv::testRetroreflection() ####################");
    for (int i=0; i < sRoughnessCount; i++) {
        printInfo("===== Roughness: %f ====================", sRoughness[i]);

        RetroreflectionBsdfFactory factory(sRoughness[i]);
        // The integrals are evaluated using uniform sampling which makes it an
        // error-prone calculation. Ideally, we'd have higher tolerance for sharper lobes.
        // Right now we set it up to 'fail' only for higher roughness values.
        TestBsdfSettings test(factory, frame, true, 0.002, 0.08,
                              (sRoughness[i] > 0.1f  &&  sRoughness[i] < 0.7f),
                              (sRoughness[i] > 0.1f), TestBsdfSettings::BSDFV);
        runTest(test, sViewAnglesTheta, 1, getSampleCount(sRoughness[i]));
    }
}

//----------------------------------------------------------------------------

void
TestBsdfv::testEyeCaustic()
{
    scene_rdl2::math::ReferenceFrame frame;

    printInfo("##### TestBsdfv::testEyeCaustic() ####################");
    for (int i=0; i < sRoughnessCount; i++) {
        printInfo("===== Roughness: %f ====================", sRoughness[i]);

        EyeCausticBsdfFactory factory(sRoughness[i]);
        // The integrals are evaluated using uniform sampling which makes it an
        // error-prone calculation. Ideally, we'd have higher tolerance for sharper lobes.
        // Right now we set it up to 'fail' only for higher roughness values.
        TestBsdfSettings test(factory, frame, false, 0.002, 0.04,
                              (sRoughness[i] > 0.1f  &&  sRoughness[i] < 0.7f),
                              (sRoughness[i] > 0.1f), TestBsdfSettings::BSDFV);
        runTest(test, sViewAnglesTheta, 1, getSampleCount(sRoughness[i]));
    }
}

//----------------------------------------------------------------------------

void
TestBsdfv::testDwaFabric()
{
    scene_rdl2::math::ReferenceFrame frame;

    printInfo("##### TestBsdfv::testDwaFabric() ####################");
    for (int i=0; i < sRoughnessCount; i++) {
        printInfo("===== Roughness: %f ====================", sRoughness[i]);
        DwaFabricBsdfFactory factory(sRoughness[i]);
        // The integrals are evaluated using uniform sampling which makes it an
        // error-prone calculation. Ideally, we'd have higher tolerance for sharper lobes.
        // Right now we set it up to 'fail' only for higher roughness values.
        TestBsdfSettings test(factory, frame, true, 0.002, 0.03,
                              (sRoughness[i] > 0.1f  &&  sRoughness[i] < 0.7f),
                              (sRoughness[i] > 0.1f), TestBsdfSettings::BSDFV);
        runTest(test, sViewAnglesTheta, 1, getSampleCount(sRoughness[i]));
    }
}

//----------------------------------------------------------------------------

void
TestBsdfv::testKajiyaKayFabric()
{
    scene_rdl2::math::ReferenceFrame frame;

    printInfo("##### TestBsdfv::testKajiyaKayFabric() ####################");
    for (int i=0; i < sRoughnessCount; i++) {
        printInfo("===== Roughness: %f ====================", sRoughness[i]);
        KajiyaKayFabricBsdfFactory factory(sRoughness[i]);
        // The integrals are evaluated using uniform sampling which makes it an
        // error-prone calculation. Ideally, we'd have higher tolerance for sharper lobes.
        // Right now we set it up to 'fail' only for higher roughness values.
        TestBsdfSettings test(factory, frame, true, 0.002, 0.03,
                              (sRoughness[i] > 0.1f  &&  sRoughness[i] < 0.7f),
                              (sRoughness[i] > 0.1f), TestBsdfSettings::BSDFV);
        runTest(test, sViewAnglesTheta, 1, getSampleCount(sRoughness[i]));
    }
}

//----------------------------------------------------------------------------

void
TestBsdfv::testAshikminhShirley()
{
    scene_rdl2::math::ReferenceFrame frame;

    printInfo("##### TestBsdfv::testAshikminhShirley() ####################");
    for (int i=0; i < sRoughnessCount; i++) {
        printInfo("===== Roughness: %f ====================", sRoughness[i]);

        AshikminhShirleyBsdfFactory factory(sRoughness[i]);
        TestBsdfSettings test(factory, frame, true, 0.008, 0.03,
                (sRoughness[i] > 0.1f), (sRoughness[i] > 0.1f), TestBsdfSettings::BSDFV);
        runTest(test, sViewAnglesTheta, 1, getSampleCount(sRoughness[i]));
    }
}


void
TestBsdfv::testAshikminhShirleyFull()
{
    scene_rdl2::math::ReferenceFrame frame;

    printInfo("##### TestBsdfv::testAshikminhShirleyFull() ####################");
    for (int i=0; i < sRoughnessCount; i++) {
        printInfo("===== Roughness: %f ====================", sRoughness[i]);

        int sampleCount = getSampleCount(sRoughness[i]) +
                          getSampleCount(sRoughness[sRoughnessCount - 1]);

        AshikminhShirleyFullBsdfFactory factory(sRoughness[i]);
        TestBsdfSettings test(factory, frame, true, 0.008, 0.02,
                (sRoughness[i] > 0.1f),
                (sRoughness[i] > 0.1f), TestBsdfSettings::BSDFV);
        runTest(test, sViewAnglesTheta, 1, sampleCount);
    }
}


void
TestBsdfv::testWardCorrected()
{
    scene_rdl2::math::ReferenceFrame frame;

    printInfo("##### TestBsdfv::testWardCorrected() ####################");
    for (int i=0; i < sRoughnessCount; i++) {
        printInfo("===== Roughness: %f ====================", sRoughness[i]);

        WardCorrectedBsdfFactory factory(sRoughness[i]);
        TestBsdfSettings test(factory, frame, true, 0.008, 0.03,
                (sRoughness[i] > 0.1f  &&  sRoughness[i] < 0.7f),
                (sRoughness[i] > 0.1f), TestBsdfSettings::BSDFV);
        runTest(test, sViewAnglesTheta, 1, getSampleCount(sRoughness[i]));
    }
}


void
TestBsdfv::testWardDuer()
{
    scene_rdl2::math::ReferenceFrame frame;

    printInfo("##### TestBsdfv::testWardDuer() ####################");
    for (int i=0; i < sRoughnessCount; i++) {
        printInfo("===== Roughness: %f ====================", sRoughness[i]);

        WardDuerBsdfFactory factory(sRoughness[i]);
        TestBsdfSettings test(factory, frame, true, 0.002, 0.03,
                // TODO: Why does the pdf integrate to values > 1 ?!?!
                false, (sRoughness[i] > 0.1f), TestBsdfSettings::BSDFV);
        runTest(test, sViewAnglesTheta, 1, getSampleCount(sRoughness[i]));
    }
}


//----------------------------------------------------------------------------

void
TestBsdfv::testHairDiffuse()
{
    scene_rdl2::math::ReferenceFrame frame;

    printInfo("##### TestBsdfv::testHairDiffuse() ####################");
    HairDiffuseBsdfFactory factory;
    TestBsdfSettings test(factory, frame, true, 0.002, 0.01, true, true, TestBsdfSettings::BSDFV);
    runTest(test, sViewAnglesTheta, 1, getSampleCount(sRoughness[sRoughnessCount - 1]));
}


void
TestBsdfv::testHairR()
{
    scene_rdl2::math::ReferenceFrame frame;

    // running into some numerical issues where
    // consistency tests fail intermittently so increasing the
    // threshold to 0.08
    static const float toleranceConsistency = 0.08f;
    static const float toleranceIntegrals   = 0.08f;
    printInfo("##### TestBsdfv::testHairR() ####################");
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
                              TestBsdfSettings::BSDFV);
        runTest(test, sViewAnglesTheta, 1, getSampleCount(sRoughness[i]));
    }
}


void
TestBsdfv::testHairTT()
{
    scene_rdl2::math::ReferenceFrame frame;

    const float aziRoughness = 0.5f;
    const float offset = 0.0f;
    // running into some numerical issues where
    // consistency tests fail intermittently so increasing the
    // threshold to 0.08
    static const float toleranceConsistency = 0.08f;
    static const float toleranceIntegrals   = 0.08f;

    printInfo("##### TestBsdfv::testHairTT() ####################");
    for (int i=0; i < sRoughnessCount; i++) {
        printInfo("===== Roughness: %f ====================", sRoughness[i]);

        HairTTBsdfFactory factory(sRoughness[i],
                                  aziRoughness,
                                  offset);
        TestBsdfSettings test(factory,
                              frame,
                              false,
                              toleranceConsistency,
                              toleranceIntegrals,
                              (sRoughness[i] > 0.1f),
                              true,
                              TestBsdfSettings::BSDFV);
        runTest(test, sViewAnglesTheta, 1, getSampleCount(sRoughness[i]));
    }
}

//----------------------------------------------------------------------------

void
TestBsdfv::testTwoLobes()
{
    scene_rdl2::math::ReferenceFrame frame;

    printInfo("##### TestBsdfv::testTwoLobes() ####################");
    for (int i=0; i < sRoughnessCount; i++) {
        printInfo("===== Roughness: %f ====================", sRoughness[i]);

        int sampleCount = getSampleCount(sRoughness[i]) +
                          getSampleCount(sRoughness[sRoughnessCount - 1]);

        TwoLobeBsdfFactory factory(sRoughness[i]);
        TestBsdfSettings test(factory, frame, true, 0.002, 0.02,
                (sRoughness[i] > 0.1f  &&  sRoughness[i] < 0.7f),
                (sRoughness[i] > 0.1f), TestBsdfSettings::BSDFV);
        runTest(test, sViewAnglesTheta, 1, sampleCount);
    }
}


void
TestBsdfv::testThreeLobes()
{
    scene_rdl2::math::ReferenceFrame frame;

    printInfo("##### TestBsdfv::testThreeLobes() ####################");
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
                true, TestBsdfSettings::BSDFV);
        runTest(test, sViewAnglesTheta, 1, sampleCount);
    }
}

//----------------------------------------------------------------------------
void
TestBsdfv::testStochasticFlakes()
{
    scene_rdl2::math::ReferenceFrame frame;

    static const size_t flakesDataSize = 100;

    printInfo("##### TestBsdfv::testStochasticFlakes() ####################");
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
                              TestBsdfSettings::BSDFV);
        runTest(test, sViewAnglesTheta, 1, getSampleCount(sRoughness[i]));

        normals.clear();
        colors.clear();
    }
}

//----------------------------------------------------------------------------

} // namespace pbr
} // namespace moonray


CPPUNIT_TEST_SUITE_REGISTRATION(moonray::pbr::TestBsdfv);

