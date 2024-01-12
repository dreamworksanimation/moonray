// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

/// @file TestBsdfOneSamplerv.cc

#include "TestBsdfOneSamplerv.h"

#include "BsdfFactory.h"
#include "TestBsdfCommon.h"
#include "TestBsdfOneSampler_ispc_stubs.h"
#include "TestUtil.h"
#include <moonray/rendering/pbr/core/PbrTLState.h>

#include <moonray/rendering/mcrt_common/ThreadLocalState.h>

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

    // Passthrough
    runBsdfTest(test, viewAnglesTheta, viewAnglesPhy, sampleCount);
}

//----------------------------------------------------------------------------

void TestBsdfOneSamplerv::setUp()
{
    setupThreadLocalData();
}

void TestBsdfOneSamplerv::tearDown()
{
    cleanupThreadLocalData();
}

TestBsdfOneSamplerv::TestBsdfOneSamplerv()
{
}

TestBsdfOneSamplerv::~TestBsdfOneSamplerv()
{
}


//----------------------------------------------------------------------------

void
TestBsdfOneSamplerv::testLambert()
{
    scene_rdl2::math::ReferenceFrame frame;

    printInfo("##### TestBsdfOneSamplerv::testLambert() ####################");
    LambertBsdfFactory factory;
    TestBsdfSettings test(factory, frame, true, 0.002, 0.01, true, true,
            TestBsdfSettings::BSDF_ONE_SAMPLERV);
    runTest(test, sViewAnglesTheta, 1, getSampleCount(sRoughness[sRoughnessCount - 1]));
}


//----------------------------------------------------------------------------

void
TestBsdfOneSamplerv::testCookTorrance()
{
    scene_rdl2::math::ReferenceFrame frame;

    printInfo("##### TestBsdfOneSamplerv::testCookTorrance() ####################");
    for (int i=0; i < sRoughnessCount; i++) {
        printInfo("===== Roughness: %f ====================", sRoughness[i]);

        CookTorranceBsdfFactory factory(sRoughness[i]);
        TestBsdfSettings test(factory, frame, true, 0.002, 0.03,
                (sRoughness[i] > 0.1f  &&  sRoughness[i] < 0.7f),
                (sRoughness[i] > 0.1f), TestBsdfSettings::BSDF_ONE_SAMPLERV);
        runTest(test, sViewAnglesTheta, 1, getSampleCount(sRoughness[i]));
    }
}


void
TestBsdfOneSamplerv::testGGXCookTorrance()
{
    scene_rdl2::math::ReferenceFrame frame;

    printInfo("##### TestBsdfOneSamplerv::testGGXCookTorrance() ####################");
    for (int i=0; i < sRoughnessCount; i++) {
        printInfo("===== Roughness: %f ====================", sRoughness[i]);

        GGXCookTorranceBsdfFactory factory(sRoughness[i]);
        TestBsdfSettings test(factory, frame, true, 0.002, 0.03,
                (sRoughness[i] > 0.1f  &&  sRoughness[i] < 0.4f),
                (sRoughness[i] > 0.1f  &&  sRoughness[i] < 0.4f),
                TestBsdfSettings::BSDF_ONE_SAMPLERV);
        runTest(test, sViewAnglesTheta, 1, getSampleCount(sRoughness[i]));
    }
}


void
TestBsdfOneSamplerv::testAnisoCookTorrance()
{
    scene_rdl2::math::ReferenceFrame frame;

    printInfo("##### TestBsdfOneSamplerv::testAnisoCookTorrance() ####################");
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
                TestBsdfSettings::BSDF_ONE_SAMPLERV);
        runTest(test, sViewAnglesTheta, 1, sampleCount);
    }
}


void
TestBsdfOneSamplerv::testTransmissionCookTorrance()
{
    scene_rdl2::math::ReferenceFrame frame;

    printInfo("##### TestBsdfOneSamplerv::testTransmissionCookTorrance() ####################");
    for (int i=0; i < sRoughnessCount; i++) {
        printInfo("===== Roughness: %f ====================", sRoughness[i]);

        TransmissionCookTorranceBsdfFactory factory(sRoughness[i]);
        TestBsdfSettings test(factory, frame, false, 0.002, 0.05,
                (sRoughness[i] > 0.2f  &&  sRoughness[i] < 0.7f),
                (sRoughness[i] > 0.2f), TestBsdfSettings::BSDF_ONE_SAMPLERV);
        runTest(test, sViewAnglesTheta, 1, 4 * getSampleCount(sRoughness[i]));
    }
}

//----------------------------------------------------------------------------

void
TestBsdfOneSamplerv::testIridescence()
{
    scene_rdl2::math::ReferenceFrame frame;

    static const int size = 2;
    const float positions[size] = {0.f, 1.f};
    const ispc::RampInterpolatorMode interpolators[size] = {
        ispc::RAMP_INTERPOLATOR_MODE_SMOOTH, 
        ispc::RAMP_INTERPOLATOR_MODE_SMOOTH };
    const scene_rdl2::math::Color colors[size] = {
        scene_rdl2::math::Color(1.f, 0.f, 0.f), scene_rdl2::math::Color(1.f, 0.f, 1.f)};

    printInfo("##### TestBsdfOneSamplerv::testIridescence() ####################");
    for (int i=0; i < sRoughnessCount; i++) {
        printInfo("===== Roughness: %f ====================", sRoughness[i]);

        IridescenceBsdfFactory factory(sRoughness[i], 1.0f, ispc::SHADING_IRIDESCENCE_COLOR_USE_HUE_INTERPOLATION,
                scene_rdl2::math::Color(1.f, 0.f, 0.f), scene_rdl2::math::Color(1.f, 0.f, 1.f), false,
                ispc::COLOR_RAMP_CONTROL_SPACE_RGB, size, positions, interpolators, colors,
                2.f, 1.f, 1.f, 1.f);
        TestBsdfSettings test(factory, frame, true, 0.002, 0.04,
                (sRoughness[i] > 0.1  &&  sRoughness[i] < 0.5),
                (sRoughness[i] > 0.1), TestBsdfSettings::BSDF_ONE_SAMPLERV);
        runTest(test, sViewAnglesTheta, 1, getSampleCount(sRoughness[i]));
    }
}


//----------------------------------------------------------------------------

void
TestBsdfOneSamplerv::testRetroreflection()
{
    scene_rdl2::math::ReferenceFrame frame;

    printInfo("##### TestBsdfOneSamplerv::testRetroreflection() ####################");
    for (int i=0; i < sRoughnessCount; i++) {
        printInfo("===== Roughness: %f ====================", sRoughness[i]);

        RetroreflectionBsdfFactory factory(sRoughness[i]);
        // The integrals are evaluated using uniform sampling which makes it an
        // error-prone calculation. Ideally, we'd have higher tolerance for sharper lobes.
        // Right now we set it up to 'fail' only for higher roughness values.
        TestBsdfSettings test(factory, frame, true, 0.002, 0.04,
                sRoughness[i] > 0.3f, sRoughness[i] > 0.3f,
                TestBsdfSettings::BSDF_ONE_SAMPLERV);
        runTest(test, sViewAnglesTheta, 1, getSampleCount(sRoughness[i]));
    }
}

//----------------------------------------------------------------------------

void
TestBsdfOneSamplerv::testEyeCaustic()
{
    scene_rdl2::math::ReferenceFrame frame;

    printInfo("##### TestBsdfOneSamplerv::testEyeCaustic() ####################");
    for (int i=0; i < sRoughnessCount; i++) {
        printInfo("===== Roughness: %f ====================", sRoughness[i]);

        EyeCausticBsdfFactory factory(sRoughness[i]);
        // The integrals are evaluated using uniform sampling which makes it an
        // error-prone calculation. Ideally, we'd have higher tolerance for sharper lobes.
        // Right now we set it up to 'fail' only for higher roughness values.
        TestBsdfSettings test(factory, frame, false , 0.002, 0.04,
                sRoughness[i] > 0.1f, sRoughness[i] > 0.1f,
                TestBsdfSettings::BSDF_ONE_SAMPLERV);
        runTest(test, sViewAnglesTheta, 1, getSampleCount(sRoughness[i]));
    }
}

//----------------------------------------------------------------------------

void
TestBsdfOneSamplerv::testDwaFabric()
{
    scene_rdl2::math::ReferenceFrame frame;

    printInfo("##### TestBsdfOneSamplerv::testDwaFabric() ####################");
    for (int i=0; i < sRoughnessCount; i++) {
        printInfo("===== SpecRoughness: %f ====================", sRoughness[i]);
        DwaFabricBsdfFactory factory(sRoughness[i]);
        // The integrals are evaluated using uniform sampling which makes it an
        // error-prone calculation. Ideally, we'd have higher tolerance for sharper lobes.
        // Right now we set it up to 'fail' only for higher roughness values.
        TestBsdfSettings test(factory, frame, true, 0.002, 0.03,
                              (sRoughness[i] > 0.1f  &&  sRoughness[i] < 0.7f),
                              (sRoughness[i] > 0.1f), TestBsdfSettings::BSDF_ONE_SAMPLERV);
        runTest(test, sViewAnglesTheta, 1, getSampleCount(sRoughness[i]));
    }
}

//----------------------------------------------------------------------------

void
TestBsdfOneSamplerv::testKajiyaKayFabric()
{
    scene_rdl2::math::ReferenceFrame frame;

    printInfo("##### TestBsdfOneSamplerv::testKajiyaKayFabric() ####################");
    for (int i=0; i < sRoughnessCount; i++) {
        printInfo("===== SpecRoughness: %f ====================", sRoughness[i]);
        // Kajiya Kay Model
        KajiyaKayFabricBsdfFactory factory(sRoughness[i]);

        // The integrals are evaluated using uniform sampling which makes it an
        // error-prone calculation. Ideally, we'd have higher tolerance for sharper lobes.
        // Right now we set it up to 'fail' only for higher roughness values.
        TestBsdfSettings test(factory, frame, true, 0.002, 0.03,
                              (sRoughness[i] > 0.1f  &&  sRoughness[i] < 0.7f),
                              (sRoughness[i] > 0.1f), TestBsdfSettings::BSDF_ONE_SAMPLERV);
        runTest(test, sViewAnglesTheta, 1, getSampleCount(sRoughness[i]));
    }
}

//----------------------------------------------------------------------------

void
TestBsdfOneSamplerv::testAshikminhShirley()
{
    scene_rdl2::math::ReferenceFrame frame;

    printInfo("##### TestBsdfOneSamplerv::testAshikminhShirley() ####################");
    for (int i=0; i < sRoughnessCount; i++) {
        printInfo("===== Roughness: %f ====================", sRoughness[i]);

        AshikminhShirleyBsdfFactory factory(sRoughness[i]);
        TestBsdfSettings test(factory, frame, true, 0.008, 0.03,
                (sRoughness[i] > 0.1f), (sRoughness[i] > 0.1f),
                TestBsdfSettings::BSDF_ONE_SAMPLERV);
        runTest(test, sViewAnglesTheta, 1, getSampleCount(sRoughness[i]));
    }
}


void
TestBsdfOneSamplerv::testAshikminhShirleyFull()
{
    scene_rdl2::math::ReferenceFrame frame;

    printInfo("##### TestBsdfOneSamplerv::testAshikminhShirleyFull() ####################");
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
                (sRoughness[i] > 0.1f), TestBsdfSettings::BSDF_ONE_SAMPLERV);
        runTest(test, sViewAnglesTheta, 1, sampleCount);
    }
}


void
TestBsdfOneSamplerv::testWardCorrected()
{
    scene_rdl2::math::ReferenceFrame frame;

    printInfo("##### TestBsdfOneSamplerv::testWardCorrected() ####################");
    for (int i=0; i < sRoughnessCount; i++) {
        printInfo("===== Roughness: %f ====================", sRoughness[i]);

        WardCorrectedBsdfFactory factory(sRoughness[i]);
        TestBsdfSettings test(factory, frame, true, 0.008, 0.03,
                (sRoughness[i] > 0.1f  &&  sRoughness[i] < 0.7f),
                (sRoughness[i] > 0.1f), TestBsdfSettings::BSDF_ONE_SAMPLERV);
        runTest(test, sViewAnglesTheta, 1, getSampleCount(sRoughness[i]));
    }
}


void
TestBsdfOneSamplerv::testWardDuer()
{
    scene_rdl2::math::ReferenceFrame frame;

    printInfo("##### TestBsdfOneSamplerv::testWardDuer() ####################");
    for (int i=0; i < sRoughnessCount; i++) {
        printInfo("===== Roughness: %f ====================", sRoughness[i]);

        WardDuerBsdfFactory factory(sRoughness[i]);
        TestBsdfSettings test(factory, frame, true, 0.002, 0.03,
                // TODO: Why does the pdf integrate to values > 1 ?!?!
                false, (sRoughness[i] > 0.1f), TestBsdfSettings::BSDF_ONE_SAMPLERV);
        runTest(test, sViewAnglesTheta, 1, getSampleCount(sRoughness[i]));
    }
}


//----------------------------------------------------------------------------

void
TestBsdfOneSamplerv::testHairDiffuse()
{
    scene_rdl2::math::ReferenceFrame frame;

    printInfo("##### TestBsdfOneSamplerv::testHairDiffuse() ####################");
    HairDiffuseBsdfFactory factory;
    TestBsdfSettings test(factory, frame, true, 0.002, 0.01, true, true,
            TestBsdfSettings::BSDF_ONE_SAMPLERV);
    runTest(test, sViewAnglesTheta, 1, getSampleCount(sRoughness[sRoughnessCount - 1]));
}

void
TestBsdfOneSamplerv::testHairR()
{
    scene_rdl2::math::ReferenceFrame frame;

    // running into some numerical issues where
    // consistency tests fail intermittently so increasing the
    // threshold to 0.08
    static const float toleranceConsistency = 0.08f;
    static const float toleranceIntegrals   = 0.06f;
    printInfo("##### TestBsdfOneSamplerv::testHairR() ####################");
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
                              TestBsdfSettings::BSDF_ONE_SAMPLERV);
        runTest(test, sViewAnglesTheta, 1, getSampleCount(sRoughness[i]));
    }
}

void
TestBsdfOneSamplerv::testHairTT()
{
    scene_rdl2::math::ReferenceFrame frame;

    // running into some numerical issues where
    // consistency tests fail intermittently so increasing the
    // threshold to 0.08
    static const float toleranceConsistency = 0.08f;
    static const float toleranceIntegrals   = 0.06f;
    printInfo("##### TestBsdfOneSamplerv::testHairTT() ####################");
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
                              TestBsdfSettings::BSDF_ONE_SAMPLERV);
        // need to bump the sample count to avoid pdfIntegral fail
        runTest(test, sViewAnglesTheta, 1, 4 * getSampleCount(sRoughness[i]));
    }
}

//----------------------------------------------------------------------------

void
TestBsdfOneSamplerv::testTwoLobes()
{
    scene_rdl2::math::ReferenceFrame frame;

    printInfo("##### TestBsdfOneSamplerv::testTwoLobes() ####################");
    for (int i=0; i < sRoughnessCount; i++) {
        printInfo("===== Roughness: %f ====================", sRoughness[i]);

        int sampleCount = getSampleCount(sRoughness[i]) +
                          getSampleCount(sRoughness[sRoughnessCount - 1]);

        TwoLobeBsdfFactory factory(sRoughness[i]);
        TestBsdfSettings test(factory, frame, true, 0.002, 0.02,
                (sRoughness[i] > 0.1f  &&  sRoughness[i] < 0.7f),
                (sRoughness[i] > 0.1f), TestBsdfSettings::BSDF_ONE_SAMPLERV);
        runTest(test, sViewAnglesTheta, 1, sampleCount);
    }
}


void
TestBsdfOneSamplerv::testThreeLobes()
{
    scene_rdl2::math::ReferenceFrame frame;

    printInfo("##### TestBsdfOneSamplerv::testThreeLobes() ####################");
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
                true, TestBsdfSettings::BSDF_ONE_SAMPLERV);
        runTest(test, sViewAnglesTheta, 1, sampleCount);
    }
}

//----------------------------------------------------------------------------

void
TestBsdfOneSamplerv::testStochasticFlakes()
{
    scene_rdl2::math::ReferenceFrame frame;

    static const size_t flakesDataSize = 100;

    printInfo("##### TestBsdfOneSamplerv::testStochasticFlakes() ####################");
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
                              TestBsdfSettings::BSDF_ONE_SAMPLERV);
        runTest(test, sViewAnglesTheta, 1, getSampleCount(sRoughness[i]));

        normals.clear();
        colors.clear();
    }
}


//----------------------------------------------------------------------------

void
TestBsdfConsistencyTask::testBsdfOneSamplerv()
{
    ispc::TestBsdfOneSamplerConsistency test;

    mcrt_common::ThreadLocalState *tls = mcrt_common::getFrameUpdateTLS();
    scene_rdl2::alloc::Arena &arena = tls->mArena;
    SCOPED_MEM(&arena);

    // initialize test input
    test.mBsdf = inTest->bsdfFactory.getBsdfv(arena, inTest->frame);
    test.mTestReciprocity = inTest->testReciprocity;
    test.mRangeStart = sampleFirst;
    test.mRangeEnd = sampleLast;
    test.mRandomSeed = randomSeed;
    test.mRandomStream = randomStream;
    test.mNg = *((const ispc::Vec3f *) &inTest->frame.getN());
    test.mTol = inTest->toleranceConsistency;

    // call the test
    ispc::TestBsdfOneSampler_testConsistency(&test);

    // check for an report errors
    // check for and report errors
    testAssert(!test.mInvalidPdf,
               "sample() returned %d invalid pdfs", test.mInvalidPdf);
    testAssert(!test.mInvalidColor,
               "sample() returned %d invalid colors", test.mInvalidColor);
    testAssert(!test.mInvalidDirection,
               "sample() returned %d invalid directions", test.mInvalidDirection);
    testAssert(!test.mInvalidEvalPdf,
               "eval() retuned %d invalid pdfs", test.mInvalidEvalPdf);
    testAssert(!test.mInconsistentEvalPdf,
               "eval() return %d inconsistent pdfs", test.mInconsistentEvalPdf);
    testAssert(!test.mInvalidEvalColor,
               "eval() returned %d invalid colors", test.mInvalidEvalColor);
    testAssert(!test.mInconsistentEvalColor,
               "eval() returned %d inconsistent colors", test.mInconsistentEvalColor);
    testAssert(!test.mInvalidRecipPdf,
               "eval() recip returned %d invalid pdfs", test.mInvalidRecipPdf);
    testAssert(!test.mInvalidRecipColor,
               "eval() recip returned %d invalid colors", test.mInvalidRecipColor);
    testAssert(!test.mInconsistentRecipColor,
               "eval() recip returned %d inconsistent colors", test.mInconsistentRecipColor);

    // update outputs
    mResult.mZeroSamplePdfCount += test.mZeroSamplePdfCount;
    mResult.mZeroEvalPdfCount += test.mZeroEvalPdfCount;
    mResult.mZeroEvalRecipPdfCount += test.mZeroEvalRecipPdfCount;
    mResult.mSampleCount += sampleLast - sampleFirst; // always one sample per item in the range (unlike BsdfSampler)
}

void
TestBsdfPdfIntegralTask::testBsdfOneSamplerv()
{
    ispc::TestBsdfOneSamplerPdfIntegral test;

    mcrt_common::ThreadLocalState *tls = mcrt_common::getFrameUpdateTLS();
    scene_rdl2::alloc::Arena &arena = tls->mArena;
    SCOPED_MEM(&arena);

    // initialize test input
    test.mBsdf = inTest->bsdfFactory.getBsdfv(arena, inTest->frame);
    test.mRangeStart = sampleFirst;
    test.mRangeEnd = sampleLast;
    test.mRandomSeed = randomSeed;
    test.mRandomStream = randomStream;
    test.mFrame = *((const ispc::ReferenceFrame *) &inTest->frame);
    test.mWo = *((const ispc::Vec3f *) &inSlice->getWo());
    test.mSpherical = mResult.mSpherical;

    // call the test
    ispc::TestBsdfOneSampler_testPdfIntegral(&test);

    // check for and report errors
    testAssert(!test.mInvalidPdf,
               "eval() returned %d invalid pdfs", test.mInvalidPdf);

    // add the integral
    mResult.mIntegral += test.mIntegral;
    mResult.mSampleCount = sampleLast - sampleFirst;
}

void
TestBsdfEvalIntegralTask::testBsdfOneSamplerv()
{
    ispc::TestBsdfOneSamplerEvalIntegral test;

    mcrt_common::ThreadLocalState *tls = mcrt_common::getFrameUpdateTLS();
    scene_rdl2::alloc::Arena &arena = tls->mArena;
    SCOPED_MEM(&arena);

    // initialize test input
    test.mBsdf = inTest->bsdfFactory.getBsdfv(arena, inTest->frame);
    test.mRangeStart = sampleFirst;
    test.mRangeEnd = sampleLast;
    test.mFrame = *((const ispc::ReferenceFrame *) &inTest->frame);
    test.mWo = *((const ispc::Vec3f *) &inSlice->getWo());

    // call the test
    ispc::TestBsdfOneSampler_testEvalIntegral(&test);

    // check for and report errors
    testAssert(!test.mInvalidPdf, "sample() returned %d invalid pdfs", test.mInvalidPdf);
    testAssert(!test.mInvalidEvalColor, "eval() returnd %d invalid colors", test.mInvalidEvalColor);
    testAssert(!test.mInvalidSampleColor, "sample() return %d invalid colors", test.mInvalidSampleColor);
    testAssert(!test.mInvalidDirection, "sample() return %d invalid directions", test.mInvalidDirection);

    // add the integral
    // it is a little confusing, but the out variables
    // are references that are allocated and
    // checked by runBsdfTest/testEvalIntegral
    mResult.mIntegralUniform += *(scene_rdl2::math::Color *) &test.mIntegralUniform;
    mResult.mIntegralImportance += *(scene_rdl2::math::Color *) &test.mIntegralImportance;
    mResult.mSampleCount = sampleLast - sampleFirst;
}

} // namespace pbr
} // namespace moonray


CPPUNIT_TEST_SUITE_REGISTRATION(moonray::pbr::TestBsdfOneSamplerv);

