// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

/// @file TestBsdfvTask.cc

#include "BsdfFactory.h"
#include "TestBsdfCommon.h"
#include "TestUtil.h"
#include "TestBsdf_ispc_stubs.h"

#include <moonray/rendering/mcrt_common/ThreadLocalState.h>
#include <scene_rdl2/common/math/Color.h>

namespace moonray {
namespace pbr {

//----------------------------------------------------------------------------

void
TestBsdfConsistencyTask::testBsdfv()
{
    ispc::TestBsdfConsistency test;

    mcrt_common::ThreadLocalState *tls = mcrt_common::getFrameUpdateTLS();
    scene_rdl2::alloc::Arena &arena = tls->mArena;
    SCOPED_MEM(&arena);

    // initialize test input
    test.mArena = (ispc::Arena *)&tls->mArena;
    test.mBsdf = inTest->bsdfFactory.getBsdfv(arena, inTest->frame);
    test.mTestReciprocity = inTest->testReciprocity;
    test.mRangeStart = sampleFirst;
    test.mRangeEnd = sampleLast;
    test.mRandomSeed = randomSeed;
    test.mRandomStream = randomStream;
    test.mMaxSamplesPerLobe = inTest->sMaxSamplesPerLobe;
    test.mNg = *((const ispc::Vec3f *) &inTest->frame.getN());
    test.mTol = inTest->toleranceConsistency;

    // call the test
    ispc::TestBsdf_testConsistency(&test);

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
    mResult.mZeroSamplePdfCount += test.mZeroSamplePdf;
    mResult.mZeroEvalPdfCount += test.mZeroEvalPdf;
    mResult.mZeroEvalRecipPdfCount += test.mZeroEvalRecipPdf;
    //PRINT(test.mSampleCount);
    mResult.mSampleCount += test.mSampleCount;
    //PRINT(mResult.mSampleCount);
}

//----------------------------------------------------------------------------

void
TestBsdfPdfIntegralTask::testBsdfv()
{
    ispc::TestBsdfPdfIntegral test;

    mcrt_common::ThreadLocalState *tls = mcrt_common::getFrameUpdateTLS();
    scene_rdl2::alloc::Arena &arena = tls->mArena;
    SCOPED_MEM(&arena);

    // initialize test input
    test.mArena = (ispc::Arena *)&tls->mArena;
    test.mBsdf = inTest->bsdfFactory.getBsdfv(arena, inTest->frame);
    test.mRangeStart = sampleFirst;
    test.mRangeEnd = sampleLast;
    test.mRandomSeed = randomSeed;
    test.mRandomStream = randomStream;
    test.mMaxSamplesPerLobe = inTest->sMaxSamplesPerLobe;
    test.mSpherical = mResult.mSpherical;
    test.mFrame = *((const ispc::ReferenceFrame *) &inTest->frame);
    test.mWo = *((const ispc::Vec3f *) &inSlice->getWo());

    // call the test
    ispc::TestBsdf_testPdfIntegral(&test);

    // check for and report errors
    testAssert(!test.mInvalidPdf,
               "eval() returned %d invalid pdfs", test.mInvalidPdf);

    // add the integral
    mResult.mIntegral += test.mIntegral;
    mResult.mSampleCount = sampleLast - sampleFirst;
}

//----------------------------------------------------------------------------

void
TestBsdfEvalIntegralTask::testBsdfv()
{
    ispc::TestBsdfEvalIntegral test;

    mcrt_common::ThreadLocalState *tls = mcrt_common::getFrameUpdateTLS();
    scene_rdl2::alloc::Arena &arena = tls->mArena;
    SCOPED_MEM(&arena);

    // initialize test input
    test.mArena = (ispc::Arena *)&tls->mArena;
    test.mBsdf = inTest->bsdfFactory.getBsdfv(arena, inTest->frame);
    test.mRangeStart = sampleFirst;
    test.mRangeEnd = sampleLast;
    test.mMaxSamplesPerLobe = inTest->sMaxSamplesPerLobe;
    test.mFrame = *((const ispc::ReferenceFrame *) &inTest->frame);
    test.mWo = *((const ispc::Vec3f *) &inSlice->getWo());

    // call the test
    ispc::TestBsdf_testEvalIntegral(&test);

    // check for and report errors
    testAssert(!test.mInvalidPdf, "sample() returned %d invalid pdfs", test.mInvalidPdf);
    testAssert(!test.mInvalidEvalColor, "eval() returned %d invalid colors", test.mInvalidEvalColor);
    testAssert(!test.mInvalidSampleColor, "sample() returned %d invalid colors", test.mInvalidSampleColor);
    testAssert(!test.mInvalidDirection, "sample() returned %d invalid directions", test.mInvalidDirection);

    // add the integral
    mResult.mIntegralUniform += *(scene_rdl2::math::Color *) &test.mIntegralUniform;
    mResult.mIntegralImportance += *(scene_rdl2::math::Color *) &test.mIntegralImportance;
    mResult.mSampleCount = sampleLast - sampleFirst;
}

} // namespace pbr
} // namespace moonray

