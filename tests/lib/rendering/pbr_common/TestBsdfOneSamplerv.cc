// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

///
/// @file TestBsdfOneSampler.cc
/// $Id$
///


#include "BsdfFactory.h"
#include "TestBsdfCommon.h"
#include "TestBsdfOneSampler_ispc_stubs.h"
#include "TestUtil.h"
#include <moonray/rendering/pbr/core/Util.h>
#include <moonray/rendering/pbr/integrator/BsdfOneSampler.h>

#include <moonray/rendering/mcrt_common/ThreadLocalState.h>
#include <moonray/rendering/shading/bsdf/Bsdf.h>
#include <moonray/rendering/shading/bsdf/BsdfSlice.h>
#include <moonray/rendering/shading/Util.h>
#include <scene_rdl2/render/util/Random.h>

namespace moonray {
namespace pbr {

using namespace scene_rdl2::math;

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

//----------------------------------------------------------------------------

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

//----------------------------------------------------------------------------

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

//----------------------------------------------------------------------------

} // namespace pbr
} // namespace moonray

