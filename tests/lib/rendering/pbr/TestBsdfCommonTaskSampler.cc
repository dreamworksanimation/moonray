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
#include <moonray/rendering/shading/Util.h>
#include <moonray/rendering/shading/bsdf/Bsdf.h>
#include <moonray/rendering/shading/bsdf/BsdfSlice.h>

#include <scene_rdl2/render/util/Random.h>

namespace moonray {
namespace pbr {

using namespace scene_rdl2::math;

//----------------------------------------------------------------------------

void
TestBsdfConsistencyTask::testBsdfSampler()
{
    int zeroSamplePdfCount = 0;
    int zeroEvalPdfCount = 0;
    int zeroEvalRecipPdfCount = 0;
    int sampleCount = 0;

    mcrt_common::ThreadLocalState *tls = mcrt_common::getFrameUpdateTLS();
    pbr::TLState* const pbrTls = MNRY_VERIFY(tls->mPbrTls.get());
    scene_rdl2::alloc::Arena &arena = tls->mArena;
    SCOPED_MEM(&arena);

    scene_rdl2::util::Random random(randomSeed, randomStream);
    shading::Bsdf* const bsdf = inTest->bsdfFactory(arena, inTest->frame);
    BsdfSample bsmp;
    const PathGuide pg;
    const Vec3f p(0.f);

    for (int sample = sampleFirst; sample != sampleLast; ++sample) {

        bool foundError = false;

        // Generate a random eye direction
        Vec3f wo;
        do {
            wo = shading::sampleSphereUniform(random.getNextFloat(), random.getNextFloat());
        } while(dot(wo, inTest->frame.getN()) < sEpsilon);
        testAssert(isOne(wo.length()), "wo is not normalized");

        // Get ready to sample, but don't include cosine term since we want to
        // test bsdf reciprocity.
        shading::BsdfSlice slice(inTest->frame.getN(), wo, false, true, ispc::SHADOW_TERMINATOR_FIX_OFF);
        BsdfSampler sampler(pbrTls->mArena, *bsdf, slice,
                            TestBsdfSettings::sMaxSamplesPerLobe, true, pg);
        sampleCount += sampler.getSampleCount();

        for (int lobeIndex = 0; lobeIndex < sampler.getLobeCount(); ++lobeIndex) {
            for (int i = 0; i < sampler.getLobeSampleCount(lobeIndex); ++i) {

                // Draw the sample and test validity
                const float r1 = random.getNextFloat();
                const float r2 = random.getNextFloat();
                sampler.sample(pbrTls, lobeIndex, p, r1, r2, bsmp);
                if (bsmp.pdf == 0.0f) {
                    zeroSamplePdfCount++;
                    continue;
                }

                // TODO: work under fixed reference frame and slice, and record
                // distribution of sampled lobes and make sure they match their albedos

                // Check the sample
                foundError |= testAssert(isValidPdf(bsmp.pdf),
                        "sample() returned invalid pdf (%f)", bsmp.pdf);
                foundError |= testAssert(isValidColor(bsmp.f),
                        "sample() returned invalid color (%f %f %f)",
                        bsmp.f[0], bsmp.f[1], bsmp.f[2]);
                foundError |= testAssert(isValidDirection(bsmp.wi),
                        "sample() returned invalid direction (%f %f %f)",
                        bsmp.wi[0], bsmp.wi[1], bsmp.wi[2]);


                float error;

                // Check consistency with eval()
                // TODO: If we sampled from a mirror lobe, then calling eval here
                // won't be consistent with bsdf.sample(), since sample() will return
                // only the mirror lobe contribution and eval() will return all
                // non-mirror lobes.
                float checkPdf;
                shading::BsdfLobe *lobe = sampler.getLobe(lobeIndex);
                Color checkF = lobe->eval(slice, bsmp.wi, &checkPdf);
                if (checkPdf == 0.0f) {
                    zeroEvalPdfCount++;
                    continue;
                }

                foundError |= testAssert(isValidPdf(checkPdf),
                        "eval() returned invalid pdf (%f)", checkPdf);
                error = computeError(bsmp.pdf, checkPdf);
                foundError |= testAssert(error < inTest->toleranceConsistency,
                        "eval() returned inconsistent pdf (error=%f)", error);

                // Check consistency with eval()
                foundError |= testAssert(isValidColor(checkF),
                        "eval() returned invalid color (%f %f %f)",
                        checkF[0], checkF[1], checkF[2]);
                for (int i=0; i < 3; i++) {
                    error = computeError(bsmp.f[i], checkF[i]);
                    foundError |= testAssert(error < inTest->toleranceConsistency,
                            "eval() returned inconsistent color (error=%f)", error);
                }


                // Check eval() reciprocity
                shading::BsdfSlice recipSlice(inTest->frame.getN(), bsmp.wi, false, true,
                    ispc::SHADOW_TERMINATOR_FIX_OFF);
                if (inTest->testReciprocity) {
                    checkF = lobe->eval(recipSlice, wo, &checkPdf);
                    if (checkPdf == 0.0f) {
                        zeroEvalRecipPdfCount++;
                        continue;
                    }
                    foundError |= testAssert(isValidPdf(checkPdf),
                            "eval() reciprocal returned invalid pdf (%f)", checkPdf);
                    foundError |= testAssert(isValidColor(checkF),
                            "eval() reciprocal returned invalid color (%f %f %f)",
                            checkF[0], checkF[1], checkF[2]);
                    // TODO: The reciprocity test should divide by the square of the index
                    // of refraction for each medium side to handle glossy transmission
                    // properly
                    for (int i=0; i < 3; i++) {
                        error = computeError(bsmp.f[i], checkF[i]);
                        foundError |= testAssert(error < inTest->toleranceConsistency,
                                "eval() reciprocal returned inconsistent color (err=%f)",
                                error);
                    }
                }

                // TODO: check other "Rules" from Bsdf API (i.e. mirror case).

                // For your debugging needs
                if (foundError) {
                    sampler.sample(pbrTls, lobeIndex, p, r1, r2, bsmp);
                    lobe->eval(slice, bsmp.wi, &checkPdf);
                    lobe->eval(recipSlice, wo, &checkPdf);
                }
            }
        }
    }

    mResult.mZeroSamplePdfCount += zeroSamplePdfCount;
    mResult.mZeroEvalPdfCount += zeroEvalPdfCount;
    mResult.mZeroEvalRecipPdfCount += zeroEvalRecipPdfCount;
    mResult.mSampleCount += sampleCount;
}


//----------------------------------------------------------------------------

void
TestBsdfPdfIntegralTask::testBsdfSampler()
{
    mcrt_common::ThreadLocalState *tls = mcrt_common::getFrameUpdateTLS();
    pbr::TLState *pbrTls = MNRY_VERIFY(tls->mPbrTls.get());
    scene_rdl2::alloc::Arena &arena = tls->mArena;
    SCOPED_MEM(&arena);

    scene_rdl2::util::Random random(randomSeed, randomStream);
    shading::Bsdf *bsdf = inTest->bsdfFactory(arena, inTest->frame);
    const PathGuide pg;
    BsdfSampler sampler(pbrTls->mArena, *bsdf, *inSlice,
                        TestBsdfSettings::sMaxSamplesPerLobe, true, pg);

    // Compute the integrated probability using uniform sampling
    float integral = 0.0f;
    for (int sample = sampleFirst; sample != sampleLast; ++sample) {

        // Compute the integrated probability using uniform sampling
        for (int lobeIndex = 0; lobeIndex < sampler.getLobeCount(); ++lobeIndex) {
            for (int i = 0; i < sampler.getLobeSampleCount(lobeIndex); ++i) {

                shading::BsdfLobe *lobe = sampler.getLobe(lobeIndex);

                Vec3f wi;
                if (mResult.mSpherical) {
                    wi = shading::sampleSphereUniform(random.getNextFloat(), random.getNextFloat());
                } else {
                    wi = shading::sampleLocalHemisphereUniform(random.getNextFloat(), random.getNextFloat());
                    if (lobe->matchesFlag(shading::BsdfLobe::TRANSMISSION)) {
                        wi = -wi;
                    }
                    wi = inTest->frame.localToGlobal(wi);
                }
                testAssert(isNormalized(wi), "wi is not normalized");

                float pdf;
                lobe->eval(*inSlice, wi, &pdf);
                if (pdf > 0.0f) {
                    testAssert(isValidPdf(pdf), "pdf() returned invalid pdf (%f)", pdf);
                    integral += pdf * sampler.getInvLobeSampleCount(lobeIndex);
                }
            }
        }
    }

    mResult.mIntegral += integral / sampler.getLobeCount();
    mResult.mSampleCount = sampleLast - sampleFirst;
}


//----------------------------------------------------------------------------

void
TestBsdfEvalIntegralTask::testBsdfSampler()
{
    mcrt_common::ThreadLocalState *tls = mcrt_common::getFrameUpdateTLS();
    pbr::TLState *pbrTls = MNRY_VERIFY(tls->mPbrTls.get());
    scene_rdl2::alloc::Arena &arena = tls->mArena;
    SCOPED_MEM(&arena);

    scene_rdl2::util::Random random(randomSeed, randomStream);
    shading::Bsdf *bsdf = inTest->bsdfFactory(arena, inTest->frame);
    const PathGuide pg;
    const Vec3f p(0.f);
    BsdfSampler sampler(pbrTls->mArena, *bsdf, *inSlice,
                        TestBsdfSettings::sMaxSamplesPerLobe, true, pg);
    BsdfSample bsmp;

    const float pdfUniform = 1.0f /
            (bsdf->getIsSpherical()  ?  sFourPi  :  sTwoPi);

    Color integralUniform(sBlack);
    Color integralImportance(sBlack);
    for (int sample = sampleFirst; sample != sampleLast; ++sample) {
        for (int lobeIndex = 0; lobeIndex < sampler.getLobeCount(); ++lobeIndex) {
            for (int i = 0; i < sampler.getLobeSampleCount(lobeIndex); ++i) {

                shading::BsdfLobe *lobe = sampler.getLobe(lobeIndex);

                const float r1 = random.getNextFloat();
                const float r2 = random.getNextFloat();

                // Compute the integrated bsdf using uniform sampling
                Vec3f wi;
                if (bsdf->getIsSpherical()) {
                    wi = shading::sampleSphereUniform(r1, r2);
                } else {
                    wi = shading::sampleLocalHemisphereUniform(r1, r2);
                    if (lobe->matchesFlag(shading::BsdfLobe::TRANSMISSION)) {
                        wi = -wi;
                    }
                    wi = inTest->frame.localToGlobal(wi);
                }
                testAssert(isOne(wi.length()), "wi is not normalized");
                Color f = lobe->eval(*inSlice, wi);
                testAssert(isValidColor(f), "eval() returned invalid color (%f %f %f)",
                           f[0], f[1], f[2]);
                integralUniform += f / pdfUniform *
                    sampler.getInvLobeSampleCount(lobeIndex);

                // Compute the integrated bsdf using importance sampling
                sampler.sample(pbrTls, lobeIndex, p, r1, r2, bsmp);
                if (bsmp.isInvalid()) {
                    continue;
                }
                testAssert(isValidPdf(bsmp.pdf), "sample() returned invalid pdf (%f)", bsmp.pdf);
                testAssert(isValidColor(bsmp.f), "sample() returned invalid color "
                           "(%f %f %f)", bsmp.f[0], bsmp.f[1], bsmp.f[2]);
                testAssert(isValidDirection(bsmp.wi), "sample() returned invalid direction "
                           "(%f %f %f)", bsmp.wi[0], bsmp.wi[1], bsmp.wi[2]);
                integralImportance += bsmp.f / bsmp.pdf *
                    sampler.getInvLobeSampleCount(lobeIndex);
            }
        }
    }

    mResult.mIntegralUniform += integralUniform;
    mResult.mIntegralImportance += integralImportance;
    mResult.mSampleCount = sampleLast - sampleFirst;
}


//----------------------------------------------------------------------------

} // namespace pbr
} // namespace moonray

