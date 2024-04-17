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
TestBsdfConsistencyTask::testBsdfOneSampler()
{
    int zeroSamplePdfCount = 0;
    int zeroEvalPdfCount = 0;
    int zeroEvalRecipPdfCount = 0;

    mcrt_common::ThreadLocalState *tls = mcrt_common::getFrameUpdateTLS();
    scene_rdl2::alloc::Arena &arena = tls->mArena;
    SCOPED_MEM(&arena);

    scene_rdl2::util::Random random(randomSeed, randomStream);
    shading::Bsdf *bsdf = inTest->bsdfFactory(arena, inTest->frame);

    for (int s = sampleFirst; s != sampleLast; ++s) {

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
        BsdfOneSampler sampler(*bsdf, slice);

        // Draw a sample according to the bsdf
        const float rLobe = random.getNextFloat();
        const float r1 = random.getNextFloat();
        const float r2 = random.getNextFloat();
        Vec3f wi;
        float pdf;
        Color color = sampler.sample(rLobe, r1, r2, wi, pdf);
        if (pdf == 0.0f) {
            zeroSamplePdfCount++;
            continue;
        }


        // TODO: work under fixed reference frame and slice, and record
        // distribution of sampled lobes and make sure they match their albedos


        // Check the sample
        foundError |= testAssert(isValidPdf(pdf),
                "sample() returned invalid pdf (%f)", pdf);
        foundError |= testAssert(isValidColor(color),
                "sample() returned invalid color (%f %f %f)",
                color[0], color[1], color[2]);
        foundError |= testAssert(isValidDirection(wi),
                "sample() returned invalid direction (%f %f %f)",
                wi[0], wi[1], wi[2]);


        float error;

        // Check consistency with eval()
        // TODO: If we sampled from a mirror lobe, then calling eval here
        // won't be consistent with bsdf.sample(), since sample() will return
        // only the mirror lobe contribution and eval() will return all
        // non-mirror lobes.
        float checkPdf;
        Color checkColor = sampler.eval(wi, checkPdf);
        if (checkPdf == 0.0f) {
            zeroEvalPdfCount++;
            continue;
        }
        foundError |= testAssert(isValidPdf(checkPdf),
                "pdf() returned invalid pdf (%f)", checkPdf);
        error = computeError(pdf, checkPdf);
        foundError |= testAssert(error < inTest->toleranceConsistency,
                "pdf() returned inconsistent pdf %f & %f (error=%f)", pdf, checkPdf, error);

        // Check consistency with eval()
        foundError |= testAssert(isValidColor(checkColor),
                "eval() returned invalid color (%f %f %f)",
                checkColor[0], checkColor[1], checkColor[2]);
        for (int i=0; i < 3; i++) {
            error = computeError(color[i], checkColor[i]);
            foundError |= testAssert(error < inTest->toleranceConsistency,
                    "eval() returned inconsistent color (error=%f)", error);
        }


        // Check eval() reciprocity
        shading::BsdfSlice recipSlice(inTest->frame.getN(), wi, false, true, ispc::SHADOW_TERMINATOR_FIX_OFF);
        if (inTest->testReciprocity) {
            BsdfOneSampler recipSampler(*bsdf, recipSlice);
            checkColor = recipSampler.eval(wo, checkPdf);
            if (checkPdf == 0.0f) {
                zeroEvalRecipPdfCount++;
                continue;
            }
            foundError |= testAssert(isValidPdf(checkPdf),
                    "pdf() reciprocal returned invalid pdf (%f)", checkPdf);
            foundError |= testAssert(isValidColor(checkColor),
                    "eval() reciprocal returned invalid color (%f %f %f)",
                    checkColor[0], checkColor[1], checkColor[2]);
            // TODO: The reciprocity test should divide by the square of the index
            // of refraction for each medium side to handle glossy transmission
            // properly
            for (int i=0; i < 3; i++) {
                error = computeError(color[i], checkColor[i]);
                foundError |= testAssert(error <= inTest->toleranceConsistency,
                        "eval() reciprocal returned inconsistent color (err=%f)",
                        error);
            }
        }

        // TODO: check other "Rules" from Bsdf API (i.e. mirror case).

        // For your debugging needs
        if (foundError) {
            color = sampler.sample(rLobe, r1, r2, wi, pdf);
            // cppcheck complains at redundant assignment if you don't comment
            // these two lines out when you check in.
            //color = sampler.eval(wi, checkPdf);
            //color = recipSampler.eval(wo, checkPdf);
        }
    }

    mResult.mZeroSamplePdfCount += zeroSamplePdfCount;
    mResult.mZeroEvalPdfCount += zeroEvalPdfCount;
    mResult.mZeroEvalRecipPdfCount += zeroEvalRecipPdfCount;
    mResult.mSampleCount += sampleLast - sampleFirst;
}


//----------------------------------------------------------------------------

void
TestBsdfPdfIntegralTask::testBsdfOneSampler()
{
    mcrt_common::ThreadLocalState *tls = mcrt_common::getFrameUpdateTLS();
    scene_rdl2::alloc::Arena &arena = tls->mArena;
    SCOPED_MEM(&arena);

    scene_rdl2::util::Random random(randomSeed, randomStream);
    shading::Bsdf *bsdf = inTest->bsdfFactory(arena, inTest->frame);
    BsdfOneSampler sampler(*bsdf, *inSlice);

    // Compute the integrated probability using uniform sampling
    float integral = 0.0f;
    for (int s = sampleFirst; s != sampleLast; ++s) {

        Vec3f wi;
        if (mResult.mSpherical) {
            wi = shading::sampleSphereUniform(random.getNextFloat(), random.getNextFloat());
        } else {
            wi = shading::sampleLocalHemisphereUniform(random.getNextFloat(), random.getNextFloat());
            if (bsdf->getType() & shading::BsdfLobe::TRANSMISSION) {
                wi = -wi;
            }
            wi = inTest->frame.localToGlobal(wi);
        }
        testAssert(isNormalized(wi), "wi is not normalized");

        float pdf;
        sampler.eval(wi, pdf);
        if (pdf > 0.0f) {
            testAssert(isValidPdf(pdf), "pdf() returned invalid pdf (%f)", pdf);
            integral += pdf;
        }
    }

    mResult.mIntegral += integral;
    mResult.mSampleCount = sampleLast - sampleFirst;
}


//----------------------------------------------------------------------------

void
TestBsdfEvalIntegralTask::testBsdfOneSampler()
{
    mcrt_common::ThreadLocalState *tls = mcrt_common::getFrameUpdateTLS();
    scene_rdl2::alloc::Arena &arena = tls->mArena;
    SCOPED_MEM(&arena);

    scene_rdl2::util::Random random(randomSeed, randomStream);
    shading::Bsdf *bsdf = inTest->bsdfFactory(arena, inTest->frame);
    BsdfOneSampler sampler(*bsdf, *inSlice);

    const float pdfUniform = 1.0f / (bsdf->getIsSpherical()  ?  sFourPi  :  sTwoPi);

    Color integralUniform(sBlack);
    Color integralImportance(sBlack);
    for (int s = sampleFirst; s != sampleLast; ++s) {

        const float r1 = random.getNextFloat();
        const float r2 = random.getNextFloat();

        // Compute the integrated bsdf using uniform sampling
        Vec3f wi;
        if (bsdf->getIsSpherical()) {
            wi = shading::sampleSphereUniform(r1, r2);
        } else {
            wi = shading::sampleLocalHemisphereUniform(r1, r2);
            if (bsdf->getType() & shading::BsdfLobe::TRANSMISSION) {
                wi = -wi;
            }
            wi = inTest->frame.localToGlobal(wi);
        }
        testAssert(isOne(wi.length()), "wi is not normalized");
        float pdf;
        Color color = sampler.eval(wi, pdf);
        testAssert(isValidColor(color), "eval() returned invalid color (%f %f %f)",
                color[0], color[1], color[2]);
        integralUniform += color / pdfUniform;

        // Compute the integrated bsdf using importance sampling
        color = sampler.sample(random.getNextFloat(), r1, r2, wi, pdf);
        if (isSampleInvalid(color, pdf)) {
            continue;
        }
        testAssert(isValidPdf(pdf), "sample() returned invalid pdf (%f)", pdf);
        testAssert(isValidColor(color), "sample() returned invalid color "
                "(%f %f %f)", color[0], color[1], color[2]);
        testAssert(isValidDirection(wi), "sample() returned invalid direction "
                "(%f %f %f)", wi[0], wi[1], wi[2]);
        integralImportance += color / pdf;
    }

    mResult.mIntegralUniform += integralUniform;
    mResult.mIntegralImportance += integralImportance;
    mResult.mSampleCount = sampleLast - sampleFirst;
}


//----------------------------------------------------------------------------

} // namespace pbr
} // namespace moonray

