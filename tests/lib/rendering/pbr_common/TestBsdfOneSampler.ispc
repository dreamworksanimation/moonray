// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

/// @file TestBsdfOneSampler.ispc

#include "TestUtil.isph"

#include <moonray/rendering/pbr/core/Util.isph>
#include <moonray/rendering/pbr/integrator/BsdfOneSampler.isph>

#include <moonray/rendering/shading/ispc/bsdf/Bsdf.isph>
#include <moonray/rendering/shading/ispc/Util.isph>

#include <scene_rdl2/common/math/ispc/ReferenceFrame.isph>

//----------------------------------------------------------------------------
// Consistency test

// inputs/outputs of the consistency test
struct TestBsdfOneSamplerConsistency
{
    // inputs
    uint32 mRandomSeed;
    uint32 mRandomStream;
    varying Bsdf *mBsdf;
    bool mTestReciprocity;
    int mRangeStart;
    int mRangeEnd;
    Vec3f mNg;
    float mTol;

    // outputs
    int mZeroSamplePdfCount;
    int mZeroEvalPdfCount;
    int mZeroEvalRecipPdfCount;

    // output errors
    int mInvalidPdf;
    int mInvalidColor;
    int mInvalidDirection;
    int mInvalidEvalPdf;
    int mInconsistentEvalPdf;
    int mInvalidEvalColor;
    int mInconsistentEvalColor;
    int mInvalidRecipPdf;
    int mInvalidRecipColor;
    int mInconsistentRecipColor;
};

export void
#pragma ignore warning(all)
TestBsdfOneSampler_testConsistency(uniform TestBsdfOneSamplerConsistency * uniform test)
{
    varying DWARNGState rand;
    seed_rng(&rand, test->mRandomSeed + programIndex, test->mRandomStream + programIndex);

    test->mZeroSamplePdfCount = 0;
    test->mZeroEvalPdfCount = 0;
    test->mZeroEvalRecipPdfCount = 0;

    test->mInvalidPdf = 0;
    test->mInvalidColor = 0;
    test->mInvalidDirection = 0;
    test->mInvalidEvalPdf = 0;
    test->mInconsistentEvalPdf = 0;
    test->mInvalidEvalColor = 0;
    test->mInconsistentEvalColor = 0;
    test->mInvalidRecipPdf = 0;
    test->mInvalidRecipColor = 0;
    test->mInconsistentRecipColor = 0;

    varying Bsdf * uniform bsdf = test->mBsdf;
    assert(bsdf->mNumLobes > 0);
    const varying Vec3f Ng = test->mNg;

    foreach (sample = test->mRangeStart ... test->mRangeEnd) {

        uniform bool foundError = false;

        // Generate a random eye direction
        Vec3f wo;
        do {
            wo = sampleSphereUniform(frandom(&rand), frandom(&rand));
        } while (dot(wo, Ng) < sEpsilon);
        assert(isNormalized(wo));

        // Get ready to sample, but don't include cosine term since we want to
        // test bsdf reciprocity
        BsdfSlice slice;
        BsdfSlice_init(&slice,
                       Ng,
                       wo,
                       BSDF_LOBE_TYPE_ALL,
                       /* includeCosineTerm = */ false,
                       /* entering = */ true,
                       SHADOW_TERMINATOR_FIX_OFF);
        BsdfOneSampler bSampler;
        BsdfOneSampler_init(&bSampler, bsdf, &slice);

        // Draw a sample according to the bsdf
        float rLobe = frandom(&rand);
        float r1 = frandom(&rand);
        float r2 = frandom(&rand);
        Vec3f wi;
        float pdf;
        Color color = BsdfOneSampler_sample(&bSampler, rLobe, r1, r2, wi, pdf, nullptr);
        if (pdf == 0.f) {
            foreach_active (lane) ++test->mZeroSamplePdfCount;
            continue;
        }

        // Check the sample
        if (!isValidPdf(pdf)) {
            foreach_active (lane) ++test->mInvalidPdf;
            foundError = true;
        }
        if (!isValidColor(color)) {
            foreach_active (lane) ++test->mInvalidColor;
            if (test->mInvalidColor > 0) {
                print("Break Here: sample = %\n", sample);
            }
            foundError = true;
        }
        if (!isValidDirection(wi)) {
            foreach_active (lane) ++test->mInvalidDirection;
            foundError = true;
        }

        // Check consistency with eval()
        // TODO: If we sampled from a mirror lobe, then calling eval here
        // won't be consistent with bsdf.sample(), since sample() will return
        // only the mirror lobe contribution and eval() will return all
        // non-mirror lobes.
        float checkPdf;
        Color checkColor = BsdfOneSampler_eval(&bSampler, wi, checkPdf, nullptr);
        if (checkPdf == 0.f) {
            foreach_active (lane) ++test->mZeroEvalPdfCount;
            continue;
        }
        if (!isValidPdf(checkPdf)) {
            foreach_active (lane) ++test->mInvalidEvalPdf;
            foundError = true;
        }
        float error = computeError(pdf, checkPdf);
        if (error > test->mTol) {
            foreach_active (lane) ++test->mInconsistentEvalPdf;
            foundError = true;
        }
        if (!isValidColor(checkColor)) {
            foreach_active (lane) ++test->mInvalidEvalColor;
            foundError = true;
        }
        float errorR = computeError(color.r, checkColor.r);
        float errorG = computeError(color.g, checkColor.g);
        float errorB = computeError(color.b, checkColor.b);
        if (errorR > test->mTol || errorG > test->mTol || errorB > test->mTol) {
            foreach_active (lane) ++test->mInconsistentEvalColor;
            foundError = true;
        }

        // Check eval() reciprocity
        if (test->mTestReciprocity) {
            BsdfSlice recipSlice;
            BsdfSlice_init(&recipSlice,
                           Ng,
                           wi,
                           BSDF_LOBE_TYPE_ALL,
                           /* includeCosineTerm = */ false,
                           /* entering = */ true,
                           SHADOW_TERMINATOR_FIX_OFF);
            BsdfOneSampler recipBSampler;
            BsdfOneSampler_init(&recipBSampler, bsdf, &recipSlice);
            checkColor = BsdfOneSampler_eval(&recipBSampler, wo, checkPdf, nullptr);
            if (checkPdf == 0.f) {
                foreach_active (lane) ++test->mZeroEvalRecipPdfCount;
                continue;
            }
            if (!isValidPdf(checkPdf)) {
                foreach_active (lane) ++test->mInvalidRecipPdf;
                foundError = true;
            }
            if (!isValidColor(checkColor)) {
                foreach_active (lane) ++test->mInvalidRecipColor;
                foundError = true;
            }
            // TODO: The reciprocity test should divide by the square of the index
            // of refraction for each medium side to handle glossy transmission
            // properly
            errorR = computeError(color.r, checkColor.r);
            errorG = computeError(color.g, checkColor.g);
            errorB = computeError(color.b, checkColor.b);
            if (errorR > test->mTol || errorG > test->mTol | errorB > test->mTol) {
                foreach_active (lane) ++test->mInconsistentRecipColor;
                foundError = true;
            }
        }

        // TODO: check other "Rules" from Bsdf API (i.e. mirror case).

        // For your debugging needs
        if (foundError) {
            color = BsdfOneSampler_sample(&bSampler, rLobe, r1, r2, wi, pdf, nullptr);
        }
    }
}

//----------------------------------------------------------------------------
// Pdf integral test

// inputs/outputs of the pdf integral test
struct TestBsdfOneSamplerPdfIntegral
{
    // inputs
    uint32 mRandomSeed;
    uint32 mRandomStream;
    varying Bsdf *mBsdf;
    int mRangeStart;
    int mRangeEnd;
    ReferenceFrame mFrame;
    Vec3f mWo;
    bool mSpherical;

    // outputs
    float mIntegral;

    // output errors
    int mInvalidPdf;
};

export void
#pragma ignore warning(all)
TestBsdfOneSampler_testPdfIntegral(uniform TestBsdfOneSamplerPdfIntegral * uniform test)
{
    DWARNGState rand;
    seed_rng(&rand, test->mRandomSeed + programIndex, test->mRandomStream + programIndex);

    test->mIntegral = 0.f;

    test->mInvalidPdf = 0;

    const varying Bsdf * uniform bsdf = test->mBsdf;
    assert(bsdf->mNumLobes > 0);
    const ReferenceFrame frame = test->mFrame;
    const Vec3f wo = test->mWo;
    const uniform bool spherical = test->mSpherical;

    BsdfSlice slice;
    BsdfSlice_init(&slice,
                   getN(frame),
                   wo,
                   BSDF_LOBE_TYPE_ALL,
                   /* includeCosineTerm = */ false,
                   /* entering = */ true,
                   SHADOW_TERMINATOR_FIX_OFF);
    BsdfOneSampler bSampler;
    BsdfOneSampler_init(&bSampler, bsdf, &slice);

    // Compute the integrated probability using uniform sampling
    float integral = 0.f;
    foreach (sample = test->mRangeStart ... test->mRangeEnd) {
        Vec3f wi = sampleSphereUniform(frandom(&rand), frandom(&rand));
        if (!spherical) {
            if (Bsdf_getType(bsdf) & BSDF_LOBE_TYPE_TRANSMISSION) {
                wi = -1.f * wi;
            }
            wi = localToGlobal(frame, wi);
        }
        assert(isNormalized(wi));

        float pdf;
        BsdfOneSampler_eval(&bSampler, wi, pdf, nullptr);
        if (pdf > 0.f) {
            if (!isValidPdf(pdf)) {
                foreach_active (lane) ++test->mInvalidPdf;
            }
            integral += pdf;
        }
    }

    // reduce the results
    foreach_active (lane) {
        test->mIntegral += extract(integral, lane);
    }
}

//----------------------------------------------------------------------------
// Eval integral test

// intputs/outputs of the eval integral test
struct TestBsdfOneSamplerEvalIntegral
{
    // inputs
    uint32 mSeed;
    varying Bsdf *mBsdf;
    int mRangeStart;
    int mRangeEnd;
    ReferenceFrame mFrame;
    Vec3f mWo;

    // outputs
    Color mIntegralUniform;
    Color mIntegralImportance;

    // output errors
    int mInvalidPdf;
    int mInvalidEvalColor;
    int mInvalidSampleColor;
    int mInvalidDirection;
};

export void
#pragma ignore warning(all)
TestBsdfOneSampler_testEvalIntegral(uniform TestBsdfOneSamplerEvalIntegral * uniform test)
{
    DWARNGState rand;
    seed_rng(&rand, test->mSeed + programIndex, 7265 + programIndex /* Select random stream */);
    foreach (i = 0 ... test->mRangeStart) {
        frandom(&rand);
    }

    test->mIntegralUniform = sBlack;
    test->mIntegralImportance = sBlack;

    test->mInvalidPdf = 0;
    test->mInvalidEvalColor = 0;
    test->mInvalidSampleColor = 0;
    test->mInvalidDirection = 0;

    const varying Bsdf * uniform bsdf = test->mBsdf;
    const ReferenceFrame frame = test->mFrame;
    const Vec3f wo = test->mWo;

    BsdfSlice slice;
    BsdfSlice_init(&slice,
                   getN(frame),
                   wo,
                   BSDF_LOBE_TYPE_ALL,
                   /* includeCosineTerm = */ false,
                   /* entering = */ true,
                   SHADOW_TERMINATOR_FIX_OFF);
    BsdfOneSampler bSampler;
    BsdfOneSampler_init(&bSampler, bsdf, &slice);

    const float pdfUniform = 1.f / (Bsdf_getIsSpherical(bsdf) ? sFourPi : sTwoPi);

    Color integralUniform = sBlack;
    Color integralImportance = sBlack;
    foreach (sample = test->mRangeStart ... test->mRangeEnd) {
        const float r1 = frandom(&rand);
        const float r2 = frandom(&rand);

        // Compute the integrated bsdf using uniform sampling
        Vec3f wi;
        if (Bsdf_getIsSpherical(bsdf)) {
            wi = sampleSphereUniform(r1, r2);
        } else {
            wi = sampleLocalHemisphereUniform(r1, r2);
            if (Bsdf_getType(bsdf) & BSDF_LOBE_TYPE_TRANSMISSION) {
                wi = -1.f * wi;
            }
            wi = localToGlobal(frame, wi);
        }
        assert(isNormalized(wi));

        float pdf;
        Color color = BsdfOneSampler_eval(&bSampler, wi, pdf, nullptr);
        if (!isValidColor(color)) {
            foreach_active (lane) ++test->mInvalidEvalColor;
        }
        integralUniform = integralUniform + color * (1.f / pdfUniform);

        // Compute the integrated bsdf using importance sampling
        color = BsdfOneSampler_sample(&bSampler, frandom(&rand), r1, r2, wi, pdf, nullptr);
        if (isSampleInvalid(color, pdf)) {
            continue;
        }
        if (!isValidPdf(pdf)) {
            foreach_active (lane) ++test->mInvalidPdf;
        }
        if (!isValidColor(color)) {
            foreach_active (lane) ++test->mInvalidSampleColor;
        }
        if (!isValidDirection(wi)) {
            foreach_active (lane) ++test->mInvalidDirection;
        }
        integralImportance = integralImportance + color * (1.f / pdf);
    }

    // outputs
    foreach_active (lane) {
        test->mIntegralUniform.r += extract(integralUniform.r, lane);
        test->mIntegralUniform.g += extract(integralUniform.g, lane);
        test->mIntegralUniform.b += extract(integralUniform.b, lane);
        test->mIntegralImportance.r += extract(integralImportance.r, lane);
        test->mIntegralImportance.g += extract(integralImportance.g, lane);
        test->mIntegralImportance.b += extract(integralImportance.b, lane);
    }
}

