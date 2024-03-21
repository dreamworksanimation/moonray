// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

//
// @file TestLights.cc
//
#include "TestLights.h"
#include "TestUtil.h"
#include "TestLights_ispc_stubs.h"

#include <moonray/rendering/geom/Api.h>
#include <moonray/rendering/geom/prim/PrimitivePrivateAccess.h>
#include <moonray/rendering/shading/Util.h>
#include <moonray/common/mcrt_macros/moonray_static_check.h>
#include <moonray/rendering/pbr/core/PbrTLState.h>
#include <moonray/common/time/Timer.h>
#include <scene_rdl2/common/math/Math.h>
#include <scene_rdl2/render/util/Random.h>

// to iterate over the [0, 1)^2 space we take (NUM_SAMPLES_PER_AXIS * NUM_SAMPLES_PER_AXIS) samples
#define NUM_SAMPLES_PER_AXIS                1000
#define GRAINSIZE_PER_AXIS                  (NUM_SAMPLES_PER_AXIS / 8)

#define NUM_SAMPLES                         (NUM_SAMPLES_PER_AXIS * NUM_SAMPLES_PER_AXIS)
#define INV_NUM_SAMPLES                     (1.0 / double(NUM_SAMPLES))

#define NUM_CAN_ILLUMINATE_TESTS            10000
#define NUM_CAN_ILLUMINATE_TESTS_GRAINSIZE  (NUM_CAN_ILLUMINATE_TESTS / 64)

#define RNG_SEED_INCREMENT                  39119


namespace moonray {
namespace pbr {

using namespace scene_rdl2::math;

//----------------------------------------------------------------------------

//
// Integrate reflected radiance at point p with normal n. We compute a reference
// value using a light tester and compare that to the corresponding importance
// sampled result returned from the real light. If both values match within some
// tolerance then we assume the sampling and evaluation are valid.
//
void
testLightRadiance(const Vec3f &p, const Vec3f &n, const LightTester *lightTester,
        int32_t initialSeed)
{
    const Light *light = lightTester->getLight();

    if (testAssert(light->canIlluminate(p, &n, 0.f, 0.f, nullptr),
                   " can't illuminate point, skipping testLightRadiance")) {
        return;
    }


    double timeCpp = 0.0;
    time::TimerDouble timerCpp(timeCpp);
    timerCpp.start();


    //
    // compute ref radiance for p
    //

    Color refRadiance = doReductionOverUnitSquare<Color>(Color(zero),
        NUM_SAMPLES_PER_AXIS, GRAINSIZE_PER_AXIS,
        [&](const tbb::blocked_range2d<unsigned> &range, const Color &current,
            float scl, float ofs) -> Color {

        double red = 0.0;
        double green = 0.0;
        double blue = 0.0;

        for (unsigned y = range.rows().begin(); y != range.rows().end(); ++y) {

            float r0 = float(y) * scl + ofs;

            for (unsigned x = range.cols().begin(); x != range.cols().end(); ++x) {

                float r1 = float(x) * scl + ofs;

                LightTester::Intersection isect;
                if (lightTester->sample(p, r0, r1, &isect)) {

                    float pdf;
                    Color radiance = lightTester->eval(isect, &pdf);

                    if (pdf != 0.f && !isBlack(radiance)) {

                        // weight by pdf
                        radiance /= pdf;

                        // assume a white lambertian brdf on all surfaces
                        radiance *= sOneOverPi;

                        // account for the cosine term in the rendering eqn
                        radiance *= saturate(dot(isect.mWi, n));

                        // accumulate
                        red += double(radiance.r);
                        green += double(radiance.g);
                        blue += double(radiance.b);
                    }
                }
            }
        }

        return current + Color(float(red * INV_NUM_SAMPLES),
                               float(green * INV_NUM_SAMPLES),
                               float(blue * INV_NUM_SAMPLES));
    });


    //
    // compute test radiance for p
    //

    Color testRadiance = doReductionOverUnitSquare<Color>(Color(zero),
        NUM_SAMPLES_PER_AXIS, GRAINSIZE_PER_AXIS,
        [&](const tbb::blocked_range2d<unsigned> &range, const Color &current,
            float scl, float ofs) -> Color {

        double red = 0.0;
        double green = 0.0;
        double blue = 0.0;

        for (unsigned y = range.rows().begin(); y != range.rows().end(); ++y)
        {
            float r0 = float(y) * scl + ofs;

            for (unsigned x = range.cols().begin(); x != range.cols().end(); ++x)
            {
                float r1 = float(x) * scl + ofs;

                Vec3f wi;
                LightIntersection isect;
                const Vec3f r(r0, r1, 0.f);
                if (light->sample(p, nullptr, 0.f, r, wi, isect, 0.0f)) {

                    float pdf;                    
                    LightFilterRandomValues filterR = {Vec2f(0.f, 0.f), Vec3f(0.f, 0.f, 0.f)};

                    Color radiance = light->eval(nullptr, wi, p, filterR, 0.f, isect, false, nullptr, 0.0f, &pdf);

                    if (pdf != 0.f && !isBlack(radiance)) {

                        // weight by pdf
                        radiance /= pdf;

                        // assume a white lambertian brdf on all surfaces
                        radiance *= sOneOverPi;

                        // account for the cosine term in the rendering eqn
                        radiance *= saturate(dot(wi, n));

                        // accumulate
                        red += double(radiance.r);
                        green += double(radiance.g);
                        blue += double(radiance.b);
                    }
                }
            }
        }
        return current + Color(float(red * INV_NUM_SAMPLES),
                               float(green * INV_NUM_SAMPLES),
                               float(blue * INV_NUM_SAMPLES));
    });


    timerCpp.stop();


    double timeIspc = 0.0;
    time::TimerDouble timerIspc(timeIspc);
    timerIspc.start();

    //
    // compute test radiance for p using ISPC light
    //

    Color testIspcRadiance = doReductionOverUnitSquare<Color>(Color(zero),
        NUM_SAMPLES_PER_AXIS, GRAINSIZE_PER_AXIS,
        [&](const tbb::blocked_range2d<unsigned> &range, const Color &current,
            float scl, float ofs) -> Color {

        ispc::Range2d r;
        initIspcRange(r, range);

        ispc::Color color;
        ispc::testLightRadiance(asIspc(p), asIspc(n), light->asIspc(),
                r, scl, ofs, true, color);

        return current + asCpp(color) * INV_NUM_SAMPLES;
    });


    timerIspc.stop();


    //
    // Print results and run test asserts
    //

    const float radianceEqualilty = 0.05f;

    printInfo(" time radiance      = %f", timeCpp);
    printInfo(" time radiance ispc = %f", timeIspc);

    printInfo(" test radiance      = [%f, %f, %f]",
              testRadiance[0], testRadiance[1], testRadiance[2]);
    printInfo(" test radiance ispc = [%f, %f, %f]",
              testIspcRadiance[0], testIspcRadiance[1], testIspcRadiance[2]);
    printInfo(" ref radiance       = [%f, %f, %f]",
              refRadiance[0], refRadiance[1], refRadiance[2]);

    testAssert(isEqual(testRadiance, refRadiance, radianceEqualilty),
               "radiance mismatch: test = [%f, %f, %f], ref = [%f, %f, %f]",
               testRadiance[0], testRadiance[1], testRadiance[2],
               refRadiance[0], refRadiance[1], refRadiance[2]);
    testAssert(isEqual(testIspcRadiance, refRadiance, radianceEqualilty),
               "radiance mismatch: testIspc = [%f, %f, %f], ref = [%f, %f, %f]",
               testIspcRadiance[0], testIspcRadiance[1], testIspcRadiance[2],
               refRadiance[0], refRadiance[1], refRadiance[2]);
}


//----------------------------------------------------------------------------

//
// Here we are integrating the constant function f(x) = 1 using the sampling
// distribution of the light over the visible domain of the light.
// That is 1 / (pdf / denom), which simplifies to (demon / pdf). Denom is a
// factor which undoes the geometric factor which was previously applied in eval.
// We later divide by the visible surface area since our result will be too big
// by that factor. (This would be equivalent to integrating f(x) = 1 / surface_area).
//
// If the final result doesn't integrate to one, we can assume that either:
// (a) the pdf/sampling routines are inconsistent with each other, or
// (b) the pdf doesn't integrate to 1.
// 
// Important note: we're integrating a constant function over the domain and
// since the function is non-zero everywhere, the pdf function *must* be non-zero
// everywhere also for this scheme to work. What this means in practice is that
// the texture maps we supply for texture must not have any black areas or at least
// just a very small amount. (This restriction only applies to this particular
// unit test, it's fine to have black areas in light texture maps in general.)
//
void
testLightPDF(const Vec3f &p, const Vec3f &n, const LightTester *lightTester,
        int32_t initialSeed)
{
    const Light *light = lightTester->getLight();

    if (testAssert(light->canIlluminate(p, &n, 0.f, 0.f, nullptr),
                   " can't illuminate point, skipping testLightPDF")) {
        return;
    }

    const double visibleSurfaceArea = double(lightTester->getVisibleSurfaceArea(p, n));
    const double surfaceArea = double(lightTester->getSurfaceArea());


    double timeCpp = 0.0;
    time::TimerDouble timerCpp(timeCpp);
    timerCpp.start();


    //
    // compute ref pdfs (only for unit test debugging)
    //

    tbb::atomic<unsigned> refValidSampleCount;
    refValidSampleCount = 0;

    double refPdf = doReductionOverUnitSquare<double>(0.0,
        NUM_SAMPLES_PER_AXIS, GRAINSIZE_PER_AXIS,
        [&](const tbb::blocked_range2d<unsigned> &range, double current,
            float scl, float ofs) -> double {

        unsigned localValidSampleCount = 0;
        double localPdf = 0.0;

        for (unsigned y = range.rows().begin(); y != range.rows().end(); ++y)
        {
            float r0 = float(y) * scl + ofs;

            for (unsigned x = range.cols().begin(); x != range.cols().end(); ++x)
            {
                float r1 = float(x) * scl + ofs;

                LightTester::Intersection isect;
                if (lightTester->sample(p, r0, r1, &isect)) {

                    float pdf;
                    lightTester->eval(isect, &pdf);

                    if (pdf != 0.f) {
                        MNRY_ASSERT(scene_rdl2::math::isnormal(pdf));

                        // undo solid angle scaling which was done in eval
                        float denom = 1.f;
                        if (!lightTester->isInfinite()) {
                            denom = absAreaToSolidAngleScale(isect.mWi, isect.mNormal, isect.mDistance);
                        }
                        localPdf += double(denom / pdf);
                        localValidSampleCount++;
                    }
                }
            }
        }

        refValidSampleCount += localValidSampleCount;

        // lightTester uses light uniform sampling so we need to divide by
        // total light surface area
        localPdf /= surfaceArea;
        localPdf *= INV_NUM_SAMPLES;

        return current + localPdf;
    });


    //
    // compute test pdf
    //

    tbb::atomic<unsigned> testValidSampleCount;
    testValidSampleCount = 0;

    double testPdf = doReductionOverUnitSquare<double>(0.0,
        NUM_SAMPLES_PER_AXIS, GRAINSIZE_PER_AXIS,
        [&](const tbb::blocked_range2d<unsigned> &range, double current,
            float scl, float ofs) -> double {

        unsigned localValidSampleCount = 0;
        double localPdf = 0.0;

        for (unsigned y = range.rows().begin(); y != range.rows().end(); ++y)
        {
            float r0 = float(y) * scl + ofs;

            for (unsigned x = range.cols().begin(); x != range.cols().end(); ++x)
            {
                float r1 = float(x) * scl + ofs;

                Vec3f wi;
                LightIntersection isect;
                const Vec3f r(r0, r1, 0.f);
                if (light->sample(p, nullptr, 0.f, r, wi, isect, 0.0f)) {

                    float pdf;
                    LightFilterRandomValues filterR = {Vec2f(0.f, 0.f), Vec3f(0.f, 0.f, 0.f)};
                    light->eval(nullptr, wi, p, filterR, 0.f, isect, false, nullptr, 0.0f, &pdf);

                    if (pdf != 0.f) {
                        MNRY_ASSERT(scene_rdl2::math::isnormal(pdf));

                        // undo solid angle scaling which was done in eval
                        float denom = 1.f;
                        if (!lightTester->isInfinite()) {
                            denom = absAreaToSolidAngleScale(wi, isect.N, isect.distance);
                        }
                        localPdf += double(denom / pdf);
                        localValidSampleCount++;
                    }
                }
            }
        }

        testValidSampleCount += localValidSampleCount;

        // Light uses light importance sampling, which samples only the visible
        // part of the light area so we need to divide by visible light surface
        // area.
        // TODO: this is do-able for the lights we have so far, but might not
        // be possible for more complex light shapes.
        localPdf /= visibleSurfaceArea;
        localPdf *= INV_NUM_SAMPLES;

        return current + localPdf;
    });


    timerCpp.stop();


    double timeIspc = 0.0;
    time::TimerDouble timerIspc(timeIspc);
    timerIspc.start();

    //
    // compute test pdf of ISPC light
    //

    tbb::atomic<unsigned> testIspcValidSampleCount;
    testIspcValidSampleCount = 0;

    double testIspcPdf = doReductionOverUnitSquare<double>(0.0,
        NUM_SAMPLES_PER_AXIS, GRAINSIZE_PER_AXIS,
        [&](const tbb::blocked_range2d<unsigned> &range, double current, float scl, float ofs) -> double {

        double localPdf = 0.0;
        unsigned localValidSampleCount = 0;

        ispc::Range2d r;
        initIspcRange(r, range);

        ispc::testLightPdf(asIspc(p), asIspc(n), light->asIspc(),
                r, scl, ofs, lightTester->isInfinite(),
                localPdf, localValidSampleCount);

        // Light uses light importance sampling, which samples only the visible
        // part of the light area so we need to divide by visible light surface
        // area.
        // TODO: this is do-able for the lights we have so far, but might not
        // be possible for more complex light shapes.
        localPdf /= visibleSurfaceArea;
        localPdf *= INV_NUM_SAMPLES;

        testIspcValidSampleCount += localValidSampleCount;
        return current + localPdf;
    });


    timerIspc.stop();


    //
    // Print results and run test asserts
    //

    static const double pdfError = 0.02;

    printInfo(" time pdf      = %f", timeCpp);
    printInfo(" time pdf ispc = %f", timeIspc);

    printInfo(" ref  pdf      = %f", refPdf);
    printInfo(" test pdf      = %f", testPdf);
    printInfo(" test pdf ispc = %f", testIspcPdf);

    if (refValidSampleCount > 0) {
        const double expectedPdf = double(refValidSampleCount) / NUM_SAMPLES;
        testAssert(isEqual(refPdf, expectedPdf, pdfError),
                "Reference pdf is wrong - (%f != %f) (%d/%d valid samples)",
                refPdf, expectedPdf, unsigned(refValidSampleCount), NUM_SAMPLES);
    } else {
        printInfo("Reference pdf didn't hit any valid samples (no big deal).");
    }

    if (testValidSampleCount > 0) {
        const double expectedPdf = double(testValidSampleCount) / NUM_SAMPLES;
        testAssert(isEqual(testPdf, expectedPdf, pdfError),
                "Test pdf is wrong - (%f != %f) (%d/%d valid samples)",
                testPdf, expectedPdf, unsigned(testValidSampleCount), NUM_SAMPLES);
    } else {
        printInfo("Test pdf didn't hit any valid samples (no big deal).");
    }

    if (testIspcValidSampleCount > 0) {
        const double expectedPdf = double(testIspcValidSampleCount) / NUM_SAMPLES;
        testAssert(isEqual(testIspcPdf, expectedPdf, pdfError),
                "Test pdf is wrong - (%f != %f) (%d/%d valid samples)",
                testIspcPdf, expectedPdf, unsigned(testIspcValidSampleCount), NUM_SAMPLES);
        testAssert(isEqual(testIspcPdf, testPdf, pdfError),
                "Test Ispc and Cpp pdf mismatch - (%f != %f)",
                testIspcPdf, testPdf);
    } else {
        printInfo("Test Ispc pdf didn't hit any valid samples (no big deal).");
    }
}

// The pdf for a point on the mesh light is the pdf of selecting a face on the
// mesh light * the pdf of picking a point on that face. Points on a face are
// uniformly sampled, so that pdf is always 1/area of the face. The pdf of
// picking the faces is dependent on the bvh traversal of that face. The bvh
// is a binary tree where each node has a left and right child. Each node has
// an importance value I >= 0. So the probability of each node is I / (IL + IR),
// where I is either IL or IR. Let Pl = I / (IL + IR) be the probability that
// a node at level l is traversed. Then the pdf for face i is Pdf_i = Product (Pli)
// over all levels l. We want to verify that Sum (Pdf_i) over all faces = 1.

void
testMeshLightFaceSelectionPDF(const Vec3f &p, const Vec3f &n, const LightTester *lightTester,
        int32_t initialSeed)
{
    const MeshLightTester* meshLightTester = static_cast<const MeshLightTester*>(lightTester);
    float pdf = meshLightTester->getIntegratedPdf(p, n);
    testAssert(isEqual(pdf, 1.0f), "Test Cpp pdf is not 1.0f - %f", pdf);
}

//----------------------------------------------------------------------------


// This test checks that canIlluminate never culls a light when we can get
// samples with contributions from it for a given (P, N) configuration.
// This is done by computing the irradiance from a light x (P, N) using sample()
// and eval() (classic monte-carlo integration), and if we get any contribution
// at all then we make sure the call to canIlluminate() returns true.

// TODO: This test guarantees that the culling test is correct, but may not
// guarantee that it is efficient (i.e. canIlluminate could cull lights not
// often enough).

// TODO:
// - Need to test canIlluminate() w/wo a normal and w/wo a radius ?
// - Need to test sample() w/wo a normal too

void
testLightCanIlluminate(const Vec3f &, const Vec3f &, const LightTester *lightTester,
        int32_t initialSeed)
{
    const Light *light = lightTester->getLight();

    const Vec3f lightPos = light->getPosition(0.f);


    double timeCpp = 0.0;
    time::TimerDouble timerCpp(timeCpp);
    timerCpp.start();


    //
    // Test C++ light implementation
    //

    tbb::atomic<int32_t> seed;
    seed = initialSeed;

    tbb::parallel_for(tbb::blocked_range<unsigned>(0, NUM_CAN_ILLUMINATE_TESTS,
                                            NUM_CAN_ILLUMINATE_TESTS_GRAINSIZE),
                      [&](tbb::blocked_range<unsigned> const &range) {
        scene_rdl2::util::Random rng(seed += RNG_SEED_INCREMENT);
        std::uniform_real_distribution<float> dist(-40.0f, 40.0f);
        for (unsigned i = range.begin(); i != range.end(); ++i) {

            // Sample p and n around the light source
            Vec3f p = Vec3f(dist(rng), dist(rng), dist(rng)) + lightPos;
            Vec3f n = shading::sampleSphereUniform(rng.getNextFloat(), rng.getNextFloat());

            // Compute the irradiance from the light using monte-carlo integration
            bool gotContribution = false;
            static const unsigned sampleCount = 1024;
            for (unsigned j=0; j < sampleCount; j++) {
                float r0 = rng.getNextFloat();
                float r1 = rng.getNextFloat();
                float r2 = rng.getNextFloat();

                Vec3f wi;
                LightIntersection isect;
                const Vec3f r(r0, r1, r2);
                if (light->sample(p, &n, 0.f, r, wi, isect, 0.0f)) {

                    float pdf;
                    LightFilterRandomValues filterR = {Vec2f(0.f, 0.f), Vec3f(0.f, 0.f, 0.f)};
                    Color radiance = light->eval(nullptr, wi, p, filterR, 0.f, isect, false, nullptr, 0.0f, &pdf);

                    if (pdf != 0.0f  &&  !isBlack(radiance)) {
                        radiance = radiance / pdf * clamp(dot(wi, n));
                        if (!isBlack(radiance)) {
                            gotContribution = true;
                            break;
                        }
                    }
                }
            }

            // If we have some light contribution, we'd better make sure the
            // light doesn't cull itself from this situation
            if (gotContribution) {
                testAssert(light->canIlluminate(p, &n, 0.f, 0.f, nullptr),
                        "canIlluminate() culled a light which would contribute");
            }
        }
    });

    timerCpp.stop();

    printInfo(" Light culling test passed");


    double timeIspc = 0.0;
    time::TimerDouble timerIspc(timeIspc);
    timerIspc.start();


    //
    // Test ISPC light implementation
    //

    seed = initialSeed;

    tbb::parallel_for(tbb::blocked_range<unsigned>(0, NUM_CAN_ILLUMINATE_TESTS,
                                            NUM_CAN_ILLUMINATE_TESTS_GRAINSIZE),
                      [&](tbb::blocked_range<unsigned> const &range) {
        int32_t localSeed = (seed += RNG_SEED_INCREMENT);

        bool success = asCppBool(ispc::testLightCanIlluminate(light->asIspc(),
                asIspc(lightPos), localSeed, range.begin(), range.end()));
        testAssert(success,
                "ispc::Light_canIlluminate() culled a light which would contribute");
    });

    timerIspc.stop();

    printInfo(" Light culling ispc test passed");

    printInfo(" time culling      = %f", timeCpp);
    printInfo(" time culling ispc = %f", timeIspc);
}


//----------------------------------------------------------------------------

//
// This tests that the results of Light::sample and Light::intersect are
// consistent with each other. We test this by randomly sampling a light, and
// then intersecting the light at the location we just sampled. The returned
// intersections from both calls should be identical, within floating point
// tolerances.
// 
// We fail this test under one of 2 conditions:
// a) Some member to the isect structures returned by sample and intersect differ
//    by a non-trivial amount, or
// b) A non-trivial percentage of total tests differ by small amount.
//
void
testLightIntersection(const Vec3f &p, const Vec3f &n, const LightTester *lightTester,
        int32_t initialSeed)
{
    const Light *light = lightTester->getLight();

    if (testAssert(light->canIlluminate(p, &n, 0.f, 0.f, nullptr),
                   " can't illuminate point, skipping testLightIntersection")) {
        return;
    }


    // turn an x, y coordinate into a uniform distributed number within [0, 1)^2
    float scl, ofs;
    getScaleOffset<float>(0, NUM_SAMPLES_PER_AXIS - 1, 0.f, 0.99999f, &scl, &ofs);


    double timeCpp = 0.0;
    time::TimerDouble timerCpp(timeCpp);
    timerCpp.start();


    //
    // Test C++ implementation
    //

    tbb::atomic<unsigned> cppIsectsEqual;
    cppIsectsEqual = 0;
    tbb::atomic<unsigned> cppNoIntersection;
    cppNoIntersection = 0;
    tbb::atomic<unsigned> cppInvalidSamples;
    cppInvalidSamples = 0;

    tbb::parallel_for (tbb::blocked_range<unsigned>(0u, NUM_SAMPLES_PER_AXIS, GRAINSIZE_PER_AXIS),
                       [&](tbb::blocked_range<unsigned> const &range) {

        unsigned localIsectsEqual = 0;
        unsigned localNoIntersection = 0;
        unsigned localInvalidSamples = 0;

        for (unsigned y = range.begin(); y != range.end(); ++y) {
            float r0 = float(y) * scl + ofs;

            for (unsigned x = 0; x != NUM_SAMPLES_PER_AXIS; ++x) {
                float r1 = float(x) * scl + ofs;

                Vec3f wi;
                LightIntersection refIsect;
                const Vec3f r(r0, r1, 0.f);
                if (light->sample(p, nullptr, 0.f, r, wi, refIsect, 0.0f)) {
                    LightIntersection testIsect;
                    if (light->intersect(p, nullptr, wi, 0.f, refIsect.distance + 1.f, testIsect)) {

                        bool passed = true;
                        bool asserted = false;

                        // test within one degree
                        if (!isEqualDirection(testIsect.N, refIsect.N, 1.f)) {
                            passed = false;
                            // test within 5 degrees
                            if (testAssert(isEqualDirection(testIsect.N, refIsect.N, 5.f),
                                       "normals differ [%f, %f, %f] != [%f, %f, %f]",
                                       testIsect.N.x, testIsect.N.y, testIsect.N.z,
                                       refIsect.N.x, refIsect.N.y, refIsect.N.z)) {
                                asserted = true;
                            }
                        }

                        if (!asserted  &&  !isEqualWrappedUv(testIsect.uv, refIsect.uv, 0.04f)) {
                            passed = false;
                            if (testAssert(isEqualWrappedUv(testIsect.uv, refIsect.uv, 0.08f),
                                       "uvs differ [%f, %f] != [%f, %f]",
                                       testIsect.uv.x, testIsect.uv.y,
                                       refIsect.uv.x, refIsect.uv.y)) {
                                asserted = true;
                            }
                        }

                        if (!asserted  &&  !isEqual(testIsect.distance, refIsect.distance, 0.001f)) {
                            passed = false;
                            if (testAssert(isEqual(testIsect.distance, refIsect.distance, 0.01f),
                                       "hit distances differ [%f] != [%f]",
                                       testIsect.distance, refIsect.distance)) {
                                asserted = true;
                            }
                        }

                        unsigned isectDataFieldsUsed = lightTester->getIsectDataFieldsUsed();
                        for (unsigned j = 0; j < isectDataFieldsUsed; ++j) {
                            if (!isEqual(testIsect.data[j], refIsect.data[j], 0.04f)) {
                                passed = false;
                                if (testAssert(isEqual(testIsect.data[j], refIsect.data[j], 0.1f),
                                           "data[%d] differs [%f] != [%f]",
                                           j, testIsect.data[j], refIsect.data[j])) {
                                    asserted = true;
                                }
                            }
                        }

                        // Useful for debugging failures
                        if (asserted) {
#ifdef DEBUG
                            // Set your breakpoint here!
                            light->sample(p, nullptr, 0.f, r, wi, refIsect, 0.0f);
                            light->intersect(p, nullptr, wi, 0.f, refIsect.distance + 1.f, testIsect);
#endif
                            return;
                        }

                        localIsectsEqual += passed ? 1 : 0;
                    } else {
                        ++localNoIntersection;
                    }
                } else {
                    ++localInvalidSamples;
                }
            }
        }

        cppIsectsEqual += localIsectsEqual;
        cppNoIntersection += localNoIntersection;
        cppInvalidSamples += localInvalidSamples;
    });

    timerCpp.stop();

    MNRY_ASSERT(cppIsectsEqual + cppNoIntersection <= NUM_SAMPLES);

    {
        unsigned numIntersected = NUM_SAMPLES - cppNoIntersection;
        double percentageValidData = double(cppIsectsEqual * 100) / double(numIntersected);
        double percentageNoIntersection = double(cppNoIntersection * 100) / double(NUM_SAMPLES);
        double percentageInvalidSamples = double(cppInvalidSamples * 100) / double(NUM_SAMPLES);

        printInfo(" C++ test:");
        printInfo(" time intersection = %f", timeCpp);
        printInfo(" %6.2f percent of intersection tests have matched data\n"
                  " %6.2f percent of tests failed to sample\n"
                  " %6.2f percent of tests returned no intersection",
                  percentageValidData, percentageInvalidSamples,
                  percentageNoIntersection);
        testAssert(percentageValidData + percentageInvalidSamples > 99.0 && percentageNoIntersection < 10.0,
            "Percentage of mismatched data is < 99%% or percentage without intersection is > 10%%");
    }


    double timeIspc = 0.0;
    time::TimerDouble timerIspc(timeIspc);
    timerIspc.start();


    //
    // Test ISPC implementation
    //

    tbb::atomic<unsigned> ispcIsectsEqual;
    ispcIsectsEqual = 0;
    tbb::atomic<unsigned> ispcNoIntersection;
    ispcNoIntersection = 0;
    tbb::atomic<unsigned> ispcInvalidSamples;
    ispcInvalidSamples = 0;

    tbb::parallel_for (tbb::blocked_range<unsigned>(0u, NUM_SAMPLES_PER_AXIS, GRAINSIZE_PER_AXIS),
                       [&](tbb::blocked_range<unsigned> const &range) {

        unsigned localIsectsEqual = 0;
        unsigned localNoIntersection = 0;
        unsigned localInvalidSamples = 0;

        ispc::Range2d r;
        r.mRowBegin = range.begin();
        r.mRowEnd = range.end();
        r.mColBegin = 0;
        r.mColEnd = NUM_SAMPLES_PER_AXIS;

        bool success = asCppBool(ispc::testLightIntersection(light->asIspc(),
                lightTester->getIsectDataFieldsUsed(), asIspc(p), r, scl, ofs,
                localIsectsEqual, localNoIntersection, localInvalidSamples));
        testAssert(success, "ispc::testLightIntersection() asserted");

        ispcIsectsEqual += localIsectsEqual;
        ispcNoIntersection += localNoIntersection;
        ispcInvalidSamples += localInvalidSamples;
    });

    timerIspc.stop();

    MNRY_ASSERT(ispcIsectsEqual + ispcNoIntersection <= NUM_SAMPLES);

    {
        unsigned numIntersected = NUM_SAMPLES - ispcNoIntersection;
        double percentageValidData = double(ispcIsectsEqual * 100) / double(numIntersected);
        double percentageNoIntersection = double(ispcNoIntersection * 100) / double(NUM_SAMPLES);
        double percentageInvalidSamples = double(ispcInvalidSamples * 100) / double(NUM_SAMPLES);

        printInfo(" ISPC test:");
        printInfo(" time intersection = %f", timeIspc);
        printInfo(" %6.2f percent of intersection tests have matched data\n"
                  " %6.2f percent of tests failed to sample\n"
                  " %6.2f percent of tests returned no intersection",
                  percentageValidData, percentageInvalidSamples,
                  percentageNoIntersection);
        testAssert(percentageValidData + percentageInvalidSamples > 99.0 && percentageNoIntersection < 10.0,
            "Percentage of mismatched data is < 99%% or percentage without intersection is > 10%%");
    }
}

//----------------------------------------------------------------------------

TestLights::TestLights() :
    mRand(123)
{
    mContext.setDsoPath(RDL2DSO_PATH);
}

void
TestLights::setUp()
{
    setupThreadLocalData();

    //
    // Create various permutations of tests for each light:
    //

    Color color = Color(one) * 64.f;
    float scaleFactor = 2.1f;

    // point lights downward along the negative y axis in world space
    Mat4f rotX = Mat4f::rotate(Vec4f(1.f, 0.f, 0.f, 0.f), deg2rad(-90.f));
    Mat4f rotY = Mat4f::rotate(Vec4f(0.f, 1.f, 0.f, 0.f), deg2rad(4.f));
    Mat4f rotZ = Mat4f::rotate(Vec4f(0.f, 0.f, 1.f, 0.f), deg2rad(10.f));
    Mat4f rot = rotX * rotY * rotZ;
    Mat4f trans = Mat4f::translate(Vec4f(0.f, 6.f, 0.f, 0.f));
    Mat4f scale = Mat4f::scale(Vec4f(scaleFactor, scaleFactor, scaleFactor, 0.f));

    Mat4f unscaledXform = rot * trans;
    Mat4f scaledXform = scale * unscaledXform;

    float width = 3.f;
    float height = 3.f;
    float radius = 2.f;
    float radiusCylinder = 0.5f;
    float lensRadius = 0.3f;
    float aspectRatio = 1.0f;
    float innerConeAngle = 10.f;
    float outerConeAngle = 45.f;
    float angularExtent = 0.53f;

    // Important: This must be an environment map without any areas of black or
    // at least a very minimal amount. See comments for testLightPDF.
    std::string basePath("/work/rd/raas/maps/env_maps/");
    const std::string texturePath = basePath + "Frozen_Waterfall_Ref.exr";

    //
    // RectLight tests:
    //

    std::shared_ptr<RectLight> rectLight;

    rectLight = std::shared_ptr<RectLight>(new RectLight(makeRectLightSceneObject(
        "rect A", &mContext, unscaledXform, color, width, height, nullptr)));
    rectLight->update(Mat4d(one));
    mRectLightTesters.push_back(std::make_shared<RectLightTester>(
            rectLight, "unscaled solid colored"));

    rectLight = std::shared_ptr<RectLight>(new RectLight(makeRectLightSceneObject(
        "rect B", &mContext, scaledXform, color, width / scaleFactor,
        height / scaleFactor, nullptr)));
    rectLight->update(Mat4d(one));
    mRectLightTesters.push_back(std::make_shared<RectLightTester>(
            rectLight, "scaled solid colored"));

    rectLight = std::shared_ptr<RectLight>(new RectLight(makeRectLightSceneObject(
        "rect C", &mContext, unscaledXform, color, width, height,
        texturePath.c_str())));
    rectLight->update(Mat4d(one));
    mRectLightTesters.push_back(std::make_shared<RectLightTester>(
            rectLight, "unscaled textured"));

    rectLight = std::shared_ptr<RectLight>(new RectLight(makeRectLightSceneObject(
        "rect D", &mContext, scaledXform, color, width / scaleFactor,
        height / scaleFactor, texturePath.c_str())));
    rectLight->update(Mat4d(one));
    mRectLightTesters.push_back(std::make_shared<RectLightTester>(
            rectLight, "scaled textured"));

    //
    // CylinderLight tests:
    //

    std::shared_ptr<CylinderLight> cylinderLight;

    cylinderLight = std::shared_ptr<CylinderLight>(new CylinderLight(makeCylinderLightSceneObject(
        "cylinder A", &mContext, unscaledXform, color, radiusCylinder, height, nullptr)));
    cylinderLight->update(Mat4d(one));
    mCylinderLightTesters.push_back(std::make_shared<CylinderLightTester>(
            cylinderLight, "unscaled solid colored"));

    cylinderLight = std::shared_ptr<CylinderLight>(new CylinderLight(makeCylinderLightSceneObject(
        "cylinder B", &mContext, scaledXform, color, radiusCylinder / scaleFactor, height / scaleFactor, nullptr)));
    cylinderLight->update(Mat4d(one));
    mCylinderLightTesters.push_back(std::make_shared<CylinderLightTester>(
            cylinderLight, "scaled solid colored"));

    cylinderLight = std::shared_ptr<CylinderLight>(new CylinderLight(makeCylinderLightSceneObject(
        "cylinder C", &mContext, unscaledXform, color, radiusCylinder, height, texturePath.c_str())));
    cylinderLight->update(Mat4d(one));
    mCylinderLightTesters.push_back(std::make_shared<CylinderLightTester>(
            cylinderLight, "unscaled textured"));

    cylinderLight = std::shared_ptr<CylinderLight>(new CylinderLight(makeCylinderLightSceneObject(
        "cylinder D", &mContext, scaledXform, color, radiusCylinder / scaleFactor, height / scaleFactor,
        texturePath.c_str())));
    cylinderLight->update(Mat4d(one));
    mCylinderLightTesters.push_back(std::make_shared<CylinderLightTester>(
            cylinderLight, "scaled textured"));

    //
    // DiskLight tests:
    //

    std::shared_ptr<DiskLight> diskLight;

    diskLight = std::shared_ptr<DiskLight>(new DiskLight(makeDiskLightSceneObject(
        "disk A", &mContext, unscaledXform, color, radius, nullptr)));
    diskLight->update(Mat4d(one));
    mDiskLightTesters.push_back(std::make_shared<DiskLightTester>(
            diskLight, "unscaled solid colored"));

    diskLight = std::shared_ptr<DiskLight>(new DiskLight(makeDiskLightSceneObject(
        "disk B", &mContext, scaledXform, color, radius / scaleFactor, nullptr)));
    diskLight->update(Mat4d(one));
    mDiskLightTesters.push_back(std::make_shared<DiskLightTester>(
            diskLight, "scaled solid colored"));

    diskLight = std::shared_ptr<DiskLight>(new DiskLight(makeDiskLightSceneObject(
        "disk C", &mContext, unscaledXform, color, radius, texturePath.c_str())));
    diskLight->update(Mat4d(one));
    mDiskLightTesters.push_back(std::make_shared<DiskLightTester>(
            diskLight, "unscaled textured"));

    diskLight = std::shared_ptr<DiskLight>(new DiskLight(makeDiskLightSceneObject(
        "disk D", &mContext, scaledXform, color, radius / scaleFactor,
        texturePath.c_str())));
    diskLight->update(Mat4d(one));
    mDiskLightTesters.push_back(std::make_shared<DiskLightTester>(
            diskLight, "scaled textured"));

    //
    // SphereLight tests:
    //

    std::shared_ptr<SphereLight> sphereLight;

    sphereLight = std::shared_ptr<SphereLight>(new SphereLight(makeSphereLightSceneObject(
        "sphere A", &mContext, unscaledXform, color, radius, nullptr)));
    sphereLight->update(Mat4d(one));
    mSphereLightTesters.push_back(std::make_shared<SphereLightTester>(
            sphereLight, "unscaled solid colored visible cap"));

    sphereLight = std::shared_ptr<SphereLight>(new SphereLight(makeSphereLightSceneObject(
        "sphere B", &mContext, scaledXform, color, radius / scaleFactor, nullptr)));
    sphereLight->update(Mat4d(one));
    mSphereLightTesters.push_back(std::make_shared<SphereLightTester>(
            sphereLight, "scaled solid colored visible cap"));

    sphereLight = std::shared_ptr<SphereLight>(new SphereLight(makeSphereLightSceneObject(
        "sphere C", &mContext, unscaledXform, color, radius,
        texturePath.c_str())));
    sphereLight->update(Mat4d(one));
    mSphereLightTesters.push_back(std::make_shared<SphereLightTester>(
            sphereLight, "unscaled textured visible cap"));

    sphereLight = std::shared_ptr<SphereLight>(new SphereLight(makeSphereLightSceneObject(
        "sphere D", &mContext, scaledXform, color, radius / scaleFactor,
        texturePath.c_str())));
    sphereLight->update(Mat4d(one));
    mSphereLightTesters.push_back(std::make_shared<SphereLightTester>(
            sphereLight, "scaled textured visible cap"));

    //
    // SpotLight tests:
    //

    std::shared_ptr<SpotLight> spotLight;

    spotLight = std::shared_ptr<SpotLight>(new SpotLight(makeSpotLightSceneObject(
        "spot A", &mContext, unscaledXform, color, lensRadius, aspectRatio,
        innerConeAngle, outerConeAngle, nullptr)));
    spotLight->update(Mat4d(one));
    mSpotLightTesters.push_back(std::make_shared<SpotLightTester>(
            spotLight, "unscaled solid colored"));

    spotLight = std::shared_ptr<SpotLight>(new SpotLight(makeSpotLightSceneObject(
        "spot B", &mContext, scaledXform, color, lensRadius / scaleFactor, aspectRatio,
        innerConeAngle, outerConeAngle, nullptr)));
    spotLight->update(Mat4d(one));
    mSpotLightTesters.push_back(std::make_shared<SpotLightTester>(
            spotLight, "scaled solid colored"));

    //
    // DistantLight tests:
    //

    std::shared_ptr<DistantLight> distantLight;

    distantLight = std::shared_ptr<DistantLight>(new DistantLight(makeDistantLightSceneObject(
        "distant A", &mContext, unscaledXform, color, angularExtent)));
    distantLight->update(Mat4d(one));
    mDistantLightTesters.push_back(std::make_shared<DistantLightTester>(
            distantLight, "solid colored"));

    //
    // EnvLight tests:
    //

    std::shared_ptr<EnvLight> envLight;

    Mat4f envXform = Mat4f(one);

    envLight = std::shared_ptr<EnvLight>(new EnvLight(makeEnvLightSceneObject(
        "env A", &mContext, envXform, color, nullptr, false)));
    envLight->update(Mat4d(one));
    mEnvLightTesters.push_back(std::make_shared<EnvLightTester>(
            envLight, "solid colored full sphere"));

    envLight = std::shared_ptr<EnvLight>(new EnvLight(makeEnvLightSceneObject(
        "env B", &mContext, envXform, color, nullptr, true)));
    envLight->update(Mat4d(one));
    mEnvLightTesters.push_back(std::make_shared<EnvLightTester>(
            envLight, "solid colored upper hemisphere"));

    envLight = std::shared_ptr<EnvLight>(new EnvLight(makeEnvLightSceneObject(
        "env C", &mContext, envXform, color, texturePath.c_str(), false)));
    envLight->update(Mat4d(one));
    mEnvLightTesters.push_back(std::make_shared<EnvLightTester>(
            envLight, "textured full sphere"));

    envLight = std::shared_ptr<EnvLight>(new EnvLight(makeEnvLightSceneObject(
        "env D", &mContext, envXform, color, texturePath.c_str(), true)));
    envLight->update(Mat4d(one));
    mEnvLightTesters.push_back(std::make_shared<EnvLightTester>(
            envLight, "textured upper hemisphere"));


    //
    // MeshLight tests:
    //

    std::shared_ptr<MeshLight> meshLight;

    // create mesh light
    Mat4f meshXform = Mat4f(one);
    meshLight = std::shared_ptr<MeshLight>(new MeshLight(makeMeshLightSceneObject(
        "mesh A", &mContext, meshXform, color, nullptr, true)));
    meshLight->update(Mat4d(one));

    // create cube
    // (-1, 1, -1) ------------ (1, 1, -1)
    //            |            |
    //            |            |
    //            |            |
    //            |            |
    //            |            |
    // (-1,-1, -1) ------------ (1,-1, -1)
    geom::PolygonMesh::FaceVertexCount faceVertexCount({4});
    geom::PolygonMesh::IndexBuffer indices({3, 2, 1, 0, //back
                                      3, 7, 6, 2, //right
                                      6, 5, 1, 2, //bottom
                                      0, 1, 5, 4, //left
                                      7, 4, 5, 6, //front
                                      0, 4, 7, 3}); //top
    geom::PolygonMesh::VertexBuffer vertices;
    vertices.push_back(Vec3fa(-1,  1, -1, 0));
    vertices.push_back(Vec3fa(-1, -1, -1, 0));
    vertices.push_back(Vec3fa( 1, -1, -1, 0));
    vertices.push_back(Vec3fa( 1,  1, -1, 0));

    vertices.push_back(Vec3fa(-1,  1, 1, 0));
    vertices.push_back(Vec3fa(-1, -1, 1, 0));
    vertices.push_back(Vec3fa( 1, -1, 1, 0));
    vertices.push_back(Vec3fa( 1,  1, 1, 0));
    std::unique_ptr<geom::PolygonMesh> mesh = geom::createPolygonMesh(
        std::move(faceVertexCount), std::move(indices),
        std::move(vertices), geom::LayerAssignmentId(0));
    geom::internal::Primitive* pImpl =
        geom::internal::PrimitivePrivateAccess::getPrimitiveImpl(mesh.get());

    // set cube to mesh light
    meshLight->setMesh(pImpl);
    meshLight->finalize();

    mMeshLightTesters.push_back(std::make_shared<MeshLightTester>(
            meshLight, "cube"));

    // make another meshLight with a texture
    meshLight = std::shared_ptr<MeshLight>(new MeshLight(makeMeshLightSceneObject(
        "mesh A", &mContext, meshXform, color, texturePath.c_str(), true)));
    meshLight->update(Mat4d(one));
    meshLight->setMesh(pImpl);
    meshLight->finalize();
    mMeshLightTesters.push_back(std::make_shared<MeshLightTester>(
            meshLight, "cube with texture"));
}

void
TestLights::tearDown()
{
    mMeshLightTesters.clear();
    mEnvLightTesters.clear();
    mDistantLightTesters.clear();
    mSpotLightTesters.clear();
    mSphereLightTesters.clear();
    mDiskLightTesters.clear();
    mRectLightTesters.clear();

    cleanupThreadLocalData();
}


/// Run a particular test on a particular surface point and direction.
void
runTestOnSinglePoint(const Vec3f &p, const Vec3f &n, const char *pointDesc,
                     TestFunction func, const LightTesters &lightTesters,
                     int32_t initialSeed)
{
    for (size_t i = 0; i < lightTesters.size(); ++i) {
        const LightTester *lightTester = lightTesters[i].get();
        printInfo("---- %s %s using a %s ----", lightTester->getLightTypeName(),
                lightTester->getDesc(), pointDesc);
        func(p, n, lightTester, initialSeed);
    }
}

// Lights are all pointing down in world space (along the negative y axis).
void
TestLights::runTestOnMultiplePoints(const std::string &funcName, TestFunction func,
        const LightTesters &lightTesters, bool closePointsOnly)
{
    printInfo("==== Testing Light %s ====", funcName.c_str());

    Vec3f p;

    // use a common normal for both sets of tests, points upwards in world space
    Vec3f n = Vec3f(0.3, 1.0f, -0.1f).normalize();

    // test a close point
    p = Vec3f(0.1f, -2.0f, 0.3f);
    runTestOnSinglePoint(p, n, "nearby point", func, lightTesters, getRandomInt32());

    // test a far point
    if (!closePointsOnly) {
        p = Vec3f(-1.0f, -100.0f, 4.0f);
        runTestOnSinglePoint(p, n, "distant point", func, lightTesters, getRandomInt32());
    }
}

uint32_t
TestLights::getRandomInt32()
{
    return mRand.getNextUInt();
}

//----------------------------------------------------------------------------

} // namespace pbr
} // namespace moonray

CPPUNIT_TEST_SUITE_REGISTRATION(moonray::pbr::TestLights);

