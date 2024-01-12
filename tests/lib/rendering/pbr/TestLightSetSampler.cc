// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

//
// @file TestLightSetSampler.cc
//

#include "TestLightSetSampler.h"
#include "TestUtil.h"
#include "TestLightSetSampler_ispc_stubs.h"

#include <moonray/rendering/pbr/core/PbrTLState.h>
#include <moonray/rendering/pbr/light/DiskLight.h>
#include <moonray/rendering/pbr/light/EnvLight.h>
#include <moonray/rendering/pbr/light/RectLight.h>
#include <moonray/rendering/pbr/light/SphereLight.h>

#include <moonray/rendering/pbr/core/PbrTLState.h>
#include <moonray/rendering/shading/bsdf/Bsdf.h>

#include <moonray/common/mcrt_macros/moonray_static_check.h>
#include <moonray/common/time/Timer.h>
#include <scene_rdl2/common/math/Math.h>
#include <scene_rdl2/render/util/Arena.h>

#include <tbb/task_scheduler_init.h>


// to iterate over the [0, 1)^2 space we take (NUM_SAMPLES_PER_AXIS * NUM_SAMPLES_PER_AXIS) samples
#define NUM_SAMPLES_PER_AXIS                1000
#define GRAINSIZE_PER_AXIS                  (NUM_SAMPLES_PER_AXIS / 8)

#define NUM_SAMPLES                         (NUM_SAMPLES_PER_AXIS * NUM_SAMPLES_PER_AXIS)
#define INV_NUM_SAMPLES                     (1.0 / double(NUM_SAMPLES))


namespace moonray {
namespace pbr {

using namespace scene_rdl2::math;


//----------------------------------------------------------------------------

//
// Integrate reflected radiance at point p with normal n. We compute this value
// both with our C++ and ISPC codepaths and make sure the results are the same
//
void
TestLightSetSampler::testLightSetRadiance(const LightSet &lightSet,
                                          const scene_rdl2::math::Vec3f &p,
                                          const scene_rdl2::math::Vec3f &n)
{
    const scene_rdl2::math::Color result = scene_rdl2::math::Color(0.268678, 0.340643, 0.539275);

    //
    // Compute radiance for (p, n) with C++ codepath
    //

    double timeCpp = 0.0;
    time::TimerDouble timerCpp(timeCpp);
    timerCpp.start();

    scene_rdl2::math::Color testRadiance = doReductionOverUnitSquare<scene_rdl2::math::Color>(scene_rdl2::math::Color(zero),
        NUM_SAMPLES_PER_AXIS, GRAINSIZE_PER_AXIS,
        [&](const tbb::blocked_range2d<unsigned> &range, const scene_rdl2::math::Color &current,
            float scl, float ofs) -> scene_rdl2::math::Color {

        // Access thread-local arena allocator
        mcrt_common::ThreadLocalState *tls = mcrt_common::getFrameUpdateTLS();
        scene_rdl2::alloc::Arena *arena = &tls->mArena;
        SCOPED_MEM(arena);

        // Create the LightSetSampler
        unsigned sampleCount = range.rows().size() * range.cols().size();
        LightSetSampler lSampler(arena, lightSet, shading::Bsdf(), p, sampleCount);

        double red = 0.0;
        double green = 0.0;
        double blue = 0.0;

        // Loop on lights
        for (int l=0; l < lSampler.getLightCount(); ++l) {

            const Light* light = lSampler.getLight(l);

            for (unsigned y = range.rows().begin(); y != range.rows().end(); ++y) {
                float r0 = float(y) * scl + ofs;
                for (unsigned x = range.cols().begin(); x != range.cols().end(); ++x) {
                    float r1 = float(x) * scl + ofs;
                    const scene_rdl2::math::Vec3f r(r0, r1, 0.f);
                    LightSample lsmp;
                    LightFilterRandomValues filterR = {scene_rdl2::math::Vec2f(0.f, 0.f),
                                                       scene_rdl2::math::Vec3f(0.f, 0.f, 0.f)};
                    lSampler.sampleIntersectAndEval(tls, light, nullptr /*LighFilterList*/, p, &n, filterR, 0.f,
                        r, lsmp, 0.0f);
                    if (lsmp.isValid()) {

                        // Compute reflected radiance from white lambertian brdf,
                        // accounting for the cosine term in the rendering eqn
                        scene_rdl2::math::Color radiance = lsmp.Li * rcp(lsmp.pdf) * sOneOverPi
                            * saturate(dot(lsmp.wi, n));

                        // Accumulate
                        red += double(radiance.r);
                        green += double(radiance.g);
                        blue += double(radiance.b);
                    }
                }
            }
        }
        return current + scene_rdl2::math::Color(float(red * INV_NUM_SAMPLES),
                               float(green * INV_NUM_SAMPLES),
                               float(blue * INV_NUM_SAMPLES));
    });

    timerCpp.stop();


    //
    // Compute radiance for (p, n) with ISPC codepath
    //

    double timeIspc = 0.0;
    time::TimerDouble timerIspc(timeIspc);
    timerIspc.start();

    scene_rdl2::math::Color testIspcRadiance = doReductionOverUnitSquare<scene_rdl2::math::Color>(scene_rdl2::math::Color(zero),
        NUM_SAMPLES_PER_AXIS, GRAINSIZE_PER_AXIS,
        [&](const tbb::blocked_range2d<unsigned> &range, const scene_rdl2::math::Color &current,
            float scl, float ofs) -> scene_rdl2::math::Color {

        // Access thread-local arena allocator
        mcrt_common::ThreadLocalState *tls = mcrt_common::getFrameUpdateTLS();
        scene_rdl2::alloc::Arena *arena = &tls->mArena;
        SCOPED_MEM(arena);

        ispc::Range2d r;
        initIspcRange(r, range);

        ispc::Color color;
        ispc::testLightSetRadiance(reinterpret_cast<ispc::Arena *>(arena),
                lightSet.asIspc(), asIspc(p), asIspc(n),
                r, scl, ofs, true, color);

        return current + asCpp(color) * INV_NUM_SAMPLES;
    });

    timerIspc.stop();


    //
    // Print results and run test asserts
    //

    const float radianceEquality = 0.001f;

    printInfo(" time radiance      = %f", timeCpp);
    printInfo(" time radiance ispc = %f", timeIspc);

    printInfo(" test radiance      = [%f, %f, %f]",
              testRadiance[0], testRadiance[1], testRadiance[2]);
    printInfo(" test radiance ispc = [%f, %f, %f]",
              testIspcRadiance[0], testIspcRadiance[1], testIspcRadiance[2]);

    testAssert(isEqual(testRadiance, result, radianceEquality),
               "radiance mismatch: test = [%f, %f, %f], result = [%f, %f, %f]",
               testRadiance[0], testRadiance[1], testRadiance[2],
               result[0], result[1], result[2]);
    testAssert(isEqual(testRadiance, testIspcRadiance, radianceEquality),
               "radiance mismatch: test = [%f, %f, %f], testIspc = [%f, %f, %f]",
               testRadiance[0], testRadiance[1], testRadiance[2],
               testIspcRadiance[0], testIspcRadiance[1], testIspcRadiance[2]);
}


//----------------------------------------------------------------------------

TestLightSetSampler::TestLightSetSampler()
{
    mContext.setDsoPath(RDL2DSO_PATH);
}


TestLightSetSampler::~TestLightSetSampler()
{
}


void
TestLightSetSampler::setUp()
{
    setupThreadLocalData();
    // See scene in lib/rendering/pbr/unittest/scenes/TestLightSetSampler.rdla
    {
        // Env Light
        scene_rdl2::math::Mat4f xform(one);
        scene_rdl2::math::Color color = 0.05f * sWhite;
        std::shared_ptr<Light> light = std::shared_ptr<Light>(
                new EnvLight(makeEnvLightSceneObject("EnvLight", &mContext,
                             xform, color, nullptr, false)));
        light->update(scene_rdl2::math::Mat4d(one));
        mLights.push_back(light);
        mLightSetA.push_back(light.get());
    }

    {
        // Sphere Light
        scene_rdl2::math::Mat4f xform = scene_rdl2::math::Mat4f::translate(scene_rdl2::math::Vec4f(0.0f, 10.0f, 0.0f, 0.0f));
        scene_rdl2::math::Color color = 1.0f * scene_rdl2::math::Color(0.9f, 0.1f, 0.1f) * powf(2.0f, 2.0f);
        std::shared_ptr<Light> light = std::shared_ptr<Light>(
                new SphereLight(makeSphereLightSceneObject("SphereLight", &mContext,
                                xform, color, 2.0f, nullptr, false)));
        light->update(scene_rdl2::math::Mat4d(one));
        mLights.push_back(light);
        mLightSetA.push_back(light.get());
    }

    {
        // Rect Light
        scene_rdl2::math::Mat4f xform = scene_rdl2::math::Mat4f::rotate(scene_rdl2::math::Vec4f(1.0f, 0.0f, 0.0f, 0.0f), deg2rad(-90.0f))
            * scene_rdl2::math::Mat4f::translate(scene_rdl2::math::Vec4f(1.5f, 8.7f, 0.0f, 0.0f));
        scene_rdl2::math::Color color = 1.0f * scene_rdl2::math::Color(0.1f, 0.9f, 0.1f) * powf(2.0f, 4.0f);
        std::shared_ptr<Light> light = std::shared_ptr<Light>(
                new RectLight(makeRectLightSceneObject("RectLight", &mContext,
                              xform, color, 2.0f, 2.0f, nullptr, false)));
        light->update(scene_rdl2::math::Mat4d(one));
        mLights.push_back(light);
        mLightSetA.push_back(light.get());
    }

    {
        // Disk Light
        scene_rdl2::math::Mat4f xform = scene_rdl2::math::Mat4f::rotate(scene_rdl2::math::Vec4f(1.0f, 0.0f, 0.0f, 0.0f), deg2rad(-90.0f))
            * scene_rdl2::math::Mat4f::translate(scene_rdl2::math::Vec4f(-1.0f, 8.25f, 0.0f, 0.0f));
        scene_rdl2::math::Color color = 1.0f * scene_rdl2::math::Color(0.1f, 0.1f, 0.9f) * powf(2.0f, 4.0f);
        std::shared_ptr<Light> light = std::shared_ptr<Light>(
                new DiskLight(makeDiskLightSceneObject("DiskLight", &mContext,
                              xform, color, 1.5f, nullptr, false)));
        light->update(scene_rdl2::math::Mat4d(one));
        mLights.push_back(light);
        mLightSetA.push_back(light.get());
    }
}


void
TestLightSetSampler::tearDown()
{
    cleanupThreadLocalData();
}


void
TestLightSetSampler::testRadiance()
{
    LightSet lightSet;

    std::vector<const LightFilterList*> lightFilterLists(mLightSetA.size(), nullptr);
    lightSet.init(&mLightSetA[0], mLightSetA.size(), &lightFilterLists[0]);

    testLightSetRadiance(lightSet,
                         scene_rdl2::math::Vec3f(0.0f, 0.0f, 0.0f),
                         scene_rdl2::math::Vec3f(0.0f, 1.0f, 0.0f));
}


//----------------------------------------------------------------------------

} // namespace pbr
} // namespace moonray

CPPUNIT_TEST_SUITE_REGISTRATION(moonray::pbr::TestLightSetSampler);

