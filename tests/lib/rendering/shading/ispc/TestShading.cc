// Copyright 2023 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

/// @file TestRender.cc

#include "TestShading.h"
#include "testrender/TestRender.h"

#include <moonray/common/time/Timer.h>
#include <moonray/rendering/bvh/shading/ShadingTLState.h>
#include <moonray/rendering/rndr/RenderDriver.h>
#include <moonray/rendering/texturing/sampler/TextureSampler.h>
#include <scene_rdl2/scene/rdl2/rdl2.h>

#include <fstream>

using namespace scene_rdl2;
using namespace moonray;
using namespace moonray::shading;
using moonray::shading::unittest::TestShading;

static void
update(scene_rdl2::rdl2::Material *mat, int testNum)
{
    double start = time::getTime();
    mat->applyUpdates();
    double end = time::getTime();
    std::cerr << "\nrender test " << testNum << " update time = "
              << end - start << '\n';
}

static void
render(const scene_rdl2::rdl2::Material *mat, int testNum,
       int width = 256, int height = 256,
       int raysPerPixel = 1,
       bool primeThePump = false)
{
    // allocate results buffer and call the test renderer
    const int bufferSize = 3 * width * height;
    uint8_t results[bufferSize];

    double start = time::getTime();
    uint64 shaderTicks = TestRender::render(mat, results, width, height, raysPerPixel, primeThePump);
    double end = time::getTime();
    std::cerr << "render test " << testNum << " render time = "
              << end - start << '\n';
    std::cerr << "render test " << testNum << " shader ticks = "
              << shaderTicks << '\n';

    // write the results as a ppm
    {
        std::ofstream ppm;
        std::stringstream filename;
        filename << "TestRender_result" << testNum << ".ppm";
        ppm.open(filename.str().c_str());
        ppm << "P6\n";
        ppm << width << " " << height << std::endl;
        const int maxPixel = 255; // 8 bit gamma corrected color
        ppm << maxPixel << std::endl;
        ppm.write((const char *) results, bufferSize * sizeof(int8_t));
        ppm.close();
    }

    // compare with the canonical
    std::ifstream ppm;
    uint8_t canonical[bufferSize];
    std::stringstream filename;
    filename << "ref/TestRender_canonical" << testNum << ".ppm";
    ppm.open(filename.str().c_str());
    ppm.getline((char *) canonical, bufferSize); // P6\0
    ppm.getline((char *) canonical, bufferSize); // width height
    ppm.getline((char *) canonical, bufferSize); // max white
    ppm.read((char *) canonical, bufferSize);
    // allow +/-1 code of tolerance
    uint8_t *resPtr = results;
    uint8_t *canPtr = canonical;
    for (int i = 0; i < bufferSize; ++i) {
        CPPUNIT_ASSERT(abs(*resPtr - *canPtr) <= 1);
        resPtr++;
        canPtr++;
    }
}

void
TestShading::render0()
{
    // build rdl2 scene - default base
    scene_rdl2::rdl2::SceneContext ctx;
    ctx.setDsoPath(ctx.getDsoPath() + ":" + RDL2DSO_PATH + ":dso/material");

#ifdef USE_DEBUG_MAP
    ctx.setDsoPath(ctx.getDsoPath() + ":dso/map");
    scene_rdl2::rdl2::Map *debugMap = nullptr;
    if (1) {
        debugMap = ctx.createSceneObject("TestDebugMap", "/debugMap")->asA<scene_rdl2::rdl2::Map>();
        scene_rdl2::rdl2::SceneObject::UpdateGuard update(debugMap);
        debugMap->set("map type", 2); // N
    }
#endif

    scene_rdl2::rdl2::Material *material =
        ctx.createSceneObject("BaseMaterial", "/base")->asA<scene_rdl2::rdl2::Material>();
    {
#ifdef USE_DEBUG_MAP
        if (debugMap) {
            material->set("emission factor", 1.f);
            material->setBinding("emission color", debugMap);
            material->set("diffuse color", scene_rdl2::rdl2::Rgb(0., 0., 0.));
            material->set("specular color", scene_rdl2::rdl2::Rgb(0., 0., 0.));
        }
#endif
    }

    update(material, 0);
    render(material, 0);
}

void
TestShading::render2()
{
    // initRenderDriver should only be called once per application.
    if (!rndr::getRenderDriver()) {

        // Create arena block pool which is shared between all threads.
        scene_rdl2::util::Ref<scene_rdl2::alloc::ArenaBlockPool> arenaBlockPool = scene_rdl2::util::alignedMallocCtorArgs<scene_rdl2::alloc::ArenaBlockPool>(CACHE_LINE_SIZE);

        // Create a RenderDriver.
        mcrt_common::TLSInitParams initParams;
        initParams.mUnitTests = true;
        initParams.mArenaBlockPool = arenaBlockPool.get();
        rndr::initRenderDriver(initParams);
    }

    // build rdl2 scene -
    //   test map shader on diffuse channel
    scene_rdl2::rdl2::SceneContext ctx;
    ctx.setDsoPath(ctx.getDsoPath() + ":" + RDL2DSO_PATH + ":dso/material:dso/map");

    scene_rdl2::rdl2::Map *checkers =
        ctx.createSceneObject("TestCheckerMap", "/checkers")->asA<scene_rdl2::rdl2::Map>();
    {
        scene_rdl2::rdl2::SceneObject::UpdateGuard update(checkers);
    }

    scene_rdl2::rdl2::Material *material =
        ctx.createSceneObject("BaseMaterial", "/base")->asA<scene_rdl2::rdl2::Material>();
    {
        scene_rdl2::rdl2::SceneObject::UpdateGuard update(material);
        material->setBinding("diffuse color", checkers);
    }
    update(material, 2);
    render(material, 2, 256, 256, 64, true);
}

void
TestShading::render3()
{
    // build rdl2 scene -
    //   test mandelbrot map shader on diffuse channel
    scene_rdl2::rdl2::SceneContext ctx;
    ctx.setDsoPath(ctx.getDsoPath() + ":" + RDL2DSO_PATH + ":dso/material:dso/map");

    scene_rdl2::rdl2::Map *map =
        ctx.createSceneObject("TestMandelbrot", "/mandelbrot")->asA<scene_rdl2::rdl2::Map>();
    {
        scene_rdl2::rdl2::SceneObject::UpdateGuard update(map);
        map->set("offset", scene_rdl2::rdl2::Vec2f(.25f, 0.f));
        map->set("scale", scene_rdl2::rdl2::Vec2f(2.f, 1.f));
        map->set("number of colors", 200);
        map->set("iterations", 1000);
    }

    scene_rdl2::rdl2::Material *material =
        ctx.createSceneObject("BaseMaterial", "/base")->asA<scene_rdl2::rdl2::Material>();
    {
        scene_rdl2::rdl2::SceneObject::UpdateGuard update(material);
        material->setBinding("diffuse color", map);
        material->set("specular color", scene_rdl2::rdl2::Rgb(0.1f, 0.1f, 0.1f));
    }
    update(material, 3);
    render(material, 3, 256, 256, 64);
}

