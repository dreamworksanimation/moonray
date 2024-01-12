// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

/// @file TestRender.cc

#include "TestRender.h"
#include "TestRender_ispc_stubs.h"

#include <fstream>

#include <moonray/common/time/Timer.h>
#include <cppunit/extensions/HelperMacros.h>
#include <moonray/rendering/bvh/shading/ShadingTLState.h>
#include <scene_rdl2/render/util/Arena.h>

using namespace scene_rdl2;

namespace moonray {
namespace shading {

class TLState;

void
TestRender::update(rdl2::Material *mat, int testNum)
{
    double start = time::getTime();

    mat->applyUpdates();
    
    double end = time::getTime();
    std::cerr << "\nrender test " << testNum << " update time = "
              << end - start << '\n';
}
uint64
TestRender::renderAndCompare(const rdl2::Material *mat,
                             int testNum,
                             int width,
                             int height,
                             int raysPerPixel,
                             bool isIndirect)
{
    // allocate results buffer and call the test renderer
    const int bufferSize = 3 * width * height;
    uint8_t results[bufferSize];

    double start = time::getTime();
    uint64 shaderTicks = render(mat, results, width, height, raysPerPixel, false, isIndirect);
    double end = time::getTime();

    std::cerr << "render test " << testNum << " render time = "
              << end - start << '\n';
    std::cerr << "render test " << testNum << " shader ticks = "
              << shaderTicks << '\n';
    
    // write the results as a ppm
    {
        std::ofstream ppm;
        std::stringstream filename;
        filename << "/tmp/TestRender_result" << testNum << ".ppm";
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
    
    return shaderTicks;
}

uint64
TestRender::render(
       const rdl2::Material *mat,
       uint8_t *results,
       int width, int height,
       int raysPerPixel,
       bool primeThePump,
       bool isIndirect)
{
    mcrt_common::ThreadLocalState *tls = mcrt_common::getFrameUpdateTLS();
    shading::TLState *shadingTls = MNRY_VERIFY(tls->mShadingTls.get());

    // initialize for texturing support
    shadingTls->initTexturingSupport();

    // setup the test render scene options
    ispc::Scene scene;
    scene.mMat = (const ispc::Material *) mat;
    scene.mWidth = width;
    scene.mHeight = height;
    scene.mRaysPerPixel = raysPerPixel;
    scene.mShadingTLState = (ispc::ShadingTLState *)shadingTls;

    // call ispc render function
    if (primeThePump) {
        ispc::render(&scene, results, isIndirect);
    }
    
    uint64 shaderTicks = ispc::render(&scene, results, isIndirect);

    return shaderTicks;
}

}
}


