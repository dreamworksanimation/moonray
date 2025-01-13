// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

/// @file main.cc

#include "TestBase.h"
#include "TestDisplace.h"
#include "TestEvalAttribute.h"
#include "TestPrimitiveAttribute.h"

#include <moonray/common/mcrt_macros/moonray_static_check.h>
#include <moonray/rendering/bvh/shading/ShadingTLState.h>
#include <moonray/rendering/mcrt_common/ThreadLocalState.h>

#include <cppunit/TestFixture.h>
#include <cppunit/extensions/HelperMacros.h>
#include <scene_rdl2/pdevunit/pdevunit.h>

int
main(int argc, char* argv[])
{
    using namespace moonray;
    using namespace moonray::shading::unittest;

    // Create arena block pool which is shared between all threads.
    scene_rdl2::util::Ref<scene_rdl2::alloc::ArenaBlockPool> arenaBlockPool =
        scene_rdl2::util::alignedMallocCtorArgs<scene_rdl2::alloc::ArenaBlockPool>(CACHE_LINE_SIZE);

    // Initialize TLS shared by all tests.
    mcrt_common::TLSInitParams initParams;
    initParams.mUnitTests = true;
    initParams.mArenaBlockPool = arenaBlockPool.get();
    initParams.initShadingTls = shading::TLState::allocTls;
    initParams.initTLSTextureSupport = shading::initTexturingSupport;
    initParams.mDesiredNumTBBThreads = 1;
    mcrt_common::initTLS(initParams);

    // Register shading unit tests.
    CPPUNIT_TEST_SUITE_REGISTRATION(TestBase);
    CPPUNIT_TEST_SUITE_REGISTRATION(TestDisplace);
    CPPUNIT_TEST_SUITE_REGISTRATION(TestEvalAttribute);
    CPPUNIT_TEST_SUITE_REGISTRATION(TestPrimitiveAttribute);

    MOONRAY_START_NON_THREADSAFE_STATIC_WRITE
    MOONRAY_FINISH_NON_THREADSAFE_STATIC_WRITE

    // Run all tests.
    int result = pdevunit::run(argc, argv);

    // Clean up TLS
    mcrt_common::cleanUpTLS();

    return result;
}

