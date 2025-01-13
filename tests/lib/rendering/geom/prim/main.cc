// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

#include "TestPrimAttr.h"
#include "TestInterpolator.h"
#include <moonray/rendering/mcrt_common/ThreadLocalState.h>
#include <scene_rdl2/pdevunit/pdevunit.h>
#include <tbb/task_scheduler_init.h>
int
main(int argc, char *argv[])
{
    // Create arena block pool which is shared between all threads.
    scene_rdl2::util::Ref<scene_rdl2::alloc::ArenaBlockPool> arenaBlockPool =
        scene_rdl2::util::alignedMallocCtorArgs<scene_rdl2::alloc::ArenaBlockPool>(CACHE_LINE_SIZE);

    moonray::mcrt_common::TLSInitParams initParams;
    initParams.mUnitTests = true;
    initParams.mDesiredNumTBBThreads = tbb::task_scheduler_init::default_num_threads();
    initParams.mArenaBlockPool = arenaBlockPool.get();
    moonray::mcrt_common::initTLS(initParams);

    CPPUNIT_TEST_SUITE_REGISTRATION(moonray::geom::unittest::TestRenderingPrimAttr);
    CPPUNIT_TEST_SUITE_REGISTRATION(moonray::geom::unittest::TestInterpolator);

    int result = pdevunit::run(argc, argv);
    moonray::mcrt_common::cleanUpTLS();
    return result;
}


