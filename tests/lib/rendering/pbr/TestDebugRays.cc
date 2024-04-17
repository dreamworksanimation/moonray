// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

#include "TestDebugRays.h"
#include <scene_rdl2/common/math/MathUtil.h>
#include <scene_rdl2/render/util/Random.h>
#include <moonray/common/mcrt_util/StaticVector.h>
#include <moonray/common/mcrt_macros/moonray_static_check.h>

#include <scene_rdl2/common/math/MathUtil.h>

#define WIDTH                       320
#define HEIGHT                      240
#define NUM_RAY_RECORDERS           13
#define NUM_VERTICES_PER_RECORDER   6789

namespace moonray {
namespace pbr {

using namespace scene_rdl2::math;

void
TestDebugRays::setUp()
{
    mDb = nullptr;

    //
    // record rays
    //
    pbr::DebugRayRecorder::enableRecording(
        scene_rdl2::math::BBox2i(
            scene_rdl2::math::Vec2i(0, 0), scene_rdl2::math::Vec2i(WIDTH, HEIGHT)),
        scene_rdl2::math::Mat4f(scene_rdl2::math::one));

    util::StaticVector<DebugRayRecorder, NUM_RAY_RECORDERS> recorders;
    for (int i = 0; i < NUM_RAY_RECORDERS; ++i) {
        recorders.emplace_back(i);
    }

    std::vector<DebugRayRecorder *> recorderList;
    for (size_t i = 0; i < NUM_RAY_RECORDERS; ++i) {
        populateRecorder(&recorders[i]);
        recorderList.push_back(&recorders[i]);
    }

    DebugRayRecorder::disableRecording();

    //
    // build
    //
    DebugRayBuilder builder;
    builder.build(WIDTH, HEIGHT, recorderList);

    //
    // export db
    //
    mDb = new DebugRayDatabase;
    builder.exportDatabase(mDb);
}

void
TestDebugRays::tearDown()
{
    delete mDb;
    mDb = nullptr;
}

void
TestDebugRays::testSortOrder()
{
    CPPUNIT_ASSERT(mDb);

    DebugRayDatabase::Iterator it(*mDb);

    unsigned screenOffset = 0;

    while (!it.isDone()) {

        DebugRayVertex const *vertex = &(*it);

        unsigned rayScreenOffset = vertex->mScreenY * WIDTH + vertex->mScreenX;

        CPPUNIT_ASSERT(rayScreenOffset >= screenOffset);
        screenOffset = rayScreenOffset;

        ++it;
    }
}

//----------------------------------------------------------------------------

void
TestDebugRays::testPrimaryRays()
{
    CPPUNIT_ASSERT(mDb);

    std::vector<uint32_t> const &primaryRays = mDb->getPrimaryRayIndices();
    size_t numPrimaryRays = primaryRays.size();
    CPPUNIT_ASSERT(numPrimaryRays);

    // count the number of primary rays using brute force iteration
    size_t numPrimaryRays2 = 0;
    DebugRayDatabase::Iterator it(*mDb);
    while (!it.isDone()) {
        DebugRayVertex const *vertex = &(*it);
        if (vertex->isRoot())
            ++numPrimaryRays2;
        ++it;
    }

    // iterate through the db and verify the mNumSubnodes member is correct,
    // this is the number of children at this particular subtree including the
    // current node

    // get first primary ray via iteration
    DebugRayVertex const *startVert = &(*DebugRayDatabase::Iterator(*mDb));
    DebugRayVertex const *currVert = startVert;
    CPPUNIT_ASSERT(startVert == mDb->getRay(primaryRays[0]));

    for (size_t i = 0; i < numPrimaryRays; ++i, ++currVert) {

        CPPUNIT_ASSERT(currVert->isRoot());

        // make sure primary indices are consistent with list of rays
        size_t ofs = currVert - startVert;
        CPPUNIT_ASSERT(ofs == primaryRays[i]);

        // loop over all children
        uint32_t numSubnodes = currVert->mNumSubnodes;
        CPPUNIT_ASSERT(numSubnodes >= 1);

        for (uint32_t j = 1; j < numSubnodes; ++j) {
            ++currVert;
            CPPUNIT_ASSERT(!currVert->isRoot());
        }
    }
}

//----------------------------------------------------------------------------

void
TestDebugRays::testRectFilter()
{
    CPPUNIT_ASSERT(mDb);

    int minX = 123;
    int maxX = 232;
    int minY = 12;
    int maxY = 121;

    // do manual iteration to get baseline count of ray vertices
    unsigned refResult = 0;

    {
        DebugRayDatabase::Iterator it(*mDb);
        while (!it.isDone()) {

            DebugRayVertex const *vertex = &(*it);

            if (vertex->mScreenX >= minX && vertex->mScreenX <= maxX &&
                vertex->mScreenY >= minY && vertex->mScreenY <= maxY)
                ++refResult;

            ++it;
        }
    }

    // verify that we get the same results when using a filter
    unsigned filterResult = 0;

    {
        DebugRayFilter filter;
        filter.mViewport = scene_rdl2::math::BBox2i(scene_rdl2::math::Vec2i(minX, minY), scene_rdl2::math::Vec2i(maxX, maxY));

        DebugRayDatabase::Iterator it(*mDb, &filter);

        while (!it.isDone()) {
            ++filterResult;
            ++it;
        }
    }

    CPPUNIT_ASSERT(refResult == filterResult);
}

//----------------------------------------------------------------------------

void
TestDebugRays::testTagFilter()
{
    CPPUNIT_ASSERT(mDb);

    // do manual iteration to get baseline count of tagged rays
    unsigned refResultAny = 0;
    unsigned refResultAll = 0;

    // pick some arbitrary tag
    uint32_t tag = 0x14200;

    {
        DebugRayDatabase::Iterator it(*mDb);
        while (!it.isDone()) {
            DebugRayVertex const *vertex = &(*it);

            if (vertex->mUserTags & tag)
                ++refResultAny;

            if ((vertex->mUserTags & tag) == tag)
                ++refResultAll;

            ++it;
        }
    }

    // verify that we get the same results when using a filter
    unsigned filterResultAny = 0;
    unsigned filterResultAll = 0;

    {
        DebugRayFilter filter;
        filter.mTags = tag;

        filter.mMatchAll = false;
        DebugRayDatabase::Iterator it(*mDb, &filter);
        while (!it.isDone()) {
            ++filterResultAny;
            ++it;
        }

        filter.mMatchAll = true;
        it = DebugRayDatabase::Iterator(*mDb, &filter);
        while (!it.isDone()) {
            ++filterResultAll;
            ++it;
        }

    }

    CPPUNIT_ASSERT(refResultAny == filterResultAny);
    CPPUNIT_ASSERT(refResultAll == filterResultAll);
}

//----------------------------------------------------------------------------

void
TestDebugRays::testSerialization()
{
    CPPUNIT_ASSERT(mDb);

    char const *filename = "unittest.raydb";

    // save mDb
    CPPUNIT_ASSERT(mDb->save(filename));

    // load back into a duplicate db
    DebugRayDatabase dupeDb;
    CPPUNIT_ASSERT(dupeDb.load(filename));

    // do some tests to verify we're dealing with identical data
    DebugRayDatabase::Iterator refIt(*mDb);
    DebugRayDatabase::Iterator dupeIt(dupeDb);

    while (!refIt.isDone()) {
        CPPUNIT_ASSERT(!dupeIt.isDone());
        CPPUNIT_ASSERT(memcmp(&(*refIt), &(*dupeIt), sizeof(DebugRayVertex)) == 0);
        ++refIt;
        ++dupeIt;
    }

    CPPUNIT_ASSERT(dupeIt.isDone());

    CPPUNIT_ASSERT(mDb->getPrimaryRayIndices() == dupeDb.getPrimaryRayIndices());
}

//----------------------------------------------------------------------------

void
TestDebugRays::populateRecorder(DebugRayRecorder *recorder)
{
    recorder->record();

    static uint32_t seed = 0x0eadbeef;
    scene_rdl2::util::Random rnd(seed);

    MOONRAY_THREADSAFE_STATIC_WRITE(++seed);

    DebugRayVertex *parent = nullptr;
    uint32_t sx = rnd.getNextUInt(WIDTH);
    uint32_t sy = rnd.getNextUInt(HEIGHT);
    for (size_t i = 0; i < NUM_VERTICES_PER_RECORDER; ++i) {

        uint32_t tags = rnd.getNextUInt();

        DebugRayVertex *vertex = nullptr;
        if (!parent) {
            parent = recorder->startNewRay(Vec3f(scene_rdl2::math::zero), sx, sy, tags);
        }
        else {
            vertex = recorder->extendRay(parent);

            // in a real app, we'd fill in the all the rest of the
            // vertex user fields here also
            vertex->mUserTags = tags;
        }

        float x = rnd.getNextFloat();
        if (x < 0.05f || parent->mDepth == 254) {
            // start new ray
            parent = nullptr;
            sx = rnd.getNextUInt(WIDTH);
            sy = rnd.getNextUInt(HEIGHT);
        } else if (x < 0.6f) {
            // continue ray
            parent = vertex;
        } else {
            // leave parent alone which will create a sibling
        }
    }

    recorder->stopRecording();
}

} // namespace pbr
} // namespace moonray

CPPUNIT_TEST_SUITE_REGISTRATION(moonray::pbr::TestDebugRays);

