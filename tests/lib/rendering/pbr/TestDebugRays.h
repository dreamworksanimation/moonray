// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <moonray/rendering/pbr/core/DebugRay.h>
#include <cppunit/extensions/HelperMacros.h>
#include <cppunit/TestFixture.h>

namespace moonray {
namespace pbr {

class TestDebugRays : public CppUnit::TestFixture
{
    CPPUNIT_TEST_SUITE(TestDebugRays);
#if 1
    CPPUNIT_TEST(testSortOrder);
    CPPUNIT_TEST(testPrimaryRays);
    CPPUNIT_TEST(testRectFilter);
    CPPUNIT_TEST(testTagFilter);
    CPPUNIT_TEST(testSerialization);
#endif
    CPPUNIT_TEST_SUITE_END();

public:
    void setUp();
    void tearDown();

    void testSortOrder();
    void testPrimaryRays();
    void testRectFilter();
    void testTagFilter();
    void testSerialization();

private:
    void populateRecorder(DebugRayRecorder *recorder);

	DebugRayDatabase *mDb;
};

} // namespace pbr
} // namespace moonray

