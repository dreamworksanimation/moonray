// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0


#pragma once

#include <scene_rdl2/scene/rdl2/rdl2.h>

#include <cppunit/TestFixture.h>
#include <cppunit/extensions/HelperMacros.h>

namespace moonray {
namespace unittest {

class TestSceneContext : public CppUnit::TestFixture
{
public:
    void setUp();
    void tearDown();

    /// Test that the update mechanism works.
    void testUpdate();

    CPPUNIT_TEST_SUITE(TestSceneContext);
#if 1
    CPPUNIT_TEST(testUpdate);
#endif
    CPPUNIT_TEST_SUITE_END();
};

} // namespace unittest
} // namespace moonray

