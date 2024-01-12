// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0


#pragma once

#include <cppunit/TestFixture.h>
#include <cppunit/extensions/HelperMacros.h>

namespace moonray {
namespace rndr {
namespace unittest {

class TestOverlappingRegions : public CppUnit::TestFixture
{
public:
    void testRegions();

    CPPUNIT_TEST_SUITE(TestOverlappingRegions);
    CPPUNIT_TEST(testRegions);
    CPPUNIT_TEST_SUITE_END();
};

} // namespace unittest
} // namespace rndr
} // namespace moonray

