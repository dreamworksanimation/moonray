// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0


#pragma once

#include <cppunit/TestFixture.h>
#include <cppunit/extensions/HelperMacros.h>

namespace moonray {
namespace rndr {
namespace unittest {

class TestActivePixelMask : public CppUnit::TestFixture
{
public:
    void testAccess();
    void testFill();
    void testDilate();

    CPPUNIT_TEST_SUITE(TestActivePixelMask);
    CPPUNIT_TEST(testAccess);
    CPPUNIT_TEST(testFill);
    CPPUNIT_TEST(testDilate);
    CPPUNIT_TEST_SUITE_END();
};

} // namespace unittest
} // namespace rndr
} // namespace moonray

