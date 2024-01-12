// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0


#pragma once

#include <cppunit/TestFixture.h>
#include <cppunit/extensions/HelperMacros.h>

namespace moonray_stats {
namespace unittest {

class TestFormatters : public CppUnit::TestFixture
{
public:
    void setUp();
    void tearDown();

    void testBytes();
    void testTime();

    CPPUNIT_TEST_SUITE(TestFormatters);
    CPPUNIT_TEST(testBytes);
    CPPUNIT_TEST(testTime);
    CPPUNIT_TEST_SUITE_END();
};

} // namespace unittest
} // namespace moonray_stats

