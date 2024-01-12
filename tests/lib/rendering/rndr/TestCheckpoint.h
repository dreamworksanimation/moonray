// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

//
//
#pragma once

#include <cppunit/TestFixture.h>
#include <cppunit/extensions/HelperMacros.h>

namespace moonray {
namespace rndr {
namespace unittest {
    
class TestCheckpoint : public CppUnit::TestFixture
{
public:
    void testKJSequenceTable();
    void testTotalCheckpointToQualitySteps();

    CPPUNIT_TEST_SUITE(TestCheckpoint);
    CPPUNIT_TEST(testKJSequenceTable);
    CPPUNIT_TEST(testTotalCheckpointToQualitySteps);
    CPPUNIT_TEST_SUITE_END();
};

} // namespace unittest
} // namespace rndr
} // namespace moonray

