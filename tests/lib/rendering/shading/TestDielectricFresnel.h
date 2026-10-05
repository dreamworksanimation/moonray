// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cppunit/extensions/HelperMacros.h>
#include <cppunit/TestFixture.h>

namespace moonray {
namespace shading {

class TestDielectricFresnel : public CppUnit::TestFixture
{
public:
    CPPUNIT_TEST_SUITE(TestDielectricFresnel);
    CPPUNIT_TEST(testNormalIncidenceFromAir);
    CPPUNIT_TEST(testRoughDielectricEnergy);
    CPPUNIT_TEST_SUITE_END();

    void testNormalIncidenceFromAir();
    void testRoughDielectricEnergy();
};

} // namespace shading
} // namespace moonray
