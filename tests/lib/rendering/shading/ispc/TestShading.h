// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

/// @file TestShading.h
#pragma once

#include <cppunit/TestFixture.h>
#include <cppunit/extensions/HelperMacros.h>

namespace moonray {
namespace shading {
namespace unittest {

class TestShading : public CppUnit::TestFixture
{
public:
    void render0();
    void render2();
    void render3();

    CPPUNIT_TEST_SUITE(TestShading);
    CPPUNIT_TEST(render0);
    CPPUNIT_TEST(render2);
    //CPPUNIT_TEST(render3);  <-----  Temporarily suppress this test for MOONRAY-4960
    CPPUNIT_TEST_SUITE_END();
};

} // namespace unittest
} // namespace shading
} // namespace moonray


