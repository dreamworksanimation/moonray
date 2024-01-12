// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

// 
// @file TestLightUtil.h
//

#pragma once
#include <cppunit/extensions/HelperMacros.h>
#include <cppunit/TestFixture.h>

namespace moonray {
namespace pbr {

//----------------------------------------------------------------------------

class TestLightUtil : public CppUnit::TestFixture
{
    CPPUNIT_TEST_SUITE(TestLightUtil);

#if 1
    CPPUNIT_TEST(testPlane);
    CPPUNIT_TEST(testFalloffCurve);
#endif

    CPPUNIT_TEST_SUITE_END();

public:
    void testPlane();
    void testFalloffCurve();
};

//----------------------------------------------------------------------------

} // namespace pbr
} // namespace moonray

