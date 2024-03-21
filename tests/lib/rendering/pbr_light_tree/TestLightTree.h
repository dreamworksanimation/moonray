// Copyright 2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <moonray/rendering/pbr/light/LightTreeUtil.h>

#include <cppunit/extensions/HelperMacros.h>
#include <cppunit/TestFixture.h>

namespace moonray {
namespace pbr {

//----------------------------------------------------------------------------

class TestLightTree : public CppUnit::TestFixture
{
    CPPUNIT_TEST_SUITE(TestLightTree);

    CPPUNIT_TEST(testCone);

    CPPUNIT_TEST_SUITE_END();

public:
    void testCone();
};

//----------------------------------------------------------------------------

} // namespace pbr
} // namespace moonray

