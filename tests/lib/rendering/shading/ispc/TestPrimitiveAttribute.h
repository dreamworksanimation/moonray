// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

/// @file TestPrimitiveAttribute.h
#pragma once

#include <moonray/rendering/shading/ispc/Shadingv.h>
#include <moonray/rendering/bvh/shading/ShadingTLState.h>
#include <cppunit/TestFixture.h>
#include <cppunit/extensions/HelperMacros.h>

namespace moonray {
namespace shading {
namespace unittest {

class TestPrimitiveAttribute : public CppUnit::TestFixture
{
public:
    void setUp();
    void tearDown();

    void testGetBoolAttribute();
    void testGetIntAttribute();
    void testGetFloatAttribute();
    void testGetColorAttribute();
    void testGetVec2fAttribute();
    void testGetVec3fAttribute();
    void testGetMat4fAttribute();

    CPPUNIT_TEST_SUITE(TestPrimitiveAttribute);
    CPPUNIT_TEST(testGetBoolAttribute);
    CPPUNIT_TEST(testGetIntAttribute);
    CPPUNIT_TEST(testGetFloatAttribute);
    CPPUNIT_TEST(testGetColorAttribute);
    CPPUNIT_TEST(testGetVec2fAttribute);
    CPPUNIT_TEST(testGetVec3fAttribute);
    CPPUNIT_TEST(testGetMat4fAttribute);
    CPPUNIT_TEST_SUITE_END();

private:
    Statev mStatev;
    shading::TLState *mTls;
};

} // namespace unittest
} // namespace shading
} // namespace moonray


