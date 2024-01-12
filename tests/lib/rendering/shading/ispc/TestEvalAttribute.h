// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

/// @file TestEvalAttribute.h
#pragma once

#include <cppunit/TestFixture.h>
#include <cppunit/extensions/HelperMacros.h>

namespace moonray {
namespace shading {
namespace unittest {

class TestEvalAttribute : public CppUnit::TestFixture
{
public:
    void testEvalColor();
    void testEvalColorComponent();
    void testGetBool();
    void testGetInt();
    void testGetFloat();
    void testGetColor();
    void testGetVec2f();
    void testGetVec3f();
    void testEvalAttrFloat();
    void testEvalAttrColor();
    void testEvalNormal();
    void testEvalAttrVec3f();
    void testEvalAttrVec2f();
    

    CPPUNIT_TEST_SUITE(TestEvalAttribute);
    CPPUNIT_TEST(testEvalColor);
    CPPUNIT_TEST(testEvalColorComponent);
    CPPUNIT_TEST(testGetBool);
    CPPUNIT_TEST(testGetInt);
    CPPUNIT_TEST(testGetFloat);
    CPPUNIT_TEST(testGetColor);
    CPPUNIT_TEST(testGetVec2f);
    CPPUNIT_TEST(testGetVec3f);
    CPPUNIT_TEST(testEvalAttrFloat);
    CPPUNIT_TEST(testEvalAttrColor);
    CPPUNIT_TEST(testEvalNormal);
    CPPUNIT_TEST(testEvalAttrVec3f);
    CPPUNIT_TEST(testEvalAttrVec2f);
    CPPUNIT_TEST_SUITE_END();
};

} // namespace unittest
} // namespace shading
} // namespace moonray


