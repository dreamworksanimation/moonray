// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cppunit/extensions/HelperMacros.h>

namespace moonray {
namespace geom {
namespace unittest {

class TestGeomApi : public CppUnit::TestFixture
{
public:
    void setUp();
    void tearDown();

    CPPUNIT_TEST_SUITE(TestGeomApi);
    CPPUNIT_TEST(testLayerAssignmentId);
    CPPUNIT_TEST(testPrimitiveAttribute);
    CPPUNIT_TEST(testVertexBufferAlignment);
    CPPUNIT_TEST(testVertexBufferVec3fa0);
    CPPUNIT_TEST(testVertexBufferVec3fa1);
    CPPUNIT_TEST(testVertexBufferVec3f0);
    CPPUNIT_TEST(testVertexBufferVec3f1);
    CPPUNIT_TEST(testVertexBufferVec3f2);
    CPPUNIT_TEST(testVertexBufferResize);
    CPPUNIT_TEST(testVertexBufferClear);
    CPPUNIT_TEST(testVertexBufferAppend);
    CPPUNIT_TEST_SUITE_END();

    void testLayerAssignmentId();
    void testPrimitiveAttribute();
    void testVertexBufferAlignment();
    void testVertexBufferVec3fa0();
    void testVertexBufferVec3fa1();
    void testVertexBufferVec3f0();
    void testVertexBufferVec3f1();
    void testVertexBufferVec3f2();
    void testVertexBufferResize();
    void testVertexBufferClear();
    void testVertexBufferAppend();

};

} // namespace unittest
} // namespace geom
} // namespace moonray

