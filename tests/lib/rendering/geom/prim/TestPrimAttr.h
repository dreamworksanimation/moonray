// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "TestPrimUtils.h"

#include <cppunit/extensions/HelperMacros.h>
#include <scene_rdl2/scene/rdl2/SceneContext.h>

namespace moonray {
namespace geom {
namespace unittest {

class TestRenderingPrimAttr : public CppUnit::TestFixture
{
public:
    void setUp();
    void tearDown();

    CPPUNIT_TEST_SUITE(TestRenderingPrimAttr);
    CPPUNIT_TEST(testConstant);
    CPPUNIT_TEST(testPart0);
    CPPUNIT_TEST(testPart1);
    CPPUNIT_TEST(testPart2);
    CPPUNIT_TEST(testUniform0);
    CPPUNIT_TEST(testUniform1);
    CPPUNIT_TEST(testUniform2);
    CPPUNIT_TEST(testFaceVarying0);
    CPPUNIT_TEST(testFaceVarying1);
    CPPUNIT_TEST(testFaceVarying2);
    CPPUNIT_TEST(testFaceVarying3);
    CPPUNIT_TEST(testVarying0);
    CPPUNIT_TEST(testVertex0);
    CPPUNIT_TEST(testVertex1);
    CPPUNIT_TEST(testVertex2);
    CPPUNIT_TEST(testTime0);
    CPPUNIT_TEST(testMixing0);
    CPPUNIT_TEST(testTransformAttributes0);
    CPPUNIT_TEST(testTransformAttributes1);
    CPPUNIT_TEST(testTransformAttributes2);
    CPPUNIT_TEST(testInstanceAttributes);
    CPPUNIT_TEST_SUITE_END();

    void testConstant();
    void testPart0();
    void testPart1();
    void testPart2();
    void testUniform0();
    void testUniform1();
    void testUniform2();
    void testFaceVarying0();
    void testFaceVarying1();
    void testFaceVarying2();
    void testFaceVarying3();
    void testVarying0();
    void testVertex0();
    void testVertex1();
    void testVertex2();
    void testTime0();
    void testMixing0();
    void testTransformAttributes0();
    void testTransformAttributes1();
    void testTransformAttributes2();
    void testInstanceAttributes();

private:
    scene_rdl2::math::Color mC0;
    scene_rdl2::math::Color mC1;
    scene_rdl2::math::Color mC2;
    scene_rdl2::math::Color mC3;
    float mF0;
    float mF1;
    float mF2;
    float mF3;
    std::string mS0;
    std::string mS1;
    RNG mRNG;
};

} // namespace unittest 
} // namespace geom
} // namespace moonray

