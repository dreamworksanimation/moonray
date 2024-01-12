// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

///
/// @file TestInterpolator
/// $Id$
///

#pragma once
#include "TestPrimUtils.h"
#include <cppunit/extensions/HelperMacros.h>

#include <scene_rdl2/scene/rdl2/SceneContext.h>
#include <moonray/rendering/geom/prim/Mesh.h>

namespace moonray {
namespace geom {
namespace unittest {

class TestInterpolator : public CppUnit::TestFixture
{
public:
    void setUp();
    void tearDown();

    CPPUNIT_TEST_SUITE(TestInterpolator);
    CPPUNIT_TEST(testCurvesConstant);
    CPPUNIT_TEST(testCurvesConstantMB);
    CPPUNIT_TEST(testCurvesUniform);
    CPPUNIT_TEST(testCurvesUniformMB);
    CPPUNIT_TEST(testCurvesVarying);
    CPPUNIT_TEST(testCurvesVaryingMB);
    CPPUNIT_TEST(testCurvesFaceVarying);
    CPPUNIT_TEST(testCurvesFaceVaryingMB);
    CPPUNIT_TEST(testCurvesVertex);
    CPPUNIT_TEST(testCurvesVertexMB);
    CPPUNIT_TEST(testMeshConstant);
    CPPUNIT_TEST(testMeshConstantMB);
    CPPUNIT_TEST(testMeshPart);
    CPPUNIT_TEST(testMeshPartMB);
    CPPUNIT_TEST(testMeshUniform);
    CPPUNIT_TEST(testMeshUniformMB);
    CPPUNIT_TEST(testMeshVarying);
    CPPUNIT_TEST(testMeshVaryingMB);
    CPPUNIT_TEST(testMeshFaceVarying);
    CPPUNIT_TEST(testMeshFaceVaryingMB);
    CPPUNIT_TEST(testMeshVertex);
    CPPUNIT_TEST(testMeshVertexMB);
    CPPUNIT_TEST_SUITE_END();

    void testCurvesConstant();
    void testCurvesConstantMB();
    void testCurvesUniform();
    void testCurvesUniformMB();
    void testCurvesVarying();
    void testCurvesVaryingMB();
    void testCurvesFaceVarying();
    void testCurvesFaceVaryingMB();
    void testCurvesVertex();
    void testCurvesVertexMB();
    void testMeshConstant();
    void testMeshConstantMB();
    void testMeshPart();
    void testMeshPartMB();
    void testMeshUniform();
    void testMeshUniformMB();
    void testMeshVarying();
    void testMeshVaryingMB();
    void testMeshFaceVarying();
    void testMeshFaceVaryingMB();
    void testMeshVertex();
    void testMeshVertexMB();

private:
    RNG mRNG;
};

} // namespace unittest
} // namespace geom
} // namespace moonray


