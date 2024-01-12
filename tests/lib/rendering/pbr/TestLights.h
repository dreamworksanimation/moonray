// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

// 
// @file TestLights.h
//

#pragma once
#include "LightTesters.h"
#include "TestUtil.h"
#include <cppunit/extensions/HelperMacros.h>
#include <cppunit/TestFixture.h>
#include <scene_rdl2/scene/rdl2/rdl2.h>
#include <scene_rdl2/render/util/Random.h>

namespace moonray {
namespace pbr {

typedef std::vector<std::shared_ptr<LightTester>> LightTesters;
typedef void (*TestFunction)(const scene_rdl2::math::Vec3f &, const scene_rdl2::math::Vec3f &,
                             const LightTester *, int32_t initialSeed);

// prototypes of test functions
void testLightRadiance(const scene_rdl2::math::Vec3f &p, const scene_rdl2::math::Vec3f &n,
                       const LightTester *light_tester, int32_t initialSeed);
void testLightPDF(const scene_rdl2::math::Vec3f &p, const scene_rdl2::math::Vec3f &n,
                  const LightTester *light_tester, int32_t initialSeed);
void testLightCanIlluminate(const scene_rdl2::math::Vec3f &p, const scene_rdl2::math::Vec3f &n,
                            const LightTester *light_tester, int32_t initialSeed);
void testLightIntersection(const scene_rdl2::math::Vec3f &p, const scene_rdl2::math::Vec3f &n,
                           const LightTester *light_tester, int32_t initialSeed);
void testMeshLightFaceSelectionPDF(const scene_rdl2::math::Vec3f &p, const scene_rdl2::math::Vec3f &n,
                          const LightTester *light_tester, int32_t initialSeed);

//----------------------------------------------------------------------------

class TestLights : public CppUnit::TestFixture
{
    CPPUNIT_TEST_SUITE(TestLights);

#if 1
    CPPUNIT_TEST(testRectLightRadiance);
    CPPUNIT_TEST(testRectLightPDF);
    CPPUNIT_TEST(testRectLightCanIlluminate);
    CPPUNIT_TEST(testRectLightIntersection);

    CPPUNIT_TEST(testCylinderLightRadiance);
    //CPPUNIT_TEST(testCylinderLightPDF);
    CPPUNIT_TEST(testCylinderLightCanIlluminate);
    CPPUNIT_TEST(testCylinderLightIntersection);

    CPPUNIT_TEST(testDiskLightRadiance);
    CPPUNIT_TEST(testDiskLightPDF);
    CPPUNIT_TEST(testDiskLightCanIlluminate);
    CPPUNIT_TEST(testDiskLightIntersection);

    CPPUNIT_TEST(testSphereLightRadiance);
    CPPUNIT_TEST(testSphereLightPDF);
    CPPUNIT_TEST(testSphereLightCanIlluminate);
    CPPUNIT_TEST(testSphereLightIntersection);

    CPPUNIT_TEST(testSpotLightRadiance);
    CPPUNIT_TEST(testSpotLightPDF);
    CPPUNIT_TEST(testSpotLightCanIlluminate);
    CPPUNIT_TEST(testSpotLightIntersection);

    CPPUNIT_TEST(testDistantLightPDF);
    CPPUNIT_TEST(testDistantLightCanIlluminate);
    CPPUNIT_TEST(testDistantLightIntersection);

    CPPUNIT_TEST(testEnvLightRadiance);
    CPPUNIT_TEST(testEnvLightPDF);
    CPPUNIT_TEST(testEnvLightCanIlluminate);
    CPPUNIT_TEST(testEnvLightIntersection);

    CPPUNIT_TEST(testMeshLightPDF);
#endif

    CPPUNIT_TEST_SUITE_END();

public:
    TestLights();

    void setUp();
    void tearDown();

    void testRectLightRadiance()        { runTestOnMultiplePoints("Radiance", testLightRadiance, mRectLightTesters, false); }
    void testRectLightPDF()             { runTestOnMultiplePoints("Pdf", testLightPDF, mRectLightTesters, false); }
    void testRectLightCanIlluminate()   { runTestOnMultiplePoints("CanIlluminate", testLightCanIlluminate, mRectLightTesters, true); }
    void testRectLightIntersection()    { runTestOnMultiplePoints("Intersection", testLightIntersection, mRectLightTesters, true); }

    void testCylinderLightRadiance()        { runTestOnMultiplePoints("Radiance", testLightRadiance, mCylinderLightTesters, false); }
    void testCylinderLightPDF()             { runTestOnMultiplePoints("Pdf", testLightPDF, mCylinderLightTesters, false); }
    void testCylinderLightCanIlluminate()   { runTestOnMultiplePoints("CanIlluminate", testLightCanIlluminate, mCylinderLightTesters, true); }
    void testCylinderLightIntersection()    { runTestOnMultiplePoints("Intersection", testLightIntersection, mCylinderLightTesters, true); }

    void testDiskLightRadiance()        { runTestOnMultiplePoints("Radiance", testLightRadiance, mDiskLightTesters, false); }
    void testDiskLightPDF()             { runTestOnMultiplePoints("Pdf", testLightPDF, mDiskLightTesters, false); }
    void testDiskLightCanIlluminate()   { runTestOnMultiplePoints("CanIlluminate", testLightCanIlluminate, mDiskLightTesters, true); }
    void testDiskLightIntersection()    { runTestOnMultiplePoints("Intersection", testLightIntersection, mDiskLightTesters, true); }

    void testSphereLightRadiance()      { runTestOnMultiplePoints("Radiance", testLightRadiance, mSphereLightTesters, false); }
    void testSphereLightPDF()           { runTestOnMultiplePoints("Pdf", testLightPDF, mSphereLightTesters, false); }
    void testSphereLightCanIlluminate() { runTestOnMultiplePoints("CanIlluminate", testLightCanIlluminate, mSphereLightTesters, true); }
    void testSphereLightIntersection()  { runTestOnMultiplePoints("Intersection", testLightIntersection, mSphereLightTesters, true); }

    void testSpotLightRadiance()        { runTestOnMultiplePoints("Radiance", testLightRadiance, mSpotLightTesters, false); }
    void testSpotLightPDF()             { runTestOnMultiplePoints("Pdf", testLightPDF, mSpotLightTesters, false); }
    void testSpotLightCanIlluminate()   { runTestOnMultiplePoints("CanIlluminate", testLightCanIlluminate, mSpotLightTesters, true); }
    void testSpotLightIntersection()    { runTestOnMultiplePoints("Intersection", testLightIntersection, mSpotLightTesters, true); }

    void testDistantLightPDF()              { runTestOnMultiplePoints("Pdf", testLightPDF, mDistantLightTesters, true); }
    void testDistantLightCanIlluminate()    { runTestOnMultiplePoints("CanIlluminate", testLightCanIlluminate, mDistantLightTesters, true); }
    void testDistantLightIntersection()     { runTestOnMultiplePoints("Intersection", testLightIntersection, mDistantLightTesters, true); }

    void testEnvLightRadiance()         { runTestOnMultiplePoints("Radiance", testLightRadiance, mEnvLightTesters, true); }
    void testEnvLightPDF()              { runTestOnMultiplePoints("Pdf", testLightPDF, mEnvLightTesters, true); }
    void testEnvLightCanIlluminate()    { runTestOnMultiplePoints("CanIlluminate", testLightCanIlluminate, mEnvLightTesters, true); }
    void testEnvLightIntersection()     { runTestOnMultiplePoints("Intersection", testLightIntersection, mEnvLightTesters, true); }

    void testMeshLightPDF()             { runTestOnMultiplePoints("Pdf", testMeshLightFaceSelectionPDF, mMeshLightTesters, true); }

private:
    /// This function is responsible for generating the surface points and directions to run the tests on.
    void runTestOnMultiplePoints(const std::string &funcName, TestFunction func, const LightTesters &light_testers, bool closePointsOnly);

    uint32_t getRandomInt32();

    scene_rdl2::rdl2::SceneContext mContext;
    scene_rdl2::util::Random mRand;

    LightTesters mRectLightTesters;
    LightTesters mCylinderLightTesters;
    LightTesters mDiskLightTesters;
    LightTesters mSphereLightTesters;
    LightTesters mSpotLightTesters;
    LightTesters mDistantLightTesters;
    LightTesters mEnvLightTesters;
    LightTesters mMeshLightTesters;
};

//----------------------------------------------------------------------------

} // namespace pbr
} // namespace moonray

