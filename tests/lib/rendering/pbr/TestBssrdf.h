// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

///
/// @file TestBssrdf.h
/// $Id$
///

#pragma once

#include <moonray/rendering/shading/bssrdf/Dipole.h>

#include <cppunit/extensions/HelperMacros.h>
#include <cppunit/TestFixture.h>
#include <scene_rdl2/common/math/Color.h>
#include <scene_rdl2/common/math/Vec3.h>

namespace moonray {
namespace pbr {

//----------------------------------------------------------------------------

///
/// @class TestBssrdf TestBssrdf.h <pbr/TestBssrdf.h>
/// @brief This class tests response and statistical properties of all the
/// brdf models we have in lib/pbr
/// 
class TestBssrdf : public CppUnit::TestFixture
{
public:
    CPPUNIT_TEST_SUITE(TestBssrdf);
#if 1
    CPPUNIT_TEST(testDipole);
#endif
    CPPUNIT_TEST_SUITE_END();

    void testDipole();

private:
    // physical init parameters
    struct PParams
    {
        scene_rdl2::math::Vec3f n;
        float eta;
        scene_rdl2::math::Color sigmaA;
        scene_rdl2::math::Color sigmaSP;
        float sceneScale;
    };

    // artist friendly init parameters
    struct AParams
    {
        scene_rdl2::math::Vec3f n;
        float eta;
        float translucentFactor;
        scene_rdl2::math::Color translucentColor;
        scene_rdl2::math::Color radius;
    };

    void test(const shading::Bssrdf &bssrdf, int sampleCount, char *name);
    void test(const shading::Bssrdfv &bssrdfv, int sampleCount, char *name);
    void test(const AParams &p, int sampleCount, char *name);
    void test(const PParams &p, int sampleCount, char *name);


    void testDistribution(const shading::Bssrdf &bssrdf, int sampleCount, char *suffix);
    void testDistribution(const shading::Bssrdfv &bssrdfv, int sampleCount, char *suffix);
    void testPdfIntegral(const shading::Bssrdf &bssrdf, int sampleCount);
    void testPdfIntegral(const shading::Bssrdfv &bssrdfv, int sampleCount);
    void testIntegral(const shading::Bssrdf &bssrdf, int sampleCount);
    void testIntegral(const shading::Bssrdfv &bssrdfv, int sampleCount);
};


//----------------------------------------------------------------------------

} // namespace pbr
} // namespace moonray

