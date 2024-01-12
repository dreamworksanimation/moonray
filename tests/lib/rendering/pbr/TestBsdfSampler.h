// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

///
/// @file TestBsdfSampler.h
/// $Id$
///

#pragma once

#include <cppunit/extensions/HelperMacros.h>
#include <cppunit/TestFixture.h>


namespace moonray {
namespace pbr {

//----------------------------------------------------------------------------

///
/// @class TestBsdfSampler TestBsdfSampler.h <pbr/unittest/TestBsdfSampler.h>
/// @brief This class tests response and statistical properties of all the
/// brdf models we have in lib/pbr
/// 
class TestBsdfSampler : public CppUnit::TestFixture
{
public:
    TestBsdfSampler();
    ~TestBsdfSampler();

    void setUp();
    void tearDown();

    CPPUNIT_TEST_SUITE(TestBsdfSampler);
#if 1
    CPPUNIT_TEST(testLambert);

    CPPUNIT_TEST(testCookTorrance);
    CPPUNIT_TEST(testGGXCookTorrance);
    CPPUNIT_TEST(testAnisoCookTorrance);
    CPPUNIT_TEST(testTransmissionCookTorrance);
    CPPUNIT_TEST(testIridescence);

    CPPUNIT_TEST(testRetroreflection);

    CPPUNIT_TEST(testEyeCaustic);

    CPPUNIT_TEST(testDwaFabric);
    CPPUNIT_TEST(testKajiyaKayFabric);

    CPPUNIT_TEST(testAshikminhShirley);
    //CPPUNIT_TEST(testAshikminhShirleyFull);
    CPPUNIT_TEST(testWardCorrected);
    CPPUNIT_TEST(testWardDuer);

    CPPUNIT_TEST(testHairDiffuse);
    CPPUNIT_TEST(testHairR);
    CPPUNIT_TEST(testHairTT);

    CPPUNIT_TEST(testTwoLobes);
    CPPUNIT_TEST(testThreeLobes);

    CPPUNIT_TEST(testStochasticFlakes);

    CPPUNIT_TEST(testUnderClearcoatLambert);
    CPPUNIT_TEST(testUnderClearcoatCookTorrance);
#endif

    CPPUNIT_TEST_SUITE_END();

    void testLambert();

    void testCookTorrance();
    void testGGXCookTorrance();
    void testAnisoCookTorrance();
    void testTransmissionCookTorrance();

    void testIridescence();

    void testRetroreflection();

    void testEyeCaustic();

    void testDwaFabric();
    void testKajiyaKayFabric();

    void testAshikminhShirley();
    void testAshikminhShirleyFull();

    void testWardCorrected();
    void testWardDuer();

    void testHairDiffuse();
    void testHairR();
    void testHairTT();

    void testTwoLobes();
    void testThreeLobes();

    void testStochasticFlakes();

    void testUnderClearcoatLambert();
    void testUnderClearcoatCookTorrance();
};


//----------------------------------------------------------------------------

} // namespace pbr
} // namespace moonray


