// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

/// @file TestBsdfOneSamplerv.h

#pragma once

#include <cppunit/extensions/HelperMacros.h>
#include <cppunit/TestFixture.h>

namespace moonray {
namespace pbr {

class TestBsdfOneSamplerv : public CppUnit::TestFixture
{
public:
    TestBsdfOneSamplerv();
    ~TestBsdfOneSamplerv();

    void setUp();
    void tearDown();

    CPPUNIT_TEST_SUITE(TestBsdfOneSamplerv);
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
};

} // namespace pbr
} // namespace moonray

