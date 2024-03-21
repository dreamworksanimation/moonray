// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

///
/// @file TestSampler.h
/// $Id$
///

#pragma once

#include <cppunit/extensions/HelperMacros.h>
#include <cppunit/TestFixture.h>

namespace moonray {
namespace pbr {


//----------------------------------------------------------------------------

///
/// @class TestSampler TestSampler.h <pbr/TestSampler.h>
///
class TestSampler : public CppUnit::TestFixture
{
public:
    CPPUNIT_TEST_SUITE(TestSampler);
#if 1
    CPPUNIT_TEST(testPrimaryDeterminism);
    CPPUNIT_TEST(testIntegratorDeterminism);
    CPPUNIT_TEST(testPrimaryDistribution);
    CPPUNIT_TEST(testIntegratorDistribution);
    CPPUNIT_TEST(testSequenceID);
    CPPUNIT_TEST(testISPCSequenceID);
    CPPUNIT_TEST(testISCPPermutations);
    CPPUNIT_TEST(testSamplePartition);
#endif
    CPPUNIT_TEST_SUITE_END();

    void testPrimaryDeterminism();
    void testIntegratorDeterminism();
    void testPrimaryDistribution();
    void testIntegratorDistribution();
    void testSequenceID();
    void testISPCSequenceID();
    void testISCPPermutations();
    void testSamplePartition();

public:
    void setUp();
    void tearDown();
};


//----------------------------------------------------------------------------

} // namespace pbr
} // namespace moonray


