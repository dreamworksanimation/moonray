// Copyright 2023-2025 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cppunit/extensions/HelperMacros.h>

class TestAtomicFunctions: public CppUnit::TestCase
{
public:
    CPPUNIT_TEST_SUITE(TestAtomicFunctions);
    CPPUNIT_TEST(testAdd);
    CPPUNIT_TEST(testMin);
    CPPUNIT_TEST(testMax);
    CPPUNIT_TEST(testClosest);
    CPPUNIT_TEST_SUITE_END();

    void testAdd();
    void testMin();
    void testMax();
    void testClosest();

private:
    // static constexpr int sTestLoopMax = 1024; // 120.51sec @cobaltcard
    // static constexpr int sTestLoopMax = 512; // 60.09sec @cobaltcard
    // static constexpr int sTestLoopMax = 256; // 30.04sec @cobaltcard
    // static constexpr int sTestLoopMax = 128; // 15.41sec @cobaltcard
    // static constexpr int sTestLoopMax = 64; // 7.44sec @cobaltcard
    static constexpr int sTestLoopMax = 32; // 4.01sec @cobaltcard
    // static constexpr int sTestLoopMax = 16; // 1.93sec @cobaltcard
    // static constexpr int sTestLoopMax = 8; // 0.96sec @cobaltcard
    // static constexpr int sTestLoopMax = 4; // 0.51sec @cobaltcard
    // static constexpr int sTestLoopMax = 2; // 0.25sec @cobaltcard
    // static constexpr int sTestLoopMax = 1; // 0.14sec @cobaltcard
};


