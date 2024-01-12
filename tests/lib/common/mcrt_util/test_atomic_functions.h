// Copyright 2023-2024 DreamWorks Animation LLC
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
};


