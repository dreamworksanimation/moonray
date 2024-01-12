// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cppunit/extensions/HelperMacros.h>

class TestAlignedElementArray: public CppUnit::TestCase
{
public:
    CPPUNIT_TEST_SUITE(TestAlignedElementArray);
    CPPUNIT_TEST(testGeneric);
    CPPUNIT_TEST(testConstructors);
    CPPUNIT_TEST(testIterators);
    CPPUNIT_TEST(testAlignment);
    CPPUNIT_TEST_SUITE_END();

    void testGeneric();
    void testConstructors();
    void testIterators();
    void testAlignment();
};


