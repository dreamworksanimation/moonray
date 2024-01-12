// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cppunit/extensions/HelperMacros.h>

class TestWait: public CppUnit::TestCase
{
public:
    CPPUNIT_TEST_SUITE(TestWait);
    CPPUNIT_TEST(testNotifyOne);
    CPPUNIT_TEST(testNotifyAll);
    CPPUNIT_TEST_SUITE_END();

    void testNotifyOne();
    void testNotifyAll();
};


