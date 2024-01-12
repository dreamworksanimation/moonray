// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cppunit/extensions/HelperMacros.h>

class TestCommonTime: public CppUnit::TestCase
{
public:
    CPPUNIT_TEST_SUITE(TestCommonTime);
    CPPUNIT_TEST(testTimer);
    CPPUNIT_TEST_SUITE_END();

    void testTimer();
};

