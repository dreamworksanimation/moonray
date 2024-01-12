// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cppunit/extensions/HelperMacros.h>

class TestRenderingRT: public CppUnit::TestCase
{
public:
    CPPUNIT_TEST_SUITE(TestRenderingRT);
    CPPUNIT_TEST(testRay);
    CPPUNIT_TEST(testIntersectPolygon);
    CPPUNIT_TEST(testIntersectInstances);
    CPPUNIT_TEST(testIntersectNestedInstances);
    CPPUNIT_TEST_SUITE_END();

    void testRay();
    void testIntersectPolygon();
    void testIntersectInstances();
    void testIntersectNestedInstances();
};


