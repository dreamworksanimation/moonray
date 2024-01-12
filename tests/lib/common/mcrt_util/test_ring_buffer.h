// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cppunit/extensions/HelperMacros.h>

class TestRingBuffer: public CppUnit::TestCase
{
public:
    CPPUNIT_TEST_SUITE(TestRingBuffer);
    CPPUNIT_TEST(testSingleItemProducer);
    CPPUNIT_TEST(testBatchItemProducer);
    CPPUNIT_TEST_SUITE_END();

    void testSingleItemProducer();
    void testBatchItemProducer();
};


