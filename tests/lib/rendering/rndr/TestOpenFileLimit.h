// Copyright 2026 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0


#pragma once

#include <cppunit/TestFixture.h>
#include <cppunit/extensions/HelperMacros.h>

#include <sys/resource.h>

namespace moonray {
namespace rndr {
namespace unittest {

class TestOpenFileLimit : public CppUnit::TestFixture
{
public:
    void setUp() override;
    void tearDown() override;

    void testGetOpenFileLimit();
    void testOpenFileLimitToString();
    void testIsOpenFileLimitLow();
    void testExceedsOpenFileLimit();

    CPPUNIT_TEST_SUITE(TestOpenFileLimit);
    CPPUNIT_TEST(testGetOpenFileLimit);
    CPPUNIT_TEST(testOpenFileLimitToString);
    CPPUNIT_TEST(testIsOpenFileLimitLow);
    CPPUNIT_TEST(testExceedsOpenFileLimit);
    CPPUNIT_TEST_SUITE_END();

private:
    struct rlimit mSavedLimit;
};

} // namespace unittest
} // namespace rndr
} // namespace moonray

