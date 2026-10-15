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

    void testFitTextureFileHandles();
    void testRaiseOpenFileLimitToMax();

    CPPUNIT_TEST_SUITE(TestOpenFileLimit);
    CPPUNIT_TEST(testFitTextureFileHandles);
    CPPUNIT_TEST(testRaiseOpenFileLimitToMax);
    CPPUNIT_TEST_SUITE_END();

private:
    struct rlimit mSavedLimit;
};

} // namespace unittest
} // namespace rndr
} // namespace moonray

