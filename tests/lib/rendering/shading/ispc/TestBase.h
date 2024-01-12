// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

/// @file TestBase.h
#pragma once

#include <cppunit/TestFixture.h>
#include <cppunit/extensions/HelperMacros.h>

namespace moonray {
namespace shading {
namespace unittest {

class TestBase : public CppUnit::TestFixture
{
public:
    void shade();

    CPPUNIT_TEST_SUITE(TestBase);
    CPPUNIT_TEST(shade);
    CPPUNIT_TEST_SUITE_END();

private:
    void shade(int);
};

} // namespace unittest
} // namespace shading
} // namespace moonray

