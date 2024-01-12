// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

/// @file TestDisplace.h
#pragma once

#include <cppunit/TestFixture.h>
#include <cppunit/extensions/HelperMacros.h>

namespace moonray {
namespace shading {
namespace unittest {

class TestDisplace : public CppUnit::TestFixture
{
public:
    void displace();

    CPPUNIT_TEST_SUITE(TestDisplace);
    CPPUNIT_TEST(displace);
    CPPUNIT_TEST_SUITE_END();

private:
    void displace(int);
};

} // namespace unittest
} // namespace shading
} // namespace moonray

