// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

///
/// @file TestDistribution.h
/// $Id$
///

#pragma once

#include "TestUtil.h"
#include <cppunit/extensions/HelperMacros.h>
#include <cppunit/TestFixture.h>
#include <string>
#include <vector>

namespace moonray {
namespace pbr {


//----------------------------------------------------------------------------

///
/// @class TestDistribution TestDistribution.h <pbr/TestDistribution.h>
/// @brief This class tests Distribution2D sampling object
/// 
class TestDistribution : public CppUnit::TestFixture
{
public:
    CPPUNIT_TEST_SUITE(TestDistribution);
#if 1
    CPPUNIT_TEST(testUniform);
    CPPUNIT_TEST(testGradient);
    CPPUNIT_TEST(testImages);
    CPPUNIT_TEST(testDiscrete);
#endif
    CPPUNIT_TEST_SUITE_END();

    void testUniform();
    void testGradient();
    void testImages();
    void testDiscrete();

private:
    void testImage(const std::string &path, const std::string &filename);
};


//----------------------------------------------------------------------------

} // namespace pbr
} // namespace moonray

