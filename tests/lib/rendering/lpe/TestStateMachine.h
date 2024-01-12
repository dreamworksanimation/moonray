// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

/// @file TestStateMachine.h

#pragma once

#include <cppunit/TestFixture.h>
#include <cppunit/extensions/HelperMacros.h>

namespace moonray {
namespace lpe {
namespace unittest {

class TestStateMachine : public CppUnit::TestFixture
{
    void testLpe();

    CPPUNIT_TEST_SUITE(TestStateMachine);
    CPPUNIT_TEST(testLpe);
    CPPUNIT_TEST_SUITE_END();
};

} // namespace unittest
} // namespace lpe
} // namespace moonray

