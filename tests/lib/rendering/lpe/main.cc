// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

/// @file main.cc

#include "TestStateMachine.h"

#include <cppunit/TestFixture.h>
#include <cppunit/extensions/HelperMacros.h>
#include <scene_rdl2/pdevunit/pdevunit.h>

int
main(int argc, char* argv[])
{
    using namespace moonray::lpe::unittest;

    CPPUNIT_TEST_SUITE_REGISTRATION(TestStateMachine);

    return pdevunit::run(argc, argv);
}


