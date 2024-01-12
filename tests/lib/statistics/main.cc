// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0


#include "TestFormatters.h"

#include <cppunit/TestFixture.h>
#include <cppunit/extensions/HelperMacros.h>
#include <scene_rdl2/pdevunit/pdevunit.h>

int
main(int argc, char* argv[])
{
    using namespace moonray_stats::unittest;

    CPPUNIT_TEST_SUITE_REGISTRATION(TestFormatters);

    return pdevunit::run(argc, argv);
}

