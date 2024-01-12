// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0


#include "TestActivePixelMask.h"
#include "TestCheckpoint.h"
#include "TestOverlappingRegions.h"
#include "TestSocketStream.h"

#include <cppunit/TestFixture.h>
#include <cppunit/extensions/HelperMacros.h>
#include <scene_rdl2/pdevunit/pdevunit.h>

int
main(int argc, char* argv[])
{
    using namespace moonray::rndr::unittest;

    CPPUNIT_TEST_SUITE_REGISTRATION(TestSocketStream);
    CPPUNIT_TEST_SUITE_REGISTRATION(TestOverlappingRegions);
    CPPUNIT_TEST_SUITE_REGISTRATION(TestCheckpoint);
    CPPUNIT_TEST_SUITE_REGISTRATION(TestActivePixelMask);

    return pdevunit::run(argc, argv);
}

