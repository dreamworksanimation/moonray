// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0


/// @file main.cc

#include "TestGeomApi.h"
#include <scene_rdl2/pdevunit/pdevunit.h>

int
main(int argc, char *argv[])
{
    CPPUNIT_TEST_SUITE_REGISTRATION(moonray::geom::unittest::TestGeomApi);
    return pdevunit::run(argc, argv);    
}




