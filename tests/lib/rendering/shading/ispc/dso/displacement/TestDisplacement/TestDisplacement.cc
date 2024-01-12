// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

/// @file TestDisplacement.cc

#include <scene_rdl2/scene/rdl2/rdl2.h>
#include "attributes.cc"

#include "TestDisplacement_ispc_stubs.h"

RDL2_DSO_CLASS_BEGIN(TestDisplacement, Displacement)
public:
    TestDisplacement(SceneClass const &sceneClass, std::string const &name);
    void update();

private:
RDL2_DSO_CLASS_END(TestDisplacement)

TestDisplacement::TestDisplacement(SceneClass const &sceneClass,
                                   std::string const &name):
Parent(sceneClass, name)
{
    mDisplaceFunc = nullptr;
    mDisplaceFuncv = (DisplaceFuncv) ispc::TestDisplacement_getDisplaceFunc();
}

void
TestDisplacement::update()
{
}

