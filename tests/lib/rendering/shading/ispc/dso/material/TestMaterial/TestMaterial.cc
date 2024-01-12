// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

/// @file TestMaterial.cc

#include <scene_rdl2/scene/rdl2/rdl2.h>
#include "attributes.cc"

#include "TestMaterial_ispc_stubs.h"

RDL2_DSO_CLASS_BEGIN(TestMaterial, Material)
public:
    TestMaterial(SceneClass const &sceneClass, std::string const &name);
    void update();

private:
RDL2_DSO_CLASS_END(TestMaterial)

TestMaterial::TestMaterial(SceneClass const &sceneClass,
                           std::string const &name):
Parent(sceneClass, name)
{
    mShadeFunc = nullptr;
    mShadeFuncv = (ShadeFuncv) ispc::TestMaterial_getShadeFunc();
}

void
TestMaterial::update()
{
}

