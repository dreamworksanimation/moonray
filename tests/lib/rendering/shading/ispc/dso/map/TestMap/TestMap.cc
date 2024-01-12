// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

/// @file TestMap.cc

#include <scene_rdl2/scene/rdl2/rdl2.h>
#include "attributes.cc"

#include "TestMap_ispc_stubs.h"

RDL2_DSO_CLASS_BEGIN(TestMap, Map)
public:
    TestMap(SceneClass const &sceneClass, std::string const &name);
    void update();

private:
RDL2_DSO_CLASS_END(TestMap)

TestMap::TestMap(SceneClass const &sceneClass,
                 std::string const &name):
Parent(sceneClass, name)
{
    mSampleFunc = nullptr;
    mSampleFuncv = (SampleFuncv) ispc::TestMap_getSampleFunc();
}

void
TestMap::update()
{
}

