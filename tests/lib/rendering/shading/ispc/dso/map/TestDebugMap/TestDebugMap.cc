// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

/// @file TestDebugMap.cc

#include <scene_rdl2/scene/rdl2/rdl2.h>
#include "attributes.cc"

#include "TestDebugMap_ispc_stubs.h"

RDL2_DSO_CLASS_BEGIN(TestDebugMap, Map)
public:
    TestDebugMap(SceneClass const &sceneClass, std::string const &name);
    void update();

private:
RDL2_DSO_CLASS_END(TestDebugMap)

TestDebugMap::TestDebugMap(SceneClass const &sceneClass,
                 std::string const &name):
Parent(sceneClass, name)
{
    mSampleFunc = nullptr;
    mSampleFuncv = (SampleFuncv) ispc::TestDebugMap_getSampleFunc();
}

void
TestDebugMap::update()
{
}

