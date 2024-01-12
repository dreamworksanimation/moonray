// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

/// @file TestCheckerMap.cc

#include <scene_rdl2/scene/rdl2/rdl2.h>
#include "attributes.cc"

#include "TestCheckerMap_ispc_stubs.h"

RDL2_DSO_CLASS_BEGIN(TestCheckerMap, Map)
public:
    TestCheckerMap(SceneClass const &sceneClass, std::string const &name);
    void update();

private:
RDL2_DSO_CLASS_END(TestCheckerMap)

TestCheckerMap::TestCheckerMap(SceneClass const &sceneClass,
                               std::string const &name):
Parent(sceneClass, name)
{
    mSampleFunc = nullptr;
    mSampleFuncv = (SampleFuncv) ispc::TestCheckerMap_getSampleFunc();
}

void
TestCheckerMap::update()
{
}

