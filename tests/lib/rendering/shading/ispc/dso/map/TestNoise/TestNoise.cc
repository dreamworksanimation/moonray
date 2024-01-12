// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

/// @file TestNoise.cc

#include <scene_rdl2/scene/rdl2/rdl2.h>
#include "attributes.cc"

#include "TestNoise_ispc_stubs.h"

RDL2_DSO_CLASS_BEGIN(TestNoise, Map)
public:
    TestNoise(SceneClass const &sceneClass, std::string const &name);
    void update();

private:
    ispc::TestNoise mIspcTestNoise;
RDL2_DSO_CLASS_END(TestNoise)

TestNoise::TestNoise(SceneClass const &sceneClass,
                               std::string const &name):
    Parent(sceneClass, name)
{
    mSampleFunc = nullptr;
    mSampleFuncv = (SampleFuncv) ispc::TestNoise_getSampleFunc();
}

void
TestNoise::update()
{
    ispc::TestNoise_update(this);
}

