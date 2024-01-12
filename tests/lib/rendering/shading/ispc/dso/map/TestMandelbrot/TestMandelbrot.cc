// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

/// @file TestMandelbrot.cc

#include <scene_rdl2/scene/rdl2/rdl2.h>
#include "attributes.cc"

#include "TestMandelbrot_ispc_stubs.h"

RDL2_DSO_CLASS_BEGIN(TestMandelbrot, Map)
public:
    TestMandelbrot(SceneClass const &sceneClass, std::string const &name);
    void update();

private:
RDL2_DSO_CLASS_END(TestMandelbrot)

TestMandelbrot::TestMandelbrot(SceneClass const &sceneClass,
                               std::string const &name):
    Parent(sceneClass, name)
{
    mSampleFunc = nullptr;
    mSampleFuncv = (SampleFuncv) ispc::TestMandelbrot_getSampleFunc();
}

void
TestMandelbrot::update()
{
}

