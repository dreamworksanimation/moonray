// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

///
/// @file TestMaterial.cc
///

#include <scene_rdl2/scene/rdl2/rdl2.h>

#include "attributes.cc"

void
testShade(const scene_rdl2::rdl2::Material*, moonray::shading::TLState *tls, const moonray::shading::State&,
        moonray::shading::BsdfBuilder &bsdfBuilder)
{
}

RDL2_DSO_CLASS_BEGIN(TestMaterial, scene_rdl2::rdl2::Material)

public:
    TestMaterial(const scene_rdl2::rdl2::SceneClass& sceneClass, const std::string& name);

RDL2_DSO_CLASS_END(TestMaterial)

TestMaterial::TestMaterial(const scene_rdl2::rdl2::SceneClass& sceneClass,
        const std::string& name) :
    Parent(sceneClass, name)
{
    mShadeFunc = testShade;
}

