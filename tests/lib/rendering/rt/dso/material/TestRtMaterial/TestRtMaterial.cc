// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

///
/// @file TestRtMaterial.cc
///

#include <scene_rdl2/scene/rdl2/rdl2.h>

#include "attributes.cc"

void
testRtShade(const scene_rdl2::rdl2::Material*, moonray::shading::TLState *tls, const moonray::shading::State&,
        moonray::shading::BsdfBuilder &bsdfBuilder)
{
}

RDL2_DSO_CLASS_BEGIN(TestRtMaterial, scene_rdl2::rdl2::Material)

public:
    TestRtMaterial(const scene_rdl2::rdl2::SceneClass& sceneClass, const std::string& name);

RDL2_DSO_CLASS_END(TestRtMaterial)

TestRtMaterial::TestRtMaterial(const scene_rdl2::rdl2::SceneClass& sceneClass,
        const std::string& name) :
    Parent(sceneClass, name)
{
    mShadeFunc = testRtShade;
}

