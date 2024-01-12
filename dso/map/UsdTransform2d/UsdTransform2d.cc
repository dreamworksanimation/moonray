// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

/// @file UsdTransform2d.cc

/// Translates, rotates, and scales UV

#include "attributes.cc"
#include "UsdTransform2d_ispc_stubs.h"

#include <moonray/common/mcrt_macros/moonray_static_check.h>
#include <moonray/rendering/shading/MapApi.h>
#include <scene_rdl2/render/util/stdmemory.h>

using namespace moonray::shading;
using namespace scene_rdl2::math;

RDL2_DSO_CLASS_BEGIN(UsdTransform2d, scene_rdl2::rdl2::Map)
public:
    UsdTransform2d(scene_rdl2::rdl2::SceneClass const &sceneClass, std::string const &name);
    ~UsdTransform2d() override;
    void update() override;

private:
    static void sample(const scene_rdl2::rdl2::Map *self, moonray::shading::TLState *tls,
                       const moonray::shading::State &state, Color *sample);

RDL2_DSO_CLASS_END(UsdTransform2d)

UsdTransform2d::UsdTransform2d(const scene_rdl2::rdl2::SceneClass& sceneClass,
        const std::string& name) :
    Parent(sceneClass, name)
{
    mSampleFunc = UsdTransform2d::sample;
    mSampleFuncv = (scene_rdl2::rdl2::SampleFuncv) ispc::UsdTransform2d_getSampleFunc();
}

UsdTransform2d::~UsdTransform2d()
{
}

void
UsdTransform2d::update()
{
}

void
UsdTransform2d::sample(const scene_rdl2::rdl2::Map* self,
                       moonray::shading::TLState *tls,
                       const moonray::shading::State& state,
                       Color* sample)
{
    const UsdTransform2d* me = static_cast<const UsdTransform2d*>(self);

    Color input = evalColor(me, attrIn, tls, state);
    Vec2d uv(input.r, input.g);

    const Vec2f scale = me->get(attrScale);
    uv.x = scale.x * uv.x;
    uv.y = scale.y * uv.y;

    const float rotation = me->get(attrRotation);
    if (!isZero(rotation)) {
        float s, c;
        const float theta = deg2rad(rotation);
        sincos(theta, &s, &c);
        const float tmpUvX = uv.x * c - uv.y * s;
        uv.y = uv.x * s + uv.y * c;
        uv.x = tmpUvX;
    }

    const Vec2f translation = me->get(attrTranslation);
    uv.x = uv.x + translation.x;
    uv.y = uv.y + translation.y;

    *sample = Color(uv.x, uv.y, 0.f);
}

