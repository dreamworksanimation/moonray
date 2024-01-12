// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0


#include "attributes.cc"
#include "ExtraAovMap_ispc_stubs.h"

#include <moonray/rendering/shading/MapApi.h>

using namespace scene_rdl2::math;

//----------------------------------------------------------------------------

RDL2_DSO_CLASS_BEGIN(ExtraAovMap, scene_rdl2::rdl2::Map)

public:
    ExtraAovMap(const scene_rdl2::rdl2::SceneClass &sceneClass, const std::string &name);
    ~ExtraAovMap() override;
    void update() override;
    bool getIsExtraAovMap(scene_rdl2::rdl2::String &label, scene_rdl2::rdl2::Bool &postScatter) const override;

private:
    static void sample(const scene_rdl2::rdl2::Map *self, moonray::shading::TLState *tls,
                       const moonray::shading::State &state, Color *sample);

RDL2_DSO_CLASS_END(ExtraAovMap)

//----------------------------------------------------------------------------

ExtraAovMap::ExtraAovMap(const scene_rdl2::rdl2::SceneClass &sceneClass, const std::string &name):
    Parent(sceneClass, name)
{
    mSampleFunc = ExtraAovMap::sample;
    mSampleFuncv = (scene_rdl2::rdl2::SampleFuncv) ispc::ExtraAovMap_getSampleFunc();
}

ExtraAovMap::~ExtraAovMap()
{
}

void
ExtraAovMap::update()
{
}

bool
ExtraAovMap::getIsExtraAovMap(scene_rdl2::rdl2::String &label, scene_rdl2::rdl2::Bool &postScatter) const
{
    // we are only valid as an extra aov if we have a non-empty label
    label = get(attrLabel);
    postScatter = get(attrPostScatter);
    return !label.empty();
}

void
ExtraAovMap::sample(const scene_rdl2::rdl2::Map *self, moonray::shading::TLState *tls,
                        const moonray::shading::State &state, Color *sample)
{
    *sample = evalColor(self, attrColor, tls, state);
}

