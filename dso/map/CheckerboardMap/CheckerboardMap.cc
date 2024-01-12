// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0


#include "attributes.cc"
#include "CheckerboardMap_ispc_stubs.h"

#include <moonray/rendering/shading/MapApi.h>

using namespace scene_rdl2::math;

//----------------------------------------------------------------------------

RDL2_DSO_CLASS_BEGIN(CheckerboardMap, scene_rdl2::rdl2::Map)

public:
    CheckerboardMap(const scene_rdl2::rdl2::SceneClass &sceneClass, const std::string &name);
    ~CheckerboardMap() override;
    void update() override;

private:
    static void sample(const scene_rdl2::rdl2::Map *self, moonray::shading::TLState *tls,
                       const moonray::shading::State &state, Color *sample);

RDL2_DSO_CLASS_END(CheckerboardMap)

//----------------------------------------------------------------------------

CheckerboardMap::CheckerboardMap(const scene_rdl2::rdl2::SceneClass &sceneClass, const std::string &name):
    Parent(sceneClass, name)
{
    mSampleFunc = CheckerboardMap::sample;
    mSampleFuncv = (scene_rdl2::rdl2::SampleFuncv) ispc::CheckerboardMap_getSampleFunc();
}

CheckerboardMap::~CheckerboardMap()
{
}

void
CheckerboardMap::update()
{
}

void
CheckerboardMap::sample(const scene_rdl2::rdl2::Map *self, moonray::shading::TLState *tls,
                        const moonray::shading::State &state, Color *sample)
{
    const CheckerboardMap *me = static_cast<const CheckerboardMap *>(self);

    Vec2f st;
    switch (me->get(attrTextureEnum)) {
    case 1:
        st = asVec2(evalVec3f(me, attrInputTextureCoordinate, tls, state));
        break;
    case 0:
    default:
        st = state.getSt();
        break;
    }

    //Put UVs in 0-1 range
    float smod = std::fmod(st[0] * me->get(attrUTiles) / 2.0f, 1.0f);
    float tmod = std::fmod(st[1] * me->get(attrVTiles) / 2.0f, 1.0f);

    //If negative, wrap it into positive range
    if (smod < 0.0f) smod += 1.0f;
    if (tmod < 0.0f) tmod += 1.0f;

    if (smod < 0.5f) {
        if (tmod < 0.5f) {
            *sample = me->get(attrColorA);
        } else {
            *sample = me->get(attrColorB);
        }
    } else {
        if (tmod < 0.5f) {
            *sample = me->get(attrColorB);
        } else {
            *sample = me->get(attrColorA);
        }
    }
}

