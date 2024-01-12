// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0


#include "attributes.cc"
#include "ListMap_ispc_stubs.h"

#include <moonray/rendering/shading/MapApi.h>

using namespace scene_rdl2::math;

//----------------------------------------------------------------------------

RDL2_DSO_CLASS_BEGIN(ListMap, scene_rdl2::rdl2::Map)

public:
    ListMap(const scene_rdl2::rdl2::SceneClass &sceneClass, const std::string &name);
    ~ListMap() override;
    void update() override;
    bool getIsListMap(std::vector<const scene_rdl2::rdl2::Map *> &maps) const override;


private:
    static void sample(const scene_rdl2::rdl2::Map *self, moonray::shading::TLState *tls,
                       const moonray::shading::State &state, Color *sample);
    std::vector<const scene_rdl2::rdl2::Map *> mMaps;

RDL2_DSO_CLASS_END(ListMap)

//----------------------------------------------------------------------------

ListMap::ListMap(const scene_rdl2::rdl2::SceneClass &sceneClass, const std::string &name):
    Parent(sceneClass, name)
{
    mSampleFunc = ListMap::sample;
    mSampleFuncv = (scene_rdl2::rdl2::SampleFuncv) ispc::ListMap_getSampleFunc();
}

ListMap::~ListMap()
{
}

void
ListMap::update()
{
    // 20 slots should be enough for most practical applications
    // There is no reason that we can't add slots if the need arises
    mMaps.clear();
    if (get(attrMap0)) mMaps.push_back(get(attrMap0)->asA<scene_rdl2::rdl2::Map>());
    if (get(attrMap1)) mMaps.push_back(get(attrMap1)->asA<scene_rdl2::rdl2::Map>());
    if (get(attrMap2)) mMaps.push_back(get(attrMap2)->asA<scene_rdl2::rdl2::Map>());
    if (get(attrMap3)) mMaps.push_back(get(attrMap3)->asA<scene_rdl2::rdl2::Map>());
    if (get(attrMap4)) mMaps.push_back(get(attrMap4)->asA<scene_rdl2::rdl2::Map>());
    if (get(attrMap5)) mMaps.push_back(get(attrMap5)->asA<scene_rdl2::rdl2::Map>());
    if (get(attrMap6)) mMaps.push_back(get(attrMap6)->asA<scene_rdl2::rdl2::Map>());
    if (get(attrMap7)) mMaps.push_back(get(attrMap7)->asA<scene_rdl2::rdl2::Map>());
    if (get(attrMap8)) mMaps.push_back(get(attrMap8)->asA<scene_rdl2::rdl2::Map>());
    if (get(attrMap9)) mMaps.push_back(get(attrMap9)->asA<scene_rdl2::rdl2::Map>());
    if (get(attrMap10)) mMaps.push_back(get(attrMap10)->asA<scene_rdl2::rdl2::Map>());
    if (get(attrMap11)) mMaps.push_back(get(attrMap11)->asA<scene_rdl2::rdl2::Map>());
    if (get(attrMap12)) mMaps.push_back(get(attrMap12)->asA<scene_rdl2::rdl2::Map>());
    if (get(attrMap13)) mMaps.push_back(get(attrMap13)->asA<scene_rdl2::rdl2::Map>());
    if (get(attrMap14)) mMaps.push_back(get(attrMap14)->asA<scene_rdl2::rdl2::Map>());
    if (get(attrMap15)) mMaps.push_back(get(attrMap15)->asA<scene_rdl2::rdl2::Map>());
    if (get(attrMap16)) mMaps.push_back(get(attrMap16)->asA<scene_rdl2::rdl2::Map>());
    if (get(attrMap17)) mMaps.push_back(get(attrMap17)->asA<scene_rdl2::rdl2::Map>());
    if (get(attrMap18)) mMaps.push_back(get(attrMap18)->asA<scene_rdl2::rdl2::Map>());
    if (get(attrMap19)) mMaps.push_back(get(attrMap19)->asA<scene_rdl2::rdl2::Map>());
}

bool
ListMap::getIsListMap(std::vector<const scene_rdl2::rdl2::Map *> &maps) const
{
    maps = mMaps;
    return true;
}

void
ListMap::sample(const scene_rdl2::rdl2::Map *self, moonray::shading::TLState *tls,
    const moonray::shading::State &state, Color *sample)
{
    *sample = sWhite;
}

