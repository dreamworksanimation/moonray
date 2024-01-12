// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0


#include "attributes.cc"
#include "SwitchMaterial_ispc_stubs.h"

#include <moonray/rendering/shading/MaterialApi.h>
#include <moonray/rendering/shading/EvalShader.h>

#include <string>

using namespace scene_rdl2::math;
using namespace moonray::shading;

//---------------------------------------------------------------------------

RDL2_DSO_CLASS_BEGIN(SwitchMaterial, scene_rdl2::rdl2::Material)

public:
    SwitchMaterial(const scene_rdl2::rdl2::SceneClass& sceneClass, const std::string& name);
    ~SwitchMaterial();
    virtual void update();

private:
    static void shade(const scene_rdl2::rdl2::Material* self, moonray::shading::TLState *tls,
                      const State& state, BsdfBuilder& bsdfBuilder);

    static float presence(const scene_rdl2::rdl2::Material* self,
                          moonray::shading::TLState *tls,
                          const moonray::shading::State& state);

    static float ior(const scene_rdl2::rdl2::Material* self,
                     moonray::shading::TLState *tls,
                     const moonray::shading::State& state);
    ispc::SwitchMaterial mIspc;


RDL2_DSO_CLASS_END(SwitchMaterial)


//---------------------------------------------------------------------------

SwitchMaterial::SwitchMaterial(const scene_rdl2::rdl2::SceneClass& sceneClass,
                               const std::string& name) :
    Parent(sceneClass, name)
{
    mShadeFunc = SwitchMaterial::shade;
    mShadeFuncv = (scene_rdl2::rdl2::ShadeFuncv) ispc::SwitchMaterial_getShadeFunc();
    mPresenceFunc = SwitchMaterial::presence;
    for (size_t i = 0; i < ispc::MAX_MATERIALS; ++i) {
        mIspc.mMaterial[i] = 0;
    }
}

SwitchMaterial::~SwitchMaterial()
{
}

void
SwitchMaterial::update()
{
    const int choice = get(attrChoice);
    if (choice < 0 || choice >= ispc::MAX_MATERIALS) {
        error("Out of range choice selection: ", choice, 
            ".   Only values between 0 and 63 are accepted.");
        return;
    }

    mIspc.mMaterial[0] = (intptr_t)get(attrMaterial0);
    mIspc.mMaterial[1] = (intptr_t)get(attrMaterial1);
    mIspc.mMaterial[2] = (intptr_t)get(attrMaterial2);
    mIspc.mMaterial[3] = (intptr_t)get(attrMaterial3);
    mIspc.mMaterial[4] = (intptr_t)get(attrMaterial4);
    mIspc.mMaterial[5] = (intptr_t)get(attrMaterial5);
    mIspc.mMaterial[6] = (intptr_t)get(attrMaterial6);
    mIspc.mMaterial[7] = (intptr_t)get(attrMaterial7);
    mIspc.mMaterial[8] = (intptr_t)get(attrMaterial8);
    mIspc.mMaterial[9] = (intptr_t)get(attrMaterial9);

    mIspc.mMaterial[10] = (intptr_t)get(attrMaterial10);
    mIspc.mMaterial[11] = (intptr_t)get(attrMaterial11);
    mIspc.mMaterial[12] = (intptr_t)get(attrMaterial12);
    mIspc.mMaterial[13] = (intptr_t)get(attrMaterial13);
    mIspc.mMaterial[14] = (intptr_t)get(attrMaterial14);
    mIspc.mMaterial[15] = (intptr_t)get(attrMaterial15);
    mIspc.mMaterial[16] = (intptr_t)get(attrMaterial16);
    mIspc.mMaterial[17] = (intptr_t)get(attrMaterial17);
    mIspc.mMaterial[18] = (intptr_t)get(attrMaterial18);
    mIspc.mMaterial[19] = (intptr_t)get(attrMaterial19);

    mIspc.mMaterial[20] = (intptr_t)get(attrMaterial20);
    mIspc.mMaterial[21] = (intptr_t)get(attrMaterial21);
    mIspc.mMaterial[22] = (intptr_t)get(attrMaterial22);
    mIspc.mMaterial[23] = (intptr_t)get(attrMaterial23);
    mIspc.mMaterial[24] = (intptr_t)get(attrMaterial24);
    mIspc.mMaterial[25] = (intptr_t)get(attrMaterial25);
    mIspc.mMaterial[26] = (intptr_t)get(attrMaterial26);
    mIspc.mMaterial[27] = (intptr_t)get(attrMaterial27);
    mIspc.mMaterial[28] = (intptr_t)get(attrMaterial28);
    mIspc.mMaterial[29] = (intptr_t)get(attrMaterial29);

    mIspc.mMaterial[30] = (intptr_t)get(attrMaterial30);
    mIspc.mMaterial[31] = (intptr_t)get(attrMaterial31);
    mIspc.mMaterial[32] = (intptr_t)get(attrMaterial32);
    mIspc.mMaterial[33] = (intptr_t)get(attrMaterial33);
    mIspc.mMaterial[34] = (intptr_t)get(attrMaterial34);
    mIspc.mMaterial[35] = (intptr_t)get(attrMaterial35);
    mIspc.mMaterial[36] = (intptr_t)get(attrMaterial36);
    mIspc.mMaterial[37] = (intptr_t)get(attrMaterial37);
    mIspc.mMaterial[38] = (intptr_t)get(attrMaterial38);
    mIspc.mMaterial[39] = (intptr_t)get(attrMaterial39);

    mIspc.mMaterial[40] = (intptr_t)get(attrMaterial40);
    mIspc.mMaterial[41] = (intptr_t)get(attrMaterial41);
    mIspc.mMaterial[42] = (intptr_t)get(attrMaterial42);
    mIspc.mMaterial[43] = (intptr_t)get(attrMaterial43);
    mIspc.mMaterial[44] = (intptr_t)get(attrMaterial44);
    mIspc.mMaterial[45] = (intptr_t)get(attrMaterial45);
    mIspc.mMaterial[46] = (intptr_t)get(attrMaterial46);
    mIspc.mMaterial[47] = (intptr_t)get(attrMaterial47);
    mIspc.mMaterial[48] = (intptr_t)get(attrMaterial48);
    mIspc.mMaterial[49] = (intptr_t)get(attrMaterial49);

    mIspc.mMaterial[50] = (intptr_t)get(attrMaterial50);
    mIspc.mMaterial[51] = (intptr_t)get(attrMaterial51);
    mIspc.mMaterial[52] = (intptr_t)get(attrMaterial52);
    mIspc.mMaterial[53] = (intptr_t)get(attrMaterial53);
    mIspc.mMaterial[54] = (intptr_t)get(attrMaterial54);
    mIspc.mMaterial[55] = (intptr_t)get(attrMaterial55);
    mIspc.mMaterial[56] = (intptr_t)get(attrMaterial56);
    mIspc.mMaterial[57] = (intptr_t)get(attrMaterial57);
    mIspc.mMaterial[58] = (intptr_t)get(attrMaterial58);
    mIspc.mMaterial[59] = (intptr_t)get(attrMaterial59);

    mIspc.mMaterial[60] = (intptr_t)get(attrMaterial60);
    mIspc.mMaterial[61] = (intptr_t)get(attrMaterial61);
    mIspc.mMaterial[62] = (intptr_t)get(attrMaterial62);
    mIspc.mMaterial[63] = (intptr_t)get(attrMaterial63);
}


//---------------------------------------------------------------------------

void
SwitchMaterial::shade(const scene_rdl2::rdl2::Material* self, moonray::shading::TLState *tls,
                      const State& state, BsdfBuilder& bsdfBuilder)
{
    const SwitchMaterial* me = static_cast<const SwitchMaterial*>(self);
    int choice = me->get(attrChoice);
    if (choice < 0 || choice >= ispc::MAX_MATERIALS) {
        return;
    }

    const scene_rdl2::rdl2::Material* mtl = reinterpret_cast<scene_rdl2::rdl2::Material*>(me->mIspc.mMaterial[choice]);
    if (mtl) {
        moonray::shading::shade(self, mtl, tls, state, bsdfBuilder);
    }
}

float
SwitchMaterial::presence(const scene_rdl2::rdl2::Material* self,
                         moonray::shading::TLState *tls,
                         const moonray::shading::State& state)
{
    const SwitchMaterial* me = static_cast<const SwitchMaterial*>(self);
    int choice = me->get(attrChoice);
    if (choice < 0 || choice >= ispc::MAX_MATERIALS) {
        return 1.0f;
    }

    const scene_rdl2::rdl2::Material* mtl = reinterpret_cast<scene_rdl2::rdl2::Material*>(me->mIspc.mMaterial[choice]);
    if (mtl) {
        return moonray::shading::presence(self, mtl, tls, state);
    } else {
        return 1.0f;
    }
}

float
SwitchMaterial::ior(const scene_rdl2::rdl2::Material* self,
                    moonray::shading::TLState *tls,
                    const moonray::shading::State& state)
{
    const SwitchMaterial* me = static_cast<const SwitchMaterial*>(self);
    int choice = me->get(attrChoice);
    if (choice < 0 || choice >= ispc::MAX_MATERIALS) {
        return 1.0f;
    }

    const scene_rdl2::rdl2::Material* mtl = reinterpret_cast<scene_rdl2::rdl2::Material*>(me->mIspc.mMaterial[choice]);
    if (mtl) {
        return moonray::shading::ior(self, mtl, tls, state);
    } else {
        return 1.0f;
    }
}

