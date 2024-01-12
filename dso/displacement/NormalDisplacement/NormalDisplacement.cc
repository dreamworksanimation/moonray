// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0


#include <moonray/rendering/bvh/shading/State.h>
#include <moonray/rendering/shading/EvalAttribute.h>

#include <scene_rdl2/scene/rdl2/rdl2.h>

#include <string>

#include "attributes.cc"
#include "NormalDisplacement_ispc_stubs.h"


using namespace scene_rdl2::math;
using namespace moonray::shading;


//---------------------------------------------------------------------------

RDL2_DSO_CLASS_BEGIN(NormalDisplacement, scene_rdl2::rdl2::Displacement)

public:
    NormalDisplacement(const scene_rdl2::rdl2::SceneClass &sceneClass, const std::string &name);
    ~NormalDisplacement();
    virtual void update();

private:
    static void displace(const Displacement *self, moonray::shading::TLState *tls,
                         const State &state, Vec3f *displace);
    
RDL2_DSO_CLASS_END(NormalDisplacement)


//---------------------------------------------------------------------------

NormalDisplacement::NormalDisplacement(const scene_rdl2::rdl2::SceneClass &sceneClass, const std::string &name) :
    Parent(sceneClass, name)
{
    mDisplaceFunc = NormalDisplacement::displace; 
    mDisplaceFuncv = (scene_rdl2::rdl2::DisplaceFuncv) ispc::NormalDisplacement_getDisplaceFunc();
}

NormalDisplacement::~NormalDisplacement()
{
}

void
NormalDisplacement::update()
{
}

void NormalDisplacement::displace(const Displacement *self, moonray::shading::TLState *tls,
                                  const State &state, Vec3f *displace)
{
    const NormalDisplacement* me = static_cast<const NormalDisplacement*>(self);
    
    float zeroValue = me->get(attrZeroValue);
    float height = evalFloatWithPreAdd(me, attrHeight, tls, state, -zeroValue);
    float heightMult = evalFloat(me, attrHeightMultiplier, tls, state);
        
    *displace = height * heightMult * state.getN();
}


//---------------------------------------------------------------------------

