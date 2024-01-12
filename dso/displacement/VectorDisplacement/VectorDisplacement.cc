// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0


#include <moonray/rendering/shading/Shading.h>

#include <scene_rdl2/scene/rdl2/rdl2.h>

#include <string>

#include "attributes.cc"
#include "VectorDisplacement_ispc_stubs.h"

#include <moonray/rendering/shading/Shading.h>
#include <scene_rdl2/scene/rdl2/rdl2.h>
#include <scene_rdl2/render/util/stdmemory.h>

#include <string>

using namespace scene_rdl2::math;
using namespace moonray::shading;


//---------------------------------------------------------------------------

RDL2_DSO_CLASS_BEGIN(VectorDisplacement, scene_rdl2::rdl2::Displacement)

public:
    VectorDisplacement(const scene_rdl2::rdl2::SceneClass &sceneClass, const std::string &name);
    ~VectorDisplacement();
    virtual void update();
        
private:
    static void displace(const Displacement *self, moonray::shading::TLState *tls,
                         const State &state, Vec3f *displace);
    
    ispc::VectorDisplacement mIspc; // must be the 1st member

    std::unique_ptr<moonray::shading::Xform> mXform;

RDL2_DSO_CLASS_END(VectorDisplacement)


//---------------------------------------------------------------------------

VectorDisplacement::VectorDisplacement(const scene_rdl2::rdl2::SceneClass &sceneClass, const std::string &name) :
    Parent(sceneClass, name)
{
    mDisplaceFunc = VectorDisplacement::displace;
    mDisplaceFuncv = (scene_rdl2::rdl2::DisplaceFuncv) ispc::VectorDisplacement_getDisplaceFunc();
}

VectorDisplacement::~VectorDisplacement()
{
}

void
VectorDisplacement::update()
{
    // Construct Xform
    mXform = fauxstd::make_unique<moonray::shading::Xform>(this);
    mIspc.mXform = mXform->getIspcXform();
}

void VectorDisplacement::displace(const Displacement *self, moonray::shading::TLState *tls,
                                  const State &state, Vec3f *displace)
{
    const VectorDisplacement* me = static_cast<const VectorDisplacement*>(self);
    
    // Evaluate vector map and apply factor
    Vec3f vector = evalVec3f(me, attrVector, tls, state);
    vector *= me->get(attrFactor);

    if (me->get(attrSourceSpace) == ispc::SOURCE_SPACE_TANGENT) {
        // Transform from tangent space to the geometry local space.
        const Vec3f &N = state.getN();
        const ReferenceFrame frame(N, normalize(state.getdPds()));
        if (me->get(attrTangentSpaceStyle) == ispc::TANGENT_SPACE_STYLE_TNB) {
            // In tangent space, the ReferenceFrame works Z-up whereas 
            // the vector displacement local space is defined as "green"-up.
            vector = frame.localToGlobal(Vec3f(vector.x, vector.z, vector.y));
        } else {
            vector = frame.localToGlobal(Vec3f(vector.x, vector.y, vector.z));
        }
    } else { // Object space
        // Transform from object space to render space
        vector = me->mXform->transformVector(ispc::SHADING_SPACE_OBJECT,
                                             ispc::SHADING_SPACE_RENDER,
                                             state,
                                             vector);
    }

    *displace = vector;
}


//---------------------------------------------------------------------------

