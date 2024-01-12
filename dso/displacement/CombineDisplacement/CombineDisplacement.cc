// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0


#include <moonray/rendering/bvh/shading/State.h>
#include <moonray/rendering/shading/EvalAttribute.h>

#include <scene_rdl2/scene/rdl2/rdl2.h>

#include <string>

#include "attributes.cc"
#include "CombineDisplacement_ispc_stubs.h"

using namespace scene_rdl2::math;
using namespace moonray::shading;

//---------------------------------------------------------------------------

RDL2_DSO_CLASS_BEGIN(CombineDisplacement, scene_rdl2::rdl2::Displacement)

public:
    CombineDisplacement(const scene_rdl2::rdl2::SceneClass &sceneClass, const std::string &name);
    ~CombineDisplacement();
    virtual void update();

private:
    static void displace(const Displacement *self, moonray::shading::TLState *tls,
                         const State &state, Vec3f *displace);
    
    bool verifyInputs();

    ispc::CombineDisplacement mIspc;
    scene_rdl2::rdl2::Displacement* mDisplacement1;
    scene_rdl2::rdl2::Displacement* mDisplacement2;

RDL2_DSO_CLASS_END(CombineDisplacement)


//---------------------------------------------------------------------------

CombineDisplacement::CombineDisplacement(const scene_rdl2::rdl2::SceneClass &sceneClass, const std::string &name) :
    Parent(sceneClass, name)
{
    mDisplaceFunc = CombineDisplacement::displace;
    mDisplaceFuncv = (scene_rdl2::rdl2::DisplaceFuncv) ispc::CombineDisplacement_getDisplaceFunc();
}

CombineDisplacement::~CombineDisplacement()
{
}

bool
CombineDisplacement::verifyInputs() {
    if (mDisplacement1 == nullptr && mDisplacement2 == nullptr) {
        warn("CombineDisplacement: No input displacement objects provided.");
    } else if (mDisplacement1 == nullptr) {
        warn("CombineDisplacement: Input 1 displacement object not provided.");
    } else if (mDisplacement2 == nullptr) {
        warn("CombineDisplacement: Input 2 displacement object not provided.");
    }

    return true;
}

void
CombineDisplacement::update()
{
    mDisplacement1 = get(attrInput1) ?
                get(attrInput1)->asA<scene_rdl2::rdl2::Displacement>() : nullptr;
    mDisplacement2 = get(attrInput2) ?
                get(attrInput2)->asA<scene_rdl2::rdl2::Displacement>() : nullptr;

    if (!verifyInputs()) {
        fatal("CombineDisplacement input validation failed");
        return;
    }

    mIspc.mDisplacement1 = mDisplacement1;
    mIspc.mDisplacement2 = mDisplacement2;

    mIspc.mDisplace1Func = (mDisplacement1 != nullptr) ? (intptr_t) mDisplacement1->mDisplaceFuncv : (intptr_t) nullptr;
    mIspc.mDisplace2Func = (mDisplacement2 != nullptr) ? (intptr_t) mDisplacement2->mDisplaceFuncv : (intptr_t) nullptr;
}

void CombineDisplacement::displace(const Displacement *self, moonray::shading::TLState *tls,
                                  const State &state, Vec3f *displace)
{
    const CombineDisplacement* me = static_cast<const CombineDisplacement*>(self);
    Vec3f delta1(0.0f), delta2(0.0f);
    if (me->mDisplacement1 != nullptr) me->mDisplacement1->displace(tls, state, &delta1);
    if (me->mDisplacement2 != nullptr) me->mDisplacement2->displace(tls, state, &delta2);
    const float scale1 = evalFloat(me, attrScale1, tls, state);
    const float scale2 = evalFloat(me, attrScale2, tls, state);

    delta1 = delta1 * scale1;
    delta2 = delta2 * scale2;

    const int operation = me->get(attrOperation);
    switch (operation) {
        case ispc::CombineOpType::ADD:
            *displace = delta1 + delta2;
            break;
        case ispc::CombineOpType::MAX_MAGNITUDE:
            // Use maximum vector based on absolute length.
            *displace = delta1.lengthSqr() > delta2.lengthSqr() ? delta1 : delta2;
            break;
        case ispc::CombineOpType::MIN_MAGNITUDE:
            // Use minimum vector based on absolute length.
            *displace = delta1.lengthSqr() < delta2.lengthSqr() ? delta1 : delta2;
            break;
        default:
            *displace = delta1;
            break;
    }
}


//---------------------------------------------------------------------------

