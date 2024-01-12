// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

/// @file UsdPrimvarReader_int.cc

#include "attributes.cc"
#include "UsdPrimvarReader_int_ispc_stubs.h"

#include <moonray/map/primvar/Primvar.h>
#include <moonray/rendering/shading/MapApi.h>

using namespace moonray::shading;
using namespace scene_rdl2::math;

//----------------------------------------------------------------------------

RDL2_DSO_CLASS_BEGIN(UsdPrimvarReader_int, scene_rdl2::rdl2::Map)

public:
    UsdPrimvarReader_int(const scene_rdl2::rdl2::SceneClass &sceneClass, const std::string &name);
    ~UsdPrimvarReader_int() override;
    void update() override;

private:
    static void sample(const scene_rdl2::rdl2::Map *self, moonray::shading::TLState *tls,
                       const moonray::shading::State &state, Color *sample);

    ispc::UsdPrimvarReader mIspc;

RDL2_DSO_CLASS_END(UsdPrimvarReader_int)

//----------------------------------------------------------------------------

UsdPrimvarReader_int::UsdPrimvarReader_int(const scene_rdl2::rdl2::SceneClass &sceneClass, const std::string &name):
    Parent(sceneClass, name)
{
    mSampleFunc = UsdPrimvarReader_int::sample;
    mSampleFuncv = (scene_rdl2::rdl2::SampleFuncv) ispc::UsdPrimvarReader_int_getSampleFunc();
}

UsdPrimvarReader_int::~UsdPrimvarReader_int()
{
}

void
UsdPrimvarReader_int::update()
{
    if (hasChanged(attrVarName)) {
        mOptionalAttributes.clear();
        const std::string& varname = get(attrVarName);
        if (varname == "id") {
            mIspc.mAttributeMapType = ispc::PRIMVAR_MAP_TYPE_ID;
            mIspc.mPrimitiveAttributeIndex = moonray::shading::StandardAttributes::sId;
            mOptionalAttributes.push_back(mIspc.mPrimitiveAttributeIndex);
        } else {
            mIspc.mAttributeMapType = ispc::PRIMVAR_MAP_TYPE_PRIMITIVE_ATTRIBUTE;
            moonray::shading::TypedAttributeKey<int> attributeKey(varname);
            mIspc.mPrimitiveAttributeIndex = attributeKey;
            mOptionalAttributes.push_back(attributeKey);
        }
        ::moonshine::primvar::createLogEvent("int",
                                             varname,
                                             mIspc.mMissingAttributeEvent,
                                             sLogEventRegistry);
    }
}

void
UsdPrimvarReader_int::sample(const scene_rdl2::rdl2::Map *self, moonray::shading::TLState *tls,
                        const moonray::shading::State &state, Color *sample)
{
    const UsdPrimvarReader_int* me = static_cast<const UsdPrimvarReader_int*>(self);

    int key = me->mIspc.mPrimitiveAttributeIndex;
    if (state.isProvided(key)) {
        const float& v = static_cast<float>(state.getAttribute(moonray::shading::TypedAttributeKey<int>(key)));
        sample->r = v;
        sample->g = v;
        sample->b = v;
    } else {
        // the primitive attribute is unavailable, use fallback parameter
        const float& v = static_cast<float>(me->get(attrFallback));
        sample->r = v;
        sample->g = v;
        sample->b = v;
        if (me->get(attrWarnWhenUnavailable)) {
            moonray::shading::logEvent(me, me->mIspc.mMissingAttributeEvent);
        }
    }
}

