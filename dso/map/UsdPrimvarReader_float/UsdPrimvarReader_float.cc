// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

/// @file UsdPrimvarReader_float.cc

#include "attributes.cc"
#include "UsdPrimvarReader_float_ispc_stubs.h"

#include <moonray/map/primvar/Primvar.h>
#include <moonray/rendering/shading/MapApi.h>

using namespace moonray::shading;
using namespace scene_rdl2::math;

//----------------------------------------------------------------------------

RDL2_DSO_CLASS_BEGIN(UsdPrimvarReader_float, scene_rdl2::rdl2::Map)

public:
    UsdPrimvarReader_float(const scene_rdl2::rdl2::SceneClass &sceneClass, const std::string &name);
    ~UsdPrimvarReader_float() override;
    void update() override;

private:
    static void sample(const scene_rdl2::rdl2::Map *self, moonray::shading::TLState *tls,
                       const moonray::shading::State &state, Color *sample);

    ispc::UsdPrimvarReader mIspc;

RDL2_DSO_CLASS_END(UsdPrimvarReader_float)

//----------------------------------------------------------------------------

UsdPrimvarReader_float::UsdPrimvarReader_float(const scene_rdl2::rdl2::SceneClass &sceneClass, const std::string &name):
    Parent(sceneClass, name)
{
    mSampleFunc = UsdPrimvarReader_float::sample;
    mSampleFuncv = (scene_rdl2::rdl2::SampleFuncv) ispc::UsdPrimvarReader_float_getSampleFunc();
}

UsdPrimvarReader_float::~UsdPrimvarReader_float()
{
}

void
UsdPrimvarReader_float::update()
{
    if (hasChanged(attrVarName)) {
        mOptionalAttributes.clear();
        const std::string& varname = get(attrVarName);
        mIspc.mAttributeMapType = ispc::PRIMVAR_MAP_TYPE_PRIMITIVE_ATTRIBUTE;
        moonray::shading::TypedAttributeKey<float> attributeKey(varname);
        mIspc.mPrimitiveAttributeIndex = attributeKey;
        mOptionalAttributes.push_back(attributeKey);
        ::moonshine::primvar::createLogEvent("float",
                                             varname,
                                             mIspc.mMissingAttributeEvent,
                                             sLogEventRegistry);
    }
}

void
UsdPrimvarReader_float::sample(const scene_rdl2::rdl2::Map *self, moonray::shading::TLState *tls,
                        const moonray::shading::State &state, Color *sample)
{
    const UsdPrimvarReader_float* me = static_cast<const UsdPrimvarReader_float*>(self);

    int key = me->mIspc.mPrimitiveAttributeIndex;
    if (state.isProvided(key)) {
        const float& v = state.getAttribute(moonray::shading::TypedAttributeKey<float>(key));
        sample->r = v;
        sample->g = v;
        sample->b = v;
    } else {
        // the primitive attribute is unavailable, use fallback parameter
        const float& v = evalFloat(me, attrFallback, tls, state);
        sample->r = v;
        sample->g = v;
        sample->b = v;
        if (me->get(attrWarnWhenUnavailable)) {
            moonray::shading::logEvent(me, me->mIspc.mMissingAttributeEvent);
        }
    }
}

