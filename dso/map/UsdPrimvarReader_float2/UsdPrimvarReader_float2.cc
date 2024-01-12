// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

/// @file UsdPrimvarReader_float2.cc

#include "attributes.cc"
#include "UsdPrimvarReader_float2_ispc_stubs.h"

#include <moonray/map/primvar/Primvar.h>
#include <moonray/rendering/shading/MapApi.h>

using namespace moonray::shading;
using namespace scene_rdl2::math;

//----------------------------------------------------------------------------

RDL2_DSO_CLASS_BEGIN(UsdPrimvarReader_float2, scene_rdl2::rdl2::Map)

public:
    UsdPrimvarReader_float2(const scene_rdl2::rdl2::SceneClass &sceneClass, const std::string &name);
    ~UsdPrimvarReader_float2() override;
    void update() override;

private:
    static void sample(const scene_rdl2::rdl2::Map *self, moonray::shading::TLState *tls,
                       const moonray::shading::State &state, Color *sample);

    ispc::UsdPrimvarReader mIspc;

RDL2_DSO_CLASS_END(UsdPrimvarReader_float2)

//----------------------------------------------------------------------------

UsdPrimvarReader_float2::UsdPrimvarReader_float2(const scene_rdl2::rdl2::SceneClass &sceneClass, const std::string &name):
    Parent(sceneClass, name)
{
    mSampleFunc = UsdPrimvarReader_float2::sample;
    mSampleFuncv = (scene_rdl2::rdl2::SampleFuncv) ispc::UsdPrimvarReader_float2_getSampleFunc();
}

UsdPrimvarReader_float2::~UsdPrimvarReader_float2()
{
}

void
UsdPrimvarReader_float2::update()
{
    if (hasChanged(attrVarName)) {
        mOptionalAttributes.clear();
        const std::string& varname = get(attrVarName);
        // We handle these special case "varname" values:
        //      "st"
        //      "surface_st"
        //      "closest_surface_st"
        if (varname == "st") {
            mIspc.mAttributeMapType = ispc::PRIMVAR_MAP_TYPE_ST;
        } else if (varname == "surface_st") {
            mIspc.mAttributeMapType = ispc::PRIMVAR_MAP_TYPE_SURFACE_ST;
            mIspc.mPrimitiveAttributeIndex = moonray::shading::StandardAttributes::sSurfaceST;
            mOptionalAttributes.push_back(mIspc.mPrimitiveAttributeIndex);
        } else if (varname == "closest_surface_uv") {
            mIspc.mAttributeMapType = ispc::PRIMVAR_MAP_TYPE_CLOSEST_SURFACE_ST;
            mIspc.mPrimitiveAttributeIndex = moonray::shading::StandardAttributes::sClosestSurfaceST;
            mOptionalAttributes.push_back(mIspc.mPrimitiveAttributeIndex);
        } else {
            // Not a special case so create key from the
            // "varname" value
            mIspc.mAttributeMapType = ispc::PRIMVAR_MAP_TYPE_PRIMITIVE_ATTRIBUTE;
            moonray::shading::TypedAttributeKey<Vec2f> attributeKey(varname);
            mIspc.mPrimitiveAttributeIndex = attributeKey;
            mOptionalAttributes.push_back(attributeKey);
        }
        ::moonshine::primvar::createLogEvent("vec2f",
                                             varname,
                                             mIspc.mMissingAttributeEvent,
                                             sLogEventRegistry);
    }
}

void
UsdPrimvarReader_float2::sample(const scene_rdl2::rdl2::Map *self, moonray::shading::TLState *tls,
                        const moonray::shading::State &state, Color *sample)
{
    const UsdPrimvarReader_float2* me = static_cast<const UsdPrimvarReader_float2*>(self);

    const int attributeMapType = me->mIspc.mAttributeMapType;
    if (attributeMapType == ispc::PRIMVAR_MAP_TYPE_ST) {
        const Vec2f& st = state.getSt();
        sample->r = st.x;
        sample->g = st.y;
        sample->b = 0.0f;
    } else {
        int key = me->mIspc.mPrimitiveAttributeIndex;
        if (state.isProvided(key)) {
            const Vec2f& v2 = state.getAttribute(moonray::shading::TypedAttributeKey<Vec2f>(key));
            sample->r = v2.x;
            sample->g = v2.y;
            sample->b = 0.0f;
        } else {
            // the primitive attribute is unavailable, use fallback parameter
            const Vec2f& v2 = evalVec2f(me, attrFallback, tls, state);
            sample->r = v2.x;
            sample->g = v2.y;
            sample->b = 0.0f;
            if (me->get(attrWarnWhenUnavailable)) {
                moonray::shading::logEvent(me, me->mIspc.mMissingAttributeEvent);
            }
        }
    }
}

