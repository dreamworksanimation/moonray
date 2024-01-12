// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

/// @file UsdPrimvarReader_point.cc

#include "attributes.cc"
#include "UsdPrimvarReader_point_ispc_stubs.h"

#include <moonray/map/primvar/Primvar.h>
#include <moonray/rendering/shading/MapApi.h>

using namespace moonray::shading;
using namespace scene_rdl2::math;

//----------------------------------------------------------------------------

RDL2_DSO_CLASS_BEGIN(UsdPrimvarReader_point, scene_rdl2::rdl2::Map)

public:
    UsdPrimvarReader_point(const scene_rdl2::rdl2::SceneClass &sceneClass, const std::string &name);
    ~UsdPrimvarReader_point() override;
    void update() override;

private:
    static void sample(const scene_rdl2::rdl2::Map *self, moonray::shading::TLState *tls,
                       const moonray::shading::State &state, Color *sample);

    ispc::UsdPrimvarReader mIspc;

RDL2_DSO_CLASS_END(UsdPrimvarReader_point)

//----------------------------------------------------------------------------

UsdPrimvarReader_point::UsdPrimvarReader_point(const scene_rdl2::rdl2::SceneClass &sceneClass, const std::string &name):
    Parent(sceneClass, name)
{
    mSampleFunc = UsdPrimvarReader_point::sample;
    mSampleFuncv = (scene_rdl2::rdl2::SampleFuncv) ispc::UsdPrimvarReader_point_getSampleFunc();
}

UsdPrimvarReader_point::~UsdPrimvarReader_point()
{
}

void
UsdPrimvarReader_point::update()
{
    if (hasChanged(attrVarName)) {
        mOptionalAttributes.clear();
        const std::string& varname = get(attrVarName);
        // We handle these special case "varname" values:
        //      "P"
        //      "surface_P"
        if (varname == "P") {
            mIspc.mAttributeMapType = ispc::PRIMVAR_MAP_TYPE_P;
        } else if (varname == "surface_P") {
            mIspc.mAttributeMapType = ispc::PRIMVAR_MAP_TYPE_SURFACE_P;
            moonray::shading::TypedAttributeKey<Vec3f> attributeKey("surface_P");
            mIspc.mPrimitiveAttributeIndex = attributeKey;
            mOptionalAttributes.push_back(attributeKey);
        } else {
            // Not a special case so create key from the
            // "varname" value
            mIspc.mAttributeMapType = ispc::PRIMVAR_MAP_TYPE_PRIMITIVE_ATTRIBUTE;
            moonray::shading::TypedAttributeKey<Vec3f> attributeKey(varname);
            mIspc.mPrimitiveAttributeIndex = attributeKey;
            mOptionalAttributes.push_back(attributeKey);
        }
        ::moonshine::primvar::createLogEvent("vec3f",
                                             varname,
                                             mIspc.mMissingAttributeEvent,
                                             sLogEventRegistry);
    }
}

void
UsdPrimvarReader_point::sample(const scene_rdl2::rdl2::Map *self, moonray::shading::TLState *tls,
                        const moonray::shading::State &state, Color *sample)
{
    const UsdPrimvarReader_point* me = static_cast<const UsdPrimvarReader_point*>(self);

    const int attributeMapType = me->mIspc.mAttributeMapType;
    if (attributeMapType == ispc::PRIMVAR_MAP_TYPE_P) {
        const Vec3f& p = state.getP();
        sample->r = p.x;
        sample->g = p.y;
        sample->b = p.z;
    } else {
        int key = me->mIspc.mPrimitiveAttributeIndex;
        if (state.isProvided(key)) {
            const Vec3f& p = state.getAttribute(moonray::shading::TypedAttributeKey<Vec3f>(key));
            sample->r = p.x;
            sample->g = p.y;
            sample->b = p.z;
        } else {
            // the primitive attribute is unavailable, use fallback parameter
            const Vec3f& p = evalVec3f(me, attrFallback, tls, state);
            sample->r = p.x;
            sample->g = p.y;
            sample->b = p.z;
            if (me->get(attrWarnWhenUnavailable)) {
                moonray::shading::logEvent(me, me->mIspc.mMissingAttributeEvent);
            }
        }
    }
}

