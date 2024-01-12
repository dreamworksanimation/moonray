// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

/// @file UsdPrimvarReader_float3.cc

#include "attributes.cc"
#include "UsdPrimvarReader_float3_ispc_stubs.h"

#include <moonray/map/primvar/Primvar.h>
#include <moonray/rendering/shading/MapApi.h>

using namespace moonray::shading;
using namespace scene_rdl2::math;

//----------------------------------------------------------------------------

RDL2_DSO_CLASS_BEGIN(UsdPrimvarReader_float3, scene_rdl2::rdl2::Map)

public:
    UsdPrimvarReader_float3(const scene_rdl2::rdl2::SceneClass &sceneClass, const std::string &name);
    ~UsdPrimvarReader_float3() override;
    void update() override;

private:
    static void sample(const scene_rdl2::rdl2::Map *self, moonray::shading::TLState *tls,
                       const moonray::shading::State &state, Color *sample);

    ispc::UsdPrimvarReader mIspc;

RDL2_DSO_CLASS_END(UsdPrimvarReader_float3)

//----------------------------------------------------------------------------

UsdPrimvarReader_float3::UsdPrimvarReader_float3(const scene_rdl2::rdl2::SceneClass &sceneClass, const std::string &name):
    Parent(sceneClass, name)
{
    mSampleFunc = UsdPrimvarReader_float3::sample;
    mSampleFuncv = (scene_rdl2::rdl2::SampleFuncv) ispc::UsdPrimvarReader_float3_getSampleFunc();
}

UsdPrimvarReader_float3::~UsdPrimvarReader_float3()
{
}

void
UsdPrimvarReader_float3::update()
{
    if (hasChanged(attrVarName)) {
        mOptionalAttributes.clear();
        mIspc.mPrimitiveAttributeType = ispc::TYPE_VEC3F;
        const std::string& varname = get(attrVarName);
        // We handle these special case "varname" values:
        //      "P"
        //      "dpds"
        //      "dpdt"
        //      "dnds"
        //      "dndt"
        //      "N"
        //      "Ng"
        //      "surface_P"
        //      "surface_N"
        //      "velocity"
        //      "acceleration"
        //      "motion"
        //      "Cd"
        //      "displayColor"
        if (varname == "P") {
            mIspc.mAttributeMapType = ispc::PRIMVAR_MAP_TYPE_P;
        } else if (varname == "dpds") {
            mIspc.mAttributeMapType = ispc::PRIMVAR_MAP_TYPE_DPDS;
        } else if (varname == "dpdt") {
            mIspc.mAttributeMapType = ispc::PRIMVAR_MAP_TYPE_DPDT;
        } else if (varname == "dnds") {
            mIspc.mAttributeMapType = ispc::PRIMVAR_MAP_TYPE_DNDS;
        } else if (varname == "dndt") {
            mIspc.mAttributeMapType = ispc::PRIMVAR_MAP_TYPE_DNDT;
        } else if (varname == "N") {
            mIspc.mAttributeMapType = ispc::PRIMVAR_MAP_TYPE_N;
        } else if (varname == "Ng") {
            mIspc.mAttributeMapType = ispc::PRIMVAR_MAP_TYPE_NG;
        } else if (varname == "surface_P") {
            mIspc.mAttributeMapType = ispc::PRIMVAR_MAP_TYPE_SURFACE_P;
            moonray::shading::TypedAttributeKey<Vec3f> attributeKey("surface_P");
            mIspc.mPrimitiveAttributeIndex = attributeKey;
            mOptionalAttributes.push_back(attributeKey);
        } else if (varname == "surface_N") {
            mIspc.mAttributeMapType = ispc::PRIMVAR_MAP_TYPE_SURFACE_N;
            moonray::shading::TypedAttributeKey<Vec3f> attributeKey("surface_N");
            mIspc.mPrimitiveAttributeIndex = attributeKey;
            mOptionalAttributes.push_back(attributeKey);
        } else if (varname == "velocity") {
            mIspc.mAttributeMapType = ispc::PRIMVAR_MAP_TYPE_VELOCITY;
            mIspc.mPrimitiveAttributeIndex = moonray::shading::StandardAttributes::sVelocity;
            mOptionalAttributes.push_back(mIspc.mPrimitiveAttributeIndex);
        } else if (varname == "acceleration") {
            mIspc.mAttributeMapType = ispc::PRIMVAR_MAP_TYPE_ACCELERATION;
            mIspc.mPrimitiveAttributeIndex = moonray::shading::StandardAttributes::sAcceleration;
            mOptionalAttributes.push_back(mIspc.mPrimitiveAttributeIndex);
        } else if (varname == "motion") {
            mIspc.mAttributeMapType = ispc::PRIMVAR_MAP_TYPE_MOTIONVEC;
            mIspc.mPrimitiveAttributeIndex = moonray::shading::StandardAttributes::sMotion;
            mOptionalAttributes.push_back(mIspc.mPrimitiveAttributeIndex);
        } else if (varname == "Cd" || varname == "displayColor") {
            mIspc.mPrimitiveAttributeType = ispc::TYPE_RGB;
            mIspc.mAttributeMapType = ispc::PRIMVAR_MAP_TYPE_PRIMITIVE_ATTRIBUTE;
            moonray::shading::TypedAttributeKey<Color> attributeKey(varname);
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
UsdPrimvarReader_float3::sample(const scene_rdl2::rdl2::Map *self, moonray::shading::TLState *tls,
                        const moonray::shading::State &state, Color *sample)
{
    const UsdPrimvarReader_float3* me = static_cast<const UsdPrimvarReader_float3*>(self);

    const int attributeMapType = me->mIspc.mAttributeMapType;
    if (attributeMapType == ispc::PRIMVAR_MAP_TYPE_P) {
        const Vec3f& p = state.getP();
        sample->r = p.x;
        sample->g = p.y;
        sample->b = p.z;
    } else if (attributeMapType == ispc::PRIMVAR_MAP_TYPE_DPDS) {
        const Vec3f& dpds = state.getdPds();
        sample->r = dpds.x;
        sample->g = dpds.y;
        sample->b = dpds.z;
    } else if (attributeMapType == ispc::PRIMVAR_MAP_TYPE_DPDT) {
        const Vec3f& dpdt = state.getdPdt();
        sample->r = dpdt.x;
        sample->g = dpdt.y;
        sample->b = dpdt.z;
    } else if (attributeMapType == ispc::PRIMVAR_MAP_TYPE_DNDS) {
        const Vec3f& dnds = state.getdNds();
        sample->r = dnds.x;
        sample->g = dnds.y;
        sample->b = dnds.z;
    } else if (attributeMapType == ispc::PRIMVAR_MAP_TYPE_DNDT) {
        const Vec3f& dndt = state.getdNdt();
        sample->r = dndt.x;
        sample->g = dndt.y;
        sample->b = dndt.z;
    } else if (attributeMapType == ispc::PRIMVAR_MAP_TYPE_N) {
        const Vec3f& N = state.getN();
        sample->r = N.x;
        sample->g = N.y;
        sample->b = N.z;
    } else if (attributeMapType == ispc::PRIMVAR_MAP_TYPE_NG) {
        const Vec3f& Ng = state.getNg();
        sample->r = Ng.x;
        sample->g = Ng.y;
        sample->b = Ng.z;
    } else {
        int key = me->mIspc.mPrimitiveAttributeIndex;
        if (state.isProvided(key)) {
            if (me->mIspc.mPrimitiveAttributeType == ispc::TYPE_RGB) {
                *sample = state.getAttribute(moonray::shading::TypedAttributeKey<Color>(key));
            } else {
                const Vec3f& v3 = state.getAttribute(moonray::shading::TypedAttributeKey<Vec3f>(key));
                sample->r = v3.x;
                sample->g = v3.y;
                sample->b = v3.z;
            }
        } else {
            // the primitive attribute is unavailable, use fallback parameter
            const Vec3f& v3 = evalVec3f(me, attrFallback, tls, state);
            sample->r = v3.x;
            sample->g = v3.y;
            sample->b = v3.z;
            if (me->get(attrWarnWhenUnavailable)) {
                moonray::shading::logEvent(me, me->mIspc.mMissingAttributeEvent);
            }
        }
    }
}

