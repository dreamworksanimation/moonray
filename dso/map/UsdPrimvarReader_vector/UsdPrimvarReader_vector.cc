// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

/// @file UsdPrimvarReader_vector.cc

#include "attributes.cc"
#include "UsdPrimvarReader_vector_ispc_stubs.h"

#include <moonray/map/primvar/Primvar.h>

#include <moonray/rendering/shading/MapApi.h>

using namespace moonray::shading;
using namespace scene_rdl2::math;

//----------------------------------------------------------------------------

RDL2_DSO_CLASS_BEGIN(UsdPrimvarReader_vector, scene_rdl2::rdl2::Map)

public:
    UsdPrimvarReader_vector(const scene_rdl2::rdl2::SceneClass &sceneClass, const std::string &name);
    ~UsdPrimvarReader_vector() override;
    void update() override;

private:
    static void sample(const scene_rdl2::rdl2::Map *self, moonray::shading::TLState *tls,
                       const moonray::shading::State &state, Color *sample);

    ispc::UsdPrimvarReader mIspc;

RDL2_DSO_CLASS_END(UsdPrimvarReader_vector)

//----------------------------------------------------------------------------

UsdPrimvarReader_vector::UsdPrimvarReader_vector(const scene_rdl2::rdl2::SceneClass &sceneClass, const std::string &name):
    Parent(sceneClass, name)
{
    mSampleFunc = UsdPrimvarReader_vector::sample;
    mSampleFuncv = (scene_rdl2::rdl2::SampleFuncv) ispc::UsdPrimvarReader_vector_getSampleFunc();
}

UsdPrimvarReader_vector::~UsdPrimvarReader_vector()
{
}

void
UsdPrimvarReader_vector::update()
{
    if (hasChanged(attrVarName)) {
        mOptionalAttributes.clear();
        const std::string& varname = get(attrVarName);
        // We handle these special case "varname" values:
        //      "dpds"
        //      "dpdt"
        //      "dnds"
        //      "dndt"
        //      "velocity"
        //      "acceleration"
        //      "motion"
        if (varname == "dpds") {
            mIspc.mAttributeMapType = ispc::PRIMVAR_MAP_TYPE_DPDS;
        } else if (varname == "dpdt") {
            mIspc.mAttributeMapType = ispc::PRIMVAR_MAP_TYPE_DPDT;
        } else if (varname == "dnds") {
            mIspc.mAttributeMapType = ispc::PRIMVAR_MAP_TYPE_DNDS;
        } else if (varname == "dndt") {
            mIspc.mAttributeMapType = ispc::PRIMVAR_MAP_TYPE_DNDT;
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
        } else {
            // Not a special case so create key from the
            // "varname" value
            mIspc.mAttributeMapType = ispc::PRIMVAR_MAP_TYPE_PRIMITIVE_ATTRIBUTE;
            moonray::shading::TypedAttributeKey<Vec3f> attributeKey(varname);
            mIspc.mPrimitiveAttributeIndex = attributeKey;
            mOptionalAttributes.push_back(attributeKey);
        }
        ::moonshine::primvar::createLogEvent("vector",
                                             varname,
                                             mIspc.mMissingAttributeEvent,
                                             sLogEventRegistry);
    }
}

void
UsdPrimvarReader_vector::sample(const scene_rdl2::rdl2::Map *self, moonray::shading::TLState *tls,
                        const moonray::shading::State &state, Color *sample)
{
    const UsdPrimvarReader_vector* me = static_cast<const UsdPrimvarReader_vector*>(self);

    const int attributeMapType = me->mIspc.mAttributeMapType;
    if (attributeMapType == ispc::PRIMVAR_MAP_TYPE_DPDS) {
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
    } else {
        int key = me->mIspc.mPrimitiveAttributeIndex;
        if (state.isProvided(key)) {
            const Vec3f& p = state.getAttribute(moonray::shading::TypedAttributeKey<Vec3f>(key));
            sample->r = p.x;
            sample->g = p.y;
            sample->b = p.z;
        } else {
            // the primitive attribute is unavailable, use fallback parameter
            const Vec3f& v = evalVec3f(me, attrFallback, tls, state);
            sample->r = v.x;
            sample->g = v.y;
            sample->b = v.z;
            if (me->get(attrWarnWhenUnavailable)) {
                moonray::shading::logEvent(me, me->mIspc.mMissingAttributeEvent);
            }
        }
    }
}

