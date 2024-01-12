// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

/// @file AttributeMap.cc

#include "attributes.cc"
#include "AttributeMap_ispc_stubs.h"

#include <moonray/rendering/shading/MapApi.h>

using namespace scene_rdl2::math;
using namespace moonray::shading;

RDL2_DSO_CLASS_BEGIN(AttributeMap, scene_rdl2::rdl2::Map)

public:
    AttributeMap(const scene_rdl2::rdl2::SceneClass& sceneClass, const std::string& name);

    ~AttributeMap();

    virtual void update();

private:
    static void sample(const scene_rdl2::rdl2::Map* self, moonray::shading::TLState *tls,
            const moonray::shading::State& state, Color* sample);

    void createLogEvent(const std::string& primAttrType,
                        const std::string& primAttrName);

    ispc::AttributeMap mData;

RDL2_DSO_CLASS_END(AttributeMap)



AttributeMap::AttributeMap(const scene_rdl2::rdl2::SceneClass& sceneClass,
        const std::string& name) : Parent(sceneClass, name)
{
    mSampleFunc = AttributeMap::sample;
    mSampleFuncv = (scene_rdl2::rdl2::SampleFuncv) ispc::AttributeMap_getSampleFunc();
}

AttributeMap::~AttributeMap()
{
}

void
AttributeMap::update()
{
    if (hasChanged(attrMapType) ||
        hasChanged(attrPrimitiveAttributeType) ||
        hasChanged(attrPrimitiveAttributeName)) {
        mOptionalAttributes.clear();
        if (get(attrMapType) == ispc::AttributeMapType::PRIMITIVE_ATTRIBUTE) {
            int type = get(attrPrimitiveAttributeType);
            const std::string& name = get(attrPrimitiveAttributeName);
            if (type == ispc::TYPE_FLOAT) {
                mData.mPrimitiveAttributeType = ispc::TYPE_FLOAT;
                moonray::shading::TypedAttributeKey<float> attributeKey(name);
                mData.mPrimitiveAttributeIndex = attributeKey;
                mOptionalAttributes.push_back(attributeKey);
                createLogEvent("float", name);
            } else if (type == ispc::TYPE_VEC2F) {
                mData.mPrimitiveAttributeType = ispc::TYPE_VEC2F;
                moonray::shading::TypedAttributeKey<Vec2f> attributeKey(name);
                mData.mPrimitiveAttributeIndex = attributeKey;
                mOptionalAttributes.push_back(attributeKey);
                createLogEvent("vec2f", name);
            } else if (type == ispc::TYPE_VEC3F) {
                mData.mPrimitiveAttributeType = ispc::TYPE_VEC3F;
                moonray::shading::TypedAttributeKey<Vec3f> attributeKey(name);
                mData.mPrimitiveAttributeIndex = attributeKey;
                mOptionalAttributes.push_back(attributeKey);
                createLogEvent("vec3f", name);
            } else if (type == ispc::TYPE_RGB) {
                mData.mPrimitiveAttributeType = ispc::TYPE_RGB;
                moonray::shading::TypedAttributeKey<Color> attributeKey(name);
                mData.mPrimitiveAttributeIndex = attributeKey;
                mOptionalAttributes.push_back(attributeKey);
                createLogEvent("rgb", name);
            } else if (type == ispc::TYPE_INT) {
                mData.mPrimitiveAttributeType = ispc::TYPE_INT;
                moonray::shading::TypedAttributeKey<int> attributeKey(name);
                mData.mPrimitiveAttributeIndex = attributeKey;
                mOptionalAttributes.push_back(attributeKey);
                createLogEvent("int", name);
            } else {
                MNRY_ASSERT(0, "unsupported primitive attribute type");
            }
        } else if (get(attrMapType) == ispc::AttributeMapType::SURFACE_P) {
            mData.mPrimitiveAttributeType = ispc::TYPE_VEC3F;
            moonray::shading::TypedAttributeKey<Vec3f> attributeKey("surface_P");
            mData.mPrimitiveAttributeIndex = attributeKey;
            createLogEvent("vec3f", "surface_P");
            mOptionalAttributes.push_back(attributeKey);
        } else if (get(attrMapType) == ispc::AttributeMapType::SURFACE_N) {
            mData.mPrimitiveAttributeType = ispc::TYPE_VEC3F;
            moonray::shading::TypedAttributeKey<Vec3f> attributeKey("surface_N");
            mData.mPrimitiveAttributeIndex = attributeKey;
            mOptionalAttributes.push_back(attributeKey);
            createLogEvent("vec3f", "surface_N");
        } else if (get(attrMapType) == ispc::AttributeMapType::SURFACE_ST) {
            mData.mPrimitiveAttributeType = ispc::TYPE_VEC2F;
            mData.mPrimitiveAttributeIndex = moonray::shading::StandardAttributes::sSurfaceST;
            mOptionalAttributes.push_back(mData.mPrimitiveAttributeIndex);
            createLogEvent("vec2f", "surface_st");
        } else if (get(attrMapType) == ispc::AttributeMapType::CLOSEST_SURFACE_ST) {
            mData.mPrimitiveAttributeType = ispc::TYPE_VEC2F;
            mData.mPrimitiveAttributeIndex = moonray::shading::StandardAttributes::sClosestSurfaceST;
            mOptionalAttributes.push_back(mData.mPrimitiveAttributeIndex);
            createLogEvent("vec2f", "closest_surface_uv");
        } else if (get(attrMapType) == ispc::AttributeMapType::ID) {
            mData.mPrimitiveAttributeType = ispc::TYPE_INT;
            mData.mPrimitiveAttributeIndex = moonray::shading::StandardAttributes::sId;
            mOptionalAttributes.push_back(mData.mPrimitiveAttributeIndex);
            createLogEvent("int", "id");
        } else if (get(attrMapType) == ispc::AttributeMapType::VELOCITY) {
            mData.mPrimitiveAttributeType = ispc::TYPE_VEC3F;
            mData.mPrimitiveAttributeIndex = moonray::shading::StandardAttributes::sVelocity;
            mOptionalAttributes.push_back(mData.mPrimitiveAttributeIndex);
            createLogEvent("vec3f", "velocity");
        } else if (get(attrMapType) == ispc::AttributeMapType::ACCELERATION) {
            mData.mPrimitiveAttributeType = ispc::TYPE_VEC3F;
            mData.mPrimitiveAttributeIndex = moonray::shading::StandardAttributes::sAcceleration;
            mOptionalAttributes.push_back(mData.mPrimitiveAttributeIndex);
            createLogEvent("vec3f", "acceleration");
        } else if (get(attrMapType) == ispc::AttributeMapType::MOTIONVEC) {
            mData.mPrimitiveAttributeType = ispc::TYPE_VEC3F;
            mData.mPrimitiveAttributeIndex = moonray::shading::StandardAttributes::sMotion;
            mOptionalAttributes.push_back(mData.mPrimitiveAttributeIndex);
            createLogEvent("vec3f", "motionvec");
        }
    }
}

void
AttributeMap::createLogEvent(const std::string& primAttrType,
                             const std::string& primAttrName)
{
    // setup an appropiate log event message
    std::ostringstream os;
    os << "Missing primitive attribute '"
        << primAttrType << " " << primAttrName
        << "', using default value";

    mData.mMissingAttributeEvent =
        sLogEventRegistry.createEvent(scene_rdl2::logging::WARN_LEVEL, os.str());
}

void
AttributeMap::sample(const scene_rdl2::rdl2::Map* self, moonray::shading::TLState *tls,
                 const moonray::shading::State& state, Color* sample)
{
    const AttributeMap* me = static_cast<const AttributeMap*>(self);
    if (    me->get(attrMapType) == ispc::AttributeMapType::PRIMITIVE_ATTRIBUTE ||
            me->get(attrMapType) == ispc::AttributeMapType::SURFACE_P ||
            me->get(attrMapType) == ispc::AttributeMapType::SURFACE_N ||
            me->get(attrMapType) == ispc::AttributeMapType::SURFACE_ST ||
            me->get(attrMapType) == ispc::AttributeMapType::CLOSEST_SURFACE_ST ||
            me->get(attrMapType) == ispc::AttributeMapType::ID ||
            me->get(attrMapType) == ispc::AttributeMapType::VELOCITY ||
            me->get(attrMapType) == ispc::AttributeMapType::ACCELERATION ||
            me->get(attrMapType) == ispc::AttributeMapType::MOTIONVEC) {

        int key = me->mData.mPrimitiveAttributeIndex;
        int type = me->mData.mPrimitiveAttributeType;
        if (state.isProvided(key)) {
            if (type == ispc::TYPE_FLOAT) {
                float f = state.getAttribute(
                    moonray::shading::TypedAttributeKey<float>(key));
                *sample = Color(f);
            } else if (type == ispc::TYPE_VEC2F) {
                const Vec2f& v2 = state.getAttribute(
                    moonray::shading::TypedAttributeKey<Vec2f>(key));
                sample->r = v2.x;
                sample->g = v2.y;
                sample->b = 0.0f;
            } else if (type == ispc::TYPE_VEC3F) {
                const Vec3f& v3 = state.getAttribute(
                    moonray::shading::TypedAttributeKey<Vec3f>(key));
                sample->r = v3.x;
                sample->g = v3.y;
                sample->b = v3.z;
            } else if (type == ispc::TYPE_RGB) {
                *sample = state.getAttribute(
                    moonray::shading::TypedAttributeKey<Color>(key));
            } else if (type == ispc::TYPE_INT) {
                int i = state.getAttribute(
                    moonray::shading::TypedAttributeKey<int>(key));
                *sample = Color((float)i);
            } else {
                // there is an attribute with the right name, but the
                // type is unknown/unsupported - so report it as missing
                // and use the default value
                *sample = evalColor(me, attrDefaultValue, tls, state);
                if (me->get(attrWarnWhenUnavailable)) {
                    moonray::shading::logEvent(me, me->mData.mMissingAttributeEvent);
                }
            }
        } else {
            // the primitive attribute is unavailable
            *sample = evalColor(me, attrDefaultValue, tls, state);
            if (me->get(attrWarnWhenUnavailable)) {
                moonray::shading::logEvent(me, me->mData.mMissingAttributeEvent);
            }
        }
    } else if (me->get(attrMapType) == ispc::AttributeMapType::P) {
        const Vec3f& p = state.getP();
        sample->r = p.x;
        sample->g = p.y;
        sample->b = p.z;
    } else if (me->get(attrMapType) == ispc::AttributeMapType::ST) {
        const Vec2f& st = state.getSt();
        sample->r = st.x;
        sample->g = st.y;
        sample->b = 0.0f;
    } else if (me->get(attrMapType) == ispc::AttributeMapType::N) {
        const Vec3f& n = state.getN();
        sample->r = n.x;
        sample->g = n.y;
        sample->b = n.z;
    } else if (me->get(attrMapType) == ispc::AttributeMapType::NG) {
        const Vec3f& ng = state.getNg();
        sample->r = ng.x;
        sample->g = ng.y;
        sample->b = ng.z;
    } else if (me->get(attrMapType) == ispc::AttributeMapType::DPDS) {
        const Vec3f& dpds = state.getdPds();
        sample->r = dpds.x;
        sample->g = dpds.y;
        sample->b = dpds.z;
    } else if (me->get(attrMapType) == ispc::AttributeMapType::DPDT) {
        const Vec3f& dpdt = state.getdPdt();
        sample->r = dpdt.x;
        sample->g = dpdt.y;
        sample->b = dpdt.z;
    } else if (me->get(attrMapType) == ispc::AttributeMapType::DNDS) {
        const Vec3f& dnds = state.getdNds();
        sample->r = dnds.x;
        sample->g = dnds.y;
        sample->b = dnds.z;
    } else if (me->get(attrMapType) == ispc::AttributeMapType::DNDT) {
        const Vec3f& dndt = state.getdNdt();
        sample->r = dndt.x;
        sample->g = dndt.y;
        sample->b = dndt.z;
    } else if (me->get(attrMapType) == ispc::AttributeMapType::MAP_COLOR) {
        *sample = evalColor(me, attrColor, tls, state);
    } else {
        MNRY_ASSERT(0, "unsupported primitive attribute type");
    }
}

