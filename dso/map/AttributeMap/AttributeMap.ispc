// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

/// @file AttributeMap.ispc

#include "attributes.isph"

#include <moonray/rendering/shading/ispc/MapApi.isph>

enum AttributeMapType {
    PRIMITIVE_ATTRIBUTE = 0,
    P = 1,
    ST = 2,
    N = 3,
    NG = 4,
    DPDS = 5,
    DPDT = 6,
    DNDS = 7,
    DNDT = 8,
    MAP_COLOR = 9,
    SURFACE_P = 12,
    SURFACE_N = 13,
    SURFACE_ST = 14,
    CLOSEST_SURFACE_ST = 15,
    ID = 16,
    VELOCITY = 17,
    ACCELERATION = 18,
    MOTIONVEC = 19
};

ISPC_UTIL_EXPORT_ENUM_TO_HEADER(AttributeMapType);

enum PrimitiveAttributeType {
    TYPE_FLOAT = 0,
    TYPE_VEC2F = 1,
    TYPE_VEC3F = 2,
    TYPE_RGB = 3,
    TYPE_INT = 4
};

ISPC_UTIL_EXPORT_ENUM_TO_HEADER(PrimitiveAttributeType);


struct AttributeMap
{
    uniform int mPrimitiveAttributeIndex;
    uniform int mPrimitiveAttributeType;
    uniform int mMissingAttributeEvent;
};

ISPC_UTIL_EXPORT_UNIFORM_STRUCT_TO_HEADER(AttributeMap);

static Color
sample(const uniform Map *            uniform  map,
             uniform ShadingTLState * uniform  tls,
       const varying State &                   state)
{
    varying Color sample;
    const uniform AttributeMap * uniform me = MAP_GET_ISPC_PTR(AttributeMap, map);
    if (    getAttrMapType(map) == PRIMITIVE_ATTRIBUTE ||
            getAttrMapType(map) == SURFACE_P ||
            getAttrMapType(map) == SURFACE_N ||
            getAttrMapType(map) == SURFACE_ST ||
            getAttrMapType(map) == CLOSEST_SURFACE_ST ||
            getAttrMapType(map) == ID ||
            getAttrMapType(map) == VELOCITY ||
            getAttrMapType(map) == ACCELERATION ||
            getAttrMapType(map) == MOTIONVEC) {

        if (isProvided(state, me->mPrimitiveAttributeIndex)) {
            if (me->mPrimitiveAttributeType == TYPE_FLOAT) {
                sample = Color_ctor(getFloatAttribute(tls, state,
                    me->mPrimitiveAttributeIndex));
            } else if (me->mPrimitiveAttributeType == TYPE_VEC2F) {
                Vec2f v2 = getVec2fAttribute(tls, state,
                    me->mPrimitiveAttributeIndex);
                sample.r = v2.x;
                sample.g = v2.y;
                sample.b = 0.0f;
            } else if (me->mPrimitiveAttributeType == TYPE_VEC3F) {
                Vec3f v3 = getVec3fAttribute(tls, state,
                    me->mPrimitiveAttributeIndex);
                sample.r = v3.x;
                sample.g = v3.y;
                sample.b = v3.z;
            } else if (me->mPrimitiveAttributeType == TYPE_RGB) {
                sample = getColorAttribute(tls, state,
                    me->mPrimitiveAttributeIndex);
            } else if (me->mPrimitiveAttributeType == TYPE_INT) {
                sample = Color_ctor((float)getIntAttribute(tls, state,
                    me->mPrimitiveAttributeIndex));
            } else {
                // there is an attribute with the right name, but the
                // type is unknown/unsupported - so report it as missing
                // and use the default value
                sample = evalAttrDefaultValue(map, tls, state);
                if (getAttrWarnWhenUnavailable(map)) {
                    logEvent(map, me->mMissingAttributeEvent);
                }
            }
        } else {
            // the primitive attribute is unavailable
            sample = evalAttrDefaultValue(map, tls, state);
            if (getAttrWarnWhenUnavailable(map)) {
                logEvent(map, me->mMissingAttributeEvent);
            }
        }
    } else if (getAttrMapType(map) == P) {
        sample.r = state.mP.x;
        sample.g = state.mP.y;
        sample.b = state.mP.z;
    } else if (getAttrMapType(map) == ST) {
        sample.r = state.mSt.x;
        sample.g = state.mSt.y;
        sample.b = 0.0f;
    } else if (getAttrMapType(map) == N) {
        sample.r = state.mN.x;
        sample.g = state.mN.y;
        sample.b = state.mN.z;
    } else if (getAttrMapType(map) == NG) {
        sample.r = state.mNg.x;
        sample.g = state.mNg.y;
        sample.b = state.mNg.z;
    } else if (getAttrMapType(map) == DPDS) {
        sample.r = state.mdPds.x;
        sample.g = state.mdPds.y;
        sample.b = state.mdPds.z;
    } else if (getAttrMapType(map) == DPDT) {
        sample.r = state.mdPdt.x;
        sample.g = state.mdPdt.y;
        sample.b = state.mdPdt.z;
    } else if (getAttrMapType(map) == DNDS) {
        sample.r = state.mdNds.x;
        sample.g = state.mdNds.y;
        sample.b = state.mdNds.z;
    } else if (getAttrMapType(map) == DNDT) {
        sample.r = state.mdNdt.x;
        sample.g = state.mdNdt.y;
        sample.b = state.mdNdt.z;
    } else if (getAttrMapType(map) == MAP_COLOR) {
        sample = evalAttrColor(map, tls, state);
    } else {
        MNRY_ASSERT(0); // unsupported primitive attribute type
    }
    return sample;
}

DEFINE_MAP_SHADER(AttributeMap, sample)

