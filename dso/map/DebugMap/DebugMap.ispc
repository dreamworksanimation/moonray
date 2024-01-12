// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

/// @file DebugMap.ispc

#include "attributes.isph"

#include <moonray/rendering/shading/ispc/MapApi.isph>

enum DebugMapType {
    P = 0,
    ST = 1,
    N = 2,
    NG = 3,
    DPDS = 4,
    DPDT = 5,
    PRIMITIVE_ATTRIBUTE = 6
};

ISPC_UTIL_EXPORT_ENUM_TO_HEADER(DebugMapType);

enum PrimitiveAttributeType {
    TYPE_FLOAT = 0,
    TYPE_VEC2F = 1,
    TYPE_VEC3F = 2,
    TYPE_RGB = 3
};

ISPC_UTIL_EXPORT_ENUM_TO_HEADER(PrimitiveAttributeType);

inline void
sample2D(const varying Vec2f& value, uniform bool checkerBoard, varying Color& sample)
{
    sample.r = value.x;
    sample.g = value.y;
    sample.b = 0.0f;

    if (checkerBoard) {
        if (((((int)(value.x * 16.0f) % 2) +
              ((int)(value.y * 16.0f) % 2)) % 2) == 0) {
            sample = sample * 0.5f;
        }
    }
}

inline void
sample3D(const varying Vec3f& value, uniform bool checkerBoard, varying Color& sample)
{
    sample.r = value.x;
    sample.g = value.y;
    sample.b = value.z;
    if (checkerBoard) {
        if ((((((int)(value.x * 16.0f) % 2) +
                ((int)(value.y * 16.0f) % 2)) +
                ((int)(value.z * 16.0f) % 2))) % 2 == 0) {
            sample = sample * 0.5f;
        }
    }
}

struct DebugMap
{
    uniform int mPrimitiveAttributeIndex;
    uniform int mPrimitiveAttributeType;
};

ISPC_UTIL_EXPORT_UNIFORM_STRUCT_TO_HEADER(DebugMap);

static Color
sample(const uniform Map *            uniform  map,
             uniform ShadingTLState * uniform  tls,
       const varying State &                   state)
{
    Color sample;
    
    const uniform DebugMap * uniform me = MAP_GET_ISPC_PTR(DebugMap, map);

    if (getAttrDebugMapType(map) == N) {
        sample3D(evalNormalInput(map, tls, state), getAttrCheckerboard(map), sample);
    } else if (getAttrDebugMapType(map) == ST) {
        sample2D(state.mSt, getAttrCheckerboard(map), sample);
    } else if (getAttrDebugMapType(map) == P) {
        sample3D(state.mP, getAttrCheckerboard(map), sample);
    } else if (getAttrDebugMapType(map) == NG) {
        sample3D(state.mNg, getAttrCheckerboard(map), sample);
    } else if (getAttrDebugMapType(map) == DPDS) {
        sample3D(state.mdPds, getAttrCheckerboard(map), sample);
    } else if (getAttrDebugMapType(map) == DPDT) {
        sample3D(state.mdPdt, getAttrCheckerboard(map), sample);
    } else if (getAttrDebugMapType(map) == PRIMITIVE_ATTRIBUTE) {
        if (me->mPrimitiveAttributeType == TYPE_FLOAT) {
            sample = Color_ctor(getFloatAttribute(tls, state, me->mPrimitiveAttributeIndex));
        } else if (me->mPrimitiveAttributeType == TYPE_VEC2F) {
            Vec2f v2 = getVec2fAttribute(tls, state, me->mPrimitiveAttributeIndex);
            sample2D(v2, getAttrCheckerboard(map), sample);
        } else if (me->mPrimitiveAttributeType == TYPE_VEC3F) {
            Vec3f v3 = getVec3fAttribute(tls, state, me->mPrimitiveAttributeIndex);
            sample3D(v3, getAttrCheckerboard(map), sample);
        } else if (me->mPrimitiveAttributeType == TYPE_RGB) {
            Color c = getColorAttribute(tls, state, me->mPrimitiveAttributeIndex);
            sample3D(Vec3f_ctor(c.r, c.g, c.b), getAttrCheckerboard(map), sample);
        }
    } else {
        sample.r = 0.f;
        sample.g = 0.f;
        sample.b = 0.f;
        return sample;
    }
    
    return sample;
}

DEFINE_MAP_SHADER(DebugMap, sample)
               
