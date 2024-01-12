// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

/// @file UsdPrimvarReader_float2.ispc

#include "attributes.isph"

#include <moonray/map/primvar/ispc/Primvar.isph>
#include <moonray/rendering/shading/ispc/MapApi.isph>

static Color
sample(const uniform Map* uniform map,
       uniform ShadingTLState* uniform tls,
       const varying State& state)
{
    varying Color sample;
    const uniform UsdPrimvarReader * uniform me = MAP_GET_ISPC_PTR(UsdPrimvarReader, map);

    const uniform int attributeMapType = me->mAttributeMapType;
    if (attributeMapType == PRIMVAR_MAP_TYPE_ST) {
        sample.r = state.mSt.x;
        sample.g = state.mSt.y;
        sample.b = 0.0f;
    } else {
        if (isProvided(state, me->mPrimitiveAttributeIndex)) {
            const varying Vec2f v2 = getVec2fAttribute(tls, state, me->mPrimitiveAttributeIndex);
            sample.r = v2.x;
            sample.g = v2.y;
            sample.b = 0.0f;
        } else {
            // the primitive attribute is unavailable
            const varying Vec2f v2 = evalAttrFallback(map, tls, state);
            sample.r = v2.x;
            sample.g = v2.y;
            sample.b = 0.0f;
            if (getAttrWarnWhenUnavailable(map)) {
                logEvent(map, me->mMissingAttributeEvent);
            }
        }
    }

    return sample;
}

DEFINE_MAP_SHADER(UsdPrimvarReader_float2, sample)

