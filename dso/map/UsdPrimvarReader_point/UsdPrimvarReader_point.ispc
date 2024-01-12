// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

/// @file UsdPrimvarReader_point.ispc

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
    if (attributeMapType == PRIMVAR_MAP_TYPE_P) {
        sample.r = state.mP.x;
        sample.g = state.mP.y;
        sample.b = state.mP.z;
    } else {
        if (isProvided(state, me->mPrimitiveAttributeIndex)) {
            const varying Vec3f p = getVec3fAttribute(tls, state, me->mPrimitiveAttributeIndex);
            sample.r = p.x;
            sample.g = p.y;
            sample.b = p.z;
        } else {
            // the primitive attribute is unavailable
            const varying Vec3f p = evalAttrFallback(map, tls, state);
            sample.r = p.x;
            sample.g = p.y;
            sample.b = p.z;
            if (getAttrWarnWhenUnavailable(map)) {
                logEvent(map, me->mMissingAttributeEvent);
            }
        }
    }

    return sample;
}

DEFINE_MAP_SHADER(UsdPrimvarReader_point, sample)

