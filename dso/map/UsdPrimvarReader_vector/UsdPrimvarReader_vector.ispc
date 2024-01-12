// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

/// @file UsdPrimvarReader_vector.ispc

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
    if (attributeMapType == PRIMVAR_MAP_TYPE_DPDS) {
        sample.r = state.mdPds.x;
        sample.g = state.mdPds.y;
        sample.b = state.mdPds.z;
    } else if (attributeMapType == PRIMVAR_MAP_TYPE_DPDT) {
        sample.r = state.mdPdt.x;
        sample.g = state.mdPdt.y;
        sample.b = state.mdPdt.z;
    } else if (attributeMapType == PRIMVAR_MAP_TYPE_DNDS) {
        sample.r = state.mdNds.x;
        sample.g = state.mdNds.y;
        sample.b = state.mdNds.z;
    } else if (attributeMapType == PRIMVAR_MAP_TYPE_DNDT) {
        sample.r = state.mdNdt.x;
        sample.g = state.mdNdt.y;
        sample.b = state.mdNdt.z;
    } else {
        if (isProvided(state, me->mPrimitiveAttributeIndex)) {
            const varying Vec3f v = getVec3fAttribute(tls, state, me->mPrimitiveAttributeIndex);
            sample.r = v.x;
            sample.g = v.y;
            sample.b = v.z;
        } else {
            // the primitive attribute is unavailable
            const varying Vec3f v = evalAttrFallback(map, tls, state);
            sample.r = v.x;
            sample.g = v.y;
            sample.b = v.z;
            if (getAttrWarnWhenUnavailable(map)) {
                logEvent(map, me->mMissingAttributeEvent);
            }
        }
    }

    return sample;
}

DEFINE_MAP_SHADER(UsdPrimvarReader_vector, sample)

