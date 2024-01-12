// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

/// @file UsdPrimvarReader_float3.ispc

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
    } else if (attributeMapType == PRIMVAR_MAP_TYPE_DPDS) {
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
    } else if (attributeMapType == PRIMVAR_MAP_TYPE_N) {
        sample.r = state.mN.x;
        sample.g = state.mN.y;
        sample.b = state.mN.z;
    } else if (attributeMapType == PRIMVAR_MAP_TYPE_NG) {
        sample.r = state.mNg.x;
        sample.g = state.mNg.y;
        sample.b = state.mNg.z;
    } else {
        if (isProvided(state, me->mPrimitiveAttributeIndex)) {
            if (me->mPrimitiveAttributeType == TYPE_RGB) {
                sample = getColorAttribute(tls, state,
                    me->mPrimitiveAttributeIndex);
            } else {
                const varying Vec3f v3 = getVec3fAttribute(tls, state, me->mPrimitiveAttributeIndex);
                sample.r = v3.x;
                sample.g = v3.y;
                sample.b = v3.z;
            }
        } else {
            // the primitive attribute is unavailable
            const varying Vec3f v3 = evalAttrFallback(map, tls, state);
            sample.r = v3.x;
            sample.g = v3.y;
            sample.b = v3.z;
            if (getAttrWarnWhenUnavailable(map)) {
                logEvent(map, me->mMissingAttributeEvent);
            }
        }
    }

    return sample;
}

DEFINE_MAP_SHADER(UsdPrimvarReader_float3, sample)

