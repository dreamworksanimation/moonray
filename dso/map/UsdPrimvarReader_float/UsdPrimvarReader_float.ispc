// Copyright 2023 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

/// @file UsdPrimvarReader_float.ispc

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

    if (isProvided(state, me->mPrimitiveAttributeIndex)) {
        const varying float v = getFloatAttribute(tls, state, me->mPrimitiveAttributeIndex);
        sample.r = v;
        sample.g = v;
        sample.b = v;
    } else {
        // the primitive attribute is unavailable
        const varying float v = evalAttrFallback(map, tls, state);
        sample.r = v;
        sample.g = v;
        sample.b = v;
        if (getAttrWarnWhenUnavailable(map)) {
            logEvent(map, tls, me->mMissingAttributeEvent);
        }
    }

    return sample;
}

DEFINE_MAP_SHADER(UsdPrimvarReader_float, sample)

