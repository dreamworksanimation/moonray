// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0


#include "attributes.isph"

#include <moonray/rendering/shading/ispc/MapApi.isph>

static Color
sample(const uniform Map* uniform map, uniform ShadingTLState* uniform tls, const varying State& state)
{
    varying Color result = evalAttrColor(map, tls, state);
    return result;
}

DEFINE_MAP_SHADER(ExtraAovMap, sample)

