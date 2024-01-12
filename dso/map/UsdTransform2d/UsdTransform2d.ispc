// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

/// @file UsdTransform2d.ispc

#include "attributes.isph"

#include <moonray/rendering/shading/ispc/MapApi.isph>

static Color
sample(const uniform Map *            uniform  map,
             uniform ShadingTLState * uniform  tls,
       const varying State &                   state)
{
    const varying Color input = evalAttrIn(map, tls, state);
    varying Vec2f uv = Vec2f_ctor(input.r, input.g);

    const uniform Vec2f scale = getAttrScale(map);
    uv.x = scale.x * uv.x;
    uv.y = scale.y * uv.y;

    const uniform float rotation = getAttrRotation(map);
    if (!isZero(rotation)) {
        float s, c;
        const uniform float theta = deg2rad(rotation);
        sincos(theta, &s, &c);
        const float tmpUvX = uv.x * c - uv.y * s;
        uv.y = uv.x * s + uv.y * c;
        uv.x = tmpUvX;
    }

    const uniform Vec2f translation = getAttrTranslation(map);
    uv.x = uv.x + translation.x;
    uv.y = uv.y + translation.y;

    return Color_ctor(uv.x, uv.y, 0.f);
}

DEFINE_MAP_SHADER(UsdTransform2d, sample)

