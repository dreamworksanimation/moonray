// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

/// @file TestMandelbrot.ispc

#include <moonray/rendering/shading/ispc/Shading.isph>
#include "attributes.isph"

static varying Color
sample(const uniform Map *            uniform map,
             uniform ShadingTLState * uniform tls,
       const varying State &                  state)
{
    uniform float colorRange = (uniform float) max(1, getAttrNumColors(map) - 1);
    if (getAttrNumColors(map) < 1) {
        return Color_ctor(0.f); // todo: report error
    }

    uniform Vec2f scale = getAttrScale(map);
    uniform Vec2f offset = getAttrOffset(map);
    varying float u = state.mSt.x * scale.x + offset.x;
    varying float v = state.mSt.y * scale.y + offset.y;
    u -= floor(u);
    v -= floor(v);
    varying float X = u * 3 - 2.f;
    varying float Y = v * 3 - 1.5f;
    varying float x = X;
    varying float y = Y;

    varying int count = 0;
    varying float dist;
    do {
        varying float oldx = x;
        x = x * x - y * y + X;
        y = 2 * oldx * y + Y;
        ++count;
        dist = x * x + y * y;
    } while (count < getAttrMaxCount(map) && dist < 4.f);

    if (count == getAttrMaxCount(map)) {
        return Color_ctor(0.f);
    }

    count = count % getAttrNumColors(map);
    varying float c = (varying float) count;
    return Color_ctor(1.f, c / colorRange, 0.f);
}

DEFINE_MAP_SHADER(TestMandelbrot, sample)
               
