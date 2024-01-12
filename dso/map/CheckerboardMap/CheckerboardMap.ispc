// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0


#include "attributes.isph"

#include <moonray/rendering/shading/ispc/MapApi.isph>

static Color
sample(const uniform Map* uniform map, uniform ShadingTLState* uniform tls, const varying State& state)
{
    varying Color result;
    varying Vec2f st;
    switch (getAttrTextureEnum(map)) {
    case 1:
        st = asAVec2f(evalAttrInputTextureCoordinate(map, tls, state));
        break;
    case 0:
    default:
        st = state.mSt;
        break;
    }

    //Put UVs in 0-1 range
    varying float smod = fmod(st.x * getAttrUTiles(map) / 2.0f, 1.0f);
    varying float tmod = fmod(st.y * getAttrVTiles(map) / 2.0f, 1.0f);
    
    //If negative, wrap it into positive range
    if (smod < 0.0f) smod += 1.0f;
    if (tmod < 0.0f) tmod += 1.0f;

    if (smod < 0.5f) {
        if (tmod < 0.5f) {
            result = getAttrColorA(map);
        } else {
            result = getAttrColorB(map);
        }
    } else {
        if (tmod < 0.5f) {
            result = getAttrColorB(map);
        } else {
            result = getAttrColorA(map);
        }
    }
    
    return result;
}

DEFINE_MAP_SHADER(CheckerboardMap, sample)

