// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

/// @file TestDebugMap.ispc

// this should be <rendering/shading/Shading.isph>
// for anything other than the unittest
#include <moonray/rendering/shading/ispc/Shading.isph>
#include "attributes.isph"

static Color
sample(const uniform Map *            uniform  map,
             uniform ShadingTLState * uniform  tls,
       const varying State &                   state)
{
    Color result;
    switch(getAttrMapType(map))
    {
    case 0: // P
        result = asAColor(state.mP);
        break;
    case 1: // St
        result = Color_ctor(state.mSt.x, state.mSt.y, 0.f);
        break;
    case 2: // N
        result = asAColor(state.mN);
        break;
    case 3: // Ng
        result = asAColor(state.mNg);
        break;
    case 4: // dPds
        result = asAColor(state.mdPds);
        break;
    case 5: // dPdt
        result = asAColor(state.mdPdt);
        break;
    case 6: // todo: hair surface st
        result.r = 0.f;
        result.g = 0.f;
        result.b = 0.f;
        break;
    case 7: // dsdx
        result.r = result.g = result.b = state.mdSdx;
        break;
    case 8: // dsdy
        result.r = result.g = result.b = state.mdSdy;
        break;
    case 9: // dtdx
        result.r = result.g = result.b = state.mdTdx;
        break;
    case 10: // dtdy
        result.r = result.g = result.b = state.mdTdy;
        break;
    }
    return result;
}

DEFINE_MAP_SHADER(TestDebugMap, sample)
               
