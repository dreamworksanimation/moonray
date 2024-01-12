// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

/// @file TestMap.ispc

// this should be <rendering/shading/Shading.isph>
// for anything other than the unittest
#include <moonray/rendering/shading/ispc/Shading.isph>
#include "attributes.isph"

static varying Color
sample(const uniform Map *            uniform map,
             uniform ShadingTLState * uniform tls,
       const varying State &                  state)
{
    switch (getAttrMode(map))
    {
    case 0:
        {
            float ans = state.mSt.x * evalFloat(map, tls, state, attrMult);
            return Color_ctor(ans);
        }
        break;
    case 1:
        {
            return getAttrColor1(map);
        }
        break;
    case 2:
        {
            if (getAttrBool1(map)) {
                return Color_ctor(1.);
            } else {
                return Color_ctor(0.);
            }
        }
        break;
    case 3:
        {
            float ans = getAttrInt1(map);
            return Color_ctor(ans);
        }
        break;
    case 4:
        {
            Vec2f v = getAttrVec21(map);
            return Color_ctor(v.x, v.y, 0.0);
        }
        break;
    case 5:
        {
            return asAColor(getAttrVec31(map));
        }
        break;
    case 6:
        {
            float ans = getAttrFloat1(map);
            return Color_ctor(ans);
        }
        break;
    case 7:
        {
            return evalAttrColor2(map, tls, state);
        }
        break;
    case 8:
        {
            return Color_ctor(evalAttrFloat2(map, tls, state));
        }
        break;
    case 9:
        {
            return asAColor(evalAttrVec32(map, tls, state));
        }
        break;

        // evalNormal test
    case 10:
        {
            return asAColor(getAttrVec3f10(map));
        }
        break;
    case 11:
        {
            return asAColor(evalNormalVec3f11(map, tls, state));
        }
        break;
    case 12:
        {
            Vec2f v2 = evalAttrVec22(map, tls, state);
            return Color_ctor(v2.x, v2.y, 0.0f);
        }
        break;

    default:
        assert(0);
    }
}

DEFINE_MAP_SHADER(TestMap, sample)
               
