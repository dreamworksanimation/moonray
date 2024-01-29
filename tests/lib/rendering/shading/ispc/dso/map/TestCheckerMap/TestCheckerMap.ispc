// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

/// @file TestCheckerMap.ispc

// this should be <rendering/shading/Shading.isph>
// for anything other than the unittest
#include <moonray/rendering/shading/ispc/Shading.isph>
#include "attributes.isph"

inline bool
isEven(varying int x)
{
    // % generates a perfomance warning
    return (x / 2) == ((x + 1) / 2);
}

// return true if we should use colorA, false if colorB
inline varying bool
checker(const varying float s, const varying float t)
{
    const varying bool evenS = isEven((varying int) floor(s));
    const varying bool evenT = isEven((varying int) floor(t));

    return (evenS && evenT) || (!evenS && !evenT);
}

// This function is not differentiable at the important points where
// isEven(s) and isEven(t) change value.
// We'll evaluate colorA and colorB with derivatives and then
// use finite differences to approximate our derivatives.
#define NUM_SAMPLES 5

static varying Color
sample(const uniform Map *            uniform map,
             uniform ShadingTLState * uniform tls,
       const varying State &                  state)
{
    const Color colorA = evalAttrColorA(map, tls, state);
    const Color colorB = evalAttrColorB(map, tls, state);
    const varying float s = state.mSt.x * rcp(getAttrUStep(map));
    const varying float t = state.mSt.y * rcp(getAttrVStep(map));
    Color result = checker(s, t)? colorA : colorB;
    return result;
}

DEFINE_MAP_SHADER(TestCheckerMap, sample)
               
