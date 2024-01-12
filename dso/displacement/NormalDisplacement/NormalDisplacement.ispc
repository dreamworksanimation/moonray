// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

/// @file NormalDisplacement.ispc

#include <moonray/rendering/shading/ispc/Shading.isph>
#include "attributes.isph"

static varying Vec3f
displace(const uniform Displacement *   uniform me,
               uniform ShadingTLState * uniform tls,
         const varying State &          state)
{
    float height = getAttrHeight(me);
    height *= evalAttrHeightMultiplier(me, tls, state);
    if (height != 0) {
        Color c;
        bool bound = evalMapBinding(me, tls, state, (const AttributeKey * uniform)attrHeight, &c);
        if (bound) {
            float avg = (c.r + c.g + c.b) / 3.;
            float zeroValue = getAttrZeroValue(me);
            height = height * (avg - zeroValue);
        }
    }
    return state.mN * height;
}

DEFINE_DISPLACEMENT_SHADER(NormalDisplacement, displace)
               
