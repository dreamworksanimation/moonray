// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

/// @file TestDisplacement.ispc

// this should be <rendering/shading/Shading.isph>
// for anything other than the unittest
#include <moonray/rendering/shading/ispc/Shading.isph>
#include "attributes.isph"

static varying Vec3f
displace(const uniform Displacement *   uniform me,
               uniform ShadingTLState * uniform tls,
         const varying State &          state)
{
    float height = evalAttrHeight(me, tls, state);
    Vec3f out = state.mN * height;
    return out;
}

DEFINE_DISPLACEMENT_SHADER(TestDisplacement, displace)

