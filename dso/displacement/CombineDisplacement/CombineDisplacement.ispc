// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

/// @file CombineDisplacement.ispc

#include "attributes.isph"

#include <moonray/rendering/shading/ispc/Shading.isph>
#include <scene_rdl2/common/platform/IspcUtil.isph>

typedef uniform intptr_t (*uniform DisplaceFn) (    const uniform Displacement *    uniform displacement,
                                                    uniform ShadingTLState *        uniform tls,
                                                    uniform int                     numState,
                                                    const varying State *           const uniform state,
                                                    varying Vec3f *                 uniform out);

enum CombineOpType {
    ADD                 =       0,
    MAX_MAGNITUDE       =       1,
    MIN_MAGNITUDE       =       2
};

ISPC_UTIL_EXPORT_ENUM_TO_HEADER(CombineOpType);

struct CombineDisplacement
{
    const uniform Displacement * uniform mDisplacement1;
    const uniform Displacement * uniform mDisplacement2;
    uniform intptr_t mDisplace1Func;
    uniform intptr_t mDisplace2Func;
};
ISPC_UTIL_EXPORT_UNIFORM_STRUCT_TO_HEADER(CombineDisplacement);

static varying Vec3f
displace(const uniform Displacement *   uniform me,
               uniform ShadingTLState * uniform tls,
         const varying State &          state)
{
    const uniform CombineDisplacement * uniform cd = DISPLACEMENT_GET_ISPC_PTR(CombineDisplacement, me);

    varying Vec3f delta1 = Vec3f_ctor(0.0f, 0.0f, 0.0f);
    varying Vec3f delta2 = Vec3f_ctor(0.0f, 0.0f, 0.0f);

    DisplaceFn displace1fn = (DisplaceFn) cd->mDisplace1Func;
    DisplaceFn displace2fn = (DisplaceFn) cd->mDisplace2Func;
    if (displace1fn != NULL) displace1fn(cd->mDisplacement1, tls, 1, &state, &delta1);
    if (displace2fn != NULL) displace2fn(cd->mDisplacement2, tls, 1, &state, &delta2);

    const uniform float scale1 = getAttrScale1(me);
    const uniform float scale2 = getAttrScale2(me);

    delta1 = delta1 * scale1;
    delta2 = delta2 * scale2;

    varying Vec3f delta;
    const uniform int operation = getAttrOperation(me);
    switch (operation) {
        case ADD:
            delta = delta1 + delta2;
            break;
        case MAX_MAGNITUDE:
            // Use maximum vector based on absolute length.
            delta = lengthSqr(delta1) > lengthSqr(delta2) ? delta1 : delta2;
            break;
        case MIN_MAGNITUDE:
            // Use minimum vector based on absolute length.
            delta = lengthSqr(delta1) < lengthSqr(delta2) ? delta1 : delta2;
            break;
        default:
            delta = delta1;
            break;
    }
    return delta;
}

DEFINE_DISPLACEMENT_SHADER(CombineDisplacement, displace)
               
