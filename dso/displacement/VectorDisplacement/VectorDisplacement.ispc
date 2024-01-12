// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

/// @file VectorDisplacement.ispc

#include "attributes.isph"

#include <moonray/rendering/shading/ispc/Shading.isph>
#include <scene_rdl2/common/platform/IspcUtil.isph>

#include <scene_rdl2/common/math/ispc/ReferenceFrame.isph>

enum SOURCE_SPACE {
    SOURCE_SPACE_TANGENT = 0,
    SOURCE_SPACE_OBJECT = 1
};
ISPC_UTIL_EXPORT_ENUM_TO_HEADER(SOURCE_SPACE);


enum TANGENT_SPACE_STYLE {
    TANGENT_SPACE_STYLE_TNB = 0,
    TANGENT_SPACE_STYLE_TBN = 1
};
ISPC_UTIL_EXPORT_ENUM_TO_HEADER(TANGENT_SPACE_STYLE);

struct VectorDisplacement
{
    const Xform *mXform;
};
ISPC_UTIL_EXPORT_UNIFORM_STRUCT_TO_HEADER(VectorDisplacement);

static varying Vec3f
displace(const uniform Displacement *   uniform me,
               uniform ShadingTLState * uniform tls,
         const varying State &          state)
{ 
    const uniform VectorDisplacement * uniform vectorDisplace =
        DISPLACEMENT_GET_ISPC_CPTR(VectorDisplacement, me);

    // Evaluate vector map and apply factor
    Vec3f vector = evalAttrVector(me, tls, state);
    vector = vector * getAttrFactor(me);

    if (getAttrSourceSpace(me) == SOURCE_SPACE_TANGENT) {
        // Transform from tangent space to the geometry local space.
        const Vec3f &N = state.mN;
        ReferenceFrame frame;
        ReferenceFrame_init(frame, N, normalize(state.mdPds));
        if (getAttrTangentSpaceStyle(me) == TANGENT_SPACE_STYLE_TNB) {
            // In tangent space, the ReferenceFrame works Z-up whereas 
            // the vector displacement local space is defined as "green"-up.
            vector = localToGlobal(frame, Vec3f_ctor(vector.x, vector.z, vector.y));
        } else {
            vector = localToGlobal(frame, Vec3f_ctor(vector.x, vector.y, vector.z));
        }
        return vector;
    } else { // Object space
        // Transform from object space to render space
        vector = transformVector(vectorDisplace->mXform,
                                 SHADING_SPACE_OBJECT,
                                 SHADING_SPACE_RENDER,
                                 state,
                                 vector);
        return vector;
    }
}

DEFINE_DISPLACEMENT_SHADER(VectorDisplacement, displace)
               
