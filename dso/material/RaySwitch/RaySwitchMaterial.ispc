// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

#include "attributes.isph"

#include <moonray/rendering/shading/ispc/EvalShader.isph>
#include <moonray/rendering/shading/ispc/MaterialApi.isph>

struct RaySwitchMaterial
{
    uniform intptr_t mDefaultMaterialObj;
    uniform bool mCutoutCameraRays;
};

export const uniform RaySwitchMaterial * uniform
RaySwitchMaterial_get(const uniform Material * uniform material)
{
    return MATERIAL_GET_ISPC_CPTR(RaySwitchMaterial, material);
}

static void
shade(const uniform Material *      uniform  me,
            uniform ShadingTLState *uniform  tls,
      const varying State                   &state,
            varying BsdfBuilder             &bsdfBuilder)
{
    const uniform RaySwitchMaterial * uniform switchMtl = RaySwitchMaterial_get(me);

    // If this material is cutting out, terminate the ray
    if (!isIndirect(state) && switchMtl->mCutoutCameraRays) {
        BsdfBuilder_setEarlyTermination(bsdfBuilder);
        return;
    }

    // We have already performed substitutions in the integrator so we don't need to switch here.
    // If there was no substitution, use the default material.

    if (switchMtl->mDefaultMaterialObj) {
        shadeMaterial((const uniform Material * uniform)(switchMtl->mDefaultMaterialObj),
                      tls, 1, &state, &bsdfBuilder);
    }
}

DEFINE_MATERIAL_SHADER(RaySwitchMaterial, shade)

