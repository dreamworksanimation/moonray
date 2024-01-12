// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

/// @file SwitchMaterial.ispc

#include "attributes.isph"

#include <moonray/rendering/shading/ispc/EvalShader.isph>
#include <moonray/rendering/shading/ispc/MaterialApi.isph>

enum SwitchConstants {
    MAX_MATERIALS = 64,
};
ISPC_UTIL_EXPORT_ENUM_TO_HEADER(SwitchConstants);

struct SwitchMaterial
{
    uniform intptr_t mMaterial[64];
};

export const uniform SwitchMaterial * uniform
SwitchMaterial_get(const uniform Material * uniform material)
{
    return MATERIAL_GET_ISPC_CPTR(SwitchMaterial, material);
}

static void
shade(const uniform Material *      uniform  me,
            uniform ShadingTLState *uniform  tls,
      const varying State                   &state,
            varying BsdfBuilder             &bsdfBuilder)
{
    const uniform SwitchMaterial * uniform switchMtl = SwitchMaterial_get(me);
    const uniform int choice = getAttrChoice(me);
    if (choice < 0 || choice > MAX_MATERIALS) {
        return;
    }

    const uniform Material * uniform mtl = (const uniform Material * uniform)(switchMtl->mMaterial[choice]);
    if (mtl) {
        shadeMaterial(mtl, tls, 1, &state, &bsdfBuilder);
    }
}

DEFINE_MATERIAL_SHADER(SwitchMaterial, shade)

