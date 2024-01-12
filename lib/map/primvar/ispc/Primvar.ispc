// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

/// @file Primvar.ispc

#include "Primvar.isph"

#include <moonray/rendering/shading/ispc/ShaderDataAux.isph>

ISPC_UTIL_EXPORT_ENUM_TO_HEADER(PrimvarMapType);
ISPC_UTIL_EXPORT_ENUM_TO_HEADER(PrimvarAttributeType);
ISPC_UTIL_EXPORT_UNIFORM_STRUCT_TO_HEADER(UsdPrimvarReader);
ISPC_UTIL_EXPORT_ENUM_TO_HEADER(PRIMVAR_Input_Source_Mode);

bool
PRIMVAR_getPosition(uniform ShadingTLState* uniform tls,
                    const varying State& state,
                    const uniform int inputSourceMode,
                    const varying Vec3f& inputPosition,
                    const uniform Xform * uniform xform,
                    const uniform SHADING_Space returnSpace,
                    const uniform int refPKey,
                    Vec3f& outputPosition,
                    Vec3f& outputPositionDdx,
                    Vec3f& outputPositionDdy,
                    Vec3f& outputPositionDdz)
{
    outputPositionDdz = Vec3f_ctor(0.f, 0.f, 0.f);
    uniform SHADING_Space inputSpace;
    if(inputSourceMode == INPUT_SOURCE_MODE_ATTR) {
        outputPosition = inputPosition;
        outputPositionDdx = Vec3f_ctor(0.f, 0.f, 0.f);
        outputPositionDdy = Vec3f_ctor(0.f, 0.f, 0.f);
        inputSpace = SHADING_SPACE_WORLD;
    } else if(inputSourceMode == INPUT_SOURCE_MODE_REF_P_REF_N) {
        if(!SHADER_DATA_getRefP(refPKey, tls, state, &outputPosition)) {
            return false;
        }
        outputPositionDdx = getdVec3fAttributedx(tls, state, refPKey);
        outputPositionDdy = getdVec3fAttributedy(tls, state, refPKey);
        inputSpace = SHADING_SPACE_WORLD;
    } else {
        outputPosition = getP(state);
        outputPositionDdx = getdPdx(state);
        outputPositionDdy = getdPdy(state);
        inputSpace = SHADING_SPACE_RENDER;
    }

    if (xform != nullptr) {
        Vec3f transformedP = transformPoint(xform,
                                            inputSpace,
                                            returnSpace,
                                            state,
                                            outputPosition);

        // transform the partials as points to include perspective divide,
        // starting with untransformed position
        Vec3f outputPositionPlusDPdx = outputPosition + outputPositionDdx;
        Vec3f outputPositionPlusDPdy = outputPosition + outputPositionDdy;

        outputPositionPlusDPdx = transformPoint(xform,
                                                inputSpace,
                                                returnSpace,
                                                state,
                                                outputPositionPlusDPdx);

        outputPositionPlusDPdy = transformPoint(xform,
                                                inputSpace,
                                                returnSpace,
                                                state,
                                                outputPositionPlusDPdy);

        outputPosition = transformedP;
        outputPositionDdx = outputPositionPlusDPdx - transformedP;
        outputPositionDdy = outputPositionPlusDPdy - transformedP;
    }

    return true;
}

bool
PRIMVAR_getNormal(uniform ShadingTLState* uniform tls,
                  const varying State& state,
                  const uniform int inputSourceMode,
                  const varying Vec3f& inputNormal,
                  const uniform Xform * uniform xform,
                  const uniform SHADING_Space returnSpace,
                  const uniform int refPKey,
                  const uniform int refNKey,
                  Vec3f& outputNormal)
{
    uniform SHADING_Space inputSpace;
    if(inputSourceMode == INPUT_SOURCE_MODE_ATTR) {
        outputNormal = inputNormal;
        inputSpace = SHADING_SPACE_WORLD;
    } else if(inputSourceMode == INPUT_SOURCE_MODE_REF_P_REF_N) {
        if (!SHADER_DATA_getRefN(refPKey, refNKey, tls, state, &outputNormal)) {
            return false;
        }
        inputSpace = SHADING_SPACE_WORLD;
    } else {
        outputNormal = getN(state);
        inputSpace = SHADING_SPACE_RENDER;
    }

    if (xform != nullptr) {
        outputNormal = transformNormal(xform,
                                       inputSpace,
                                       returnSpace,
                                       state,
                                       outputNormal);
    }

    if (inputSourceMode == INPUT_SOURCE_MODE_REF_P_REF_N && isEntering(state) == false) {
        // Flip reference space normals on exiting a surface
        // (shading normals are already inverted)
        outputNormal = Vec3f_ctor(-1.0f * outputNormal.x,
                                  -1.0f * outputNormal.y,
                                  -1.0f * outputNormal.z);
    }

    outputNormal = normalize(outputNormal);

    return true;
}

