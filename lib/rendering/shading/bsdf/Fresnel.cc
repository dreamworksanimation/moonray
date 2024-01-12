// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

///
///
#include "Fresnel.h"
#include <scene_rdl2/render/util/Arena.h>

namespace moonray {
namespace shading {

Fresnel *
createFresnel(scene_rdl2::alloc::Arena* arena,
              const Fresnelv *fresnelv,
              int lane)
{
    Fresnel *fresnel = nullptr;

    if (fresnelv && ((1 << lane) & fresnelv->mMask)) {

        switch (fresnelv->mType) {

        case ispc::FRESNEL_TYPE_NONE:
            break;

        case ispc::FRESNEL_TYPE_SCHLICK_FRESNEL:
            fresnel = arena->allocWithArgs<SchlickFresnel>(arena, (const SchlickFresnelv *)fresnelv, lane);
            break;

        case ispc::FRESNEL_TYPE_ONE_MINUS_ROUGH_SCHLICK_FRESNEL:
            fresnel = arena->allocWithArgs<OneMinusRoughSchlickFresnel>(arena, (const OneMinusRoughSchlickFresnelv *)fresnelv, lane);
            break;

        case ispc::FRESNEL_TYPE_ONE_MINUS_FRESNEL:
            fresnel = arena->allocWithArgs<OneMinusFresnel>(arena, (const OneMinusFresnelv *)fresnelv, lane);
            break;

        case ispc::FRESNEL_TYPE_ONE_MINUS_VELVET_FRESNEL:
            fresnel = arena->allocWithArgs<OneMinusVelvetFresnel>(arena, (const OneMinusVelvetFresnelv *)fresnelv, lane);
            break;

        case ispc::FRESNEL_TYPE_ONE_MINUS_ROUGH_FRESNEL:
            fresnel = arena->allocWithArgs<OneMinusRoughFresnel>(arena, (const OneMinusRoughFresnelv *)fresnelv, lane);
            break;

        case ispc::FRESNEL_TYPE_CONDUCTOR_FRESNEL:
            fresnel = arena->allocWithArgs<ConductorFresnel>(arena, (const ConductorFresnelv *)fresnelv, lane);
            break;

        case ispc::FRESNEL_TYPE_DIELECTRIC_FRESNEL:
            fresnel = arena->allocWithArgs<DielectricFresnel>(arena, (const DielectricFresnelv *)fresnelv, lane);
            break;

        case ispc::FRESNEL_TYPE_MULTIPLE_TRANSMISSION_FRESNEL:
            fresnel = arena->allocWithArgs<MultipleTransmissionFresnel>(arena, (const MultipleTransmissionFresnelv *)fresnelv, lane);
            break;

        default:
            MNRY_ASSERT(0);
        }
    }
    return fresnel;
}

} // namespace shading
} // namespace moonray

