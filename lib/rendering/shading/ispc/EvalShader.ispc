// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

/// @file EvalShader.ispc

#include "EvalShader.isph"

#include <moonray/rendering/shading/ispc/bsdf/Bsdf.isph>
#include <moonray/rendering/shading/ispc/Closure.isph>

#include <moonray/rendering/bvh/shading/ispc/State.isph>
#include <scene_rdl2/scene/rdl2/rdl2.isph>

typedef void (*ShadePtr)(const uniform Material *          uniform material,
                               uniform ShadingTLState *    uniform tls,
                               uniform int                         numState,
                         const varying State * const       uniform state,
                               varying BsdfBuilder *       uniform bsdfBuilder);

extern "C" void
CPP_setBsdfLabels(const uniform Material * uniform material,
                  uniform ShadingTLState * uniform tls,
                  const uniform int nstate,
                  const varying State * const uniform state,
                  varying Bsdf * uniform bsdf,
                  const uniform int parentLobeCount);

void
shadeMaterial(const uniform Material * uniform material,
                    uniform ShadingTLState * uniform tls,
              const uniform int nstate,
              const varying State * const uniform state,
                    varying BsdfBuilder * uniform bsdfBuilder)
{
    int64 * uniform matObjPtr = (int64 * uniform) material;
    int64 * uniform ptr = (int64 * uniform) *(matObjPtr + (MATERIAL_SHADEFUNCV_OFFSET / 8));
    assert(ptr != NULL);

    varying Bsdf * const uniform bsdf = asBsdf(BsdfBuilder_getClosure(*bsdfBuilder));
    const uniform int parentLobeCount = Bsdf_getLobeCount(bsdf);

    ShadePtr shadeFn = (ShadePtr) ptr;
    shadeFn(material, tls, nstate, state, bsdfBuilder);


    // hack in the bsdf label ids
    CPP_setBsdfLabels(material,
                      tls,
                      nstate,
                      state,
                      bsdf,
                      parentLobeCount);
}

