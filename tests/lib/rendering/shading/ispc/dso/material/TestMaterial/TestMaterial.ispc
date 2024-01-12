// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

/// @file TestMaterial.ispc

// this should be <rendering/shading/Shading.isph>
// for anything other than the unittest
#include <moonray/rendering/shading/ispc/Shading.isph>
#include <moonray/rendering/shading/ispc/Closure.isph>
#include "attributes.isph"

static void
shade(const uniform Material *      uniform  me,
            uniform ShadingTLState *uniform  tls,
      const varying State                    &state,
            varying BsdfBuilder              &bsdfBuilder)
{
    varying Closure * uniform closure = BsdfBuilder_getClosure(bsdfBuilder);

    // evaluate color
    Color c1 = evalAttrColorNonComp(me, tls, state);
    Color c2 = evalCompColor(me, tls, state);

    // non-component color as the scale on the first lobe
    Closure_addLambertBsdfLobe(closure, tls, state,
                               /* scale = */ c1,
                               /* fresnel = */ NULL,
                               /* underFresnel = */ NULL,
                               /* normal = */ state.mN,
                               /* albedo = */ (const varying Color)sWhite,
                               /* isReflection = */ true,
                               /* labels = */ 0);


    // component color as the scale on the second lobe
    Closure_addCookTorranceBsdfLobe(closure, tls, state,
                                    /* scale = */ c2,
                                    /* fresnel = */ NULL,
                                    /* underFresnel = */ NULL,
                                    /* normal = */ state.mN,
                                    /* roughness = */ 1.f,
                                    /* favg */ sWhite,
                                    /* favgInv */ sWhite,
                                    /* labels = */ 0);
}

DEFINE_MATERIAL_SHADER(TestMaterial, shade)

