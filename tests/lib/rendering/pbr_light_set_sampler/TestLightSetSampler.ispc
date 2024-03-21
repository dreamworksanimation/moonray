// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

//
//

#include "TestUtil.isph"
#include <moonray/rendering/pbr/integrator/LightSetSampler.isph>

#include <moonray/rendering/shading/ispc/bsdf/Bsdf.isph>

//----------------------------------------------------------------------------

export void
testLightSetRadiance(
        uniform Arena * uniform arena,
        const uniform LightSet * uniform lightSet,
        const uniform Vec3f &p, const uniform Vec3f &n,
        const uniform Range2d &range,
        uniform float scl, uniform float ofs,
        uniform bool normalCulling,
        uniform Color &result)
{
    Vec3f pv = p;
    Vec3f nv = n;

    varying Vec3f cullingNormal = nv;
    if (!normalCulling) {
        setInvalidCullingNormal(&cullingNormal);
    }
    // Create the LightSetSampler
    uniform uint32_t sampleCount = (range.mRowEnd - range.mRowBegin)
                                 * (range.mColEnd - range.mColBegin);
    LightSetSampler lSampler;
    Bsdf bsdf;
    LightSetSampler_init(&lSampler, arena, lightSet, &bsdf, pv, sampleCount);

    double red = 0.0;
    double green = 0.0;
    double blue = 0.0;

    uint32_t lightCount = LightSetSampler_getLightCount(&lSampler);
    for (uniform uint32_t l = 0; l < lightCount; l++) {

        const uniform Light * uniform light = LightSetSampler_getLight(&lSampler, l);
        const uniform LightFilterList * uniform lightFilterList = LightSetSampler_getLightFilterList(&lSampler, l);

        // TODO / FIXME: this assumes a uniform sample count per light, an
        // assumption that will be broken when the LightSetSampler uses
        // an adaptive number of samples based on pv. See LightSetSampler_init()
        // for details.
        for (uniform uint32_t y=range.mRowBegin; y < range.mRowEnd; y++) {
            foreach (x = range.mColBegin ... range.mColEnd) {
                varying Vec3f r;
                r.x = (float)y * scl + ofs;
                r.y = (float)x * scl + ofs;
                r.z = 0.0f;

                LightSample lsmp;
                LightFilterRandomValues filterR = {Vec2f_ctor(0.f, 0.f), Vec3f_ctor(0.f, 0.f, 0.f)};
                LightSetSampler_sampleIntersectAndEval(&lSampler, nullptr, light, lightFilterList, pv, cullingNormal, 
                        filterR, 0.f, r, lsmp, 0.0f);
                if (LightSample_isValid(&lsmp)) {

                    // Compute reflected radiance from white lambertian brdf,
                    // accounting for the cosine term in the rendering eqn
                    float cosTheta = dot(lsmp.wi, nv);
                    Color radiance = lsmp.Li * rcp(lsmp.pdf) * sOneOverPi
                            * clamp(cosTheta, 0.0f, 1.0f);

                    // Accumulate
                    red += (double)radiance.r;
                    green += (double)radiance.g;
                    blue += (double)radiance.b;
                }
            }
        }
    }

    // TODO: add a reduce_add/min/max/etc() functions for most types
    result = Color_ctor(
            (uniform float)reduce_add(red),
            (uniform float)reduce_add(green),
            (uniform float)reduce_add(blue));
}


//----------------------------------------------------------------------------

