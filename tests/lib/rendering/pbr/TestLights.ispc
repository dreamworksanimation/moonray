// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

//
//

#include "TestUtil.isph"
#include <moonray/rendering/pbr/light/Light.isph>
#include <moonray/rendering/pbr/core/Util.isph>

#include <moonray/rendering/shading/ispc/Util.isph>

//----------------------------------------------------------------------------

export void
testLightRadiance(const uniform Vec3f &p, const uniform Vec3f &n,
        const uniform Light * uniform light,
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

    double red = 0.0;
    double green = 0.0;
    double blue = 0.0;

    for (uniform uint32_t y=range.mRowBegin; y < range.mRowEnd; y++) {
        foreach (x = range.mColBegin ... range.mColEnd) {
            varying Vec3f r;
            r.x = (float)y * scl + ofs;
            r.y = (float)x * scl + ofs;
            r.z = 0.0f;

            Vec3f wi;
            LightIntersection isect;
            if (Light_sample(light, pv, cullingNormal, 0.f, r, wi, isect, 0.0f)) {

                float pdf;                
                LightFilterRandomValues filterR = {Vec2f_ctor(0.f, 0.f), Vec3f_ctor(0.f, 0.f, 0.f)};
                Color radiance = Light_eval(light, nullptr, wi, pv, filterR, 0.f, isect, false, nullptr, 0.0f, &pdf);

                if (pdf != 0.0f  &&  !isBlack(radiance)) {

                    // weight by pdf
                    // assume a white lambertian brdf on all surfaces
                    // account for the cosine term in the rendering eqn
                    float cosTheta = dot(wi, nv);
                    radiance = radiance *
                            (sOneOverPi * clamp(cosTheta, 0.0f, 1.0f) * rcp(pdf));

                    // accumulate
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

export void
testLightPdf(const uniform Vec3f &p, const uniform Vec3f &n,
        const uniform Light * uniform light,
        const uniform Range2d &range,
        uniform float scl, uniform float ofs,
        uniform bool isInfinite,
        uniform double &resultPdf, uniform uint32_t &resultValidSampleCount)
{
    Vec3f pv = p;
    Vec3f nv = n;

    double sum = 0.0;
    uint32_t validSampleCount = 0;

    varying Vec3f cullingNormal;
    setInvalidCullingNormal(&cullingNormal);

    for (uniform uint32_t y=range.mRowBegin; y < range.mRowEnd; y++) {
        foreach (x = range.mColBegin ... range.mColEnd) {
            varying Vec3f r;
            r.x = (float)y * scl + ofs;
            r.y = (float)x * scl + ofs;
            r.z = 0.0f;

            Vec3f wi;
            LightIntersection isect;
            if (Light_sample(light, pv, cullingNormal, 0.f, r, wi, isect, 0.0f)) {

                float pdf;
                LightFilterRandomValues filterR = {Vec2f_ctor(0.f, 0.f), Vec3f_ctor(0.f, 0.f, 0.f)};
                Light_eval(light, nullptr, wi, pv, filterR, 0.f, isect, false, nullptr, 0.0f, &pdf);

                if (pdf != 0.0f) {
                    MNRY_ASSERT(isnormal(pdf));

                    // Convert pdf back to area measure, undo-ing the same
                    // scaling that is done in eval (only for non-infinite lights)
                    float denom = 1.0f;
                    if (!isInfinite) {
                        denom = absAreaToSolidAngleScale(wi, isect.N, isect.distance);
                    }
                    sum += (double)(denom / pdf);
                    ++validSampleCount;
                }
            }
        }
    }

    resultPdf = reduce_add(sum);
    resultValidSampleCount = reduce_add(validSampleCount);
}


//----------------------------------------------------------------------------

export uniform bool
testLightCanIlluminate(const uniform Light * uniform light,
        const uniform Vec3f &lightPos, uniform int32_t seed,
        uniform uint32_t imin, uniform uint32_t imax)
{
    // Prep varying random number generator from given uniform seed
    uniform DWARNGState rng;
    seed_rng(&rng, seed);
    int32_t seedArray[programCount];
    for (uniform int32_t k=0; k < programCount; k++) {
        seedArray[k] = random(&rng);
    }

    int32_t seedv = seedArray[programIndex];
    DWARNGState rngv;
    seed_rng(&rngv, seedv, programIndex);


    // Loop over samples
    uniform bool success = true;
    foreach (i = imin ... imax) {

        // Sample p and n around the light source
        Vec3f p = Vec3f_ctor((frandom(&rngv) * 2.0f - 1.0f) * 40.0f,
                             (frandom(&rngv) * 2.0f - 1.0f) * 40.0f,
                             (frandom(&rngv) * 2.0f - 1.0f) * 40.0f) + lightPos;
        Vec3f n = sampleSphereUniform(frandom(&rngv), frandom(&rngv));

        // Compute if there is any contribution from the light
        bool gotContribution = false;
        static const uniform int32_t sampleCount = 1024;
        for (uniform uint32_t j=0; j < sampleCount; j++) {
            varying Vec3f r;
            r.x = frandom(&rngv);
            r.y = frandom(&rngv);
            r.z = frandom(&rngv);

            Vec3f wi;
            LightIntersection isect;
            if (Light_sample(light, p, n, 0.f, r, wi, isect, 0.0f)) {

                float pdf;
                LightFilterRandomValues filterR = {Vec2f_ctor(0.f, 0.f), Vec3f_ctor(0.f, 0.f, 0.f)};
                Color radiance = Light_eval(light, nullptr, wi, p, filterR, 0.f, isect, false, nullptr, 0.0f, &pdf);

                if (pdf != 0.0f  &&  !isBlack(radiance)) {
                    radiance = radiance * clamp(dot(wi, n), 0.0f, 1.0f) * rcp(pdf);
                    if (!isBlack(radiance)) {
                        gotContribution = true;
                    }
                }
            }
        }

        // If we have some light contribution, we'd better make sure thec
        // light doesn't cull itself from this situation
        if (gotContribution) {
            if (!Light_canIlluminate(light, p, n, 0.f, 0.f, nullptr)) {
                success = false;
            }
        }
    }

    return success;
}


//----------------------------------------------------------------------------

export uniform bool
testLightIntersection(const uniform Light * uniform light,
        uniform uint32_t isectDataFieldsUsed,
        const uniform Vec3f &p,
        const uniform Range2d &range,
        uniform float scl, uniform float ofs,
        uniform uint32_t &isectsEqual,
        uniform uint32_t &noIntersection,
        uniform uint32_t &invalidSamples)
{
    uniform bool success = true;

    Vec3f pv = p;

    uint32_t localIsectsEqual = 0;
    uint32_t localNoIntersection = 0;
    uint32_t localInvalidSamples = 0;

    varying Vec3f cullingNormal;
    setInvalidCullingNormal(&cullingNormal);

    for (uniform uint32_t y=range.mRowBegin; y < range.mRowEnd; y++) {
        foreach (x = range.mColBegin ... range.mColEnd) {
            varying Vec3f r;
            r.x = (float)y * scl + ofs;
            r.y = (float)x * scl + ofs;
            r.z = 0.0f;

            Vec3f wi;
            LightIntersection refIsect;
            if (Light_sample(light, pv, cullingNormal, 0.f, r, wi, refIsect, 0.0f)) {

                LightIntersection testIsect;
                if (Light_intersect(light, pv, cullingNormal, wi, 0.f, refIsect.distance + 1.0f, testIsect)) {

                    bool passed = true;
                    bool asserted = false;

                    // test within one degree
                    if (!isEqualDirection(testIsect.N, refIsect.N, 1.f)) {
                        passed = false;
                        // test within 5 degrees
                        if (!isEqualDirection(testIsect.N, refIsect.N, 5.f)) {
                            print("Error: normals differ:\n"
                                  "[%f,\n %f,\n %f] != \n[%f,\n %f,\n %f]\n",
                                   testIsect.N.x, testIsect.N.y, testIsect.N.z,
                                   refIsect.N.x, refIsect.N.y, refIsect.N.z);
                            asserted = true;
                        }
                    }

                    if (!asserted  &&  !isEqualWrappedUv(testIsect.uv, refIsect.uv, 0.04f)) {
                        passed = false;
                        if (!isEqualWrappedUv(testIsect.uv, refIsect.uv, 0.08f)) {
                            print("Error: uvs differ:\n"
                                  "[%f,\n %f] != \n[%f,\n %f]\n",
                                   testIsect.uv.x, testIsect.uv.y,
                                   refIsect.uv.x, refIsect.uv.y);
                            asserted = true;
                        }
                    }

                    if (!asserted  &&  !isEqual(testIsect.distance, refIsect.distance, 0.001f)) {
                        passed = false;
                        if (!isEqual(testIsect.distance, refIsect.distance, 0.01f)) {
                            print("Error: hit distances differ:\n"
                                  "[%f] != \n[%f]\n",
                                   testIsect.distance, refIsect.distance);
                            asserted = true;
                        }
                    }

                    for (uniform uint32_t j = 0; j < isectDataFieldsUsed; ++j) {
                        if (!isEqual(testIsect.data[j], refIsect.data[j], 0.04f)) {
                            passed = false;
                            if (!isEqual(testIsect.data[j], refIsect.data[j], 0.1f)) {
                                print("Error: data[%d] differs:\n"
                                      "[%f] != \n[%f]\n",
                                      j, testIsect.data[j], refIsect.data[j]);
                                asserted = true;
                            }
                        }
                    }

                    // Useful for debugging failures
                    if (asserted) {
#ifdef DEBUG
                        // Set your breakpoint here!
                        Light_sample(light, pv, cullingNormal, 0.f, r, wi, refIsect, 0.0f);
                        Light_intersect(light, pv, cullingNormal, wi, 0.f, refIsect.distance + 1.0f, testIsect);
#endif
                        success = false;
                    }

                    localIsectsEqual += passed ? 1 : 0;
                } else {
                    ++localNoIntersection;
                }
            } else {
                ++localInvalidSamples;
            }
        }
    }

    isectsEqual = reduce_add(localIsectsEqual);
    noIntersection = reduce_add(localNoIntersection);
    invalidSamples = reduce_add(localInvalidSamples);

    return success;
}


//----------------------------------------------------------------------------

