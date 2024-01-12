// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

/// @file TestBssrdf.ispc

#include <moonray/rendering/shading/ispc/bssrdf/Dipole.isph>
#include <moonray/rendering/shading/ispc/bssrdf/Bssrdf.isph>
#include <scene_rdl2/common/math/ispc/Color.isph>
#include <scene_rdl2/common/math/ispc/Vec3.isph>
#include <scene_rdl2/common/platform/Platform.isph>

export void
#pragma ignore warning(all)
TestBssrdf_physicalInit(varying DipoleBssrdf * uniform bssrdf,
                        const uniform Vec3f * uniform inN,
                        const uniform float inEta,
                        const uniform Color * uniform inSigmaA,
                        const uniform Color * uniform inSigmaSP,
                        const uniform float inSceneScale)
{
    Vec3f N = *inN;
    float eta = inEta;
    Color sigmaA = *inSigmaA;
    Color sigmaSP = *inSigmaSP;
    float sceneScale = inSceneScale;
    DipoleBssrdf_init(bssrdf, N, eta, sigmaA, sigmaSP, sceneScale, 
                      nullptr, 0);
}

export void
#pragma ignore warning(all)
TestBssrdf_artistInit(varying DipoleBssrdf * uniform bssrdf,
                      const uniform Vec3f * uniform inN,
                      const uniform float inEta,
                      const uniform Color * uniform inTranslucentColor,
                      const uniform Color * uniform inRadius)
{
    Vec3f N = *inN;
    float eta = inEta;
    Color translucentColor = *inTranslucentColor;
    Color radius = *inRadius;
    DipoleBssrdf_init(bssrdf, N, translucentColor, radius,
                      nullptr, 0);
}

export void
#pragma ignore warning(all)
TestBssrdf_distribution(const varying Bssrdf * uniform bssrdf,
                        const uniform int sampleCount,
                        uniform float * uniform u,
                        uniform float * uniform v,
                        uniform float * uniform x,
                        uniform float * uniform red,
                        uniform float * uniform grn,
                        uniform float * uniform blu)

{
    foreach (i = 0 ... sampleCount) {
        Vec3f xi;
        float r;
        Bssrdf_sampleLocal(bssrdf, u[i], v[i], xi, r);
        u[i] = xi.x;
        v[i] = xi.y;

        Color rd = Bssrdf_eval(bssrdf, r);
        x[i] = r;
        red[i] = rd.r;
        grn[i] = rd.g;
        blu[i] = rd.b;
    }
}

export void
#pragma ignore warning(all)
TestBssrdf_pdfIntegral(const varying Bssrdf * uniform bssrdf,
                       const uniform int sampleCount,
                       const uniform float * uniform u,
                       const uniform float * uniform v,
                       uniform Color * uniform result)
{
    float maxRadius = Bssrdf_getMaxRadius(bssrdf);
    Color integralUniform = Color_ctor(0.f);
    float pdf = rcp(4.f * maxRadius * maxRadius);
    foreach (i = 0 ... sampleCount) {
        Vec3f xi;
        xi.x = (u[i] - 0.5f) * 2.0f * maxRadius;
        xi.y = (v[i] - 0.5f) * 2.0f * maxRadius;
        xi.z = 0.0f;

        float r = length(xi);

        integralUniform = integralUniform +
            Color_ctor(Bssrdf_pdfLocal(bssrdf, r));
    }

    // now things get weird.  since we are to return a single uniform
    // Color value to represent the integral, we must assume that
    // all lanes of the bssrdf are active and equal.  the test
    // should have been set up that way
    result->r = reduce_add(integralUniform.r);
    result->g = reduce_add(integralUniform.g);
    result->b = reduce_add(integralUniform.b);
    // given these same assumptions, pdf should be equal on all lanes
    MNRY_ASSERT(reduce_equal(pdf));
    *result = *result * rcp(sampleCount * extract(pdf, 0));
}

export void
#pragma ignore warning(all)
TestBssrdf_integral(const varying Bssrdf * uniform bssrdf,
                    const uniform int impSampleCount,
                    const uniform int uniSampleCount,
                    const uniform float * uniform impU,
                    const uniform float * uniform impV,
                    const uniform float * uniform uniU,
                    const uniform float * uniform uniV,
                    uniform Color * uniform outImpIntegral,
                    uniform Color * uniform outUniIntegral,
                    uniform Color * uniform outMaxRadiusReflectance,
                    uniform float * uniform outAvgSampledRadius,
                    uniform float * uniform outMaxSampledRadius,
                    uniform float * uniform outMaxRadius)
{
    // Compute diffuse reflectance using importance sampling
    float avgSampledRadius = 0.0f;
    float maxSampledRadius = 0.0f;
    Color integralImp = Color_ctor(0.f);
    foreach (i = 0 ... impSampleCount) {
        Vec3f xi;
        float r;
        float pdf = Bssrdf_sampleLocal(bssrdf, impU[i], impV[i], xi, r);

        avgSampledRadius += r;
        integralImp = integralImp + Bssrdf_eval(bssrdf, r) * rcp(pdf);
        maxSampledRadius = max(r, maxSampledRadius);
    }
    outImpIntegral->r = reduce_add(integralImp.r) / impSampleCount;
    outImpIntegral->g = reduce_add(integralImp.g) / impSampleCount;
    outImpIntegral->b = reduce_add(integralImp.b) / impSampleCount;
    *outAvgSampledRadius = reduce_add(avgSampledRadius) / impSampleCount;
    *outMaxSampledRadius = reduce_max(maxSampledRadius);

    // Compute diffuse reflectance using uniform sampling
    float maxRadius = Bssrdf_getMaxRadius(bssrdf);
    Color integralUniform = Color_ctor(0.f);
    float pdf =  rcp(4.0f * maxRadius * maxRadius);
    foreach (i = 0 ... uniSampleCount) {
        Vec3f xi;
        xi.x = (uniU[i] - 0.5f) * 2.0f * maxRadius;
        xi.y = (uniV[i] - 0.5f) * 2.0f * maxRadius;
        xi.z = 0.0f;

        float r = length(xi);

        integralUniform = integralUniform + Bssrdf_eval(bssrdf, r);
    }
    outUniIntegral->r = reduce_add(integralUniform.r);
    outUniIntegral->g = reduce_add(integralUniform.g);
    outUniIntegral->b = reduce_add(integralUniform.b);
    MNRY_ASSERT(reduce_equal(pdf));
    *outUniIntegral = *outUniIntegral * rcp(uniSampleCount * extract(pdf, 0));
    MNRY_ASSERT(reduce_equal(maxRadius));
    *outMaxRadius = extract(maxRadius, 0);

    Color maxRadiusReflectance = Bssrdf_eval(bssrdf, maxRadius);
    outMaxRadiusReflectance->r = extract(maxRadiusReflectance.r, 0);
    outMaxRadiusReflectance->g = extract(maxRadiusReflectance.g, 0);
    outMaxRadiusReflectance->b = extract(maxRadiusReflectance.b, 0);
}

