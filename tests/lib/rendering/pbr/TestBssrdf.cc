// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

///
/// @file TestBssrdf.cc
/// $Id$
///


#include "TestBssrdf.h"
#include "TestUtil.h"
#include "TestBssrdf_ispc_stubs.h"

#include <moonray/rendering/pbr/sampler/IntegratorSample.h>
#include <moonray/rendering/shading/bssrdf/Dipole.h>
#include <scene_rdl2/common/math/MathUtil.h>
#include <scene_rdl2/render/util/Random.h>


namespace moonray {
namespace pbr {


using namespace scene_rdl2::math;

//----------------------------------------------------------------------------

static void
generateSamples(const int sampleCount, FloatArray &u, FloatArray &v)
{
    IntegratorSample2D sampler(SequenceID{}, sampleCount, 0);
    u.resize(sampleCount);
    v.resize(sampleCount);

    float s[2];
    for (int i = 0; i < sampleCount; ++i) {
        constexpr unsigned depth = 0;
        sampler.getSample(s, depth);
        u.at(i) = s[0];
        v.at(i) = s[1];
    }
}

//----------------------------------------------------------------------------

static void
printDistribution(const FloatArray &u, const FloatArray &v,
                  const FloatArray &x, const FloatArray &red,
                  const FloatArray &grn, const FloatArray &blu,
                  const std::string &prefix)
{
    // NOTE: if saving to "mat/...." you'll likely need to manually
    // mkdir the mat/ directory in your build area in order to generate
    // the distribution files.  save2dTxtFile does not create directories.
    std::string filename;

    filename = prefix;
    filename += ".txt";
    save2dTxtFile(filename, u, v);
    
    filename = prefix;
    filename += "-red";
    filename += ".txt";
    save2dTxtFile(filename, x, red);

    filename = prefix;
    filename += "-grn";
    filename += ".txt";
    save2dTxtFile(filename, x, grn);

    filename = prefix;
    filename += "-blu";
    filename += ".txt";
    save2dTxtFile(filename, x, blu);
}

void
TestBssrdf::testDistribution(const shading::Bssrdfv &bssrdfv, int sampleCount, char *suffix)
{
    FloatArray u, v;
    generateSamples(sampleCount, u, v);

    FloatArray x, red, grn, blu;
    x.resize(sampleCount);
    red.resize(sampleCount);
    grn.resize(sampleCount);
    blu.resize(sampleCount);

    ispc::TestBssrdf_distribution(&bssrdfv, sampleCount, &u[0], &v[0], &x[0],
                                  &red[0], &grn[0], &blu[0]);

    std::string prefix = std::string("mat/testBssrdfvDistribution-") + suffix;
    printDistribution(u, v, x, red, grn, blu, prefix);
}

void
TestBssrdf::testDistribution(const shading::Bssrdf &bssrdf, int sampleCount, char *suffix)
{
    FloatArray u, v;
    generateSamples(sampleCount, u, v);

    FloatArray x, red, grn, blu;
    x.resize(sampleCount);
    red.resize(sampleCount);
    grn.resize(sampleCount);
    blu.resize(sampleCount);

    for (int i=0; i < sampleCount; i++) {
        Vec3f xi;
        float r;
        /* float pdf = */ bssrdf.sampleLocal(u[i], v[i], xi, r);
        u[i] = xi[0];
        v[i] = xi[1];

        Color rd = bssrdf.eval(r);
        x[i] = r;
        red[i] = rd[0];
        grn[i] = rd[1];
        blu[i] = rd[2];
    }

    std::string prefix = std::string("mat/testBssrdfDistribution-") + suffix;
    printDistribution(u, v, x, red, grn, blu, prefix);
}

//----------------------------------------------------------------------------

static void
printIntegralUniform(const Color &integralUniform)
{
    std::ostringstream integralUniformStr;
    integralUniformStr << integralUniform;

    printInfo("pdf uni = %s", integralUniformStr.str().c_str());
}

void
TestBssrdf::testPdfIntegral(const shading::Bssrdfv &bssrdfv, int sampleCount)
{
    sampleCount *= 100;
    FloatArray u, v;
    generateSamples(sampleCount, u, v);

    ispc::Col3f result;
    ispc::TestBssrdf_pdfIntegral(&bssrdfv, sampleCount, &u[0], &v[0], &result);
    printIntegralUniform(Color(result.r, result.g, result.b));
}

void
TestBssrdf::testPdfIntegral(const shading::Bssrdf &bssrdf, int sampleCount)
{
    sampleCount *= 100;
    FloatArray u, v;
    generateSamples(sampleCount, u, v);

    float maxRadius = bssrdf.getMaxRadius();

    Color integralUniform = sBlack;
    float pdf = 1.0f / (4.0f * maxRadius * maxRadius);
    for (int i=0; i < sampleCount; i++) {
        Vec3f xi;
        xi[0] = (u[i] - 0.5f) * 2.0f * maxRadius;
        xi[1] = (v[i] - 0.5f) * 2.0f * maxRadius;
        xi[2] = 0.0f;

        float r = xi.length();

        integralUniform += Color(bssrdf.pdfLocal(r));
    }
    integralUniform /= sampleCount * pdf;

    printIntegralUniform(integralUniform);
}

//----------------------------------------------------------------------------

static void
printIntegralResults(const Color &integralImp, const Color &integralUniform,
                     const Color &maxRadiusReflectance, float avgSampledRadius,
                     float maxSampledRadius, float maxRadius)
{
    std::ostringstream integralImpStr;
    integralImpStr << integralImp;

    std::ostringstream integralUniformStr;
    integralUniformStr << integralUniform;

    std::ostringstream diffuseReflectanceStr;
    diffuseReflectanceStr << maxRadiusReflectance;

    printInfo("reflectance imp = %s", integralImpStr.str().c_str());
    printInfo("reflectance uni = %s", integralUniformStr.str().c_str());
    printInfo("avg sampled radius = %f", avgSampledRadius);
    printInfo("max sampled radius = %f", maxSampledRadius);
    printInfo("max radius = %f", maxRadius);
    printInfo("reflectance(max radius) = %s", diffuseReflectanceStr.str().c_str());
}

void
TestBssrdf::testIntegral(const shading::Bssrdfv &bssrdfv, int sampleCount)
{
    FloatArray impU, impV;
    FloatArray uniU, uniV;
    generateSamples(sampleCount, impU, impV);
    generateSamples(10 * sampleCount, uniU, uniV);

    ispc::Col3f integralImp;
    ispc::Col3f integralUni;
    ispc::Col3f maxRadiusReflectance;
    float avgSampledRadius;
    float maxSampledRadius;
    float maxRadius;
    ispc::TestBssrdf_integral(&bssrdfv, sampleCount, 10 * sampleCount,
                              &impU[0], &impV[0], &uniU[0], &uniV[0],
                              &integralImp, &integralUni, &maxRadiusReflectance,
                              &avgSampledRadius, &maxSampledRadius, &maxRadius);

    printIntegralResults(Color(integralImp.r, integralImp.g, integralImp.b),
                         Color(integralUni.r, integralUni.g, integralUni.b),
                         Color(maxRadiusReflectance.r, maxRadiusReflectance.g, maxRadiusReflectance.b),
                         avgSampledRadius, maxSampledRadius, maxRadius);
}

void
TestBssrdf::testIntegral(const shading::Bssrdf &bssrdf, int sampleCount)
{
    FloatArray u, v;
    generateSamples(sampleCount, u, v);

    // Compute diffuse reflectance using importance sampling
    float avgSampledRadius = 0.0f;
    float maxSampledRadius = 0.0f;
    Color integralImp = sBlack;
    for (int i=0; i < sampleCount; i++) {
        Vec3f xi;
        float r;
        float pdf = bssrdf.sampleLocal(u[i], v[i], xi, r);

        avgSampledRadius += r;
        integralImp += bssrdf.eval(r) / pdf;
        maxSampledRadius = max(r, maxSampledRadius);
    }
    avgSampledRadius /= sampleCount;
    integralImp /= sampleCount;


    // Compute diffuse reflectance using uniform sampling
    sampleCount *= 10;
    generateSamples(sampleCount, u, v);

    float maxRadius = bssrdf.getMaxRadius();

    Color integralUniform = sBlack;
    float pdf = 1.0f / (4.0f * maxRadius * maxRadius);
    for (int i=0; i < sampleCount; i++) {
        Vec3f xi;
        xi[0] = (u[i] - 0.5f) * 2.0f * maxRadius;
        xi[1] = (v[i] - 0.5f) * 2.0f * maxRadius;
        xi[2] = 0.0f;

        float r = xi.length();

        integralUniform += bssrdf.eval(r);
    }
    integralUniform /= sampleCount * pdf;

    printIntegralResults(integralImp, integralUniform, bssrdf.eval(maxRadius),
                         avgSampledRadius, maxSampledRadius, maxRadius);

}


void
TestBssrdf::test(const shading::Bssrdf &bssrdf, int sampleCount, char *name)
{
    printInfo("----- %s -----", name);

    testDistribution(bssrdf, sampleCount, name);
    testPdfIntegral(bssrdf, 100.0f * sampleCount);
    testIntegral(bssrdf, 100.0f * sampleCount);
}

void
TestBssrdf::test(const shading::Bssrdfv &bssrdfv, int sampleCount, char *name)
{
    printInfo("----- %s v -----", name);

    testDistribution(bssrdfv, sampleCount, name);
    testPdfIntegral(bssrdfv, 100.0f * sampleCount);
    testIntegral(bssrdfv, 100.0f * sampleCount);
}

void
TestBssrdf::test(const PParams &p, int sampleCount, char *name)
{
    shading::DipoleBssrdf bssrdf(p.n, p.eta, p.sigmaA, p.sigmaSP, p.sceneScale,
                                 nullptr, nullptr);
    test(bssrdf, sampleCount, name);
    shading::DipoleBssrdfv dipoleBssrdfv;
    ispc::TestBssrdf_physicalInit(&dipoleBssrdfv,
                                  (const ispc::Vec3f *) &p.n,
                                  p.eta,
                                  (const ispc::Col3f *) &p.sigmaA,
                                  (const ispc::Col3f *) &p.sigmaSP,
                                  p.sceneScale);
    shading::Bssrdfv &bssrdfv = reinterpret_cast<shading::Bssrdfv &>(dipoleBssrdfv);
    test(bssrdfv, sampleCount, name);
}

void
TestBssrdf::test(const AParams &p, int sampleCount, char *name)
{
    shading::DipoleBssrdf bssrdf(p.n, p.eta, p.translucentColor, p.radius,
                                 nullptr, nullptr);
    test(bssrdf, sampleCount, name);
    shading::DipoleBssrdfv dipoleBssrdfv;
    ispc::TestBssrdf_artistInit(&dipoleBssrdfv,
                                (const ispc::Vec3f *) &p.n,
                                p.eta,
                                (const ispc::Col3f *) &p.translucentColor,
                                (const ispc::Col3f *) &p.radius);
    shading::Bssrdfv &bssrdfv = reinterpret_cast<shading::Bssrdfv &>(dipoleBssrdfv);
    test(bssrdfv, sampleCount, name);
}

//----------------------------------------------------------------------------

static const int sSampleCount = 100;

void
TestBssrdf::testDipole()
{
    ReferenceFrame frame;

    printInfo("##### testDipole() ####################");

    // Test ctor with physical paramters
    PParams a { Vec3f(0.0f, 0.0f, 1.0f), 1.3f,
            Color(0.032f, 0.17f, 0.48f), Color(0.74f, 0.88f, 1.01f), 0.001f };
    test(a, sSampleCount, const_cast<char *>("skin1"));

    PParams b { Vec3f(0.0f, 0.0f, 1.0f), 1.5f,
            Color(0.0021f, 0.0041f, 0.0071f), Color(2.19f, 2.62f, 3.00f), 0.001f };
    test(b, sSampleCount, const_cast<char *>("marble"));

    PParams c { Vec3f(0.0f, 0.0f, 1.0f), 1.3f,
            Color(0.0002f, 0.0028f, 0.0163f), Color(7.38, 5.47f, 3.15f), 0.001f };
    test(c, sSampleCount, const_cast<char *>("cream"));

    // Test ctor with artist-friendly parameters
    AParams d { Vec3f(0.0f, 0.0f, 1.0f), 1.3f, 1.0f, Color(0.5f, 0.0f, 0.0f),
            5.0f * Color(1.0f, 1.0f, 1.0f) };
    test(d, sSampleCount, const_cast<char *>("test1"));

    AParams e { Vec3f(0.0f, 0.0f, 1.0f), 1.3f, 1.0f, Color(1.0f, 1.0f, 1.0f),
            5.0f * Color(1.0f, 1.0f, 1.0f) };
    test(e, sSampleCount, const_cast<char *>("test2"));

    AParams f { Vec3f(0.0f, 0.0f, 1.0f), 1.3f, 1.0f, Color(0.44f, 0.22f, 0.13f),
            5.0f * Color(1.0f, 0.213f, 0.141f) };
    test(f, sSampleCount, const_cast<char *>("test3"));

    AParams g { Vec3f(0.0f, 0.0f, 1.0f), 1.3f, 2.0f, Color(0.44f, 0.22f, 0.13f),
            5.0f * Color(1.0f, 0.213f, 0.141f) };
    test(g, sSampleCount, const_cast<char *>("test4"));

    AParams h { Vec3f(0.0f, 0.0f, 1.0f), 1.3f, 1.0f, Color(0.7, 0.389, 0.319),
            3.0f * Color(0.7, 0.16, 0.079) };
    test(h, sSampleCount, const_cast<char *>("test5"));
}


//----------------------------------------------------------------------------

} // namespace pbr
} // namespace moonray


CPPUNIT_TEST_SUITE_REGISTRATION(moonray::pbr::TestBssrdf);


