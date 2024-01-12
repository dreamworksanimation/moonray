// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

///
/// @file TestHair.cc
/// $Id$
///


#include "TestHair.h"
#include "TestHair_ispc_stubs.h"

#include <moonray/rendering/shading/bsdf/BsdfSlice.h>
#include <moonray/rendering/shading/bsdf/Fresnel.h>
#include <moonray/rendering/shading/bsdf/hair/BsdfHairLobes.h>
#include <moonray/rendering/shading/bsdf/hair/HairUtil.h>
#include <moonray/rendering/shading/bsdf/hair/HairState.h>
#include <scene_rdl2/common/math/Color.h>
#include <scene_rdl2/common/math/Math.h>
#include <scene_rdl2/render/util/stdmemory.h>

namespace moonray {
namespace shading {

using namespace scene_rdl2::math;
using namespace moonray::shading;

namespace {

static const float sTolerance = 0.0001;

bool isEqualColor(const Color &a, const ispc::Color &b, const float eps)
{
    const Color bb(asCpp(b));
    return isEqual(a, bb, eps);
}

bool isEqualVec3f(const Vec3f &a, const ispc::Vec3f &b, const float eps)
{
    const Vec3f bb(asCpp(b));
    return isEqual(a, bb, eps);
}

} // end anonymous namespace

TestHair::TestHair() :
    mNg(0.0f, 0.0f, 1.0f),
    mWi(normalize(Vec3f(-0.25f, -0.15f, 0.59f))),
    mWo(normalize(Vec3f(0.5f, 0.75f, 0.25f)))
{
    // setup slices
    mBsdfSliceCpp = fauxstd::make_unique<BsdfSlice>(
        mNg, mWo, true, true, ispc::SHADOW_TERMINATOR_FIX_OFF);

    mBsdfSliceIspc = deleted_unique_ptr<void>(
            ispc::createBsdfSlice(asIspc(mNg), asIspc(mWo), true, true),
            [](void * p) { ispc::deleteBsdfSlice(p); } );

    // setup HairState
    static const Vec3f hairDir = normalize(Vec3f(1.0f, 1.0f, 1.0f));
    static const Color hairColor(0.65f, 0.21f, 0.25f);
    static const float azimRoughness = 0.22f;
    static const Color hairSigmaA = HairUtil::computeAbsorptionCoefficients(hairColor,
                                                                            azimRoughness);
    static const Vec2f hairUv(0.33f, 0.66f);
    static const float H = hairUv[1];
    static const float eta = 1.45;
    static const float hairRotation = 1.0f;
    static const Vec3f hairNormal = Vec3f(1.0f, 0.0f, 0.0f);

    mHairStateCpp = fauxstd::make_unique<HairState>(mWo,
                                                    mWi,
                                                    hairDir,
                                                    H,
                                                    eta,
                                                    hairSigmaA,
                                                    hairRotation,
                                                    hairNormal);

    mHairStateIspc = deleted_unique_ptr<void>(
            ispc::createHairState(asIspc(mWo),
                                  asIspc(mWi),
                                  asIspc(hairDir),
                                  H,
                                  eta,
                                  asIspc(hairSigmaA),
                                  hairRotation,
                                  asIspc(hairNormal)),
            [](void * p) { ispc::deleteHairState(p); } );

    // setup Fresnel
    mFresnelCpp = fauxstd::make_unique<DielectricFresnel>(1.0f, eta);

    mFresnelIspc = deleted_unique_ptr<void>(
            ispc::createFresnel(1.0f, eta),
            [](void * p) { ispc::deleteFresnel(p); } );
}

TestHair::~TestHair()
{
}

void TestHair::testHairUtil()
{
    static const float f = 0.25f;
    static const Color c(0.25f, 0.55f, 0.75f);

    CPPUNIT_ASSERT(isEqual(Pow<2>(f),  ispc::testPow2(f),  sTolerance));
    CPPUNIT_ASSERT(isEqual(Pow<3>(f),  ispc::testPow3(f),  sTolerance));
    CPPUNIT_ASSERT(isEqual(Pow<4>(f),  ispc::testPow4(f),  sTolerance));
    CPPUNIT_ASSERT(isEqual(Pow<20>(f), ispc::testPow20(f), sTolerance));
    CPPUNIT_ASSERT(isEqual(Pow<22>(f), ispc::testPow22(f), sTolerance));

    CPPUNIT_ASSERT(isEqual(HairUtil::safeSqrt(f), ispc::testSafeSqrt(f), sTolerance));
    CPPUNIT_ASSERT(isEqual(scene_rdl2::math::sqr(f), ispc::testSqr(f), sTolerance));
    CPPUNIT_ASSERT(isEqual(scene_rdl2::math::sinh(f), ispc::testSinh(f), sTolerance));

    CPPUNIT_ASSERT(isEqual(HairUtil::azimuthalVar(f), ispc::testAzimuthalVar(f), sTolerance));
    CPPUNIT_ASSERT(isEqual(HairUtil::longitudinalVar(f), ispc::testLongitudinalVar(f), sTolerance));

    // test computeAbsorptionCoefficients()
    {
        ispc::Color ispcResult;
        ispc::testComputeAbsorptionCoefficients(asIspc(c), f, ispcResult);
        CPPUNIT_ASSERT(isEqualColor(HairUtil::computeAbsorptionCoefficients(c, f), ispcResult, sTolerance));
    }

    CPPUNIT_ASSERT(isEqual(HairUtil::besselIO(f), ispc::testBesselIO(f), sTolerance));
    CPPUNIT_ASSERT(isEqual(HairUtil::logBesselIO(f), ispc::testLogBesselIO(f), sTolerance));

    // test deonLongitudinalM() func
    {
        static const float variance = 0.0123f;
        static const float thetaI = scene_rdl2::math::sPi / 4.0f;
        static const float thetaO = scene_rdl2::math::sPi / 6.0f;
        static const float sinThetaI = scene_rdl2::math::sin(thetaI);
        static const float cosThetaI = scene_rdl2::math::cos(thetaI);
        static const float sinThetaO = scene_rdl2::math::sin(thetaO);
        static const float cosThetaO = scene_rdl2::math::cos(thetaO);

        CPPUNIT_ASSERT(isEqual(
                    HairUtil::deonLongitudinalM(variance, sinThetaI, cosThetaI, sinThetaO, cosThetaO),
                    ispc::testDeonLongitudinalM(variance, sinThetaI, cosThetaI, sinThetaO, cosThetaO),
                    sTolerance));
    }

    // test logistic functions
    {
        static const float x = 0.3425f;
        static const float s = 0.6345f;
        static const float a = 0.15f;
        static const float b = 0.85f;

        CPPUNIT_ASSERT(isEqual(HairUtil::logisticFunction(x, s), ispc::testLogisticFunction(x, s), sTolerance));
        CPPUNIT_ASSERT(isEqual(HairUtil::logisticCDF(x, s), ispc::testLogisticCDF(x, s), sTolerance));
        CPPUNIT_ASSERT(isEqual(HairUtil::trimmedLogisticFunction(x, s, a, b), ispc::testTrimmedLogisticFunction(x, s, a, b), sTolerance));
        CPPUNIT_ASSERT(isEqual(HairUtil::sampleTrimmedLogistic(x, s, a, b), ispc::testSampleTrimmedLogistic(x, s, a, b), sTolerance));
    }

    // test gaussian/cauchy functions
    {
        static const float stddev = 0.1425f;
        static const float x = 0.6345f;
        CPPUNIT_ASSERT(isEqual(HairUtil::unitGaussianForShade(stddev, x), ispc::testUnitGaussianForShade(stddev, x), sTolerance));
        CPPUNIT_ASSERT(isEqual(HairUtil::unitCauchyForShade(stddev, x), ispc::testUnitCauchyForShade(stddev, x), sTolerance));
        CPPUNIT_ASSERT(isEqual(HairUtil::unitCauchyForSample(stddev, x), ispc::testUnitCauchyForSample(stddev, x), sTolerance));
    }

    // test compact1By1()
    {
        static const uint32_t x = 54321;
        const uint32_t a = HairUtil::compact1By1(x);
        const uint32_t b = ispc::testCompact1By1(x);
        CPPUNIT_ASSERT( a == b );
    }

    // test demuxFloat()
    {
        const Vec2f a = HairUtil::demuxFloat(f);
        ispc::Vec2f b;
        ispc::testDemuxFloat(f, b);
        CPPUNIT_ASSERT(isEqual(a.x, asCpp(b).x, sTolerance));
        CPPUNIT_ASSERT(isEqual(a.y, asCpp(b).y, sTolerance));
    }
}

void TestHair::testLobe(HairBsdfLobe * lobeCpp, void * lobeIspc)
{
    // test members
    {
        CPPUNIT_ASSERT(isEqual(lobeCpp->mSinAlpha,
                               ispc::getSinAlpha(lobeIspc),
                               sTolerance));
        CPPUNIT_ASSERT(isEqual(lobeCpp->mCosAlpha,
                               ispc::getCosAlpha(lobeIspc),
                               sTolerance));
        CPPUNIT_ASSERT(isEqual(lobeCpp->mLongitudinalRoughness,
                               ispc::getLongitudinalRoughness(lobeIspc),
                               sTolerance));
        CPPUNIT_ASSERT(isEqual(lobeCpp->mLongitudinalVariance,
                               ispc::getLongitudinalVariance(lobeIspc),
                               sTolerance));
    }
    // test evalMTerm()
    {
        const float a = lobeCpp->evalMTerm(*mHairStateCpp);
        const float b = ispc::evalMTerm(lobeIspc,
                                        mHairStateIspc.get());
        CPPUNIT_ASSERT(isEqual(a, b, sTolerance));
    }
    // test evalNTermWithAbsorption()
    {
        const Color result_a = lobeCpp->evalNTermWithAbsorption(*mHairStateCpp);
        ispc::Color result_b;
        ispc::evalNTermWithAbsorption(lobeIspc,
                                      mHairStateIspc.get(),
                                      result_b);
        CPPUNIT_ASSERT(isEqualColor(result_a, result_b, sTolerance));
    }
    // test evalPhiPdf()
    {
        const float a = lobeCpp->evalPhiPdf(*mHairStateCpp);
        const float b = ispc::evalPhiPdf(lobeIspc,
                                         mHairStateIspc.get());
        CPPUNIT_ASSERT(isEqual(a, b, sTolerance));
    }
    // test evalThetaPdf()
    {
        // check some internals first
        const float thetaI_a = mHairStateCpp->thetaI();
        const float thetaI_b = ispc::getThetaI(mHairStateIspc.get());
        CPPUNIT_ASSERT(isEqual(thetaI_a, thetaI_b, sTolerance));
        const float thetaO_a = mHairStateCpp->thetaO();
        const float thetaO_b = ispc::getThetaO(mHairStateIspc.get());
        CPPUNIT_ASSERT(isEqual(thetaO_a, thetaO_b, sTolerance));

        const float a = lobeCpp->evalThetaPdf(*mHairStateCpp);
        const float b = ispc::evalThetaPdf(lobeIspc,
                                           mHairStateIspc.get());
        CPPUNIT_ASSERT(isEqual(a, b, sTolerance));
    }
    // test sampleTheta()
    {
        // choose a random
        const float r1 = 0.123f;

        const float thetaO = mHairStateCpp->thetaO();
        float thetaI_a, thetaI_b;
        float pdf_a, pdf_b;
        pdf_a = lobeCpp->sampleTheta(r1, thetaO, thetaI_a);
        pdf_b = ispc::sampleTheta(lobeIspc, r1, thetaO, thetaI_b);
        CPPUNIT_ASSERT(isEqual(pdf_a, pdf_b, sTolerance));
        CPPUNIT_ASSERT(isEqual(thetaI_a, thetaI_b, sTolerance));
    }
    // test samplePhi()
    {
        // choose a random
        const float r2 = 0.567;
        const float phiO = mHairStateCpp->phiO();

        float phiI_a, phiI_b;
        float pdf_a = lobeCpp->samplePhi(r2,
                                         phiO,
                                         phiI_a);
        float pdf_b = ispc::samplePhi(lobeIspc,
                                      r2,
                                      phiO,
                                      phiI_b);
        CPPUNIT_ASSERT(isEqual(pdf_a, pdf_b, sTolerance));
        CPPUNIT_ASSERT(isEqual(phiI_a, phiI_b, sTolerance));
    }
    // test evalPdf()
    {
        const float a = lobeCpp->evalPdf(*mHairStateCpp);
        const float b = ispc::evalPdf(lobeIspc,
                                      mHairStateIspc.get());
        CPPUNIT_ASSERT(isEqual(a, b, sTolerance));
    }
    // test eval()
    {
        float pdf_a, pdf_b;
        const Color result_a = lobeCpp->eval(*mBsdfSliceCpp, mWi, &pdf_a);
        ispc::Color result_b;
        ispc::evalLobe(lobeIspc, mBsdfSliceIspc.get(), asIspc(mWi), result_b, pdf_b);
        CPPUNIT_ASSERT(isEqual(pdf_a, pdf_b, sTolerance));
        CPPUNIT_ASSERT(isEqualColor(result_a, result_b, sTolerance));
    }
    // test sample()
    {
        // choose some randoms
        const float r1 = 0.123f;
        const float r2 = 0.567f;

        float pdf_a, pdf_b;
        Vec3f wi_a;
        ispc::Vec3f wi_b;
        const Color result_a = lobeCpp->sample(*mBsdfSliceCpp, r1, r2, wi_a, pdf_a);
        ispc::Color result_b;
        ispc::sampleLobe(lobeIspc, mBsdfSliceIspc.get(), r1, r2, wi_b, pdf_b, result_b);

        CPPUNIT_ASSERT(isEqualVec3f(wi_a, wi_b, sTolerance));
        CPPUNIT_ASSERT(isEqual(pdf_a, pdf_b, sTolerance));
        CPPUNIT_ASSERT(isEqualColor(result_a, result_b, sTolerance));
    }
}

void TestHair::testRLobe()
{
    // setup lobes
    const float longShift = -3.0f;
    const float longRoughness = 0.5f;
    static const Vec3f hairDir = normalize(Vec3f(1.0f, 1.0f, 1.0f));
    static const Color hairColor(0.65f, 0.21f, 0.25f);
    static const Vec2f hairUv(0.33f, 0.66f);
    static const float eta = 1.45;

    HairRLobe lobeCpp(hairDir,
                      hairUv,
                      1.0f, //mediumIOR
                      eta,
                      ispc::HAIR_FRESNEL_DIELECTRIC_CYLINDER,
                      0.0f, //cuticle layer thickness
                      longShift,
                      longRoughness);
    lobeCpp.setFresnel(mFresnelCpp.get());

    deleted_unique_ptr<void> lobeIspc(
            ispc::createRLobe(asIspc(hairDir),
                              asIspc(hairUv),
                              eta,
                              longShift,
                              longRoughness,
                              mFresnelIspc.get()),
            [](void * p) { ispc::deleteLobe(p); } );

    testLobe(&lobeCpp, lobeIspc.get());
}

void TestHair::testTRTLobe()
{
    // setup lobes
    const float longShift = -3.0f;
    const float longRoughness = 0.5f;
    static const float azimRoughness = 0.22f;
    static const Vec3f hairDir = normalize(Vec3f(1.0f, 1.0f, 1.0f));
    static const Color hairColor(0.65f, 0.21f, 0.25f);
    static const Color hairSigmaA = shading::HairUtil::computeAbsorptionCoefficients(hairColor,
                                                                                 azimRoughness);
    static const Vec2f hairUv(0.33f, 0.66f);
    static const float eta = 1.45;
    static const bool showGlint = true;
    static const float glintRoughness = 0.5f;
    static const float glintEccentricity = 0.85f;
    static const float glintSaturation = 0.5f;
    static const float hairRotation = 1.0f;
    static const Vec3f hairNormal = Vec3f(0.0f, 1.0f, 0.0f);

    HairTRTLobe lobeCpp(hairDir,
                        hairUv,
                        1.0f, //mediumIOR
                        eta,
                        ispc::HAIR_FRESNEL_DIELECTRIC_CYLINDER,
                        0.0f, //cuticle layer thickness
                        longShift,
                        longRoughness,
                        hairColor,
                        hairSigmaA,
                        sWhite,
                        showGlint,
                        glintRoughness,
                        glintEccentricity,
                        glintSaturation,
                        hairRotation,
                        hairNormal);
    lobeCpp.setFresnel(mFresnelCpp.get());

    deleted_unique_ptr<void> lobeIspc(
            ispc::createTRTLobe(asIspc(hairDir),
                                asIspc(hairUv),
                                eta,
                                longShift,
                                longRoughness,
                                asIspc(hairColor),
                                asIspc(hairSigmaA),
                                showGlint,
                                glintRoughness,
                                glintEccentricity,
                                glintSaturation,
                                hairRotation,
                                asIspc(hairNormal),
                                mFresnelIspc.get()),
            [](void * p) { ispc::deleteLobe(p); } );

    testLobe(&lobeCpp, lobeIspc.get());
}

void TestHair::testTTLobe()
{
    // setup lobes
    const float longShift = -3.0f;
    const float longRoughness = 0.5f;
    const float azimRoughness = 0.4f;
    static const Vec3f hairDir = normalize(Vec3f(1.0f, 1.0f, 1.0f));
    static const Color hairColor(0.65f, 0.21f, 0.25f);
    static const Color hairSigmaA = shading::HairUtil::computeAbsorptionCoefficients(hairColor,
                                                                                 azimRoughness);
    static const Vec2f hairUv(0.33f, 0.66f);
    static const float eta = 1.45;

    HairTTLobe lobeCpp(hairDir,
                       hairUv,
                       1.0f, //mediumIOR
                       eta,
                       ispc::HAIR_FRESNEL_DIELECTRIC_CYLINDER,
                       0.0f, //cuticle layer thickness
                       longShift,
                       longRoughness,
                       azimRoughness,
                       hairColor,
                       hairSigmaA);
    lobeCpp.setFresnel(mFresnelCpp.get());

    deleted_unique_ptr<void> lobeIspc(
            ispc::createTTLobe(asIspc(hairDir),
                               asIspc(hairUv),
                               eta,
                               longShift,
                               longRoughness,
                               azimRoughness,
                               asIspc(hairColor),
                               asIspc(hairSigmaA),
                               mFresnelIspc.get()),
            [](void * p) { ispc::deleteLobe(p); } );

    testLobe(&lobeCpp, lobeIspc.get());
}

void TestHair::testTRRTLobe()
{
    // setup lobes
    const float longRoughness = 0.5f;
    static const Vec3f hairDir = normalize(Vec3f(1.0f, 1.0f, 1.0f));
    static const Color hairColor(0.65f, 0.21f, 0.25f);
    static const Color hairSigmaA = shading::HairUtil::computeAbsorptionCoefficients(hairColor,
                                                                                 0.5f);
    static const Vec2f hairUv(0.33f, 0.66f);
    static const float eta = 1.45;

    HairTRRTLobe lobeCpp(hairDir,
                         hairUv,
                         1.0f, //mediumIOR
                         eta,
                         ispc::HAIR_FRESNEL_DIELECTRIC_CYLINDER,
                         0.0f, //cuticle layer thickness
                         longRoughness,
                         hairColor,
                         hairSigmaA);
    lobeCpp.setFresnel(mFresnelCpp.get());

    deleted_unique_ptr<void> lobeIspc(
            ispc::createTRRTLobe(
                    asIspc(hairDir),
                    asIspc(hairUv),
                    eta,
                    0.0f,
                    longRoughness,
                    asIspc(hairColor),
                    asIspc(hairSigmaA),
                    mFresnelIspc.get()),
            [](void * p) { ispc::deleteLobe(p); } );

    testLobe(&lobeCpp, lobeIspc.get());
}

void TestHair::testHairState()
{
    // pick a couple of random directions to test with
    const Vec3f wi_2(normalize(Vec3f(-0.55f, -0.33f, 0.85f)));

    // test calculateAbsorptionTerm()
    // check mAbsorptionTerm
    {
        const Color a = mHairStateCpp->mAbsorptionTerm;
        ispc::Color b;
        ispc::getAbsorptionTerm(mHairStateIspc.get(), b);
        CPPUNIT_ASSERT(isEqualColor(a, b, sTolerance));
    }

    // test updateLocalFrame()
    // check mWo, mHairNorm, mHairBiNorm, mThetaO, mSinThetaO, mCosThetaO, mAbsorptionTerm
    {
        { // mHairNorm
            const Vec3f a = mHairStateCpp->mHairNorm;
            ispc::Vec3f b;
            ispc::getHairNorm(mHairStateIspc.get(), b);
            CPPUNIT_ASSERT(isEqualVec3f(a, b, sTolerance));
        }
        { // mHairBiNorm
            const Vec3f a = mHairStateCpp->mHairBinorm;
            ispc::Vec3f b;
            ispc::getHairBiNorm(mHairStateIspc.get(), b);
            CPPUNIT_ASSERT(isEqualVec3f(a, b, sTolerance));
        }

        CPPUNIT_ASSERT(isEqual(mHairStateCpp->mThetaO, ispc::getThetaO(mHairStateIspc.get()), sTolerance));
        CPPUNIT_ASSERT(isEqual(mHairStateCpp->mSinThetaO, ispc::getSinThetaO(mHairStateIspc.get()), sTolerance));
        CPPUNIT_ASSERT(isEqual(mHairStateCpp->mCosThetaO, ispc::getCosThetaO(mHairStateIspc.get()), sTolerance));

        { // mAbsorptionTerm
            const Color a = mHairStateCpp->mAbsorptionTerm;
            ispc::Color b;
            ispc::getAbsorptionTerm(mHairStateIspc.get(), b);
            CPPUNIT_ASSERT(isEqualColor(a, b, sTolerance));
        }
    }

    // test updateAngles() (the one called from eval)
    // check mWi, mThetaI, mPhiI, mSinThetaI, mCosThetaI,
    // mPhiD, mThetaD, mCosineTerm
    {
        CPPUNIT_ASSERT(isEqual(mHairStateCpp->mThetaI, ispc::getThetaI(mHairStateIspc.get()), sTolerance));
        CPPUNIT_ASSERT(isEqual(mHairStateCpp->mPhiI, ispc::getPhiI(mHairStateIspc.get()), sTolerance));
        CPPUNIT_ASSERT(isEqual(mHairStateCpp->mSinThetaI, ispc::getSinThetaI(mHairStateIspc.get()), sTolerance));
        CPPUNIT_ASSERT(isEqual(mHairStateCpp->mCosThetaI, ispc::getCosThetaI(mHairStateIspc.get()), sTolerance));
        CPPUNIT_ASSERT(isEqual(mHairStateCpp->mPhiD, ispc::getPhiD(mHairStateIspc.get()), sTolerance));
        CPPUNIT_ASSERT(isEqual(mHairStateCpp->mThetaD, ispc::getThetaD(mHairStateIspc.get()), sTolerance));
    }

    // test updateAngles() (the one called from sample)
    // check mWi, mThetaI, mPhiI, mPhiD, mThetaD
    {
        const float phiI = 0.75f;
        const float thetaI = 0.15f;
        mHairStateCpp->updateAngles(wi_2, phiI, thetaI);
        ispc::updateAngles(mHairStateIspc.get(), asIspc(wi_2), phiI, thetaI);

        CPPUNIT_ASSERT(isEqual(mHairStateCpp->mThetaI, ispc::getThetaI(mHairStateIspc.get()), sTolerance));
        CPPUNIT_ASSERT(isEqual(mHairStateCpp->mPhiI, ispc::getPhiI(mHairStateIspc.get()), sTolerance));
        CPPUNIT_ASSERT(isEqual(mHairStateCpp->mPhiD, ispc::getPhiD(mHairStateIspc.get()), sTolerance));
        CPPUNIT_ASSERT(isEqual(mHairStateCpp->mThetaD, ispc::getThetaD(mHairStateIspc.get()), sTolerance));
    }
}

} // namespace shading
} // namespace moonray


CPPUNIT_TEST_SUITE_REGISTRATION(moonray::shading::TestHair);


