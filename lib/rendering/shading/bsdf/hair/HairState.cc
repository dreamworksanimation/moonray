// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

///
/// @file BsdfHair.cc
/// $Id$
///

#include "HairState.h"
#include <moonray/rendering/shading/Util.h>

#include <scene_rdl2/common/math/ReferenceFrame.h>
#include <scene_rdl2/common/math/Mat4.h>

using namespace scene_rdl2::math;

namespace moonray {
namespace shading {

constexpr float HairState::sPhiO;

HairState::HairState(const Vec3f &wo,
                     const Vec3f &wi,
                     const Vec3f &hairDir,
                     const float h,
                     const float refractiveIndex,
                     const Color& sigmaA,
                     const float hairRotation,
                     const Vec3f &hairNormal):
    mWo(wo),
    mWi(wi),
    mHairDir(hairDir),
    mEta(refractiveIndex),
    mSigmaA(sigmaA)
{
    mUsingCustomNormal = !isZero(hairRotation) && !isZero(hairNormal);

    calculateHairFrame(wo,
                       hairDir,
                       hairRotation,
                       hairNormal);

    computeAngles(wi,
                  hairDir,
                  mThetaI,
                  mPhiI);

    updateAngles(wi,
                 mPhiI,
                 mThetaI);

    calculateAbsorptionTerm(h,
                            refractiveIndex,
                            sigmaA);

    mSinGammaO = h;
    mCosGammaO = HairUtil::safeSqrt(1.0f - h*h);
}

HairState::HairState(const Vec3f &wo,
                     const Vec3f &hairDir,
                     const float h,
                     const float refractiveIndex,
                     const Color& sigmaA,
                     const float hairRotation,
                     const Vec3f &hairNormal):
    mWo(wo),
    mHairDir(hairDir),
    mEta(refractiveIndex),
    mSigmaA(sigmaA)
{
    mUsingCustomNormal = !isZero(hairRotation) && !isZero(hairNormal);
    mPhiI = 0.0f;
    mPhiD = 0.0f;

    calculateHairFrame(wo,
                       hairDir,
                       hairRotation,
                       hairNormal);

    calculateAbsorptionTerm(h,
                            refractiveIndex,
                            sigmaA);

    mSinGammaO = h;
    mCosGammaO = HairUtil::safeSqrt(1.0f - h*h);
}

HairState::HairState(const Vec3f &wo,
                     const Vec3f &hairDir)
{
    mUsingCustomNormal = false;

    calculateHairFrame(wo,
                       hairDir);
    mSinGammaO = 0.0f;
    mCosGammaO = 1.0f;
}

void
HairState::calculateHairFrame(const Vec3f& wo,
                              const Vec3f& hairDir,
                              const float hairRotation,
                              const Vec3f& hairNormal)
{
    float UdotO = dot(wo, hairDir);
    UdotO = clamp(UdotO, -1.0f, 1.0f);

    if (mUsingCustomNormal) {
        mHairNorm = Vec3f(hairNormal);
        scene_rdl2::math::Mat4f mat = scene_rdl2::math::Mat4f::rotate(hairDir, hairRotation);
        mHairNorm = transformVector(mat, mHairNorm);
    } else {
        // Set hair normal to align with eyeDir's projection onto normal plane
        // to ease of computation
        mHairNorm   = wo - (UdotO * hairDir);
    }
    mHairBinorm = cross(mHairNorm, hairDir);

    if (lengthSqr(mHairBinorm) > sEpsilon) {
        mHairNorm   = normalize(mHairNorm);
        mHairBinorm = normalize(mHairBinorm);
    } else {
        // rare case that wo is parallel to hairdir, need a safe fallback
        // to avoid NaN normal/binormal
        ReferenceFrame frame(wo);
        mHairNorm  = frame.getX();
        mHairBinorm = frame.getY();
    }

    mThetaO = sHalfPi - acos(UdotO);
    sincos(mThetaO, &mSinThetaO, &mCosThetaO);

    if (mUsingCustomNormal) {
        scene_rdl2::math::Vec3f proj = normalize(wo - hairDir * dot(wo, hairDir));
        const float cosPhiO = clamp(dot(proj, mHairNorm), -1.0f, 1.0f);
        const float sinPhiO = clamp(dot(proj, mHairBinorm), -1.0f, 1.0f);
        mPhiO = atan2(sinPhiO, cosPhiO);
    } else {
        // normal is aligned to view direction
        mPhiO = sPhiO;
    }

    // "Light Scattering from Human Hair Fibers" - Marschner et al
    // Appendix B
    const float tmp = HairUtil::safeSqrt(mEta*mEta - mSinThetaO*mSinThetaO);
    mEtaP  = tmp / mCosThetaO;
    mEtaPP = mEta*mEta*mCosThetaO / tmp;

}

// From PBRTv3
void
HairState::calculateAbsorptionTerm(const float  h,
                                   const float  refractiveIndex,
                                   const Color& sigmaA)
{
    mAbsorptionTerm = scene_rdl2::math::sWhite;

    float cosGammaT = 0.0f;
    // Calculate EtaP
    if (!isZero(cosThetaO())) {
        const float sinGammaT = h / mEtaP;
        cosGammaT = HairUtil::safeSqrt(1.0f - sinGammaT*sinGammaT);
    }
    const float sinThetaT = sinThetaO() / refractiveIndex;
    const float cosThetaT = HairUtil::safeSqrt(1.0f - sinThetaT*sinThetaT);
    if (!isZero(cosThetaT))
        mAbsorptionTerm = scene_rdl2::math::exp(-sigmaA * 2.0f * cosGammaT / cosThetaT);
}

// static function for calculating absorption based
// on a modified eta and etaP
Color
HairState::calculateAbsorptionTerm(const float  h,
                                   const float  eta,
                                   const Color& sigmaA,
                                   const float etaP,
                                   const float cosThetaO,
                                   const float sinThetaO)
{
    Color absorptionTerm = scene_rdl2::math::sWhite;

    float cosGammaT = 0.0f;
    // Calculate EtaP
    if (!isZero(cosThetaO)) {
        const float sinGammaT = h / etaP;
        cosGammaT = HairUtil::safeSqrt(1.0f - sinGammaT*sinGammaT);
    }
    const float sinThetaT = sinThetaO / eta;
    const float cosThetaT = HairUtil::safeSqrt(1.0f - sinThetaT*sinThetaT);
    if (!isZero(cosThetaT))
        absorptionTerm = scene_rdl2::math::exp(-sigmaA * 2.0f * cosGammaT / cosThetaT);

    return absorptionTerm;
}

// Compute the direction in hair local frame (outTheta, outPhi)
// outTheta is in [-pi/2, pi/2]
// outPhi is in the range [0, pi] when !mUsingCustomNormal
// and in the range [-pi, pi] when mUsingCustomNormal
// WARNING: Don't use these angles to sample directions!!!
void
HairState::computeAngles(const Vec3f &direction,
                         const Vec3f &hairDir,
                         float &outTheta,
                         float &outPhi) const
{
    const float cosTheta = dot(direction,
                               hairDir);
    outTheta = sHalfPi - acos(clamp(cosTheta, -1.0f, 1.0f));

    // Project direction into azimuth plane
    Vec3f proj = direction - hairDir * dot(direction, hairDir);
    float l = lengthSqr(proj);
    if (l <= sEpsilon) { outPhi = 0.0f; return; }
    proj /= sqrt(l);

    const float cosPhi = clamp(dot(proj, mHairNorm), -1.0f, 1.0f);
    if (mUsingCustomNormal) {
        const float sinPhi = clamp(dot(proj, mHairBinorm), -1.0f, 1.0f);
        outPhi = atan2(sinPhi, cosPhi);
    } else {
        // normal is aligned to view direction
        // WARNING: We exploit the symmetry of the bsdf in the azimuth, to use
        // acos instead of atan2. We get a phi value only in half the possible
        // range of directions. Don't use these angles to generate directions!!!
        // Only use these angles to evaluate the pdf or the bsdf.
        outPhi = acos(cosPhi);
    }
}

void
HairState::updateAngles(const Vec3f &wi,
                        const float phiI,
                        const float thetaI)
{
    mThetaI = thetaI;
    mPhiI = phiI;

    sincos(mThetaI, &mSinThetaI, &mCosThetaI);

    if (mUsingCustomNormal) {
        scene_rdl2::math::Vec3f proj = normalize(mWo - mHairDir * dot(mWo, mHairDir));
        const float cosPhiO = clamp(dot(proj, mHairNorm), -1.0f, 1.0f);
        const float sinPhiO = clamp(dot(proj, mHairBinorm), -1.0f, 1.0f);
        mPhiO = atan2(sinPhiO, cosPhiO);
    } else {
        // normal is aligned to view direction
        mPhiO = sPhiO;
    }

    mPhiD   = rangeAngle(mPhiO - mPhiI);
    mPhiH   = (mPhiO + mPhiI) * 0.5f;
    mThetaD = scene_rdl2::math::abs(mThetaO - mThetaI)    * 0.5f;

    mCosPhiDOverTwo = cos(mPhiD*0.5f);
    sincos(mThetaD, &mSinThetaD, &mCosThetaD);

    // "Light Scattering from Human Hair Fibers" - Marschner et al
    // Appendix B
    const float tmp = sqrt(mEta*mEta - mSinThetaD*mSinThetaD);
    mEtaP  = tmp / mCosThetaD;
    mEtaPP = mEta*mEta*mCosThetaD / tmp;

}

//----------------------------------------------------------------------------

} // namespace shading
} // namespace moonray

