// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

#include "BsdfHair.isph"

#include <scene_rdl2/common/platform/IspcUtil.isph>
#include <scene_rdl2/common/math/ispc/Mat4.isph>

void
HairState_calculateHairFrame(varying HairState * uniform hairState,
                             const varying Vec3f& wo,
                             const varying Vec3f& hairDir,
                             const varying float hairRotation,
                             const varying Vec3f& hairNormal)
{

    float UdotO = dot(wo, hairDir);
    UdotO = clamp(UdotO, -1.0f, 1.0f);

    hairState->mHairDir = hairDir;

    if (hairState->mUsingCustomNormal) {
        hairState->mHairNorm = Vec3f_ctor(hairNormal.x, hairNormal.y, hairNormal.z);
        Mat4f mat = Mat4f_rotate(hairDir, hairRotation);
        hairState->mHairNorm = transformVector(mat, hairState->mHairNorm);
    } else {
        // Set hair normal to align with eyeDir's projection onto normal plane
        // to ease of computation
        hairState->mHairNorm   = wo - (UdotO * hairDir);
    }

    hairState->mHairBiNorm = cross(hairState->mHairNorm, hairDir);

    if (lengthSqr(hairState->mHairBiNorm) > sEpsilon) {
        hairState->mHairNorm   = normalize(hairState->mHairNorm);
        hairState->mHairBiNorm = normalize(hairState->mHairBiNorm);
    } else {
        // rare case that wo is parallel to hairdir, need a safe fallback
        // to avoid NaN normal/binormal
        ReferenceFrame frame;
        ReferenceFrame_init(frame, wo);
        hairState->mHairNorm  = frame.mX;
        hairState->mHairBiNorm = frame.mY;
    }

    hairState->mThetaO = sHalfPi - dw_acos(UdotO);
    sincos(hairState->mThetaO, &hairState->mSinThetaO, &hairState->mCosThetaO);

    if (hairState->mUsingCustomNormal) {
        Vec3f proj = normalize(wo - hairDir * dot(wo, hairDir));
        float cosPhiO = clamp(dot(proj, hairState->mHairNorm), -1.0f, 1.0f);
        float sinPhiO = clamp(dot(proj, hairState->mHairBiNorm), -1.0f, 1.0f);
        hairState->mPhiO = atan2(sinPhiO, cosPhiO);
    } else {
        // normal is aligned to view direction
        hairState->mPhiO = HairState_sPhiO;
    }

    float tmp = sqrt(hairState->mEta * hairState->mEta -
                     hairState->mSinThetaO * hairState->mSinThetaO);
    hairState->mEtaP  = tmp / hairState->mCosThetaO;
    hairState->mEtaPP = hairState->mEta * hairState->mEta *
            hairState->mCosThetaO / tmp;
}

// From PBRTv3
void
HairState_calculateAbsorptionTerm(varying HairState * uniform hairState,
                                  const varying float h,
                                  const varying float refractiveIndex,
                                  const varying Color& sigmaAC)
{
    float cosGammaT = 0.0f;

    // Calculate EtaP
    if (!isZero(hairState->mCosThetaO)) {
        float sinGammaT = h / hairState->mEtaP;
        cosGammaT = HairUtil_safeSqrt(1.0f - sinGammaT*sinGammaT);
    }
    float sinThetaT = hairState->mSinThetaO / refractiveIndex;
    float cosThetaT = HairUtil_safeSqrt(1.0f - HairUtil_sqr(sinThetaT));
    if (!isZero(cosThetaT)) {
        // workaround for missing neg() function for Color type,
        // and missing exp() function for color type
        const varying float * uniform sigmaA = asArray(sigmaAC);
        const float k = 2.0f * cosGammaT / cosThetaT;
        hairState->mAbsorptionTerm = Color_ctor(exp(-sigmaA[0] * k),
                                                exp(-sigmaA[1] * k),
                                                exp(-sigmaA[2] * k));
    } else {
        hairState->mAbsorptionTerm = sWhite;
    }
}

// function for calculating absorption based
// on a modified eta and etaP
Color
HairState_calculateAbsorptionTerm(const varying float h,
                                  const varying float eta,
                                  const varying Color& sigmaAC,
                                  const varying float etaP,
                                  const varying float cosThetaO,
                                  const varying float sinThetaO)
{
    Color absorptionTerm = sWhite;
    float cosGammaT = 0.0f;

    // Calculate EtaP
    if (!isZero(cosThetaO)) {
        float sinGammaT = h / etaP;
        cosGammaT = HairUtil_safeSqrt(1.0f - sinGammaT*sinGammaT);
    }
    float sinThetaT = sinThetaO / eta;
    float cosThetaT = HairUtil_safeSqrt(1.0f - HairUtil_sqr(sinThetaT));
    if (!isZero(cosThetaT)) {
        // workaround for missing neg() function for Color type,
        // and missing exp() function for color type
        const varying float * uniform sigmaA = asArray(sigmaAC);
        const float k = 2.0f * cosGammaT / cosThetaT;

        absorptionTerm = Color_ctor(exp(-sigmaA[0] * k),
                                         exp(-sigmaA[1] * k),
                                         exp(-sigmaA[2] * k));
    }
    return absorptionTerm;
}

// Compute the direction in hairState local frame (outTheta, outPhi)
// outTheta is in [-pi/2, pi/2]
// outPhi is in the range [0, pi] when !mUsingCustomNormal
// and in the range [-pi, pi] when mUsingCustomNormal
// WARNING: Don't use these angles to sample directions!!!
void
HairState_computeAngles(varying HairState * uniform hairState,
                        const varying Vec3f& direction,
                        const varying Vec3f& hairDir,
                        varying float& outTheta,
                        varying float& outPhi)
{
    const float cosTheta = dot(direction,
                               hairDir);
    outTheta = sHalfPi - dw_acos(clamp(cosTheta, -1.0f, 1.0f));

    // Project direction into azimuth plane
    Vec3f proj = direction - hairDir * dot(direction, hairDir);
    const float l = lengthSqr(proj);
    if (l <= sEpsilon) { outPhi = 0.0f; return; }
    proj = proj / sqrt(l);

    float cosPhi = clamp(dot(proj, hairState->mHairNorm), -1.0f, 1.0f);
    if (hairState->mUsingCustomNormal) {
        float sinPhi = clamp(dot(proj, hairState->mHairBiNorm), -1.0f, 1.0f);
        outPhi = atan2(sinPhi, cosPhi);
    } else {
        // WARNING: We exploit the symmetry of the bsdf in the azimuth, to use
        // acos instead of atan2. We get a phi value only in half the possible
        // range of directions. Don't use these angles to generate directions!!!
        // Only use these angles to evaluate the pdf or the bsdf.
        outPhi = dw_acos(cosPhi);
    }
}

void
HairState_updateAngles(varying HairState * uniform hairState,
                       const varying Vec3f &direction,
                       const varying float phiI,
                       const varying float thetaI)
{
    // This version is called from sample(), where we already know the
    // projection angles ahead of time and we're pretty much guaranteed wi
    // is changing

    hairState->mThetaI = thetaI;
    hairState->mPhiI = phiI;

    sincos(thetaI, &hairState->mSinThetaI, &hairState->mCosThetaI);

    if (hairState->mUsingCustomNormal) {
        Vec3f proj = normalize(hairState->mWo - hairState->mHairDir * dot(hairState->mWo, hairState->mHairDir));
        float cosPhiO = clamp(dot(proj, hairState->mHairNorm), -1.0f, 1.0f);
        float sinPhiO = clamp(dot(proj, hairState->mHairBiNorm), -1.0f, 1.0f);
        hairState->mPhiO = atan2(sinPhiO, cosPhiO);
    } else {
        // normal is aligned to view direction
        hairState->mPhiO = HairState_sPhiO;
    }

    hairState->mPhiD = rangeAngle(hairState->mPhiO - hairState->mPhiI);
    hairState->mPhiH = (hairState->mPhiO + hairState->mPhiI) * 0.5f;
    hairState->mThetaD = abs(hairState->mThetaO - hairState->mThetaI) * 0.5f;

    hairState->mCosPhiDOverTwo = cos(hairState->mPhiD*0.5f);
    sincos(hairState->mThetaD,
           &hairState->mSinThetaD, &hairState->mCosThetaD);

    float tmp = sqrt(hairState->mEta * hairState->mEta -
                     hairState->mSinThetaD * hairState->mSinThetaD);
    hairState->mEtaP  = tmp / hairState->mCosThetaD;
    hairState->mEtaPP = hairState->mEta * hairState->mEta *
            hairState->mCosThetaD / tmp;

}

// Called from Sample
Vec3f
HairState_localToGlobal(varying HairState * uniform hairState,
                        const varying Vec3f &direction)
{
    return (direction.x * hairState->mHairDir +
            direction.y * hairState->mHairNorm +
            direction.z * hairState->mHairBiNorm);
}

Vec3f
HairState_localToGlobal(varying HairState * uniform hairState,
                        const varying float x,
                        const varying float y,
                        const varying float z)
{
    return (x * hairState->mHairDir +
            y * hairState->mHairNorm +
            z * hairState->mHairBiNorm);

}

void
HairState_init(varying HairState * uniform hairState,
               const varying Vec3f& wo,
               const varying Vec3f& wi,
               const varying Vec3f& hairDir,
               const varying float h,
               const varying float refractiveIndex,
               const varying Color& sigmaA,
               const varying float hairRotation,
               const varying Vec3f& hairNormal)
{
    hairState->mWo = wo;
    hairState->mEta = refractiveIndex;
    hairState->mSigmaA = sigmaA;

    hairState->mUsingCustomNormal = !isZero(hairRotation) && !isZero(hairNormal);

    HairState_calculateHairFrame(hairState,
                                 wo,
                                 hairDir,
                                 hairRotation,
                                 hairNormal);

    HairState_computeAngles(hairState,
                            wi,
                            hairDir,
                            hairState->mThetaI,
                            hairState->mPhiI);

    HairState_updateAngles(hairState,
                           wi,
                           hairState->mPhiI,
                           hairState->mThetaI);

    HairState_calculateAbsorptionTerm(hairState,
                                      h,
                                      refractiveIndex,
                                      sigmaA);

    hairState->mSinGammaO = h;
    hairState->mCosGammaO = HairUtil_safeSqrt(1.0f - h*h);

}

void
HairState_init(varying HairState * uniform hairState,
               const varying Vec3f& wo,
               const varying Vec3f& hairDir,
               const varying float h,
               const varying float refractiveIndex,
               const varying Color& sigmaA,
               const varying float hairRotation,
               const varying Vec3f& hairNormal)
{
    hairState->mWo = wo;
    hairState->mHairDir = hairDir;
    hairState->mEta = refractiveIndex;
    hairState->mSigmaA = sigmaA;

    hairState->mUsingCustomNormal = !isZero(hairRotation) && !isZero(hairNormal);
    hairState->mPhiI = 0.0f;
    hairState->mPhiD = 0.0f;

    HairState_calculateHairFrame(hairState,
                                 wo,
                                 hairDir,
                                 hairRotation,
                                 hairNormal);

    HairState_calculateAbsorptionTerm(hairState,
                                      h,
                                      refractiveIndex,
                                      sigmaA);

    hairState->mSinGammaO = h;
    hairState->mCosGammaO = HairUtil_safeSqrt(1.0f - h*h);
}

void
HairState_init(varying HairState * uniform hairState,
               const varying Vec3f& wo,
               const varying Vec3f& hairDir)
{
    const varying float hairRotation = 0.0f;
    const varying Vec3f hairNormal = Vec3f_ctor(0.0f);
    hairState->mUsingCustomNormal = false;

    HairState_calculateHairFrame(hairState,
                                 wo,
                                 hairDir,
                                 hairRotation,
                                 hairNormal);
    hairState->mSinGammaO = 0.0f;
    hairState->mCosGammaO = 1.0f;
}

#pragma ignore warning(all)
ISPC_UTIL_EXPORT_STRUCT_TO_HEADER(HairState);


