// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

///
/// @file HairState.h
/// $Id$
///

#pragma once

#include "HairUtil.h"

//#include <moonray/rendering/shading/ispc/bsdf/hair/BsdfHair_ispc_stubs.h>
#include <scene_rdl2/common/math/Color.h>
#include <scene_rdl2/common/math/Vec3.h>

namespace moonray {
namespace shading {

//----------------------------------------------------------------------------
///
/// @class HairState <shading/hair/HairState.h>
/// @brief This holds data that's common/shared between all the lobes
///
class HairState
{
    // allow unittest to access internals
	friend class TestHair;

public:

    // Constructor / Destructor
    // The given hair direction is assumed to be normalized by the caller
	// During Eval
    HairState(const scene_rdl2::math::Vec3f &wo,
              const scene_rdl2::math::Vec3f &wi,
              const scene_rdl2::math::Vec3f &hairDirection,
              const float h,
              const float refractiveIndex,
              const scene_rdl2::math::Color& sigmaA,
              const float hairRotation = 0.f,
              const scene_rdl2::math::Vec3f &hairNormal = scene_rdl2::math::Vec3f(0.f));

    // When Sampling from Glossy
    HairState(const scene_rdl2::math::Vec3f &wo,
              const scene_rdl2::math::Vec3f &hairDirection,
              const float h,
              const float refractiveIndex,
              const scene_rdl2::math::Color& sigmaA,
              const float hairRotation = 0.f,
              const scene_rdl2::math::Vec3f &hairNormal = scene_rdl2::math::Vec3f(0.f));

    // When Sampling from Diffuse
    HairState(const scene_rdl2::math::Vec3f &wo,
              const scene_rdl2::math::Vec3f &hairDirection);

    ~HairState(){}

    finline scene_rdl2::math::Color sigmaA()           const { return mSigmaA;         }
    finline scene_rdl2::math::Color absorptionTerm()   const { return mAbsorptionTerm; }

    finline scene_rdl2::math::Vec3f wo() const { return mWo; }
    finline scene_rdl2::math::Vec3f wi() const { return mWi; }

    finline scene_rdl2::math::Vec3f hairNormal()   const { return mHairNorm;   }
    finline scene_rdl2::math::Vec3f hairBinormal() const { return mHairBinorm; }

    finline float eta()   const { return mEta;   }
    finline float etaP()  const { return mEtaP;  }
    finline float etaPP() const { return mEtaPP; }

    finline float sinThetaI() const { return mSinThetaI; }
    finline float sinThetaO() const { return mSinThetaO; }
    finline float cosThetaI() const { return mCosThetaI; }
    finline float cosThetaO() const { return mCosThetaO; }
    finline float sinGammaO() const { return mSinGammaO; }
    finline float cosGammaO() const { return mCosGammaO; }
    finline float sinThetaD() const { return mSinThetaD; }
    finline float cosThetaD() const { return mCosThetaD; }
    finline float cosPhiDOverTwo() const { return mCosPhiDOverTwo; }

    // Getters for azimuthal angles
    finline float phiD() const { return mPhiD; }
    finline float phiH() const { return mPhiH; }

    finline float thetaI() const { return mThetaI; }
    finline float thetaO() const { return mThetaO; }
    finline float phiO()   const { return mPhiO; }

    void updateAngles(const scene_rdl2::math::Vec3f &direction,
                      float phiI,
                      float thetaI);

    scene_rdl2::math::Vec3f
    localToGlobal(const scene_rdl2::math::Vec3f& wi)
    {
        return (wi[0] * mHairDir +
                wi[1] * mHairNorm +
                wi[2] * mHairBinorm);

    }

    scene_rdl2::math::Vec3f
    localToGlobal(const float x,
                  const float y,
                  const float z)
    {
        return (x * mHairDir +
                y * mHairNorm +
                z * mHairBinorm);
    }


    // Calculate the Absorption Term
    static scene_rdl2::math::Color calculateAbsorptionTerm(const float  h,
                                               const float  eta,
                                               const scene_rdl2::math::Color& sigmaA,
                                               const float etaP,
                                               const float cosThetaO,
                                               const float sinThetaO);

private:
    // Helperfunctions ----------------------------------------------

    void calculateHairFrame(const scene_rdl2::math::Vec3f& wo,
                            const scene_rdl2::math::Vec3f& hairDirection,
                            const float hairRotation = 0.f,
                            const scene_rdl2::math::Vec3f& hairNormal = scene_rdl2::math::Vec3f(0.f));

    // Project a given direction vector into its longitudinal and azimuthal
    // component angles. This assumes that updateLocalFrame() has already been
    // called.
    void computeAngles(const scene_rdl2::math::Vec3f &direction,
                       const scene_rdl2::math::Vec3f &hairDir,
                       float &outTheta,
                       float &outPhi) const;

    // Calculate the Absorption Term
    void calculateAbsorptionTerm(const float  h,
                                 const float  refractiveIndex,
                                 const scene_rdl2::math::Color& sigmaA);

    scene_rdl2::math::Vec3f mWo, mWi;

    // Hair Reference Frame
    scene_rdl2::math::Vec3f mHairDir;
    scene_rdl2::math::Vec3f mHairNorm;
    scene_rdl2::math::Vec3f mHairBinorm;

    float mEta;
    float mEtaP, mEtaPP;

    float mSinGammaO;
    float mCosGammaO;

    float mThetaD;
    float mPhiD;
    float mPhiH;

    // Also depends on mWi
    float mSinThetaD;
    float mCosThetaD;
    float mCosPhiDOverTwo;

    float mPhiI;
    float mPhiO;
    float mThetaO, mThetaI;
    float mSinThetaI, mCosThetaI;
    float mSinThetaO, mCosThetaO;

    scene_rdl2::math::Color mSigmaA;
    scene_rdl2::math::Color mAbsorptionTerm;

    bool mUsingCustomNormal;

    // PhiO is always 0.0 when not using custom normal
    static constexpr float sPhiO = 0.0f;
};

} // namespace shading
} // namespace moonray


