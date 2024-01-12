// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

///
/// @file BsdfHair.h
/// $Id$
///

#pragma once

#include "HairState.h"
#include <moonray/rendering/shading/bsdf/Bsdf.h>

#include <moonray/rendering/shading/ispc/BsdfComponent_ispc_stubs.h>
#include <moonray/rendering/shading/ispc/bsdf/hair/BsdfHair_ispc_stubs.h>

#include <scene_rdl2/common/math/Color.h>
#include <scene_rdl2/common/math/Vec2.h>
#include <scene_rdl2/common/math/Vec3.h>

// Uncomment this to use uniform sampling
/* #define PBR_HAIR_USE_UNIFORM_SAMPLING 1 */

namespace moonray {
namespace shading {

//----------------------------------------------------------------------------
///
/// @class HairBsdfLobe BsdfHair.h <shading/BsdfHair.h>
/// @brief A non-functional interface class that defines some common
///        data members and accessors for all the hair lobes
/// 
class HairBsdfLobe : public BsdfLobe
{
    // allow unittest to access internals
    friend class TestHair;
public:
    // Constructor / Destructor. Note that shift and width are in radians
    HairBsdfLobe(Type type,
                 const scene_rdl2::math::Vec3f &hairDir,
                 const scene_rdl2::math::Vec2f &hairUV,
                 const float mediumIor,
                 const float ior,
                 const ispc::HairFresnelType fresnelType,
                 const float layers,
                 const float longShift,
                 const float longRoughness,
                 const float azimRoughness = 0.75f,
                 const scene_rdl2::math::Color &hairColor = scene_rdl2::math::sWhite,
                 const scene_rdl2::math::Color &hairSigmaA = scene_rdl2::math::sWhite,
                 const scene_rdl2::math::Color &tint = scene_rdl2::math::sWhite,
                 const float hairRotation = 0.f,
                 const scene_rdl2::math::Vec3f &hairNormal = scene_rdl2::math::Vec3f(scene_rdl2::math::zero));

    virtual ~HairBsdfLobe() {}

    // BsdfLobe API functions
    scene_rdl2::math::Color albedo(const BsdfSlice &slice) const override { return (getScale() * mTint); }

    scene_rdl2::math::Color eval(const BsdfSlice &slice,
                     const scene_rdl2::math::Vec3f &wi,
                     float *pdf = NULL) const override;
    scene_rdl2::math::Color sample(const BsdfSlice &slice,
                       float r1, float r2,
                       scene_rdl2::math::Vec3f &wi,
                       float &pdf) const override;

    void differentials(const scene_rdl2::math::Vec3f &wo,
                       const scene_rdl2::math::Vec3f &wi,
                       float r1, float r2,
                       const scene_rdl2::math::Vec3f &dNdx,
                       const scene_rdl2::math::Vec3f &dNdy,
                       scene_rdl2::math::Vec3f &dDdx, scene_rdl2::math::Vec3f &dDdy) const override;

    bool getProperty(Property property,
                     float *dest) const override;

    // getters
    float azimuthalVariance() const { return mAzimuthalVariance; }

    void show(std::ostream& os, const std::string& indent) const override
    {
        const scene_rdl2::math::Color& scale = getScale();
        const Fresnel * fresnel = getFresnel();
        os << indent << "    " << "scale: "
            << scale.r << " " << scale.g << " " << scale.b << "\n";
        os << indent << "    " << "hair dir: "
            << mHairDir.x << " " << mHairDir.y << " " << mHairDir.z << "\n";
        os << indent << "    " << "hair uv: "
            << mHairUV.x << " " << mHairUV.y << "\n";
        os << indent << "    " << "medium ior: " << mMediumIOR << "\n";
        os << indent << "    " << "eta: " << mIOR << "\n";
        os << indent << "    " << "fresnel type: " << mFresnelType << "\n";
        os << indent << "    " << "cuticle layer thickness: " << mCuticleLayerThickness << "\n";
        os << indent << "    " << "longitudinal roughness: " << mLongitudinalRoughness << "\n";
        os << indent << "    " << "longitudinal variance: " << mLongitudinalVariance << "\n";
        os << indent << "    " << "longitudinal shift: " << mLongitudinalShift << "\n";
        os << indent << "    " << "azimuthal roughness: " << mAzimuthalRoughness << "\n";
        os << indent << "    " << "azimuthal variance: " << mAzimuthalVariance << "\n";
        os << indent << "    " << "hair color: "
            << mHairColor.r << " " << mHairColor.g << " " << mHairColor.b << "\n";
        os << indent << "    " << "sigmaA: "
            << mSigmaA.r << " " << mSigmaA.g << " " << mSigmaA.b << "\n";
        os << indent << "    " << "sin alpha: " << mSinAlpha << "\n";
        os << indent << "    " << "cos alpha: " << mCosAlpha << "\n";
        os << indent << "    " << "H: " << mH << "\n";
        os << indent << "    " << "dD factor: " << mdDFactor << "\n";
        os << indent << "    " << "tint: "
            << mTint.r << " " << mTint.g << " " << mTint.b << "\n";
        os << indent << "    " << "hair rotation: " << mHairRotation << "\n";
        os << indent << "    " << "hair normal: "
                    << mHairNormal.x << " " << mHairNormal.y << " " << mHairNormal.z << "\n";

        if (fresnel) {
            fresnel->show(os, indent + "    ");
        }
    }

protected:
    // Internal Functions
    virtual float evalPdf(const HairState&) const;

    virtual scene_rdl2::math::Color fresnel(const HairState&,
                                const float cosTheta) const
    { return scene_rdl2::math::sWhite; }

    virtual scene_rdl2::math::Color absorption(const HairState&) const
    { return scene_rdl2::math::sWhite; }

    virtual scene_rdl2::math::Color evalBsdf(const HairState& hairState,
                                 bool includeCosineTerm) const;

    virtual float evalMTerm(const HairState&)  const;
    virtual scene_rdl2::math::Color evalNTermWithAbsorption(const HairState&)  const;

    virtual float evalThetaPdf(const HairState&) const;
    virtual float evalPhiPdf(const HairState&)   const;

    // Returns the PDF for sampling Theta
    virtual float sampleTheta(float  r,
                              float  thetaO,
                              float& thetaI) const;

    // Returns the PDF for sampling Phi
    virtual float samplePhi(float  r,
                            float  phiO,
                            float& phiI) const;

    virtual scene_rdl2::math::Color samplingWeight(const HairState&) const
    { return scene_rdl2::math::sWhite; }

    // Helper Function to Calculate the Hair Fresnel Response
    scene_rdl2::math::Color evalHairFresnel(const HairState& hairState,
                                float cosTheta) const;

    finline const scene_rdl2::math::Color&  getHairColor() const                { return mHairColor; }
    finline const scene_rdl2::math::Color&  getHairSigmaA() const               { return mSigmaA; }
    finline const scene_rdl2::math::Color&  getTint() const                     { return mTint; }
    finline       float         getRefractiveIndex() const          { return mIOR; }
    finline       float         getLongitudinalShift() const        { return mLongitudinalShift; }
    finline       float         getLongitudinalRoughness() const    { return mLongitudinalRoughness; }
    finline       float         getAzimuthalRoughness() const       { return mAzimuthalRoughness; }

    finline bool isLayeredFresnel() const { return mFresnelType == ispc::HAIR_FRESNEL_LAYERED_CUTICLES; }

private:
    float evalNTerm(const HairState&)  const;

    // Derive a directional differential size that varies according to width
    // Experimentally, we found that we want around 0.075 when width is
    // 0 degrees and 0.25 when it is 15 degrees. Also taking care that
    // mLongitudinalWidth is expressed in radians, not degrees...
    // 15.0f/180.f = 0.0833f
    static constexpr float sdDFactorMin = 0.075f;
    static constexpr float sdDFactorMax = 0.25f;
    static constexpr float sdDFactorSlope = (sdDFactorMax - sdDFactorMin)
                                                  / (0.0833f * scene_rdl2::math::sPi);

    scene_rdl2::math::Vec3f mHairDir;
    scene_rdl2::math::Vec2f mHairUV;

    float mMediumIOR;
    float mIOR;

    ispc::HairFresnelType mFresnelType;
    float mCuticleLayerThickness;

    // Longitudinal Roughness and Variance
    float mLongitudinalRoughness;
    float mLongitudinalVariance;

    // Azimuthal Roughness and Variance
    float mAzimuthalRoughness;
    float mAzimuthalVariance;

    // Hair Color and Absorption Coeffs
    scene_rdl2::math::Color mHairColor;
    scene_rdl2::math::Color mSigmaA;

    // sincos(offset)
    float mLongitudinalShift;
    float mSinAlpha;
    float mCosAlpha;

    // Parameterize the position across the hair-cross-section, h in [-1,1]
    float mH;

    // Ray Differentials Spread Factor
    float mdDFactor;

    // non-physical post-eval tinting, for extra artistic control
    scene_rdl2::math::Color mTint;

    float mHairRotation;
    scene_rdl2::math::Vec3f mHairNormal;

    friend class HairOneSampleLobe;
};

} // namespace shading
} // namespace moonray


