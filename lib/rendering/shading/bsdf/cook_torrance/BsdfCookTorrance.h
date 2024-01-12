// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

///
/// @file BsdfCookTorrance.h
/// $Id$
///

#pragma once

#include <moonray/rendering/shading/bsdf/Bsdf.h>
#include <moonray/rendering/shading/bsdf/BsdfSlice.h>
#include <moonray/rendering/shading/ispc/bsdf/cook_torrance/BsdfCookTorrance_ispc_stubs.h>
#include <moonray/rendering/shading/ispc/bsdf/cook_torrance/BsdfTransmissionCookTorrance_ispc_stubs.h>
#include <moonray/rendering/shading/ispc/BsdfComponent_ispc_stubs.h>

#include <scene_rdl2/common/math/Color.h>
#include <scene_rdl2/common/math/ReferenceFrame.h>
#include <scene_rdl2/common/math/Vec3.h>
#include <scene_rdl2/common/platform/IspcUtil.h>

namespace moonray {
namespace shading {


// Sample with a slightly widened roughness at grazing angles
#define PBR_CT_SAMPLE_WIDENED_ROUGHNESS 1

//----------------------------------------------------------------------------

///
/// @class CookTorranceBsdfLobe BsdfCookTorrance.h <shading/BsdfCookTorrance.h>
/// @brief The Cook-Torrance bsdf using Beckmann Distribution.
/// 
class CookTorranceBsdfLobe : public BsdfLobe
{
public:
    // Constructor / Destructor
    // Assumes N and anisoDirection are normalized
    CookTorranceBsdfLobe(const scene_rdl2::math::Vec3f &N, float roughness,
                         const scene_rdl2::math::Color& favg = scene_rdl2::math::sBlack,
                         const scene_rdl2::math::Color& favgInv = scene_rdl2::math::sBlack,
                         float etaI = 1.0f,
                         float etaT = 1.5f,
                         bool coupledWithTransmission = false);
    virtual ~CookTorranceBsdfLobe() {}

    // BsdfLobe API
    virtual scene_rdl2::math::Color eval(const BsdfSlice &slice, const scene_rdl2::math::Vec3f &wi, float *pdf = NULL) const override;
    virtual scene_rdl2::math::Color sample(const BsdfSlice &slice, float r1, float r2,
            scene_rdl2::math::Vec3f &wi, float &pdf) const override;

    void differentials(const scene_rdl2::math::Vec3f &wo, const scene_rdl2::math::Vec3f &wi, float r1, float r2,
            const scene_rdl2::math::Vec3f &dNdx, const scene_rdl2::math::Vec3f &dNdy,
            scene_rdl2::math::Vec3f &dDdx, scene_rdl2::math::Vec3f &dDdy) const override;

    finline scene_rdl2::math::Color albedo(const BsdfSlice &slice) const override
    {
        float cosThetaWo = scene_rdl2::math::max(dot(mFrame.getN(), slice.getWo()), 0.0f);
        // TODO: Improve this approximation!
        return computeScaleAndFresnel(cosThetaWo);
    }

    bool getProperty(Property property, float *dest) const override;

    void show(std::ostream& os, const std::string& indent) const override
    {
        const scene_rdl2::math::Color& scale = getScale();
        const scene_rdl2::math::Vec3f& N = mFrame.getN();
        const Fresnel * fresnel = getFresnel();
        os << indent << "[CookTorranceBsdfLobe]\n";
        os << indent << "    " << "scale: "
            << scale.r << " " << scale.g << " " << scale.b << "\n";
        os << indent << "    " << "N: "
            << N.x << " " << N.y << " " << N.z << "\n";
        os << indent << "    " << "roughness^2: " << mRoughness << "\n";
        if (fresnel) {
            fresnel->show(os, indent + "    ");
        }
    }

    // Widen roughness at grazing angles, to reduce maximum weight
    // Also called the "Walter trick". See sect. 5.3 "Modified Sampling Distrubtion"
    static float
    widenRoughness(float roughness, float cosNO)
    {
    #if PBR_CT_SAMPLE_WIDENED_ROUGHNESS
        return (1.2f - 0.2f * scene_rdl2::math::sqrt(cosNO)) * roughness;
    #else
        return roughness;
    #endif
    }

protected:
    // Derive a directional differential scale that varies according to roughness
    // Experimentally, we found that we want 1.0 when roughness is 0
    // and 8 when it is 1.
    static constexpr float sdDFactorMin = 1.0f;
    static constexpr float sdDFactorMax = 8.0f;
    static constexpr float sdDFactorSlope = (sdDFactorMax - sdDFactorMin);

    scene_rdl2::math::ReferenceFrame mFrame;
    float mInputRoughness;
    float mRoughness;
    float mInvRoughness;
    float mdDFactor;

    // Energy Compensation Params
    // Ref:"Revisiting Physically Based Shading", Kulla'17
    // Average Fresnel Reflectance
    scene_rdl2::math::Color mFavg;
    // Average Fresnel Reflectance when Exiting
    scene_rdl2::math::Color mFavgInv;
    float mEtaI, mEtaT;
    // Bool to keep track of whether
    // this reflection lobe is coupled with a transmission
    // lobe. This information is used to compensate missing
    // energy correctly.
    bool mCoupledWithTransmission;

protected:
    // Energy Compensation Helper Functions
    float energyCompensationWeight() const;
    float energyCompensationPDF(
            ispc::MicrofacetDistribution type,
            float cosNI) const;
    void sampleEnergyCompensation(
            ispc::MicrofacetDistribution type,
            const scene_rdl2::math::Vec3f& wo,
            float r1, float r2,
            scene_rdl2::math::Vec3f& wi) const;
    scene_rdl2::math::Color evalEnergyCompensation(
            ispc::MicrofacetDistribution type,
            float cosNO, float cosNI,
            bool includeCosineTerm) const;

private:
    void computeBeckmannMicrofacet(float r1, float r2,
                                   float cosNO,
                                   scene_rdl2::math::Vec3f &m) const;

};


///
/// @class TransmissionCookTorranceBsdfLobe BsdfCookTorrance.h <shading/BsdfCookTorrance.h>
/// @brief The Cook-Torrance transmission bsdf using Beckmann Distribution.
/// Note: This bsdf lobe only works when roughness > 0 and when
/// iorIncident != iorTransmitted
///
class TransmissionCookTorranceBsdfLobe : public BsdfLobe
{
public:
    // Constructor / Destructor
    // Assumes N and anisoDirection are normalized
    // Pass IOR for incident and transmitted media
    TransmissionCookTorranceBsdfLobe(const scene_rdl2::math::Vec3f &N, float roughness,
                                     float iorIncident,
                                     float iorTransmitted,
                                     const scene_rdl2::math::Color &tint,
                                     float favg,
                                     float favgInv,
                                     float abbe = 0.0f);
    ~TransmissionCookTorranceBsdfLobe() {}

    // BsdfLobe API
    scene_rdl2::math::Color eval(const BsdfSlice &slice, const scene_rdl2::math::Vec3f &wi, float *pdf = NULL) const override;
    scene_rdl2::math::Color sample(const BsdfSlice &slice, float r1, float r2,
            scene_rdl2::math::Vec3f &wi, float &pdf) const override;

    void differentials(const scene_rdl2::math::Vec3f &wo, const scene_rdl2::math::Vec3f &wi, float r1, float r2,
            const scene_rdl2::math::Vec3f &dNdx, const scene_rdl2::math::Vec3f &dNdy,
            scene_rdl2::math::Vec3f &dDdx, scene_rdl2::math::Vec3f &dDdy) const override;

    finline scene_rdl2::math::Color albedo(const BsdfSlice &slice) const override
    {
        float cosThetaWo = scene_rdl2::math::max(dot(mFrame.getN(), slice.getWo()), 0.0f);
        // TODO: Improve this approximation!
        return mTint * computeScaleAndFresnel(cosThetaWo);
    }

    bool getProperty(Property property, float *dest) const override;

    void show(std::ostream& os, const std::string& indent) const override
    {
        const scene_rdl2::math::Color& scale = getScale();
        const scene_rdl2::math::Vec3f& N = mFrame.getN();
        const Fresnel * fresnel = getFresnel();
        os << indent << "[TransmissionCookTorranceBsdfLobe]\n";
        os << indent << "    " << "scale: "
            << scale.r << " " << scale.g << " " << scale.b << "\n";
        os << indent << "    " << "N: "
            << N.x << " " << N.y << " " << N.z << "\n";
        os << indent << "    " << "roughness^2: " << mRoughness << "\n";
        os << indent << "    " << "tint: "
            << mTint.r << " " << mTint.g << " " << mTint.b << "\n";
        os << indent << "    " << "etaI: " << mEtaI << "\n";
        os << indent << "    " << "etaT: " << mEtaT << "\n";
        os << indent << "    " << "allow dispersion: " << (mAllowDispersion ? "true" : "false") << "\n";
        os << indent << "    " << "etaRGB: " << mEtaR << " " << mEtaG << " " << mEtaB << "\n";

        if (fresnel) {
            fresnel->show(os, indent + "    ");
        }
    }

protected:
    // BsdfLobe API
    scene_rdl2::math::Color eval(const BsdfSlice &slice,
               const scene_rdl2::math::Vec3f &wi,
               float iorTransmitted,
               float *pdf = NULL) const;

    // Derive a directional differential scale that varies according to roughness
    // Experimentally, we found that we want 1.0 when roughness is 0
    // and 8 when it is 1.
    static constexpr float sdDFactorMin = 1.0f;
    static constexpr float sdDFactorMax = 8.0f;
    static constexpr float sdDFactorSlope = (sdDFactorMax - sdDFactorMin);

    scene_rdl2::math::ReferenceFrame mFrame;
    scene_rdl2::math::Color mTint;
    float mInputRoughness;
    float mRoughness;
    float mInvRoughness;
    float mdDFactor;
    float mEtaI;
    float mEtaT;
    float mNeta;
    float mEtaR, mEtaG, mEtaB;
    bool  mAllowDispersion;
    // Energy Compensation Params
    // Ref:"Revisiting Physically Based Shading", Kulla'17
    // Average Fresnel Reflectance
    float mFavg;
    float mFavgInv;
};

///
/// @class GGXCookTorranceBsdfLobe BsdfCookTorrance.h <shading/BsdfCookTorrance.h>
/// @brief The Cook-Torrance bsdf using GGX Distribution.
/// 
class GGXCookTorranceBsdfLobe : public CookTorranceBsdfLobe
{
public:
    // Constructor / Destructor
    // Assumes N and anisoDirection are normalized
    GGXCookTorranceBsdfLobe(const scene_rdl2::math::Vec3f &N, float roughness,
                            const scene_rdl2::math::Color& favg = scene_rdl2::math::sBlack,
                            const scene_rdl2::math::Color& favgInv = scene_rdl2::math::sBlack,
                            float etaI = 1.0f,
                            float etaT = 1.5f,
                            bool coupledWithTransmission = false);
    ~GGXCookTorranceBsdfLobe() {}

    // BsdfLobe API
    scene_rdl2::math::Color eval(const BsdfSlice &slice, const scene_rdl2::math::Vec3f &wi, float *pdf = NULL) const override;
    scene_rdl2::math::Color sample(const BsdfSlice &slice, float r1, float r2,
                 scene_rdl2::math::Vec3f &wi, float &pdf) const override;

protected:
    // These two functions allow sharing of common code between GGXCookTorrance and GlitterGGXCookTorrance
    // It is a specialization required to specify the normal (frame) to be used for the bsdf sampling/evaluation
    scene_rdl2::math::Color eval(const BsdfSlice &slice, const scene_rdl2::math::Vec3f &wi, float *pdf,
               const float cosNO, const float cosNI, const scene_rdl2::math::ReferenceFrame& frame) const;

    // Returns wi given a frame and wo
    scene_rdl2::math::Vec3f sample(const BsdfSlice &slice, float cosNO, float r1, float r2,
                 const scene_rdl2::math::ReferenceFrame& frame) const;

    void show(std::ostream& os, const std::string& indent) const override
    {
        const scene_rdl2::math::Color& scale = getScale();
        const scene_rdl2::math::Vec3f& N = mFrame.getN();
        const Fresnel * fresnel = getFresnel();
        os << indent << "[GGXCookTorranceBsdfLobe]\n";
        os << indent << "    " << "scale: "
            << scale.r << " " << scale.g << " " << scale.b << "\n";
        os << indent << "    " << "N: "
            << N.x << " " << N.y << " " << N.z << "\n";
        os << indent << "    " << "roughness^2: " << mRoughness << "\n";
        if (fresnel) {
            fresnel->show(os, indent + "    ");
        }
    }
};

///
/// @class GlitterGGXCookTorranceBsdfLobe BsdfCookTorrance.h <shading/BsdfCookTorrance.h>
/// @brief The Cook-Torrance bsdf using GGX Distribution, specialized to handled
/// perturbed normals used for large glitter flakes.
///
/// Note: The main modification in this lobe is that when the flake normal is too
/// perturbed to have wo and wi within the shading hemisphere (wrt to surface normal),
/// sampling/evaluation is done using the surface shading normal instead. This is a
/// hack to prevent occurence of black flakes but allow high variation in the flake normals
///
class GlitterGGXCookTorranceBsdfLobe : public GGXCookTorranceBsdfLobe
{
public:
    // Constructor / Destructor
    // Assumes N and anisoDirection are normalized
    GlitterGGXCookTorranceBsdfLobe(const scene_rdl2::math::Vec3f &N, const scene_rdl2::math::Vec3f &flakeN,
                                   float roughness, const scene_rdl2::math::Color& favg);
    ~GlitterGGXCookTorranceBsdfLobe() {}

    // BsdfLobe API
    scene_rdl2::math::Color eval(const BsdfSlice &slice, const scene_rdl2::math::Vec3f &wi, float *pdf = NULL) const override;
    scene_rdl2::math::Color sample(const BsdfSlice &slice, float r1, float r2,
            scene_rdl2::math::Vec3f &wi, float &pdf) const override;

    bool getProperty(Property property, float *dest) const override;

    void show(std::ostream& os, const std::string& indent) const override
    {
        const scene_rdl2::math::Color& scale = getScale();
        const scene_rdl2::math::Vec3f& N = mFrame.getN();
        const Fresnel * fresnel = getFresnel();
        os << indent << "[GlitterGGXCookTorranceBsdfLobe]\n";
        os << indent << "    " << "scale: "
            << scale.r << " " << scale.g << " " << scale.b << "\n";
        os << indent << "    " << "N: "
            << N.x << " " << N.y << " " << N.z << "\n";
        if (fresnel) {
            fresnel->show(os, indent + "    ");
        }
    }

protected:
    scene_rdl2::math::Vec3f mFlakeNormal;
};


///
/// @class BerryCookTorranceBsdfLobe BsdfCookTorrance.h <shading/BsdfCookTorrance.h>
/// @brief The Cook-Torrance bsdf using Berry Distribution.
///        WARNING: This Bsdf is not energy preserving. Do not use!
/// 
class BerryCookTorranceBsdfLobe : public CookTorranceBsdfLobe
{
public:
    // Constructor / Destructor
    // Assumes N and anisoDirection are normalized
    BerryCookTorranceBsdfLobe(const scene_rdl2::math::Vec3f &N, float roughness);
    ~BerryCookTorranceBsdfLobe() {}

    // BsdfLobe API
    scene_rdl2::math::Color eval(const BsdfSlice &slice, const scene_rdl2::math::Vec3f &wi, float *pdf = NULL) const override;
    scene_rdl2::math::Color sample(const BsdfSlice &slice, float r1, float r2,
            scene_rdl2::math::Vec3f &wi, float &pdf) const override;

    void show(std::ostream& os, const std::string& indent) const override
    {
        const scene_rdl2::math::Color& scale = getScale();
        const scene_rdl2::math::Vec3f& N = mFrame.getN();
        const Fresnel * fresnel = getFresnel();
        os << indent << "[BerryCookTorranceBsdfLobe]\n";
        os << indent << "    " << "scale: "
            << scale.r << " " << scale.g << " " << scale.b << "\n";
        os << indent << "    " << "N: "
            << N.x << " " << N.y << " " << N.z << "\n";
        if (fresnel) {
            fresnel->show(os, indent + "    ");
        }
    }
};


///
/// @class AnisoCookTorranceBsdfLobe  BsdfCookTorrance.h <shading/BsdfCookTorrance.h>
/// @brief The Anisotropic Beckmann Cook-Torrance distribution from the paper:
///        "Understanding the Masking-Shadowing Function in Microfacet-Based BRDFs"
///        Journal of Computer Graphics Techniques, JCGT, 2014
class AnisoCookTorranceBsdfLobe : public CookTorranceBsdfLobe
{
public:
    // Constructor / Destructor
    // Assumes N and anisoDirection are normalized
    AnisoCookTorranceBsdfLobe(const scene_rdl2::math::Vec3f &N, const scene_rdl2::math::Vec3f &anisoDirection,
            float roughnessU, float roughnessV);
    ~AnisoCookTorranceBsdfLobe() {}

    // BsdfLobe API
    scene_rdl2::math::Color eval(const BsdfSlice &slice, const scene_rdl2::math::Vec3f &wi, float *pdf = NULL) const override;
    scene_rdl2::math::Color sample(const BsdfSlice &slice, float r1, float r2,
            scene_rdl2::math::Vec3f &wi, float &pdf) const override;
    bool getProperty(Property property, float *dest) const override;

    void show(std::ostream& os, const std::string& indent) const override
    {
        const scene_rdl2::math::Color& scale = getScale();
        const scene_rdl2::math::Vec3f& N = mFrame.getN();
        const Fresnel * fresnel = getFresnel();
        os << indent << "[AnisoCookTorranceBsdfLobe]\n";
        os << indent << "    " << "scale: "
            << scale.r << " " << scale.g << " " << scale.b << "\n";
        os << indent << "    " << "N: "
            << N.x << " " << N.y << " " << N.z << "\n";
        os << indent << "    " << "roughness^2 X: " << mRoughness  << "\n";
        os << indent << "    " << "roughness^2 Y: " << mRoughnessV << "\n";
        if (fresnel) {
            fresnel->show(os, indent + "    ");
        }
    }
protected:
    float mInputRoughnessV;
    float mRoughnessV;
};

//----------------------------------------------------------------------------

ISPC_UTIL_TYPEDEF_STRUCT(AnisoCookTorranceBsdfLobe, AnisoCookTorranceBsdfLobev);
ISPC_UTIL_TYPEDEF_STRUCT(BerryCookTorranceBsdfLobe, BerryCookTorranceBsdfLobev);
ISPC_UTIL_TYPEDEF_STRUCT(CookTorranceBsdfLobe, CookTorranceBsdfLobev);
ISPC_UTIL_TYPEDEF_STRUCT(GGXCookTorranceBsdfLobe, GGXCookTorranceBsdfLobev);
ISPC_UTIL_TYPEDEF_STRUCT(TransmissionCookTorranceBsdfLobe, TransmissionCookTorranceBsdfLobev);

} // namespace shading
} // namespace moonray

