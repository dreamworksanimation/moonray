// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

///
/// @file BsdfFabric.h
/// $Id$
///

#pragma once

#include <moonray/rendering/shading/bsdf/Bsdf.h>
#include <moonray/rendering/shading/bsdf/BsdfSlice.h>
#include <moonray/rendering/shading/Util.h>

#include <moonray/rendering/shading/ispc/bsdf/fabric/BsdfFabric_ispc_stubs.h>
#include <moonray/rendering/shading/ispc/bsdf/fabric/BsdfFabricVelvet_ispc_stubs.h>

#include <scene_rdl2/common/math/Color.h>
#include <scene_rdl2/common/math/ReferenceFrame.h>
#include <scene_rdl2/common/math/Vec3.h>
#include <scene_rdl2/common/math/MathUtil.h>

namespace moonray {
namespace shading {

//----------------------------------------------------------------------------
///
/// @class FabricBsdfLobe
/// @brief The fabric BSDF Interface common functions, inherit from this
///
class FabricBsdfLobe : public BsdfLobe
{
protected:
    // Constructor / Destructor
    FabricBsdfLobe(const scene_rdl2::math::Vec3f &N,
                   const scene_rdl2::math::Vec3f &T,
                   const scene_rdl2::math::Vec3f &threadDirection,
                   const float threadElevation,
                   const float roughness,
                   const scene_rdl2::math::Color& color = scene_rdl2::math::sWhite);

    ~FabricBsdfLobe() {}

    scene_rdl2::math::Color eval(const BsdfSlice &slice,
                     const scene_rdl2::math::Vec3f &wi,
                     float *pdf = NULL) const override
    {
        return scene_rdl2::math::sBlack;
    }
    scene_rdl2::math::Color sample(const BsdfSlice &slice,
                       float r1, float r2,
                       scene_rdl2::math::Vec3f &wi,
                       float &pdf) const override
    {
        return scene_rdl2::math::sBlack;
    }

    void differentials(const scene_rdl2::math::Vec3f &wo, const scene_rdl2::math::Vec3f &wi,
                       float r1, float r2,
                       const scene_rdl2::math::Vec3f &dNdx, const scene_rdl2::math::Vec3f &dNdy,
                       scene_rdl2::math::Vec3f &dDdx, scene_rdl2::math::Vec3f &dDdy) const override;
    scene_rdl2::math::Color albedo(const BsdfSlice &slice) const override;
    bool getProperty(Property property, float *dest) const override;

    void show(std::ostream& os, const std::string& indent) const override
    {
        const scene_rdl2::math::Color& scale = getScale();
        const Fresnel * fresnel = getFresnel();
        os << indent << "[FabricBsdfLobe]\n";
        os << indent << "    " << "scale: "
            << scale.r << " " << scale.g << " " << scale.b << "\n";
        os << indent << "    " << "color: "
            << mColor.r << " " << mColor.g << " " << mColor.b << "\n";
        os << indent << "    " << "roughness: " << mRoughness << "\n";
        os << indent << "    " << "specular exponent: " << mSpecularExponent << "\n";
        os << indent << "    " << "thread direction: "
            << mThreadDirection.x << " " << mThreadDirection.y << " " << mThreadDirection.z << "\n";
        os << indent << "    " << "normalization factor: " << mNormalizationFactor << "\n";

        if (fresnel) {
            fresnel->show(os, indent + "    ");
        }
    }

protected:
    // Derive a directional differential scale that varies according to roughness
    // TODO - need to derive factors similar to the hair lobe
    static constexpr float sdDFactorMin = 1.0f;
    static constexpr float sdDFactorMax = 8.0f;
    static constexpr float sdDFactorSlope = (sdDFactorMax - sdDFactorMin);

    scene_rdl2::math::ReferenceFrame mFrame;
    float mRoughness;
    float mSpecularExponent;

    float mdDFactor;
    float mNormalizationFactor;

    scene_rdl2::math::Vec3f mThreadDirection;

    scene_rdl2::math::Color mColor;

protected:
    // Initialize the thread direction vector for fabric
    // 'thread elevation' is used to rotate the fabric thread vertically
    void calculateThreadDirection(const scene_rdl2::math::Vec3f& threadDirection,
                                  float threadElevation);

    inline float safeASin(float sinTheta) const
    {
        // Clamp to slightly less than the max range to avoid NaNs
        sinTheta = scene_rdl2::math::clamp(sinTheta, -.99f, .99f);
        return scene_rdl2::math::asin(sinTheta);
    }
    inline float safeACos(float cosTheta) const
    {
        // Clamp to slightly less than the max range to avoid NaNs
        cosTheta = scene_rdl2::math::clamp(cosTheta, -.99f, .99f);
        return scene_rdl2::math::acos(cosTheta);
    }
};

///
/// @class DwaFabricBsdfLobe BsdfFabric.h <shading/fabric/BsdfFabric.h>
/// @brief The DWA fabric specular lobe (1 - |sin(thetaH)|)^N
/// thetaH = 0.5*(thetaO+thetaI), theta wrt normal plane to hair tangent
///
class DwaFabricBsdfLobe : public FabricBsdfLobe
{
public:
    DwaFabricBsdfLobe(const scene_rdl2::math::Vec3f &N,
                      const scene_rdl2::math::Vec3f &T,
                      const scene_rdl2::math::Vec3f &threadDirection,
                      const float threadElevation,
                      const float roughness,
                      const scene_rdl2::math::Color& color = scene_rdl2::math::sWhite);
    ~DwaFabricBsdfLobe() override {}

    // BsdfLobe API
    scene_rdl2::math::Color eval(const BsdfSlice &slice,
                     const scene_rdl2::math::Vec3f &wi,
                     float *pdf = NULL) const override;
    scene_rdl2::math::Color sample(const BsdfSlice &slice,
                       float r1, float r2,
                       scene_rdl2::math::Vec3f &wi, float &pdf) const override;

    void show(std::ostream& os, const std::string& indent) const override
    {
        const scene_rdl2::math::Color& scale = getScale();
        const Fresnel * fresnel = getFresnel();
        os << indent << "[DwaFabricBsdfLobe]\n";
        os << indent << "    " << "scale: "
            << scale.r << " " << scale.g << " " << scale.b << "\n";
        os << indent << "    " << "color: "
            << mColor.r << " " << mColor.g << " " << mColor.b << "\n";
        os << indent << "    " << "roughness: " << mRoughness << "\n";
        os << indent << "    " << "specular exponent: " << mSpecularExponent << "\n";
        os << indent << "    " << "thread direction: "
            << mThreadDirection.x << " " << mThreadDirection.y << " " << mThreadDirection.z << "\n";
        os << indent << "    " << "normalization factor: " << mNormalizationFactor << "\n";

        if (fresnel) {
            fresnel->show(os, indent + "    ");
        }
    }

private:
    void calculateNormalizationFactor();

    void sampleVector(const scene_rdl2::math::Vec3f& wo,
                      float r1, float r2,
                      scene_rdl2::math::Vec3f& wi) const;
};

///
/// @class KajiyaKayFabricBsdfLobe BsdfFabric.h <shading/fabric/BsdfFabric.h>
/// @brief The KajiyaKay cosine specular lobe cos(thetaD)^N
/// thetaD = thetaO - thetaIPrime, theta wrt hair tangent
///
class KajiyaKayFabricBsdfLobe : public FabricBsdfLobe
{
public:
    KajiyaKayFabricBsdfLobe(const scene_rdl2::math::Vec3f &N,
                      const scene_rdl2::math::Vec3f &T,
                      const scene_rdl2::math::Vec3f &threadDirection,
                      const float threadElevation,
                      const float roughness);
    ~KajiyaKayFabricBsdfLobe() override {}

    // BsdfLobe API
    scene_rdl2::math::Color eval(const BsdfSlice &slice,
                     const scene_rdl2::math::Vec3f &wi,
                     float *pdf = NULL) const override;
    scene_rdl2::math::Color sample(const BsdfSlice &slice,
                       float r1, float r2,
                       scene_rdl2::math::Vec3f &wi, float &pdf) const override;

    void show(std::ostream& os, const std::string& indent) const override
    {
        const scene_rdl2::math::Color& scale = getScale();
        const Fresnel * fresnel = getFresnel();
        os << indent << "[KajiyaKayFabricBsdfLobe]\n";
        os << indent << "    " << "scale: "
            << scale.r << " " << scale.g << " " << scale.b << "\n";
        if (fresnel) {
            fresnel->show(os, indent + "    ");
        }
    }

private:
    void calculateNormalizationFactor();

    void sampleVector(const scene_rdl2::math::Vec3f& wo,
                      float r1, float r2,
                      scene_rdl2::math::Vec3f& wi) const;
};

//----------------------------------------------------------------------------
///
/// @class FabricVelvetBsdfLobe
/// @brief A lobe with granzing-angle velvet highlights
///
class FabricVelvetBsdfLobe : public BsdfLobe
{
public:
    // Constructor / Destructor
    FabricVelvetBsdfLobe(const scene_rdl2::math::Vec3f &N,
                         const float roughness,
                         const scene_rdl2::math::Color& color = scene_rdl2::math::sWhite);

    ~FabricVelvetBsdfLobe() override {}

    // BsdfLobe API
    scene_rdl2::math::Color eval(const BsdfSlice &slice,
                     const scene_rdl2::math::Vec3f &wi,
                     float *pdf = NULL) const override;
    scene_rdl2::math::Color sample(const BsdfSlice &slice,
                       float r1, float r2,
                       scene_rdl2::math::Vec3f &wi,
                       float &pdf) const override;

    void differentials(const scene_rdl2::math::Vec3f &wo, const scene_rdl2::math::Vec3f &wi,
                       float r1, float r2,
                       const scene_rdl2::math::Vec3f &dNdx, const scene_rdl2::math::Vec3f &dNdy,
                       scene_rdl2::math::Vec3f &dDdx, scene_rdl2::math::Vec3f &dDdy) const override;

    scene_rdl2::math::Color albedo(const BsdfSlice &slice) const override;

    bool getProperty(Property property, float *dest) const override
    {
        bool handled = true;
        switch (property)
        {
        case PROPERTY_COLOR:
        {
            *dest       = mColor[0];
            *(dest + 1) = mColor[1];
            *(dest + 2) = mColor[2];
            break;
        }
        case PROPERTY_PBR_VALIDITY:
        {
            // Same as Albedo PBR Validity
            scene_rdl2::math::Color res = computeAlbedoPbrValidity(mColor);
            *dest       = res.r;
            *(dest + 1) = res.g;
            *(dest + 2) = res.b;
            break;
        }
        case PROPERTY_ROUGHNESS:
        {
            *dest       = mRoughness;
            *(dest + 1) = mRoughness;
            break;
        }
        case PROPERTY_NORMAL:
            {
                const scene_rdl2::math::Vec3f &N = mFrame.getN();
                *dest       = N.x;
                *(dest + 1) = N.y;
                *(dest + 2) = N.z;
            }
            break;
        default:
            handled = BsdfLobe::getProperty(property, dest);
            break;
        }
        return handled;
    }

    void show(std::ostream& os, const std::string& indent) const override
    {
        const scene_rdl2::math::Color& scale = getScale();
        const Fresnel * fresnel = getFresnel();
        os << indent << "[FabricVelvetBsdfLobe]\n";
        os << indent << "    " << "scale: "
            << scale.r << " " << scale.g << " " << scale.b << "\n";
        os << indent << "    " << "color: "
            << mColor.r << " " << mColor.g << " " << mColor.b << "\n";
        os << indent << "    " << "roughness: " << mRoughness << "\n";
        os << indent << "    " << "specular exponent: " << mSpecularExponent << "\n";
        if (fresnel) {
            fresnel->show(os, indent + "    ");
        }
    }

protected:
    // Derive a directional differential scale that varies according to roughness
    static constexpr float sdDFactorMin = 1.0f;
    static constexpr float sdDFactorMax = 8.0f;
    static constexpr float sdDFactorSlope = (sdDFactorMax - sdDFactorMin);

    scene_rdl2::math::ReferenceFrame mFrame;
    scene_rdl2::math::Color mColor;
    float mRoughness;
    float mSpecularExponent;
    float mNormalizationFactor;

    float mdDFactor;
};

//----------------------------------------------------------------------------
ISPC_UTIL_TYPEDEF_STRUCT(FabricBsdfLobe, FabricBsdfLobev);
ISPC_UTIL_TYPEDEF_STRUCT(FabricVelvetBsdfLobe, FabricVelvetBsdfLobev);

} // namespace shading
} // namespace moonray

