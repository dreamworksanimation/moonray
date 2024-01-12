// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

///
/// @file BsdfHairLobes.h
/// $Id$
///

#pragma once

#include "BsdfHair.h"

#include <scene_rdl2/common/math/Color.h>
#include <scene_rdl2/common/math/Vec2.h>
#include <scene_rdl2/common/math/Vec3.h>

namespace moonray {
namespace shading {


///
/// @class HairLobe BsdfHairLobes.h <pbr/bsdf/hair/BsdfHairLobes.h>
/// @brief Energy Conserving Hair Model based on the following papers:
/// [1] An Energy-conserving Hair Reflectance Model - D'eon et al Sig'11
/// [2] A Practical and Controllable Hair and Fur Model for Production Path Tracing - Chiang et al EGSR'16
/// [3] Importance Sampling for Physically-Based Hair Fiber Models, D'eon et al Sig'13

///
/// @class HairRLobe BsdfHair.h <pbr/BsdfHairR.h>
/// @brief The hair reflection (R) lobe.
/// 
class HairRLobe : public HairBsdfLobe
{
public:
    // Constructor / Destructor. Note that shift and width are in radians
    HairRLobe(const scene_rdl2::math::Vec3f &hairDir,
              const scene_rdl2::math::Vec2f &hairUV,
              const float mediumIOR,
              const float ior,
              ispc::HairFresnelType fresnelType,
              const float layerThickness,
              const float longShift,
              const float longRoughness,
              const scene_rdl2::math::Color &tint = scene_rdl2::math::sWhite) :
                  HairBsdfLobe(Type(REFLECTION | GLOSSY),
                               hairDir,
                               hairUV,
                               mediumIOR,
                               ior,
                               fresnelType,
                               layerThickness,
                               longShift,
                               longRoughness,
                               0.75f, // Not actually used by R lobe, but set to match pre shading_api value
                               scene_rdl2::math::sWhite,
                               scene_rdl2::math::sWhite,
                               tint)
    {}

    ~HairRLobe() {}

    // Override Fresnel
    scene_rdl2::math::Color fresnel(const HairState&, const float) const override;

    void show(std::ostream& os, const std::string& indent) const override
    {
        os << indent << "[HairRLobe]\n";
        // use parent class func to print all members
        HairBsdfLobe::show(os, indent);

    }
};

///
/// @class HairTRTLobe BsdfHair.h <pbr/BsdfHair.h>
/// @brief The hair secondary/internal reflection (TRT) lobe.
///
class HairTRTLobe : public HairBsdfLobe
{
public:
    // Constructor / Destructor
    HairTRTLobe(const scene_rdl2::math::Vec3f &hairDir,
                const scene_rdl2::math::Vec2f &hairUV,
                const float mediumIOR,
                const float ior,
                ispc::HairFresnelType fresnelType,
                const float layerThickness,
                const float longShift,
                const float longRoughness,
                const scene_rdl2::math::Color &hairColor,
                const scene_rdl2::math::Color &hairSigmaA,
                const scene_rdl2::math::Color &tint = scene_rdl2::math::sWhite,
                const bool showGlint = false,
                const float glintRoughness = 0.5f,
                const float glintEccentricity = 1.0f,
                const float glintSaturation = 1.0f,
                const float hairRotation = 0.f,
                const scene_rdl2::math::Vec3f &hairNormal = scene_rdl2::math::Vec3f(scene_rdl2::math::zero)) :
                    HairBsdfLobe(Type(REFLECTION | GLOSSY),
                                 hairDir,
                                 hairUV,
                                 mediumIOR,
                                 ior,
                                 fresnelType,
                                 layerThickness,
                                 longShift,
                                 longRoughness,
                                 0.75f,
                                 hairColor,
                                 hairSigmaA,
                                 tint,
                                 hairRotation,
                                 hairNormal),
                     mShowGlint(showGlint),
                     mGlintRoughness(glintRoughness),
                     mGlintEccentricity(glintEccentricity),
                     mGlintSaturation(glintSaturation)
    {}
    ~HairTRTLobe() {}

    // Override Fresnel and Absorption Definitions
    scene_rdl2::math::Color fresnel(const HairState&, const float) const override;
    scene_rdl2::math::Color absorption(const HairState&) const override;

    // Override N term with absorption
    scene_rdl2::math::Color evalNTermWithAbsorption(const HairState&) const override;

    void show(std::ostream& os, const std::string& indent) const override
    {
        os << indent << "[HairTRTLobe]\n";
        // use parent class func to print all members
        HairBsdfLobe::show(os, indent);
        os << indent << "    " << "show glint: " << mShowGlint << "\n";
        os << indent << "    " << "glint roughness: " << mGlintRoughness << "\n";
        os << indent << "    " << "glint eccentricity: " << mGlintEccentricity << "\n";
        os << indent << "    " << "glint saturation: " << mGlintSaturation << "\n";
    }

private:
    scene_rdl2::math::Color glintAbsorption(const HairState& hairState,
                                const float etaStar,
                                const float etaP) const;
    float evalNTermTRT(const HairState& hairState) const;
    // This function evaluates the glint component of the TRT lobe
    // and returns a weight for the remaining TRT response which
    // is dependent on the glint response
    float evalNTermGlint(const HairState& hairState,
                         float& etaStar,
                         float& etaP,
                         float& trtWeight) const;
    bool mShowGlint;
    float mGlintRoughness;
    float mGlintEccentricity;
    float mGlintSaturation;
};

///
/// @class HairTTLobe BsdfHair.h <pbr/BsdfHair.h>
/// @brief The hair brdf transmission (TT) lobe.
///
class HairTTLobe : public HairBsdfLobe
{
public:
    // Constructor / Destructor
    HairTTLobe(const scene_rdl2::math::Vec3f &hairDir,
               const scene_rdl2::math::Vec2f &hairUV,
               const float mediumIOR,
               const float ior,
               ispc::HairFresnelType fresnelType,
               const float layerThickness,
               const float longShift,
               const float longRoughness,
               const float azimuthalRoughness,
               const scene_rdl2::math::Color &hairColor,
               const scene_rdl2::math::Color &hairSigmaA,
               const scene_rdl2::math::Color &tint = scene_rdl2::math::sWhite,
               const float saturationTT = 1.0f) :
                   HairBsdfLobe(Type(TRANSMISSION | GLOSSY),
                                hairDir,
                                hairUV,
                                mediumIOR,
                                ior,
                                fresnelType,
                                layerThickness,
                                longShift,
                                longRoughness,
                                azimuthalRoughness,
                                hairColor,
                                hairSigmaA,
                                tint),
                   mSaturation(saturationTT)
    {}

    ~HairTTLobe() {}

    // Override Fresnel and Absorption Definitions
    scene_rdl2::math::Color fresnel(const HairState&, const float) const override;
    scene_rdl2::math::Color absorption(const HairState&) const override;

    // Override N term
    scene_rdl2::math::Color evalNTermWithAbsorption(const HairState&) const override;
    float evalPhiPdf(const HairState&) const override;
    float samplePhi(float  r2,
                    float  phiO,
                    float& phiI) const override;

    bool getProperty(Property property,
                     float *dest) const override;

    void show(std::ostream& os, const std::string& indent) const override
    {
        os << indent << "[HairTTLobe]\n";
        // use parent class func to print all members
        HairBsdfLobe::show(os, indent);
        os << indent << "    " << "transmission saturation: " << mSaturation << "\n";
    }

private:
    float evalNTerm(const HairState&) const;
    // MOONSHINE-1238
    // Shader trick to further saturate/desaturate Transmission
    void applySaturation(scene_rdl2::math::Color& c,
                         const HairState&) const;

private:
    float mSaturation;
};

///
/// @class HairTRRTLobe BsdfHair.h <pbr/BsdfHair.h>
/// @brief The energy compensation lobe mentioned in [2], Section 3.4
/// This lobe creates a geometric series from the events TRRT and following and adds them up.
class HairTRRTLobe : public HairBsdfLobe
{
public:
    // Constructor / Destructor
    HairTRRTLobe(const scene_rdl2::math::Vec3f &hairDir,
                 const scene_rdl2::math::Vec2f &hairUV,
                 const float mediumIOR,
                 const float ior,
                 ispc::HairFresnelType fresnelType,
                 const float layerThickness,
                 const float longRoughness,
                 const scene_rdl2::math::Color &hairColor,
                 const scene_rdl2::math::Color &hairSigmaA,
                 const scene_rdl2::math::Color &tint = scene_rdl2::math::sWhite) :
                     HairBsdfLobe(Type(REFLECTION | GLOSSY),
                                  hairDir,
                                  hairUV,
                                  mediumIOR,
                                  ior,
                                  fresnelType,
                                  layerThickness,
                                  0.0f,
                                  longRoughness,
                                  0.75f,
                                  hairColor,
                                  hairSigmaA,
                                  tint)
    {
    }
    ~HairTRRTLobe() {}

    // Override Fresnel Definition
    scene_rdl2::math::Color fresnel(const HairState&, const float) const override;

    // Override N term
    scene_rdl2::math::Color evalNTermWithAbsorption(const HairState&) const override;
    float evalPhiPdf(const HairState&) const override;
    float samplePhi(float  r2,
                    float  phiO,
                    float& phiI) const override;

    void show(std::ostream& os, const std::string& indent) const override
    {
        os << indent << "[HairTRRTLobe]\n";
        // use parent class func to print all members
        HairBsdfLobe::show(os, indent);
    }

private:
    float evalNTerm(const HairState&) const;
    // TRRT Compensation Factor
    // [2], Section 3.4
    scene_rdl2::math::Color compensationFactor(const scene_rdl2::math::Color& f,
                                               const scene_rdl2::math::Color& oneMinusF,
                                               const scene_rdl2::math::Color& absorption) const;

    friend class HairOneSampleLobe;
};

} // namespace shading
} // namespace moonray


