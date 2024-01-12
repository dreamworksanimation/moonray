// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

///
/// @file BsdfHairOneSampler.h
/// $Id$
///

#pragma once

#include "BsdfHairLobes.h"
#include <moonray/rendering/shading/ispc/bsdf/hair/BsdfHair_ispc_stubs.h>

#include <scene_rdl2/common/math/Color.h>
#include <scene_rdl2/common/math/Vec2.h>
#include <scene_rdl2/common/math/Vec3.h>

namespace moonray {
namespace shading {

///
/// @class HairLobe BsdfHair.h <shading/bsdf/hair/BsdfHair.h>
/// @brief An Energy Conserving Hair Model based on the following papers:
/// [1] An Energy-conserving Hair Reflectance Model - D'eon et al Sig'11
/// [2] A Practical and Controllable Hair and Fur Model for Production Path Tracing - Chiang et al EGSR'16
/// [3] Importance Sampling for Physically-Based Hair Fiber Models, D'eon et al Sig'13

class HairOneSampleLobe : public HairBsdfLobe
{
public:
    HairOneSampleLobe(const scene_rdl2::math::Vec3f& hairDir,
                      const scene_rdl2::math::Vec2f& hairUV,
                      const float mediumIOR,
                      const float ior,
                      const ispc::HairFresnelType fresnelType,
                      const float layerThickness,
                      bool  showR,
                      float shiftR,
                      float roughnessR,
                      const scene_rdl2::math::Color& tintR,
                      bool  showTT,
                      float shiftTT,
                      float roughnessTT,
                      float azimuthalRoughnessTT,
                      const scene_rdl2::math::Color& tintTT,
                      float saturationTT,
                      bool  showTRT,
                      float shiftTRT,
                      float roughnessTRT,
                      const scene_rdl2::math::Color& tintTRT,
                      bool  showGlint,
                      float roughnessGlint,
                      float eccentricityGlint,
                      float saturationGlint,
                      const float hairRotation,
                      const scene_rdl2::math::Vec3f& hairNormal,
                      bool  showTRRT,
                      const scene_rdl2::math::Color &hairColor,
                      const scene_rdl2::math::Color &hairSigmaA):
        HairBsdfLobe(Type(REFLECTION | TRANSMISSION | GLOSSY),
                     hairDir,
                     hairUV,
                     mediumIOR,
                     ior,
                     fresnelType,
                     layerThickness,
                     0.1f, //longShift
                     0.1f, //longRoughness
                     0.1f, //azimRoughness
                     hairColor,
                     hairSigmaA,
                     scene_rdl2::math::sWhite,
                     hairRotation,
                     hairNormal),
        mShowR(showR),
        mShowTT(showTT),
        mShowTRT(showTRT),
        mShowTRRT(showTRRT),
        rLobe(hairDir, hairUV, mediumIOR, ior, fresnelType, layerThickness,
              shiftR, roughnessR, tintR),
        ttLobe(hairDir, hairUV, mediumIOR, ior, fresnelType, layerThickness,
               shiftTT, roughnessTT, azimuthalRoughnessTT, hairColor, hairSigmaA, tintTT, saturationTT),
        trtLobe(hairDir, hairUV, mediumIOR, ior, fresnelType, layerThickness,
                shiftTRT, roughnessTRT, hairColor, hairSigmaA, tintTRT, showGlint,
                roughnessGlint, eccentricityGlint, saturationGlint, hairRotation, hairNormal),
        trrtLobe(hairDir, hairUV, mediumIOR, ior, fresnelType, layerThickness,
                 1.0f, hairColor, hairSigmaA)
    {}

    virtual ~HairOneSampleLobe() {}

    scene_rdl2::math::Color eval(const BsdfSlice &slice,
                     const scene_rdl2::math::Vec3f &wi,
                     float *pdf = NULL) const override;

    scene_rdl2::math::Color sample(const BsdfSlice &slice,
                       float r1, float r2,
                       scene_rdl2::math::Vec3f &wi,
                       float &pdf) const override;

    bool getProperty(Property property,
                     float *dest) const override;

    void show(std::ostream& os, const std::string& indent) const override
    {
        const scene_rdl2::math::Color& scale = getScale();
        const Fresnel * fresnel = getFresnel();
        os << indent << "[HairOneSampleLobe]\n";
        os << indent << "    " << "scale: "
            << scale.r << " " << scale.g << " " << scale.b << "\n";
        const std::string indentMore(indent + "    ");
        if (fresnel) {
            fresnel->show(os, indentMore);
        }
        // show each sub lobe
        rLobe.show(os, indentMore);
        ttLobe.show(os, indentMore);
        trtLobe.show(os, indentMore);
        trrtLobe.show(os, indentMore);
    }

protected:

    // Override Base Hair Functions for PDF and BSDF
    float evalPdf(const HairState& hairState) const override;
    scene_rdl2::math::Color evalBsdf(const HairState& hairState,
                         bool includeCosineTerm = true) const override;

private:
    bool mShowR, mShowTT, mShowTRT, mShowTRRT;

    HairRLobe    rLobe;
    HairTTLobe   ttLobe;
    HairTRTLobe  trtLobe;
    HairTRRTLobe trrtLobe;

    std::array<float, 3> mSampleProbabilities = { {0.0f, 0.0f, 0.0f} };
    std::array<float, 3> mCDF = { {0.0f, 0.0f, 0.0f} };

    float evalPdf(const HairState& hairState,
                  const float (&weights)[3]) const;

    void calculateSamplingWeightsAndCDF(const HairState& hairState,
                                        float (&weights)[3],
                                        float (&cdf)[3]) const;
};

} // namespace shading
} // namespace moonray


