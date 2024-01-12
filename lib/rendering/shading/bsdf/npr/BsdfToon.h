// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

///
/// @file BsdfToon.h
/// $Id$
///

// A simple cel-shading lobe with user controls for
// each threshold position, interpolator and color.

#pragma once

#include <moonray/rendering/shading/bsdf/BsdfLambert.h>
#include <moonray/rendering/shading/Util.h>
#include <moonray/rendering/shading/RampControl.h>
#include <moonray/rendering/shading/bsdf/cook_torrance/BsdfCookTorrance.h>
#include <moonray/rendering/shading/bsdf/hair/BsdfHairLobes.h>

#include <scene_rdl2/common/math/Color.h>
#include <scene_rdl2/common/math/ReferenceFrame.h>
#include <scene_rdl2/common/math/Vec3.h>
#include <scene_rdl2/common/platform/IspcUtil.h>

namespace moonray {
namespace shading {

//----------------------------------------------------------------------------
class HairToonSpecularBsdfLobe : public BsdfLobe
{
public:
    HairToonSpecularBsdfLobe(const scene_rdl2::math::Vec3f &N,
                             const float intensity,
                             const scene_rdl2::math::Color& tint,
                             int numRampPoints,
                             const float* rampPositions,
                             const ispc::RampInterpolatorMode* rampInterpolators,
                             const float* rampValues,
                             const bool enableIndirectReflections,
                             const float indirectReflectionsRoughness,
                             const float indirectReflectionsIntensity,
                             const scene_rdl2::math::Vec3f &hairDir,
                             const scene_rdl2::math::Vec2f &hairUV,
                             const float mediumIOR,
                             const float ior,
                             ispc::HairFresnelType fresnelType,
                             const float layerThickness,
                             const float longShift,
                             const float longRoughness);

    ~HairToonSpecularBsdfLobe() {}

    // BsdfLobe API
    scene_rdl2::math::Color eval(const BsdfSlice &slice,
                     const scene_rdl2::math::Vec3f &wi,
                     float *pdf = NULL) const override;

    scene_rdl2::math::Color sample(const BsdfSlice &slice,
                       float r1,
                       float r2,
                       scene_rdl2::math::Vec3f &wi,
                       float &pdf) const override;

    void differentials(const scene_rdl2::math::Vec3f &wo,
                       const scene_rdl2::math::Vec3f &wi,
                       float r1, float r2,
                       const scene_rdl2::math::Vec3f &dNdx,
                       const scene_rdl2::math::Vec3f &dNdy,
                       scene_rdl2::math::Vec3f &dDdx,
                       scene_rdl2::math::Vec3f &dDdy) const override;

    finline scene_rdl2::math::Color albedo(const BsdfSlice &slice) const override
    {
        // This approximation is the same a cook torrance
        float cosThetaWo = scene_rdl2::math::max(dot(mFrame.getN(), slice.getWo()), 0.0f);
        // TODO: Improve this approximation!
        return computeScaleAndFresnel(cosThetaWo);
    }

protected:
    void show(std::ostream& os, const std::string& indent) const override;

private:
    scene_rdl2::math::ReferenceFrame mFrame;
    float mIntensity;
    scene_rdl2::math::Color mTint;
    bool mEnableIndirectReflections;
    float mIndirectReflectionsIntensity;
    HairRLobe mDirectHairLobe;
    HairRLobe mIndirectHairLobe;
    FloatRampControl mRampControl;
};

class ToonSpecularBsdfLobe : public BsdfLobe
{
public:
    ToonSpecularBsdfLobe(const scene_rdl2::math::Vec3f &N,
                         const float intensity,
                         const scene_rdl2::math::Color& tint,
                         float rampInputScale,
                         int numRampPoints,
                         const float* rampPositions,
                         const ispc::RampInterpolatorMode* rampInterpolators,
                         const float* rampValues,
                         const float stretchU,
                         const float stretchV,
                         const scene_rdl2::math::Vec3f &dPds,
                         const scene_rdl2::math::Vec3f &dPdt,
                         const bool enableIndirectReflections,
                         const float indirectReflectionsRoughness,
                         const float indirectReflectionsIntensity,
                         Fresnel * fresnel);

    ~ToonSpecularBsdfLobe() {}

    // BsdfLobe API
    scene_rdl2::math::Color eval(const BsdfSlice &slice,
                     const scene_rdl2::math::Vec3f &wi,
                     float *pdf = NULL) const override;

    scene_rdl2::math::Color sample(const BsdfSlice &slice,
                       float r1,
                       float r2,
                       scene_rdl2::math::Vec3f &wi,
                       float &pdf) const override;

    void differentials(const scene_rdl2::math::Vec3f &wo,
                       const scene_rdl2::math::Vec3f &wi,
                       float r1, float r2,
                       const scene_rdl2::math::Vec3f &dNdx,
                       const scene_rdl2::math::Vec3f &dNdy,
                       scene_rdl2::math::Vec3f &dDdx,
                       scene_rdl2::math::Vec3f &dDdy) const override;

    bool getProperty(Property property, float *dest) const override;

    finline scene_rdl2::math::Color albedo(const BsdfSlice &slice) const override
    {
        // This approximation is the same a cook torrance
        float cosThetaWo = scene_rdl2::math::max(dot(mFrame.getN(), slice.getWo()), 0.0f);
        // TODO: Improve this approximation!
        return computeScaleAndFresnel(cosThetaWo);
    }

protected:
    void show(std::ostream& os, const std::string& indent) const override;

private:
    scene_rdl2::math::ReferenceFrame mFrame;
    FloatRampControl mRampControl;
    float mIntensity;
    scene_rdl2::math::Color mTint;
    float mRampInputScale;
    float mStretchU;
    float mStretchV;
    scene_rdl2::math::Vec3f mdPds;
    scene_rdl2::math::Vec3f mdPdt;
    bool mEnableIndirectReflections;
    float mIndirectReflectionsIntensity;
    CookTorranceBsdfLobe mIndirectLobe;
};

class ToonBsdfLobe : public LambertBsdfLobe
{
public:
    // Constructor / Destructor
    ToonBsdfLobe(const scene_rdl2::math::Vec3f &N,
                 const scene_rdl2::math::Color &albedo,
                 int numRampPoints,
                 const float* rampPositions,
                 const ispc::RampInterpolatorMode* rampInterpolators,
                 const scene_rdl2::math::Color* rampColors,
                 const bool extendRamp) :
        LambertBsdfLobe(N, albedo, true),
        mExtendRamp(extendRamp)
    {
        mRampControl.init(numRampPoints, rampPositions, rampColors, rampInterpolators,
                          ispc::COLOR_RAMP_CONTROL_SPACE_RGB);
    }

    finline scene_rdl2::math::Color eval(const BsdfSlice &slice,
                             const scene_rdl2::math::Vec3f &wi,
                             float *pdf = NULL) const override;

    void show(std::ostream& os, const std::string& indent) const override;

private: 
    ColorRampControl mRampControl;
    bool mExtendRamp;
};


//----------------------------------------------------------------------------
/* ISPC_UTIL_TYPEDEF_STRUCT(ToonBsdfLobe, ToonBsdfLobev); */

} // namespace shading
} // namespace moonray

