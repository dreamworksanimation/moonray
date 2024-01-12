// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

///
/// @file BsdfHair.h
/// $Id$
///

#pragma once

#include <moonray/rendering/shading/bsdf/Bsdf.h>

//#include <moonray/rendering/shading/ispc/bsdf/hair/BsdfHair_ispc_stubs.h>

#include <scene_rdl2/common/math/Color.h>
#include <scene_rdl2/common/math/Vec3.h>

namespace moonray {
namespace shading {

//----------------------------------------------------------------------------
///
/// @class HairDiffuseLobe BsdfHair.h <shading/BsdfHair.h>
/// @brief A BsdfLobe class made to define the hair diffuse component
///        using the simple Kajiya-Kay diffuse.
///        See "Rendering Fur with Three-Dimensional Textures"
///         Kajiya, J.T. and Kay, T. L., SIGGRAPH 1989 proceedings pp 271-280
/// 
class HairDiffuseLobe : public BsdfLobe
{
public:
    // Constructor / Destructor
    HairDiffuseLobe(const scene_rdl2::math::Vec3f& hairDir,
                    const scene_rdl2::math::Color& colorRefl,
                    const scene_rdl2::math::Color& colorTrans) :
        BsdfLobe(Type(REFLECTION | DIFFUSE),
                 DifferentialFlags(IGNORES_INCOMING_DIFFERENTIALS),
                 true,
                 PROPERTY_ROUGHNESS | PROPERTY_NORMAL |
                 PROPERTY_COLOR | PROPERTY_PBR_VALIDITY),
        mHairDir(hairDir),
        mHairColorRefl(colorRefl),
        mHairColorTrans(colorTrans),
        mMaxDiffAngle(scene_rdl2::math::sHalfPi)
    {
        setIsHair(true);
    }
    ~HairDiffuseLobe() {}

    // BsdfLobe API functions
    scene_rdl2::math::Color albedo(const BsdfSlice &slice) const override;
    scene_rdl2::math::Color eval(const BsdfSlice &slice, const scene_rdl2::math::Vec3f &wi, float *pdf = NULL) const override;
    scene_rdl2::math::Color sample(const BsdfSlice &slice, float r1, float r2,
            scene_rdl2::math::Vec3f &wi, float &pdf) const override;

    void differentials(const scene_rdl2::math::Vec3f &wo, const scene_rdl2::math::Vec3f &wi,
                       float r1, float r2, const scene_rdl2::math::Vec3f &dNdx, const scene_rdl2::math::Vec3f &dNdy,
                       scene_rdl2::math::Vec3f &dDdx, scene_rdl2::math::Vec3f &dDdy) const override;

    virtual bool getProperty(Property property, float *dest) const override;

    void show(std::ostream& os, const std::string& indent) const override
    {
        const scene_rdl2::math::Color& scale = getScale();
        const Fresnel * fresnel = getFresnel();
        os << indent << "[HairDiffuseLobe]\n";
        os << indent << "    " << "scale: "
            << scale.r << " " << scale.g << " " << scale.b << "\n";
        os << indent << "    " << "refl color: "
            << mHairColorRefl.r << " " << mHairColorRefl.g << " " << mHairColorRefl.b << "\n";
        os << indent << "    " << "trans color: "
            << mHairColorTrans.r << " " << mHairColorTrans.g << " " << mHairColorTrans.b << "\n";
        if (fresnel) {
            fresnel->show(os, indent + "    ");
        }
    }

protected:
    float evalPdf(const scene_rdl2::math::Vec3f &wi) const;

    virtual scene_rdl2::math::Color evalBsdf(const scene_rdl2::math::Vec3f& wo,
                                             const scene_rdl2::math::Vec3f &wi,
                                             bool includeCosineTerm) const;

    scene_rdl2::math::Vec3f mHairDir;

    // Reflection and Transmission colors
    scene_rdl2::math::Color mHairColorRefl;
    scene_rdl2::math::Color mHairColorTrans;
    float mMaxDiffAngle;
};

} // namespace shading
} // namespace moonray


