// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

///
/// @file BsdfUnderClearcoat.h
/// $Id$
///

#pragma once

#include "BsdfUnder.h"

#include <scene_rdl2/common/math/Color.h>
#include <scene_rdl2/common/math/Vec3.h>

namespace moonray {
namespace shading {

///
/// @class UnderClearcoatBsdfLobe BsdfUnderClearcoat.h <pbr/BsdfUnderClearcoat.h>
/// @brief An "under" bsdf lobe adapter, which *bends* the rays on refraction,
/// evaluates the under lobe using refracted rays and also attenuates the lobe it wraps
/// according to the distance traveled in clearcoat
///
class UnderClearcoatBsdfLobe : public UnderBsdfLobe
{
public:
    // Constructor / Destructor
    UnderClearcoatBsdfLobe(BsdfLobe *under,
                           scene_rdl2::alloc::Arena* arena,
                           const scene_rdl2::math::Vec3f &N,
                           const float etaI,
                           const float etaT,
                           const float thickness,
                           const scene_rdl2::math::Color &attenuationColor,
                           const float attenuationWeight,
                           const bool  passThroughTIRWhenSampling = true)
        : UnderBsdfLobe(under, N, thickness, attenuationColor, attenuationWeight)
        , mExitingFresnel(arena->allocWithArgs<DielectricFresnel>(etaT, etaI))
        , mNeta(etaI * scene_rdl2::math::rcp(etaT))
        , mPassThroughTIRWhenSampling(passThroughTIRWhenSampling)
        {}

    // BsdfLobe API
    scene_rdl2::math::Color eval(const BsdfSlice &slice,
                     const scene_rdl2::math::Vec3f &wi,
                     float *pdf = NULL) const override;

    scene_rdl2::math::Color sample(const BsdfSlice &slice,
                       float r1, float r2,
                       scene_rdl2::math::Vec3f &wi,
                       float &pdf) const override;

    finline scene_rdl2::math::Color albedo(const BsdfSlice &slice) const override
    {
        // forward to base class
        return UnderBsdfLobe::albedo(slice);
    }

    finline void differentials(const scene_rdl2::math::Vec3f &wo, const scene_rdl2::math::Vec3f &wi,
            float r1, float r2, const scene_rdl2::math::Vec3f &dNdx, const scene_rdl2::math::Vec3f &dNdy,
            scene_rdl2::math::Vec3f &dDdx, scene_rdl2::math::Vec3f &dDdy) const override
    {
        // forward to base class
        UnderBsdfLobe::differentials(wo, wi, r1, r2, dNdx, dNdy, dDdx, dDdy);
    }

    bool getProperty(Property property, float *dest) const override
    {
        // forward to base class
        return UnderBsdfLobe::getProperty(property, dest);
    }

    void show(std::ostream& os, const std::string& indent) const override
    {
        const scene_rdl2::math::Color& scale = getScale();
        const Fresnel * fresnel = getFresnel();
        os << indent << "[UnderClearcoatBsdfLobe]\n";
        os << indent << "    " << "scale: "
            << scale.r << " " << scale.g << " " << scale.b << "\n";
        os << indent << "    " << "N: "
            << mN.x << " " << mN.y << " " << mN.z << "\n";
        os << indent << "    " << "neta: " << mNeta << "\n";
        os << indent << "    " << "thickness: " << mThickness << "\n";
        os << indent << "    " << "attenuation color: "
            << mAttenuationColor.r << " " << mAttenuationColor.g << " " << mAttenuationColor.b << "\n";
        os << indent << "    " << "pass through TIR when sampling: "
            << (mPassThroughTIRWhenSampling ? "true" : "false") << "\n";

        mUnder->show(os, indent + "    ");

        if (fresnel) {
            fresnel->show(os, indent + "    ");
        }
    }

protected:
    finline static scene_rdl2::math::Color maxColor(const scene_rdl2::math::Color& c, const float x)
    {
        return scene_rdl2::math::Color(scene_rdl2::math::max(c.r, x),
                                       scene_rdl2::math::max(c.g, x),
                                       scene_rdl2::math::max(c.b, x));
    }

    // Fresnel Object that reverses the IORs for rays exiting Clearcoat
    Fresnel* mExitingFresnel;
    // Ratio of etaI/etaT
    float mNeta;

    // When sampling the Under lobe through the Clearcoat interface,
    // it is very likely to run into situations where the "sampled" vector
    // from the under lobe results in TIR when exiting the Clearcoat interface and
    // might have to be discarded. At this point, in lieu of generating another sample
    // until we find one that exits, or discarding this sample entirely,
    // we assume that the ray undergoes TIR through the interface, bounces around
    // multiple times and eventually comes out through the interface in the same
    // direction. This helps energy conservation and stops the under lobe from
    // going too dark because of discarded samples.

    // Note that we disable this option when unit testing the UnderClearcoat interface
    // to ensure the consistency tests pass for the sample() and eval() functions.
    bool mPassThroughTIRWhenSampling;
};

} // namespace shading
} // namespace moonray

