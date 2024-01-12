// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

///
/// @file BsdfUnderClearcoat.h
/// $Id$
///

#pragma once

#include "BsdfUnderClearcoat.h"

#include <scene_rdl2/common/math/Color.h>
#include <scene_rdl2/common/math/Vec3.h>

namespace moonray {
namespace shading {

///
/// @class UnderClearcoatTransmissionBsdfLobe BsdfUnderClearcoat.h <pbr/BsdfUnderClearcoat.h>
/// @brief An "under" bsdf lobe adapter for transmission, which bends omega_o rays on refraction,
/// evaluates the under lobe using refracted ray and also attenuates the lobe it wraps
/// according to the distance traveled in clearcoat
///
class UnderClearcoatTransmissionBsdfLobe : public UnderClearcoatBsdfLobe
{
public:
    // Constructor / Destructor
    UnderClearcoatTransmissionBsdfLobe(BsdfLobe *under,
                                       scene_rdl2::alloc::Arena* arena,
                                       const scene_rdl2::math::Vec3f &N,
                                       const float etaI,
                                       const float etaT,
                                       const float thickness,
                                       const scene_rdl2::math::Color &attenuationColor,
                                       const float attenuationWeight)
        : UnderClearcoatBsdfLobe(under, arena, N,
                                 etaI, etaT,
                                 thickness,
                                 attenuationColor,
                                 attenuationWeight,
                                 true)
        {}

    // Override eval and sample
    scene_rdl2::math::Color eval(const BsdfSlice &slice,
                     const scene_rdl2::math::Vec3f &wi,
                     float *pdf = NULL) const override;

    scene_rdl2::math::Color sample(const BsdfSlice &slice,
                       float r1, float r2,
                       scene_rdl2::math::Vec3f &wi,
                       float &pdf) const override;

    void show(std::ostream& os, const std::string& indent) const override
    {
        const scene_rdl2::math::Color& scale = getScale();
        const Fresnel * fresnel = getFresnel();
        os << indent << "[UnderClearcoatTransmissionBsdfLobe]\n";
        os << indent << "    " << "scale: "
            << scale.r << " " << scale.g << " " << scale.b << "\n";
        mUnder->show(os, indent + "    ");
        if (fresnel) {
            fresnel->show(os, indent + "    ");
        }
    }

};

} // namespace shading
} // namespace moonray

