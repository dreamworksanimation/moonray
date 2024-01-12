// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

///
/// @file CookTorranceEnergyCompensation.h
/// $Id$
///

/*
 * This file is used to compensate for energy loss from
 * microfacet reflection lobes as outlined in Walter'07.
 * We use the Kelemen'01 energy compensation BRDF as described in
 * Kulla'17 to be added to the regular microfacet BRDF.
 * This file describes the term to be added in reflection
 * to be perfectly energy 'preserving'. It also provides
 * sampling curves, PDF and a one-sample weight to sample
 * this term efficiently.
 * Ref:
 * Revisiting Physically Based Shading, PBS Kulla'17
 */

#pragma once

#include <scene_rdl2/common/math/Color.h>
#include <scene_rdl2/common/math/Vec3.h>
#include <scene_rdl2/common/math/ReferenceFrame.h>
#include <moonray/rendering/shading/ispc/BsdfComponent_ispc_stubs.h>

namespace moonray {
namespace shading {

//----------------------------------------------------------------------------
class CookTorranceEnergyCompensation
{
public:

    // Energy Compensation
    static scene_rdl2::math::Color eval(
            ispc::MicrofacetDistribution type,
            const float cosNO,
            const float cosNI,
            float inputRoughness,
            const scene_rdl2::math::Color& favg,
            bool includeCosineTerm);
    static void sample(
            ispc::MicrofacetDistribution type,
            const scene_rdl2::math::ReferenceFrame& frame,
            float r1, float r2,
            float roughness,
            scene_rdl2::math::Vec3f& wi);
    static float pdf(
            ispc::MicrofacetDistribution type,
            float cosNI,
            float roughness);
    static float weight(
            float roughness);
};

//----------------------------------------------------------------------------

} // namespace shading
} // namespace moonray

