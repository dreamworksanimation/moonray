// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

///
///

#pragma once

#include "Bssrdf.h"

#include <moonray/rendering/shading/ispc/bssrdf/NormalizedDiffusion_ispc_stubs.h>
#include <scene_rdl2/common/math/Color.h>
#include <scene_rdl2/common/math/Vec3.h>

namespace moonray {
namespace shading {

ISPC_UTIL_TYPEDEF_STRUCT(NormalizedDiffusionBssrdf, NormalizedDiffusionBssrdfv);

//----------------------------------------------------------------------------

///
/// @class NormalizedDiffusionBssrdf NormalizedDiffusion.h <shading/bssrdf/NormalizedDiffusion.h>
/// @brief Derived class to implement the Normalized Diffusion Bssrdf with importance sampling.
///
/// This Bssrdf implements "Normalized Diffusion" approximation as described in:
/// [1] "Approximate Reflectance Profiles for Efficient Subsurface Scattering",
///     Burley Christensen, Pixar Technical Memo #15-04 - July, 2014
///     http://graphics.pixar.com/library/ApproxBSSRDF/paper.pdf
///
/// [2] http://www-history.mcs.st-andrews.ac.uk/history/HistTopics/Quadratic_etc_equations.html
///
class NormalizedDiffusionBssrdf : public Bssrdf
{
public:
    // Artist-friendly constructor
    NormalizedDiffusionBssrdf(const scene_rdl2::math::Vec3f &N, const scene_rdl2::math::Color &albedo,
                              const scene_rdl2::math::Color &radius, const scene_rdl2::rdl2::Material* material,
                              const scene_rdl2::rdl2::EvalNormalFunc evalNormalFn);
    NormalizedDiffusionBssrdf(scene_rdl2::alloc::Arena *arena, const NormalizedDiffusionBssrdfv &bssrdfv, int lane);

    ~NormalizedDiffusionBssrdf() {}

    /// Sampling API:
    /// This function returns R(r), equation (3) in [1]
    scene_rdl2::math::Color eval(float r, bool global = false) const override;

    /// We use exact importance sampling by inverting the CDF in eq (11) in [1]
    float sampleLocal(float r1, float r2, scene_rdl2::math::Vec3f &dPi, float &r) const override;

    /// This is only needed for unit-testing and verify that the pdf integrates to 1.
    float pdfLocal(float r) const override;

    /// For NormalizedDiffusion, diffuseReflectance is same as the Albedo [1]
    scene_rdl2::math::Color diffuseReflectance() const override { return mAlbedo; }

    bool getProperty(Property property, float *dest) const override;

    void show(std::ostream& os, const std::string& indent) const override;

private:
    // Copy is disabled
    NormalizedDiffusionBssrdf(const NormalizedDiffusionBssrdf &other) =delete;
    const NormalizedDiffusionBssrdf &operator=(const NormalizedDiffusionBssrdf &other) =delete;

    scene_rdl2::math::Color mAlbedo;    // 'A'  surface albedo
    scene_rdl2::math::Color mDmfp;      // 'ld' diffuse mean free path length, aka 'translucency radius'
    scene_rdl2::math::Color mD;         // 'd'  shaping parameter

    // Used for local importance sampling
    scene_rdl2::math::Color mChannelCdf;
    scene_rdl2::math::Color mMaxR;                // per-channel distance at which R(r) < 0.001
};

} // namespace shading
} // namespace moonray

