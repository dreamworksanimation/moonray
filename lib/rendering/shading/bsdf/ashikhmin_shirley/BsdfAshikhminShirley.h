// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

///
/// @file BsdfAshikhminShirley.h
/// $Id$
///

#pragma once

#include <moonray/rendering/shading/bsdf/Bsdf.h>
#include <moonray/rendering/shading/bsdf/BsdfSlice.h>
#include <moonray/rendering/shading/Util.h>

#include <moonray/rendering/shading/ispc/bsdf/ashikhmin_shirley/BsdfAshikhminShirley_ispc_stubs.h>

#include <scene_rdl2/common/math/Color.h>
#include <scene_rdl2/common/math/ReferenceFrame.h>
#include <scene_rdl2/common/math/Vec3.h>

namespace moonray {
namespace shading {


//----------------------------------------------------------------------------

///
/// @class AshikhminShirleyGlossyBsdfLobe BsdfAshikhminShirley.h <rendering/shading/ashikhmin_shirley/BsdfAshikhminShirley.h>
/// @brief The Ashikhmin Shirley Glossy Anisotropic Brdf
/// See the paper "An Anisotropic Phong BRDF ModelAn Anisotropic Phong BRDF Model",
///               By Michael Ashikhmin and Peter Shirley
///               Journal of Graphics Tools, 2000, vol. 5, p25-32.
/// This corresponds to equation (4) of the paper, except for the Fresnel
/// term. It's up to the material to set a SchlickFresnel via setFresnel()
/// The importance sampling uses equation (9) and (10), and the pdf is equation (8).
///
class AshikhminShirleyGlossyBsdfLobe : public BsdfLobe
{
public:
    // Constructor / Destructor
    AshikhminShirleyGlossyBsdfLobe(const scene_rdl2::math::Vec3f &N, const scene_rdl2::math::Vec3f &anisoDirection,
            float roughnessU, float roughnessV);
    ~AshikhminShirleyGlossyBsdfLobe() {}

    // BsdfLobe API
    scene_rdl2::math::Color eval(const BsdfSlice &slice, const scene_rdl2::math::Vec3f &wi, float *pdf = NULL) const override;
    scene_rdl2::math::Color sample(const BsdfSlice &slice, float r1, float r2,
            scene_rdl2::math::Vec3f &wi, float &pdf) const override;

    bool getProperty(Property attr, float *dest) const override;

    finline scene_rdl2::math::Color albedo(const BsdfSlice &slice) const override
    {
        float cosThetaWo = scene_rdl2::math::max(dot(mFrame.getN(), slice.getWo()), 0.0f);
        return computeScaleAndFresnel(cosThetaWo);
    }

    void differentials(const scene_rdl2::math::Vec3f &wo, const scene_rdl2::math::Vec3f &wi,
            float r1, float r2, const scene_rdl2::math::Vec3f &dNdx, const scene_rdl2::math::Vec3f &dNdy,
            scene_rdl2::math::Vec3f &dDdx, scene_rdl2::math::Vec3f &dDdy) const override;

    void show(std::ostream& os, const std::string& indent) const override
    {
        const scene_rdl2::math::Color& scale = getScale();
        const Fresnel * const fresnel = getFresnel();
        os << indent << "[AshikhminShirleyGlossyBsdfLobe]\n";
        os << indent << "    " << "scale: "
            << scale.r << " " << scale.g << " " << scale.b << "\n";
        if (fresnel) {
            fresnel->show(os, indent + "    ");
        }
    }

private:
    // Derive a directional differential scale that varies according to roughness
    // Experimentally, we found that we want 1.0 when roughness is 0
    // and 8 when it is 1.
    static constexpr float sdDFactorMin = 1.0f;
    static constexpr float sdDFactorMax = 8.0f;
    static constexpr float sdDFactorSlope = (sdDFactorMax - sdDFactorMin);

    scene_rdl2::math::ReferenceFrame mFrame;

    float mInputRoughnessU;
    float mInputRoughnessV;
    float mExponentU;
    float mExponentV;
    float mScaleFactor;
    float mSampleFactor;
    float mdDFactor;
};


///
/// @class AshikhminShirleyDiffuseBsdfLobe BsdfAshikhminShirley.h <pbr/BsdfAshikhminShirley.h>
/// @brief The Ashikhmin Shirley Diffuse Brdf
/// This corresponds to equation (5) of the paper, with the omission of the
/// Rd * (1 - Rs) term. That term should be set via setScale() by the material.
///
class AshikhminShirleyDiffuseBsdfLobe : public BsdfLobe
{
public:
    // Constructor / Destructor
    AshikhminShirleyDiffuseBsdfLobe(const scene_rdl2::math::Vec3f &N);
    ~AshikhminShirleyDiffuseBsdfLobe()  {}

    // BsdfLobe API
    scene_rdl2::math::Color eval(const BsdfSlice &slice, const scene_rdl2::math::Vec3f &wi, float *pdf = NULL) const override;
    finline scene_rdl2::math::Color sample(const BsdfSlice &slice, float r1, float r2,
            scene_rdl2::math::Vec3f &wi, float &pdf) const override
    {
        wi = mFrame.localToGlobal(sampleLocalHemisphereCosine(r1, r2));
        return eval(slice, wi, &pdf);
    }

    bool getProperty(Property attr, float *dest) const override;

    finline scene_rdl2::math::Color albedo(const BsdfSlice &slice) const override
    {
        float cosThetaWo = scene_rdl2::math::max(dot(mFrame.getN(), slice.getWo()), 0.0f);
        return computeScaleAndFresnel(cosThetaWo);
    }

    // no surface curvature required
    finline void differentials(const scene_rdl2::math::Vec3f &wo, const scene_rdl2::math::Vec3f &wi,
            float r1, float r2, const scene_rdl2::math::Vec3f &dNdx, const scene_rdl2::math::Vec3f &dNdy,
            scene_rdl2::math::Vec3f &dDdx, scene_rdl2::math::Vec3f &dDdy) const override
    {
        // See BsdfLambert.h for details
        localHemisphereCosineDifferentials(r1, r2, dDdx, dDdy);
        squarifyRectangle(dDdx, dDdy);

        dDdx = mFrame.localToGlobal(dDdx);
        dDdy = mFrame.localToGlobal(dDdy);
    }

    void show(std::ostream& os, const std::string& indent) const override
    {
        const scene_rdl2::math::Color& scale = getScale();
        const Fresnel * const fresnel = getFresnel();
        os << indent << "[AshikhminShirleyDiffuseBsdfLobe]\n";
        os << indent << "    " << "scale: "
            << scale.r << " " << scale.g << " " << scale.b << "\n";
        if (fresnel) {
            fresnel->show(os, indent + "    ");
        }
    }

private:
    scene_rdl2::math::ReferenceFrame mFrame;
};


//----------------------------------------------------------------------------
ISPC_UTIL_TYPEDEF_STRUCT(AshikhminShirleyDiffuseBsdfLobe, AshikhminShirleyDiffuseBsdfLobev);
ISPC_UTIL_TYPEDEF_STRUCT(AshikhminShirleyGlossyBsdfLobe, AshikhminShirleyGlossyBsdfLobev);

} // namespace shading
} // namespace moonray


