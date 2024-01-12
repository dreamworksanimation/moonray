// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

///
/// @file BsdfWard.h
/// $Id$
///

#pragma once

#include <moonray/rendering/shading/bsdf/Bsdf.h>
#include <moonray/rendering/shading/bsdf/BsdfSlice.h>
#include <moonray/rendering/shading/ispc/bsdf/ward/BsdfWard_ispc_stubs.h>

#include <scene_rdl2/common/math/Color.h>
#include <scene_rdl2/common/math/Vec3.h>
#include <scene_rdl2/common/platform/IspcUtil.h>

namespace moonray {
namespace shading {


//----------------------------------------------------------------------------

///
/// @class WardBsdfLobe BsdfWard.h <shading/BsdfWard.h>
/// @brief The Ward bsdf basics.
/// 
class WardBsdfLobe : public BsdfLobe
{
public:
    /// Constructor / Destructor
    /// Assumes N and anisoDirection are normalized
    WardBsdfLobe(const scene_rdl2::math::Vec3f &N, const scene_rdl2::math::Vec3f &anisoDirection,
            float roughnessU, float roughnessV, bool squareRoughness);
    virtual ~WardBsdfLobe() =0;

    // Accessor
    void setRoughness(float roughnessU, float roughnessV, bool squareRoughness);
    void getRoughness(float &roughnessU, float &roughnessV) const;

    /// Note returns the anisoDirection after being orthogonalized with N
    const scene_rdl2::math::Vec3f &getAnisotropicDirection() const  {  return mFrame.getX();  }

    
    // BsdfLobe API
    virtual scene_rdl2::math::Color eval(const BsdfSlice &slice, const scene_rdl2::math::Vec3f &wi, float *pdf = NULL) const override = 0;
    virtual scene_rdl2::math::Color sample(const BsdfSlice &slice, float r1, float r2,
            scene_rdl2::math::Vec3f &wi, float &pdf) const override = 0;

    void differentials(const scene_rdl2::math::Vec3f &wo, const scene_rdl2::math::Vec3f &wi,
            float r1, float r2, const scene_rdl2::math::Vec3f &dNdx, const scene_rdl2::math::Vec3f &dNdy,
            scene_rdl2::math::Vec3f &dDdx, scene_rdl2::math::Vec3f &dDdy) const override;

    finline scene_rdl2::math::Color albedo(const BsdfSlice &slice) const override
    {
        float cosThetaWo = scene_rdl2::math::max(dot(mFrame.getN(), slice.getWo()), 0.0f);
        return computeScaleAndFresnel(cosThetaWo);
    }

    bool getProperty(Property property, float *dest) const override;

    void show(std::ostream& os, const std::string& indent) const override
    {
        const scene_rdl2::math::Color& scale = getScale();
        const Fresnel * fresnel = getFresnel();
        os << indent << "[WardBsdfLobe]\n";
        os << indent << "    " << "scale: "
            << scale.r << " " << scale.g << " " << scale.b;
        if (fresnel) {
            fresnel->show(os, indent + "    ");
        }
    }

protected:
    // Derive a directional differential scale that varies according to roughness
    // Experimentally, we found that we want 1.0 when roughness is 0
    // and 8 when it is 1.
    static constexpr float sdDFactorMin = 1.0f;
    static constexpr float sdDFactorMax = 8.0f;
    static constexpr float sdDFactorSlope = (sdDFactorMax - sdDFactorMin);

    scene_rdl2::math::ReferenceFrame mFrame;

    float mInputRoughnessU;
    float mInputRoughnessV;
    float mRoughnessU;
    float mRoughnessV;
    float mInvRoughUSqrd;
    float mInvRoughVSqrd;
    float mScaleFactor;
    float mdDFactor;
};


///
/// @class WardOriginalBsdfLobe WardOriginalBsdfLobe.h <shading/WardOriginalBsdfLobe.h>
/// @brief The original Ward bsdf. This is the original implementation of the
///     Ward bsdf as printed in the uncorrected version of the paper:
///     "Measuring and modeling anisotropic reflection" SIGGRAPH '92.
///     (This is the version currently sampled by the raytracing light)
///     The BRDF can be derived by following the procedure outlined in "Notes on
///     the Ward BRDF" Section 3.3 using the original Ward sampling functions for
///     theta_h and phi_h. When using these sampling functions and without
///     weighting the samples, the probability function that
///     results from this procedure is also the BRDF. For reference this
///     equation is:
///
///         phi = azimuthal angle of the half vector relative to the normal.
///         theta = inclination angle of the half vector relative to the normal
///                 (ie. cos(theta) = N.dot(half))
///
///         F = (cos^2(phi) / roughnessX^2) + (sin^2(phi) / roughnessZ^2)
///         S = theta / (4 * PI * half.dot(wo) * roughnessX * roughnessZ * sin(theta))
///         BRDF = S * exp(-F * theta^2)
///
class WardOriginalBsdfLobe : public WardBsdfLobe
{
public:
    /// Constructor / Destructor
    /// Assumes N and anisoDirection are normalized
    WardOriginalBsdfLobe(const scene_rdl2::math::Vec3f &N, const scene_rdl2::math::Vec3f &anisoDirection,
            float roughnessU, float roughnessV, bool squareRoughness);
    ~WardOriginalBsdfLobe() {}

    // BsdfLobe API
    scene_rdl2::math::Color eval(const BsdfSlice &slice, const scene_rdl2::math::Vec3f &wi, float *pdf = NULL) const override;
    scene_rdl2::math::Color sample(const BsdfSlice &slice, float r1, float r2, scene_rdl2::math::Vec3f &wi, float &pdf) const override;

    void show(std::ostream& os, const std::string& indent) const override
    {
        const scene_rdl2::math::Color& scale = getScale();
        const Fresnel * fresnel = getFresnel();
        os << indent << "[WardOriginalBsdfLobe]\n";
        os << indent << "    " << "scale: "
            << scale.r << " " << scale.g << " " << scale.b;
        if (fresnel) {
            fresnel->show(os, indent + "    ");
        }
    }

private:
    float pdf(const scene_rdl2::math::Vec3f &wo, const scene_rdl2::math::Vec3f &wi) const;
};


///
/// @class WardCorrectedBsdfLobe WardCorrectedBsdfLobe.h <shading/WardCorrectedBsdfLobe.h>
/// @brief The corrected Ward bsdf. This is the corrected implementation of the
///     Ward bsdf as printed in the paper:
///     "Notes on the Ward BRDF", Tech report 2005, by Bruce Walter.
///     See equation (4) for the brdf, equations (6) and (7) for the sampling,
///     and equation (9) for the sampling pdf.
///     CAREFUL: the notation in that paper has wo and wi swapped
class WardCorrectedBsdfLobe : public WardBsdfLobe
{
public:
    /// Constructor / Destructor
    /// Assumes N and anisoDirection are normalized
    WardCorrectedBsdfLobe(const scene_rdl2::math::Vec3f &N, const scene_rdl2::math::Vec3f &anisoDirection,
            float roughnessU, float roughnessV, bool squareRoughness);
    ~WardCorrectedBsdfLobe() override {}

    // BsdfLobe API
    scene_rdl2::math::Color eval(const BsdfSlice &slice, const scene_rdl2::math::Vec3f &wi, float *pdf = NULL) const override;
    scene_rdl2::math::Color sample(const BsdfSlice &slice, float r1, float r2, scene_rdl2::math::Vec3f &wi, float &pdf) const override;

    void show(std::ostream& os, const std::string& indent) const override
    {
        const scene_rdl2::math::Color& scale = getScale();
        const Fresnel * fresnel = getFresnel();
        os << indent << "[WardCorrectedBsdfLobe]\n";
        os << indent << "    " << "scale: "
            << scale.r << " " << scale.g << " " << scale.b;
        if (fresnel) {
            fresnel->show(os, indent + "    ");
        }
    }
};


///
/// @class WardDuerBsdfLobe WardDuerBsdfLobe.h <shading/WardDuerBsdfLobe.h>
/// @brief The Ward-Duer BSDF with Bounded Albedo.
///   This is the implementation based primarily on the technical report
///   which serves as a further extension on the Ward-Duer BRDF (which itself
///   changes the normalization factor of the Ward BRDF, as presented by Arne Duer).
///   "Bounding the Albedo of the Ward Reflectance Model"
///   By Walter Geisler-Moroder and Arne Duer, Technical Rpt 2010
///
///   This more or less is identical to the "Corrected" Ward lobe above,
///   Except it changes the normalization factor and weighting of the samples
///   The new normalization is described in eq. (23) of the tech report and
///   The new weighting scheme is described in eq. (21)
///
class WardDuerBsdfLobe : public WardBsdfLobe
{
public:
    /// Constructor / Destructor
    /// Assumes N and anisoDirection are normalized
    WardDuerBsdfLobe(const scene_rdl2::math::Vec3f &N, const scene_rdl2::math::Vec3f &anisoDirection,
            float roughnessU, float roughnessV, bool squareRoughness);
    ~WardDuerBsdfLobe() override {}

    // BsdfLobe API
    scene_rdl2::math::Color eval(const BsdfSlice &slice, const scene_rdl2::math::Vec3f &wi, float *pdf = NULL) const override;
    scene_rdl2::math::Color sample(const BsdfSlice &slice, float r1, float r2, scene_rdl2::math::Vec3f &wi, float &pdf) const override;

    void show(std::ostream& os, const std::string& indent) const override
    {
        const scene_rdl2::math::Color& scale = getScale();
        const Fresnel * fresnel = getFresnel();
        os << indent << "[WardDuerBsdfLobe]\n";
        os << indent << "    " << "scale: "
            << scale.r << " " << scale.g << " " << scale.b;
        if (fresnel) {
            fresnel->show(os, indent + "    ");
        }
    }
private:
    scene_rdl2::math::Color eval(const BsdfSlice &slice, const scene_rdl2::math::Vec3f &wi, const scene_rdl2::math::Vec3f &H, float *pdf) const;
};


//----------------------------------------------------------------------------
ISPC_UTIL_TYPEDEF_STRUCT(WardCorrectedBsdfLobe, WardCorrectedBsdfLobev);
ISPC_UTIL_TYPEDEF_STRUCT(WardDuerBsdfLobe, WardDuerBsdfLobev);
ISPC_UTIL_TYPEDEF_STRUCT(WardOriginalBsdfLobe, WardOriginalBsdfLobev);

} // namespace shading
} // namespace moonray


