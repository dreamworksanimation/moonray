// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

///
/// @file BsdfRetroreflection.h
/// $Id$
///

#pragma once

#include <moonray/rendering/shading/Util.h>
#include "Bsdf.h"
#include "BsdfSlice.h"

#include <moonray/rendering/shading/ispc/bsdf/BsdfRetroreflection_ispc_stubs.h>

#include <scene_rdl2/common/math/Color.h>
#include <scene_rdl2/common/math/ReferenceFrame.h>
#include <scene_rdl2/common/math/Vec3.h>
#include <scene_rdl2/common/platform/IspcUtil.h>

namespace moonray {
namespace shading {

//----------------------------------------------------------------------------

///
/// @class RetroReflectionBsdfLobe BsdfRetroreflection.h <shading/BsdfRetroreflection.h>
/// @brief retro-reflective normalized phong lobe
/// References:
/// [1] Lafortune et al, ReportCW 1994, "Using the Modified Phong Reflectance Model for Physically Based Rendering"
/// [2] Walter et al, EGSR 2007, "Microfacet Models for Refraction through Rough Surfaces"
/// [3] Igehy, Siggraph 1999,  "Tracing Ray Differentials"
class RetroreflectionBsdfLobe : public BsdfLobe
{
public:
    // Constructor / Destructor
    RetroreflectionBsdfLobe(const scene_rdl2::math::Vec3f &N, const float roughness) :
        BsdfLobe(Type(REFLECTION | GLOSSY), DifferentialFlags(0), false,
                 PROPERTY_NORMAL | PROPERTY_ROUGHNESS),

        mN(N),
        mInputRoughness(roughness)
    {
        // Apply roughness squaring to linearize roughness response
        // See "Physically-Based Shading at Disney" Siggraph course notes.
        mRoughness = mInputRoughness * mInputRoughness;
        if (mRoughness < 0.001) {
            mRoughness = 0.001;
        }

        // Use a directional differential scale that varies with roughness
        // Taken from CookTorrance roughness-based differential computation
        mdDFactor = sdDFactorMin + mRoughness * sdDFactorSlope;
    }

    // BsdfLobe API
    scene_rdl2::math::Color eval(const BsdfSlice &slice, const scene_rdl2::math::Vec3f &wi, float *pdf = NULL) const override
    {

        // Prepare for early exit
        if (pdf != NULL) {
            *pdf = 0.0f;
        }

        const float cosThetaI = dot(mN, wi);
        if (cosThetaI <= 0.0f)      return scene_rdl2::math::sBlack;

        const float cosOmegaO_OmegaI = dot(slice.getWo(), wi);
        if (cosOmegaO_OmegaI <= 0.0f)      return scene_rdl2::math::sBlack;

        //convert roughness to phong exponent (see [2])
        const float alphaP = 2.0f * scene_rdl2::math::rcp(mRoughness * mRoughness) - 2.0f;

        // section 3.1 in [1]
        const float power = scene_rdl2::math::pow(cosOmegaO_OmegaI, alphaP) *
            scene_rdl2::math::sOneOverTwoPi;
        const float normalizationFactor2 = (alphaP + 2.0f);
        const scene_rdl2::math::Color f = computeScaleAndFresnel(cosThetaI) *
                normalizationFactor2 * power *
                (slice.getIncludeCosineTerm() ? cosThetaI : 1.0);

        // Compute pdf of sampling ([1])
        // section 3.3.1 in [1])
        if (pdf != NULL) {
            const float normalizationFactor1 = (alphaP + 1.0f);
            *pdf = normalizationFactor1 * power;
        }

        // Soften hard shadow terminator due to shading normals
        const float Gs = slice.computeShadowTerminatorFix(mN, wi);

        return Gs * f;
    }


    scene_rdl2::math::Color sample(const BsdfSlice &slice, float r1, float r2,
                       scene_rdl2::math::Vec3f &wi, float &pdf) const override
    {
        const float cosNO = dot(mN, slice.getWo());
        if (cosNO <= 0.0f) {
            wi = scene_rdl2::math::Vec3f(scene_rdl2::math::zero);
            pdf = 0.0f;
            return scene_rdl2::math::sBlack;
        }

        // sample the retro-reflective phong lobe
        // section 3.3.1 in [1]
        const float alphaP = 2.0f * scene_rdl2::math::rcp(mRoughness * mRoughness) - 2.0f;
        const float cosThetaM = pow(r1, scene_rdl2::math::rcp(alphaP + 1.0f));
        const float sinThetaM = sqrt(1.0f - cosThetaM*cosThetaM);
        const float phiM = 2.0f * scene_rdl2::math::sPi * r2;

        scene_rdl2::math::Vec3f m = computeLocalSphericalDirection(cosThetaM, sinThetaM, phiM);

        //sample along the outgoing vector
        scene_rdl2::math::ReferenceFrame frame(slice.getWo());
        wi = frame.localToGlobal(m);

        return eval(slice, wi, &pdf);
    }


    scene_rdl2::math::Color albedo(const BsdfSlice &slice) const override
    {
        float cosThetaWo = dot(mN, slice.getWo());
        return computeScaleAndFresnel(scene_rdl2::math::abs(cosThetaWo));
    }

    void differentials(const scene_rdl2::math::Vec3f &wo, const scene_rdl2::math::Vec3f &wi,
            float r1, float r2, const scene_rdl2::math::Vec3f &dNdx, const scene_rdl2::math::Vec3f &dNdy,
            scene_rdl2::math::Vec3f &dDdx, scene_rdl2::math::Vec3f &dDdy) const override
    {
        // Reverse incoming differentials, based on [3]
        // Factors taken from CookTorrance roughness-based differential computation
        // TODO: Not sure this is even close to correct.
        dDdx *= -mdDFactor;
        dDdy *= -mdDFactor;
    }

    bool getProperty(Property property, float *dest) const override
    {
        bool handled = true;
        switch (property)
        {
        case PROPERTY_ROUGHNESS:
            *dest       = mInputRoughness;
            *(dest + 1) = mInputRoughness;
            break;
        case PROPERTY_NORMAL:
            {
                *dest       = mN.x;
                *(dest + 1) = mN.y;
                *(dest + 2) = mN.z;
            }
            break;
        default:
            handled = BsdfLobe::getProperty(property, dest);
            break;
        }

        return handled;
    }

    void show(std::ostream& os, const std::string& indent) const override
    {
        const scene_rdl2::math::Color& scale = getScale();
        const Fresnel * const fresnel = getFresnel();
        os << indent << "[RetroreflectionBsdfLobe]\n";
        os << indent << "    " << "scale: "
            << scale.r << " " << scale.g << " " << scale.b << "\n";
        if (fresnel) {
            fresnel->show(os, indent + "    ");
        }
    }

private:
    // Derive a directional differential scale that varies according to roughness
    // Copied from Cook-Torrance BRDF (BsdfCookTorrance.h)
    static constexpr float sdDFactorMin = 1.0f;
    static constexpr float sdDFactorMax = 8.0f;
    static constexpr float sdDFactorSlope = (sdDFactorMax - sdDFactorMin);

    scene_rdl2::math::Vec3f mN;
    float mInputRoughness;
    float mRoughness;
    float mdDFactor;
};

//----------------------------------------------------------------------------
ISPC_UTIL_TYPEDEF_STRUCT(RetroreflectionBsdfLobe, RetroreflectionBsdfLobev);

} // namespace shading
} // namespace moonray


