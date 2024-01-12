// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

///
/// @file BsdfEyeCaustic.h
/// $Id$
///

#pragma once

#include "Bsdf.h"
#include "BsdfSlice.h"
#include <moonray/rendering/shading/Util.h>

#include <moonray/rendering/shading/ispc/bsdf/BsdfEyeCaustic_ispc_stubs.h>

#include <scene_rdl2/common/math/Color.h>
#include <scene_rdl2/common/math/ReferenceFrame.h>
#include <scene_rdl2/common/math/Vec3.h>
#include <scene_rdl2/common/platform/IspcUtil.h>

namespace moonray {
namespace shading {

//----------------------------------------------------------------------------

///
/// @class EyeCausticBsdfLobe BsdfEyeCaustic.h <rendeirng/shading/BsdfEyeCaustic.h>
/// @brief normalized (N.L)^Exponent lobe
/// This lobe creates a Phong-like specular lobe, but along the surface normal instead of
/// the 'reflection vector'. It is called "EyeCaustic" because it is empirically modeled
/// based on the caustics observed in eyeballs from refractions through the cornea.
/// When used in conjunction with 'iris-bulge' (concave bending of the flat iris-geo)
/// via normal mapping, the concave normals, combined with the eye caustic lobe,
/// produce the effect of light 'pooling' on the iris, opposite to the light-reflection
/// on the cornea. This gives the caustic-like behavior desired in our characters' eyes.
/// References for Normalized Phong and Sampling:
/// [1] Lafortune et al, ReportCW 1994, "Using the Modified Phong Reflectance Model for Physically Based Rendering"
/// [2] Walter et al, EGSR 2007, "Microfacet Models for Refraction through Rough Surfaces"
class EyeCausticBsdfLobe : public BsdfLobe
{
public:
    // Constructor / Destructor
    EyeCausticBsdfLobe(const scene_rdl2::math::Vec3f& N,
                       const scene_rdl2::math::Vec3f& irisN,
                       const scene_rdl2::math::Color& causticColor,
                       const float exponent) :
        BsdfLobe(Type(REFLECTION | GLOSSY),
                 DifferentialFlags(0),
                 false,
                 PROPERTY_NORMAL),
        mFrame(irisN),
        mCausticColor(causticColor),
        mN(N)
        {
            // As mentioned in [2], Phong exponents relates to Beckmann roughness using:
            // exponent = 2 * pow(roughness, -2) -2
            // Using a minimum roughness of 0.01, gives us the maximum exponent allowed for this lobe.
            const float maxExponent = 20000.0f;
            mExponent = scene_rdl2::math::clamp(exponent, 0.1f, maxExponent);
            const float normalizedExponent = mExponent * scene_rdl2::math::rcp(maxExponent);
            // Use a directional differential scale that varies with roughness
            // Taken from CookTorrance roughness-based differential computation
            mdDFactor = sdDFactorMin + normalizedExponent * sdDFactorSlope;

            mNormalizationFactor = (mExponent + 2) * scene_rdl2::math::sOneOverTwoPi;
        }



    // BsdfLobe API
    scene_rdl2::math::Color eval(const BsdfSlice &slice, const scene_rdl2::math::Vec3f &wi, float *pdf = NULL) const override
    {
        // Prepare for early exit
        if (pdf != NULL) {
            *pdf = 0.0f;
        }

        const float cosThetaO = dot(mN, slice.getWo());
        if (cosThetaO <= 0.0f) return scene_rdl2::math::sBlack;

        const float eyeCausticBrdfCosThetaI = dot(mFrame.getN(), wi);
        if (eyeCausticBrdfCosThetaI <= 0.0f)      return scene_rdl2::math::sBlack;

        // Normalization Factor from [1]
        const float normalizedPhong = mNormalizationFactor * powf(eyeCausticBrdfCosThetaI, mExponent);

        // Note: we assume this lobe has been setup with a OneMinus*Fresnel
        // as we want to use 1 - specular_fresnel. Also notice we use
        // cosThetaWo to evaluate the fresnel term, as an approximation of what
        // hDotWi would be for the specular lobe.
        const scene_rdl2::math::Color f = mCausticColor * computeScaleAndFresnel(cosThetaO) * normalizedPhong *
                (slice.getIncludeCosineTerm() ? eyeCausticBrdfCosThetaI : 1.0);

        // Compute pdf of sampling ([1])
        // section 3.3.1 in [1])
        if (pdf != NULL) {
            const float normalizationFactor = (mExponent + 1) * scene_rdl2::math::sOneOverTwoPi;
            *pdf = normalizationFactor * powf(eyeCausticBrdfCosThetaI, mExponent);
        }

        // Soften hard shadow terminator due to shading normals
        const float Gs = slice.computeShadowTerminatorFix(mFrame.getN(), wi);

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

        // see [1] for PDF inversion
        const float cosThetaM = scene_rdl2::math::pow(r1, scene_rdl2::math::rcp(mExponent+1));
        const float sinThetaM = scene_rdl2::math::sqrt(1 - cosThetaM*cosThetaM);
        const float phiM = scene_rdl2::math::sTwoPi * r2;

        scene_rdl2::math::Vec3f m = computeLocalSphericalDirection(cosThetaM, sinThetaM, phiM);

        // Sample along the normal
        wi = mFrame.localToGlobal(m);

        return eval(slice, wi, &pdf);
    }


    scene_rdl2::math::Color albedo(const BsdfSlice &slice) const override
    {
        float cosThetaWo = dot(mN, slice.getWo());
        return mCausticColor * computeScaleAndFresnel(scene_rdl2::math::abs(cosThetaWo));
    }

    void differentials(const scene_rdl2::math::Vec3f &wo, const scene_rdl2::math::Vec3f &wi,
            float r1, float r2, const scene_rdl2::math::Vec3f &dNdx, const scene_rdl2::math::Vec3f &dNdy,
            scene_rdl2::math::Vec3f &dDdx, scene_rdl2::math::Vec3f &dDdy) const override
    {
        // Light reflected along N, D = N
        // Scaled by roughness
        dDdx = dNdx * mdDFactor;
        dDdy = dNdy * mdDFactor;

    }

    bool getProperty(Property property, float *dest) const override
    {
        bool handled = true;
        switch (property)
        {
        case PROPERTY_NORMAL:
            {
                *dest       = mFrame.getN().x;
                *(dest + 1) = mFrame.getN().y;
                *(dest + 2) = mFrame.getN().z;
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
        os << indent << "[EyeCausticBsdfLobe]\n";
        os << indent << "    " << "scale: "
            << scale.r << " " << scale.g << " " << scale.b << "\n";
        os << indent << "    " << "caustic color: "
            << mCausticColor.r << " " << mCausticColor.g << " " << mCausticColor.b << "\n";
        os << indent << "    " << "exponent: " << mExponent << "\n";
        os << indent << "    " << "iris N: " << mFrame.getN().x << " " << mFrame.getN().y << " "<< mFrame.getN().z << "\n";
        os << indent << "    " << "N: " << mN.x << " " << mN.x << " " << mN.z << "\n";

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

    scene_rdl2::math::ReferenceFrame mFrame;
    scene_rdl2::math::Color mCausticColor;
    scene_rdl2::math::Vec3f mN;
    float mExponent;
    float mNormalizationFactor;
    float mdDFactor;
};

//----------------------------------------------------------------------------
ISPC_UTIL_TYPEDEF_STRUCT(EyeCausticBsdfLobe, EyeCausticBsdfLobev);

} // namespace shading
} // namespace moonray


