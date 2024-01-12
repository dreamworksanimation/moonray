// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

///
/// @file BsdfMirror.h
/// $Id$
///

#pragma once

#include "Bsdf.h"
#include "BsdfSlice.h"

#include <moonray/rendering/shading/ispc/bsdf/BsdfMirror_ispc_stubs.h>

#include <scene_rdl2/common/platform/IspcUtil.h>
#include <scene_rdl2/common/math/Color.h>
#include <scene_rdl2/common/math/Vec3.h>
#include <scene_rdl2/common/math/Math.h>
#include <scene_rdl2/common/math/MathUtil.h>

namespace moonray {
namespace shading {


//----------------------------------------------------------------------------
///

/// @class MirrorReflectionBsdfLobe BsdfMirror.h <shading/BsdfMirror.h>
/// @brief The mirror reflection brdf delta function (scale * fresnel / cos(theta))
/// 
class MirrorReflectionBsdfLobe : public BsdfLobe
{
public:
    // Constructor / Destructor
    MirrorReflectionBsdfLobe(const scene_rdl2::math::Vec3f &N) :
        BsdfLobe(Type(REFLECTION | MIRROR),
                 DifferentialFlags(0),
                 false,
                 PROPERTY_NORMAL | PROPERTY_ROUGHNESS),
        mN(N)   {}

    // BsdfLobe API
    finline scene_rdl2::math::Color eval(const BsdfSlice &slice, const scene_rdl2::math::Vec3f &wi, float *pdf = NULL) const override
    {
        if (pdf != NULL) {
            *pdf = 0.0f;
        }
        return scene_rdl2::math::Color(0.0f, 0.0f, 0.0f);
    }


    finline scene_rdl2::math::Color sample(const BsdfSlice &slice, float r1, float r2,
            scene_rdl2::math::Vec3f &wi, float &pdf) const override
    {
        // Reflect wo
        float cosTheta = computeReflectionDirection(mN, slice.getWo(), wi);
        cosTheta = scene_rdl2::math::abs(cosTheta);

        // Probability of sampling the delta function is 1.
        pdf = 1.0f;

        // Compute contribution
        return computeScaleAndFresnel(cosTheta) *
                (slice.getIncludeCosineTerm()  ?  1.0f  :  1.0f / cosTheta);
    }


    finline scene_rdl2::math::Color albedo(const BsdfSlice &slice) const override
    {
        float cosThetaWo = dot(mN, slice.getWo());
        return computeScaleAndFresnel(scene_rdl2::math::abs(cosThetaWo));
    }

    void differentials(const scene_rdl2::math::Vec3f &wo, const scene_rdl2::math::Vec3f &wi,
            float r1, float r2, const scene_rdl2::math::Vec3f &dNdx, const scene_rdl2::math::Vec3f &dNdy,
            scene_rdl2::math::Vec3f &dDdx, scene_rdl2::math::Vec3f &dDdy) const override
    {
        computeReflectionDirectionDifferential(wo, wi, mN, dNdx, dNdy, dDdx, dDdy);
    }

    bool getProperty(Property property, float *dest) const override
    {
        bool handled = true;

        switch (property) {
        case PROPERTY_ROUGHNESS:
            {
                *dest       = 0.0f;
                *(dest + 1) = 0.0f;
            }
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
        }

        return handled;
    }

    void show(std::ostream& os, const std::string& indent) const override
    {
        const scene_rdl2::math::Color& scale = getScale();
        const Fresnel * const fresnel = getFresnel();
        os << indent << "[MirrorReflectionBsdfLobe]\n";
        os << indent << "    " << "scale: "
            << scale.r << " " << scale.g << " " << scale.b << "\n";
        os << indent << "    " << "N: "
            << mN.x << " " << mN.y << " " << mN.z << "\n";
        if (fresnel) {
            fresnel->show(os, indent + "    ");
        }
    }

private:
    scene_rdl2::math::Vec3f mN;
};

//----------------------------------------------------------------------------
///
/// @class MirrorTransmissionBsdfLobe BsdfMirror.h <shading/BsdfMirror.h>
/// @brief The mirror transmission brdf delta function (scale * fresnel / cos(theta))
///
class MirrorTransmissionBsdfLobe : public BsdfLobe
{
public:
    /// Constructor / Destructor
    /// Pass in neta, the ratio of the ior for incident / transmitted material.
    /// This is used appropriately by the renderer at sampling time whether
    /// we're entering or leaving the material.
    /// Important: This lobe should be setup with a OneMinusFresnel
    MirrorTransmissionBsdfLobe(const scene_rdl2::math::Vec3f &N,
                               float etaI,
                               float etaT,
                               const scene_rdl2::math::Color &tint,
                               float abbeNumber = 0.0f) :
        BsdfLobe(Type(TRANSMISSION | MIRROR),
                 DifferentialFlags(0),
                 false,
                 PROPERTY_NORMAL | PROPERTY_ROUGHNESS),
        mN(N),
        mEtaI(etaI),
        mEtaT(etaT),
        mTint(tint)
    {
        // A value of zero is used to turn OFF dispersion.
        // Values range from below 25 for very dense flint glasses, around 34 for polycarbonate plastics,
        // up to 65 for common crown glasses, and 75 to 85 for some fluorite and phosphate crown glasses.
        if (scene_rdl2::math::isZero(abbeNumber)) {
            mAllowDispersion = false;
        } else {
            shading::computeSpectralIOR(etaT,
                                        abbeNumber,
                                        &mEtaR,
                                        &mEtaG,
                                        &mEtaB);
            mAllowDispersion = true;
        }
    }

    // BsdfLobe API
    finline scene_rdl2::math::Color eval(const BsdfSlice &slice, const scene_rdl2::math::Vec3f &wi, float *pdf = NULL) const  override
    {
        if (pdf != NULL) {
            *pdf = 0.0f;
        }
        return scene_rdl2::math::Color(0.0f, 0.0f, 0.0f);
    }

    finline scene_rdl2::math::Color sample(const BsdfSlice &slice,
                         float r1,
                         float r2,
                         scene_rdl2::math::Vec3f &wi,
                         float &pdf) const override
    {
        MNRY_ASSERT(!scene_rdl2::math::isZero(mEtaT));

        float neta;
        scene_rdl2::math::Color dispersionColor = scene_rdl2::math::sWhite;
        if (mAllowDispersion) {
            float sampledEta;
            shading::sampleSpectralIOR(r1,
                                       mEtaR,
                                       mEtaG,
                                       mEtaB,
                                       &sampledEta,
                                       &pdf,
                                       dispersionColor);
            neta = mEtaI / sampledEta;
        } else {
            // Probability of sampling the delta function is 1.
            neta = mEtaI / mEtaT;
            pdf = 1.0f;
        }

        // Compute the transmission direction
        float cosThetaWo, cosThetaWi;
        if (!computeRefractionDirection(mN,
                                        slice.getWo(),
                                        neta,
                                        wi,
                                        cosThetaWo,
                                        cosThetaWi)) {

            // Total internal reflection is handled by the reflection lobe
            // fresnel reflectance
            pdf = 0.0f;
            return scene_rdl2::math::Color(0.0f, 0.0f, 0.0f);
        }


        // Note: we assume this lobe has been setup with a OneMinus*Fresnel
        // as we want to use 1 - specular_fresnel. Also notice we use
        // cosThetaWo to evaluate the fresnel term, as an approximation of what
        // hDotWi would be for the specular lobe.
        return mTint * computeScaleAndFresnel(cosThetaWo) * dispersionColor *
                (slice.getIncludeCosineTerm()  ?  1.0f  :  1.0f / cosThetaWi);
    }

    finline scene_rdl2::math::Color albedo(const BsdfSlice &slice) const override
    {
        float cosThetaWo = dot(mN, slice.getWo());
        return mTint * computeScaleAndFresnel(scene_rdl2::math::abs(cosThetaWo));
    }

    void differentials(const scene_rdl2::math::Vec3f &wo, const scene_rdl2::math::Vec3f &wi,
            float r1, float r2, const scene_rdl2::math::Vec3f &dNdx, const scene_rdl2::math::Vec3f &dNdy,
            scene_rdl2::math::Vec3f &dDdx, scene_rdl2::math::Vec3f &dDdy) const override
    {
        computeRefractionDirectionDifferential((mEtaI/mEtaT),
                                               wo, wi, mN, dNdx, dNdy,
                                               dDdx, dDdy);
    }

    bool getProperty(Property property, float *dest) const override
    {
        bool handled = true;

        switch (property) {
        case PROPERTY_ROUGHNESS:
            {
                *dest       = 0.0f;
                *(dest + 1) = 0.0f;
            }
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
        }

        return handled;
    }

    void show(std::ostream& os, const std::string& indent) const override
    {
        const scene_rdl2::math::Color& scale = getScale();
        const Fresnel * const fresnel = getFresnel();
        os << indent << "[MirrorTransmissionBsdfLobe]\n";
        os << indent << "    " << "scale: "
            << scale.r << " " << scale.g << " " << scale.b << "\n";
        os << indent << "    " << "N: "
            << mN.x << " " << mN.y << " " << mN.z << "\n";
        if (fresnel) {
            fresnel->show(os, indent + "    ");
        }
    }

private:
    scene_rdl2::math::Vec3f mN;
    float mEtaI, mEtaT;
    scene_rdl2::math::Color mTint;
    float mEtaR, mEtaG, mEtaB;
    bool mAllowDispersion;
};


//----------------------------------------------------------------------------

///
/// @class MirrorRetroreflectionBsdfLobe BsdfMirror.h <shading/BsdfMirror.h>
/// @brief The mirror retroreflection brdf delta function (scale * fresnel / cos(theta))
///
class MirrorRetroreflectionBsdfLobe : public BsdfLobe
{
public:
    // Constructor / Destructor
    MirrorRetroreflectionBsdfLobe(const scene_rdl2::math::Vec3f &N) :
        BsdfLobe(Type(REFLECTION | MIRROR),
                 DifferentialFlags(0),
                 false,
                 PROPERTY_NORMAL | PROPERTY_ROUGHNESS),
        mN(N)   {}

    // BsdfLobe API
    finline scene_rdl2::math::Color eval(const BsdfSlice &slice, const scene_rdl2::math::Vec3f &wi, float *pdf = NULL) const override
    {
        if (pdf != NULL) {
            *pdf = 0.0f;
        }
        return scene_rdl2::math::Color(0.0f, 0.0f, 0.0f);
    }


    finline scene_rdl2::math::Color sample(const BsdfSlice &slice, float r1, float r2,
            scene_rdl2::math::Vec3f &wi, float &pdf) const override
    {
        // Retroreflect wo
        wi = slice.getWo();

        // Probability of sampling the delta function is 1.
        pdf = 1.0f;

        float cosTheta = dot(mN, slice.getWo());
        // Compute contribution
        return computeScaleAndFresnel(cosTheta) *
                (slice.getIncludeCosineTerm()  ?  1.0f  :  1.0/cosTheta);
    }


    finline scene_rdl2::math::Color albedo(const BsdfSlice &slice) const override
    {
        float cosThetaWo = dot(mN, slice.getWo());
        return computeScaleAndFresnel(scene_rdl2::math::abs(cosThetaWo));
    }

    void differentials(const scene_rdl2::math::Vec3f &wo, const scene_rdl2::math::Vec3f &wi,
            float r1, float r2, const scene_rdl2::math::Vec3f &dNdx, const scene_rdl2::math::Vec3f &dNdy,
            scene_rdl2::math::Vec3f &dDdx, scene_rdl2::math::Vec3f &dDdy) const override
    {
        // TODO: Shouldn't this be a no-op for dDd{x,y} ?
        dDdx = -dDdx;
        dDdy = -dDdy;
    }

    bool getProperty(Property property, float *dest) const override
    {
        bool handled = true;

        switch (property) {
        case PROPERTY_ROUGHNESS:
            {
                *dest       = 0.0f;
                *(dest + 1) = 0.0f;
            }
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
        }

        return handled;
    }

    void show(std::ostream& os, const std::string& indent) const override
    {
        const scene_rdl2::math::Color& scale = getScale();
        const Fresnel * const fresnel = getFresnel();
        os << indent << "[MirrorRetroreflectionBsdfLobe]\n";
        os << indent << "    " << "scale: "
            << scale.r << " " << scale.g << " " << scale.b << "\n";
        os << indent << "    " << "N: "
            << mN.x << " " << mN.y << " " << mN.z << "\n";
        if (fresnel) {
            fresnel->show(os, indent + "    ");
        }
    }

private:
    scene_rdl2::math::Vec3f mN;
};

//----------------------------------------------------------------------------
ISPC_UTIL_TYPEDEF_STRUCT(MirrorReflectionBsdfLobe, MirrorReflectionBsdfLobev);
ISPC_UTIL_TYPEDEF_STRUCT(MirrorTransmissionBsdfLobe, MirrorTransmissionBsdfLobev);
ISPC_UTIL_TYPEDEF_STRUCT(MirrorRetroreflectionBsdfLobe, MirrorRetroreflectionBsdfLobev);

} // namespace shading
} // namespace moonray


