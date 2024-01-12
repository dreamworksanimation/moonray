// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

///
/// @file BsdfUnder.h
/// $Id$
///

#pragma once

#include <moonray/rendering/shading/bsdf/Bsdf.h>
#include <moonray/rendering/shading/bsdf/BsdfSlice.h>

#include <moonray/rendering/shading/Shading.h>
#include <moonray/rendering/shading/ispc/bsdf/under/BsdfUnder_ispc_stubs.h>

#include <scene_rdl2/common/math/Color.h>
#include <scene_rdl2/common/math/Vec3.h>
#include <scene_rdl2/common/platform/IspcUtil.h>

namespace moonray {
namespace shading {

//----------------------------------------------------------------------------

///
/// @class UnderBsdfLobe BsdfUnder.h <shading/BsdfUnder.h>
/// @brief An "under" bsdf lobe adapter, which uses its Fresnel as a OneMinus*Fresnel
/// from another lobe that is layered on top
/// 
class UnderBsdfLobe : public BsdfLobe
{
public:
    // Constructor / Destructor
    UnderBsdfLobe(BsdfLobe *under,
                  const scene_rdl2::math::Vec3f &N,
                  const float thickness = 0.0f,
                  const scene_rdl2::math::Color attenuationColor = scene_rdl2::math::sWhite,
                  const float attenuationWeight = 1.0f) :
        BsdfLobe(under->getType(),
                 under->getDifferentialFlags(),
                 under->getIsSpherical(),
                 under->getPropertyFlags()),
        mUnder(under),
        mN(N),
        mThickness(thickness),
        mAttenuationColor(attenuationColor),
        mAttenuationWeight(attenuationWeight)
    {}

    // BsdfLobe API
    finline scene_rdl2::math::Color eval(const BsdfSlice &slice, const scene_rdl2::math::Vec3f &wi, float *pdf = NULL) const override
    {
        // Note: we assume this lobe has been setup with a OneMinus*Fresnel
        // We use cosThetaWo to evaluate the fresnel term, as an approximation
        // of what hDotWi would be for the "over" lobe.
        const float cosThetaWo = (!getFresnel()  ?  1.0f  :
                        scene_rdl2::math::max(dot(mN, slice.getWo()), 0.0f));

        const float cosThetaWi =  scene_rdl2::math::max(dot(mN, wi), 0.0f);

        // Forward to the under lobe and apply fresnel
        return (computeScaleAndFresnel(cosThetaWo) *
                computeTransmission(cosThetaWo, cosThetaWi) *
                mUnder->eval(slice, wi, pdf));

    }


    finline scene_rdl2::math::Color sample(const BsdfSlice &slice, float r1, float r2,
                               scene_rdl2::math::Vec3f &wi, float &pdf) const override
    {
        // See eval() for info
        const float cosThetaWo = (!getFresnel()  ?  1.0f  :
                scene_rdl2::math::max(dot(mN, slice.getWo()), 0.0f));

        const scene_rdl2::math::Color underResult = mUnder->sample(slice, r1, r2, wi, pdf);
        const float cosThetaWi = scene_rdl2::math::max(dot(mN, wi), 0.0f);

        // Forward to the under lobe and apply fresnel
        return (computeScaleAndFresnel(cosThetaWo) *
                computeTransmission(cosThetaWo, cosThetaWi) *
                underResult);
    }

    finline scene_rdl2::math::Color albedo(const BsdfSlice &slice) const override
    {
        // See eval() for info
        float cosThetaWo = (!getFresnel()  ?  1.0f  :
                scene_rdl2::math::max(dot(mN, slice.getWo()), 0.0f));

        // Forward to the under lobe and apply fresnel
        return computeScaleAndFresnel(cosThetaWo) * mUnder->albedo(slice);
    }

    finline void differentials(const scene_rdl2::math::Vec3f &wo,
                               const scene_rdl2::math::Vec3f &wi,
                               float r1, float r2, const scene_rdl2::math::Vec3f &dNdx,
                               const scene_rdl2::math::Vec3f &dNdy,
                               scene_rdl2::math::Vec3f &dDdx,
                               scene_rdl2::math::Vec3f &dDdy) const override
    {
        // Forward to the under lobe
        mUnder->differentials(wo, wi, r1, r2, dNdx, dNdy, dDdx, dDdy);
    }

    bool getProperty(Property property, float *dest) const override
    {
        // THINK: interesting question, handle PROPERTY_NORMAL ourselves?
        return mUnder->getProperty(property, dest);
    }

    void show(std::ostream& os, const std::string& indent) const override
    {
        const scene_rdl2::math::Color& scale = getScale();
        const Fresnel * fresnel = getFresnel();
        os << indent << "[UnderBsdfLobe]\n";
        os << indent << "    " << "scale: "
            << scale.r << " " << scale.g << " " << scale.b << "\n";
        mUnder->show(os, indent + "    ");
        if (fresnel) {
            fresnel->show(os, indent + "    ");
        }
    }

protected:
    // Attenuate the under layer based on 'thickness' and 'attenuation color'
    // and distance traveled through the clearcoat layer
    scene_rdl2::math::Color computeTransmission(float cosThetaR1,
                                                float cosThetaR2) const
    {
        if (scene_rdl2::math::isZero(mThickness)) {
            return scene_rdl2::math::sWhite;
        }

        float rcp1 = 0.0f, rcp2 = 0.0f;
        if (!scene_rdl2::math::isZero(cosThetaR1)) rcp1 = scene_rdl2::math::rcp(scene_rdl2::math::abs(cosThetaR1));
        if (!scene_rdl2::math::isZero(cosThetaR2)) rcp2 = scene_rdl2::math::rcp(scene_rdl2::math::abs(cosThetaR2));

        // length of light path within clearcoat
        const float distanceTraveled = mThickness * (rcp1 + rcp2);

        /// The physical formula for attenuation is:
        /// T = exp(absorptionCoeff * distanceTraveled)
        /// For user convenience we allow specifying attenuationColor at unit distance which can be converted
        /// into absorptionCoeff = log(attenuationColor)/1
        /// Using the fact that exp(log(a)) = a,
        /// T = pow(attenuationColor, distanceTraveled)

        const scene_rdl2::math::Color T = scene_rdl2::math::Color(
            scene_rdl2::math::pow(mAttenuationColor.r, distanceTraveled),
            scene_rdl2::math::pow(mAttenuationColor.g, distanceTraveled),
            scene_rdl2::math::pow(mAttenuationColor.b, distanceTraveled));
        return scene_rdl2::math::lerp(scene_rdl2::math::sWhite, T, mAttenuationWeight);
    }

protected:
    BsdfLobe *mUnder;
    scene_rdl2::math::Vec3f mN;

    // Thickness for Attenuation
    float mThickness;
    // Attenuation Color
    scene_rdl2::math::Color mAttenuationColor;
    // Attenuation Weight
    float mAttenuationWeight;
};

//----------------------------------------------------------------------------
ISPC_UTIL_TYPEDEF_STRUCT(UnderBsdfLobe, UnderBsdfLobev);

} // namespace shading
} // namespace moonray

