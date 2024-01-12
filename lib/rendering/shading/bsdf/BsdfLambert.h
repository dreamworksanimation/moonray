// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

///
/// @file BsdfLambert.h
/// $Id$
///

#pragma once

#include "Bsdf.h"
#include "BsdfSlice.h"
#include <moonray/rendering/shading/PbrValidity.h>
#include <moonray/rendering/shading/Util.h>

#include <moonray/rendering/shading/ispc/bsdf/BsdfLambert_ispc_stubs.h>

#include <scene_rdl2/common/math/Color.h>
#include <scene_rdl2/common/math/ReferenceFrame.h>
#include <scene_rdl2/common/math/Vec3.h>
#include <scene_rdl2/common/platform/IspcUtil.h>

namespace moonray {
namespace shading {

// See Substance for Unity: Chapter 01-01 Understanding PBR
// https://youtu.be/ueC2qGzWrgQ?t=914
#define LAMBERT_PBR_VALIDITY_MIN 0.031896
#define LAMBERT_PBR_VALIDITY_MAX 0.871367

///
/// @class LambertBsdfLobe BsdfLambert.h <shading/bsdf/BsdfLambert.h>
/// @brief The Lambert diffuse brdf (f = kd / PI)
/// 
class LambertBsdfLobe : public BsdfLobe
{
public:
    // Constructor / Destructor
    LambertBsdfLobe(const scene_rdl2::math::Vec3f &N, const scene_rdl2::math::Color& albedo, bool reflection /* = true */) :
        BsdfLobe(Type((reflection ? REFLECTION : TRANSMISSION) | DIFFUSE),
                 DifferentialFlags(IGNORES_INCOMING_DIFFERENTIALS), false,
                 PROPERTY_NORMAL | PROPERTY_ROUGHNESS | PROPERTY_PBR_VALIDITY),
        mFrame(N),
        mAlbedo(albedo) {}

    // BsdfLobe API
    finline scene_rdl2::math::Color eval(const BsdfSlice &slice, const scene_rdl2::math::Vec3f &wi, float *pdf = NULL) const override
    {
        const float cosThetaWi = scene_rdl2::math::max(dot(mFrame.getN(), wi), 0.0f);
        if (pdf != NULL) {
            *pdf = cosThetaWi * scene_rdl2::math::sOneOverPi;
        }

        // Note: we assume this lobe has been setup with a OneMinus*Fresnel
        // as we want to use 1 - specular_fresnel. Also notice we use
        // cosThetaWo to evaluate the fresnel term, as an approximation of what
        // hDotWi would be for the specular lobe.
        float cosThetaWo = 1.0f;

        if (getFresnel()) {
            const scene_rdl2::math::Vec3f N = (matchesFlag(REFLECTION)) ? mFrame.getN() : -mFrame.getN();
            cosThetaWo = scene_rdl2::math::max(dot(N, slice.getWo()), 0.0f);
        }

        // Soften hard shadow terminator due to shading normals
        const float Gs = (matchesFlag(REFLECTION)) ?
                         slice.computeShadowTerminatorFix(mFrame.getN(), wi) :
                         1.0f;

        return Gs * mAlbedo * computeScaleAndFresnel(cosThetaWo) * scene_rdl2::math::sOneOverPi *
                (slice.getIncludeCosineTerm()  ?  cosThetaWi  :  1.0f);
    }


    finline scene_rdl2::math::Color sample(const BsdfSlice &slice, float r1, float r2,
            scene_rdl2::math::Vec3f &wi, float &pdf) const override
    {
        wi = mFrame.localToGlobal(sampleLocalHemisphereCosine(r1, r2));
        return eval(slice, wi, &pdf);
    }


    finline scene_rdl2::math::Color albedo(const BsdfSlice &slice) const override
    {
        float cosThetaWo = scene_rdl2::math::max(dot(mFrame.getN(), slice.getWo()), 0.0f);
        return computeScaleAndFresnel(cosThetaWo) * mAlbedo;
    }

    // no surface curvature required
    finline void differentials(const scene_rdl2::math::Vec3f &wo, const scene_rdl2::math::Vec3f &wi,
            float r1, float r2, const scene_rdl2::math::Vec3f &dNdx, const scene_rdl2::math::Vec3f &dNdy,
            scene_rdl2::math::Vec3f &dDdx, scene_rdl2::math::Vec3f &dDdy) const override
    {
        // The hemisphere cosine sampling direction derivative seems like a very
        // good approximation to the full derivative for diffuse sampling. This
        // is why we ignore the input differentials.

        // TODO: How can we avoid computing these twice (once here, once in
        // sample()->sampleLocalHemisphereCosine()) ?
        localHemisphereCosineDifferentials(r1, r2, dDdx, dDdy);

        // The differentials form rectangles which get long and thin close
        // to the hemisphere pole and equator. We prefer having ray
        // differentials that form a square, but preserve the ray footprint.
        squarifyRectangle(dDdx, dDdy);

        dDdx = mFrame.localToGlobal(dDdx);
        dDdy = mFrame.localToGlobal(dDdy);
    }

    bool getProperty(Property property, float *dest) const override
    {
        bool handled = true;

        switch (property)
        {
        case PROPERTY_ROUGHNESS:
            *dest       = 1.0f;
            *(dest + 1) = 1.0f;
            break;
        case PROPERTY_NORMAL:
            {
                const scene_rdl2::math::Vec3f &N = mFrame.getN();
                *dest       = N.x;
                *(dest + 1) = N.y;
                *(dest + 2) = N.z;
            }
            break;
        case PROPERTY_PBR_VALIDITY:
            {
                scene_rdl2::math::Color res = computeAlbedoPbrValidity(mAlbedo);
                *dest       = res.r;
                *(dest + 1) = res.g;
                *(dest + 2) = res.b;
            }
        break;
        case PROPERTY_COLOR:
            {
                // Special case handling for lambert color property since albedo isn't included
                // in mScale anymore.
                scene_rdl2::math::Color result = mAlbedo * computeScaleAndFresnel(1.f);
                *dest       = result.r;
                *(dest + 1) = result.g;
                *(dest + 2) = result.b;
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
        const scene_rdl2::math::Vec3f& N = mFrame.getN();
        const Fresnel * const fresnel = getFresnel();

        os << indent << "[LambertBsdfLobe] "
            << (matchesFlag(REFLECTION) ? "(reflection)" : "(transmission)") << "\n";
        os << indent << "    " << "scale: "
            << scale.r << " " << scale.g << " " << scale.b << "\n";
        os << indent << "    " << "N: "
            << N.x << " " << N.y << " " << N.z << "\n";
        os << indent << "    " << "albedo: "
            << mAlbedo.r << " " << mAlbedo.g << " " << mAlbedo.b << "\n";
        if (fresnel) {
            fresnel->show(os, indent + "    ");
        }
    }

protected:
    scene_rdl2::math::ReferenceFrame mFrame;
    scene_rdl2::math::Color mAlbedo;
};


//----------------------------------------------------------------------------
ISPC_UTIL_TYPEDEF_STRUCT(LambertBsdfLobe, LambertBsdfLobev);

} // namespace shading
} // namespace moonray

