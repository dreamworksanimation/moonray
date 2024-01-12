// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

///
/// @file BsdfFlatDiffuse.h
/// $Id$
///

#pragma once

#include <moonray/rendering/shading/bsdf/BsdfOrenNayar.h>
#include <moonray/rendering/shading/bsdf/Bsdf.h>
#include <moonray/rendering/shading/bsdf/BsdfSlice.h>
#include <moonray/rendering/shading/PbrValidity.h>
#include <moonray/rendering/shading/Util.h>

#include <moonray/rendering/shading/ispc/bsdf/npr/BsdfFlatDiffuse_ispc_stubs.h>

#include <scene_rdl2/common/math/Color.h>
#include <scene_rdl2/common/math/ReferenceFrame.h>
#include <scene_rdl2/common/math/Vec3.h>
#include <scene_rdl2/common/platform/IspcUtil.h>

namespace moonray {
namespace shading {

static finline float
getOrenNayarFalloffIntensity(const scene_rdl2::math::Vec3f& n,
                             const scene_rdl2::math::Vec3f& v,
                             const scene_rdl2::math::Vec3f& l,
                             const float a,
                             const float b,
                             const float terminatorShift)
{
    // We are using the Fujii implementation or Oren-Nayar.
    // The PBRT implementation was also tested but resulted
    // in more energy loss.
    // https://mimosa-pudica.net/improved-oren-nayar.html
    // Note: For the flat diffuse lobe type we introduce an
    // additional terminatorShift control that reshapes the n dot l
    // response
    const float nl = scene_rdl2::math::max(
        0.0f, scene_rdl2::math::cos(
            scene_rdl2::math::min(
                scene_rdl2::math::sHalfPi, scene_rdl2::math::acos(
                    scene_rdl2::math::clamp(scene_rdl2::math::dot(n, l), -1.0f, 1.0f)) *
                terminatorShift)));
    const float nv = scene_rdl2::math::max(scene_rdl2::math::dot(n, v), 0.0f);
    const float s = scene_rdl2::math::dot(l, v) - nl * nv;

    float t = 1.0f;
    if (s > 0.0f) {
        t = scene_rdl2::math::max(nl, nv) + scene_rdl2::math::sEpsilon;
    }

    return nl * (a + b * s/t);
}

///
/// @class FlatDiffuseBsdfLobe BsdfFlatDiffuse.h <shading/bsdf/npr/BsdfFlatDiffuse.h>
/// @brief  This is a totally made-up NPR "flat diffuse" BRDF that
///         matches Oren-Nayar when flatness = 0.  It works by bending
///         the normal towards the light as flatness goes from 0 to 1.
/// 
class FlatDiffuseBsdfLobe : public OrenNayarBsdfLobe
{
public:
    // Constructor / Destructor
    FlatDiffuseBsdfLobe(const scene_rdl2::math::Vec3f &N,
                        const scene_rdl2::math::Color& albedo,
                        float roughness,
                        float terminatorShift,
                        float flatness,
                        float flatnessFalloff,
                        bool reflection) :
        OrenNayarBsdfLobe(N, albedo, roughness, reflection),
        mTerminatorShift(terminatorShift),
        mFlatness(flatness),
        mFlatnessFalloff(flatnessFalloff)
    {}

    // BsdfLobe API
    finline scene_rdl2::math::Color eval(const BsdfSlice &slice, const scene_rdl2::math::Vec3f &wi, float *pdf = NULL) const override
    {
        // When flatness approaches 1.0 it tends to produce an extremely harsh terminator,
        // which can lead to polygonal artifacts where self-shadowing occurs.  This remapping
        // attempts to apply a targeted softening effect at the terminator - similar to the
        // "shadow terminator fix" that we used to soften bump mapping

        // Calculate the flatness falloff off by remapping
        // the angle between the light and the normal.
        // Desmos expression:
        // https://www.desmos.com/calculator/z1iufecqmf
        const float terminatorShift = mTerminatorShift + 1.0f;
        const float theta = scene_rdl2::math::min(scene_rdl2::math::sHalfPi, terminatorShift *
                            scene_rdl2::math::acos(scene_rdl2::math::clamp(scene_rdl2::math::dot(mFrame.getN(), wi), -1.0f, 1.0f)));
        const float a = 1.0f - mFlatnessFalloff;
        // The power of "a" controls the linearity of the falloff parameter.
        // Raising to the power of 4 slightly biases the control towards
        // the low end giving more control over a low falloff value.  The
        // constant 100.0 controls how sharply the falloff occurs when the
        // parameter is set to 1.0.
        const float b = 1.0f + a * a * a * a * 100.0f;
        const float c = scene_rdl2::math::sHalfPi / b;
        const float d = c - scene_rdl2::math::sHalfPi;
        const float t = (theta > scene_rdl2::math::sHalfPi - c) ? scene_rdl2::math::cos((theta + d) * b) : 1.0f;
        const float flatness = mFlatness * t;
        const scene_rdl2::math::Vec3f flatN = scene_rdl2::math::normalize(scene_rdl2::math::lerp(mFrame.getN(), wi, flatness));

        const float cosThetaWi = getOrenNayarFalloffIntensity(flatN,
                                                              slice.getWo(), wi,
                                                              mA, mB,
                                                              terminatorShift);

        if (pdf != NULL) {
            *pdf = cosThetaWi * scene_rdl2::math::sOneOverPi;
        }

        // Note: we assume this lobe has been setup with a OneMinus*Fresnel
        // as we want to use 1 - specular_fresnel. Also notice we use
        // cosThetaWo to evaluate the fresnel term, as an approximation of what
        // hDotWi would be for the specular lobe.
        float cosThetaWo = 1.0f;
        if (getFresnel()) {
            const scene_rdl2::math::Vec3f N = (matchesFlag(REFLECTION)) ? flatN : -flatN;
            cosThetaWo = scene_rdl2::math::max(dot(N, slice.getWo()), 0.0f);
        }

        // Soften hard shadow terminator due to shading normals
        const float Gs = (matchesFlag(REFLECTION)) ?
                         slice.computeShadowTerminatorFix(flatN, wi) :
                         1.0f;

        const scene_rdl2::math::Color result =  mAlbedo * scene_rdl2::math::sOneOverPi * Gs *
               computeScaleAndFresnel(cosThetaWo) *
               (slice.getIncludeCosineTerm()  ?  cosThetaWi  :  1.0f);

        // This is an ad hoc factor that tries to minimize energy
        // loss while also minimizing the difference in brightness
        // compared with lambertian (i.e. flatnesss = 0).
        const float normalizationFactor = scene_rdl2::math::lerp(1.0f,
                                                     0.75f,
                                                     scene_rdl2::math::bias(mFlatness, 0.75f));

        return result * normalizationFactor;
    }

    void show(std::ostream& os, const std::string& indent) const override
    {
        const scene_rdl2::math::Color& scale = getScale();
        const scene_rdl2::math::Vec3f& N = mFrame.getN();
        const Fresnel * const fresnel = getFresnel();

        os << indent << "[FlatDiffuseBsdfLobe] "
            << (matchesFlag(REFLECTION) ? "(reflection)" : "(transmission)") << "\n";
        os << indent << "    " << "scale: "
            << scale.r << " " << scale.g << " " << scale.b << "\n";
        os << indent << "    " << "N: "
            << N.x << " " << N.y << " " << N.z << "\n";
        os << indent << "    " << "albedo: "
            << mAlbedo.r << " " << mAlbedo.g << " " << mAlbedo.b << "\n";
        os << indent << "roughness: " << mRoughness << "\n";
        os << indent << "terminator shift: " << mTerminatorShift << "\n";
        os << indent << "flatness: " << mFlatness << "\n";
        os << indent << "flatness falloff: " << mFlatnessFalloff << "\n";
        if (fresnel) {
            fresnel->show(os, indent + "    ");
        }
    }

private:
    float mTerminatorShift;
    float mFlatness;
    float mFlatnessFalloff;
};


//----------------------------------------------------------------------------
ISPC_UTIL_TYPEDEF_STRUCT(FlatDiffuseBsdfLobe, FlatDiffuseBsdfLobev);

} // namespace shading
} // namespace moonray

