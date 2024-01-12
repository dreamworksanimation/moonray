// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

///
/// @file BsdfHairLobe.cc
/// $Id$
///

#include "HairState.h"
#include "BsdfHairDiffuse.h"
#include <moonray/rendering/shading/bsdf/BsdfSlice.h>
#include <moonray/rendering/shading/Util.h>

namespace moonray {
namespace shading {


using namespace scene_rdl2::math;

// Uncomment this to use uniform sampling
// TODO: The importance sampling that came with this lobe is slightly
// biased compared to uniform sampling the sphere. Also in the case where
// the mMaxDiffAngle < 90, directions almost parallel to the hair cause
// a lot of noise (not so small eval() divided by very small pdf() values).
// Uniform sampling fixes both issues, and finding a better non-biased
// importance sampling scheme is left for future work.
//#define PBR_HAIR_DIFFUSE_USE_UNIFORM_SAMPLING

//----------------------------------------------------------------------------

// Cheap approximation of the diffuse albedo.
Color HairDiffuseLobe::albedo(const BsdfSlice &slice) const
{
    Vec3f I = slice.getWo();
    float cosTheta = dot(I, mHairDir);
    return getScale() * scene_rdl2::math::sqrt(scene_rdl2::math::abs(1.0f - cosTheta * cosTheta)) / sPiSqr;
}


Color HairDiffuseLobe::evalBsdf(const Vec3f& wo,
                                const Vec3f &wi,
                                bool includeCosineTerm) const
{
    const Vec3f &hairDir = mHairDir;

    const Vec3f hd_cross_i = cross(hairDir, wi);

    Color hairColor = mHairColorRefl;
    if (!isEqual(mHairColorRefl, mHairColorTrans)) {

        const Vec3f hd_cross_o = cross(hairDir, wo);

        const float denominator =
            scene_rdl2::math::max(length(hd_cross_i) * length(hd_cross_o),
                                  sEpsilon);

        // Compute the proportion of transmission color vs. reflection color.
        const float kappa = dot(hd_cross_i, hd_cross_o) / denominator;
        const float frontIntensity = (1.0f + kappa) / 2.0f;
        const float backIntensity = 1.0f - frontIntensity;

        hairColor = frontIntensity * mHairColorRefl +
                    backIntensity  * mHairColorTrans;
    }

    // Here we want a lambertian reflectivity and we need to include
    // the cosine term per the slice. If we don't include the cosine term, then
    // the mMaxDiffAngle feature is incompatible (a hack anyways) and not
    // enabled at all.
    //result /= cosTerm;
    Color result = hairColor;
    if (includeCosineTerm) {
        // Note: cos(thetaI) == sin(sin_hd_i)
        float min_sin_hd_i = scene_rdl2::math::cos(mMaxDiffAngle);
        float cos_hd_i = dot(hairDir, wi);
        float sin_hd_i = max(min_sin_hd_i, scene_rdl2::math::sqrt(scene_rdl2::math::abs(1 - cos_hd_i * cos_hd_i)));
        result *= sin_hd_i / sPiSqr;
    } else {
        result *= 1.0f / sPiSqr;
    }

    return getScale() * result;
}


float HairDiffuseLobe::evalPdf(const Vec3f &wi) const
{
#ifdef PBR_HAIR_DIFFUSE_USE_UNIFORM_SAMPLING

    return 1.0f / (4.0f * sPi);

#else
    Vec3f hairDir = mHairDir;
    float cosTheta = dot(wi, hairDir);
    float pdf = scene_rdl2::math::sqrt(std::max(1.0f - cosTheta*cosTheta, 0.f));

    return pdf / sPiSqr;
#endif
}

Color HairDiffuseLobe::eval(const BsdfSlice &slice, const Vec3f &wi, float *pdf) const
{
    if (pdf != nullptr) {
        *pdf = evalPdf(wi);
    }
    return evalBsdf(slice.getWo(),
                    wi,
                    slice.getIncludeCosineTerm());
}


Color HairDiffuseLobe::sample(const BsdfSlice &slice, float r1, float r2, Vec3f &wi, float &pdf) const
{
    HairState hairState(slice.getWo(),
                        mHairDir);

#ifdef PBR_HAIR_DIFFUSE_USE_UNIFORM_SAMPLING

    Vec3f uvw = shading::sampleSphereUniform(r1, r2);
    wi = (uvw[0] * mHairDir) +
         (uvw[1] * hairState.hairNormal()) +
         (uvw[2] * hairState.hairBinormal());

    // We shouldn't need to normalize!
    MNRY_ASSERT(isNormalized(wi));

    pdf = 1.0f / (4.0f * sPi);

#else

    float sinPhi, cosPhi;
    
    // Basically, we're sampling a Lambertian in the longitudinal
    // plane and uniform circular in the azimuthal plane.
    float u = 2.0f * (r1 - 0.5f);
    float v = r2;

    float sinTheta = u;
    float cosTheta = scene_rdl2::math::sqrt(1.0f - u*u);
    float phi = v * sTwoPi;
    sincos(phi, &sinPhi, &cosPhi);

    // Compute the light direction vector for shading.
    float uWgt = sinTheta;
    float vWgt = cosTheta * sinPhi;
    float wWgt = cosTheta * cosPhi;
    wi = (uWgt * mHairDir) +
         (vWgt * hairState.hairNormal()) +
         (wWgt * hairState.hairBinormal());
    wi.normalize();

    pdf = cosTheta / sPiSqr;
#endif
    return evalBsdf(slice.getWo(),
                    wi,
                    slice.getIncludeCosineTerm());
}

bool
HairDiffuseLobe::getProperty(Property property, float *dest) const
{
    bool handled = true;

    switch (property)
    {
    case PROPERTY_COLOR:
        {
            *dest       = mHairColorRefl[0];
            *(dest + 1) = mHairColorRefl[1];
            *(dest + 2) = mHairColorRefl[2];
        }
        break;
    case PROPERTY_ROUGHNESS:
        {
            *dest       = 1.0f;
            *(dest + 1) = 1.0f;
        }
        break;
    case PROPERTY_NORMAL:
        {
            *dest       = mHairDir.x;
            *(dest + 1) = mHairDir.y;
            *(dest + 2) = mHairDir.z;
        }
        break;
    case PROPERTY_PBR_VALIDITY:
        {
            // TODO: decide if refl/trans colors are valid
            *dest       = 0.0f;
            *(dest + 1) = 1.0f;
            *(dest + 2) = 0.0f;
        }
    default:
        handled = BsdfLobe::getProperty(property, dest);
        break;
    }

    return handled;
}

void
HairDiffuseLobe::differentials(const Vec3f &wo, const Vec3f &wi,
                               float r1, float r2, const Vec3f &dNdx, const Vec3f &dNdy,
                               Vec3f &dDdx, Vec3f &dDdy) const
{
    Vec3f H;
    if (!computeNormalizedHalfVector(wo, wi, H)) {
        H = mHairDir;
    }
    computeReflectionDirectionDifferentialNoCurvature(wo, wi, H, dDdx, dDdy);
    // TODO: This needs to be tested
    dDdx *= 100.0f;
    dDdy *= 100.0f;
}

//----------------------------------------------------------------------------

} // namespace shading
} // namespace moonray

