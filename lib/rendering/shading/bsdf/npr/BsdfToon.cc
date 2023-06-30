// Copyright 2023 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

///
/// @file BsdfToon.cc
/// $Id$
///
//
#include "BsdfToon.h"
#include <scene_rdl2/common/math/MathUtil.h>
#include <moonray/rendering/shading/Util.h>

#include <algorithm>

namespace {

// Rodrigues' rotation formula, assume axis is normalized
scene_rdl2::math::Vec3f
rotateVector(const scene_rdl2::math::Vec3f& v, const scene_rdl2::math::Vec3f& axis, float theta) {
    float ct, st;
    scene_rdl2::math::sincos(theta, &st, &ct);
    return ct * v + st * scene_rdl2::math::cross(axis, v) + scene_rdl2::math::dot(axis, v) * (1.f - ct) * axis;
}

} // end anonymous namespace


namespace moonray {
namespace shading {

using namespace scene_rdl2::math;

HairToonSpecularBsdfLobe::HairToonSpecularBsdfLobe(const Vec3f &N,
                                                   const float intensity,
                                                   const Color& tint,
                                                   int numRampPoints,
                                                   const float* rampPositions,
                                                   const ispc::RampInterpolatorMode* rampInterpolators,
                                                   const float* rampValues,
                                                   HairRLobe* directHairLobe) :
    BsdfLobe(Type(REFLECTION | GLOSSY), DifferentialFlags(0), false,
             PROPERTY_NORMAL | PROPERTY_ROUGHNESS),
    mFrame(N),
    mIntensity(intensity),
    mTint(tint),
    mDirectHairLobe(directHairLobe),
    mIndirectHairLobe(nullptr),
    mIndirectReflectionsIntensity(0.0f)
{
    mRampControl.init(numRampPoints, rampPositions, rampValues, rampInterpolators);
}

HairToonSpecularBsdfLobe::HairToonSpecularBsdfLobe(const Vec3f &N,
                                                   const float intensity,
                                                   const Color& tint,
                                                   int numRampPoints,
                                                   const float* rampPositions,
                                                   const ispc::RampInterpolatorMode* rampInterpolators,
                                                   const float* rampValues,
                                                   HairRLobe* directHairLobe,
                                                   HairRLobe* indirectHairLobe,
                                                   const float indirectReflectionsIntensity) :
    BsdfLobe(Type(REFLECTION | GLOSSY), DifferentialFlags(0), false,
             PROPERTY_NORMAL | PROPERTY_ROUGHNESS),
    mFrame(N),
    mIntensity(intensity),
    mTint(tint),
    mDirectHairLobe(directHairLobe),
    mIndirectHairLobe(indirectHairLobe),
    mIndirectReflectionsIntensity(indirectReflectionsIntensity)
{
    mRampControl.init(numRampPoints, rampPositions, rampValues, rampInterpolators);
}

Color
HairToonSpecularBsdfLobe::sample(const BsdfSlice &slice,
                                 float r1,
                                 float r2,
                                 Vec3f &wi,
                                 float &pdf) const
{
    if (mIndirectHairLobe) {
        return mIndirectHairLobe->sample(slice, r1, r2, wi, pdf) * mIndirectReflectionsIntensity;
    } else {
        pdf = 0.0f;
        return sBlack;
    }
}

Color
HairToonSpecularBsdfLobe::eval(const BsdfSlice &slice,
                               const Vec3f &wi,
                               float *pdf) const
{
    const Color hairEval = mDirectHairLobe->eval(slice, wi, pdf);
    const float ramp = mRampControl.eval1D(scene_rdl2::math::luminance(hairEval));
    return mTint * ramp * mIntensity * scene_rdl2::math::sOneOverPi;
}

void
HairToonSpecularBsdfLobe::differentials(const Vec3f &wo,
                                        const Vec3f &wi,
                                        float r1, float r2,
                                        const Vec3f &dNdx,
                                        const Vec3f &dNdy,
                                        Vec3f &dDdx, Vec3f &dDdy) const
{
    // This function is only called if indirect reflections are enabled.
    // If that's the case we call the internal hair lobe's differentials
    // function.
    if (mIndirectHairLobe) {
        mIndirectHairLobe->differentials(wo, wi, r1, r2, dNdx, dNdy, dDdx, dDdy);
    }
}

void
HairToonSpecularBsdfLobe::show(std::ostream& os,
                               const std::string& indent) const
{
    const Color& scale = getScale();
    const Vec3f& N = mFrame.getN();
    const Fresnel * fresnel = getFresnel();
    os << indent << "[ToonSpecularBsdfLobe]\n";
    os << indent << "    " << "scale: "
        << scale.r << " " << scale.g << " " << scale.b << "\n";
    os << indent << "    " << "N: "
        << N.x << " " << N.y << " " << N.z << "\n";
    os << indent << "    " << "intensity: " << mIntensity << "\n";
    os << indent << "    " << "tint: "
        << mTint.r << " " << mTint.g << " " << mTint.b << "\n";
    if (fresnel) {
        fresnel->show(os, indent + "    ");
    }
    if (mDirectHairLobe) {
        mDirectHairLobe->show(os, indent + "    ");
    }
    if (mIndirectHairLobe) {
        mIndirectHairLobe->show(os, indent + "    ");
    }
}

ToonSpecularBsdfLobe::ToonSpecularBsdfLobe(const Vec3f &N,
                                           const float intensity,
                                           const float roughness,
                                           const Color& tint,
                                           int numRampPoints,
                                           const float* rampPositions,
                                           const ispc::RampInterpolatorMode* rampInterpolators,
                                           const float* rampValues,
                                           const float stretchU,
                                           const float stretchV,
                                           const scene_rdl2::math::Vec3f &dPds,
                                           const scene_rdl2::math::Vec3f &dPdt) :
    BsdfLobe(Type(REFLECTION | GLOSSY), DifferentialFlags(0), false,
             PROPERTY_NORMAL | PROPERTY_ROUGHNESS),
    mFrame(N),
    mIntensity(intensity),
    mRoughness(roughness),
    mTint(tint),
    mStretchU(stretchU),
    mStretchV(stretchV),
    mdPds(dPds),
    mdPdt(dPdt),
    mIndirectLobe(nullptr),
    mIndirectReflectionsIntensity(0.0f)
{    
    mRampControl.init(numRampPoints, rampPositions, rampValues, rampInterpolators);
}

ToonSpecularBsdfLobe::ToonSpecularBsdfLobe(const Vec3f &N,
                                           const float intensity,
                                           const float roughness,
                                           const Color& tint,
                                           int numRampPoints,
                                           const float* rampPositions,
                                           const ispc::RampInterpolatorMode* rampInterpolators,
                                           const float* rampValues,
                                           const float stretchU,
                                           const float stretchV,
                                           const scene_rdl2::math::Vec3f &dPds,
                                           const scene_rdl2::math::Vec3f &dPdt,
                                           CookTorranceBsdfLobe* indirectLobe,
                                           const float indirectReflectionsIntensity) :
    BsdfLobe(Type(REFLECTION | GLOSSY), DifferentialFlags(0), false,
             PROPERTY_NORMAL | PROPERTY_ROUGHNESS),
    mFrame(N),
    mIntensity(intensity),
    mRoughness(roughness),
    mTint(tint),
    mStretchU(stretchU),
    mStretchV(stretchV),
    mdPds(dPds),
    mdPdt(dPdt),
    mIndirectLobe(indirectLobe),
    mIndirectReflectionsIntensity(indirectReflectionsIntensity)
{    
    mRampControl.init(numRampPoints, rampPositions, rampValues, rampInterpolators);
}

Color
ToonSpecularBsdfLobe::sample(const BsdfSlice &slice,
                             float r1,
                             float r2,
                             Vec3f &wi,
                             float &pdf) const
{
    if (mIndirectLobe) {
        return mIndirectLobe->sample(slice, r1, r2, wi, pdf) * mIndirectReflectionsIntensity;
    } else {
        pdf = 0.0f;
        return sBlack;
    }
}

Color
ToonSpecularBsdfLobe::eval(const BsdfSlice &slice,
                           const Vec3f &wi,
                           float *pdf) const
{

    Vec3f N = mFrame.getN();
    const Vec3f& wo = slice.getWo();
    Vec3f R = wi - 2.0f * scene_rdl2::math::dot(wi, N) * N;

    // Rotate N to "stretch" the specular highlight
    const float dot_u_l = scene_rdl2::math::dot(R, mdPds);
    const float dot_u_c = scene_rdl2::math::dot(wo, mdPds);
    const float dot_u = dot_u_l + dot_u_c;
    const float rot_u = scene_rdl2::math::clamp(mStretchU * dot_u, -0.5f, 0.5f);
    N = rotateVector(N, mdPdt, rot_u);

    const float dot_v_l = scene_rdl2::math::dot(R, mdPdt);
    const float dot_v_c = scene_rdl2::math::dot(wo, mdPdt);
    const float dot_v = dot_v_l + dot_v_c;
    const float rot_v = scene_rdl2::math::clamp(-mStretchV * dot_v, -0.5f, 0.5f);
    N = rotateVector(N, mdPds, rot_v);

    const float cosNO = dot(N, wo);
    const float cosNI = dot(N, wi);

    // Prepare for early exit
    if (pdf != NULL) {
        *pdf = 0.0f;
    }

    if (cosNO <= 0.0f || cosNI <= 0.0f) {
        return sBlack;
    }

    // https://docs.google.com/presentation/d/10lYRt3FpQEOu8vrCoGdc_ngJnRXkzFi2B69cEUFkM48/edit
    // Uses acos to get a more linear response for easier ramp remapping
    R = wi - 2.0f * scene_rdl2::math::dot(wi, N) * N;
    float specAngle = scene_rdl2::math::max(0.0f, dot(-wo, R));
    specAngle = 1.0f - 2.0f * (scene_rdl2::math::acos(scene_rdl2::math::clamp(specAngle, -1.0f, 1.0f)) / scene_rdl2::math::sPi);
    const float exponent = (1.0f - mRoughness) * 100.0f;
    specAngle = scene_rdl2::math::pow(specAngle, exponent);

    if (specAngle <= 0.0f) {
        return sBlack;
    }

    if (pdf != NULL) {
        *pdf = 0.5f * scene_rdl2::math::sOneOverPi * specAngle;
    }

    const float ramp = mRampControl.eval1D(specAngle);
    return mTint * ramp * mIntensity * scene_rdl2::math::sOneOverPi;
}

void
ToonSpecularBsdfLobe::differentials(const Vec3f &wo,
                                    const Vec3f &wi,
                                    float r1, float r2,
                                    const Vec3f &dNdx,
                                    const Vec3f &dNdy,
                                    Vec3f &dDdx, Vec3f &dDdy) const
{
    // This function is only called if indirect reflections are enabled.
    // If that's the case we call the internal cook torrance lobe's
    // differentials function.
    if (mIndirectLobe) {
        mIndirectLobe->differentials(wo, wi, r1, r2, dNdx, dNdy, dDdx, dDdy);
    }
}

bool
ToonSpecularBsdfLobe::getProperty(Property property, float *dest) const
{
    bool handled = true;

    switch (property)
    {
    case PROPERTY_ROUGHNESS:
        *dest       = mRoughness;
        *(dest + 1) = mRoughness;
        break;
    case PROPERTY_NORMAL:
        {
            const Vec3f &N = mFrame.getN();
            *dest       = N.x;
            *(dest + 1) = N.y;
            *(dest + 2) = N.z;
        }
        break;
    default:
        handled = BsdfLobe::getProperty(property, dest);
        break;
    }

    return handled;
}

void
ToonSpecularBsdfLobe::show(std::ostream& os,
                           const std::string& indent) const
{
    const Color& scale = getScale();
    const Vec3f& N = mFrame.getN();
    const Fresnel * fresnel = getFresnel();
    os << indent << "[ToonSpecularBsdfLobe]\n";
    os << indent << "    " << "scale: "
        << scale.r << " " << scale.g << " " << scale.b << "\n";
    os << indent << "    " << "N: "
        << N.x << " " << N.y << " " << N.z << "\n";
    os << indent << "    " << "intensity: " << mIntensity << "\n";
    os << indent << "    " << "roughness: " << mRoughness << "\n";
    os << indent << "    " << "tint: "
        << mTint.r << " " << mTint.g << " " << mTint.b << "\n";
    os << indent << "    " << "stretch u: " << mStretchU << "\n";
    os << indent << "    " << "stretch v: " << mStretchU << "\n";
    os << indent << "    " << "dPds: "
        << mdPds.x << " " << mdPds.y << " " << mdPds.z << "\n";
    os << indent << "    " << "dPdt: "
        << mdPdt.x << " " << mdPdt.y << " " << mdPdt.z << "\n";
    if (fresnel) {
        fresnel->show(os, indent + "    ");
    }
    if (mIndirectLobe) {
        mIndirectLobe->show(os, indent + "    ");
    }
}

scene_rdl2::math::Color
ToonBsdfLobe::eval(const BsdfSlice &slice,
                   const Vec3f &wi,
                   float *pdf) const
{
    Vec3f N = mFrame.getN();
    if (mExtendRamp && dot(N, wi) < 0) {
        // Make shading normal perpendicular to light direction
        // and nudge towards light with lerp
        N = normalize(lerp(cross(wi, cross(N, wi)),
                      wi,
                      sEpsilon));  // Must be > 1e-9
    }
    float cosThetaWi = max(dot(N, wi), 0.0f);

    if (pdf != NULL) {
        *pdf = cosThetaWi * sOneOverPi;
    }

    // Note: we assume this lobe has been setup with a OneMinus*Fresnel
    // as we want to use 1 - specular_fresnel. Also notice we use
    // cosThetaWo to evaluate the fresnel term, as an approximation of what
    // hDotWi would be for the specular lobe.
    float cosThetaWo = 1.0f;

    if (getFresnel()) {
        cosThetaWo = max(dot(N, slice.getWo()), 0.0f);
    }

    Color ramp = mRampControl.eval1D(1.0f - cosThetaWi);

    const Color albedo = clamp(ramp*mAlbedo, sBlack, sWhite);
    // Soften hard shadow terminator due to shading normals
    const float Gs = slice.computeShadowTerminatorFix(N, wi);

    // We use the upper bound of 1/2pi as the normalization factor
    // for the maximum possible toon step function
    // TODO: a more accurate normalization factor for the varying
    // step can be computed if so desired.
    Color result = Gs * albedo *
                   computeScaleAndFresnel(cosThetaWo) * sOneOverTwoPi;
    return result;
}

void
ToonBsdfLobe::show(std::ostream& os,
                   const std::string& indent) const
{
    const Color& scale = getScale();
    const Fresnel * const fresnel = getFresnel();
    os << indent << "[ToonBsdfLobe]\n";
    os << indent << "    " << "scale: "
        << scale.r << " " << scale.g << " " << scale.b << "\n";
    os << indent << "    " << "base color: "
        << mAlbedo.r << " " << mAlbedo.g << " " << mAlbedo.b << "\n";
    if (fresnel) {
        fresnel->show(os, indent + "    ");
    }
}

} // namespace shading
} // namespace moonray

