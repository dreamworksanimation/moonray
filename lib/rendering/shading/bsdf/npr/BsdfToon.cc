// Copyright 2023-2024 DreamWorks Animation LLC
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
                                                   const bool enableIndirectReflections,
                                                   const float indirectReflectionsRoughness,
                                                   const float indirectReflectionsIntensity,
                                                   const scene_rdl2::math::Vec3f &hairDir,
                                                   const scene_rdl2::math::Vec2f &hairUV,
                                                   const float mediumIOR,
                                                   const float ior,
                                                   ispc::HairFresnelType fresnelType,
                                                   const float layerThickness,
                                                   const float longShift,
                                                   const float longRoughness) :
    BsdfLobe(Type(REFLECTION | GLOSSY), DifferentialFlags(0), false,
             PROPERTY_NORMAL | PROPERTY_ROUGHNESS),
    mFrame(N),
    mIntensity(intensity),
    mTint(tint),
    mEnableIndirectReflections(enableIndirectReflections),
    mIndirectReflectionsIntensity(indirectReflectionsIntensity),
    mDirectHairLobe(hairDir, hairUV,
                    mediumIOR, ior,
                    fresnelType,
                    layerThickness,
                    longShift, longRoughness,
                    tint),
    mIndirectHairLobe(hairDir, hairUV,
                      mediumIOR, ior,
                      fresnelType,
                      layerThickness,
                      longShift, indirectReflectionsRoughness,
                      tint)
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
    if (mEnableIndirectReflections) {
        return mIndirectHairLobe.sample(slice, r1, r2, wi, pdf) * mIndirectReflectionsIntensity;
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
    const Color hairEval = mDirectHairLobe.eval(slice, wi, pdf);
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
    if (mEnableIndirectReflections) {
        mIndirectHairLobe.differentials(wo, wi, r1, r2, dNdx, dNdy, dDdx, dDdy);
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
    mDirectHairLobe.show(os, indent + "    ");
    if (mEnableIndirectReflections) {
        mIndirectHairLobe.show(os, indent + "    ");
    }
}

ToonSpecularBsdfLobe::ToonSpecularBsdfLobe(const Vec3f &N,
                                           const float intensity,
                                           const float fresnelBlend,
                                           const Color& tint,
                                           float rampInputScale,
                                           int numRampPoints,
                                           const float* rampPositions,
                                           const ispc::RampInterpolatorMode* rampInterpolators,
                                           const float* rampValues,
                                           const float stretchU,
                                           const float stretchV,
                                           const scene_rdl2::math::Vec3f &dPds,
                                           const scene_rdl2::math::Vec3f &dPdt,
                                           const bool enableIndirectReflections,
                                           const float indirectReflectionsRoughness,
                                           const float indirectReflectionsIntensity,
                                           Fresnel * fresnel) :
    BsdfLobe(Type(REFLECTION | GLOSSY), DifferentialFlags(0), false,
             PROPERTY_NORMAL | PROPERTY_ROUGHNESS),
    mFrame(N),
    mIntensity(intensity),
    mFresnelBlend(fresnelBlend),
    mNormalization(1.0f),
    mTint(tint),
    mRampInputScale(rampInputScale),
    mStretchU(stretchU),
    mStretchV(stretchV),
    mdPds(dPds),
    mdPdt(dPdt),
    mEnableIndirectReflections(enableIndirectReflections),
    mIndirectReflectionsIntensity(indirectReflectionsIntensity),
    mIndirectLobe(N, indirectReflectionsRoughness)
{
    mRampControl.init(numRampPoints,
                      rampPositions,
                      rampValues,
                      rampInterpolators);

    mIndirectLobe.setFresnel(fresnel);

    // Approximate a normalization term that attempt to preserve energy
    // under different ramp settings.  Narrow/tight ramps should produce
    // small/bright highlights, and wider/broad ramps should produce dimmer
    // highlights.

    // We'll divide the ramp into a number of slices and for each slice
    // compute the area of the spherical segment representing that slice,
    // multipled by the ramp value representing that segment.

    // The toon ramp's input in [0,1) in eval() represents theta in [0,pi/2).
    // (There is an optimization/approximation that attempts to avoid acos.)
    constexpr int numSegments = 64; // this approximation is coarse, maybe too coarse.
    constexpr float dx = 1.0f / numSegments;
    constexpr float dx_half = 0.5f * dx;
    float totalArea = 0.0f;
    float rampAvg = 0.0f;

    for (int i = 0; i < numSegments; ++i) {
        // Scale segment area by ramp eval.
        const float ramp = mRampControl.eval1D((i * dx + dx_half) / mRampInputScale);
        rampAvg += ramp;

        if (isZero(ramp)) {
            continue;
        }

        const float theta0 = i * dx * sHalfPi;
        const float theta1 = (i+1) * dx * sHalfPi;

        // Area of spherical caps using r and theta:
        // A = 2 * pi * r*r (1 - cosTheta), where r = 1
        // The segment area is the difference of the spherical caps
        // defined at the bottom and top edge of the segment
        const float a0 = sTwoPi * (1.0f - scene_rdl2::math::cos(theta0));
        const float a1 = sTwoPi * (1.0f - scene_rdl2::math::cos(theta1));

        const float segmentArea = (a1 - a0) * ramp;
        totalArea += segmentArea;
    }

    // Use the pre-sampled ramp to determine average value and scale the
    // fresnel's weight to try to conserve energy.
    rampAvg /= numSegments;
    fresnel->setWeight(fresnel->getWeight() * rampAvg);

    if (!isZero(totalArea)) {
        mNormalization = sTwoPi / totalArea; // reciprocal
    }
}

Color
ToonSpecularBsdfLobe::sample(const BsdfSlice &slice,
                             float r1,
                             float r2,
                             Vec3f &wi,
                             float &pdf) const
{
    if (mEnableIndirectReflections) {
        return mIndirectLobe.sample(slice, r1, r2, wi, pdf) * mIndirectReflectionsIntensity;
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
    Vec3f R = wi - 2.0f * dot(wi, N) * N;

    // Rotate N to "stretch" the specular highlight
    const float dot_u_l = dot(R, mdPds);
    const float dot_u_c = dot(wo, mdPds);
    const float dot_u = dot_u_l + dot_u_c;
    const float rot_u = clamp(mStretchU * dot_u, -0.5f, 0.5f);
    N = rotateVector(N, mdPdt, rot_u);

    const float dot_v_l = dot(R, mdPdt);
    const float dot_v_c = dot(wo, mdPdt);
    const float dot_v = dot_v_l + dot_v_c;
    const float rot_v = clamp(-mStretchV * dot_v, -0.5f, 0.5f);
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

    // Note: we assume this lobe has been setup with a OneMinus*Fresnel
    // as we want to use 1 - specular_fresnel. Also notice we use
    // cosThetaWo to evaluate the fresnel term, as an approximation of what
    // hDotWi would be for the specular lobe.
    float cosThetaWo = 1.0f;

    if (getFresnel()) {
        cosThetaWo = max(dot(N, slice.getWo()), 0.0f);
    }

    // Reflection vector using modified N
    R = wi - 2.0f * dot(wi, N) * N;

    // acos approximation
    const float thetaRO = scene_rdl2::math::pow(1.0f - clamp(dot(-wo,  R), 0.0f, 1.0f), 0.56f);

    if (thetaRO <= 0.0f) {
        return sBlack;
    }

    const float ramp = mRampControl.eval1D(thetaRO / mRampInputScale);
    if (pdf != NULL) {
        *pdf = sOneOverTwoPi * ramp;
    }

    // ad-hoc "shadow/masking" terms
    const Color f = lerp(getScale(),
                         mNormalization * computeScaleAndFresnel(cosThetaWo),
                         mFresnelBlend);
    return ramp * mIntensity * mTint * sOneOverPi * f;
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
    if (mEnableIndirectReflections) {
        mIndirectLobe.differentials(wo, wi, r1, r2, dNdx, dNdy, dDdx, dDdy);
    }
}

bool
ToonSpecularBsdfLobe::getProperty(Property property, float *dest) const
{
    bool handled = true;

    switch (property)
    {
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
    if (mEnableIndirectReflections) {
        mIndirectLobe.show(os, indent + "    ");
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

    Color ramp = mRampControl.eval1D((1.0f - cosThetaWi) / mRampInputScale);

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

