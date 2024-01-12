// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

///
///

#pragma once

#include "Bsdf.h"
#include "BsdfSlice.h"
#include <moonray/rendering/shading/Util.h>
#include <moonray/rendering/shading/Iridescence.h>
#include <moonray/rendering/shading/RampControl.h>

#include <scene_rdl2/common/math/Color.h>
#include <scene_rdl2/common/math/ColorSpace.h>
#include <scene_rdl2/common/math/MathUtil.h>
#include <scene_rdl2/common/math/Vec3.h>
#include <scene_rdl2/common/platform/IspcUtil.h>

namespace moonray {
namespace shading {

namespace {

finline float
fitInRange(float value, float oldMin, float oldMax, float newMin, float newMax) {
    return ((newMax - newMin) * (value - oldMin) / (oldMax - oldMin)) + newMin;
}

scene_rdl2::math::Color
computeIridescenceColor(float nDotV, const scene_rdl2::math::Color& primaryColor, const scene_rdl2::math::Color& secondaryColor,
                        bool flipHue, float thickness, float exponent)
{
    using scene_rdl2::math::rgbToHue;
    using scene_rdl2::math::hueToRgb;
    // 1 - nDotV, to reverse the color spectrum between primaryColor and secondaryColor
    // so that secondaryColor appears at grazing angles and primaryColor appears straight on.
    float index = scene_rdl2::math::pow(scene_rdl2::math::abs(1.f - nDotV), exponent) * thickness;
    index = scene_rdl2::math::fmod(index, 1.f);

    scene_rdl2::math::Color iridescence;
    if (scene_rdl2::math::isEqual(primaryColor, secondaryColor)) {
        // If colors are equal, interpolate around the hue spectrum back to the same color.
        if (flipHue) {
            // Nothing to interpolate in the counter-clockwise direction.
            iridescence = primaryColor;
        } else {
            // Interpolate clockwise around the whole hue wheel back to the same color.
            const float huePrimary = scene_rdl2::math::rgbToHue(primaryColor);
            // Remap range to [hue value, hue value + 1] so that the primary color appears
            // at near normal and glancing angles and the rest of the hues are interpolated between.
            const float clampedIndex = fitInRange(index, 0.f, 1.f, huePrimary, huePrimary + 1.f);
            iridescence = scene_rdl2::math::hueToRgb(clampedIndex);
        }
    } else {
        // If colors are not equal, interpolate a range around the hue wheel and lerp from
        // the secondaryColor back to the primaryColor.
        float huePrimary = scene_rdl2::math::rgbToHue(primaryColor);
        float hueSecondary = rgbToHue(secondaryColor);
        // If the hue is not flipped but the secondary color is greater than the primary,
        // we need to wrap around the secondary color to maintain clockwise interpolation around the wheel.
        // If the hue is flipped but primary is less than secondary, we need to wrap around
        // the primary color so that it interpolates in reverse, counter-clockwise.
        if (hueSecondary < huePrimary && !flipHue) {
            hueSecondary += 1.f;
        } else if (huePrimary < hueSecondary && flipHue) {
            huePrimary += 1.f;
        }

        // Lerp back to the primary color from the secondary color using part of the
        // nDotV index. The hue spectrum is normally divided into 6 sections or 1/6ths (0.166667),
        // but since the hue range is limited, these sections become proportionally larger.
        // The lerpBackDistance is computed by inverting this value and adding 1 to add another "section,"
        // and then inverting it again to make the lerp distance 1/x, for x = number of sections.
        const float lerpBackDistance = 1.f / (scene_rdl2::math::abs(hueSecondary - huePrimary) / 0.166667 + 1.f);
        const float spectrumDistance = 1.f - lerpBackDistance;
        const float clampedIndex = fitInRange(index, 0.f, spectrumDistance, huePrimary, hueSecondary);
        if (index < spectrumDistance) {
            iridescence = hueToRgb(clampedIndex);
        } else {
            iridescence = scene_rdl2::math::lerp(hueToRgb(hueSecondary), hueToRgb(huePrimary), (index - spectrumDistance)/lerpBackDistance );
        }
    }

    return iridescence;
}

finline bool
computeAverageOfHalfVectorCosines(const scene_rdl2::math::Vec3f& wo, const scene_rdl2::math::Vec3f& wi, float& avgCos)
{
    scene_rdl2::math::Vec3f m;
    if (!computeNormalizedHalfVector(wo, wi, m)) {
        return false;
    }
    const float cosMO = dot(m, wo);
    const float cosMI = dot(m, wi);

    // There is a slight difference between cosMO (hDotV) and cosMI (hDotL) which causes the unit test
    // to fail. Averaging the two circumvents this issue and results in identical results
    // as using just cosMO.
    avgCos = (cosMO + cosMI) * 0.5f;
    return true;
}

} // anonymous namespace

///
/// @class IridescenceBsdfLobe  BsdfIridescence.h <shading/bsdf/BsdfIridescence.h>
/// @brief Wrapper BSDF that takes a specular bsdf under-lobe
/// and multiplies it by a non-physical artist friendly iridescence color.
class IridescenceBsdfLobe : public BsdfLobe
{
public:
    IridescenceBsdfLobe(BsdfLobe* lobe,
                        const scene_rdl2::math::Vec3f& N,
                        const float iridescence,
                        ispc::SHADING_IridescenceColorMode colorControl,
                        const scene_rdl2::math::Color &primary,
                        const scene_rdl2::math::Color &secondary,
                        const bool flipHue,
                        const ispc::ColorRampControlSpace rampInterpolationMode,
                        const int numRampPoints,
                        const float* const rampPositions,
                        const ispc::RampInterpolatorMode* const rampInterpolators,
                        const scene_rdl2::math::Color* const rampColors,
                        const float thickness,
                        const float exponent,
                        const float iridescenceAt0,
                        const float iridescenceAt90) :
        BsdfLobe(lobe->getType(), lobe->getDifferentialFlags(),
                 lobe->getIsSpherical(), lobe->getPropertyFlags()),
        mChildLobe(lobe), mN(N), mIridescence(iridescence), mColorControl(colorControl),
        mPrimary(primary), mSecondary(secondary), mFlipHue(flipHue),
        mThickness(thickness), mExponent(exponent),
        mIridescenceAt0(iridescenceAt0), mIridescenceAt90(iridescenceAt90)
    {
        if (mColorControl == ispc::SHADING_IRIDESCENCE_COLOR_USE_RAMP) {
            mRampControl.init(numRampPoints, rampPositions, rampColors, rampInterpolators, rampInterpolationMode);
        }
    }

    // BsdfLobe API
    finline scene_rdl2::math::Color eval(const BsdfSlice &slice,
                                         const scene_rdl2::math::Vec3f &wi,
                                         float *pdf = NULL) const override
    {
        // Mirror lobe will not be evaluated!
        if (mChildLobe->getType() & MIRROR) {
            if (pdf) *pdf = 0.0f;
            return scene_rdl2::math::sBlack;
        }

        // Glossy lobe
        // Evaluate the underlying bsdf
        const scene_rdl2::math::Color lobeResult = mChildLobe->eval(slice, wi, pdf);

        // use half vector dot products for iridescence computation
        float avgCos;
        if (!computeAverageOfHalfVectorCosines(slice.getWo(), wi, avgCos)) {
            return scene_rdl2::math::sBlack;
        }
        return tintWithIridescence(avgCos, lobeResult) * computeScaleAndFresnel(avgCos);
    }
    finline scene_rdl2::math::Color sample(const BsdfSlice &slice, float r1, float r2,
            scene_rdl2::math::Vec3f &wi, float &pdf) const override
    {
        // Sample the child spec lobe
        const scene_rdl2::math::Color res = mChildLobe->sample(slice, r1, r2, wi, pdf);

        // Mirror
        // *Optimization* : using NdotV for iridescence color computation
        if (mChildLobe->getType() & MIRROR) {
            const float cosThetaNV = dot(mN, slice.getWo());
            return tintWithIridescence(cosThetaNV, res) * computeScaleAndFresnel(cosThetaNV);
        }

        // Glossy - use half vector dot products for iridescence computation
        float avgCos;
        if (!computeAverageOfHalfVectorCosines(slice.getWo(), wi, avgCos)) {
            return scene_rdl2::math::sBlack;
        }
        return tintWithIridescence(avgCos, res) * computeScaleAndFresnel(avgCos);
    }

    finline scene_rdl2::math::Color albedo(const BsdfSlice& slice) const override
    {
        return mChildLobe->albedo(slice);
    }

    finline void differentials(const scene_rdl2::math::Vec3f&
                               wo, const scene_rdl2::math::Vec3f& wi,
                               float r1, float r2,
                               const scene_rdl2::math::Vec3f& dNdx,
                               const scene_rdl2::math::Vec3f& dNdy,
                               scene_rdl2::math::Vec3f& dDdx,
                               scene_rdl2::math::Vec3f& dDdy) const override
    {
        mChildLobe->differentials(wo, wi, r1, r2, dNdx, dNdy, dDdx, dDdy);
    }

    bool getProperty(Property property, float *dest) const override
    {
        return mChildLobe->getProperty(property, dest);
    }

    void show(std::ostream& os, const std::string& indent) const override
    {
        const scene_rdl2::math::Color& scale = getScale();
        const Fresnel * const fresnel = getFresnel();
        os << indent << "[BsdfIridescence]\n";
        os << indent << "    " << "scale: "
            << scale.r << " " << scale.g << " " << scale.b << "\n";
        if (fresnel) {
            fresnel->show(os, indent + "    ");
        }
        mChildLobe->show(os, indent + "    ");
    }

private:
    scene_rdl2::math::Color tintWithIridescence(
        const float cosThetaHV, const scene_rdl2::math::Color& lobeColor) const
    {
        // Early exit
        if (isBlack(lobeColor)) {
            return scene_rdl2::math::sBlack;
        }

        scene_rdl2::math::Color iridescence;
        if (mColorControl == ispc::SHADING_IRIDESCENCE_COLOR_USE_HUE_INTERPOLATION) {
            // Creates a view-dependent color by indexing into the hue spectrum between
            // the given primary and secondary color.
            iridescence = computeIridescenceColor(
                cosThetaHV, mPrimary, mSecondary, mFlipHue, mThickness, mExponent);
        } else {    // SHADING_IRIDESCENCE_COLOR_USE_RAMP
            // Creates a view-dependent color by indexing into the user provided ramp.            
            float index = scene_rdl2::math::pow(scene_rdl2::math::abs(1.f - cosThetaHV), mExponent) * mThickness;
            index = scene_rdl2::math::fmod(index, 1.f);
            iridescence = mRampControl.eval1D(index);
        }

        const float att = scene_rdl2::math::saturate(
            scene_rdl2::math::lerp(mIridescenceAt90, mIridescenceAt0, cosThetaHV));
        iridescence = scene_rdl2::math::lerp(
            scene_rdl2::math::sWhite, iridescence, att * mIridescence);

        // Multiply lobe result with iridescence approximation.
        return lobeColor * iridescence;
    }

private:
    BsdfLobe* mChildLobe;
    scene_rdl2::math::Vec3f mN;
    float mIridescence;
    ispc::SHADING_IridescenceColorMode mColorControl;
    scene_rdl2::math::Color mPrimary;
    scene_rdl2::math::Color mSecondary;
    bool  mFlipHue;
    ColorRampControl mRampControl;
    float mThickness;
    float mExponent;
    float mIridescenceAt0;
    float mIridescenceAt90;
};

//----------------------------------------------------------------------------

} // namespace shading
} // namespace moonray

