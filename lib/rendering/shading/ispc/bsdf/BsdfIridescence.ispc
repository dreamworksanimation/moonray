// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

/// @file BsdfIridescence.ispc

#include "BsdfIridescence.isph"
#include "BsdfSlice.isph"
#include <moonray/rendering/shading/ispc/Util.isph>

#include <scene_rdl2/common/math/ispc/ColorSpace.isph>
#include <scene_rdl2/common/platform/IspcUtil.isph>

//----------------------------------------------------------------------------
// These two functions are moved here from bsdf/Iridescence.isph
inline varying float
fitInRange(varying float value, const varying float oldMin, varying float oldMax, varying float newMin, varying float newMax)
{
    return ((newMax - newMin) * (value - oldMin) / (oldMax - oldMin)) + newMin;
}

inline varying Color
computeIridescenceColor(const varying float nDotV, const varying Color& primaryColor, const varying Color& secondaryColor,
                        const varying bool flipHue, varying float thickness, varying float exponent)
{
    // 1 - nDotV, to reverse the color spectrum between primaryColor and secondaryColor
    // so that secondaryColor appears at grazing angles and primaryColor appears straight on.
    varying float index = pow(abs(1.f - nDotV), exponent) * thickness;
    index = fmod(index, 1.f);

    varying Color iridescence;
    if (isEqual(primaryColor, secondaryColor)) {
        // If colors are equal, interpolate around the hue spectrum back to the same color.
        if (flipHue) {
            // Nothing to interpolate in the counter-clockwise direction.
            iridescence = primaryColor;
        } else {
            // Interpolate clockwise around the whole hue wheel back to the same color.
            const varying float huePrimary = rgbToHue(primaryColor);
            // Remap range to [hue value, hue value + 1] so that the primary color appears
            // at near normal and glancing angles and the rest of the hues are interpolated between.
            const varying float clampedIndex = fitInRange(index, 0.f, 1.f, huePrimary, huePrimary + 1.f);
            iridescence = hueToRgb(clampedIndex);
        }
    } else {
        // If colors are not equal, interpolate a range around the hue wheel and lerp from
        // the secondaryColor back to the primaryColor.
        varying float huePrimary = rgbToHue(primaryColor);
        varying float hueSecondary = rgbToHue(secondaryColor);
        // If the hue is not flipped but the secondary color is greater than the primary,
        // we need to wrap around the secondary color to maintain clockwise interpolation around the wheel.
        // If the hue is flipped but primary is less than secondary, we need to wrap around
        // the primary color so that it interpolates in reverse, counter-clockwise.
        if (hueSecondary < huePrimary && !flipHue) {
            hueSecondary = hueSecondary + 1.f;
        } else if (huePrimary < hueSecondary && flipHue) {
            huePrimary = huePrimary + 1.f;
        }

        // Lerp back to the primary color from the secondary color using part of the
        // nDotV index. The hue spectrum is normally divided into 6 sections or 1/6ths (0.166667),
        // but since the hue range is limited, these sections become proportionally larger.
        // The lerpBackDistance is computed by inverting this value and adding 1 to add another "section,"
        // and then inverting it again to make the lerp distance 1/x, for x = number of sections.
        const varying float lerpBackDistance = 1.f / (abs(hueSecondary - huePrimary) / 0.166667 + 1.f);
        const varying float spectrumDistance = 1.f - lerpBackDistance;
        const varying float clampedIndex = fitInRange(index, 0.f, spectrumDistance, huePrimary, hueSecondary);
        if (index < spectrumDistance) {
            iridescence = hueToRgb(clampedIndex);
        } else {
            iridescence = lerp(hueToRgb(hueSecondary), hueToRgb(huePrimary), (index - spectrumDistance)/lerpBackDistance );
        }
    }

    return iridescence;
}
//----------------------------------------------------------------------------

varying Color
tintWithIridescence(const varying IridescenceBsdfLobe * uniform iLobe,
                    const float cosThetaHV,
                    const Color& lobeColor)
{
    // Early exit
    if (isBlack(lobeColor)) {
        return sBlack;
    }

    varying Color iridescence;
    cif (iLobe->mColorControl == SHADING_IRIDESCENCE_COLOR_USE_HUE_INTERPOLATION) {
        // Creates a view-dependent color by indexing into the hue spectrum between
        // the given primary and secondary color.
        iridescence = computeIridescenceColor(cosThetaHV, iLobe->mPrimary, iLobe->mSecondary,
                                              iLobe->mFlipHue, iLobe->mThickness, iLobe->mExponent);
    } else {    // SHADING_IRIDESCENCE_COLOR_USE_RAMP
        // Creates a view-dependent color by indexing into the user provided ramp.
        varying float index = pow(abs(1.f - cosThetaHV), iLobe->mExponent) * iLobe->mThickness;
        index = fmod(index, 1.f);
        iridescence = ColorRampControl_eval1D(index, &iLobe->mRampControl);
    }

    const float att = saturate(lerp(iLobe->mIridescenceAt90, iLobe->mIridescenceAt0, cosThetaHV));
    iridescence = lerp(sWhite, iridescence, att * iLobe->mIridescence);

    // Multiply Cook-Torrance result with iridescence approximation.
    return lobeColor * iridescence;
}

inline varying bool
computeAverageOfHalfVectorCosines(const varying Vec3f& wo, const varying Vec3f& wi,
                                  varying float& avgCos)
{
    Vec3f m;
    if (!computeNormalizedHalfVector(wo, wi, m)) {
        return false;
    }
    // Iridescence
    const varying float cosMO = dot(m, wo);
    const varying float cosMI = dot(m, wi);

    // There is a slight difference between cosMO (hDotV) and cosMI (hDotL) which causes the unit test
    // to fail. Averaging the two circumvents this issue and results in identical results
    // as using just cosMO.
    avgCos = (cosMO + cosMI) * 0.5f;
    return true;
}

varying Color
IridescenceBsdfLobe_eval(const varying BsdfLobe * uniform lobe,
                         const varying BsdfSlice &slice,
                         const Vec3f &wi,
                         varying float * uniform pdf)
{
    const varying IridescenceBsdfLobe * uniform iLobe =
            (const varying IridescenceBsdfLobe * uniform) lobe;

    // Mirror lobe
    if (BsdfLobe_getType(iLobe->mChildLobe) & BSDF_LOBE_TYPE_MIRROR) {
        if (pdf) *pdf = 0.0f;
        return sBlack;
    }

    // Glossy lobe
    // Evaluate the underlying bsdf
    const varying Color lobeResult = BsdfLobe_eval(iLobe->mChildLobe, slice, wi, pdf);

    // use half vector dot products for iridescence computation
    varying float avgCos;
    if (!computeAverageOfHalfVectorCosines(slice.mWo, wi, avgCos)) {
        return sBlack;
    }
    return tintWithIridescence(iLobe, avgCos, lobeResult) * BsdfLobe_computeScaleAndFresnel(lobe, avgCos);
}

varying Color
IridescenceBsdfLobe_sample(const varying BsdfLobe * uniform lobe,
                           const varying BsdfSlice &slice,
                           varying float r1, varying float r2,
                           varying Vec3f &wi, varying float &pdf)
{
    const varying IridescenceBsdfLobe * uniform iLobe =
            (const varying IridescenceBsdfLobe * uniform) lobe;

    const varying Color res = BsdfLobe_sample(iLobe->mChildLobe, slice, r1, r2, wi, pdf);

    // Mirror
    // *Optimization* : using NdotV for iridescence color computation
    if (BsdfLobe_getType(iLobe->mChildLobe) & BSDF_LOBE_TYPE_MIRROR) {
        const float cosThetaNV = dot(iLobe->mN, slice.mWo);
        return tintWithIridescence(iLobe, cosThetaNV, res) * BsdfLobe_computeScaleAndFresnel(lobe, cosThetaNV);;
    }

    // Glossy - use half vector dot products for iridescence computation
    varying float avgCos;
    if (!computeAverageOfHalfVectorCosines(slice.mWo, wi, avgCos)) {
        return sBlack;
    }
    return tintWithIridescence(iLobe, avgCos, res) * BsdfLobe_computeScaleAndFresnel(lobe, avgCos);;
}

varying Color
IridescenceBsdfLobe_albedo(const varying BsdfLobe * uniform lobe,
                           const varying BsdfSlice &slice)
{
    const varying IridescenceBsdfLobe * uniform iLobe =
            (const varying IridescenceBsdfLobe * uniform) lobe;

    return BsdfLobe_albedo(iLobe->mChildLobe, slice);
}

void
IridescenceBsdfLobe_differentials(const varying BsdfLobe * uniform lobe,
                                  const varying Vec3f &wo,
                                  const varying Vec3f &wi,
                                  varying float r1, varying float r2,
                                  const varying Vec3f &dNdx,
                                  const varying Vec3f &dNdy,
                                  varying Vec3f &dDdx, varying Vec3f &dDdy)
{
    const varying IridescenceBsdfLobe * uniform iLobe =
            (const varying IridescenceBsdfLobe * uniform) lobe;

    BsdfLobe_differentials(iLobe->mChildLobe, wo, wi, r1, r2, dNdx, dNdy, dDdx, dDdy);
}

uniform bool
IridescenceBsdfLobe_getProperty(const varying BsdfLobe * uniform lobe,
                                uniform BsdfLobeProperty property,
                                varying float * uniform dest)
{
    const varying IridescenceBsdfLobe * uniform iLobe =
            (const varying IridescenceBsdfLobe * uniform) lobe;

    return BsdfLobe_getProperty(iLobe->mChildLobe, property, dest);
}

