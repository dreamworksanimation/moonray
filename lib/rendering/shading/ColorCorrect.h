// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

//
//

#pragma once

#include <scene_rdl2/scene/rdl2/rdl2.h>
#include <scene_rdl2/common/math/ColorSpace.h>

namespace moonray {
namespace shading {

inline scene_rdl2::math::Color
clampColor(const scene_rdl2::math::Color&c)
{
    scene_rdl2::math::Color r;
    for (int i = 0; i < 3; ++i) {
        r[i] = scene_rdl2::math::clamp(c[i]);
    }
    return r;
}

inline float
lerpOpt(const float v0, const float v1, const float t)
{
    // lerpOpt = lerp optimized.
    // Optimized version of
    // return (1.0 - t) * v0 + t * v1;
    return (v1 - v0) * t + v0;
}

inline scene_rdl2::math::Color
lerpOpt(const scene_rdl2::math::Color& v0, const scene_rdl2::math::Color& v1, const float t)
{
    // lerpOpt = lerp optimized.
    // Optimized version of
    // return (1.0 - t) * v0 + t * v1;
    return (v1 - v0) * t + v0;
}

inline float
computeLuminance(const scene_rdl2::math::Color& rgb)
{
    return rgb.r * 0.212671f + rgb.g * 0.715160f + rgb.b * 0.072169f;
}

inline void
applyHueShift(const float hueShift, scene_rdl2::math::Color& result)
{
    if (scene_rdl2::math::isZero(hueShift)) {
        return;
    }

    scene_rdl2::math::Color hsv = scene_rdl2::math::rgbToHsv(result);
    hsv.r = scene_rdl2::math::fmod(hsv.r + hueShift, 1.f);
    result = scene_rdl2::math::hsvToRgb(hsv);
}

inline void
applySaturation(const float saturation, scene_rdl2::math::Color& result)
{
    float y = computeLuminance(result);
    result.r = lerpOpt(y, result.r, saturation);
    result.g = lerpOpt(y, result.g, saturation);
    result.b = lerpOpt(y, result.b, saturation);
}

inline void
applySaturation(const scene_rdl2::math::Color& saturation, scene_rdl2::math::Color& result)
{
    float y = computeLuminance(result);
    result.r = lerpOpt(y, result.r, saturation.r);
    result.g = lerpOpt(y, result.g, saturation.g);
    result.b = lerpOpt(y, result.b, saturation.b);
}

inline void
applySaturationWithoutPreservingLuminance(const float saturation, scene_rdl2::math::Color& result)
{
    const float m = scene_rdl2::math::max(result.r, scene_rdl2::math::max(result.g, result.b));
    result.r = lerpOpt(m, result.r, saturation);
    result.g = lerpOpt(m, result.g, saturation);
    result.b = lerpOpt(m, result.b, saturation);
}

inline void
applyContrast(float contrast, float& result)
{
    if (scene_rdl2::math::isZero(contrast)) {
        return;
    }

    contrast = scene_rdl2::math::clamp(contrast, -1.0f, 1.0f);

    if (contrast < 0) {
        contrast *= -1.0f;
        result = scene_rdl2::math::lerp(result, 0.5f, contrast);
    } else {
        if (scene_rdl2::math::isEqual(contrast, 1.0f)) contrast = 0.999f;
        contrast = 1.0f / (1.0f - contrast);
        result = contrast * (result - 0.5f) + 0.5f;
    }
}

inline void
applyContrast(scene_rdl2::math::Color contrast, scene_rdl2::math::Color& result)
{
    applyContrast(contrast.r, result.r);
    applyContrast(contrast.g, result.g);
    applyContrast(contrast.b, result.b);
}

inline void
applyNukeContrast(const scene_rdl2::math::Color& contrast, scene_rdl2::math::Color& result)
{
    // Mimic Nuke's ColorCorrect node's contrast function.
    // Pivot around %18.
    const float pivot = 0.18f;
    const float invPivot = 1.0f / pivot;
    if (result.r > 0) {
        result.r = scene_rdl2::math::pow(result.r * invPivot, contrast.r) * pivot;
    } else {
        result.r = result.r * scene_rdl2::math::pow(invPivot, contrast.r) * pivot;
    }
    if (result.g > 0) {
        result.g = scene_rdl2::math::pow(result.g * invPivot, contrast.g) * pivot;
    } else {
        result.g = result.g * scene_rdl2::math::pow(invPivot, contrast.g) * pivot;
    }
    if (result.b > 0) {
        result.b = scene_rdl2::math::pow(result.b * invPivot, contrast.b) * pivot;
    } else {
        result.b = result.b * scene_rdl2::math::pow(invPivot, contrast.b) * pivot;
    }
}

inline void
applyGamma(const scene_rdl2::math::Color& gamma, scene_rdl2::math::Color& result)
{
    if (result.r > 0) result.r = scene_rdl2::math::pow(result.r, gamma.r);
    if (result.g > 0) result.g = scene_rdl2::math::pow(result.g, gamma.g);
    if (result.b > 0) result.b = scene_rdl2::math::pow(result.b, gamma.b);
}

inline void
applyGainAndOffset(const scene_rdl2::math::Color& gain,
                   const scene_rdl2::math::Color& offset, scene_rdl2::math::Color& result)
{
    result.r = result.r * gain.r + offset.r;
    result.g = result.g * gain.g + offset.g;
    result.b = result.b * gain.b + offset.b;
}

inline void
applyTMI(const scene_rdl2::math::Color& temperature, scene_rdl2::math::Color& result)
{
    // TMI(E)
    // Color Temperature (T)
    // Magenta/Green (M):
    // Energy (E) / Intensity (I):

    const float oneOverSix = 1.0 / 6.0;
    const float oneOverThree = 1.0 / 3.0;

    const float kT = temperature.r; // yellow/blue vector (temperature), default range [-2,+2]
    const float kM = temperature.g; // green/magenta vector, default range [-2,+2]
    const float kI = temperature.b; // intensity is more useful in a lighting context
                                    // than simple intensity, default range 0..4
                                    // the range for k_E should be -8..+8.

    const float expScale = scene_rdl2::math::pow(2.0f, kI);
    const float rScale = (6.0f * expScale + 2.0f * kM - 3.0f * kT) * oneOverSix;
    const float gScale = (3.0f * expScale - 2.0f * kM) * oneOverThree;
    const float bScale = rScale + kT;

    result.r = result.r * rScale;
    result.g = result.g * gScale;
    result.b = result.b * bScale;
}

} // namespace shading
} // namespace moonray


