// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

///
/// @file RampControl.h
///

#pragma once

#include <moonray/rendering/shading/ispc/RampControl_ispc_stubs.h>
#include <scene_rdl2/common/math/Color.h>
#include <scene_rdl2/common/math/Vec2.h>

namespace moonray {
namespace shading {

/// @class FloatRampControl
/// @brief Helper class that maps a set of float inputs to float outputs and
/// performs interpolation between 2 control points based on interpolationTypes
class FloatRampControl {
public:
    FloatRampControl() : mIspc() {}
    ~FloatRampControl() {}

    void init(const int numEntries,
              const float* inputs,
              const float* outputs,
              const ispc::RampInterpolatorMode* interpolators);

    /// evaluates the ramp to return an output float value based on a 1D input 't'
    /// @param t input position to be evaluated on the ramp
    /// @return float result based on ramp control and input 't'
    float eval1D(float t) const;

    /// evaluates the ramp to return an output float value based on a 2D input 'uv'
    /// @param uv input position to be evaluated on the ramp
    /// @param rampType2D type of 2d interpolation
    /// @param inputRamp custom input ramp if rampType2D is RAMP_INTERPOLATOR_2D_TYPE_INPUT
    /// @return float result based on ramp control and input 'uv'
    float eval2D(const scene_rdl2::math::Vec2f& uv,
                 ispc::RampInterpolator2DType rampType2D,
                 float inputRamp = 0.0f) const;

    /// Gets ispc object for vector mode
    HUD_AS_ISPC_METHODS(FloatRampControl);
private:
    ispc::FloatRampControl mIspc;
};

/// @class ColorRampControl
/// @brief Helper class that maps a set of float inputs to Color outputs and
/// performs interpolation between 2 control points based on interpolationTypes
class ColorRampControl {
public:
    ColorRampControl() : mIspc() {}
    ~ColorRampControl() {}

    void init(const int numEntries,
              const float* inputs,
              const scene_rdl2::math::Color* outputs,
              const ispc::RampInterpolatorMode* interpolatorTypes,
              ispc::ColorRampControlSpace colorSpace,
              const bool applyHueBlendAdjustment = true);

    /// evaluates the ramp to return an output Color value based on a 1D input 't'
    /// @param t input position to be evaluated on the ramp
    /// @return Color result based on ramp control and input 't'
    scene_rdl2::math::Color eval1D(float t) const;

    /// evaluates the ramp to return an output Color value based on a 2D input 'uv'
    /// @param t input position to be evaluated on the ramp
    /// @param rampType2D type of 2d interpolation
    /// @param inputRamp custom input ramp if rampType2D is RAMP_INTERPOLATOR_2D_TYPE_INPUT
    /// @return Color result based on ramp control and input 'uv'
    scene_rdl2::math::Color eval2D(const scene_rdl2::math::Vec2f& uv,
                                   ispc::RampInterpolator2DType rampType2D,
                                   float inputRamp = 0.0f) const;

    /// Gets ispc object for vector mode
    HUD_AS_ISPC_METHODS(ColorRampControl);

private:
    void blendAdjustment(scene_rdl2::math::Color& left,
                         scene_rdl2::math::Color& right) const;

    ispc::ColorRampControl mIspc;
};

} // end namespace shading
} // end namespace moonray
