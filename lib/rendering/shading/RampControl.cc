// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

///
/// @file RampControlImpl.cc
///

#include "RampControl.h"

#include <scene_rdl2/common/math/Color.h>
#include <scene_rdl2/common/math/ColorSpace.h>
#include <scene_rdl2/common/math/Mat4.h>
#include <scene_rdl2/common/math/Math.h>
#include <scene_rdl2/common/math/Vec4.h>
#include <scene_rdl2/render/util/stdmemory.h>

using namespace scene_rdl2;
using namespace scene_rdl2::math;

namespace {

// Simple line slope calculation: (y2 - y1) / (x2 - x1)
inline float
computeLineSlope(float i0, float o0, float i1, float o1)
{
    return i0 == i1 ? 0.0f : (o1 - o0) / (i1 - i0);
}

inline Color
computeLineSlope(float i0, const Color& o0, float i1, const Color& o1)
{
    return i0 == i1 ? Color(0.0f) : (o1 - o0) / (i1 - i0);
}

// Compute the curve slope at one control vertex required to enforce monotonicity
// on a cubic hermite spline
// Based on Fritsch-Carlson method
// Inputs are four control vertices, with slopes being calculated at cv1 and cv2
// @param "end" is used to indicate boundary cases, where
// end = 1 if cv1 is the first cv, end = 2 if cv2 is the last cv, end = 0 no boundary case
void
computeFKMonotonoSlope(float i0, float o0,
                       float i1, float o1,
                       float i2, float o2,
                       float i3, float o3,
                       int end,
                       float& slope1, float& slope2)
{
    // Compute secant line slopes
    // if end==3, then there are only 2 control points
    // in that case we set all slope information to 0
    // which ends up being a smoothstep operation
    const float delta2 = (end != 2 && end != 3) ? computeLineSlope(i2, o2, i3, o3) : 0.0f;
    const float delta1 = end != 3 ? computeLineSlope(i1, o1, i2, o2) : 0.0f;
    const float delta0 = (end != 1 && end != 3) ? computeLineSlope(i0, o0, i1, o1) : 0.0f;

    if (isZero(delta1)) {
        slope1 = 0.0f;
        slope2 = 0.0f;
        return;
    }

    // Initialize curve slope to be convex combination of slopes of adjacent data
    // We use the average of the secant slopes
    if (end == 1) { // cv0 is not provided (beginning of the spline)
        slope1 = delta1;
        slope2 = (sign(delta1) == sign(delta2)) ? (delta2 + delta1) / 2.0f : 0.0f;
    } else if (end == 2) { // cv3 is not provided (end of the spline)
        slope1 = (sign(delta0) == sign(delta1)) ? (delta0 + delta1) / 2.0f : 0.0f;
        slope2 = delta1;
    } else { // all four cvs are provided (non-boundary cases on the spline)
        slope1 = (sign(delta0) == sign(delta1)) ? (delta0 + delta1) / 2.0f : 0.0f;
        slope2 = (sign(delta1) == sign(delta2)) ? (delta2 + delta1) / 2.0f : 0.0f;
    }

    const float alpha = slope1 / delta1;
    const float beta = slope2 / delta1;

    if (alpha < 0.0f || beta < 0.0f) {
        // Data not monotone, return calculated slopes as initialized
        return;
    } else {
        if (alpha*alpha + beta*beta > 9.0f) {
            const float tau = 3.0f / sqrt(alpha*alpha + beta*beta);
            slope1 = tau * slope1;
            slope2 = tau * slope2;
        }
    }
}

void
computeFKMonotonoSlope(float i0, const Color& o0,
                       float i1, const Color& o1,
                       float i2, const Color& o2,
                       float i3, const Color& o3,
                       int end,
                       Color& slope1, Color& slope2)
{
    computeFKMonotonoSlope(i0, o0.r,
                           i1, o1.r,
                           i2, o2.r,
                           i3, o3.r,
                           end, slope1.r, slope2.r);
    computeFKMonotonoSlope(i0, o0.g,
                           i1, o1.g,
                           i2, o2.g,
                           i3, o3.g,
                           end, slope1.g, slope2.g);
    computeFKMonotonoSlope(i0, o0.b,
                           i1, o1.b,
                           i2, o2.b,
                           i3, o3.b,
                           end, slope1.b, slope2.b);
}

// Given four control vertices in 1D and a curve parameterization
// value t in [0, 1], returns a value interpolated by Centripital
// Catmull Rom spline.
finline float
interpolateCatmullRom(const float t, const float p[4])
{
    Vec4f controlPoints(p[0], p[1], p[2], p[3]);
    static const Mat4f catmullMatrix( 0.0f,  2.0f,  0.0f,  0.0f,
                                     -1.0f,  0.0f,  1.0f,  0.0f,
                                      2.0f, -5.0f,  4.0f, -1.0f,
                                     -1.0f,  3.0f, -3.0f,  1.0f);

    const Vec4f tVec(1.0f, t, t * t, t * t * t);
    return dot(0.5f * tVec * catmullMatrix, controlPoints);
}

finline Color
interpolateCatmullRom(const float t, const Color p[4])
{
    const float r[4] = {p[0].r, p[1].r, p[2].r, p[3].r};
    const float g[4] = {p[0].g, p[1].g, p[2].g, p[3].g};
    const float b[4] = {p[0].b, p[1].b, p[2].b, p[3].b};

    return Color(interpolateCatmullRom(t, r),
                 interpolateCatmullRom(t, g),
                 interpolateCatmullRom(t, b));
}

// Returns a value interpolated by cubic hermite spline, given
// two CVs and end slopes
finline float
interpolateCubicHermite(const float t,
                        const float x1, const float x2,
                        const float y1, const float y2,
                        const float slope1, const float slope2)
{
    const float H = x2 - x1;
    const float t2 = t * t;
    const float t3 = t2 * t;

    // Compute cubic hermite basis functions
    const float h00 = 2.0f * t3 - 3.0f * t2 + 1.0f;
    const float h10 = t3 - 2.0f * t2 + t;
    const float h01 = (-2.0f * t3) + 3.0f * t2;
    const float h11 = t3 - t2;

    return h00 * y1 + H * h10 * slope1 + h01 * y2 + H * h11 * slope2;
}

// Returns a value interpolated by cubic hermite spline, given
// two CVs and end slopes
finline Color
interpolateCubicHermite(const float t,
                        const float x1, const float x2,
                        const Color& y1, const Color& y2,
                        const Color& slope1, const Color& slope2)
{
    Color result;

    result.r = interpolateCubicHermite(t, x1, x2, y1.r, y2.r, slope1.r, slope2.r);
    result.g = interpolateCubicHermite(t, x1, x2, y1.g, y2.g, slope1.g, slope2.g);
    result.b = interpolateCubicHermite(t, x1, x2, y1.b, y2.b, slope1.b, slope2.b);

    return result;
}

void
evaluateMonotoneSlopes(const int numEntries,
                       const float* const inputs,
                       const float* const outputs,
                       const ispc::RampInterpolatorMode* const interpolators,
                       float* const slopes)
{
    // If any of the ramp inputs has monotone cubic interpolation, find and cache
    // curve slopes at each point
    bool usesCubicInterpolation = false;
    for (int i = 0; i < numEntries; ++i) {
        if(interpolators[i] == ispc::RAMP_INTERPOLATOR_MODE_MONOTONECUBIC) {
            usesCubicInterpolation = true;
            break;
        }
    }

    if (!usesCubicInterpolation) {
        return;
    }

    // Compute curve slopes at each control vertex to enforce monotonicity
    int idx;
    for (idx = 0; idx < numEntries - 1; idx += 2) {
        int k0 = idx - 1, k1 = idx, k2 = idx + 1, k3 = idx + 2;
        int end = 0;

        if (idx == 0) {
            k0 = k1; // id0 will not be used
            end = 1;
        }
        if (idx+1 == numEntries - 1) {
            k3 = k2; // id3 will not be used
            end = (end == 1) ? 3 : 2; //3 is special case if there are only 2 points
        }
        computeFKMonotonoSlope(inputs[k0], outputs[k0],
                               inputs[k1], outputs[k1],
                               inputs[k2], outputs[k2],
                               inputs[k3], outputs[k3],
                               end, slopes[k1], slopes[k2]);
    }
    if (idx == numEntries - 1) {
        // Last CV hasn't been assigned a slope yet, if odd number of ramp points
        const int lastIdx = numEntries - 1;
        slopes[lastIdx] = computeLineSlope(inputs[lastIdx - 1],
                                           outputs[lastIdx - 1],
                                           inputs[lastIdx],
                                           outputs[lastIdx]);
    }
}

void
evaluateMonotoneSlopes(const int numEntries,
                       const float* const inputs,
                       const Color* const outputs,
                       const ispc::RampInterpolatorMode* const interpolators,
                       Color* const slopes)
{
    // If any of the ramp inputs has monotone cubic interpolation, find and cache
    // curve slopes at each point
    bool usesCubicInterpolation = false;
    for(int i=0; i<numEntries; i++) {
        if(interpolators[i] == ispc::RAMP_INTERPOLATOR_MODE_MONOTONECUBIC) {
            usesCubicInterpolation = true;
            break;
        }
    }

    if (!usesCubicInterpolation) {
        return;
    }

    // Compute curve slopes at each control vertex to enforce monotonicity
    int idx;
    for (idx = 0; idx < numEntries - 1; idx+=2) {
        int k0 = idx - 1, k1 = idx, k2 = idx + 1, k3 = idx + 2;
        int end = 0;

        if (idx == 0) {
            k0 = k1; // id0 will not be used
            end = 1;
        }
        if (idx+1 == numEntries - 1) {
            k3 = k2; // id3 will not be used
            end = (end == 1)? 3:2; //3 is special case if there are only 2 points
        }
        computeFKMonotonoSlope(inputs[k0], outputs[k0],
                               inputs[k1], outputs[k1],
                               inputs[k2], outputs[k2],
                               inputs[k3], outputs[k3],
                               end, slopes[k1], slopes[k2]);
    }
    if (idx == numEntries - 1) {
        // Last CV hasn't been assigned a slope yet, if odd number of ramp points
        const int lastIdx = numEntries - 1;
        slopes[lastIdx] = computeLineSlope(inputs[lastIdx - 1], outputs[lastIdx - 1],
                                           inputs[lastIdx], outputs[lastIdx]);
    }
}

// Struct to hold one ramp data point
template<class OutputType>
struct RampCv
{
    float input;
    OutputType output;
    ispc::RampInterpolatorMode interp;
};

template<class OutputType>
void
sortEntries(const uint32_t numEntries,
            float* const inputs,
            OutputType* const outputs,
            ispc::RampInterpolatorMode* const interpolators)
{
    // check to see if the points are already sorted
    bool isSorted = true;
    for (uint32_t i = 0; i < numEntries - 1; ++i) {
        if (inputs[i] > inputs[i + 1]) {
            isSorted = false;
            break;
        }
    }

    // skip sorting if its already sorted.
    if (isSorted) { return; }

    // pack the points into array of structs for sorting
    RampCv<OutputType> rampCvs[ispc::RAMP_MAX_POINTS];

    // populate array of structs
    for (uint32_t idx = 0; idx < numEntries; ++idx) {
        rampCvs[idx].input = inputs[idx];
        rampCvs[idx].output = outputs[idx];
        rampCvs[idx].interp = interpolators[idx];
    }

    // perform sorting
    std::sort(rampCvs, rampCvs + numEntries,
              [](const RampCv<OutputType>& a, const RampCv<OutputType>& b) {
                  return a.input < b.input;
              });

    // unpack structs back into arrays
    for (uint32_t idx = 0; idx < numEntries; ++idx) {
        inputs[idx] = rampCvs[idx].input;
        outputs[idx] = rampCvs[idx].output;
        interpolators[idx] = rampCvs[idx].interp;
    }
}

void
findRampSpan(const int numEntries,
             const float* inputs,
             const float t,
             int& leftIdx,
             int& rightIdx) {
    // Find keyframe to use for interpolation based on t
    // keyframe is interpolated based on the type provided in the point closest to the left.
    leftIdx = 0; // Constant interpolation by default
    rightIdx = numEntries - 1;

    // Assign -1 to the indices if pos is outside the range of the defined ramp
    // This can happen when the ramp isn't defined over the entirity of [0, 1] range
    if (t < inputs[leftIdx]) {
        leftIdx = -1;
        return;
    }
    if (t > inputs[rightIdx]) {
        rightIdx = -1;
        return;
    }

    // Bisect to find the right interpolation span (positions and colors are sorted)
    while (rightIdx - leftIdx > 1) {
        int mid = (leftIdx + rightIdx) / 2;
        if (inputs[mid] > t) {
            rightIdx = mid;
        } else {
            leftIdx = mid;
        }
    }
}

template<class OutputType>
OutputType
computeCubicSplineValue(const int numEntries,
                        const float* inputs,
                        const OutputType* outputs,
                        const OutputType* slopes,
                        const int interval,
                        const float dist,
                        const int interpolator) {

    // Control points of cubic spline
    OutputType p[4];
    int intervals[4];
    float t = dist;

    intervals[1] = interval;
    // set p0
    if (interval == 0) {
        intervals[0] = intervals[1];
    } else {
        intervals[0] = interval - 1;
    }

    // set p2 and p3
    if (interval == numEntries - 2) {
        intervals[2] = interval + 1;
        intervals[3] = intervals[2];
    } else if (interval == numEntries - 1) {
        intervals[2] = intervals[1];
        intervals[3] = intervals[2];
        t = 0.f;
    } else {
        intervals[2] = interval + 1;
        intervals[3] = interval + 2;
    }

    for (int i = 0; i < 4; ++i)
        p[i] = outputs[intervals[i]];

    OutputType result = OutputType(math::zero);

    if (interpolator == ispc::RAMP_INTERPOLATOR_MODE_CATMULLROM) {
        result = interpolateCatmullRom(t, p);
    } else if (interpolator == ispc::RAMP_INTERPOLATOR_MODE_MONOTONECUBIC) {
        result = interpolateCubicHermite(t,
                                         inputs[intervals[1]], inputs[intervals[2]],
                                         p[1], p[2],
                                         slopes[intervals[1]], slopes[intervals[2]]);
    }
    return result;
}


template<class OutputType>
OutputType
evaluateFourOutputRamp(const Vec2f& uv,
                       const OutputType& input0, const OutputType& input1,
                       const OutputType& input2, const OutputType& input3)
{
    // Corner 1
    float weight = (1.0f - uv.x) * (1.0f - uv.y);
    OutputType result = weight * input0;

    // Corner 2
    weight = uv.x * (1.0f - uv.y);
    result += weight * input1;

    // Corner 3
    weight = (1.0f - uv.x) * uv.y;
    result += weight * input2;

    // Corner 4
    weight = uv.x * uv.y;
    result += weight * input3;

    return result;
}

template<class OutputType, typename BlendAdjustmentCallbackType>
OutputType
eval1DRamp(const int numEntries,
           const float* inputs,
           const OutputType* outputs,
           const ispc::RampInterpolatorMode* interpolators,
           const OutputType* slopes,
           float t,
           const BlendAdjustmentCallbackType& blendAdjustmentCallback)
{
    // Find which span the current position belongs to
    // Interpolation at leftIdx determines the interpolation to use for the span
    int leftIdx, rightIdx;
    findRampSpan(numEntries, inputs, t, leftIdx, rightIdx);

    // Check if pos is outside the range of the defined ramp
    if (leftIdx == -1) {
        return outputs[0];
    }
    if (rightIdx == -1) {
        return outputs[numEntries - 1];
    }

    // No interpolation if distance is too small
    const float spanLength = inputs[rightIdx] - inputs[leftIdx];
    if (math::isZero(spanLength)) {
        return outputs[leftIdx];
    }

    OutputType result(scene_rdl2::math::zero);
    float weight = 0.0f;

    // t is converted to [0 1] range
    t = (t - inputs[leftIdx]) / spanLength;

    int interpolator = interpolators[leftIdx];

    switch (interpolator) {
    case ispc::RAMP_INTERPOLATOR_MODE_NONE:
        weight = 0.0f;
        break;
    case ispc::RAMP_INTERPOLATOR_MODE_LINEAR:
        weight = t;
        break;
    case ispc::RAMP_INTERPOLATOR_MODE_EXPONENTIAL_UP:
        weight = t * t;
        break;
    case ispc::RAMP_INTERPOLATOR_MODE_EXPONENTIAL_DOWN:
        weight = 1.0f - t;
        weight = 1.0f - (weight * weight);
        break;
    case ispc::RAMP_INTERPOLATOR_MODE_SMOOTH:
        weight = math::sin(t *  math::sHalfPi);
        break;
    case ispc::RAMP_INTERPOLATOR_MODE_CATMULLROM:
    case ispc::RAMP_INTERPOLATOR_MODE_MONOTONECUBIC:
        result = computeCubicSplineValue(numEntries, inputs, outputs, slopes, leftIdx, t, interpolator);
        return result;
    }

    OutputType leftOutput = outputs[leftIdx];
    OutputType rightOutput = outputs[rightIdx];
    blendAdjustmentCallback(leftOutput, rightOutput);
    result = leftOutput + weight * (rightOutput - leftOutput);
    return result;
}

template<class OutputType, typename BlendAdjustmentCallbackType>
OutputType
eval2DRamp(const int numEntries,
           const float* inputs,
           const OutputType* outputs,
           const ispc::RampInterpolatorMode* interpolators,
           const OutputType* slopes,
           Vec2f uv,
           ispc::RampInterpolator2DType rampType2D,
           float inputRamp,
           const BlendAdjustmentCallbackType& blendAdjustmentCallback)
{
    OutputType result(math::zero);

    switch (rampType2D) {
    case ispc::RAMP_INTERPOLATOR_2D_TYPE_V_RAMP:
        result = eval1DRamp(numEntries, inputs, outputs, interpolators, slopes, uv.y, blendAdjustmentCallback);
        break;
    case ispc::RAMP_INTERPOLATOR_2D_TYPE_U_RAMP:
        result = eval1DRamp(numEntries, inputs, outputs, interpolators, slopes, uv.x, blendAdjustmentCallback);
        break;
    case ispc::RAMP_INTERPOLATOR_2D_TYPE_DIAGONAL_RAMP:
        result = eval1DRamp(numEntries, inputs, outputs, interpolators, slopes, 0.5f * (uv.x + uv.y),
                          blendAdjustmentCallback);
        break;
    case ispc::RAMP_INTERPOLATOR_2D_TYPE_RADIAL_RAMP:
        {
            float value = math::atan2(uv.x - 0.5f, uv.y - 0.5f);
            value = 0.5f * (1.0f + value / math::sPi);
            result = eval1DRamp(numEntries, inputs, outputs, interpolators, slopes, value, blendAdjustmentCallback);
        }
        break;
    case ispc::RAMP_INTERPOLATOR_2D_TYPE_CIRCULAR_RAMP:
        {
            uv.x -= 0.5f;
            uv.y -= 0.5f;
            float value = uv.x * uv.x + uv.y * uv.y;
            if (value > math::sEpsilon) {
                value = math::sqrt(2.0f * value);
            } else {
                value = 0.0f;
            }
            result = eval1DRamp(numEntries, inputs, outputs, interpolators, slopes, value, blendAdjustmentCallback);
        }
        break;
    case ispc::RAMP_INTERPOLATOR_2D_TYPE_BOX_RAMP:
        {
            const float value = 2.0f * math::max(math::abs(uv.x - 0.5f), math::abs(uv.y - 0.5f));
            result = eval1DRamp(numEntries, inputs, outputs, interpolators, slopes, value, blendAdjustmentCallback);
        }
        break;
    case ispc::RAMP_INTERPOLATOR_2D_TYPE_UxV_RAMP:
        result = eval1DRamp(numEntries, inputs, outputs, interpolators, slopes,
                          2.0f * math::abs(uv.y - 0.5f), blendAdjustmentCallback);
        result *= eval1DRamp(numEntries, inputs, outputs, interpolators, slopes,
                           2.0f * math::abs(uv.x - 0.5f), blendAdjustmentCallback);
        break;
    case ispc::RAMP_INTERPOLATOR_2D_TYPE_FOUR_CORNER_RAMP: // Special case, not calling evaluateRampColor()
        {
            // we skip calling blendAdjustmentCallback for this case as the callback deals with only
            // 2 outputs instead of 4
            MNRY_ASSERT(numEntries == 4);
            result = evaluateFourOutputRamp(uv,
                                           outputs[0],
                                           outputs[1],
                                           outputs[2],
                                           outputs[3]);
        }
        break;
    case ispc::RAMP_INTERPOLATOR_2D_TYPE_INPUT:
        {
            result = eval1DRamp(numEntries,
                                inputs, outputs, interpolators, slopes,
                                inputRamp, blendAdjustmentCallback);
        }
        break;
    default:
        MNRY_ASSERT(0);
    }
    return result;
}

} // end anonymous namespace


namespace moonray {
namespace shading {

void
FloatRampControl::init(int numEntries,
                       const float* inputs,
                       const float* outputs,
                       const ispc::RampInterpolatorMode* interpolators)
{
    // ignore any extra points
    numEntries = math::min(numEntries, ispc::RAMP_MAX_POINTS);

    mIspc.mNumEntries = numEntries;

    for (int i = 0; i < numEntries; ++i) {
        mIspc.mInputs[i] = inputs[i];
        mIspc.mOutputs[i] = outputs[i];
        mIspc.mInterpolators[i] = interpolators[i];
    }

    sortEntries(numEntries, mIspc.mInputs, mIspc.mOutputs, mIspc.mInterpolators);

    // compute slopes
    evaluateMonotoneSlopes(numEntries,
                           mIspc.mInputs,
                           mIspc.mOutputs,
                           mIspc.mInterpolators,
                           mIspc.mSlopes);
}

float
FloatRampControl::eval1D(float t) const {
    if (mIspc.mNumEntries == 0) {
        return  0.0f;
    }
    auto blendAdjustmentCallback = [&](const float&, const float&) { };
    float result = eval1DRamp(mIspc.mNumEntries, mIspc.mInputs, mIspc.mOutputs, mIspc.mInterpolators,
                             mIspc.mSlopes, t, blendAdjustmentCallback);
    return result;
}

float
FloatRampControl::eval2D(const Vec2f& uv,
                               ispc::RampInterpolator2DType rampType2D,
                               float inputRamp) const {
    if (mIspc.mNumEntries == 0) {
        return 1.0f;
    }
    auto blendAdjustmentCallback = [&](float& left, float& right) { };
    float result = eval2DRamp(mIspc.mNumEntries, mIspc.mInputs, mIspc.mOutputs, mIspc.mInterpolators,
                             mIspc.mSlopes, uv, rampType2D, inputRamp, blendAdjustmentCallback);

    return result;
}

void
ColorRampControl::init(int numEntries,
                       const float* inputs,
                       const Color* outputs,
                       const ispc::RampInterpolatorMode* interpolators,
                       const ispc::ColorRampControlSpace colorSpace,
                       const bool applyHueBlendAdjustment)
{
    mIspc.mApplyHueBlendAdjustment = applyHueBlendAdjustment;

    // ignore any extra points
    numEntries = math::min(numEntries, ispc::RAMP_MAX_POINTS);

    // copy inputs/outputs/interpolators
    mIspc.mNumEntries = numEntries;
    for (int i = 0; i < numEntries; ++i) {
        mIspc.mInputs[i] = inputs[i];
        asCpp(mIspc.mOutputs[i]) = outputs[i];
        mIspc.mInterpolators[i] = interpolators[i];
    }

    sortEntries(numEntries, mIspc.mInputs, mIspc.mOutputs, mIspc.mInterpolators);

    evaluateMonotoneSlopes(numEntries,
                           mIspc.mInputs,
                           asCpp(mIspc.mOutputs),
                           mIspc.mInterpolators,
                           asCpp(mIspc.mSlopes));

    mIspc.mColorSpace = colorSpace;

    // CATMULLROM interpolation requires 4 control points. But HSV has special logic to wrap
    // around for 2 control points and dealing with wrap for 4 control points is not supported
    // if any CATMULLROM interpolator type is detected, then we force RGB ramp interpolation mode
    if(mIspc.mColorSpace == ispc::COLOR_RAMP_CONTROL_SPACE_HSV ||
       mIspc.mColorSpace == ispc::COLOR_RAMP_CONTROL_SPACE_HSL) {
        for(int i = 0; i < numEntries; i++) {
            if (mIspc.mInterpolators[i] == ispc::RAMP_INTERPOLATOR_MODE_CATMULLROM) {
                mIspc.mColorSpace = ispc::COLOR_RAMP_CONTROL_SPACE_RGB;
                break;
            }
        }
    }

    //convert all colors to right space based on the color space mode
    if(mIspc.mColorSpace == ispc::COLOR_RAMP_CONTROL_SPACE_HSV) {
        for(int i = 0; i < numEntries; i++) {
            Color rgb = asCpp(mIspc.mOutputs[i]);
            asCpp(mIspc.mOutputs[i]) = rgbToHsv(rgb);
        }
    } else if(mIspc.mColorSpace == ispc::COLOR_RAMP_CONTROL_SPACE_HSL) {
        for(int i = 0; i < numEntries; i++) {
            Color rgb = asCpp(mIspc.mOutputs[i]);
            asCpp(mIspc.mOutputs[i]) = rgbToHsl(rgb);
        }
    }
}

void
ColorRampControl::blendAdjustment(Color& left, Color& right) const {
    if (mIspc.mColorSpace == ispc::COLOR_RAMP_CONTROL_SPACE_HSV ||
        mIspc.mColorSpace == ispc::COLOR_RAMP_CONTROL_SPACE_HSL) {
        if(mIspc.mApplyHueBlendAdjustment) {
            // HSV/HSL mode
            // Ensure that HSV/HSL color blending takes the shortest route
            // around the hue wheel.
            if ((left.r + 1.f) - right.r < 0.5f ) {
                left.r += 1.f;
            } else if ((right.r + 1.f) - left.r < 0.5f) {
                right.r += 1.f;
            }
        }
    }
}

Color
ColorRampControl::eval1D(float t) const
{
    if (mIspc.mNumEntries == 0) {
        return math::sBlack;
    }

    Color result = eval1DRamp(mIspc.mNumEntries,
                            mIspc.mInputs,
                            reinterpret_cast<const Color*>(mIspc.mOutputs),
                            mIspc.mInterpolators,
                            reinterpret_cast<const Color*>(mIspc.mSlopes),
                            t,
                            [&](Color& left, Color& right) {
                                blendAdjustment(left, right);
                            });

    // convert back to HSV/HSL after evaluation if needed
    if (mIspc.mColorSpace == ispc::COLOR_RAMP_CONTROL_SPACE_HSV) {
        result = math::hsvToRgb(result);
    } else if (mIspc.mColorSpace == ispc::COLOR_RAMP_CONTROL_SPACE_HSL) {
        result = math::hslToRgb(result);
    }

    return result;
}

Color
ColorRampControl::eval2D(const Vec2f& uv,
                               ispc::RampInterpolator2DType rampType2D,
                               float inputRamp) const
{
    if (mIspc.mNumEntries == 0) {
        return math::sBlack;
    }

    Color result;
    const Color* outputs = reinterpret_cast<const Color*>(mIspc.mOutputs);

    if (mIspc.mNumEntries == 1) {
        result = outputs[0];
    } else {
        result = eval2DRamp(mIspc.mNumEntries,
                              mIspc.mInputs,
                              outputs,
                              mIspc.mInterpolators,
                              reinterpret_cast<const Color*>(mIspc.mSlopes),
                              uv,
                              rampType2D,
                              inputRamp,
                              [&](Color& left, Color& right) {
                                  blendAdjustment(left, right);
                              });
    }

    // convert back to HSV/HSL after evaluation if needed
    if (mIspc.mColorSpace == ispc::COLOR_RAMP_CONTROL_SPACE_HSV) {
        result = scene_rdl2::math::hsvToRgb(result);
    } else if (mIspc.mColorSpace == ispc::COLOR_RAMP_CONTROL_SPACE_HSL) {
        result = scene_rdl2::math::hslToRgb(result);
    }

    return result;
}

} // end namespace shading
} // end namespace moonray
