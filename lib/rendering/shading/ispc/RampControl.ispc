// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

/// @file RampControl.ispc

#include "RampControl.isph"

#include <scene_rdl2/common/platform/IspcUtil.isph>
#include <scene_rdl2/common/math/ispc/ColorSpace.isph>
#include <scene_rdl2/common/math/ispc/Mat4.isph>
#include <scene_rdl2/common/math/ispc/Vec4.isph>

ISPC_UTIL_EXPORT_ENUM_TO_HEADER(RampConstants);
ISPC_UTIL_EXPORT_ENUM_TO_HEADER(RampInterpolatorMode);
ISPC_UTIL_EXPORT_ENUM_TO_HEADER(ColorRampControlSpace);
ISPC_UTIL_EXPORT_ENUM_TO_HEADER(RampInterpolator2DType);

ISPC_UTIL_EXPORT_UNIFORM_STRUCT_TO_HEADER(FloatRampControl);
ISPC_UTIL_EXPORT_UNIFORM_STRUCT_TO_HEADER(ColorRampControl);


// Simple line slope calculation: (y2 - y1) / (x2 - x1)
varying float
computeLineSlope(varying float i0, varying float o0, varying float i1, varying float o1)
{
    return i0 == i1 ? 0.0f : (o1 - o0) / (i1 - i0);
}

// TODO: move this to scene_rdl2 common/math library ?
varying float
sign(varying float value)
{
    return value < 0.0f ? -1.0f : 1.0f;
}

// Compute the curve slope at one control vertex required to enforce monotonicity
// on a cubic hermite spline
// Based on Fritsch-Carlson method
// Inputs are four control vertices, with slopes being calculated at cv1 and cv2
// @param "end" is used to indicate boundary cases, where
// end = 1 if cv1 is the first cv, end = 2 if cv2 is the last cv, end = 0 no boundary case
void
computeFKMonotonoSlope(varying float i0, varying float o0,
                       varying float i1, varying float o1,
                       varying float i2, varying float o2,
                       varying float i3, varying float o3,
                       varying int end,
                       varying float * varying slope1,
                       varying float * varying slope2)
{
    // Compute secant line slopes
    // if end==3, then there are only 2 control points
    // in that case we set all slope information to 0
    // which ends up being a smoothstep operation
    const float delta2 = (end != 2 && end != 3) ? computeLineSlope(i2, o2, i3, o3) : 0.0f;
    const float delta1 = end != 3 ? computeLineSlope(i1, o1, i2, o2) : 0.0f;
    const float delta0 = (end != 1 && end != 3) ? computeLineSlope(i0, o0, i1, o1) : 0.0f;

    if (isZero(delta1)) {
        *slope1 = 0.0f;
        *slope2 = 0.0f;
        return;
    }

    // Initialize curve slope to be convex combination of slopes of adjacent data
    // We use the average of the secant slopes
    if (end == 1) { // cv0 is not provided (beginning of the spline)
        *slope1 = delta1;
        *slope2 = (sign(delta1) == sign(delta2)) ? (delta2 + delta1) / 2.0f : 0.0f;
    } else if (end == 2) { // cv3 is not provided (end of the spline)
        *slope1 = (sign(delta0) == sign(delta1)) ? (delta0 + delta1) / 2.0f : 0.0f;
        *slope2 = delta1;
    } else { // all four cvs are provided (non-boundary cases on the spline)
        *slope1 = (sign(delta0) == sign(delta1)) ? (delta0 + delta1) / 2.0f : 0.0f;
        *slope2 = (sign(delta1) == sign(delta2)) ? (delta2 + delta1) / 2.0f : 0.0f;
    }

    const float alpha = *slope1 / delta1;
    const float beta = *slope2 / delta1;

    if (alpha < 0.0f || beta < 0.0f) {
        // Data not monotone, return calculated slopes as initialized
        return;
    } else {
        if (alpha*alpha + beta*beta > 9.0f) {
            const float tau = 3.0f / sqrt(alpha*alpha + beta*beta);
            *slope1 = tau * *slope1;
            *slope2 = tau * *slope2;
        }
    }
}

// Given four control vertices in 1D and a curve parameterization
// value t in [0, 1], returns a value interpolated by Centripital
// Catmull Rom spline.
varying float
interpolateCatmullRom(const varying float t, const varying float p[4])
{
    const varying Vec4f controlPoints = Vec4f_ctor(p[0], p[1], p[2], p[3]);
    const uniform Mat4f catmullMatrix = Mat4f_ctor(Vec4f_ctor( 0.0f,  2.0f,  0.0f,  0.0f),
                                                   Vec4f_ctor(-1.0f,  0.0f,  1.0f,  0.0f),
                                                   Vec4f_ctor( 2.0f, -5.0f,  4.0f, -1.0f),
                                                   Vec4f_ctor(-1.0f,  3.0f, -3.0f,  1.0f));

    const varying Vec4f tVec = Vec4f_ctor(1, t, t * t, t * t * t);
    return dot(0.5f * tVec * catmullMatrix, controlPoints);
}

varying Color
interpolateCatmullRom(const varying float t, const varying Color p[4])
{
    const float r[4] = {p[0].r, p[1].r, p[2].r, p[3].r};
    const float g[4] = {p[0].g, p[1].g, p[2].g, p[3].g};
    const float b[4] = {p[0].b, p[1].b, p[2].b, p[3].b};

    return Col3f_ctor(interpolateCatmullRom(t, r),
                      interpolateCatmullRom(t, g),
                      interpolateCatmullRom(t, b));
}

varying float
interpolateCubicHermite(const varying float t,
                        const varying float x1, const varying float x2,
                        const varying float y1, const varying float y2,
                        const varying float slope1, const varying float slope2)
{
    const varying float H = x2 - x1;
    const varying float t2 = t * t;
    const varying float t3 = t2 * t;

    // Compute cubic hermite basis functions
    const varying float h00 = 2.0f * t3 - 3.0f * t2 + 1.0f;
    const varying float h10 = t3 - 2.0f * t2 + t;
    const varying float h01 = (-2.0f * t3) + 3.0f * t2;
    const varying float h11 = t3 - t2;

    return h00 * y1 + H * h10 * slope1 + h01 * y2 + H * h11 * slope2;
}

varying Color
interpolateCubicHermite(const varying float t,
                        const varying float x1, const varying float x2,
                        const varying Color y1, const varying Color y2,
                        const varying Color slope1, const varying Color slope2)
{
    return Col3f_ctor(
        interpolateCubicHermite(t, x1, x2, y1.r, y2.r, slope1.r, slope2.r),
        interpolateCubicHermite(t, x1, x2, y1.g, y2.g, slope1.g, slope2.g),
        interpolateCubicHermite(t, x1, x2, y1.b, y2.b, slope1.b, slope2.b));
}

void
evaluateMonotoneSlopes(varying FloatRampControl* uniform rampControl)
{
    // If any of the ramp inputs has monotone cubic interpolation, find and cache
    // curve slopes at each point
    bool usesCubicInterpolation = false;
    cfor (int i = 0; i < rampControl->mNumEntries; ++i) {
        cif(rampControl->mInterpolators[i] == RAMP_INTERPOLATOR_MODE_MONOTONECUBIC) {
            usesCubicInterpolation = true;
            break;
        }
    }

    cif (!usesCubicInterpolation) {
        return;
    }

    // Compute curve slopes at each control vertex to enforce monotonicity
    varying int idx;
    cfor (idx = 0; idx < rampControl->mNumEntries - 1; idx += 2) {
        int k0 = idx - 1, k1 = idx, k2 = idx + 1, k3 = idx + 2;
        int end = 0;

        cif (idx == 0) {
            k0 = k1; // id0 will not be used
            end = 1;
        }
        cif (idx+1 == rampControl->mNumEntries - 1) {
            k3 = k2; // id3 will not be used
            end = (end == 1) ? 3 : 2; //3 is special case if there are only 2 points
        }
        computeFKMonotonoSlope(rampControl->mInputs[k0], rampControl->mOutputs[k0],
                               rampControl->mInputs[k1], rampControl->mOutputs[k1],
                               rampControl->mInputs[k2], rampControl->mOutputs[k2],
                               rampControl->mInputs[k3], rampControl->mOutputs[k3],
                               end, &rampControl->mSlopes[k1], &rampControl->mSlopes[k2]);
    }
    cif (idx == rampControl->mNumEntries - 1) {

        // Last CV hasn't been assigned a slope yet, if odd number of ramp points
        const int lastIdx = rampControl->mNumEntries - 1;
        rampControl->mSlopes[lastIdx] = computeLineSlope(rampControl->mInputs[lastIdx - 1],
                                                         rampControl->mOutputs[lastIdx - 1],
                                                         rampControl->mInputs[lastIdx],
                                                         rampControl->mOutputs[lastIdx]);
    }
}

void
evaluateMonotoneSlopes(varying ColorRampControl* uniform rampControl)
{
    // If any of the ramp inputs has monotone cubic interpolation, find and cache
    // curve slopes at each point
    bool usesCubicInterpolation = false;
    cfor (int i = 0; i < rampControl->mNumEntries; ++i) {
        cif(rampControl->mInterpolators[i] == RAMP_INTERPOLATOR_MODE_MONOTONECUBIC) {
            usesCubicInterpolation = true;
            break;
        }
    }

    cif (!usesCubicInterpolation) {
        return;
    }

    // Compute curve slopes at each control vertex to enforce monotonicity
    varying int idx;
    cfor (idx = 0; idx < rampControl->mNumEntries - 1; idx += 2) {
        int k0 = idx - 1, k1 = idx, k2 = idx + 1, k3 = idx + 2;
        int end = 0;

        cif (idx == 0) {
            k0 = k1; // id0 will not be used
            end = 1;
        }
        cif (idx+1 == rampControl->mNumEntries - 1) {
            k3 = k2; // id3 will not be used
            end = (end == 1)? 3:2; //3 is special case if there are only 2 points
        }
        computeFKMonotonoSlope(rampControl->mInputs[k0], rampControl->mOutputs[k0].r,
                               rampControl->mInputs[k1], rampControl->mOutputs[k1].r,
                               rampControl->mInputs[k2], rampControl->mOutputs[k2].r,
                               rampControl->mInputs[k3], rampControl->mOutputs[k3].r,
                               end, &rampControl->mSlopes[k1].r, &rampControl->mSlopes[k2].r);
        computeFKMonotonoSlope(rampControl->mInputs[k0], rampControl->mOutputs[k0].g,
                               rampControl->mInputs[k1], rampControl->mOutputs[k1].g,
                               rampControl->mInputs[k2], rampControl->mOutputs[k2].g,
                               rampControl->mInputs[k3], rampControl->mOutputs[k3].g,
                               end, &rampControl->mSlopes[k1].g, &rampControl->mSlopes[k2].g);
        computeFKMonotonoSlope(rampControl->mInputs[k0], rampControl->mOutputs[k0].b,
                               rampControl->mInputs[k1], rampControl->mOutputs[k1].b,
                               rampControl->mInputs[k2], rampControl->mOutputs[k2].b,
                               rampControl->mInputs[k3], rampControl->mOutputs[k3].b,
                               end, &rampControl->mSlopes[k1].b, &rampControl->mSlopes[k2].b);
    }
    cif (idx == rampControl->mNumEntries - 1) {

        // Last CV hasn't been assigned a slope yet, if odd number of ramp points
        const int lastIdx = rampControl->mNumEntries - 1;
        rampControl->mSlopes[lastIdx].r = computeLineSlope(rampControl->mInputs[lastIdx - 1],
                                                           rampControl->mOutputs[lastIdx - 1].r,
                                                           rampControl->mInputs[lastIdx],
                                                           rampControl->mOutputs[lastIdx].r);
        rampControl->mSlopes[lastIdx].g = computeLineSlope(rampControl->mInputs[lastIdx - 1],
                                                           rampControl->mOutputs[lastIdx - 1].g,
                                                           rampControl->mInputs[lastIdx],
                                                           rampControl->mOutputs[lastIdx].g);
        rampControl->mSlopes[lastIdx].b = computeLineSlope(rampControl->mInputs[lastIdx - 1],
                                                           rampControl->mOutputs[lastIdx - 1].b,
                                                           rampControl->mInputs[lastIdx],
                                                           rampControl->mOutputs[lastIdx].b);
    }
}

#define FIND_RAMP_SPAN_BODY                                 \
    leftIdx = 0;                                            \
    rightIdx = numEntries - 1;                              \
                                                            \
    if (t < inputs[leftIdx]) {                              \
        leftIdx = -1;                                       \
        return;                                             \
    }                                                       \
    if (t > inputs[rightIdx]) {                             \
        rightIdx = -1;                                      \
        return;                                             \
    }                                                       \
                                                            \
    while (rightIdx - leftIdx > 1) {                        \
        varying int mid = (leftIdx + rightIdx) / 2;         \
        if (inputs[mid] > t) {                              \
            rightIdx = mid;                                 \
        } else {                                            \
            leftIdx = mid;                                  \
        }                                                   \
    }

void
findRampSpan(const uniform int numEntries,
             const uniform float* uniform inputs,
             const varying float t,
             varying int& leftIdx,
             varying int& rightIdx)
{
    FIND_RAMP_SPAN_BODY
}

void
findRampSpan(const varying int numEntries,
             const varying float* uniform inputs,
             const varying float t,
             varying int& leftIdx,
             varying int& rightIdx)
{
    FIND_RAMP_SPAN_BODY
}

#define COMPUTE_CUBIC_SPLINE_FLOAT_BODY                                         \
    float p[4];                                                                 \
    int intervals[4];                                                           \
    float t = dist;                                                             \
                                                                                \
    intervals[1] = interval;                                                    \
    if (interval == 0) {                                                        \
        intervals[0] = intervals[1];                                            \
    } else {                                                                    \
        intervals[0] = interval - 1;                                            \
    }                                                                           \
                                                                                \
    if (interval == rampControl->mNumEntries - 2) {                             \
        intervals[2] = interval + 1;                                            \
        intervals[3] = intervals[2];                                            \
    } else if (interval == rampControl->mNumEntries - 1) {                      \
        intervals[2] = intervals[1];                                            \
        intervals[3] = intervals[2];                                            \
        t = 0.f;                                                                \
    } else {                                                                    \
        intervals[2] = interval + 1;                                            \
        intervals[3] = interval + 2;                                            \
    }                                                                           \
                                                                                \
    for (uniform int i = 0; i < 4; ++i)                                         \
        p[i] = rampControl->mOutputs[intervals[i]];                             \
                                                                                \
    float result = 0.0f;                                                        \
                                                                                \
    if(interpolator == RAMP_INTERPOLATOR_MODE_CATMULLROM) {                     \
        result = interpolateCatmullRom(t, p);                                   \
    } else if (interpolator == RAMP_INTERPOLATOR_MODE_MONOTONECUBIC) {          \
        result = interpolateCubicHermite(t,                                     \
                                         rampControl->mInputs[intervals[1]],    \
                                         rampControl->mInputs[intervals[2]],    \
                                         p[1], p[2],                            \
                                         rampControl->mSlopes[intervals[1]],    \
                                         rampControl->mSlopes[intervals[2]]);   \
    }                                                                           \
    return result;

varying float
computeCubicSplineValue(const varying int interval,
                        const varying float dist,
                        const varying int interpolator,
                        const uniform FloatRampControl* uniform rampControl)
{
    COMPUTE_CUBIC_SPLINE_FLOAT_BODY
}

varying float
computeCubicSplineValue(const varying int interval,
                        const varying float dist,
                        const varying int interpolator,
                        const varying FloatRampControl* uniform rampControl)
{
    COMPUTE_CUBIC_SPLINE_FLOAT_BODY
}

#define COMPUTE_SPLINE_COLOR_BODY                                               \
    Color p[4];                                                                 \
    int intervals[4];                                                           \
    float t = dist;                                                             \
                                                                                \
    intervals[1] = interval;                                                    \
    if (interval == 0) {                                                        \
        intervals[0] = intervals[1];                                            \
    } else {                                                                    \
        intervals[0] = interval - 1;                                            \
    }                                                                           \
                                                                                \
    if (interval == rampControl->mNumEntries - 2) {                             \
        intervals[2] = interval + 1;                                            \
        intervals[3] = intervals[2];                                            \
    } else if (interval == rampControl->mNumEntries - 1) {                      \
        intervals[2] = intervals[1];                                            \
        intervals[3] = intervals[2];                                            \
        t = 0.f;                                                                \
    } else {                                                                    \
        intervals[2] = interval + 1;                                            \
        intervals[3] = interval + 2;                                            \
    }                                                                           \
                                                                                \
    for (uniform int i = 0; i < 4; ++i)                                         \
        p[i] = rampControl->mOutputs[intervals[i]];                             \
                                                                                \
    Color result = sBlack;                                                      \
                                                                                \
    if(interpolator == RAMP_INTERPOLATOR_MODE_CATMULLROM) {                     \
        result = interpolateCatmullRom(t, p);                                   \
    } else if (interpolator == RAMP_INTERPOLATOR_MODE_MONOTONECUBIC) {          \
        result.r = interpolateCubicHermite(t,                                   \
                                           rampControl->mInputs[intervals[1]],  \
                                           rampControl->mInputs[intervals[2]],  \
                                           p[1].r, p[2].r,                      \
                                           rampControl->mSlopes[intervals[1]].r,\
                                           rampControl->mSlopes[intervals[2]].r);\
        result.g = interpolateCubicHermite(t,                                   \
                                           rampControl->mInputs[intervals[1]],  \
                                           rampControl->mInputs[intervals[2]],  \
                                           p[1].g, p[2].g,                      \
                                           rampControl->mSlopes[intervals[1]].g,\
                                           rampControl->mSlopes[intervals[2]].g);\
        result.b = interpolateCubicHermite(t,                                   \
                                           rampControl->mInputs[intervals[1]],  \
                                           rampControl->mInputs[intervals[2]],  \
                                           p[1].b, p[2].b,                      \
                                           rampControl->mSlopes[intervals[1]].b,\
                                           rampControl->mSlopes[intervals[2]].b);\
    }                                                                           \
    return result;

varying Color
computeCubicSplineValue(const varying int interval,
                        const varying float dist,
                        const varying int interpolator,
                        const uniform ColorRampControl* uniform rampControl)
{
    COMPUTE_SPLINE_COLOR_BODY
}

varying Color
computeCubicSplineValue(const varying int interval,
                        const varying float dist,
                        const varying int interpolator,
                        const varying ColorRampControl* uniform rampControl)
{
    COMPUTE_SPLINE_COLOR_BODY
}

#define EVAL_FOUR_BODY(OutputType)                      \
    float weight = (1.0f - uv.x) * (1.0f - uv.y);       \
    OutputType result = weight * input0;                \
                                                        \
    weight = uv.x * (1.0f - uv.y);                      \
    result = result + weight * input1;                  \
                                                        \
    weight = (1.0f - uv.x) * uv.y;                      \
    result = result + weight * input2;                  \
                                                        \
    weight = uv.x * uv.y;                               \
    result = result + weight * input3;                  \
                                                        \
    return result;

varying float
evaluateFourOutputRamp(const varying Vec2f& uv,
                       uniform float input0, uniform float input1,
                       uniform float input2, uniform float input3)
{
    EVAL_FOUR_BODY(float)
}

varying float
evaluateFourOutputRamp(const varying Vec2f& uv,
                       varying float input0, varying float input1,
                       varying float input2, varying float input3)
{
    EVAL_FOUR_BODY(float)
}

varying Color
evaluateFourOutputRamp(const varying Vec2f& uv,
                       const uniform Color& input0, const uniform Color& input1,
                       const uniform Color& input2, const uniform Color& input3)
{
    EVAL_FOUR_BODY(Color)
}

varying Color
evaluateFourOutputRamp(const varying Vec2f& uv,
                       const varying Color& input0, const varying Color& input1,
                       const varying Color& input2, const varying Color& input3)
{
    EVAL_FOUR_BODY(Color)
}

void
blendAdjustment(const ColorRampControlSpace colorspace,
                Color& left, Color& right)
{
    if (colorspace == COLOR_RAMP_CONTROL_SPACE_HSV ||
        colorspace == COLOR_RAMP_CONTROL_SPACE_HSL) {
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

#define EVAL_1D_FLOAT_BODY                                                                      \
    if (rampControl->mNumEntries == 0) {                                                        \
        return 0.0f;                                                                            \
    }                                                                                           \
                                                                                                \
    int leftIdx, rightIdx;                                                                      \
    findRampSpan(rampControl->mNumEntries, rampControl->mInputs, t, leftIdx, rightIdx);         \
                                                                                                \
    cif (leftIdx == -1) {                                                                       \
        return rampControl->mOutputs[0];                                                        \
    }                                                                                           \
    cif (rightIdx == -1) {                                                                      \
        return rampControl->mOutputs[rampControl->mNumEntries - 1];                             \
    }                                                                                           \
                                                                                                \
    const float spanLength = rampControl->mInputs[rightIdx] - rampControl->mInputs[leftIdx];    \
    cif (isZero(spanLength)) {                                                                  \
        return rampControl->mOutputs[leftIdx];                                                  \
    }                                                                                           \
                                                                                                \
    float result = 0.0f;                                                                        \
    float weight = 0.0f;                                                                        \
                                                                                                \
    t = (t - rampControl->mInputs[leftIdx]) / spanLength;                                       \
                                                                                                \
    RampInterpolatorMode interpolator = rampControl->mInterpolators[leftIdx];                   \
                                                                                                \
    switch (interpolator) {                                                                     \
    case RAMP_INTERPOLATOR_MODE_NONE:                                                           \
        weight = 0.0f;                                                                          \
        break;                                                                                  \
    case RAMP_INTERPOLATOR_MODE_LINEAR:                                                         \
        weight = t;                                                                             \
        break;                                                                                  \
    case RAMP_INTERPOLATOR_MODE_EXPONENTIAL_UP:                                                 \
        weight = t * t;                                                                         \
        break;                                                                                  \
    case RAMP_INTERPOLATOR_MODE_EXPONENTIAL_DOWN:                                               \
        weight = 1.0f - t;                                                                      \
        weight = 1.0f - (weight * weight);                                                      \
        break;                                                                                  \
    case RAMP_INTERPOLATOR_MODE_SMOOTH:                                                         \
        weight = sin(t *  sHalfPi);                                                             \
        break;                                                                                  \
    case RAMP_INTERPOLATOR_MODE_CATMULLROM:                                                     \
    case RAMP_INTERPOLATOR_MODE_MONOTONECUBIC:                                                  \
        result = computeCubicSplineValue(leftIdx, t, interpolator, rampControl);                \
        return result;                                                                          \
    }                                                                                           \
                                                                                                \
    float leftOutput = rampControl->mOutputs[leftIdx];                                          \
    float rightOutput = rampControl->mOutputs[rightIdx];                                        \
    result = leftOutput + weight * (rightOutput - leftOutput);                                  \

#define EVAL_1D_COLOR_BODY                                                                      \
    if (rampControl->mNumEntries == 0) {                                                        \
        return sBlack;                                                                          \
    }                                                                                           \
                                                                                                \
    int leftIdx, rightIdx;                                                                      \
    findRampSpan(rampControl->mNumEntries, rampControl->mInputs, t, leftIdx, rightIdx);         \
                                                                                                \
    cif (leftIdx == -1) {                                                                       \
        return rampControl->mOutputs[0];                                                        \
    }                                                                                           \
    cif (rightIdx == -1) {                                                                      \
        return rampControl->mOutputs[rampControl->mNumEntries - 1];                             \
    }                                                                                           \
                                                                                                \
    const float spanLength = rampControl->mInputs[rightIdx] - rampControl->mInputs[leftIdx];    \
    cif (isZero(spanLength)) {                                                                  \
        return rampControl->mOutputs[leftIdx];                                                  \
    }                                                                                           \
                                                                                                \
    Color result = sBlack;                                                                      \
    float weight = 0.0f;                                                                        \
                                                                                                \
    t = (t - rampControl->mInputs[leftIdx]) / spanLength;                                       \
                                                                                                \
    RampInterpolatorMode interpolator = rampControl->mInterpolators[leftIdx];                   \
                                                                                                \
    switch (interpolator) {                                                                     \
    case RAMP_INTERPOLATOR_MODE_NONE:                                                           \
        weight = 0.0f;                                                                          \
        break;                                                                                  \
    case RAMP_INTERPOLATOR_MODE_LINEAR:                                                         \
        weight = t;                                                                             \
        break;                                                                                  \
    case RAMP_INTERPOLATOR_MODE_EXPONENTIAL_UP:                                                 \
        weight = t * t;                                                                         \
        break;                                                                                  \
    case RAMP_INTERPOLATOR_MODE_EXPONENTIAL_DOWN:                                               \
        weight = 1.0f - t;                                                                      \
        weight = 1.0f - (weight * weight);                                                      \
        break;                                                                                  \
    case RAMP_INTERPOLATOR_MODE_SMOOTH:                                                         \
        weight = sin(t *  sHalfPi);                                                             \
        break;                                                                                  \
    case RAMP_INTERPOLATOR_MODE_CATMULLROM:                                                     \
    case RAMP_INTERPOLATOR_MODE_MONOTONECUBIC:                                                  \
        result = computeCubicSplineValue(leftIdx, t, interpolator, rampControl);                \
        return result;                                                                          \
    }                                                                                           \
                                                                                                \
    Color leftOutput = rampControl->mOutputs[leftIdx];                                          \
    Color rightOutput = rampControl->mOutputs[rightIdx];                                        \
    if (rampControl->mApplyHueBlendAdjustment) {                                                \
        blendAdjustment(rampControl->mColorSpace, leftOutput, rightOutput);                     \
    }                                                                                           \
    result = leftOutput + weight * (rightOutput - leftOutput);                                  \

varying float
FloatRampControl_eval1D(varying float t,
                        const uniform FloatRampControl* uniform rampControl)
{
    EVAL_1D_FLOAT_BODY
    return result;
}

varying float
FloatRampControl_eval1D(varying float t,
                        const varying FloatRampControl* uniform rampControl)
{
    EVAL_1D_FLOAT_BODY
    return result;
}

varying Color
eval1D(varying float t,
       const uniform ColorRampControl* uniform rampControl)
{
    EVAL_1D_COLOR_BODY
    return result;
}

varying Color
eval1D(varying float t,
       const varying ColorRampControl* uniform rampControl)
{
    EVAL_1D_COLOR_BODY
    return result;
}

varying Color
ColorRampControl_eval1D(varying float t,
                        const uniform ColorRampControl* uniform rampControl)
{
    EVAL_1D_COLOR_BODY

    if (rampControl->mColorSpace == COLOR_RAMP_CONTROL_SPACE_HSV) {
        result = hsvToRgb(result);
    } else if (rampControl->mColorSpace == COLOR_RAMP_CONTROL_SPACE_HSL) {
        result = hslToRgb(result);
    }

    return result;
}

varying Color
ColorRampControl_eval1D(varying float t,
                        const varying ColorRampControl* uniform rampControl)
{
    EVAL_1D_COLOR_BODY

    if (rampControl->mColorSpace == COLOR_RAMP_CONTROL_SPACE_HSV) {
        result = hsvToRgb(result);
    } else if (rampControl->mColorSpace == COLOR_RAMP_CONTROL_SPACE_HSL) {
        result = hslToRgb(result);
    }

    return result;
}

#define EVAL_2D_FLOAT_BODY                                                                      \
    float result = 0.0f;                                                                        \
                                                                                                \
    switch (rampType2D) {                                                                       \
    case RAMP_INTERPOLATOR_2D_TYPE_V_RAMP:                                                      \
        result = FloatRampControl_eval1D(uv.y, rampControl);                                    \
        break;                                                                                  \
    case RAMP_INTERPOLATOR_2D_TYPE_U_RAMP:                                                      \
        result = FloatRampControl_eval1D(uv.x, rampControl);                                    \
        break;                                                                                  \
    case RAMP_INTERPOLATOR_2D_TYPE_DIAGONAL_RAMP:                                               \
        result = FloatRampControl_eval1D(0.5f * (uv.x + uv.y), rampControl);                    \
        break;                                                                                  \
    case RAMP_INTERPOLATOR_2D_TYPE_RADIAL_RAMP:                                                 \
        {                                                                                       \
            float value = atan2(uv.x - 0.5f, uv.y - 0.5f);                                      \
            value = 0.5f * (1.0f + value / sPi);                                                \
            result = FloatRampControl_eval1D(value, rampControl);                               \
        }                                                                                       \
        break;                                                                                  \
    case RAMP_INTERPOLATOR_2D_TYPE_CIRCULAR_RAMP:                                               \
        {                                                                                       \
            uv.x -= 0.5f;                                                                       \
            uv.y -= 0.5f;                                                                       \
            float value = uv.x * uv.x + uv.y * uv.y;                                            \
            if (value > sEpsilon) {                                                             \
                value = sqrt(2.0f * value);                                                     \
            } else {                                                                            \
                value = 0.0f;                                                                   \
            }                                                                                   \
            result = FloatRampControl_eval1D(value, rampControl);                               \
        }                                                                                       \
        break;                                                                                  \
    case RAMP_INTERPOLATOR_2D_TYPE_BOX_RAMP:                                                    \
        {                                                                                       \
            const float value = 2.0f * max(abs(uv.x - 0.5f), abs(uv.y - 0.5f));                 \
            result = FloatRampControl_eval1D(value, rampControl);                               \
        }                                                                                       \
        break;                                                                                  \
    case RAMP_INTERPOLATOR_2D_TYPE_UxV_RAMP:                                                    \
        result = FloatRampControl_eval1D(2.0f * abs(uv.y - 0.5f), rampControl) *                \
                 FloatRampControl_eval1D(2.0f * abs(uv.x - 0.5f), rampControl);                 \
        break;                                                                                  \
    case RAMP_INTERPOLATOR_2D_TYPE_FOUR_CORNER_RAMP:                                            \
        MNRY_ASSERT(rampControl->mNumEntries == 4);                                              \
        result = evaluateFourOutputRamp(uv,                                                     \
                                        rampControl->mOutputs[0],                               \
                                        rampControl->mOutputs[1],                               \
                                        rampControl->mOutputs[2],                               \
                                        rampControl->mOutputs[3]);                              \
        break;                                                                                  \
    case RAMP_INTERPOLATOR_2D_TYPE_INPUT:                                                       \
        result = FloatRampControl_eval1D(inputRamp, rampControl);                               \
        break;                                                                                  \
    default:                                                                                    \
        MNRY_ASSERT(0);                                                                          \
    }                                                                                           \
    return result;


#define EVAL_2D_COLOR_BODY                                                                      \
    Color result = sBlack;                                                                      \
                                                                                                \
    switch (rampType2D) {                                                                       \
    case RAMP_INTERPOLATOR_2D_TYPE_V_RAMP:                                                      \
        result = eval1D(uv.y, rampControl);                                                     \
        break;                                                                                  \
    case RAMP_INTERPOLATOR_2D_TYPE_U_RAMP:                                                      \
        result = eval1D(uv.x, rampControl);                                                     \
        break;                                                                                  \
    case RAMP_INTERPOLATOR_2D_TYPE_DIAGONAL_RAMP:                                               \
        result = eval1D(0.5f * (uv.x + uv.y), rampControl);                                     \
        break;                                                                                  \
    case RAMP_INTERPOLATOR_2D_TYPE_RADIAL_RAMP:                                                 \
        {                                                                                       \
            float value = atan2(uv.x - 0.5f, uv.y - 0.5f);                                      \
            value = 0.5f * (1.0f + value / sPi);                                                \
            result = eval1D(value, rampControl);                                                \
        }                                                                                       \
        break;                                                                                  \
    case RAMP_INTERPOLATOR_2D_TYPE_CIRCULAR_RAMP:                                               \
        {                                                                                       \
            uv.x -= 0.5f;                                                                       \
            uv.y -= 0.5f;                                                                       \
            float value = uv.x * uv.x + uv.y * uv.y;                                            \
            if (value > sEpsilon) {                                                             \
                value = sqrt(2.0f * value);                                                     \
            } else {                                                                            \
                value = 0.0f;                                                                   \
            }                                                                                   \
            result = eval1D(value, rampControl);                                                \
        }                                                                                       \
        break;                                                                                  \
    case RAMP_INTERPOLATOR_2D_TYPE_BOX_RAMP:                                                    \
        {                                                                                       \
            const float value = 2.0f * max(abs(uv.x - 0.5f), abs(uv.y - 0.5f));                 \
            result = eval1D(value, rampControl);                                                \
        }                                                                                       \
        break;                                                                                  \
    case RAMP_INTERPOLATOR_2D_TYPE_UxV_RAMP:                                                    \
        result = eval1D(2.0f * abs(uv.y - 0.5f), rampControl);                                  \
        result = result * eval1D(2.0f * abs(uv.x - 0.5f), rampControl);                         \
        break;                                                                                  \
    case RAMP_INTERPOLATOR_2D_TYPE_FOUR_CORNER_RAMP:                                            \
        MNRY_ASSERT(rampControl->mNumEntries == 4);                                              \
        result = evaluateFourOutputRamp(uv,                                                     \
                                        rampControl->mOutputs[0],                               \
                                        rampControl->mOutputs[1],                               \
                                        rampControl->mOutputs[2],                               \
                                        rampControl->mOutputs[3]);                              \
        break;                                                                                  \
    case RAMP_INTERPOLATOR_2D_TYPE_INPUT:                                                       \
        result = eval1D(inputRamp, rampControl);                                                \
        break;                                                                                  \
    default:                                                                                    \
        MNRY_ASSERT(0);                                                                          \
    }                                                                                           \
                                                                                                \
    if (rampControl->mColorSpace == COLOR_RAMP_CONTROL_SPACE_HSV) {                             \
        result = hsvToRgb(result);                                                              \
    } else if (rampControl->mColorSpace == COLOR_RAMP_CONTROL_SPACE_HSL) {                      \
        result = hslToRgb(result);                                                              \
    }                                                                                           \
    return result;

varying float
FloatRampControl_eval2D(varying Vec2f uv,
                        const uniform RampInterpolator2DType rampType2D,
                        const uniform float inputRamp,
                        const varying FloatRampControl * uniform rampControl)
{
    EVAL_2D_FLOAT_BODY
}

varying float
FloatRampControl_eval2D(varying Vec2f uv,
                        const varying RampInterpolator2DType rampType2D,
                        const varying float inputRamp,
                        const varying FloatRampControl * uniform rampControl)
{
    EVAL_2D_FLOAT_BODY
}

varying Color
ColorRampControl_eval2D(varying Vec2f uv,
                        const uniform RampInterpolator2DType rampType2D,
                        const varying float inputRamp,
                        const uniform ColorRampControl * uniform rampControl)
{
    EVAL_2D_COLOR_BODY
}

varying Color
ColorRampControl_eval2D(varying Vec2f uv,
                        const varying RampInterpolator2DType rampType2D,
                        const varying float inputRamp,
                        const varying ColorRampControl * uniform rampControl)
{
    EVAL_2D_COLOR_BODY
}

varying bool
isSorted(const varying float* uniform inputs,
         const varying int numEntries)
{
    for (varying int i = 0; i < numEntries - 1; ++i) {
        if (inputs[i] > inputs[i + 1])
            return false;
    }
    return true;
}

varying bool
findMonotoneCubicInterpolation(const varying RampInterpolatorMode* uniform interpolators,
                               const varying int numEntries)
{
    for (varying int i = 0; i < numEntries - 1; ++i) {
        if (interpolators[i] == RAMP_INTERPOLATOR_MODE_MONOTONECUBIC)
            return true;
    }
    return false;
}

varying bool
findCatmullRomInterpolation(const varying RampInterpolatorMode* uniform interpolators,
                            const varying int numEntries)
{
    for (varying int i = 0; i < numEntries - 1; ++i) {
        if(interpolators[i] == RAMP_INTERPOLATOR_MODE_CATMULLROM)
            return true;
    }
    return false;
}

void
FloatRampControl_init(varying FloatRampControl* uniform rampControl,
                      varying int numEntries,
                      const varying float* uniform inputs,
                      const varying float* uniform outputs,
                      const varying RampInterpolatorMode* uniform interpolators)
{
    //the caller is expected to sort the data before calling this function
    MNRY_ASSERT(isSorted(inputs, numEntries));

    // ignore any extra points
    numEntries = min((varying int) RAMP_MAX_POINTS, numEntries);

    rampControl->mNumEntries = numEntries;

    cfor (varying int i = 0; i < numEntries; ++i) {
        rampControl->mInputs[i] = inputs[i];
        rampControl->mOutputs[i] = outputs[i];
        rampControl->mInterpolators[i] = interpolators[i];
    }

    // compute slopes
    cif (findMonotoneCubicInterpolation(rampControl->mInterpolators, rampControl->mNumEntries)) {
        evaluateMonotoneSlopes(rampControl);
    }
}

void
ColorRampControl_init(varying ColorRampControl* uniform rampControl,
                      varying int numEntries,
                      const varying float* uniform inputs,
                      const varying Color* uniform outputs,
                      const varying RampInterpolatorMode* uniform  interpolators,
                      const varying ColorRampControlSpace colorSpace,
                      const varying bool applyHueBlendAdjustment)
{
    //the caller is expected to sort the data before calling this function
    MNRY_ASSERT(isSorted(inputs, numEntries));

    // ignore any extra points
    numEntries = min((varying int) RAMP_MAX_POINTS, numEntries);

    rampControl->mNumEntries = numEntries;
    rampControl->mColorSpace = colorSpace;
    rampControl->mApplyHueBlendAdjustment = applyHueBlendAdjustment;

    cfor (varying int i = 0; i < numEntries; ++i) {
        rampControl->mInputs[i] = inputs[i];
        rampControl->mOutputs[i] = outputs[i];
        rampControl->mInterpolators[i] = interpolators[i];
    }

    // compute slopes
    cif (findMonotoneCubicInterpolation(rampControl->mInterpolators, rampControl->mNumEntries)) {
        evaluateMonotoneSlopes(rampControl);
    }

    // FIXME this behavior is not very intuitive to an artist
    // CATMULLROM interpolation requires 4 control points. But HSV has special logic to wrap
    // around for 2 control points and dealing with wrap for 4 control points is not supported
    // if any CATMULLROM interpolator type is detected, then we force RGB ramp interpolation mode
    cif (rampControl->mColorSpace == COLOR_RAMP_CONTROL_SPACE_HSV ||
         rampControl->mColorSpace == COLOR_RAMP_CONTROL_SPACE_HSL) {
       cif (findCatmullRomInterpolation(rampControl->mInterpolators, rampControl->mNumEntries)) {
            rampControl->mColorSpace = COLOR_RAMP_CONTROL_SPACE_RGB;
       }
    }

    //convert all colors to right space based on the color space mode
    cif (rampControl->mColorSpace == COLOR_RAMP_CONTROL_SPACE_HSV) {
        cfor (varying int i = 0; i < numEntries; ++i)
            rampControl->mOutputs[i] = rgbToHsv(rampControl->mOutputs[i]);
    } else cif (rampControl->mColorSpace == COLOR_RAMP_CONTROL_SPACE_HSL) {
        cfor (varying int i = 0; i < numEntries; ++i)
            rampControl->mOutputs[i] = rgbToHsl(rampControl->mOutputs[i]);
    }
}


