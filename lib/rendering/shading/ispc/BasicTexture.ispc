// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

/// @file BasicTexture.ispc

#include "BasicTexture.isph"

ISPC_UTIL_EXPORT_UNIFORM_STRUCT_TO_HEADER(BASIC_TEXTURE_StaticData);
ISPC_UTIL_EXPORT_UNIFORM_STRUCT_TO_HEADER(BASIC_TEXTURE_Data);

// Defined in MipSelector.ispc
varying float
computeMipSelector(
    const varying float dsdx,
    const varying float dtdx,
    const varying float dsdy,
    const varying float dtdy
);

extern "C" void
CPP_oiioTexture(const uniform BASIC_TEXTURE_Data * uniform tx,
                uniform ShadingTLState * uniform tls,
                const uniform uint32_t displacement,
                const uniform int pathType,
                const uniform float * uniform derivatives,
                const uniform float * uniform st,
                uniform float * uniform);                


Col4f
BASIC_TEXTURE_sample(
    const uniform BASIC_TEXTURE_Data * uniform tx,
    uniform ShadingTLState * uniform  tls,
    const varying State& state,
    varying Vec2f st,
    varying float (&derivatives)[4]
) {
    if (!tx->mIsValid && tx->mUseDefaultColor) {
        return Col4f_ctor(tx->mDefaultColor.r,
                          tx->mDefaultColor.g,
                          tx->mDefaultColor.b,
                          1.0f);
    }

    StartAccumulatorFn startAccumulator = (StartAccumulatorFn) tx->mBasicTextureStaticDataPtr->mStartAccumulator;
    StopAccumulatorFn stopAccumulator = (StopAccumulatorFn) tx->mBasicTextureStaticDataPtr->mStopAccumulator;

    uniform uint32_t displacement = isDisplacement(state);
    uint32_t blendMips = blendMips(state);

    float mipSelector = computeMipSelector(
        derivatives[0], derivatives[1],
        derivatives[2], derivatives[3]
    );

    // Thread local accumulator for profiling
    uniform intptr_t accumulator = startAccumulator(tls, displacement);

    Col4f sampleResult;

    PathType pathType = getPathType(state);

    foreach_active(lane) {
        uniform uint32_t displacement_lane = displacement;
        uniform int pathType_lane = extract((int)pathType, lane);
        uniform float derivatives_lane[4];
        derivatives_lane[0] = extract(derivatives[0], lane);
        derivatives_lane[1] = extract(derivatives[1], lane);
        derivatives_lane[2] = extract(derivatives[2], lane);
        derivatives_lane[3] = extract(derivatives[3], lane);
        uniform float st_lane[2];
        st_lane[0] = extract(st.x, lane);
        st_lane[1] = extract(st.y, lane);
        uniform float sampleresult_lane[4];

        CPP_oiioTexture(tx,
                        tls,
                        displacement_lane,
                        pathType_lane,
                        derivatives_lane,
                        st_lane,
                        sampleresult_lane);

        sampleResult.r = insert(sampleResult.r, lane, sampleresult_lane[0]);
        sampleResult.g = insert(sampleResult.g, lane, sampleresult_lane[1]);
        sampleResult.b = insert(sampleResult.b, lane, sampleresult_lane[2]);
        sampleResult.a = insert(sampleResult.a, lane, sampleresult_lane[3]);
    }

    stopAccumulator(accumulator);

    return sampleResult;
}

Col4f
BASIC_TEXTURE_sample(
    const uniform BASIC_TEXTURE_Data * varying tx,
    uniform ShadingTLState * uniform  tls,
    const varying State& state,
    varying Vec2f st,
    varying float (&derivatives)[4]
) {
    varying Col4f result;
    intptr_t txInt = (intptr_t)tx;
    foreach_active(lane) {
        uniform intptr_t txUniformInt = extract(txInt, lane);
        const uniform BASIC_TEXTURE_Data * uniform uniformTx = (const uniform BASIC_TEXTURE_Data * uniform)txUniformInt;
        result = BASIC_TEXTURE_sample(uniformTx, tls, state, st, derivatives);
    }
    return result;
}

void
BASIC_TEXTURE_getDimensions(
    const uniform BASIC_TEXTURE_Data * uniform tx,
    int &x, int& y)
{
    x = tx->mWidth;
    y = tx->mHeight;
}

float
BASIC_TEXTURE_getPixelAspectRatio(const uniform BASIC_TEXTURE_Data * uniform tx)
{
    return tx->mPixelAspectRatio;
}

