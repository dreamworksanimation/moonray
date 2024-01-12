// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0


/// @file UdimTexture.ispc

#include "UdimTexture.isph"

ISPC_UTIL_EXPORT_UNIFORM_STRUCT_TO_HEADER(UDIM_TEXTURE_StaticData);
ISPC_UTIL_EXPORT_UNIFORM_STRUCT_TO_HEADER(UDIM_TEXTURE_Data);

// Defined in MipSelector.ispc
varying float
computeMipSelector(
    const varying float dsdx,
    const varying float dtdx,
    const varying float dsdy,
    const varying float dtdy);

extern "C" void
CPP_oiioUdimTexture(const uniform UDIM_TEXTURE_Data * uniform tx,
                    uniform ShadingTLState * uniform tls,
                    const uniform uint32_t displacement,
                    const uniform int pathType,
                    const uniform float * uniform derivatives,
                    const uniform int udim,
                    const uniform float * uniform st,
                    uniform float * uniform);                

int
UDIM_TEXTURE_compute_udim(
    const uniform UDIM_TEXTURE_Data * uniform tx,
    uniform ShadingTLState * uniform  tls,
    varying Vec2f st)
{
    if (st.x < 0.f || st.x > 10) {
        logEvent((const uniform SceneObject * uniform) tx->mShader,
                 tx->mErrorUdimOutOfRangeU);
        return -1;
    }
    if (st.y < 0.f) {
        logEvent((const uniform SceneObject * uniform) tx->mShader,
                 tx->mErrorUdimOutOfRangeV);
        return -1;
    }
    return (int)st.x + (int)st.y * 10;
}

Col4f
UDIM_TEXTURE_sample(
    const uniform UDIM_TEXTURE_Data * uniform tx,
    uniform ShadingTLState * uniform  tls,
    const varying State& state,
    varying int udim,
    varying Vec2f st,
    varying float (&derivatives)[4])
{
    if (!tx->mIsValid && tx->mUseDefaultColor) {
        return Col4f_ctor(tx->mDefaultColor.r,
                          tx->mDefaultColor.g,
                          tx->mDefaultColor.b,
                          1.0f);
    }

    StartAccumulatorFnUdim startAccumulator = (StartAccumulatorFnUdim) tx->mUdimTextureStaticDataPtr->mStartAccumulator;
    StopAccumulatorFnUdim stopAccumulator = (StopAccumulatorFnUdim) tx->mUdimTextureStaticDataPtr->mStopAccumulator;

    uniform uint32_t displacement = isDisplacement(state);
    uint32_t blendMips = blendMips(state);

    float mipSelector = computeMipSelector(
        derivatives[0], derivatives[1],
        derivatives[2], derivatives[3]
    );

    // Thread local accumulator for profiling
    uniform intptr_t accumulator = startAccumulator(tls, displacement);

    Col4f sampleResult;

    // Sony texture system

    PathType pathType = getPathType(state);

    foreach_active(lane) {
        uniform uint32_t displacement_lane = displacement;
        uniform int pathType_lane = extract((int)pathType, lane);
        uniform float derivatives_lane[4];
        derivatives_lane[0] = extract(derivatives[0], lane);
        derivatives_lane[1] = extract(derivatives[1], lane);
        derivatives_lane[2] = extract(derivatives[2], lane);
        derivatives_lane[3] = extract(derivatives[3], lane);
        uniform int udim_lane = extract(udim, lane);
        uniform float st_lane[2];
        st_lane[0] = extract(st.x, lane);
        st_lane[1] = extract(st.y, lane);
        uniform float sampleresult_lane[4];

        CPP_oiioUdimTexture(tx,
                            tls,
                            displacement_lane,
                            pathType_lane,
                            derivatives_lane,
                            udim_lane,
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

void
UDIM_TEXTURE_getDimensions(
    const uniform UDIM_TEXTURE_Data * uniform tx,
    varying int udim,
    varying int& x, 
    varying int& y)
{
    x = tx->mWidths[udim];
    y = tx->mHeights[udim];
}

float
UDIM_TEXTURE_getPixelAspectRatio(
    const uniform UDIM_TEXTURE_Data * uniform tx,
    int udim)
{
    return tx->mPixelAspectRatios[udim];
}

