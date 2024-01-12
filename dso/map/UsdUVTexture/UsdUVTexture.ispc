// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

/// @file UsdUVTexture.ispc

#include "attributes.isph"

#include <moonray/rendering/shading/ispc/BasicTexture.isph>
#include <moonray/rendering/shading/ispc/ColorCorrect.isph>
#include <moonray/rendering/shading/ispc/MapApi.isph>
#include <moonray/rendering/shading/ispc/UdimTexture.isph>

enum OutputMode {
    OUTPUT_RGB = 0,
    OUTPUT_R   = 1,
    OUTPUT_G   = 2,
    OUTPUT_B   = 3,
    OUTPUT_A   = 4
};
ISPC_UTIL_EXPORT_ENUM_TO_HEADER(OutputMode);

struct StaticUsdUVTextureData {
    // shade time error events - these can be static as long as we
    // ensure they have the same value across all instances.
    // no conditional registration of log events is allowed.
    uniform int sErrorInvalidUdimCoord;
};
ISPC_UTIL_EXPORT_UNIFORM_STRUCT_TO_HEADER(StaticUsdUVTextureData);

struct UsdUVTexture
{
    const uniform BASIC_TEXTURE_Data * uniform mTexture;
    const uniform UDIM_TEXTURE_Data * uniform mUdimTexture;
    uniform Color mFatalColor;

    uniform StaticUsdUVTextureData* uniform mUsdUVTextureDataPtr;
};
ISPC_UTIL_EXPORT_UNIFORM_STRUCT_TO_HEADER(UsdUVTexture);

static Color
sample(const uniform Map* uniform map,
       uniform ShadingTLState* uniform tls,
       const varying State& state)
{
    const uniform UsdUVTexture * uniform me = MAP_GET_ISPC_PTR(UsdUVTexture, map);

    Color result = Color_ctor(0.0f, 0.0f, 0.0f);

    const Vec2f uv = evalAttrSt(map, tls, state);
    varying Vec2f st = uv;

    float dsdx, dtdx, dsdy, dtdy;
    dsdx = dsdy = dtdx = dtdy = 0.0f; // scalar mode also sets these to zero

    varying Col4f tx;
    if (me->mTexture) {
        // Invert t coord.
        st.y = 1.0 - st.y;

        // Set and scale derivatives.
        float derivatives[4] = { dsdx, -dtdx, dsdy, -dtdy };

        tx = BASIC_TEXTURE_sample(me->mTexture,
                                  tls,
                                  state,
                                  st,
                                  derivatives);
    } else if (me->mUdimTexture) {
        // compute udim index
        varying int udim = UDIM_TEXTURE_compute_udim(me->mUdimTexture,
                                                     tls, st);
        if (udim == -1) {
            logEvent(map, me->mUsdUVTextureDataPtr->sErrorInvalidUdimCoord);
            return me->mUdimTexture->mFatalColor;
        }

        // take fractional parts of st
        st.x = st.x - (int)st.x;
        st.y = st.y - (int)st.y;

        // Invert t coord.
        st.y = 1.0 - st.y;

        // Set and scale derivatives.
        float derivatives[4] = { dsdx, -dtdx,dsdy, -dtdy };

        tx = UDIM_TEXTURE_sample(me->mUdimTexture,
                                 tls,
                                 state,
                                 udim,
                                 st,
                                 derivatives);
    }

    switch (getAttrOutputMode(map)) {
    case OUTPUT_R:
        result = Color_ctor(tx.r);
        break;
    case OUTPUT_G:
        result = Color_ctor(tx.g);
        break;
    case OUTPUT_B:
        result = Color_ctor(tx.b);
        break;
    case OUTPUT_A:
        result = Color_ctor(tx.a);
        break;
    default:
        result = Color_ctor(tx.r, tx.g, tx.b);
    }

    Color scl = getAttrScale(map);
    Color bs = getAttrBias(map);
    result = result * scl + bs;

    return result;
}

DEFINE_MAP_SHADER(UsdUVTexture, sample)

