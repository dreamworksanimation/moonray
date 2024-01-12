// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

#include "attributes.isph"

#include <moonray/rendering/pbr/core/Util.isph>
#include <moonray/rendering/shading/ispc/BasicTexture.isph>
#include <moonray/rendering/shading/ispc/UdimTexture.isph>
#include <moonray/rendering/shading/ispc/ColorCorrect.isph>
#include <moonray/rendering/shading/ispc/MapApi.isph>

enum InputTextureType {
    ST = 0,
    SURFACE_ST = 1,
    INPUT_TEXTURE_COORDINATES = 2,
    CLOSEST_SURFACE_ST = 3,
};
ISPC_UTIL_EXPORT_ENUM_TO_HEADER(InputTextureType);

struct StaticImageMapData {
    // shade time error events - these can be static as long as we
    // ensure they have the same value across all instances.
    // no conditional registration of log events is allowed.
    uniform int sErrorInvalidUdimCoord;
};
ISPC_UTIL_EXPORT_UNIFORM_STRUCT_TO_HEADER(StaticImageMapData);

struct ImageMap
{
    const uniform BASIC_TEXTURE_Data * uniform mTexture;
    const uniform UDIM_TEXTURE_Data * uniform mUdimTexture;
    uniform bool mApplyColorCorrection;
    uniform Color mFatalColor;
    uniform int mHairSurfaceSTKey;

    uniform StaticImageMapData* uniform mImageMapDataPtr;

    uniform intptr_t mRandFn;
};
ISPC_UTIL_EXPORT_UNIFORM_STRUCT_TO_HEADER(ImageMap);

varying float
getVaryingRand(uniform intptr_t randFn)
{
    typedef uniform float (* uniform GetRandFn_t)();
    GetRandFn_t getRandFn = (GetRandFn_t)randFn;
    varying float result = 0.0;
    foreach_active(lane) {
        result = insert(result, lane, getRandFn());
    }
    return result;
}

inline bool
getSTCoordinates(const uniform Map * uniform map,
                 uniform ShadingTLState * uniform  tls,
                 const varying State& state,
                 int surfaceStIndex,
                 Vec2f& st, float& dsdx, float& dtdx, float& dsdy, float& dtdy)
{
    bool isValidST = true;

    switch (getAttrTextureEnum(map)) {
    case ST:
        {
            st = state.mSt;
            dsdx = state.mdSdx;
            dtdx = state.mdTdx;
            dsdy = state.mdSdy;
            dtdy = state.mdTdy;
        }
        break;

    case SURFACE_ST:
    case CLOSEST_SURFACE_ST:
        // Hair Surface/Closest Surface UVs
        st = getVec2fAttribute(tls, state, surfaceStIndex);

        // reset texture lookup dimensions to (for now) perform point sampling
        // (assume each hair has a "point sample" wide diameter
        dsdx = dtdx = dsdy = dtdy = 0.f;
        break;

    case INPUT_TEXTURE_COORDINATES:
        {
            const Vec3f uvw = evalAttrInputTextureCoordinate(map, tls, state);
            st = asAVec2f(uvw);
            dsdx = dsdy = dtdx = dtdy = 0.0;

            if (uvw.z < 0.f) {
                // signal to caller that the 'w' coordinate was negative and
                // thus this ST coord should be considered 'invalid'
                isValidST = false;
            }
        }
        break;

    default:
        st = state.mSt;
        break;
    }

    return isValidST;
}

static void
applyColorCorrection(const uniform Map * uniform map, Color& result)
{
    if (getAttrSaturationEnabled(map)) {
        const uniform Color sat = getAttrSaturation(map);
        applySaturation(sat.r, sat.g, sat.b,
                        result.r, result.g, result.b);
    }    Vec2f st;
    float dsdx, dtdx, dsdy, dtdy;

    if (getAttrContrastEnabled(map)) {
        const uniform Color contrast = getAttrContrast(map);
        applyNukeContrast(  contrast.r, contrast.g, contrast.b,
                            result.r, result.g, result.b);
    }

    if (getAttrGammaEnabled(map)) {
        const uniform float epsilon = 0.000001;
        const uniform Color gam = getAttrGammaAdjust(map);
        const uniform Color invGamma = Color_ctor(  1.0f / max(epsilon, gam.r),
                                                    1.0f / max(epsilon, gam.g),
                                                    1.0f / max(epsilon, gam.b));
        applyGamma( invGamma.r, invGamma.g, invGamma.b,
                    result.r, result.g, result.b);
    }

    if (getAttrGainOffsetEnabled(map)) {
        const uniform Color gain = getAttrGain(map);
        const uniform Color offset = getAttrOffsetAdjust(map);
        applyGainAndOffset( gain.r, gain.g, gain.b,
                            offset.r, offset.g, offset.b,
                            result.r, result.g, result.b);
    }

    if (getAttrTMIControlEnabled(map)) {
        const uniform Vec3f tmi = getAttrTMI(map);
        applyTMI(tmi.x, tmi.y, tmi.z, result.r, result.g, result.b);
    }
}

static void
rotateTexCoords(uniform float theta, uniform const Vec2f& rotationCenter,
        varying Vec2f& st, varying float& dsdx, varying float& dsdy,
        varying float& dtdx, varying float& dtdy)
{
    uniform Mat3f R = Mat3f_ctor( cos(theta), -sin(theta),    0,
                                  sin(theta), cos(theta),     0,
                                  0,          0,              1);
    Vec3f st3 = Vec3f_ctor(st.x, st.y, 0.f);
    uniform Vec3f rotationCenter3 = Vec3f_ctor(rotationCenter.x, rotationCenter.y, 0.f);
    // Translate rotation center to origin.
    st3 = st3 - rotationCenter3;
    // Rotate.
    st3 = st3 * R;
    // Translate rotation center back.
    st3 = st3 + rotationCenter3;
    st.x = st3.x;
    st.y = st3.y;

    // Rotate derivatives.
    Vec3f dsdxy3 = Vec3f_ctor(dsdx, dsdy, 0.f);
    Vec3f dtdxy3 = Vec3f_ctor(dtdx, dtdy, 0.f);
    dsdxy3 = dsdxy3 * transpose(R);
    dtdxy3 = dtdxy3 * transpose(R);
    dsdx = dsdxy3.x;
    dsdy = dsdxy3.y;
    dtdx = dtdxy3.x;
    dtdy = dtdxy3.y;
}

static Color
sample(const uniform Map* uniform map,
       uniform ShadingTLState* uniform tls,
       const varying State& state)
{
    const uniform ImageMap * uniform me = MAP_GET_ISPC_PTR(ImageMap, map);

    Color result = Color_ctor(0.f, 0.f, 0.f);

    Vec2f st;
    float dsdx, dtdx, dsdy, dtdy;
    uniform int32_t surfaceStIndex = me->mHairSurfaceSTKey;
    const bool isValidST = getSTCoordinates(map, tls, state, surfaceStIndex, st, dsdx, dtdx, dsdy, dtdy);

    // A negative w value signals an "invalid" texture coordinate. This is
    // purely a convention.  An example usage would be where a tex coordinate
    // generator map (eg. camera projection) wants to signal to Image map that
    // the coordinate should be considered out of range, and thus we should
    // simply return black.
    if (!isValidST) {
        return sBlack;
    }

    const varying float mipBias = 1.0f + evalAttrMipBias(map, tls, state);
    const uniform Vec2f scale = getAttrScale(map);
    varying int udim = -1;
    varying int width, height;
    if (me->mTexture) {
        // rotation and scaling only for non-udim case
        const uniform Vec2f offset = getAttrOffset(map);
        const uniform Vec2f rotationCenter = getAttrRotationCenter(map);

        // Rotate coords and derivatives.
        const uniform float theta = deg2rad(getAttrRotationAngle(map));
        if (!isZero(theta)) {
            rotateTexCoords(theta,
                            rotationCenter,
                            st,
                            dsdx, dsdy, dtdx, dtdy);
        }
        // Scale and translate coords.
        st.x = scale.x * st.x + offset.x;
        st.y = scale.y * st.y + offset.y;

        // Invert t coord.
        st.y = 1.0 - st.y;

        BASIC_TEXTURE_getDimensions(me->mTexture, width, height);
    } else if (me->mUdimTexture) {
        // compute udim index
        udim = UDIM_TEXTURE_compute_udim(me->mUdimTexture, tls, st);
        if (udim == -1) {
            logEvent(map, me->mImageMapDataPtr->sErrorInvalidUdimCoord);
            return Color_ctor(me->mUdimTexture->mFatalColor.r,
                              me->mUdimTexture->mFatalColor.g,
                              me->mUdimTexture->mFatalColor.b);
        }

        // take fractional parts of st
        st.x = st.x - (int)st.x;
        st.y = st.y - (int)st.y;

        // Invert t coord
        st.y = 1.0 - st.y;

        UDIM_TEXTURE_getDimensions(me->mUdimTexture, udim, width, height);
    }

    varying Col4f tx;

    if (me->mTexture) {
        // Set and scale derivatives.
        float derivatives[4] = { dsdx * scale.x * mipBias,
                                -dtdx * scale.x * mipBias,
                                 dsdy * scale.y * mipBias,
                                -dtdy * scale.y * mipBias };

        tx = BASIC_TEXTURE_sample(me->mTexture,
                                  tls,
                                  state,
                                  st,
                                  derivatives);
    } else if (me->mUdimTexture) {
        // Set and scale derivatives.
        float derivatives[4] = { dsdx * mipBias,
                                -dtdx * mipBias,
                                 dsdy * mipBias,
                                -dtdy * mipBias };

        tx = UDIM_TEXTURE_sample(me->mUdimTexture,
                                 tls,
                                 state,
                                 udim,
                                 st,
                                 derivatives);
    }


    if (getAttrAlphaOnly(map)) {
        result = Color_ctor(tx.a, tx.a, tx.a);
    } else {
        result = Color_ctor(tx.r, tx.g, tx.b);
    }

    if (me->mApplyColorCorrection) {
        applyColorCorrection(map, result);
    }

    return result;
}

DEFINE_MAP_SHADER(ImageMap, sample)

