// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

#include "LightFilter.isph"
#include <moonray/rendering/pbr/core/Util.isph>
#include <moonray/rendering/shading/ispc/Shading.isph>
#include <scene_rdl2/common/platform/IspcUtil.isph>

//----------------------------------------------------------------------------

ISPC_UTIL_EXPORT_UNIFORM_STRUCT_TO_HEADER(CookieLightFilter);

export uniform uint32_t
CookieLightFilter_hudValidation(uniform bool verbose)
{
    COOKIE_LIGHT_FILTER_VALIDATION;
}

//----------------------------------------------------------------------------

varying bool
CookieLightFilter_canIlluminate(const uniform LightFilter * uniform lif,
                                const varying CanIlluminateData &data)
{
    const uniform CookieLightFilter * uniform lf = (const uniform CookieLightFilter * uniform)lif;

    Vec3f projectorPos = lerp(lf->mProjectorPos[0], lf->mProjectorPos[1], data.time);
    Vec3f projectorDir = lerp(lf->mProjectorDir[0], lf->mProjectorDir[1], data.time);
    Vec3f shadingVec = data.shadingPointPosition - projectorPos;

    if (dot(shadingVec, projectorDir) < 0.f) {
        // The shading point is behind the projector, we don't need to evaluate the
        // map shader.
        switch (lf->mOutsideProjection) {
        case BLACK:
            return false;
        case WHITE:
            return true;
        default:
            return false;
        }
    }
    return true;
}

void
sampleMapShader(uniform ShadingTLState * uniform tls,
                const uniform int64 * const uniform mapObjPtr,
                varying Vec2f st,
                varying Color * uniform color)
{
    Vec3f zeroVec = {0.f, 0.f, 0.f};

    varying State state;
    Address64_clear(&state.mGeometryObject);
    Address64_clear(&state.mMaterial);
    Address64_clear(&state.mTable);
    Address64_clear(&state.mData);
    state.mLayerAssignmentId = 0;
    state.mId1 = 0;
    state.mId2 = 0;
    state.mId3 = 0;
    state.mSt = st;
    state.mP = zeroVec;
    state.mNg = zeroVec;
    state.mN = zeroVec;
    state.mdPds = zeroVec;
    state.mdPdt = zeroVec;
    state.mdSdx = 0.f;
    state.mdSdy = 0.f;
    state.mdTdx = 0.f;
    state.mdTdy = 0.f;
    Flags_clearAll(&(state.mFlags));

    int64 * uniform ptr = (int64 * uniform) *(mapObjPtr + (MAP_SAMPLEFUNCV_OFFSET / 8));

    SamplePtr sampleFn = (SamplePtr) ptr;
    sampleFn((const Map * uniform) mapObjPtr, tls, &state, color);
}

void
CookieLightFilter_eval(const uniform LightFilter * uniform lif,
                       const varying EvalData& data,
                       varying Color * uniform filterValue)
{
    const uniform CookieLightFilter * uniform lf = (const uniform CookieLightFilter * uniform)lif;

    if (!lf->mMapShader) {
        *filterValue = Color_ctor(1.f);
        return;
    }

    Vec3f projectorPos = lerp(lf->mProjectorPos[0], lf->mProjectorPos[1], data.time);
    Vec3f projectorDir = lerp(lf->mProjectorDir[0], lf->mProjectorDir[1], data.time);
    Vec3f shadingVec = data.shadingPointPosition - projectorPos;
    float distanceToProjector = length(shadingVec);

    if (dot(shadingVec, projectorDir) < 0.f) {
        // The shading point is behind the projector, we don't need to evaluate the
        // map shader.
        switch (lf->mOutsideProjection) {
        case BLACK:
            *filterValue = Color_ctor(0.f);
            return;
        case WHITE:
            *filterValue = Color_ctor(1.f);
            return;
        default:
            *filterValue = Color_ctor(0.f);
            return;
        }
    }

    // We need the screen space position of the render space point.
    Vec4f spp = {data.shadingPointPosition.x,
                 data.shadingPointPosition.y,
                 data.shadingPointPosition.z,
                 1.f};
    Vec3f screenP0 = transformH(lf->mProjectorR2S[0], spp);
    Vec3f screenP1 = transformH(lf->mProjectorR2S[1], spp);
    Vec3f screenP = lerp(screenP0, screenP1, data.time);

    // Right now we are in screen space where (0, 0) is the center of the projection.
    // However, for maps, the center is (0.5, 0.5) and we want the map centered
    // in the center of the projection.  So we need to add an 0.5 offset.
    screenP.x += 0.5f;
    screenP.y += 0.5f;

    float filterRadius = 0.f; // aka blur value
    if (distanceToProjector <= lf->mBlurNearDistance) {
        filterRadius = lf->mBlurNearValue;
    } else if (distanceToProjector <= lf->mBlurMidpoint) {
        float dt = lf->mBlurMidpoint - lf->mBlurNearDistance;
        if (dt > 0.f) {
            float t = (distanceToProjector - lf->mBlurNearDistance) / dt;
            filterRadius = lerp(lf->mBlurNearValue, lf->mBlurMidValue, t);
        } else {
            filterRadius = lf->mBlurMidValue;
        }
    } else if (distanceToProjector <= lf->mBlurFarDistance) {
        float dt = lf->mBlurFarDistance - lf->mBlurMidpoint;
        if (dt > 0.f) {
            float t = (distanceToProjector - lf->mBlurMidpoint) / dt;
            filterRadius = lerp(lf->mBlurMidValue, lf->mBlurFarValue, t);
        } else {
            filterRadius = lf->mBlurFarValue;
        }
    } else {
        filterRadius = lf->mBlurFarValue;
    }

    // Box filter, kept here for debugging.
    // Vec2f offset = {(r.x - 0.5f) * filterRadius, (r.y - 0.5f) * filterRadius};

    Vec2f offset;

    switch (lf->mBlurType) {
    case GAUSSIAN:
        // Quadratic bspline "gaussian" filter, which is the same filter the regular camera uses.
        offset.x = quadraticBSplineWarp(data.randVar.r2.x) * filterRadius;
        offset.y = quadraticBSplineWarp(data.randVar.r2.y) * filterRadius;
        break;
    case CIRCULAR:
        offset.x = data.randVar.r2.x;
        offset.y = data.randVar.r2.y;
        toUnitDisk(offset.x, offset.y);
        offset.x *= filterRadius;
        offset.y *= filterRadius;
        break;
    default:
        MNRY_ASSERT(false);
    }

    Vec2f st = {screenP.x + offset.x, screenP.y + offset.y};

    if (st.x < 0.f || st.x > 1.f ||
        st.y < 0.f || st.y > 1.f) {
        switch (lf->mOutsideProjection) {
        case BLACK:
            *filterValue = Color_ctor(0.f);
            return;
        case WHITE:
            *filterValue = Color_ctor(1.f);
            return;
        }
    }

    sampleMapShader(data.tls, lf->mMapShader, st, filterValue);

    filterValue->r = 1.f - lf->mDensity + filterValue->r * lf->mDensity;
    filterValue->g = 1.f - lf->mDensity + filterValue->g * lf->mDensity;
    filterValue->b = 1.f - lf->mDensity + filterValue->b * lf->mDensity;

    if (lf->mInvert) {
        *filterValue = 1.f - *filterValue;
    }
}

//----------------------------------------------------------------------------

export void
CookieLightFilter_init(uniform CookieLightFilter * uniform lightfilter)
{
    LightFilter_init((uniform LightFilter * uniform)lightfilter, 
            CookieLightFilter_canIlluminate, CookieLightFilter_eval);
}

//----------------------------------------------------------------------------

