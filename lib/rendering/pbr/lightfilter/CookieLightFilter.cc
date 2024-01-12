// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

#include "CookieLightFilter.h"

#include <moonray/common/mcrt_macros/moonray_static_check.h>
#include <moonray/rendering/bvh/shading/Intersection.h>
#include <moonray/rendering/bvh/shading/ShadingTLState.h>
#include <moonray/rendering/bvh/shading/State.h>
#include <moonray/rendering/pbr/core/Util.h>
#include <moonray/rendering/pbr/lightfilter/CookieLightFilter_ispc_stubs.h>

namespace moonray {
namespace pbr{

using namespace scene_rdl2;
using namespace scene_rdl2::math;

bool CookieLightFilter::sAttributeKeyInitialized;
rdl2::AttributeKey<rdl2::SceneObject*> CookieLightFilter::sProjectorKey;
rdl2::AttributeKey<rdl2::Mat4d> CookieLightFilter::sProjectorXformKey;
rdl2::AttributeKey<rdl2::Int> CookieLightFilter::sProjectorTypeKey;
rdl2::AttributeKey<rdl2::Float> CookieLightFilter::sProjectorFocalKey;
rdl2::AttributeKey<rdl2::Float> CookieLightFilter::sProjectorFilmWidthApertureKey;
rdl2::AttributeKey<rdl2::Float> CookieLightFilter::sProjectorPixelAspectRatioKey;
rdl2::AttributeKey<rdl2::SceneObject *> CookieLightFilter::sMapShaderKey;
rdl2::AttributeKey<rdl2::Float> CookieLightFilter::sBlurNearDistanceKey;
rdl2::AttributeKey<rdl2::Float> CookieLightFilter::sBlurMidpointKey;
rdl2::AttributeKey<rdl2::Float> CookieLightFilter::sBlurFarDistanceKey;
rdl2::AttributeKey<rdl2::Float> CookieLightFilter::sBlurNearValueKey;
rdl2::AttributeKey<rdl2::Float> CookieLightFilter::sBlurMidValueKey;
rdl2::AttributeKey<rdl2::Float> CookieLightFilter::sBlurFarValueKey;
rdl2::AttributeKey<rdl2::Int> CookieLightFilter::sBlurType;
rdl2::AttributeKey<rdl2::Int> CookieLightFilter::sOutsideProjection;
rdl2::AttributeKey<rdl2::Float> CookieLightFilter::sDensityKey;
rdl2::AttributeKey<rdl2::Bool> CookieLightFilter::sInvertKey;


HUD_VALIDATOR(CookieLightFilter);

CookieLightFilter::CookieLightFilter(const rdl2::LightFilter* rdlLightFilter) :
    LightFilter(rdlLightFilter),
    mMapShader(nullptr)
{
    if (mRdlLightFilter) {
        initAttributeKeys(mRdlLightFilter->getSceneClass());
    }

    ispc::CookieLightFilter_init((ispc::CookieLightFilter *)this->asIspc());
}

void
CookieLightFilter::initAttributeKeys(const rdl2::SceneClass &sc)
{
    if (sAttributeKeyInitialized) {
        return;
    }

    MOONRAY_START_NON_THREADSAFE_STATIC_WRITE

    sAttributeKeyInitialized = true;

    sProjectorKey = sc.getAttributeKey<rdl2::SceneObject*>("projector");
    sProjectorXformKey = sc.getAttributeKey<rdl2::Mat4d>("node_xform");
    sProjectorTypeKey = sc.getAttributeKey<rdl2::Int>("projector_type");
    sProjectorFocalKey = sc.getAttributeKey<rdl2::Float>("projector_focal");
    sProjectorFilmWidthApertureKey = sc.getAttributeKey<rdl2::Float>("projector_film_width_aperture");
    sProjectorPixelAspectRatioKey = sc.getAttributeKey<rdl2::Float>("projector_pixel_aspect_ratio");
    sMapShaderKey = sc.getAttributeKey<rdl2::SceneObject *>("texture_map");
    sBlurNearDistanceKey = sc.getAttributeKey<rdl2::Float>("blur_near_distance");
    sBlurMidpointKey = sc.getAttributeKey<rdl2::Float>("blur_midpoint");
    sBlurFarDistanceKey = sc.getAttributeKey<rdl2::Float>("blur_far_distance");
    sBlurNearValueKey = sc.getAttributeKey<rdl2::Float>("blur_near_value");
    sBlurMidValueKey = sc.getAttributeKey<rdl2::Float>("blur_mid_value");
    sBlurFarValueKey = sc.getAttributeKey<rdl2::Float>("blur_far_value");
    sBlurType = sc.getAttributeKey<rdl2::Int>("blur_type");
    sOutsideProjection = sc.getAttributeKey<rdl2::Int>("outside_projection");
    sDensityKey = sc.getAttributeKey<rdl2::Float>("density");
    sInvertKey = sc.getAttributeKey<rdl2::Bool>("invert");

    MOONRAY_FINISH_NON_THREADSAFE_STATIC_WRITE
}

math::Mat4f
CookieLightFilter::computePerspectiveProjectionMatrix(float t) const
{
    // Get focal length (mm)
    float focal = mRdlLightFilter->get(sProjectorFocalKey, t);
    focal = max(focal, 0.1f);

    // Get film-back width (mm)
    float filmWidth = mRdlLightFilter->get(sProjectorFilmWidthApertureKey);
    filmWidth = max(filmWidth, 0.01f);
    float halfFilmWidth = filmWidth * 0.5f;

    // Get near plane
    float near = 1.f;

    // Get far plane
    float far = 10000.f;

    // Note that the cookie filter doesn't clip to near/far planes but reasonable
    // defaults need to be specified for the projection matrix to be constructed.
    // Other values would also work and produce an equivalent projection.

    // Get pixel aspect ratio
    float par = mRdlLightFilter->get(sProjectorPixelAspectRatioKey);
    if (isEqual(par, 0.0f)) {
        par = 1.0f;
    }

    // Compute window corners on the near plane
    float factor = near / focal;
    float windowNear[4];
    windowNear[0] = (-2.f * halfFilmWidth) * factor;
    windowNear[1] = (-2.f * halfFilmWidth) * factor / par;
    windowNear[2] = (2.f * halfFilmWidth) * factor;
    windowNear[3] = (2.f * halfFilmWidth) * factor / par;

    // See Appendix C.5 of the SGI Graphics Library Programming Guide
    // for format of projection matrix.
    Mat4f c2s;
    c2s[0][0] = (2.0f * near) / (windowNear[2] - windowNear[0]);
    c2s[0][1] = 0.0f;
    c2s[0][2] = 0.0f;
    c2s[0][3] = 0.0f;

    c2s[1][0] = 0.0f;
    c2s[1][1] = (2.0f * near) / (windowNear[3] - windowNear[1]);
    c2s[1][2] = 0.0f;
    c2s[1][3] = 0.0f;

    c2s[2][0] = (windowNear[2] + windowNear[0]) / (windowNear[2] - windowNear[0]);
    c2s[2][1] = (windowNear[3] + windowNear[1]) / (windowNear[3] - windowNear[1]);
    c2s[2][2] = -(far + near) / (far - near);
    c2s[2][3] = -1.0f;

    c2s[3][0] = 0.0f;
    c2s[3][1] = 0.0f;
    c2s[3][2] = -(2.0f * far * near) / (far - near);
    c2s[3][3] = 0.0f;

    return c2s;
}

math::Mat4f
CookieLightFilter::computeOrthoProjectionMatrix(float t) const
{
    // Get film-back width (mm)
    float filmWidth = mRdlLightFilter->get(sProjectorFilmWidthApertureKey);
    filmWidth = max(filmWidth, 0.01f);
    float halfFilmWidth = filmWidth * 0.5f;

    // Get near plane
    float near = 1.f;

    // Get far plane
    float far = 10000.f;

    // Get pixel aspect ratio
    float par = mRdlLightFilter->get(sProjectorPixelAspectRatioKey);
    if (isEqual(par, 0.0f)) {
        par = 1.0f;
    }

    // Compute window corners on the near plane
    float windowNear[4];
    windowNear[0] = (-2.f * halfFilmWidth);
    windowNear[1] = (-2.f * halfFilmWidth) / par;
    windowNear[2] = (2.f * halfFilmWidth);
    windowNear[3] = (2.f * halfFilmWidth) / par;

    // See Appendix C.5 of the SGI Graphics Library Programming Guide
    // for format of projection matrix.
    Mat4f c2s;
    c2s[0][0] = 2.0f / (windowNear[2] - windowNear[0]);
    c2s[0][1] = 0.0f;
    c2s[0][2] = 0.0f;
    c2s[0][3] = 0.0f;

    c2s[1][0] = 0.0f;
    c2s[1][1] = 2.0f / (windowNear[3] - windowNear[1]);
    c2s[1][2] = 0.0f;
    c2s[1][3] = 0.0f;

    c2s[2][0] = 0.0f;
    c2s[2][1] = 0.0f;
    c2s[2][2] = -2.0f / (far - near);
    c2s[2][3] = 0.0f;

    c2s[3][0] = -(windowNear[2] + windowNear[0])/(windowNear[2] - windowNear[0]);
    c2s[3][1] = -(windowNear[3] + windowNear[1])/(windowNear[3] - windowNear[1]);
    c2s[3][2] = -(far + near) / (far - near);
    c2s[3][3] = 1.0f;

    return c2s;
}

bool
isValidXform(const Mat4d& xf)
{
    // Check for zero scale
    return !(isZero(xf.vx[0]) && isZero(xf.vx[1]) && isZero(xf.vx[2]) &&
             isZero(xf.vy[0]) && isZero(xf.vy[1]) && isZero(xf.vy[2]) &&
             isZero(xf.vz[0]) && isZero(xf.vz[1]) && isZero(xf.vz[2]));
}

void
CookieLightFilter::update(const LightFilterMap& /*lightFilters*/,
                          const Mat4d& world2Render)
{
    if (!mRdlLightFilter) {
        return;
    }

    // Update projector render2Screen.  This transform is the guts of the filter.
    rdl2::SceneObject* projectorSo = mRdlLightFilter->get(sProjectorKey);
    if (projectorSo) {
        rdl2::Camera *projector = projectorSo->asA<rdl2::Camera>();
        if (projector != nullptr && projector->doesSupportProjectionMatrix()) {

            Mat4d render2World = world2Render.inverse();
            Mat4d projector2World0 = projector->get(rdl2::Node::sNodeXformKey, 0.0f);
            Mat4d projector2World1 = projector->get(rdl2::Node::sNodeXformKey, 1.0f);
            Mat4d world2Projector0 = projector2World0.inverse();
            Mat4d world2Projector1 = projector2World1.inverse();

            Mat4d projector2Screen = Mat4d(projector->computeProjectionMatrix(0.f, // t
                                                                              {-2.f, -2.f, 2.f, 2.f}, // window
                                                                              0.f)); // interocularOffset

            mProjectorR2S[0] = Mat4f(render2World * world2Projector0 * projector2Screen);
            mProjectorR2S[1] = Mat4f(render2World * world2Projector1 * projector2Screen);

            Mat4d projector2Render0 = projector2World0 * world2Render;
            Mat4d projector2Render1 = projector2World1 * world2Render;
            mProjectorPos[0] = projector2Render0.row3().toVec3();
            mProjectorPos[1] = projector2Render1.row3().toVec3();
            mProjectorDir[0] = -projector2Render0.row2().toVec3();
            mProjectorDir[1] = -projector2Render1.row2().toVec3();

        } else {
            mRdlLightFilter->error("Projector is invalid or does not have a projection matrix");
            mProjectorR2S[0] = Mat4f(math::one);
            mProjectorR2S[1] = Mat4f(math::one);
            mProjectorPos[0] = {0.f, 0.f, 0.f};
            mProjectorPos[1] = {0.f, 0.f, 0.f};
            mProjectorDir[0] = {0.f, 0.f, 1.f};
            mProjectorDir[1] = {0.f, 0.f, 1.f};
        }
    } else {
        // use the projector attributes
        Mat4d render2World = world2Render.inverse();
        Mat4d projector2World0 = mRdlLightFilter->get(sProjectorXformKey, 0.f);
        Mat4d projector2World1 = mRdlLightFilter->get(sProjectorXformKey, 1.f);

        if (!isValidXform(projector2World0) || !isValidXform(projector2World1)) {
            mRdlLightFilter->error("node_xform is invalid");
        }

        Mat4d world2Projector0 = projector2World0.inverse();
        Mat4d world2Projector1 = projector2World1.inverse();

        Mat4d projector2Screen;

        switch (mRdlLightFilter->get(sProjectorTypeKey)) {
        case PERSPECTIVE:
            projector2Screen = Mat4d(computePerspectiveProjectionMatrix(0.f));
        break;
        case ORTHOGRAPHIC:
            projector2Screen = Mat4d(computeOrthoProjectionMatrix(0.f));
        break;
        default:
            MNRY_ASSERT(false);
        }

        mProjectorR2S[0] = Mat4f(render2World * world2Projector0 * projector2Screen);
        mProjectorR2S[1] = Mat4f(render2World * world2Projector1 * projector2Screen);

        Mat4d projector2Render0 = projector2World0 * world2Render;
        Mat4d projector2Render1 = projector2World1 * world2Render;
        mProjectorPos[0] = projector2Render0.row3().toVec3();
        mProjectorPos[1] = projector2Render1.row3().toVec3();
        mProjectorDir[0] = -projector2Render0.row2().toVec3();
        mProjectorDir[1] = -projector2Render1.row2().toVec3();
    }

    // update image map
    rdl2::SceneObject* mapSo = mRdlLightFilter->get(sMapShaderKey);
    if (mapSo) {
        // ispc ptr is int64
        mMapShader = (const int64_t *) mapSo->asA<rdl2::Map>();
        if (mMapShader == nullptr) {
            mRdlLightFilter->error("Map shader is invalid");
        }
    } else {
        mRdlLightFilter->error("Map shader does not exist");
        mMapShader = nullptr;
    }

    mBlurNearDistance = mRdlLightFilter->get<rdl2::Float>(sBlurNearDistanceKey);
    mBlurMidpoint = mRdlLightFilter->get<rdl2::Float>(sBlurMidpointKey);
    mBlurFarDistance = mRdlLightFilter->get<rdl2::Float>(sBlurFarDistanceKey);
    mBlurNearValue = mRdlLightFilter->get<rdl2::Float>(sBlurNearValueKey);
    mBlurMidValue = mRdlLightFilter->get<rdl2::Float>(sBlurMidValueKey);
    mBlurFarValue = mRdlLightFilter->get<rdl2::Float>(sBlurFarValueKey);
    mBlurType = mRdlLightFilter->get<rdl2::Int>(sBlurType);

    mOutsideProjection = mRdlLightFilter->get<rdl2::Int>(sOutsideProjection);

    mDensity = clamp(mRdlLightFilter->get<rdl2::Float>(sDensityKey), 0.f, 1.f);
    mInvert = mRdlLightFilter->get<rdl2::Bool>(sInvertKey);
}

bool
CookieLightFilter::canIlluminate(const CanIlluminateData& data) const
{
    Vec3f projectorPos = lerp(mProjectorPos[0], mProjectorPos[1], data.time);
    Vec3f projectorDir = lerp(mProjectorDir[0], mProjectorDir[1], data.time);
    Vec3f shadingVec = data.shadingPointPosition - projectorPos;

    if (dot(shadingVec, projectorDir) < 0.f) {
        // The shading point is behind the projector.
        switch (mOutsideProjection) {
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

bool
CookieLightFilter::needsSamples() const {
    return mBlurNearValue > 0.f || mBlurMidValue > 0.f || mBlurFarValue > 0.f;
}

Color
CookieLightFilter::eval(const EvalData& data) const
{
    if (!mMapShader) {
        return Color(1.f);
    }

    Vec3f projectorPos = lerp(mProjectorPos[0], mProjectorPos[1], data.time);
    Vec3f projectorDir = lerp(mProjectorDir[0], mProjectorDir[1], data.time);
    Vec3f shadingVec = data.shadingPointPosition - projectorPos;
    float distanceToProjector = length(shadingVec);

    if (dot(shadingVec, projectorDir) < 0.f) {
        // The shading point is behind the projector, we don't need to evaluate the
        // map shader.
        switch (mOutsideProjection) {
        case BLACK:
            return Color(0.f);
        case WHITE:
            return Color(1.f);
        default:
            return Color(0.f);
        }
    }

    // We need the screen space position of the render space point.
    Vec3f screenP0 = transformH(mProjectorR2S[0], data.shadingPointPosition);
    Vec3f screenP1 = transformH(mProjectorR2S[1], data.shadingPointPosition);
    Vec3f screenP = lerp(screenP0, screenP1, data.time);

    // Right now we are in screen space where (0, 0) is the center of the projection.
    // However, for maps, the center is (0.5, 0.5) and we want the map centered
    // in the center of the projection.  So we need to add an 0.5 offset.
    screenP.x += 0.5f;
    screenP.y += 0.5f;

    // Based on the distance to the projector, we interpolate the filter radius
    // depending on whether it is in the near/mid/far interval.
    float filterRadius = 0.f; // aka blur value
    if (distanceToProjector <= mBlurNearDistance) {
        filterRadius = mBlurNearValue;
    } else if (distanceToProjector <= mBlurMidpoint) {
        float dt = mBlurMidpoint - mBlurNearDistance;
        if (dt > 0.f) {
            float t = (distanceToProjector - mBlurNearDistance) / dt;
            filterRadius = lerp(mBlurNearValue, mBlurMidValue, t);
        } else {
            filterRadius = mBlurMidValue;
        }
    } else if (distanceToProjector <= mBlurFarDistance) {
        float dt = mBlurFarDistance - mBlurMidpoint;
        if (dt > 0.f) {
            float t = (distanceToProjector - mBlurMidpoint) / dt;
            filterRadius = lerp(mBlurMidValue, mBlurFarValue, t);
        } else {
            filterRadius = mBlurFarValue;
        }
    } else {
        filterRadius = mBlurFarValue;
    }

    // Move the screen space position based on random sampling the filter

    // Box filter, kept here for debugging.
    // Vec2f offset((r.x - 0.5f) * filterRadius, (r.y - 0.5f) * filterRadius;

    Vec2f offset;

    switch (mBlurType) {
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

    Vec2f st(screenP.x + offset.x, screenP.y + offset.y);

    if (st.x < 0.f || st.x > 1.f ||
        st.y < 0.f || st.y > 1.f) {
        switch (mOutsideProjection) {
        case BLACK:
            return Color(0.f);
        case WHITE:
            return Color(1.f);
        }
    }

    // Lookup the map value using the displaced screen space position
    Color mapValue = sampleMapShader(data.tls, st);

    // Apply density scaling to allow partial light filtering
    mapValue = Color(1.f - mDensity) + mapValue * mDensity;

    // Invert the filter?
    if (mInvert) {
        mapValue = 1.f - mapValue;
    }

    return mapValue;
}

Color
CookieLightFilter::sampleMapShader(mcrt_common::ThreadLocalState* tls,
                                   const Vec2f& st) const
{
    Vec3f p(0.f);
    Vec3f n(0.f);

    shading::Intersection isect;
    isect.initMapEvaluation(&tls->mArena,
                            nullptr,  // attr table
                            nullptr,  // geom
                            nullptr,  // layer
                            0,        // assignment id
                            p,
                            n,
                            0, 0, // All texture derivatives are set to 0.
                            st,
                            0, 0,
                            0, 0);

    Color result;
    MNRY_ASSERT(mMapShader);
    const rdl2::Map* const map = reinterpret_cast<const rdl2::Map*>(mMapShader);
    MNRY_ASSERT(map);
    map->sample(tls->mShadingTls.get(), shading::State(&isect), &result);
    return result;
}

} //namespace pbr
} //namespace moonray

