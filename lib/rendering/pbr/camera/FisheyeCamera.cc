// Copyright 2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

#include "FisheyeCamera.h"

#include <moonray/common/mcrt_macros/moonray_static_check.h>
#include <scene_rdl2/scene/rdl2/rdl2.h>

namespace moonray {
namespace pbr {

using namespace scene_rdl2::math;

bool FisheyeCamera::sAttributeKeyInitialized = false;
scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Int>   FisheyeCamera::sMappingKey;
scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Int>   FisheyeCamera::sFormatKey;
scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Float> FisheyeCamera::sZoomKey;


FisheyeCamera::FisheyeCamera(const scene_rdl2::rdl2::Camera* rdlCamera) :
    Camera(rdlCamera),
    mMapping(0),
    mRadialScale(0.0f),
    mDerivScale(0.0f)
{
    initAttributeKeys(rdlCamera->getSceneClass());
}

void FisheyeCamera::initAttributeKeys(const scene_rdl2::rdl2::SceneClass& sceneClass)
{
    if (sAttributeKeyInitialized) {
        return;
    }

    MOONRAY_START_NON_THREADSAFE_STATIC_WRITE
    sAttributeKeyInitialized = true;
    sMappingKey = sceneClass.getAttributeKey<scene_rdl2::rdl2::Int>  ("mapping");
    sFormatKey  = sceneClass.getAttributeKey<scene_rdl2::rdl2::Int>  ("format");
    sZoomKey    = sceneClass.getAttributeKey<scene_rdl2::rdl2::Float>("zoom");
    MOONRAY_FINISH_NON_THREADSAFE_STATIC_WRITE
}

bool FisheyeCamera::getIsDofEnabledImpl() const
{
    return false;
}

void FisheyeCamera::updateImpl(const Mat4d& world2render)
{
    mMapping = getRdlCamera()->get(sMappingKey);

    const float w = getRegionWindowWidth();
    const float h = getRegionWindowHeight();

    int format  = getRdlCamera()->get(sFormatKey);
    float diameter;
    switch (format) {
    case FORMAT_CIRCULAR:
        diameter = scene_rdl2::math::min(w, h);
        break;
    case FORMAT_CROPPED:
        diameter = scene_rdl2::math::max(w, h);
        break;
    case FORMAT_DIAGONAL:
        diameter = scene_rdl2::math::sqrt(w*w + h*h);
        break;
    default:
        MNRY_ASSERT_REQUIRE(false, "Unsupported case label.");
        break;
    }
    mRadialScale = 2.0f / diameter;

    const float zoom = getRdlCamera()->get(sZoomKey);
    mRadialScale /= zoom;

    mDerivScale = 1.0f / (mRadialScale * h);
}

void FisheyeCamera::createRayImpl(mcrt_common::RayDifferential* dstRay,
                                    float x,
                                    float y,
                                    float time,
                                    float /*lensU*/,
                                    float /*lensV*/) const
{
    // Compute transforms
    Mat4f ct2render;    // "camera space at ray time" --> render space
    if (getMotionBlur()) {
        ct2render = computeCamera2Render(time);
    } else {
        time = 0.0f;
        ct2render = getCamera2Render();
    }

    const Vec3f org    = transformPoint(ct2render, Vec3f(0, 0, 0));
    const Vec3f dir    = transformVector(ct2render, createDirection(x, y));
    const Vec3f dir_dx = transformVector(ct2render, createDirection(x+1.0f, y));
    const Vec3f dir_dy = transformVector(ct2render, createDirection(x, y+1.0f));

    // A fisheye lens needs to capture only the hemisphere in front of the camera,
    // i.e. with -ve z-component. We invalidate the ray if it has +ve z-component.
    // This invalidation is done by setting the ray's near and far to sMaxValue.
    float near, far;
    if (dir.z <= 0.0f) {
        near = getNear();
        far  = getFar();
    } else {
        // invalidate backwards-pointing rays
        near = scene_rdl2::math::sMaxValue;
        far  = scene_rdl2::math::sMaxValue;
    }

    *dstRay = mcrt_common::RayDifferential(
        org, dir,
        org, dir_dx,
        org, dir_dy,
        near, far, time, 0);
}

Vec3f FisheyeCamera::createDirection(float X, float Y) const
{
    const float W = getRegionWindowWidth();
    const float H = getRegionWindowHeight();

    X -= 0.5f * W;
    Y -= 0.5f * H;

    const float D = scene_rdl2::math::sqrt(X*X + Y*Y);
    const float R = D * mRadialScale;

    float sintheta, costheta;
    switch (mMapping) {
    case MAPPING_STEREOGRAPHIC: {
        // theta = 2.0f * atan(R);
        float Q = 1.0f / (1.0f + R*R);
        sintheta = 2.0f * R * Q;
        costheta = (1.0f - R*R) * Q;
        break;
    }
    case MAPPING_EQUIDISTANT:
        // theta = sHalfPi * R;
        sincos(sHalfPi * R, &sintheta, &costheta);
        break;
    case MAPPING_EQUISOLID_ANGLE:
        // theta = 2.0f * asin(R * sqrt(0.5f));
        sintheta = R * scene_rdl2::math::sqrt(2.0f - R*R);
        costheta = 1.0f - R*R;
        break;
    case MAPPING_ORTHOGRAPHIC:
        // theta = asin(R);
        sintheta = R;
        costheta = scene_rdl2::math::sqrt(1.0f - R*R);
        break;
    default:
        MNRY_ASSERT_REQUIRE(false, "Unsupported case label.");
        break;
    }

    const float cosphi = D ? X/D : 1.0f;
    const float sinphi = D ? Y/D : 0.0f;

    Vec3f dir(sintheta * cosphi, sintheta * sinphi, -costheta);

    return dir;
}

void
FisheyeCamera::computeFishtumImpl(mcrt_common::Fishtum *f, float t,
                                      bool useRenderRegion) const
{
    f->mRadialScale = mRadialScale;
    f->mDerivScale  = mDerivScale;
    f->mMapping     = mMapping;
    f->mWidth       = useRenderRegion ? (int)getRegionWindowWidth()  : (int)getApertureWindowWidth();
    f->mHeight      = useRenderRegion ? (int)getRegionWindowHeight() : (int)getApertureWindowHeight();
}

} // namespace pbr
} // namespace moonray

