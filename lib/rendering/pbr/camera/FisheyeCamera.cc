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
    mRadialScale(0.0f)
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

    const float w = getApertureWindowWidth();
    const float h = getApertureWindowHeight();

    int format  = getRdlCamera()->get(sFormatKey);
    float diameter;
    switch (format) {
    case 0:
        // circular
        diameter = scene_rdl2::math::min(w, h);
        break;
    case 1:
        // cropped
        diameter = scene_rdl2::math::max(w, h);
        break;
    case 2:
    default:
        // diagonal
        diameter = scene_rdl2::math::sqrt(w*w + h*h);
        break;
    }
    mRadialScale = 2.0f / diameter;

    const float zoom = getRdlCamera()->get(sZoomKey);
    mRadialScale /= zoom;
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

    *dstRay = mcrt_common::RayDifferential(
        org, dir,
        org, dir_dx,
        org, dir_dy,
        getNear(), getFar(), time, 0);
}

Vec3f FisheyeCamera::createDirection(float x, float y) const
{
    const float w = getApertureWindowWidth();
    const float h = getApertureWindowHeight();

    x -= 0.5f * w;
    y -= 0.5f * h;

    const float d = scene_rdl2::math::sqrt(x*x + y*y);
    const float r = d * mRadialScale;

    float sintheta, costheta;
    switch (mMapping) {
    case 0: {
        // stereographic
        // theta = 2.0f * atan(r);
        float q = 1.0f / (1.0f + r*r);
        sintheta = 2.0f * r * q;
        costheta = (1.0f - r*r) * q;
    }
    break;
    case 1: {
        // equidistant
        // theta = sHalfPi * r;
        sincos(sHalfPi * r, &sintheta, &costheta);
    }
    break;
    case 2: {
        // equisolid angle
        // theta = 2.0f * asin(r * sqrt(0.5f));
        sintheta = r * scene_rdl2::math::sqrt(2.0f - r*r);
        costheta = 1.0f - r*r;
    }
    break;
    case 3:
    default: {
        // orthographic
        // theta = asin(r);
        sintheta = r;
        costheta = scene_rdl2::math::sqrt(1.0f - r*r);
    }
    break;
    }

    const float cosphi = x / d;
    const float sinphi = y / d;

    Vec3f dir(sintheta * cosphi, sintheta * sinphi, -costheta);

    return dir;
}

} // namespace pbr
} // namespace moonray

