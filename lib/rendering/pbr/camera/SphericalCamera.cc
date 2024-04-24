// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

#include "SphericalCamera.h"

#include <moonray/common/mcrt_macros/moonray_static_check.h>
#include <scene_rdl2/scene/rdl2/rdl2.h>

namespace moonray {
namespace pbr {

using namespace scene_rdl2::math;

bool SphericalCamera::sAttributeKeyInitialized = false;
scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Bool>  SphericalCamera::sInsideOutKey;
scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Float> SphericalCamera::sOffsetRadiusKey;


SphericalCamera::SphericalCamera(const scene_rdl2::rdl2::Camera* rdlCamera) :
    Camera(rdlCamera),
    mInsideOut(false),
    mOffsetRadius(0.0f)
{
    initAttributeKeys(rdlCamera->getSceneClass());
}

void SphericalCamera::initAttributeKeys(const scene_rdl2::rdl2::SceneClass& sceneClass)
{
    if (sAttributeKeyInitialized) {
        return;
    }

    MOONRAY_START_NON_THREADSAFE_STATIC_WRITE
    sAttributeKeyInitialized = true;
    sInsideOutKey    = sceneClass.getAttributeKey<scene_rdl2::rdl2::Bool>("inside_out");
    sOffsetRadiusKey = sceneClass.getAttributeKey<scene_rdl2::rdl2::Float>("offset_radius");
    MOONRAY_FINISH_NON_THREADSAFE_STATIC_WRITE
}

bool SphericalCamera::getIsDofEnabledImpl() const
{
    return false;
}

void SphericalCamera::updateImpl(const Mat4d& world2render)
{
    mInsideOut    = getRdlCamera()->get(sInsideOutKey);
    mOffsetRadius = getRdlCamera()->get(sOffsetRadiusKey);
}

void SphericalCamera::createRayImpl(mcrt_common::RayDifferential* dstRay,
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

    if (mInsideOut) {
        const Vec3f offset_org = org + mOffsetRadius * dir;
        *dstRay = mcrt_common::RayDifferential(
            offset_org, -dir,
            offset_org, -dir_dx,
            offset_org, -dir_dy,
            getNear(), getFar(), time, 0);
    } else {
        *dstRay = mcrt_common::RayDifferential(
            org, dir,
            org, dir_dx,
            org, dir_dy,
            getNear(), getFar(), time, 0);
    }
}

Vec3f SphericalCamera::createDirection(float x, float y) const
{
    const float width  = getApertureWindowWidth();
    const float height = getApertureWindowHeight();

    const float theta = sPi * y / height;
    const float phi = 2.0f * sPi * x / width;

    float sintheta, costheta;
    float sinphi, cosphi;

    // theta is in [0, pi) (excluding filter importance sampling). Subtract from
    // pi to reverse image y.
    sincos(sPi - theta, &sintheta, &costheta);
    sincos(phi, &sinphi, &cosphi);

    return Vec3f(sintheta * cosphi, costheta, sintheta * sinphi);
}

} // namespace pbr
} // namespace moonray

