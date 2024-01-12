// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

#include "SphericalCamera.h"
#include <scene_rdl2/scene/rdl2/rdl2.h>

namespace moonray {
namespace pbr {

using namespace scene_rdl2::math;

SphericalCamera::SphericalCamera(const scene_rdl2::rdl2::Camera* rdlCamera) :
    Camera(rdlCamera)
{
}

bool SphericalCamera::getIsDofEnabledImpl() const
{
    return false;
}

void SphericalCamera::updateImpl(const Mat4d& world2render)
{
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

    const Vec3f cameraOrigin = transformPoint(ct2render, Vec3f(0, 0, 0));
    *dstRay = mcrt_common::RayDifferential(
        cameraOrigin, transformVector(ct2render, createDirection(x, y)),
        cameraOrigin, transformVector(ct2render, createDirection(x + 1.0f, y)),
        cameraOrigin, transformVector(ct2render, createDirection(x, y + 1.0f)),
        getNear(), getFar(), time, 0);
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


