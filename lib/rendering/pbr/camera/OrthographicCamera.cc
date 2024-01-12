// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

#include "Camera.h"
#include "OrthographicCamera.h"

#include <moonray/rendering/mcrt_common/Ray.h>
#include <scene_rdl2/scene/rdl2/rdl2.h>

namespace moonray {
namespace pbr {

using namespace scene_rdl2::math;

namespace {
// Pass by value on purpose. We'd be making a copy anyway. This allows the
// compiler to optimize in some cases.
finline mcrt_common::Ray modifyForDofOrtho(mcrt_common::Ray ray,
                                           float focalDistance,
                                           float lensX,
                                           float lensY)
{
    // Project back from the near clipping plane to the camera plane -- the
    // direction is in the negative z-direction.
    Vec3f cameraPlanePoint = ray.org;
    cameraPlanePoint.z = 0.0f;

    // The focal plane is perpendicular to the z-axis, and the ray now starts on
    // the camera plane. Calculate the t value through the center of of the lens.
    const Vec3f pfocus = cameraPlanePoint - Vec3f(0, 0, focalDistance);

    //   We add the lens offset instead of setting it (like in PBRT or our
    //   perspective camera), because setting it causes the rays to converge,
    //   giving a perspective view instead of orthographic.
    //
    //   In one dimension, it looks like this. Our initial point is 'x'. Our
    //   generated lens point is 'o'.
    //
    //   Camera     Focus
    //
    //   |           |
    //   x --------->|
    //   |         / |
    //   |        /  |
    //   |       /   |
    //   |      /    |
    //   |     /     |
    //   |    /      |
    //   |   /       |
    //   |  /        |
    //   | /         |
    //   |/          |
    //   o           |
    //   |           |
    //
    //   Points near the top in our picture will tend to generate lens points
    //   below it (just due to the position relative to a uniform distribution).
    //   Therefore, the directions from points initially at the top will tend
    //   to point upwards.  similarly, positions initially near the bottom will
    //   tend to point downward.

    // Offset the camera plane point by the lens samples.
    cameraPlanePoint += Vec3f(lensX, lensY, 0.0f);

    // Create a ray that goes from the new camera plane point to the focus point.
    ray.dir = normalize(pfocus - cameraPlanePoint);


    // Now we project from the camera plane along the ray direction back so
    // we're on the near plane again.

    // We have the camera point point (P). We want it offset to the near plane
    // (n === ray.org.z). Project n onto the direction vector using the dot
    // product, and offset from P along the direction vector by that length.

    // Camera     Near
    //
    //  |          |
    //  |          |
    // P|          |
    //  |\         |
    //  | \        |
    //  |  \       |
    //  |   \      |
    //  |    \     |
    //  |     \    |
    //  |      \   |
    //  |       \  |
    //  |        \ |
    //  |         \|
    //  |----------|
    //  |     n    |\
    //  |          | \
    //  |          |  \
    //  |          |   \
    //  |          |    \
    //  |          |     _|
    //  |          |
    //  |          |

    // Since n has two components as 0, we can turn this into a single mult
    // (the compiler would probably do it for us...).

    //const Vec3f n(0, 0, ray.org.z);
    //const float s = dot(n, ray.dir);
    // is equivalent to the following:
    const float s = ray.dir.z * ray.org.z;

    ray.org = cameraPlanePoint + ray.dir * s;

    return ray;
}
} // namespace

// dstRay is returned in render space.
void
OrthographicCamera::createDOFRay(mcrt_common::RayDifferential *dstRay,
                                 const Vec3f &raster,
                                 float lensX,
                                 float lensY,
                                 float time) const
{
    // TODO: adjust ray differentials for non-zero origin due to dof

    const Mat4f ct2render = computeCamera2Render(time);
    const Mat4f render2camera = computeRender2Camera(time);
    const Mat4f rt2render = computeRaster2Render(time, ct2render);
    const Mat4f rt2camera = rt2render * render2camera;

    const Vec3f ctStartPrimary =
        transformH(rt2camera, raster);
    const Vec3f ctStartAuxX =
        transformH(rt2camera, raster + Vec3f(1.0f, 0.0f, 0.0f));
    const Vec3f ctStartAuxY =
        transformH(rt2camera, raster + Vec3f(0.0f, 1.0f, 0.0f));

    mcrt_common::Ray p(ctStartPrimary, Vec3f(0.0f, 0.0f, -1.0f), 0);
    mcrt_common::Ray ax(ctStartAuxX, Vec3f(0.0f, 0.0f, -1.0f), 0);
    mcrt_common::Ray ay(ctStartAuxY, Vec3f(0.0f, 0.0f, -1.0f), 0);

    p  = modifyForDofOrtho(p,  getDofFocusDistance(), lensX, lensY);
    ax = modifyForDofOrtho(ax, getDofFocusDistance(), lensX, lensY);
    ay = modifyForDofOrtho(ay, getDofFocusDistance(), lensX, lensY);

    const Vec3f pdir  = transformH(ct2render, p.dir);
    const Vec3f axdir = transformH(ct2render, ax.dir);
    const Vec3f aydir = transformH(ct2render, ay.dir);

    *dstRay = mcrt_common::RayDifferential(
        transformH(ct2render, p.org), normalize(pdir),
        transformH(ct2render, ax.org), normalize(axdir),
        transformH(ct2render, ay.org), normalize(aydir),
        0.0f, getFar() - getNear(), time, 0);
}

finline void
OrthographicCamera::createSimpleRay(mcrt_common::RayDifferential *dstRay,
                                    const Vec3f &raster,
                                    float time) const
{
    const Mat4f ct2render = computeCamera2Render(time);
    const Mat4f rt2render = computeRaster2Render(time, ct2render);

    // Transform camera origin from camera-space at time t to camera-space at t=0
    const Vec3f renderStartPrimary =
        transformH(rt2render, raster);
    const Vec3f renderStartAuxX =
        transformH(rt2render, raster + Vec3f(1.0f, 0.0f, 0.0f));
    const Vec3f renderStartAuxY =
        transformH(rt2render, raster + Vec3f(0.0f, 1.0f, 0.0f));

    // Transform raster position on the near clipping plane, from raster-space
    // at time t to camera-space at t=0
    const Vec3f renderDir = normalize(transformH(ct2render, Vec3f(0.0f, 0.0f, -1.0f)));

    *dstRay = mcrt_common::RayDifferential(
        renderStartPrimary, renderDir,
        renderStartAuxX, renderDir,
        renderStartAuxY, renderDir,
        0.0f, getFar() - getNear(), time, 0);
}

OrthographicCamera::OrthographicCamera(const scene_rdl2::rdl2::Camera* rdlCamera) :
    ProjectiveCamera(rdlCamera)
{
}

void
OrthographicCamera::updateImpl(const Mat4d &world2render)
{
    ProjectiveCamera::updateImpl(world2render);
}

// Compute projection matrix
Mat4f
OrthographicCamera::computeC2S(float t) const
{
    const auto cam = getRdlCamera();
    return cam->computeProjectionMatrix(t, getWindow(), 0.0f);
}

float
OrthographicCamera::getFocalDistance() const
{
    return 1.0f;
}

} // namespace pbr
} // namespace moonray


