// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

#include "Camera.h"
#include "PerspectiveCamera.h"

#include <moonray/common/mcrt_macros/moonray_static_check.h>
#include <scene_rdl2/scene/rdl2/rdl2.h>

namespace moonray {
namespace pbr {

using namespace scene_rdl2;
using namespace scene_rdl2::math;

bool                            PerspectiveCamera::sAttributeKeyInitialized = false;
rdl2::AttributeKey<rdl2::Float> PerspectiveCamera::sFocalKey;

rdl2::AttributeKey<rdl2::Int>   PerspectiveCamera::sStereoView;
rdl2::AttributeKey<rdl2::Float> PerspectiveCamera::sStereoInterocularDistance;
rdl2::AttributeKey<rdl2::Float> PerspectiveCamera::sStereoConvergenceDistance;

namespace {
// focalPoint is the eye position, not to be confused with a point on the focal
// plane.
finline Vec3f
getFocusPlaneHitPoint(const Vec3f& focalPoint, const Vec3f& raster, const Mat4f& rt2c,
                      float focusDistance)
{
    const Vec3f dir = normalize(transformH(rt2c, raster) - focalPoint);

    // We want to take our initial ray (origin at mFocalPoint) and find its
    // intersection with the focal plane. All work is done in camera space.
#if 0
    const Vec3f focalPlaneNormal(0, 0, +1);
    const Vec3f focalPlanePoint(0, 0, -focusDistance);

    // Ray/plane intersection
    const float t = dot(focalPlanePoint - focalPoint, focalPlaneNormal) / dot(dir, focalPlaneNormal);
#else
    // Since we have a bunch of one-component vectors, the above code simplifies
    // to some one-dimensional math. Yay!
    // Since 'focalPoint' is in camera space, it only has 'x' set, so we don't
    // have to subtract the z value when simplifying the above vector
    // subtraction.
    const float n = -focusDistance;
    const float d = dir.z;
    const float t = n/d;
#endif
    const Vec3f pFocus = focalPoint + dir * t;
    return pFocus;
}

finline float
getClipPlaneT(float dist, const Vec3f& rayOrigin, const Vec3f& rayDirection)
{
#if 0
    // Again, here's the math for the ray/plane intersection, but because of
    // some assumptions about our coordinate space, the math simplifies.
    const Vec3f normal(0, 0, +1);
    const Vec3f planePoint(0, 0, -dist);

    // Ray/plane intersection
    const float t = dot(planePoint - rayOrigin, normal) / dot(rayDirection, normal);
    return t;
#else
    const float t = (-dist - rayOrigin.z) / rayDirection.z;
    return t;
#endif
}

// Compute signed interocular offset due to stereo (translation in camera
// space along x axis).
// Warning: This function doesn't work in CENTER mode.
finline float
computeInterocularOffset(StereoView stereoView,
                         float interocularDistance)
{
    MNRY_ASSERT(stereoView != StereoView::CENTER);

    float interocularOffset =
        (stereoView == StereoView::LEFT ? 0.5f : -0.5f) *
        interocularDistance;

    return interocularOffset;
}
} // namespace

void PerspectiveCamera::initAttributeKeys(const rdl2::SceneClass& sceneClass)
{
    if (sAttributeKeyInitialized) {
        return;
    }

    MOONRAY_START_NON_THREADSAFE_STATIC_WRITE

    sAttributeKeyInitialized = true;

    sFocalKey = sceneClass.getAttributeKey<rdl2::Float>("focal");

    sStereoView = sceneClass.getAttributeKey<rdl2::Int>("stereo_view");
    sStereoInterocularDistance = sceneClass.getAttributeKey<rdl2::Float>("stereo_interocular_distance");
    sStereoConvergenceDistance = sceneClass.getAttributeKey<rdl2::Float>("stereo_convergence_distance");

    MOONRAY_FINISH_NON_THREADSAFE_STATIC_WRITE
}

void
PerspectiveCamera::createDOFRay(mcrt_common::RayDifferential *dstRay,
                                const Vec3f &raster,
                                float lensX,
                                float lensY,
                                float time) const
{
    // TODO: adjust ray differentials for non-zero origin due to dof

    const Mat4f ct2render = computeCamera2Render(time);
    const Mat4f rt2ct     = computeRaster2Camera(time);

    // Transform camera origin on the lens, from camera-space at time t to
    // camera-space at t=0

    // mFocalPoint is in camera space
    const Vec3f nx(1.0f, 0.0f, 0.0f);
    const Vec3f ny(0.0f, 1.0f, 0.0f);
    const Vec3f ctHitPoint      = getFocusPlaneHitPoint(mFocalPoint,
                                                        raster,
                                                        rt2ct,
                                                        getDofFocusDistance());
    const Vec3f ctHitPointAuxX  = getFocusPlaneHitPoint(mFocalPoint,
                                                        raster + nx,
                                                        rt2ct,
                                                        getDofFocusDistance());
    const Vec3f ctHitPointAuxY  = getFocusPlaneHitPoint(mFocalPoint,
                                                        raster + ny,
                                                        rt2ct,
                                                        getDofFocusDistance());

    // Change the origin to that of the lens sample + mFocalPoint
    const Vec3f ctLensPoint = mFocalPoint + Vec3f(lensX, lensY, 0.0f);
    const Vec3f renderLensPoint = transformPoint(ct2render, ctLensPoint);

    const Vec3f ctDirPrimary = ctHitPoint - ctLensPoint;
    const Vec3f dirPrimaryNormalized = normalize(ctDirPrimary);
    const float tNear = getClipPlaneT(getNear(), mFocalPoint,
                                     dirPrimaryNormalized);
    const float tFar  = getClipPlaneT(getFar(), mFocalPoint,
                                     dirPrimaryNormalized);

    // Change the direction to the point intersected at the focal plane - the new
    // origin.
    *dstRay = mcrt_common::RayDifferential(
        renderLensPoint, normalize(transformVector(ct2render, ctDirPrimary)),
        renderLensPoint, normalize(transformVector(ct2render, ctHitPointAuxX - ctLensPoint)),
        renderLensPoint, normalize(transformVector(ct2render, ctHitPointAuxY - ctLensPoint)),
        tNear, tFar, time, 0);
}

void
PerspectiveCamera::createSimpleRay(mcrt_common::RayDifferential *dstRay,
                                   const Vec3f &raster,
                                   float time) const
{
    const Mat4f ct2render = computeCamera2Render(time);
    const Mat4f rt2ct     = computeRaster2Camera(time);

    const Vec3f renderStart  = transformPoint(ct2render, mFocalPoint);
    const Vec3f ctDirPrimary = transformH(rt2ct, raster) - mFocalPoint;
    const Vec3f ctDirAuxX    = transformH(rt2ct, raster + Vec3f(1.0f, 0.0f, 0.0f)) - mFocalPoint;
    const Vec3f ctDirAuxy    = transformH(rt2ct, raster + Vec3f(0.0f, 1.0f, 0.0f)) - mFocalPoint;

    const Vec3f dirPrimaryNormalized = normalize(ctDirPrimary);
    const float tNear = getClipPlaneT(getNear(), mFocalPoint,
                                     dirPrimaryNormalized);
    const float tFar  = getClipPlaneT(getFar(), mFocalPoint,
                                     dirPrimaryNormalized);

    *dstRay = mcrt_common::RayDifferential(
        renderStart, normalize(transformVector(ct2render, ctDirPrimary)),
        renderStart, normalize(transformVector(ct2render, ctDirAuxX)),
        renderStart, normalize(transformVector(ct2render, ctDirAuxy)),
        tNear, tFar, time, 0);
}

PerspectiveCamera::PerspectiveCamera(const rdl2::Camera* rdlCamera) :
    ProjectiveCamera(rdlCamera)
{
    initAttributeKeys(rdlCamera->getSceneClass());
}

void
PerspectiveCamera::updateImpl(const Mat4d &world2render)
{
    ProjectiveCamera::updateImpl(world2render);

    // Compute camera focal point in camera space
    // See diagram pbr/doc/stereo_projection_top_view.jpg for details.
    mFocalPoint = Vec3f(zero);
    const StereoView stereoView = getStereoViewImpl();
    if (stereoView != StereoView::CENTER) {
        mFocalPoint.x = -computeInterocularOffset(stereoView,
                                                  getRdlCamera()->get(
                                                      sStereoInterocularDistance));
    }
}

// Compute projection matrix
Mat4f
PerspectiveCamera::computeC2S(float t) const
{
    const auto cam = getRdlCamera();

    // Compute additional horizontal film offset and translation due to stereo
    // See diagram pbr/doc/stereo_projection_top_view.jpg for details.
    float interocularOffset = 0.0f;
    const StereoView stereoView = getStereoViewImpl();
    if (stereoView != StereoView::CENTER) {
        // Get stereo properties
        interocularOffset = computeInterocularOffset(stereoView,
                                                     cam->get(sStereoInterocularDistance));
    }

    return cam->computeProjectionMatrix(t, getWindow(), interocularOffset);
}

// Compute projection matrix
Mat4f
PerspectiveCamera::computeRegionC2S(float t) const
{
    const auto cam = getRdlCamera();

    // Compute additional horizontal film offset and translation due to stereo
    // See diagram pbr/doc/stereo_projection_top_view.jpg for details.
    float interocularOffset = 0.0f;
    const StereoView stereoView = getStereoViewImpl();
    if (stereoView != StereoView::CENTER) {
        // Get stereo properties
        interocularOffset = computeInterocularOffset(stereoView,
                                                     cam->get(sStereoInterocularDistance));
    }

    return cam->computeProjectionMatrix(t, getRenderRegion(), interocularOffset);
}

float
PerspectiveCamera::getFocalDistance() const
{
    return getRdlCamera()->get(sFocalKey);
}

void
PerspectiveCamera::computeFrustumImpl(mcrt_common::Frustum *f, float t,
                                      bool useRenderRegion) const
{
    // C2S is an OpenGL-style projection matrix so we can just extract the
    //  clip plane coefficients (Ax+By+Cz+D=0) by adding/subtracting matrix columns.
    //
    // Derivation: Let v = [x y z w=1] a point in camera space,
    //               C2S = 4x4 column-major projection matrix with 4 columns col0..3
    // Then v' = v * C2S = [v dot col0, v dot col1, v dot col2, v dot col3]
    //                   = [x', y', z', w']
    // For v' to be in the frustum:
    //      -w' < x' < w'
    //      -w' < y' < w'
    //      -w' < z' < w'
    // To test x' against left clipping plane, use the inequality:
    //      -w' < x'
    //      -(v dot col3) < (v dot col0)
    //      0 < (v dot col0) + (v dot col3)
    //      0 < v dot (col0 + col3)
    //      0 < x * (col0.x + col3.x) + y * (col0.y + col3.y) + z * (col0.z + col3.z) + w * (col0.w + col3.w)
    //      ... but w=1:
    //      0 < x * (col0.x + col3.x) + y * (col0.y + col3.y) + z * (col0.z + col3.z) + (col0.w + col3.w)
    //      ... this is a plane equation of the form:
    //      0 < x * A + y * B + c * Z + D
    //      ... just add col0 and col3 to get the ABCD coefficients
    //      ... similar derivation for the other planes

    if (useRenderRegion) {
        f->mC2S = computeRegionC2S(t);
        f->mViewport[0] = (int)getRegionToApertureOffsetX();
        f->mViewport[1] = (int)getRegionToApertureOffsetY();
        f->mViewport[2] = f->mViewport[0] + (int)getRegionWindowWidth();
        f->mViewport[3] = f->mViewport[1] + (int)getRegionWindowHeight();
    } else {
        f->mC2S = computeC2S(t);
        f->mViewport[0] = 0;
        f->mViewport[1] = 0;
        f->mViewport[2] = (int)getApertureWindowWidth();
        f->mViewport[3] = (int)getApertureWindowHeight();
    }

    // right: col 3 - col 0
    f->mClipPlanes[0][0] = f->mC2S[0][3] - f->mC2S[0][0];
    f->mClipPlanes[0][1] = f->mC2S[1][3] - f->mC2S[1][0];
    f->mClipPlanes[0][2] = f->mC2S[2][3] - f->mC2S[2][0];
    f->mClipPlanes[0][3] = f->mC2S[3][3] - f->mC2S[3][0];
    // left: col 3 + col 0
    f->mClipPlanes[1][0] = f->mC2S[0][3] + f->mC2S[0][0];
    f->mClipPlanes[1][1] = f->mC2S[1][3] + f->mC2S[1][0];
    f->mClipPlanes[1][2] = f->mC2S[2][3] + f->mC2S[2][0];
    f->mClipPlanes[1][3] = f->mC2S[3][3] + f->mC2S[3][0];
    // bottom: col 3 + col 1
    f->mClipPlanes[2][0] = f->mC2S[0][3] + f->mC2S[0][1];
    f->mClipPlanes[2][1] = f->mC2S[1][3] + f->mC2S[1][1];
    f->mClipPlanes[2][2] = f->mC2S[2][3] + f->mC2S[2][1];
    f->mClipPlanes[2][3] = f->mC2S[3][3] + f->mC2S[3][1];
    // top: col 3 - col 1
    f->mClipPlanes[3][0] = f->mC2S[0][3] - f->mC2S[0][1];
    f->mClipPlanes[3][1] = f->mC2S[1][3] - f->mC2S[1][1];
    f->mClipPlanes[3][2] = f->mC2S[2][3] - f->mC2S[2][1];
    f->mClipPlanes[3][3] = f->mC2S[3][3] - f->mC2S[3][1];
    // far: col3 - col 2
    f->mClipPlanes[4][0] = f->mC2S[0][3] - f->mC2S[0][2];
    f->mClipPlanes[4][1] = f->mC2S[1][3] - f->mC2S[1][2];
    f->mClipPlanes[4][2] = f->mC2S[2][3] - f->mC2S[2][2];
    f->mClipPlanes[4][3] = f->mC2S[3][3] - f->mC2S[3][2];
    // near: col3 + col2
    f->mClipPlanes[5][0] = f->mC2S[0][3] + f->mC2S[0][2];
    f->mClipPlanes[5][1] = f->mC2S[1][3] + f->mC2S[1][2];
    f->mClipPlanes[5][2] = f->mC2S[2][3] + f->mC2S[2][2];
    f->mClipPlanes[5][3] = f->mC2S[3][3] + f->mC2S[3][2];
}

StereoView PerspectiveCamera::getStereoViewImpl() const
{
    return static_cast<StereoView>(getRdlCamera()->get(sStereoView));
}

} // namespace pbr
} // namespace moonray


