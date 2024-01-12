// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

#include "Camera.h"

#include <moonray/rendering/pbr/core/Util.h>

#include <scene_rdl2/common/except/exceptions.h>
#include <scene_rdl2/scene/rdl2/rdl2.h>

namespace moonray {
namespace pbr {

using namespace scene_rdl2::math;

bool
Camera::xformChanged() const
{
    return getRdlCamera()->hasChanged(scene_rdl2::rdl2::Node::sNodeXformKey);
}

/// Update all camera properties from the scene_rdl2::rdl2::Camera and given World to
/// Render transform.
void
Camera::update(const Mat4d& world2render)
{
    float near = getNear();
    float far = getFar();
    if (near < 0.f || far <= near) {
        mRdlCamera->error("Camera has invalid near/far values.  ",
            "near: " , near , " far: " , far);
    }

    // Recompute world and trace transforms
    mCamera2World = computeC2W(0.0f);
    bool isRotTrans;
    mWorld2Camera = rtInverse(mCamera2World, isRotTrans);
    if (!isRotTrans) {
        mRdlCamera->warn("Camera has scale or shear, results may be non-deterministic\n");
    }
    mCamera2Render = toFloat(mCamera2World * world2render);
    mRender2Camera = rtInverse(mCamera2Render);

    // Compute camera space transforms between shutter close
    // and shutter open
    Mat4d cameraClose2World = computeC2W(1.0f);
    const Mat4f cameraClose2CameraOpen = toFloat(cameraClose2World * getWorld2Camera());
    mCameraClose2CameraOpen = cameraClose2CameraOpen;
    const Mat4f cameraOpen2CameraClose = toFloat((cameraClose2World * getWorld2Camera()).inverse());
    // rotations
    mCameraClose2CameraOpenRot = Quaternion3f(asVec3(cameraClose2CameraOpen.vx),
                                              asVec3(cameraClose2CameraOpen.vy),
                                              asVec3(cameraClose2CameraOpen.vz));
    // This quaternion could be computed as <r, -i, -j, -k> of
    // the previous one.  But for robustness reasons, compute it directly
    // from the source matrix.
    mCameraOpen2CameraCloseRot = Quaternion3f(asVec3(cameraOpen2CameraClose.vx),
                                              asVec3(cameraOpen2CameraClose.vy),
                                              asVec3(cameraOpen2CameraClose.vz));
    // When slerping two generic quaternions, a and b, this check is "if (dot(a, b) < 0.f)".
    // But since we are slerping these quaternions always and only with <1, 0, 0, 0>,
    // checking against the .r component is sufficient.  Both of these conditionals
    // should always be either both true or both false, but it is conceivably possible
    // that rounding errors could cause this to not be the case.
    if (mCameraClose2CameraOpenRot.r < 0.f) mCameraClose2CameraOpenRot *= -1.0f;
    if (mCameraOpen2CameraCloseRot.r < 0.f) mCameraOpen2CameraCloseRot *= -1.0f;
    // translations
    mCameraClose2CameraOpenTrans = asVec3(cameraClose2CameraOpen.vw);
    mCameraOpen2CameraCloseTrans = asVec3(cameraOpen2CameraClose.vw);

    const scene_rdl2::rdl2::SceneVariables& vars =
        getRdlCamera()->getSceneClass().getSceneContext()->getSceneVariables();

    scene_rdl2::math::HalfOpenViewport aperture = vars.getRezedApertureWindow();
    scene_rdl2::math::HalfOpenViewport region = vars.getRezedRegionWindow();

    mApertureWindowWidth = aperture.width();
    mApertureWindowHeight = aperture.height();
    mRegionWindowWidth = region.width();
    mRegionWindowHeight = region.height();
    mRegionToApertureOffsetX = region.mMinX - aperture.mMinX;
    mRegionToApertureOffsetY = region.mMinY - aperture.mMinY;

    updateImpl(world2render);
}

Mat4f
Camera::computeCamera2Render(float time) const
{
    // "camera space at ray time" --> render space
    if (getMotionBlur()) {
        // slerp rotation, lerp translation to compute camera(time) to camera(0) matrix
        const Quaternion3f r = slerp(Quaternion3f(scene_rdl2::math::one), mCameraClose2CameraOpenRot, time);
        const Vec3f t = mCameraClose2CameraOpenTrans * time;
        const Mat4f cameraTime2CameraOpen = Mat4f(r, Vec4f(t.x, t.y, t.z, 1.f));

        // Then apply camera to render transform
        return cameraTime2CameraOpen * getCamera2Render();
    } else {
        return getCamera2Render();
    }
}

Mat4f
Camera::computeRender2Camera(float time) const
{
    // render space -> "camera space at ray time"
    if (getMotionBlur()) {
        // slerp rotation, lerp translation to compute camera(0) to camera(time) matrix
        const Quaternion3f r = slerp(Quaternion3f(scene_rdl2::math::one), mCameraOpen2CameraCloseRot, time);
        const Vec3f t = mCameraOpen2CameraCloseTrans * time;
        const Mat4f cameraOpen2CameraTime = Mat4f(r, Vec4f(t.x, t.y, t.z, 1.f));

        // Then apply camera to render transform
        return getRender2Camera() * cameraOpen2CameraTime;
    } else {
        return getRender2Camera();
    }
}

Mat4d
Camera::computeC2W(float t) const
{
    return getRdlCamera()->get(scene_rdl2::rdl2::Node::sNodeXformKey, t);
}

void
Camera::computeFrustumImpl(mcrt_common::Frustum *frust, float t, bool useRenderRegion) const
{
    throw scene_rdl2::except::NotImplementedError("No frustum implemented for this camera type.");
}

void
Camera::bakeUvMapsImpl()
{
    // empty
}

void
Camera::getRequiredPrimAttributesImpl(shading::PerGeometryAttributeKeySet & /*keys*/) const
{
    // empty
}

bool
Camera::getMotionBlur() const
{
    return mRdlCamera->getSceneClass().getSceneContext()->getSceneVariables().get(
        scene_rdl2::rdl2::SceneVariables::sEnableMotionBlur);
}

float
Camera::computeZDistance(const Vec3f &p, const Vec3f &o, float time) const
{
    return computeZDistanceImpl(p, o, time);
}

float
Camera::computeZDistanceImpl(const Vec3f &p, const Vec3f &o, float time) const
{
    // The default implemenation computes the euclidean distance between
    // the ray hit point p, and the ray origin o.  This should be a good
    // default for all but projective cameras.  It assumes that the camera
    // space has no scales, which ensures that distance in render space is equal
    // to distance in all camera spaces.  So the time parameter is ignored.
    return (p - o).length();
}

} // namespace pbr
} // namespace moonray

