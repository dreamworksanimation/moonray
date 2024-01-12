// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "StereoView.h"

#include <moonray/rendering/bvh/shading/AttributeKey.h>
#include <moonray/rendering/mcrt_common/Frustum.h>
#include <moonray/rendering/mcrt_common/Ray.h>

#include <scene_rdl2/common/math/Mat4.h>
#include <scene_rdl2/common/math/Quaternion.h>
#include <scene_rdl2/common/math/Vec3.h>
#include <scene_rdl2/scene/rdl2/Camera.h>

#include <array>

namespace scene_rdl2 {
namespace rdl2 {
    class Camera;
}
}

namespace moonray {

namespace pbr {

class Camera
{
public:
    explicit Camera(const scene_rdl2::rdl2::Camera* rdlCamera) :
        mRdlCamera(rdlCamera),
        mApertureWindowWidth(0.f),
        mApertureWindowHeight(0.f),
        mRegionWindowWidth(0.f),
        mRegionWindowHeight(0.f),
        mRegionToApertureOffsetX(0.f),
        mRegionToApertureOffsetY(0.f)
    {
    }

    Camera(const Camera &other) = delete;
    const Camera &operator=(const Camera &other) = delete;

    virtual ~Camera() = default;

    /// Get to the scene_rdl2::rdl2::Camera
    const scene_rdl2::rdl2::Camera* getRdlCamera() const {  return mRdlCamera;  }

    /// If true, lights need to be updated
    bool xformChanged() const;

    float getApertureWindowWidth() const        { return mApertureWindowWidth; }
    float getApertureWindowHeight() const       { return mApertureWindowHeight; }
    float getRegionWindowWidth() const          { return mRegionWindowWidth; }
    float getRegionWindowHeight() const         { return mRegionWindowHeight; }
    float getRegionToApertureOffsetX() const    { return mRegionToApertureOffsetX; }
    float getRegionToApertureOffsetY() const    { return mRegionToApertureOffsetY; }

    /// Returns the transforms to go between world space, render space and
    /// camera-space at shutter-open (t=0) (where t is expressed in ray time;
    /// see createRayImpl() below).
    /// Whether we render from the center/left/right stereo view, camera space
    /// is always the same and defined by the center camera (as when stereo is
    /// disabled). The camera view translation due to the interocular distance
    /// is part of the camera projection transform (C2S).
    const scene_rdl2::math::Mat4d &getCamera2World() const { return mCamera2World; }
    const scene_rdl2::math::Mat4d &getWorld2Camera() const { return mWorld2Camera; }
    const scene_rdl2::math::Mat4f &getCamera2Render() const { return mCamera2Render; }
    const scene_rdl2::math::Mat4f &getRender2Camera() const { return mRender2Camera; }

    scene_rdl2::math::Mat4f computeCamera2Render(float time) const;
    scene_rdl2::math::Mat4f computeRender2Camera(float time) const;

    /// Update all camera properties from the scene_rdl2::rdl2::Camera and given World to
    /// Render transform.
    void update(const scene_rdl2::math::Mat4d& world2render);

    /// Create a ray given (x, y) coordinates in region space.
    void createRay(mcrt_common::RayDifferential* dstRay,
                           float x,
                           float y,
                           float time,
                           float lensU,
                           float lensV,
                           bool createDifferentials) const
    {


        createRayImpl(dstRay, x + mRegionToApertureOffsetX,
                      y + mRegionToApertureOffsetY,
                      time, lensU, lensV);

        // turn off ray differentials if unwanted (uncommon case)
        dstRay->mFlags.set(mcrt_common::RayDifferential::HAS_DIFFERENTIALS, createDifferentials);
    }

    bool getIsDofEnabled() const {  return getIsDofEnabledImpl(); }

    float getNear() const { return mRdlCamera->get(scene_rdl2::rdl2::Camera::sNearKey); }
    float getFar() const { return mRdlCamera->get(scene_rdl2::rdl2::Camera::sFarKey); }

    bool hasFrustum() const { return hasFrustumImpl(); }
    void computeFrustum(mcrt_common::Frustum *frust, float t, bool useRenderRegion) const
    {
        return computeFrustumImpl(frust, t, useRenderRegion);
    }

    void bakeUvMaps() { bakeUvMapsImpl(); }
    void getRequiredPrimAttributes(shading::PerGeometryAttributeKeySet &keys) const
    {
        getRequiredPrimAttributesImpl(keys);
    }

    float getShutterBias() const { return mRdlCamera->get( scene_rdl2::rdl2::Camera::sMbShutterBiasKey ); }

    StereoView getStereoView() const
    {
        return getStereoViewImpl();
    }

    bool getMotionBlur() const;

    /// Given a point p in render space, compute its z-distance
    /// from the camera at ray time t.  For projective cameras, this is
    /// the value typically computed in "depth" maps.  For non-projective
    /// cameras the function returns the euclidean distance between p
    /// and the render space ray origin o.  The ray origin is needed to
    /// support cameras that do no have a constant location
    /// (such as the BakeCamera).
    float computeZDistance(const scene_rdl2::math::Vec3f &p, const scene_rdl2::math::Vec3f &o, float time) const;

protected:
    /// Compute W <--> C matrices at time t
    scene_rdl2::math::Mat4d computeC2W(float t) const;

private:
    virtual bool getIsDofEnabledImpl() const = 0;
    virtual bool hasFrustumImpl() const { return false; }
    virtual void computeFrustumImpl(mcrt_common::Frustum *frust, float t, bool useRenderRegion) const;
    virtual void bakeUvMapsImpl();
    virtual void getRequiredPrimAttributesImpl(shading::PerGeometryAttributeKeySet &keys) const;
    virtual float computeZDistanceImpl(const scene_rdl2::math::Vec3f &p, const scene_rdl2::math::Vec3f &o,
                                       float time) const;
    virtual void updateImpl(const scene_rdl2::math::Mat4d& world2render) = 0;

    virtual void createRayImpl(mcrt_common::RayDifferential* dstRay,
                           float x,
                           float y,
                           float time,
                           float lensU,
                           float lensV) const = 0;

    virtual StereoView getStereoViewImpl() const { return StereoView::CENTER; }

    const scene_rdl2::rdl2::Camera* mRdlCamera;

    // The details of the global scene aperture window and region window are
    // accounted for in these quantities.
    float mApertureWindowWidth;
    float mApertureWindowHeight;
    float mRegionWindowWidth;
    float mRegionWindowHeight;
    float mRegionToApertureOffsetX;
    float mRegionToApertureOffsetY;

    // Transforms at shutter-open (t=0)
    scene_rdl2::math::Mat4d mCamera2World;
    scene_rdl2::math::Mat4d mWorld2Camera;
    scene_rdl2::math::Mat4f mCamera2Render;
    scene_rdl2::math::Mat4f mRender2Camera;

    // Camera-space at shutter-close (t=1) to camera-space at shutter-open (t=0)
    scene_rdl2::math::Quaternion3f  mCameraClose2CameraOpenRot;
    scene_rdl2::math::Vec3f         mCameraClose2CameraOpenTrans;
    scene_rdl2::math::Quaternion3f  mCameraOpen2CameraCloseRot;
    scene_rdl2::math::Vec3f         mCameraOpen2CameraCloseTrans;
    scene_rdl2::math::Mat4f         mCameraClose2CameraOpen;
};

} // namespace pbr
} // namespace moonray


