// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "Camera.h"

#include "LensDistribution.h"
#include <moonray/rendering/pbr/core/Distribution.h>

#include <scene_rdl2/common/math/Mat4.h>
#include <scene_rdl2/common/math/Vec3.h>
#include <scene_rdl2/scene/rdl2/AttributeKey.h>
#include <scene_rdl2/scene/rdl2/Camera.h>
#include <array>

namespace scene_rdl2 {
namespace rdl2 {
class SceneClass;
}
}
namespace moonray {
namespace pbr {

class ProjectiveCamera : public Camera
{
public:
    explicit ProjectiveCamera(const scene_rdl2::rdl2::Camera* rdlCamera);

    scene_rdl2::math::Mat4f computeCamera2Screen(float time) const;

protected:
    static scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Bool>  sDofKey;
    static scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Float> sDofApertureKey;
    static scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Float> sDofFocusDistance;

    static scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Float> sHorizontalFilmOffset;
    static scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Float> sVerticalFilmOffset;

    /// Update all camera properties from the scene_rdl2::rdl2::Camera and given World to
    /// Render transform.
    void updateImpl(const scene_rdl2::math::Mat4d &world2render) override;

    float getDofFocusDistance() const { return mDofFocusDistance; }
    std::array<float, 4> getWindow() const { return mWindow; };
    std::array<float, 4> getRenderRegion() const { return mRenderRegion; };

    scene_rdl2::math::Mat4f computeRaster2Camera(float time) const;
    scene_rdl2::math::Mat4f computeRaster2Render(float time, const scene_rdl2::math::Mat4f& ct2render) const;

private:
    void initAttributeKeys(const scene_rdl2::rdl2::SceneClass& sceneClass);

    bool getIsDofEnabledImpl() const override {  return mDof  &&  mDofLensRadius > 0.0f;  }
    float computeZDistanceImpl(const scene_rdl2::math::Vec3f &p, const scene_rdl2::math::Vec3f &o,
                               float time) const final;

    /// Create a primary ray *in render space*, given:
    /// - pixel in viewport coordinates)
    /// - subpixel in the range [0,subpixelRate)
    /// - subpixel offset within the pixel in the range [0,1)
    /// - time sample in the range [0,1) mapping to [shutterOpen,shutterClose)
    /// - disk lens sample in the unit-disc, mapping to the lens aperture
    void createRayImpl(mcrt_common::RayDifferential* dstRay,
                       float x,
                       float y,
                       float time,
                       float lensU,
                       float lensV) const final;

    virtual void createDOFRay(mcrt_common::RayDifferential* dstRay,
                              const scene_rdl2::math::Vec3f& Pr,
                              float lensX,
                              float lensY,
                              float time) const = 0;

    virtual void createSimpleRay(mcrt_common::RayDifferential* dstRay,
                                 const scene_rdl2::math::Vec3f& Pr,
                                 float time) const = 0;

    virtual float getFocalDistance() const = 0;

private:
    static bool sAttributeKeyInitialized;

    /// Compute C <--> S matrices at time t
    virtual scene_rdl2::math::Mat4f computeC2S(float t) const = 0;

    std::array<float, 4> mWindow;
    std::array<float, 4> mRenderRegion;

    // Raster-space to camera-space at shutter-open (t=0) and at shutter-close (t=1)
    scene_rdl2::math::Mat4f mRtoC0;
    scene_rdl2::math::Mat4f mRtoC1;

    // Camera-space to screen-space at shutter-open (t=0) and at shutter-close (t=1)
    scene_rdl2::math::Mat4f mC0toS;
    scene_rdl2::math::Mat4f mC1toS;
    
    // Depth-of-field settings
    bool mDof;
    float mDofLensRadius;
    float mDofFocusDistance;

    LensDistribution mLensDistribution;
};


} // namespace pbr
} // namespace moonray


