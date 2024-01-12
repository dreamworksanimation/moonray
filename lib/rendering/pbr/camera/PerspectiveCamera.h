// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ProjectiveCamera.h"

#include <scene_rdl2/common/math/Mat4.h>
#include <scene_rdl2/common/math/Vec3.h>
#include <scene_rdl2/scene/rdl2/AttributeKey.h>

namespace scene_rdl2 {
namespace rdl2 {
class ProjectiveCamera;
class SceneClass;
}
}
namespace moonray {
namespace pbr {

///
/// @class PerspectiveCamera Camera.h <pbr/Camera.h>
/// @brief Gets camera properties from an scene_rdl2::rdl2::Camera, and informs other libs
/// about them as needed.
///
class PerspectiveCamera : public ProjectiveCamera
{
public:
    /// Constructor
    explicit PerspectiveCamera(const scene_rdl2::rdl2::Camera* rdlCamera);


private:
    void initAttributeKeys(const scene_rdl2::rdl2::SceneClass& sceneClass);

    bool hasFrustumImpl() const override { return true; }
    void computeFrustumImpl(mcrt_common::Frustum *frust, float t,
                            bool useRenderRegion) const override;
    void updateImpl(const scene_rdl2::math::Mat4d &world2render) override;
    StereoView getStereoViewImpl() const override;

    void createDOFRay(mcrt_common::RayDifferential* dstRay,
                      const scene_rdl2::math::Vec3f& Pr,
                      float lensX,
                      float lensY,
                      float time) const override;

    void createSimpleRay(mcrt_common::RayDifferential* dstRay,
                         const scene_rdl2::math::Vec3f& Pr,
                         float time) const override;

    scene_rdl2::math::Mat4f computeC2S(float t) const override;
    scene_rdl2::math::Mat4f computeRegionC2S(float t) const;
    float getFocalDistance() const override;

    // Camera focal point in camera space (non-zero for stereo views L/R)
    scene_rdl2::math::Vec3f mFocalPoint;

    static bool                            sAttributeKeyInitialized;

    // Both focal length and film width should be in the same units. They
    // can both be in mm, even though the scene scale default is cm.
    static scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Float> sFocalKey;

    static scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Int>   sStereoView;
    static scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Float> sStereoInterocularDistance;
    static scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Float> sStereoConvergenceDistance;
};

} // namespace pbr
} // namespace moonray


