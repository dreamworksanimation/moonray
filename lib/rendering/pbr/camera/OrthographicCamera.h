// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ProjectiveCamera.h"

#include <scene_rdl2/common/math/Mat4.h>
#include <scene_rdl2/common/math/Vec3.h>
#include <scene_rdl2/scene/rdl2/AttributeKey.h>

namespace scene_rdl2 {
namespace rdl2 {
class SceneClass;
}
}
namespace moonray {
namespace pbr {

class ProjectiveCamera;

///
/// @class OrthographicCamera Camera.h <pbr/Camera.h>
/// @brief Gets camera properties from an scene_rdl2::rdl2::Camera, and informs other libs
/// about them as needed.
///
class OrthographicCamera : public ProjectiveCamera
{
public:
    /// Constructor
    explicit OrthographicCamera(const scene_rdl2::rdl2::Camera* rdlCamera);

private:
    void updateImpl(const scene_rdl2::math::Mat4d &world2render) override;
    void createDOFRay(mcrt_common::RayDifferential* dstRay,
                      const scene_rdl2::math::Vec3f& Pr,
                      float lensX,
                      float lensY,
                      float time) const override;

    void createSimpleRay(mcrt_common::RayDifferential* dstRay,
                         const scene_rdl2::math::Vec3f& Pr,
                         float time) const override;

    scene_rdl2::math::Mat4f computeC2S(float t) const override;
    float getFocalDistance() const override;
};

} // namespace pbr
} // namespace moonray


