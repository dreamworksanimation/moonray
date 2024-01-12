// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "Camera.h"

#include <scene_rdl2/common/math/Mat4.h>
#include <scene_rdl2/common/math/Vec3.h>

namespace moonray {
namespace pbr {

class SphericalCamera : public Camera
{
public:
    /// Constructor
    explicit SphericalCamera(const scene_rdl2::rdl2::Camera* rdlCamera);

private:
    bool getIsDofEnabledImpl() const override;

    void updateImpl(const scene_rdl2::math::Mat4d& world2render) override;

    void createRayImpl(mcrt_common::RayDifferential* dstRay,
                       float x,
                       float y,
                       float time,
                       float lensU,
                       float lensV) const override;

    scene_rdl2::math::Vec3f createDirection(float x, float y) const;

};

} // namespace pbr
} // namespace moonray


