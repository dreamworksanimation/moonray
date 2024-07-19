// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "Camera.h"

namespace moonray {
namespace pbr {

class SphericalCamera : public Camera
{
public:
    /// Constructor
    explicit SphericalCamera(const scene_rdl2::rdl2::Camera* rdlCamera);

private:
    void initAttributeKeys(const scene_rdl2::rdl2::SceneClass& sceneClass);

    bool getIsDofEnabledImpl() const override;

    void updateImpl(const scene_rdl2::math::Mat4d& world2render) override;

    void createRayImpl(mcrt_common::RayDifferential* dstRay,
                       float x,
                       float y,
                       float time,
                       float lensU,
                       float lensV) const override;

    scene_rdl2::math::Vec3f createDirection(float x, float y) const;

    static bool sAttributeKeyInitialized;
    static scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Float> sMinLatitudeKey;
    static scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Float> sMaxLatitudeKey;
    static scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Float> sLatitudeZoomOffsetKey;
    static scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Float> sMinLongitudeKey;
    static scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Float> sMaxLongitudeKey;
    static scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Float> sLongitudeZoomOffsetKey;
    static scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Float> sFocalKey;
    static scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Bool>  sInsideOutKey;
    static scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Float> sOffsetRadiusKey;

    float mThetaScale;
    float mThetaOffset;
    float mPhiScale;
    float mPhiOffset;
    bool  mInsideOut;
    float mOffsetRadius;
};

} // namespace pbr
} // namespace moonray


