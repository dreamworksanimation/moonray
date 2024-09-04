// Copyright 2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "Camera.h"

namespace moonray {
namespace pbr {

class FisheyeCamera : public Camera
{
public:
    /// Constructor
    explicit FisheyeCamera(const scene_rdl2::rdl2::Camera* rdlCamera);

    scene_rdl2::math::Vec3f createDirection(float x, float y) const;
    bool testBBoxOverlaps(const scene_rdl2::math::BBox3f& bbox) const;
    float screenSpaceDerivative(const scene_rdl2::math::Vec3f &v) const;

private:
    // These need to match up with the enums declared in dso/camera/FisheyeCamera/attributes.cc
    enum Mapping {
        MAPPING_STEREOGRAPHIC,
        MAPPING_EQUIDISTANT,
        MAPPING_EQUISOLID_ANGLE,
        MAPPING_ORTHOGRAPHIC,
    };
    enum Format {
        FORMAT_CIRCULAR,
        FORMAT_CROPPED,
        FORMAT_DIAGONAL,
    };

    void initAttributeKeys(const scene_rdl2::rdl2::SceneClass& sceneClass);

    bool getIsDofEnabledImpl() const override;

    void updateImpl(const scene_rdl2::math::Mat4d& world2render) override;

    void createRayImpl(mcrt_common::RayDifferential* dstRay,
                       float x,
                       float y,
                       float time,
                       float lensU,
                       float lensV) const override;

    scene_rdl2::math::Vec2f projectPoint(const scene_rdl2::math::Vec3f &v) const;
    bool isInView(const scene_rdl2::math::Vec3f &v) const;

    static bool sAttributeKeyInitialized;
    static scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Int>   sMappingKey;
    static scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Int>   sFormatKey;
    static scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Float> sZoomKey;

    int mMapping;
    float mRadialScale;
    float mDerivScale;
};

} // namespace pbr
} // namespace moonray


