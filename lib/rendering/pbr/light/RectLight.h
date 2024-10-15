// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0


#pragma once
#include "Light.h"

#include <scene_rdl2/common/math/BBox.h>
#include <scene_rdl2/common/math/Mat4.h>
#include <scene_rdl2/common/math/Vec3.h>

// Forward declaration of the ISPC types
namespace ispc {
    struct RectLight;
}


namespace scene_rdl2 {
namespace rdl2 {
    class Light;
}
}

namespace moonray {
namespace pbr {

//----------------------------------------------------------------------------

/// @brief Implements light sampling for rectangular lights.

/// A rectangular area light is internally implement by modeling a rectangle on the
/// xy plane centered at the origin in local space, covering the region
/// [-mLocalWidth/2, mLocalWidth/2] x [-mLocalHeight/2, mLocalHeight/2], although note
/// that in render space these dimensions may also be scaled by the mLocal2Render matrix.
/// It emits light along the +z axis in local space.

class RectLight : public LocalParamLight
{
    friend class RectLightTester;

public:
    /// Constructor / Destructor
    explicit RectLight(const scene_rdl2::rdl2::Light* rdlLight, bool uniformSampling = false);
    virtual ~RectLight();

    /// HUD validation and type casting
    static uint32_t hudValidation(bool verbose) {
        RECT_LIGHT_VALIDATION;
    }
    HUD_AS_ISPC_METHODS(RectLight);


    virtual bool update(const scene_rdl2::math::Mat4d& world2render) override;

    /// Intersection and sampling API
    virtual bool canIlluminate(const scene_rdl2::math::Vec3f p, const scene_rdl2::math::Vec3f *n, float time, float radius,
            const LightFilterList* lightFilterList) const override;
    virtual bool isBounded() const override;
    virtual bool isDistant() const override;
    virtual bool isEnv() const override;
    virtual scene_rdl2::math::BBox3f getBounds() const override;
    virtual bool intersect(const scene_rdl2::math::Vec3f &p, const scene_rdl2::math::Vec3f *n, const scene_rdl2::math::Vec3f &wi, float time,
            float maxDistance, LightIntersection &isect) const override;
    virtual bool sample(const scene_rdl2::math::Vec3f &p, const scene_rdl2::math::Vec3f *n, float time, const scene_rdl2::math::Vec3f& r,
            scene_rdl2::math::Vec3f &wi, LightIntersection &isect, float rayDirFootprint) const override;
    virtual scene_rdl2::math::Color eval(mcrt_common::ThreadLocalState* tls, const scene_rdl2::math::Vec3f &wi, const scene_rdl2::math::Vec3f &p,
            const LightFilterRandomValues& filterR, float time, const LightIntersection &isect, bool fromCamera,
            const LightFilterList *lightFilterList, float rayDirFootprint, float *pdf = nullptr) const override;
    virtual scene_rdl2::math::Vec3f getEquiAngularPivot(const scene_rdl2::math::Vec3f& r, float time) const override;

    float getThetaO() const override { return scene_rdl2::math::sPi * 0.5f * mSpread; }
    float getThetaE() const override { return scene_rdl2::math::sPi * 0.5f; }

protected:
    bool updateTransforms(const scene_rdl2::math::Mat4f &local2Render, int ti) override;

private:
    void initAttributeKeys(const scene_rdl2::rdl2::SceneClass &sc);

    void computeCorners(scene_rdl2::math::Vec3f *corners, float time) const;
    // This function computes the perpendicular distance from the plane of the RectLight to the point p.
    // The distance will be positive on the lit side of a RectLight with regular or reverse sideness,
    // or the absolute value of the distance for a 2-sided Rectlight.
    float planeDistance(const scene_rdl2::math::Vec3f &p, float time) const;
    // Get region of overlap between RectLight and square region of influence.
    // The square is centered at localP and has a side length 2 * localLength. 
    // We want to get the center of the rectangular bound and its width and height.
    // Returns true if the region overlaps with the light.
    bool getOverlapBounds(const scene_rdl2::math::Vec2f& localP, float localLength, 
            scene_rdl2::math::Vec2f& center, float& width, float& height) const;

    /// Copy is disabled
    RectLight(const RectLight &other);
    const RectLight &operator=(const RectLight &other);

    RECT_LIGHT_MEMBERS;

    //
    // Cached attribute keys:
    //
    // cppcheck-suppress duplInheritedMember
    static bool                             sAttributeKeyInitialized;
    static scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Bool>   sNormalizedKey;
    static scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Bool>   sApplySceneScaleKey;
    static scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Float>  sWidthKey;
    static scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Float>  sHeightKey;
    static scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Float>  sSpreadKey;
    static scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Int>    sSidednessKey;
    DECLARE_ATTR_SKEYS_CLEAR_RADIUS
};

//----------------------------------------------------------------------------

} // namespace pbr
} // namespace moonray

