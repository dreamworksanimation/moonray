// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0


#pragma once

#include "Light.h"

#include <scene_rdl2/common/math/BBox.h>
#include <scene_rdl2/common/math/Color.h>
#include <scene_rdl2/common/math/Mat4.h>
#include <scene_rdl2/common/math/Vec3.h>

// Forward declaration of the ISPC types
namespace ispc {
    struct DiskLight;
}


namespace scene_rdl2 {
namespace rdl2 {
    class Light;
}
}

namespace moonray {
namespace pbr {

//----------------------------------------------------------------------------

/// @brief Implements light sampling for disk lights.

/// A disk light is internally implement by modeling a unit circle on the
/// xy plane in local space. It emits light along the +z axis in local space.

class DiskLight : public LocalParamLight
{
    friend class DiskLightTester;

public:
    /// Constructor / Destructor
    explicit DiskLight(const scene_rdl2::rdl2::Light* rdlLight);
    virtual ~DiskLight();

    /// HUD validation and type casting
    static uint32_t hudValidation(bool verbose) {
        DISK_LIGHT_VALIDATION;
    }
    HUD_AS_ISPC_METHODS(DiskLight);


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

private:
    void initAttributeKeys(const scene_rdl2::rdl2::SceneClass &sc);

    /// Copy is disabled
    DiskLight(const DiskLight &other);
    const DiskLight &operator=(const DiskLight &other);

    // This function computes the perpendicular distance from the plane of the DiskLight to the point p.
    // The distance will be positive on the lit side of a DiskLight with regular or reverse sideness,
    // or the absolute value of the distance for a 2-sided Disklight.
    float planeDistance(const scene_rdl2::math::Vec3f &p, float time) const;
    // Inputs: render space shading point and distance to light.
    // Outputs: center and half side length of valid square region of light influence if spread < 1.
    void getSpreadSquare(const scene_rdl2::math::Vec3f& renderP, const float renderDistance, const float time,
            scene_rdl2::math::Vec3f& center, float& halfSideLength) const;

    DISK_LIGHT_MEMBERS;

    //
    // Cached attribute keys:
    //
    // cppcheck-suppress duplInheritedMember
    static bool                             sAttributeKeyInitialized;
    static scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Bool>   sNormalizedKey;
    static scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Bool>   sApplySceneScaleKey;
    static scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Float>  sRadiusKey;
    static scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Float>  sSpreadKey;
    static scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Int>    sSidednessKey;
    DECLARE_ATTR_SKEYS_CLEAR_RADIUS
};

//----------------------------------------------------------------------------

} // namespace pbr
} // namespace moonray

