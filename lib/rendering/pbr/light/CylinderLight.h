// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0


#pragma once
#include "Light.h"

#include <scene_rdl2/common/math/BBox.h>
#include <scene_rdl2/common/math/Color.h>
#include <scene_rdl2/common/math/Mat4.h>
#include <scene_rdl2/common/math/Vec2.h>
#include <scene_rdl2/common/math/Vec3.h>

// Forward declaration of the ISPC types
namespace ispc {
    struct CylinderLight;
}


namespace scene_rdl2 {
namespace rdl2 {
    class Light;
}
}

namespace moonray {
namespace pbr {

//----------------------------------------------------------------------------

/// @brief Implements light sampling for cylinder lights.

/// A cylinder area light is internally implement by modeling a circle of given
/// radius on the xz plane in local space and the height of the cylinder spans
/// the y-axis over the range [0.5*height..-0.5*height].
/// The local uv parameterization has u spanning along the height of the
/// cylinder (u=[0..1] maps to y=[0.5*height..-0.5*height]), and v around
/// its circumference.

class CylinderLight : public LocalParamLight
{
    friend class CylinderLightTester;

public:
    /// Constructor / Destructor
    explicit CylinderLight(const scene_rdl2::rdl2::Light* rdlLight);
    virtual ~CylinderLight();

    /// HUD validation and type casting
    static uint32_t hudValidation(bool verbose) {
        CYLINDER_LIGHT_VALIDATION;
    }
    HUD_AS_ISPC_METHODS(CylinderLight);


    virtual bool update(const scene_rdl2::math::Mat4d& world2render) override;

    /// Intersection and sampling API
    // TODO: Can we cull cylinder lights ?
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

    float getThetaO() const override { return scene_rdl2::math::sPi; }
    float getThetaE() const override { return scene_rdl2::math::sPi * 0.5f; }

private:
    void initAttributeKeys(const scene_rdl2::rdl2::SceneClass &sc);

    /// Copy is disabled
    CylinderLight(const CylinderLight &other);
    const CylinderLight &operator=(const CylinderLight &other);

    // Compute local space position from uv cylinder parameterization.
    finline scene_rdl2::math::Vec3f uv2local(const scene_rdl2::math::Vec2f &uv) const
    {
        float phi = uv.y * scene_rdl2::math::sTwoPi;
        float sinPhi, cosPhi;
        scene_rdl2::math::sincos(phi, &sinPhi, &cosPhi);

        return scene_rdl2::math::Vec3f(mLocalRadius * cosPhi,
                     (uv.x - mUvOffset.x) * scene_rdl2::math::rcp(mUvScale.x),
                     mLocalRadius * sinPhi);
    }

    // Compute uv cylinder parameterization from local space position.
    finline scene_rdl2::math::Vec2f local2uv(const scene_rdl2::math::Vec3f &pos) const
    {
        float u = pos.y * mUvScale.x + mUvOffset.x;

        float phi = scene_rdl2::math::atan2(pos.z, pos.x);
        float v = (phi < 0.0f  ?  phi + scene_rdl2::math::sTwoPi  :  phi) * scene_rdl2::math::sOneOverTwoPi;

        MNRY_ASSERT(finite(u));
        MNRY_ASSERT(finite(v));

        return scene_rdl2::math::Vec2f(u, v);
    }

    CYLINDER_LIGHT_MEMBERS;

    //
    // Cached attribute keys:
    //
    // cppcheck-suppress duplInheritedMember
    static bool                                                     sAttributeKeyInitialized;
    static scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Bool>   sNormalizedKey;
    static scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Bool>   sApplySceneScaleKey;
    static scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Float>  sRadiusKey;
    static scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Float>  sHeightKey;
    static scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Int>    sSidednessKey;
    DECLARE_ATTR_SKEYS_CLEAR_RADIUS
};

//----------------------------------------------------------------------------

} // namespace pbr
} // namespace moonray

