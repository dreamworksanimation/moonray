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
    struct SphereLight;
}


namespace scene_rdl2 {
namespace rdl2 {
    class Light;
}
}

namespace moonray {
namespace pbr {

//----------------------------------------------------------------------------

/// @brief Implements light sampling for Sphere lights.

/// A sphere light is internally implement by modeling a unit sphere around the
/// origin in local space. Light is emitted in all directions.

class SphereLight : public LocalParamLight
{
    friend class SphereLightTester;

public:
    /// Constructor / Destructor
    explicit SphereLight(const scene_rdl2::rdl2::Light* rdlLight, bool uniformSampling = false);
    virtual ~SphereLight();

    /// HUD validation and type casting
    static uint32_t hudValidation(bool verbose) {
        SPHERE_LIGHT_VALIDATION;
    }
    HUD_AS_ISPC_METHODS(SphereLight);


    virtual bool update(const scene_rdl2::math::Mat4d& world2render) override;

    /// Accessors to render space properties
    /// TODO: mb scale radiance
    finline float getRadius() const     {  return mLocal2RenderScale[0];  }
    finline float getInvRadius() const  {  return mRender2LocalScale[0];  }

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
    virtual scene_rdl2::math::Vec3f getEquiAngularPivot(const scene_rdl2::math::Vec3f& r, float time) const override {
        return getPosition(time);
    }

    float getThetaO() const override { return scene_rdl2::math::sPi; }
    float getThetaE() const override { return scene_rdl2::math::sPi * 0.5f; }

private:
    void initAttributeKeys(const scene_rdl2::rdl2::SceneClass &sc);

    /// Copy is disabled
    SphereLight(const SphereLight &other);
    const SphereLight &operator=(const SphereLight &other);

    SPHERE_LIGHT_MEMBERS;

    //
    // Cached attribute keys:
    //
    // cppcheck-suppress duplInheritedMember
    static bool                             sAttributeKeyInitialized;
    static scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Bool>   sNormalizedKey;
    static scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Bool>   sApplySceneScaleKey;
    static scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Float>  sRadiusKey;
    static scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Int>    sSidednessKey;
    DECLARE_ATTR_SKEYS_CLEAR_RADIUS
};

//----------------------------------------------------------------------------

} // namespace pbr
} // namespace moonray

