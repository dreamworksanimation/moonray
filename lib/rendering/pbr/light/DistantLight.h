// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0


#pragma once

#include "Light.h"

#include <scene_rdl2/common/math/Color.h>
#include <scene_rdl2/common/math/Mat4.h>
#include <scene_rdl2/common/math/ReferenceFrame.h>
#include <scene_rdl2/common/math/Vec3.h>

// Forward declaration of the ISPC types
namespace ispc {
    struct DistantLight;
}


namespace scene_rdl2 {
namespace rdl2 {
    class Light;
}
}

namespace moonray {
namespace pbr {

//----------------------------------------------------------------------------

/// @brief Implements light sampling for distant light.

class DistantLight : public Light
{
    friend class DistantLightTester;

public:
    /// Constructor / Destructor
    explicit DistantLight(const scene_rdl2::rdl2::Light* rdlLight, bool uniformSampling = false);
    virtual ~DistantLight();

    /// HUD validation and type casting
    static uint32_t hudValidation(bool verbose) {
        DISTANT_LIGHT_VALIDATION;
    }
    HUD_AS_ISPC_METHODS(DistantLight);


    virtual bool update(const scene_rdl2::math::Mat4d& world2render) override;

    /// Intersection and sampling API
    virtual bool canIlluminate(const scene_rdl2::math::Vec3f p, const scene_rdl2::math::Vec3f *n, float time, float radius,
            const LightFilterList* lightFilterList) const override;
    virtual bool isBounded() const override;
    virtual bool isDistant() const override;
    virtual bool isEnv() const override;
    virtual bool intersect(const scene_rdl2::math::Vec3f &p, const scene_rdl2::math::Vec3f *n, const scene_rdl2::math::Vec3f &wi, float time,
            float maxDistance, LightIntersection &isect) const override;
    virtual bool sample(const scene_rdl2::math::Vec3f &p, const scene_rdl2::math::Vec3f *n, float time, const scene_rdl2::math::Vec3f& r,
            scene_rdl2::math::Vec3f &wi, LightIntersection &isect, float rayDirFootprint) const override;
    virtual scene_rdl2::math::Color eval(mcrt_common::ThreadLocalState* tls, const scene_rdl2::math::Vec3f &wi, const scene_rdl2::math::Vec3f &p,
            const LightFilterRandomValues& filterR, float time, const LightIntersection &isect, bool fromCamera,
            const LightFilterList *lightFilterList, float rayDirFootprint, float *pdf = nullptr) const override;
    virtual scene_rdl2::math::Vec3f getEquiAngularPivot(const scene_rdl2::math::Vec3f& r, float time) const override;

    // Unbounded lights aren't included in the LightTree sampling BVH, so these values aren't needed
    float getThetaO() const override { return 0.f; }
    float getThetaE() const override { return 0.f; }

private:
    void initAttributeKeys(const scene_rdl2::rdl2::SceneClass &sc);

    scene_rdl2::math::Vec3f localToGlobal(const scene_rdl2::math::Vec3f &v,
                              float time) const;
    scene_rdl2::math::Vec3f globalToLocal(const scene_rdl2::math::Vec3f &v,
                              float time) const;
    scene_rdl2::math::Xform3f globalToLocalXform(float time, bool needed = true) const;

    /// Copy is disabled
    DistantLight(const DistantLight &other);
    const DistantLight &operator=(const DistantLight &other);

    DISTANT_LIGHT_MEMBERS;

    //
    // Cached attribute keys:
    //
    // cppcheck-suppress duplInheritedMember
    static bool                                                     sAttributeKeyInitialized;
    static scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Bool>   sNormalizedKey;
    static scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Float>  sAngularExtent;
};

//----------------------------------------------------------------------------

} // namespace pbr
} // namespace moonray

