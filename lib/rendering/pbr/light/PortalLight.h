// Copyright 2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "Light.h"
#include "RectLight.h"

// Forward declaration of the ISPC types
namespace ispc {
    struct PortalLight;
}

namespace scene_rdl2 {
namespace rdl2 {
    class Light;
}
}

namespace moonray {
namespace pbr {

//----------------------------------------------------------------------------

/// @brief Implements portal light sampling.

/// In situations where a light (like an EnvLight) is mostly occluded by geometry (e.g. a room with a window),
/// we can use a PortalLight to focus sampling toward a particular region, preventing us from taking occluded samples.
/// This PortalLight behaves like a RectLight, except that, for evaluation, it allows the ray to continue on to
/// the light behind it (mRefLight) and uses the radiance from that intersection point. The color, exposure, and
/// intensity on the PortalLight act as multipliers to the radiance we get from the reference light, mRefLight.

class PortalLight : public RectLight
{

public:
    /// Constructor / Destructor
    explicit PortalLight(const scene_rdl2::rdl2::Light* rdlLight);
    virtual ~PortalLight();

    /// HUD validation and type casting
    static uint32_t hudValidation(bool verbose) {
        PORTAL_LIGHT_VALIDATION;
    }
    HUD_AS_ISPC_METHODS(PortalLight);

    const scene_rdl2::rdl2::Light* getRefRdlLight() const { return mRefRdlLight; }

    /// Is this light (and the referenced light) active?
    bool isOn() const override {
        return mRdlLight->get(scene_rdl2::rdl2::Light::sOnKey) && mRefRdlLight->get(scene_rdl2::rdl2::Light::sOnKey);
    }

    /// Set the portal's reference light and indicate to that light that
    /// it will be using portal light sampling instead
    void setRefLight(Light* refLight) {
        refLight->turnOnPortal();
        mRefLight = refLight;
    }

    virtual bool update(const scene_rdl2::math::Mat4d& world2render) override;
    virtual bool canIlluminate(const scene_rdl2::math::Vec3f p, const scene_rdl2::math::Vec3f *n, float time, float radius,
            const LightFilterList* lightFilterList, const PathVertex* pv) const override;
    virtual bool intersect(const scene_rdl2::math::Vec3f &p, const scene_rdl2::math::Vec3f *n,
            const scene_rdl2::math::Vec3f &wi, float time, float maxDistance, LightIntersection &isect) const override;
    virtual bool sample(const scene_rdl2::math::Vec3f &p, const scene_rdl2::math::Vec3f *n, float time,
            const scene_rdl2::math::Vec3f& r, scene_rdl2::math::Vec3f &wi, LightIntersection &isect,
            float rayDirFootprint) const override;
    virtual scene_rdl2::math::Color eval(mcrt_common::ThreadLocalState* tls, const scene_rdl2::math::Vec3f &wi, 
            const scene_rdl2::math::Vec3f &p, const LightFilterRandomValues& filterR, float time, 
            const LightIntersection &isect, bool fromCamera, const LightFilterList *lightFilterList, 
            const PathVertex *pv, float rayDirFootprint, float *visibility, float *pdf) const override;

private:
    void initAttributeKeys(const scene_rdl2::rdl2::SceneClass &sc);

    PORTAL_LIGHT_MEMBERS;

    static scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::SceneObject*>   sRefLight;
    static bool                                                             sAttributeKeyInitialized;
    static scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Bool>           sNormalizedKey;
    static scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Bool>           sApplySceneScaleKey;
    static scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Float>          sWidthKey;
    static scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Float>          sHeightKey;
    static scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Float>          sSpreadKey;
    static scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Int>            sSidednessKey;
    DECLARE_ATTR_SKEYS_CLEAR_RADIUS
};

//----------------------------------------------------------------------------

} // namespace pbr
} // namespace moonray

