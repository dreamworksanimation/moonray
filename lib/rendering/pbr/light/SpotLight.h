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
    struct SpotLight;
}


namespace scene_rdl2 {
namespace rdl2 {
    class Light;
}
}

namespace moonray {

namespace pbr {

//----------------------------------------------------------------------------

/// @brief Implements light sampling for spotlights.

class SpotLight : public LocalParamLight
{
    friend class SpotLightTester;

public:
    /// Constructor / Destructor
    explicit SpotLight(const scene_rdl2::rdl2::Light* rdlLight);
    virtual ~SpotLight();

    /// HUD validation and type casting
    static uint32_t hudValidation(bool verbose) {
        SPOT_LIGHT_VALIDATION;
    }
    HUD_AS_ISPC_METHODS(SpotLight);


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

    float getThetaO() const override { 
        return 0.f; 
    }
    float getThetaE() const override 
    { 
        // find outer angle
        // if lens is elliptical, find larger outer angle
        float aspectRatio = scene_rdl2::math::max(mRcpAspectRatio, 1.f); 
        // outer angle = atan( aspect ratio (if y > x) * opposite / adjacent )
        return scene_rdl2::math::atan(aspectRatio * (mFocalRadius + mLensRadius) / mFocalDistance);    
    }

protected:
    void initAttributeKeys(const scene_rdl2::rdl2::SceneClass &sc);

    void computeCorners(scene_rdl2::math::Vec3f *corners, float time) const;

    /// Copy is disabled
    SpotLight(const SpotLight &other);
    const SpotLight &operator=(const SpotLight &other);

    // Helper functions for converting between the spotlight's coordinate systems
    scene_rdl2::math::Vec2f lensToFocal(const scene_rdl2::math::Vec2f &lensCoords,  const scene_rdl2::math::Vec3f &localP) const;
    scene_rdl2::math::Vec2f focalToLens(const scene_rdl2::math::Vec2f &focalCoords, const scene_rdl2::math::Vec3f &localP) const;
    scene_rdl2::math::Vec2f getNormalizedLensCoords(const scene_rdl2::math::Vec2f &lensCoords) const;
    scene_rdl2::math::Vec2f getNormalizedFocalCoords(const scene_rdl2::math::Vec2f &focalCoords) const;
    scene_rdl2::math::Vec2f getLensCoords(const scene_rdl2::math::Vec2f &normalizedLensCoords) const;
    scene_rdl2::math::Vec2f getFocalCoords(const scene_rdl2::math::Vec2f &normalizedFocalCoords) const;
    scene_rdl2::math::Vec2f getUvsFromNormalized(const scene_rdl2::math::Vec2f& normalizedCoords) const;

    SPOT_LIGHT_MEMBERS;

    //
    // Cached attribute keys:
    //
    // cppcheck-suppress duplInheritedMember
    static bool                                                     sAttributeKeyInitialized;
    static scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Bool>   sNormalizedKey;
    static scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Bool>   sApplySceneScaleKey;
    static scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Float>  sLensRadiusKey;
    static scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Float>  sAspectRatioKey;
    static scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Float>  sFocalPlaneDistanceKey;
    static scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Float>  sOuterConeAngleKey;
    static scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Float>  sInnerConeAngleKey;
    static scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Int>    sAngleFalloffTypeKey;
    static scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Float>  sAngleFalloffExponentKey;
    static scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Float>  sBlackLevelKey;
    DECLARE_ATTR_SKEYS_CLEAR_RADIUS
};


//----------------------------------------------------------------------------

} // namespace pbr
} // namespace moonray


