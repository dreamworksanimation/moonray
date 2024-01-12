// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "LightFilter.h"

#include <moonray/rendering/pbr/core/Distribution.h>

#include <scene_rdl2/common/math/Color.h>

// The cookie light filter takes a render space position and transforms it to
// a screen space position for a specified camera projection.  The screen space
// position is used to lookup a value in a Map shader, which becomes the filter value.
// The screen space position may be displaced by a random amount to apply a blur filter.

// Forward declaration of the ISPC types
namespace ispc {
    struct CookieLightFilter_v2;
}

namespace moonray {
namespace pbr {

class CookieLightFilter_v2 : public LightFilter
{
public:
    void initAttributeKeys(const scene_rdl2::rdl2::SceneClass &sc);

    CookieLightFilter_v2() {}
    CookieLightFilter_v2(const scene_rdl2::rdl2::LightFilter* rdlLightFilter);

    virtual ~CookieLightFilter_v2() override; 

    virtual void update(const LightFilterMap& lightFilters,
                        const scene_rdl2::math::Mat4d& world2render) override;
    virtual bool canIlluminate(const CanIlluminateData& data) const override;
    virtual scene_rdl2::math::Color eval(const EvalData& data) const override;
    virtual bool needsSamples() const override;

    /// HUD validation and type casting
    static uint32_t hudValidation(bool verbose) {
        COOKIE_LIGHT_FILTER_V2_VALIDATION;
    }
    HUD_AS_ISPC_METHODS(CookieLightFilter_v2);

private:

    static bool isValidXform(const scene_rdl2::math::Mat4d& xf);
    scene_rdl2::math::Mat4f computePerspectiveProjectionMatrix(float t) const;
    scene_rdl2::math::Mat4f computeOrthoProjectionMatrix(float t) const;

    COOKIE_LIGHT_FILTER_V2_MEMBERS;

    static bool sAttributeKeyInitialized;
    static scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::SceneObject*> sProjectorKey;
    static scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Mat4d> sProjectorXformKey;
    static scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Int> sProjectorTypeKey;
    static scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Float> sProjectorFocalKey;
    static scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Float> sProjectorFilmWidthApertureKey;
    static scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Float> sProjectorPixelAspectRatioKey;
    static scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Float> sBlurNearDistanceKey;
    static scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Float> sBlurMidpointKey;
    static scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Float> sBlurFarDistanceKey;
    static scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Float> sBlurNearValueKey;
    static scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Float> sBlurMidValueKey;
    static scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Float> sBlurFarValueKey;
    static scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Int> sBlurType;
    static scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Int> sOutsideProjection;
    static scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Float> sDensityKey;
    static scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Bool> sInvertKey;
    static scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::String> sTextureKey;
    static scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Rgb> sGammaKey;
};

} // namespace pbr
} // namespace moonray

