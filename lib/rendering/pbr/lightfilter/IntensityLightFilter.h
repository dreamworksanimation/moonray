// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

//


#pragma once

#include "LightFilter.h"

#include <scene_rdl2/common/math/Color.h>

// Forward declaration of the ISPC types
namespace ispc {
    struct IntensityLightFilter;
}

namespace moonray {
namespace pbr {


class IntensityLightFilter : public LightFilter
{
public:
    void initAttributeKeys(const scene_rdl2::rdl2::SceneClass &sc);

    IntensityLightFilter() : mRadianceMod(1.0f) {}
    IntensityLightFilter(const scene_rdl2::rdl2::LightFilter* rdlLightFilter);

    virtual ~IntensityLightFilter() override {}

    virtual void update(const LightFilterMap& lightFilters,
                        const scene_rdl2::math::Mat4d& world2render) override;
    virtual bool canIlluminate(const CanIlluminateData &data) const override;
    virtual scene_rdl2::math::Color eval(const EvalData& data) const override;

    /// HUD validation and type casting
    static uint32_t hudValidation(bool verbose) {
        INTENSITY_LIGHT_FILTER_VALIDATION;
    }
    HUD_AS_ISPC_METHODS(IntensityLightFilter);

private:
    INTENSITY_LIGHT_FILTER_MEMBERS;

    static bool sAttributeKeyInitialized;
    static scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Float> sIntensityKey;
    static scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Float> sExposureKey;
    static scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Rgb> sColorKey;
    static scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Bool> sInvertKey;
};

} // namespace pbr
} // namespace moonray

