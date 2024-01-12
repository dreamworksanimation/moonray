// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

//

#pragma once

#include "LightFilter.h"

#include <moonray/rendering/shading/RampControl.h>
#include <scene_rdl2/common/math/Color.h>

// Forward declaration of the ISPC types
namespace ispc {
    struct ColorRampLightFilter;
}

namespace moonray {
namespace pbr {

class ColorRampLightFilter : public LightFilter
{
public:
    void initAttributeKeys(const scene_rdl2::rdl2::SceneClass &sc);

    ColorRampLightFilter() {}
    ColorRampLightFilter(const scene_rdl2::rdl2::LightFilter* rdlLightFilter);

    virtual ~ColorRampLightFilter() override {}

    virtual void update(const LightFilterMap& lightFilters,
                        const scene_rdl2::math::Mat4d& world2render) override;
    virtual bool canIlluminate(const CanIlluminateData& data) const override;
    virtual scene_rdl2::math::Color eval(const EvalData& data) const override;

    /// HUD validation and type casting
    static uint32_t hudValidation(bool verbose) {
        COLOR_RAMP_LIGHT_FILTER_VALIDATION;
    }
    HUD_AS_ISPC_METHODS(ColorRampLightFilter);
  
private:
    COLOR_RAMP_LIGHT_FILTER_MEMBERS;

    static bool sAttributeKeyInitialized;
    static scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Mat4d> sNodeXformKey;
    static scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Bool> sUseXformKey;
    static scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Float> sBeginDistanceKey;
    static scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Float> sEndDistanceKey;
    static scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::RgbVector> sColorsKey;
    static scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::FloatVector> sDistancesKey;
    static scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::IntVector> sInterpolationTypesKey;
    static scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Float> sIntensityKey;
    static scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Float> sDensityKey;
    static scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Int> sModeKey;
    static scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Int> sWrapModeKey;
    
    static scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::RgbVector> sOutDistancesKey;
    static scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::FloatVector> sInDistancesKey;
    static scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::IntVector>   attrInterpolationTypes;
};

} // namespace pbr
} // namespace moonray

