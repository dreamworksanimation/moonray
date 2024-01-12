// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

//


#pragma once

#include "LightFilter.h"

#include <scene_rdl2/common/math/Color.h>

// Forward declaration of the ISPC types
namespace ispc {
    struct DecayLightFilter;
}

namespace moonray {
namespace pbr {


class DecayLightFilter : public LightFilter
{
public:
    void initAttributeKeys(const scene_rdl2::rdl2::SceneClass &sc);

    DecayLightFilter() :
        mFalloffNear(false), mFalloffFar(false),
        mNearStart(0.0f), mNearEnd(0.0f), mFarStart(0.0f), mFarEnd(0.0f) {}
    DecayLightFilter(const scene_rdl2::rdl2::LightFilter* rdlLightFilter);

    virtual ~DecayLightFilter() override {}

    virtual void update(const LightFilterMap& lightFilters,
                        const scene_rdl2::math::Mat4d& world2render) override;
    virtual bool canIlluminate(const CanIlluminateData& data) const override;
    virtual scene_rdl2::math::Color eval(const EvalData& data) const override;

    /// HUD validation and type casting
    static uint32_t hudValidation(bool verbose) {
        DECAY_LIGHT_FILTER_VALIDATION;
    }
    HUD_AS_ISPC_METHODS(DecayLightFilter);
  
private:
    DECAY_LIGHT_FILTER_MEMBERS;

    static bool sAttributeKeyInitialized;
    static scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Bool> sFalloffNearKey;
    static scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Bool> sFalloffFarKey;
    static scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Float> sNearStartKey;
    static scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Float> sNearEndKey;
    static scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Float> sFarStartKey;
    static scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Float> sFarEndKey;
};

} // namespace pbr
} // namespace moonray

