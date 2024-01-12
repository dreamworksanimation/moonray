// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "LightFilter.h"

#include <scene_rdl2/common/math/Color.h>

// Forward declaration of the ISPC types
namespace ispc {
    struct CombineLightFilter;
}

namespace moonray {
namespace pbr {

class CombineLightFilter : public LightFilter
{
public:
    void initAttributeKeys(const scene_rdl2::rdl2::SceneClass &sc);

    CombineLightFilter();
    CombineLightFilter(const scene_rdl2::rdl2::LightFilter* rdlLightFilter);

    virtual ~CombineLightFilter() override;

    virtual void update(const LightFilterMap& lightFilters,
                        const scene_rdl2::math::Mat4d& world2render) override;
    virtual bool canIlluminate(const CanIlluminateData& data) const override;
    virtual scene_rdl2::math::Color eval(const EvalData& data) const override;
    virtual bool needsLightXform() const override;

    /// HUD validation and type casting
    static uint32_t hudValidation(bool verbose) {
        COMBINE_LIGHT_FILTER_VALIDATION;
    }
    HUD_AS_ISPC_METHODS(CombineLightFilter);

private:

    COMBINE_LIGHT_FILTER_MEMBERS;

    static bool sAttributeKeyInitialized;
    static scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::SceneObjectVector> sLightFiltersKey;
    static scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Int> sModeKey;

};

} // namespace pbr
} // namespace moonray

