// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0


#pragma once

#include "LightFilter.h"

#include <scene_rdl2/common/math/Color.h>
#include <moonray/rendering/shading/RampControl.h>

// The VDB light filter takes a render space position and transforms it to
// a camera space position. The camera space position is used to lookup a value 
// in a Map shader, which becomes the filter value. The camera space position may be displaced
// by a random amount to apply a blur filter.

// Forward declaration of the ISPC types
namespace ispc {
    struct VdbLightFilter;
}

namespace moonray {

//forward declarations within moonray namespace
namespace shading {    
    class OpenVdbSampler;
}
namespace pbr {

class VdbLightFilter : public LightFilter
{
public:
    void initAttributeKeys(const scene_rdl2::rdl2::SceneClass &sc);

    VdbLightFilter() : mDensitySampler(nullptr) {}
    VdbLightFilter(const scene_rdl2::rdl2::LightFilter* rdlLightFilter);

    ~VdbLightFilter() override;

    void update(const LightFilterMap& lightFilters,
                        const scene_rdl2::math::Mat4d& world2render) override;
    bool canIlluminate(const CanIlluminateData& data) const override;
    virtual bool needsSamples() const override;
    scene_rdl2::math::Color eval(const EvalData& data) const override;

    /// HUD validation and type casting
    static uint32_t hudValidation(bool verbose) {
        VDB_LIGHT_FILTER_VALIDATION;
    }
    
    static void sampleVdb(const scene_rdl2::rdl2::LightFilter* lightFilter,
                          moonray::shading::TLState *tls,
                          const float px, 
                          const float py, 
                          const float pz,
                          float *outDensity);
    HUD_AS_ISPC_METHODS(VdbLightFilter);

private:

    bool isValidXform(const scene_rdl2::math::Mat4d& xf);

    VDB_LIGHT_FILTER_MEMBERS;

    static bool sAttributeKeyInitialized;    
    static scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Mat4d> sVdbXformKey;
    static scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Int> sVdbInterpolation;
    static scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Float> sDensityRemapInputMinKey;
    static scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Float> sDensityRemapInputMaxKey;
    static scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Float> sDensityRemapOutputMinKey;
    static scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Float> sDensityRemapOutputMaxKey;
    static scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Bool> sDensityRemapRescaleEnableKey;
    static scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::FloatVector> sDensityRemapOutputsKey;
    static scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::FloatVector> sDensityRemapInputsKey;
    static scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::IntVector> sDensityRemapInterpTypesKey;
    static scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::String> sVdbMapKey;
    static scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::String> sDensityGridNameKey;
    static scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Rgb> sColorTintNameKey;
    static scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Float> sBlurValueKey;
    static scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Int> sBlurType;
    static scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Bool> sInvertDensityKey;

};

} // namespace pbr
} // namespace moonray

