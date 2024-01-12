// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

//


#pragma once

#include "LightFilter.h"
#include <scene_rdl2/common/math/Color.h>
#include <moonray/rendering/shading/RampControl.h>

// Forward declaration of the ISPC types
namespace ispc {
    struct RodLightFilter;
}

namespace moonray {
namespace pbr {


class RodLightFilter : public LightFilter
{
public:
    void initAttributeKeys(const scene_rdl2::rdl2::SceneClass &sc);

    RodLightFilter() {}
    RodLightFilter(const scene_rdl2::rdl2::LightFilter* rdlLightFilter);

    virtual ~RodLightFilter() override {}

    virtual void update(const LightFilterMap& lightFilters,
                        const scene_rdl2::math::Mat4d& world2render) override;
    virtual bool canIlluminate(const CanIlluminateData& data) const override;
    virtual scene_rdl2::math::Color eval(const EvalData& data) const override;

    /// HUD validation and type casting
    static uint32_t hudValidation(bool verbose) {
        ROD_LIGHT_FILTER_VALIDATION;
    }
    HUD_AS_ISPC_METHODS(RodLightFilter);

private:
    ROD_LIGHT_FILTER_MEMBERS;

    bool updateParamAndTransforms(const scene_rdl2::math::Mat4f &local2Render0,
                                  const scene_rdl2::math::Mat4f &local2Render1);
    bool updateTransforms(const scene_rdl2::math::Mat4f &local2Render, int ti);
    finline bool isMb() const { return mMb; }
    scene_rdl2::math::Vec3f slerpPointRender2Local(const scene_rdl2::math::Vec3f &p, float time) const;

    scene_rdl2::math::Vec3f xformPointRender2Local(const scene_rdl2::math::Vec3f &p, float time) const
    {
        if (!isMb()) return transformPoint(mRender2Local0, p);

        return slerpPointRender2Local(p, time);
    }
     
    bool isOutsideInfluence(const scene_rdl2::math::Vec3f &p, float rad) const;
    float signedDistanceRoundBox(const scene_rdl2::math::Vec3f &p ) const;

    static bool sAttributeKeyInitialized;
    static scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Mat4d> sNodeXformKey;
    static scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Float> sWidthKey;
    static scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Float> sDepthKey;
    static scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Float> sHeightKey;
    static scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Float> sRadiusKey;
    static scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Float> sEdgeKey;
    static scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Rgb> sColorKey;
    static scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Float> sIntensityKey;
    static scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Float> sDensityKey;
    static scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Bool> sInvertKey;
    static scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::FloatVector> sRampInKey;
    static scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::FloatVector> sRampOutKey;
    static scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::IntVector> sRampInterpolationTypesKey;
};

} // namespace pbr
} // namespace moonray

