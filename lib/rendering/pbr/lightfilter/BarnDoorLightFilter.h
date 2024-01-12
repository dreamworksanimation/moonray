// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "LightFilter.h"

#include <scene_rdl2/common/math/Color.h>

// The Barn Door light filter is based on the on set technique of hinging doors
// to light fixtures to control and shape and spill of light. As it simulates
// being a light attachment, it works best with positional lights i.e. NOT
// directional and env lights. There are two modes to this filter; physical and
// analytic. Physical giving us a more accurate representation of real shadows,
// looking at the size of the light, the distance of the barn door to the
// subject etc. Analytic skews results in favour of artistic control over
// physical accuracy.
//
// The user configures a rectangular portal which is defined in a similar
// fashion as the CookieLightFilter. There is a projector which has a position
// and direction and projects essentially a procedural texture: a blurry
// rectangle. The blurry rectangle represents the portal defined by the ends of
// the four flaps of a real barn door, here known as the "flap opening". It can
// have rounded corners, blurry edges, and be resized (as if adjusting the angle
// of the virtual flaps). The focal distance determines how far away the portal
// sits from the filter origin.
//
// In analytic mode we use the projection of the shading point directly into
// screen space.
//
// For physical mode we intersect the light ray with the plane of the
// rectangular portal, then project that point into screen space.
//
// Once the point is in the screen space, it is checked to see if its within the
// (rounded, blurry) rectangle.


// Forward declaration of the ISPC types
namespace ispc {
    struct BarnDoorLightFilter;
}

namespace moonray {
namespace pbr {

class BarnDoorLightFilter : public LightFilter
{
public:
    void initAttributeKeys(const scene_rdl2::rdl2::SceneClass &sc);

    BarnDoorLightFilter();
    BarnDoorLightFilter(const scene_rdl2::rdl2::LightFilter* rdlLightFilter);

    virtual ~BarnDoorLightFilter() override {}

    virtual void update(const LightFilterMap& lightFilters,
                        const scene_rdl2::math::Mat4d& world2render) override;
    virtual bool canIlluminate(const CanIlluminateData& data) const override;
    virtual scene_rdl2::math::Color eval(const EvalData& data) const override;
    virtual bool needsLightXform() const override { return mUseLightXform; }

    /// HUD validation and type casting
    static uint32_t hudValidation(bool verbose) {
        BARN_DOOR_LIGHT_FILTER_VALIDATION;
    }
    HUD_AS_ISPC_METHODS(BarnDoorLightFilter);

private:

    bool updateParamAndTransforms(const scene_rdl2::math::Mat4f &local2Render0,
                                  const scene_rdl2::math::Mat4f &local2Render1);
    bool updateTransforms(const scene_rdl2::math::Mat4f &local2Render, int ti);
    finline bool isMb() const { return (mMb != 0); }
    scene_rdl2::math::Xform3f getSlerpXformRender2Local(float time) const;
    scene_rdl2::math::Xform3f getXformRender2Local(float time) const
    {
        return isMb() ? getSlerpXformRender2Local(time) : mRender2Local0;
    }

    static scene_rdl2::math::Mat4f getPerspectiveProjectionMatrix();
    static scene_rdl2::math::Mat4f getOrthoProjectionMatrix();

    BARN_DOOR_LIGHT_FILTER_MEMBERS;

    static bool sAttributeKeyInitialized;
    static scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Mat4d> sProjectorXformKey;
    static scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Int> sProjectorTypeKey;
    static scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Float> sProjectorFocalDistKey;
    static scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Float> sProjectorWidthKey;
    static scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Float> sProjectorHeightKey;
    static scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Float> sRadiusKey;
    static scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Float> sEdgeKey;
    static scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Float> sEdgeScaleTopKey;
    static scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Float> sEdgeScaleBottomKey;
    static scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Float> sEdgeScaleLeftKey;
    static scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Float> sEdgeScaleRightKey;
    static scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Int> sPreBarnModeKey;
    static scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Float> sPreBarnDistKey;
    static scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Float> sDensityKey;
    static scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Bool> sInvertKey;
    static scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Int> sModeKey;
    static scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Float> sSizeTopKey;
    static scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Float> sSizeBottomKey;
    static scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Float> sSizeLeftKey;
    static scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Float> sSizeRightKey;
    static scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Bool> sUseLightXformKey;
    static scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Float> sRotationKey;
    static scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Rgb> sColorKey;
};

} // namespace pbr
} // namespace moonray

