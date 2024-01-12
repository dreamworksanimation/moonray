// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

#include "ColorRampLightFilter.h"

#include <moonray/rendering/pbr/lightfilter/ColorRampLightFilter_ispc_stubs.h>
#include <moonray/rendering/shading/ispc/RampControl_ispc_stubs.h>
#include <moonray/common/mcrt_macros/moonray_static_check.h>
#include <scene_rdl2/common/math/ReferenceFrame.h>

namespace moonray {
namespace pbr{


using namespace scene_rdl2;
using namespace scene_rdl2::math;

bool ColorRampLightFilter::sAttributeKeyInitialized;
rdl2::AttributeKey<rdl2::Mat4d> ColorRampLightFilter::sNodeXformKey;
rdl2::AttributeKey<rdl2::Bool> ColorRampLightFilter::sUseXformKey;
rdl2::AttributeKey<rdl2::Float> ColorRampLightFilter::sBeginDistanceKey;
rdl2::AttributeKey<rdl2::Float> ColorRampLightFilter::sEndDistanceKey;
rdl2::AttributeKey<rdl2::RgbVector> ColorRampLightFilter::sColorsKey;
rdl2::AttributeKey<rdl2::FloatVector> ColorRampLightFilter::sDistancesKey;
rdl2::AttributeKey<rdl2::IntVector> ColorRampLightFilter::sInterpolationTypesKey;
rdl2::AttributeKey<rdl2::Float> ColorRampLightFilter::sIntensityKey;
rdl2::AttributeKey<rdl2::Float> ColorRampLightFilter::sDensityKey;
rdl2::AttributeKey<rdl2::Int> ColorRampLightFilter::sModeKey;
rdl2::AttributeKey<rdl2::Int> ColorRampLightFilter::sWrapModeKey;

HUD_VALIDATOR(ColorRampLightFilter);

ColorRampLightFilter::ColorRampLightFilter(const rdl2::LightFilter* rdlLightFilter) :
    LightFilter(rdlLightFilter)
{
    if (mRdlLightFilter) {
        initAttributeKeys(mRdlLightFilter->getSceneClass());
    }
    ispc::ColorRampLightFilter_init((ispc::ColorRampLightFilter *)this->asIspc());
}

void
ColorRampLightFilter::initAttributeKeys(const rdl2::SceneClass &sc)
{
    if (sAttributeKeyInitialized) {
        return;
    }

    MOONRAY_START_NON_THREADSAFE_STATIC_WRITE

    sAttributeKeyInitialized = true;

    sNodeXformKey = sc.getAttributeKey<rdl2::Mat4d>("node_xform");
    sUseXformKey = sc.getAttributeKey<rdl2::Bool>("use_xform");
    sBeginDistanceKey = sc.getAttributeKey<rdl2::Float>("begin_distance");
    sEndDistanceKey = sc.getAttributeKey<rdl2::Float>("end_distance");
    sColorsKey = sc.getAttributeKey<rdl2::RgbVector>("colors");
    sDistancesKey = sc.getAttributeKey<rdl2::FloatVector>("distances");
    sInterpolationTypesKey = sc.getAttributeKey<rdl2::IntVector>("interpolation_types");
    sIntensityKey = sc.getAttributeKey<rdl2::Float>("intensity");
    sDensityKey = sc.getAttributeKey<rdl2::Float>("density");
    sModeKey = sc.getAttributeKey<rdl2::Int>("mode");
    sWrapModeKey = sc.getAttributeKey<rdl2::Int>("wrap_mode");

    MOONRAY_FINISH_NON_THREADSAFE_STATIC_WRITE
}

void
ColorRampLightFilter::update(const LightFilterMap& /*lightFilters*/,
                             const Mat4d& world2render)
{
    if (!mRdlLightFilter) {
        return;
    }

    Mat4d render2world = world2render.inverse();
    Mat4d local2world0 = mRdlLightFilter->get<rdl2::Mat4d>(sNodeXformKey, 0.f);
    Mat4d local2world1 = mRdlLightFilter->get<rdl2::Mat4d>(sNodeXformKey, 1.f);
    Mat4d world2local0 = local2world0.inverse();
    Mat4d world2local1 = local2world1.inverse();
    mXform[0] = Mat4f(render2world * world2local0);
    mXform[1] = Mat4f(render2world * world2local1);
    mUseXform = mRdlLightFilter->get<rdl2::Bool>(sUseXformKey);

    float beginDistance = mRdlLightFilter->get<rdl2::Float>(sBeginDistanceKey);
    float endDistance = mRdlLightFilter->get<rdl2::Float>(sEndDistanceKey);
    if (beginDistance >= endDistance) {
        mRdlLightFilter->error(
            "Color ramp light filter begin distance is >= end distance, using defaults");
        beginDistance = 0.f;
        endDistance = 1.f;
    }
    std::vector<Color> colorsVec = mRdlLightFilter->get<rdl2::RgbVector>(sColorsKey);
    std::vector<float> distancesVec = mRdlLightFilter->get<rdl2::FloatVector>(sDistancesKey);
    std::vector<int> interpolationTypesVec = mRdlLightFilter->get<rdl2::IntVector>(sInterpolationTypesKey);

    if (distancesVec.size() != colorsVec.size() || distancesVec.size() != interpolationTypesVec.size()) {
        mRdlLightFilter->error(
            "Color ramp light filter distances, colors and interpolation types are different sizes, using defaults");
        colorsVec = {rdl2::Rgb(1.f, 1.f, 1.f), rdl2::Rgb(0.f, 0.f, 0.f)};
        distancesVec = {0.f, 1.f};
        interpolationTypesVec = {1, 1};
    }

    // remap distances to startDistance to endDistance range
    float scale = endDistance - beginDistance;
    for (int i = 0; i < distancesVec.size(); i++) {
        distancesVec[i] = beginDistance + scale * distancesVec[i];
    }

    mColorRamp.init(
        distancesVec.size(),
        distancesVec.data(),
        colorsVec.data(),
        reinterpret_cast<const ispc::RampInterpolatorMode*>(interpolationTypesVec.data()),
        ispc::COLOR_RAMP_CONTROL_SPACE_RGB);

    mIntensity = mRdlLightFilter->get<rdl2::Float>(sIntensityKey);
    mDensity = clamp(mRdlLightFilter->get<rdl2::Float>(sDensityKey), 0.f, 1.f);
    mMode = mRdlLightFilter->get<rdl2::Int>(sModeKey);
    mWrapMode = mRdlLightFilter->get<rdl2::Int>(sWrapModeKey);
}

bool
ColorRampLightFilter::canIlluminate(const CanIlluminateData& data) const
{
    return true;
}

Color
ColorRampLightFilter::eval(const EvalData& data) const
{
    float dist;
    switch (mMode) {
    case RADIAL:
        if (mUseXform) {
            Vec3f xformedPoint0 = transformPoint(mXform[0], data.shadingPointPosition);
            Vec3f xformedPoint1 = transformPoint(mXform[1], data.shadingPointPosition);
            dist = lerp(length(xformedPoint0), length(xformedPoint1), data.time);
        } else {
            dist = data.isect->distance;
        }
    break;
    case DIRECTIONAL:
        if (mUseXform) {
            Vec3f xformedPoint0 = transformPoint(mXform[0], data.shadingPointPosition);
            Vec3f xformedPoint1 = transformPoint(mXform[1], data.shadingPointPosition);
            Vec3f xformedPoint = lerp(xformedPoint0, xformedPoint1, data.time);
            dist = -xformedPoint.z;
        } else {
            Plane plane(data.lightPosition, data.lightDirection);
            dist = plane.getDistance(data.shadingPointPosition);
            if (dot(data.shadingPointPosition - data.lightPosition, data.lightDirection) < 0.f) {
                // Restores the sign of dist because plane.getDistance() returns the abs distance to the plane
                dist *= -1.f;
            }
        }

        switch (mWrapMode) {
        case EXTEND:
            // Causes f(dist) = f(0) for dist < 0
            if (dist < 0.f) dist = 0.f;
        break;
        case MIRROR:
            // Causes f(dist) = f(-dist), mirroring about dist = 0
            dist = fabs(dist);
        break;
        default:
            MNRY_ASSERT(false);
        }

    break;
    default:
        MNRY_ASSERT(false);
    }

    Color filterVal = mColorRamp.eval1D(dist);
    
    // Scale the filter color value by the intensity
    filterVal *= mIntensity;

    // Apply density scaling to allow partial light filtering
    filterVal = Color(1.f - mDensity) + filterVal * mDensity;

    return filterVal;
}


} //namespace pbr
} //namespace moonray

