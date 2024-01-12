// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0


#include "IntensityLightFilter.h"

#include <moonray/rendering/pbr/lightfilter/IntensityLightFilter_ispc_stubs.h>
#include <moonray/common/mcrt_macros/moonray_static_check.h>

namespace moonray {
namespace pbr{

using namespace scene_rdl2;
using namespace scene_rdl2::math;

bool IntensityLightFilter::sAttributeKeyInitialized;
scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Float> IntensityLightFilter::sIntensityKey;
scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Float> IntensityLightFilter::sExposureKey;
scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Rgb> IntensityLightFilter::sColorKey;
scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Bool> IntensityLightFilter::sInvertKey;

HUD_VALIDATOR(IntensityLightFilter);

IntensityLightFilter::IntensityLightFilter(const scene_rdl2::rdl2::LightFilter* rdlLightFilter) :
                                           LightFilter(rdlLightFilter),
                                           mRadianceMod(1.0f)
{
    if (mRdlLightFilter) {
        initAttributeKeys(mRdlLightFilter->getSceneClass());
    }

    ispc::IntensityLightFilter_init((ispc::IntensityLightFilter *)this->asIspc());
}

void
IntensityLightFilter::initAttributeKeys(const scene_rdl2::rdl2::SceneClass &sc)
{
    if (sAttributeKeyInitialized) {
        return;
    }

    MOONRAY_START_NON_THREADSAFE_STATIC_WRITE

    sAttributeKeyInitialized = true;

    sIntensityKey = sc.getAttributeKey<scene_rdl2::rdl2::Float>("intensity");
    sExposureKey = sc.getAttributeKey<scene_rdl2::rdl2::Float>("exposure");
    sColorKey = sc.getAttributeKey<scene_rdl2::rdl2::Rgb>("color");
    sInvertKey = sc.getAttributeKey<scene_rdl2::rdl2::Bool>("invert");

    MOONRAY_FINISH_NON_THREADSAFE_STATIC_WRITE
}

void
IntensityLightFilter::update(const LightFilterMap& /*lightFilters*/,
                             const Mat4d& world2render)
{
    if (!mRdlLightFilter) {
        return;
    }

    float intensity = mRdlLightFilter->get<scene_rdl2::rdl2::Float>(sIntensityKey);
    float exposure = mRdlLightFilter->get<scene_rdl2::rdl2::Float>(sExposureKey);
    math::Color color = mRdlLightFilter->get<scene_rdl2::rdl2::Rgb>(sColorKey);
    bool invert = mRdlLightFilter->get<scene_rdl2::rdl2::Bool>(sInvertKey);

    mRadianceMod = intensity * color * powf(2.0f, exposure);
    if (invert) {
        if (!isZero(mRadianceMod.r)) {
            mRadianceMod.r = 1.f / mRadianceMod.r;
        }
        if (!isZero(mRadianceMod.g)) {
            mRadianceMod.g = 1.f / mRadianceMod.g;
        }
        if (!isZero(mRadianceMod.b)) {
            mRadianceMod.b = 1.f / mRadianceMod.b;
        }
    }
}

bool
IntensityLightFilter::canIlluminate(const CanIlluminateData &data) const
{
    return !math::isBlack(mRadianceMod);
}


Color
IntensityLightFilter::eval(const EvalData& data) const
{
    return mRadianceMod;
}

} //namespace pbr
} //namespace moonray

