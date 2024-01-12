// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0


#include "DecayLightFilter.h"

#include <moonray/rendering/pbr/lightfilter/DecayLightFilter_ispc_stubs.h>
#include <moonray/common/mcrt_macros/moonray_static_check.h>

namespace moonray {
namespace pbr{

using namespace scene_rdl2;
using namespace scene_rdl2::math;

bool DecayLightFilter::sAttributeKeyInitialized;
scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Bool> DecayLightFilter::sFalloffNearKey;
scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Bool> DecayLightFilter::sFalloffFarKey;
scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Float> DecayLightFilter::sNearStartKey;
scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Float> DecayLightFilter::sNearEndKey;
scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Float> DecayLightFilter::sFarStartKey;
scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Float> DecayLightFilter::sFarEndKey;

HUD_VALIDATOR(DecayLightFilter);

DecayLightFilter::DecayLightFilter(const scene_rdl2::rdl2::LightFilter* rdlLightFilter) :
    LightFilter(rdlLightFilter),
    mFalloffNear(false), mFalloffFar(false),
    mNearStart(0.0f), mNearEnd(0.0f), mFarStart(0.0f), mFarEnd(0.0f)
{
    if (mRdlLightFilter) {
        initAttributeKeys(mRdlLightFilter->getSceneClass());
    }

    ispc::DecayLightFilter_init((ispc::DecayLightFilter *)this->asIspc());
}

void
DecayLightFilter::initAttributeKeys(const scene_rdl2::rdl2::SceneClass &sc)
{
    if (sAttributeKeyInitialized) {
        return;
    }

    MOONRAY_START_NON_THREADSAFE_STATIC_WRITE

    sAttributeKeyInitialized = true;

    sFalloffNearKey = sc.getAttributeKey<scene_rdl2::rdl2::Bool>("falloff_near");
    sFalloffFarKey = sc.getAttributeKey<scene_rdl2::rdl2::Bool>("falloff_far");
    sNearStartKey = sc.getAttributeKey<scene_rdl2::rdl2::Float>("near_start");
    sNearEndKey = sc.getAttributeKey<scene_rdl2::rdl2::Float>("near_end");
    sFarStartKey = sc.getAttributeKey<scene_rdl2::rdl2::Float>("far_start");
    sFarEndKey = sc.getAttributeKey<scene_rdl2::rdl2::Float>("far_end");

    MOONRAY_FINISH_NON_THREADSAFE_STATIC_WRITE
}

void
DecayLightFilter::update(const LightFilterMap& /*lightFilters*/,
                         const Mat4d& world2render)
{
    if (!mRdlLightFilter) {
        return;
    }
    mFalloffNear = mRdlLightFilter->get<scene_rdl2::rdl2::Bool>(sFalloffNearKey);
    mFalloffFar = mRdlLightFilter->get<scene_rdl2::rdl2::Bool>(sFalloffFarKey);
    mNearStart = mRdlLightFilter->get<scene_rdl2::rdl2::Float>(sNearStartKey);
    mNearEnd = scene_rdl2::math::max(mRdlLightFilter->get<scene_rdl2::rdl2::Float>(sNearEndKey), mNearStart);
    mFarStart = mRdlLightFilter->get<scene_rdl2::rdl2::Float>(sFarStartKey);
    // If there is a near falloff, we want the far falloff to start afterwards,
    // otherwise there is a discontinuity in the light decay.
    if (mFalloffNear) {
        mFarStart = scene_rdl2::math::max(mFarStart, mNearEnd);
    }
    mFarEnd = scene_rdl2::math::max(mRdlLightFilter->get<scene_rdl2::rdl2::Float>(sFarEndKey), mFarStart);
}

bool
DecayLightFilter::canIlluminate(const CanIlluminateData& data) const
{
    // Approximate light and shading point as two spheres. We can compute the
    // minimum and maximum distances between any 2 points in those two spheres.
    float distance = scene_rdl2::math::length(data.lightPosition - data.shadingPointPosition);
    float buffer = data.lightRadius + data.shadingPointRadius;
    float minDistance = buffer > distance ? 0.f : distance - buffer;
    float maxDistance = distance + buffer;

    bool success = true;

    if (mFalloffNear) {
        success &= maxDistance >= mNearStart;
    }

    if (mFalloffFar) {
        success &= minDistance <= mFarEnd;
    }

    return success;
}


Color
DecayLightFilter::eval(const EvalData& data) const
{
    float dist = data.isect->distance;

    if ((mFalloffNear && dist < mNearStart) ||
        (mFalloffFar && dist > mFarEnd)) {
        return Color(0.0f);

    } else if (mFalloffNear && dist < mNearEnd) {
        return Color((dist - mNearStart) / (mNearEnd - mNearStart));

    } else if (mFalloffFar && dist > mFarStart) {
        return Color((mFarEnd - dist) / (mFarEnd - mFarStart));
    }

    return Color(1.0f);
}

} //namespace pbr
} //namespace moonray

