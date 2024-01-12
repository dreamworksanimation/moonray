// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

#include "CombineLightFilter.h"

#include <moonray/common/mcrt_macros/moonray_static_check.h>
#include <moonray/rendering/pbr/core/Util.h>
#include <moonray/rendering/pbr/lightfilter/CombineLightFilter_ispc_stubs.h>

namespace moonray {
namespace pbr{

using namespace scene_rdl2;
using namespace scene_rdl2::math;

bool CombineLightFilter::sAttributeKeyInitialized;
rdl2::AttributeKey<rdl2::SceneObjectVector> CombineLightFilter::sLightFiltersKey;
rdl2::AttributeKey<rdl2::Int> CombineLightFilter::sModeKey;


HUD_VALIDATOR(CombineLightFilter);

CombineLightFilter::CombineLightFilter() :
    mNumLightFilters(0),
    mMode(MULTIPLY)
{
}

CombineLightFilter::CombineLightFilter(const rdl2::LightFilter* rdlLightFilter) :
    LightFilter(rdlLightFilter),
    mNumLightFilters(0),
    mMode(MULTIPLY)
{
    if (mRdlLightFilter) {
        initAttributeKeys(mRdlLightFilter->getSceneClass());
    }

    ispc::CombineLightFilter_init((ispc::CombineLightFilter *)this->asIspc());
}

CombineLightFilter::~CombineLightFilter()
{
}

void
CombineLightFilter::initAttributeKeys(const rdl2::SceneClass &sc)
{
    if (sAttributeKeyInitialized) {
        return;
    }

    MOONRAY_START_NON_THREADSAFE_STATIC_WRITE

    sAttributeKeyInitialized = true;

    sLightFiltersKey = sc.getAttributeKey<rdl2::SceneObjectVector>("light_filters");
    sModeKey = sc.getAttributeKey<rdl2::Int>("mode");

    MOONRAY_FINISH_NON_THREADSAFE_STATIC_WRITE
}

void
CombineLightFilter::update(const LightFilterMap& lightFilters,
                           const Mat4d& world2Render)
{
    if (!mRdlLightFilter) {
        return;
    }

    mLightFiltersVec.clear();

    const rdl2::SceneObjectVector& rdlLightFilters =
        mRdlLightFilter->get<rdl2::SceneObjectVector>(sLightFiltersKey);

    for (rdl2::SceneObject* sceneObject : rdlLightFilters) {
        const rdl2::LightFilter *rdlLightFilter = sceneObject->asA<rdl2::LightFilter>();
        if (rdlLightFilter == nullptr) {
            // not a light filter, ignore
            continue;
        }
        if (!rdlLightFilter->isOn()) {
            // only enabled light filters will be in the LightFilterMap
            continue; 
        }
        auto search = lightFilters.find(rdlLightFilter);
        MNRY_ASSERT(search != lightFilters.end());
        const LightFilter *filter = search->second.get();
        mLightFiltersVec.push_back(filter);
    }
    mLightFilters = mLightFiltersVec.data();
    mNumLightFilters = mLightFiltersVec.size();

    mMode = mRdlLightFilter->get<rdl2::Int>(sModeKey);
}

bool 
CombineLightFilter::needsLightXform() const { 
    for (int i = 0; i < mLightFiltersVec.size(); i++) {
        if (mLightFiltersVec[i]->needsLightXform()) {
            return true;
        }
    }
    return false;
}

bool
CombineLightFilter::canIlluminate(const CanIlluminateData& data) const
{
    for (int i = 0; i < mLightFiltersVec.size(); i++) {
        if (mLightFiltersVec[i]->canIlluminate(data)) {
            return true;
        }
    }
    return false;
}

Color
CombineLightFilter::eval(const EvalData& data) const
{
    Color val;

    switch (mMode) {
    case MULTIPLY:
        val = Color(1.f);
        for (int i = 0; i < mLightFiltersVec.size(); i++) {
            val *= mLightFiltersVec[i]->eval(data);
        }
    break;
    case MIN:
        val = Color(1.f);
        for (int i = 0; i < mLightFiltersVec.size(); i++) {
            Color filterVal = mLightFiltersVec[i]->eval(data);
            val.r = std::min(val.r, filterVal.r);
            val.g = std::min(val.g, filterVal.g);
            val.b = std::min(val.b, filterVal.b);
        }
    break;
    case MAX:
        val = Color(0.f);
        for (int i = 0; i < mLightFiltersVec.size(); i++) {
            Color filterVal = mLightFiltersVec[i]->eval(data);
            val.r = std::max(val.r, filterVal.r);
            val.g = std::max(val.g, filterVal.g);
            val.b = std::max(val.b, filterVal.b);
        }
    break;
    case ADD:
        val = Color(0.f);
        for (int i = 0; i < mLightFiltersVec.size(); i++) {
            val += mLightFiltersVec[i]->eval(data);
        }
        val.r = std::min(val.r, 1.f);
        val.g = std::min(val.g, 1.f);
        val.b = std::min(val.b, 1.f);
    break;
    case SUBTRACT:
        val = Color(1.f);
        for (int i = 0; i < mLightFiltersVec.size(); i++) {
            if (i == 0) {
                val = mLightFiltersVec[i]->eval(data);
            } else {
                val -= mLightFiltersVec[i]->eval(data);
            }
        }
        val.r = std::max(val.r, 0.f);
        val.g = std::max(val.g, 0.f);
        val.b = std::max(val.b, 0.f);
    break;
    default:
        MNRY_ASSERT(false);
    }

    return val;
}

} //namespace pbr
} //namespace moonray

