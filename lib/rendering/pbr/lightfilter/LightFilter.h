// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

//


#pragma once

#include "LightFilter.hh"

#include <moonray/rendering/pbr/light/LightUtil.h>

#include <moonray/rendering/mcrt_common/ThreadLocalState.h>
#include <moonray/rendering/pbr/lightfilter/LightFilterList_ispc_stubs.h>

#include <scene_rdl2/scene/rdl2/LightFilter.h>
#include <scene_rdl2/common/math/Color.h>
#include <scene_rdl2/common/math/Mat4.h>
#include <scene_rdl2/common/math/Vec3.h>
#include <scene_rdl2/common/platform/HybridUniformData.h>

// Forward declaration of the ISPC types
namespace ispc {
    struct LightFilter;
}

namespace moonray {
namespace pbr {

struct LightFilterRandomValues
{
    scene_rdl2::math::Vec2f r2;
    scene_rdl2::math::Vec3f r3;
};

class LightFilter;
typedef std::unordered_map<const scene_rdl2::rdl2::LightFilter*, std::unique_ptr<LightFilter>> LightFilterMap;


class LightFilter
{
public:
    LightFilter() :
        mCanIlluminateFn(nullptr),
        mEvalFn(nullptr),
        mRdlLightFilter(nullptr) {}
    LightFilter(const scene_rdl2::rdl2::LightFilter* rdlLightFilter) :
        mCanIlluminateFn(nullptr),
        mEvalFn(nullptr),
        mRdlLightFilter(rdlLightFilter){}

    virtual ~LightFilter() {}

    /// HUD validation and type casting
    static uint32_t hudValidation(bool verbose) {
        LIGHT_FILTER_VALIDATION;
    }
    HUD_AS_ISPC_METHODS(LightFilter);

    virtual void update(const LightFilterMap& lightFilters,
                        const scene_rdl2::math::Mat4d& world2render) = 0;

    struct CanIlluminateData
    {
        scene_rdl2::math::Vec3f lightPosition;
        float lightRadius;
        scene_rdl2::math::Vec3f shadingPointPosition;
        scene_rdl2::math::Xform3f lightRender2LocalXform;
        float shadingPointRadius;
        float time;
    };

    struct EvalData
    {
        mcrt_common::ThreadLocalState* tls;
        const LightIntersection* isect;
        scene_rdl2::math::Vec3f lightPosition;
        scene_rdl2::math::Vec3f lightDirection;
        scene_rdl2::math::Vec3f shadingPointPosition;
        LightFilterRandomValues randVar;
        float time;
        scene_rdl2::math::Xform3f lightRender2LocalXform;
        scene_rdl2::math::Vec3f wi;   // direction of incoming light
    };

    virtual bool canIlluminate(const CanIlluminateData& data) const = 0;
    virtual scene_rdl2::math::Color eval(const EvalData& data) const = 0;
    virtual bool needsLightXform() const { return false; }
    virtual bool needsSamples() const { return false; }

protected:

    LIGHT_FILTER_MEMBERS;

private:
    /// Copy is disabled
    LightFilter(const LightFilter &other);
    const LightFilter &operator=(const LightFilter &other);
};

class LightFilterList
{
public:
    LightFilterList() :
        mLightFilters(nullptr),
        mLightFilterCount(0),
        mNeedsLightXform(false) {}

    /// HUD validation and type casting
    static uint32_t hudValidation(bool verbose) {
        LIGHT_FILTER_LIST_VALIDATION;
    }

    void init(std::unique_ptr<const LightFilter* []>&& lightFilters, int count)
    {
        // LightFilterList now owns the array of LightFilters.
        mLightFilters = std::move(lightFilters);
        mLightFilterCount = count;
        mNeedsLightXform = false;

        for (int i = 0; i < mLightFilterCount; i++) {
            if (mLightFilters[i]->needsLightXform()) {
                mNeedsLightXform = true;
                break;
            }
        }
    }

    int getLightFilterCount() const { return mLightFilterCount; }
    const LightFilter *getLightFilter(int idx) const { return mLightFilters[idx]; }
    bool needsLightXform() const { return mNeedsLightXform; }

private:
    LIGHT_FILTER_LIST_MEMBERS;
};

HUD_VALIDATOR(LightFilterList);

inline bool canIlluminateLightFilterList(const LightFilterList* lightFilterList,
                                         const LightFilter::CanIlluminateData& data)
{
    MNRY_ASSERT(lightFilterList);
    size_t lightFilterCount = lightFilterList->getLightFilterCount();
    for (size_t i = 0; i < lightFilterCount; ++i) {
        const LightFilter *lightFilter = lightFilterList->getLightFilter(i);
        MNRY_ASSERT(lightFilter);
        if (!lightFilter->canIlluminate(data)) {
            return false;
        }
    }
    return true;
}

inline void evalLightFilterList(const LightFilterList* lightFilterList,
                                const LightFilter::EvalData& data,
                                scene_rdl2::math::Color& radiance)
{
    MNRY_ASSERT(lightFilterList);
    size_t lightFilterCount = lightFilterList->getLightFilterCount();
    for (size_t i = 0; i < lightFilterCount; ++i) {
        const LightFilter *lightFilter = lightFilterList->getLightFilter(i);
        MNRY_ASSERT(lightFilter);
        radiance *= lightFilter->eval(data);
    }
}

} //namespace pbr
} //namespace moonray

