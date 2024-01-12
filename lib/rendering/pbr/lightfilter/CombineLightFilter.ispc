// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

#include "LightFilter.isph"
#include <moonray/rendering/pbr/core/Util.isph>
#include <scene_rdl2/common/platform/IspcUtil.isph>

//----------------------------------------------------------------------------

ISPC_UTIL_EXPORT_UNIFORM_STRUCT_TO_HEADER(CombineLightFilter);

export uniform uint32_t
CombineLightFilter_hudValidation(uniform bool verbose)
{
    COMBINE_LIGHT_FILTER_VALIDATION;
}

//----------------------------------------------------------------------------

varying bool
CombineLightFilter_canIlluminate(const uniform LightFilter * uniform lif,
                                 const varying CanIlluminateData &data)
{
    const uniform CombineLightFilter * uniform lf = (const uniform CombineLightFilter * uniform)lif;
    const uniform LightFilter ** uniform childFilters = (const uniform LightFilter **) lf->mLightFilters;

    for (uniform int i = 0; i < lf->mNumLightFilters; i++) {
        if (LightFilter_canIlluminate(childFilters[i], data)) {
            return true;
        }
    }

    return false;
}

void
CombineLightFilter_eval(const uniform LightFilter * uniform lif,
                        const varying EvalData& data,
                        varying Color * uniform filterValue)
{
    const uniform CombineLightFilter * uniform lf = (const uniform CombineLightFilter * uniform)lif;
    const uniform LightFilter ** uniform childFilters = (const uniform LightFilter **) lf->mLightFilters;

    switch (lf->mMode) {
    case MULTIPLY:
        *filterValue = Color_ctor(1.f);
        for (uniform int i = 0; i < lf->mNumLightFilters; i++) {
            Color childFilterVal;
            LightFilter_eval(childFilters[i], data, &childFilterVal);
            *filterValue = *filterValue * childFilterVal;
        }
    break;
    case MIN:
        *filterValue = Color_ctor(1.f);
        for (uniform int i = 0; i < lf->mNumLightFilters; i++) {
            Color childFilterVal;
            LightFilter_eval(childFilters[i], data, &childFilterVal);
            filterValue->r = min(filterValue->r, childFilterVal.r);
            filterValue->g = min(filterValue->g, childFilterVal.g);
            filterValue->b = min(filterValue->b, childFilterVal.b);
        }
    break;
    case MAX:
        *filterValue = Color_ctor(0.f);
        for (uniform int i = 0; i < lf->mNumLightFilters; i++) {
            Color childFilterVal;
            LightFilter_eval(childFilters[i], data, &childFilterVal);
            filterValue->r = max(filterValue->r, childFilterVal.r);
            filterValue->g = max(filterValue->g, childFilterVal.g);
            filterValue->b = max(filterValue->b, childFilterVal.b);
        }
    break;
    case ADD:
        *filterValue = Color_ctor(0.f);
        for (uniform int i = 0; i < lf->mNumLightFilters; i++) {
            Color childFilterVal;
            LightFilter_eval(childFilters[i], data, &childFilterVal);
            *filterValue = *filterValue + childFilterVal;
        }
        filterValue->r = min(filterValue->r, 1.f);
        filterValue->g = min(filterValue->g, 1.f);
        filterValue->b = min(filterValue->b, 1.f);
    break;
    case SUBTRACT:
        *filterValue = Color_ctor(1.f);
        for (uniform int i = 0; i < lf->mNumLightFilters; i++) {
            Color childFilterVal;
            LightFilter_eval(childFilters[i], data, &childFilterVal);
            if (i == 0) {
                *filterValue = childFilterVal;
            } else {
                *filterValue = *filterValue - childFilterVal;
            }
        }
        filterValue->r = max(filterValue->r, 0.f);
        filterValue->g = max(filterValue->g, 0.f);
        filterValue->b = max(filterValue->b, 0.f);
    break;
    default:
        MNRY_ASSERT(false);
    }
}

//----------------------------------------------------------------------------

export void
CombineLightFilter_init(uniform CombineLightFilter * uniform lightfilter)
{
    LightFilter_init((uniform LightFilter * uniform)lightfilter, 
            CombineLightFilter_canIlluminate, CombineLightFilter_eval);
}

//----------------------------------------------------------------------------

