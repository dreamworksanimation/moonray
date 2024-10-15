// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0


#include "LightFilter.isph"
#include <scene_rdl2/common/platform/IspcUtil.isph>

//----------------------------------------------------------------------------

ISPC_UTIL_EXPORT_UNIFORM_STRUCT_TO_HEADER(IntensityLightFilter);

export uniform uint32_t
IntensityLightFilter_hudValidation(uniform bool verbose)
{
    INTENSITY_LIGHT_FILTER_VALIDATION;
}


//----------------------------------------------------------------------------

varying bool
IntensityLightFilter_canIlluminate(const uniform LightFilter * uniform lif,
        const varying CanIlluminateData &data)
{
    const uniform IntensityLightFilter * uniform lightfilter = (const uniform IntensityLightFilter * uniform)lif;

    return !isBlack(lightfilter->mRadianceMod);
}

void
IntensityLightFilter_eval(const uniform LightFilter * uniform lif,
        const varying EvalData& data,
        varying Color * uniform filterValue)
{
    const uniform IntensityLightFilter * uniform lightfilter = (const uniform IntensityLightFilter * uniform)lif;

    *filterValue = lightfilter->mRadianceMod;
}

//----------------------------------------------------------------------------

export void
IntensityLightFilter_init(uniform IntensityLightFilter * uniform lightfilter)
{
    LightFilter_init((uniform LightFilter * uniform)lightfilter,
            IntensityLightFilter_canIlluminate, IntensityLightFilter_eval);
}


//----------------------------------------------------------------------------

