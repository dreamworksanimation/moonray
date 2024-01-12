// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0


#include "LightFilter.isph"
#include <scene_rdl2/common/platform/IspcUtil.isph>

//----------------------------------------------------------------------------

ISPC_UTIL_EXPORT_UNIFORM_STRUCT_TO_HEADER(LightFilterList);

export uniform uint32_t
LightFilterList_hudValidation(uniform bool verbose)
{
    LIGHT_FILTER_LIST_VALIDATION;
}

//----------------------------------------------------------------------------

