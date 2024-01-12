// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0


#include "RayState.isph"

export uniform uint32_t
Subpixel_hvdValidation(uniform bool verbose)
{
    SUBPIXEL_VALIDATION(VLEN);
}

export uniform uint32_t
PathVertex_hvdValidation(uniform bool verbose)
{
    PATH_VERTEX_VALIDATION(VLEN);
}

export uniform uint32_t
RayState_hvdValidation(uniform bool verbose)
{
    RAY_STATE_VALIDATION(VLEN);
}


