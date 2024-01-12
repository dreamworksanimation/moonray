// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0


#include "Ray.isph"

export uniform uint32_t
RayExtension_hvdValidation(uniform bool verbose)
{
    MCRT_COMMON_RAY_EXTENSION_VALIDATION(VLEN);
}

export uniform uint32_t
RayDifferential_hvdValidation(uniform bool verbose)
{
    MCRT_COMMON_RAY_DIFFERENTIAL_VALIDATION(VLEN);
}

