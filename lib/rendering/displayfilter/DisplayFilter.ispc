// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

/// @file DisplayFilter.isph

#include "DisplayFilter.isph"

#include <scene_rdl2/common/platform/IspcUtil.isph>

#pragma ignore warning(all)
ISPC_UTIL_EXPORT_STRUCT_TO_HEADER(DisplayFilterState);

varying float
DISPLAYFILTER_mixAndMask(varying float mix,
                         const uniform InputBuffer const * uniform mask,
                         varying unsigned int x,
                         varying unsigned int y,
                         uniform bool invert)
{
    if (isZero(mix)) {
        return mix;
    }
    if (mask != nullptr) {
        const Color maskPixel = InputBuffer_getPixel(mask, x, y);
        float maskValue = (maskPixel.r + maskPixel.g + maskPixel.b) / 3.0f;
        if (invert) {
            maskValue = 1 - maskValue;
        }
        mix = mix * maskValue;
    }
    return mix;
}

