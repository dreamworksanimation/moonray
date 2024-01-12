// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

// Math type conversion functions that we can't include in OptixGPUMath.h
// because they would pull in the regular MoonRay headers into the GPU-only code.

#pragma once

#include "OptixGPUMath.h"
#include <moonray/rendering/geom/Types.h>

namespace moonray {
namespace rt {

inline OptixGPUXform
mat43ToOptixGPUXform(const geom::Mat43& mat)
{
    // Note that Mat43 is a 4x3 matrix with the translation as the bottom row,
    // while OptixGPUXform is a 3x4 matrix with the translation as the last column.
    // The core 3x3 part of the matrix is transposed.
    OptixGPUXform xf;
    xf.m[0][0] = mat.row0().x;
    xf.m[0][1] = mat.row1().x;
    xf.m[0][2] = mat.row2().x;
    xf.m[1][0] = mat.row0().y;
    xf.m[1][1] = mat.row1().y;
    xf.m[1][2] = mat.row2().y;
    xf.m[2][0] = mat.row0().z;
    xf.m[2][1] = mat.row1().z;
    xf.m[2][2] = mat.row2().z;
    // translation part
    xf.m[0][3] = mat.row3().x;
    xf.m[1][3] = mat.row3().y;
    xf.m[2][3] = mat.row3().z;

    return xf;
}

} // namespace rt
} // namespace moonray

