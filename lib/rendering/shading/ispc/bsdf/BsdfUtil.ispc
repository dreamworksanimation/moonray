// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

/// @file BsdfUtil.ispc


#include "BsdfUtil.isph"

void
BsdfUtil_indent(const uniform uint8_t ntab)
{
    for (uniform uint8_t i = 0; i < ntab; ++i) {
        print("    ");
    }
}



