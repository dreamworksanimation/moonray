// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

/// @file AttributeTable.ispc

#include "AttributeTable.isph"

export uniform uint32_t
AttributeTable_hvdValidation(uniform bool verbose)
{
    SHADING_ATTRIBUTE_TABLE_VALIDATION(VLEN);
}

