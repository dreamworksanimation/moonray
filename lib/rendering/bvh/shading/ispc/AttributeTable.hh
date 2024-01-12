// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

/// @file AttributeTable.hh
#pragma once

#include <scene_rdl2/common/platform/HybridVaryingData.hh>

#define SHADING_ATTRIBUTE_TABLE_MEMBERS                                 \
    HVD_MEMBER(int32_t, mNumKeys);                                      \
    HVD_MEMBER(int32_t, mTotalSize);                                    \
    HVD_PTR(int32_t *, mKeyOffset);                                     \
    /* sizeof(std::vector<AttributeKey>) == 24 */                       \
    HVD_CPP_MEMBER(std::vector<AttributeKey>, mRequiredAttributes, 24); \
    HVD_CPP_MEMBER(std::vector<AttributeKey>, mOptionalAttributes, 24); \
    HVD_CPP_MEMBER(std::vector<AttributeKey>, mDifferentialAttributes, 24)

#define SHADING_ATTRIBUTE_TABLE_VALIDATION(vlen)                \
    HVD_BEGIN_VALIDATION(AttributeTable, vlen);                 \
    HVD_VALIDATE(AttributeTable, mNumKeys);                     \
    HVD_VALIDATE(AttributeTable, mKeyOffset);                   \
    HVD_VALIDATE(AttributeTable, mTotalSize);                   \
    HVD_VALIDATE(AttributeTable, mRequiredAttributes);          \
    HVD_VALIDATE(AttributeTable, mOptionalAttributes);          \
    HVD_VALIDATE(AttributeTable, mDifferentialAttributes);      \
    HVD_END_VALIDATION

