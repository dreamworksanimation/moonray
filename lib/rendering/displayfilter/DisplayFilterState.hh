// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0


#pragma once

#include <scene_rdl2/common/platform/HybridVaryingData.hh>

#define DISPLAYFILTER_STATE_MEMBERS                     \
    HVD_MEMBER(unsigned int, mOutputPixelX);            \
    HVD_MEMBER(unsigned int, mOutputPixelY);            \
    HVD_MEMBER(unsigned int, mImageWidth);              \
    HVD_MEMBER(unsigned int, mImageHeight)

#define DISPLAYFILTER_STATE_VALIDATION(vlen)            \
    HVD_BEGIN_VALIDATION(DisplayFilterState, vlen);     \
    HVD_VALIDATE(DisplayFilterState, mOutputPixelX);    \
    HVD_VALIDATE(DisplayFilterState, mOutputPixelY);    \
    HVD_VALIDATE(DisplayFilterState, mImageWidth);      \
    HVD_VALIDATE(DisplayFilterState, mImageHeight);     \
    HVD_END_VALIDATION

