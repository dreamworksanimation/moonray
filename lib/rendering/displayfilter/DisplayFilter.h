// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0


#pragma once

#include "DisplayFilterState.hh"
#include "DisplayFilter_ispc_stubs.h"

#include <scene_rdl2/common/platform/IspcUtil.h>
#include <scene_rdl2/scene/rdl2/Types.h>

namespace moonray {
namespace displayfilter {

struct DisplayFilterState
{
    DISPLAYFILTER_STATE_MEMBERS;

    static uint32_t hvdValidation(bool verbose)
    {
        DISPLAYFILTER_STATE_VALIDATION(VLEN);
    }

    void init(unsigned int outputPixelX, unsigned int outputPixelY, unsigned int width, unsigned int height)
    {
        mOutputPixelX = outputPixelX;
        mOutputPixelY = outputPixelY;
        mImageWidth = width;
        mImageHeight = height;
    }
};

ISPC_UTIL_TYPEDEF_STRUCT(DisplayFilterState, DisplayFilterStatev);

// This data is provided to a DisplayFilter's getInputData() function.
struct InitializeData
{
    unsigned int mImageWidth;
    unsigned int mImageHeight;

    InitializeData(unsigned int width, unsigned int height) :
        mImageWidth(width),
        mImageHeight(height)
    {
    }
};

// This data is to be filled in by the DisplayFilter's getInputData() function.
// mInputs:  A vector of the required inputs
// mWindowWidths:  the window width required for each input, in the same order as the inputs
struct InputData
{
    scene_rdl2::rdl2::SceneObjectVector mInputs;
    std::vector<int> mWindowWidths;
};

} // namespace displayfilter
} // namespace moonray

