// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

#include <moonray/rendering/displayfilter/DisplayFilter.isph>

enum DisplayType {
    STRETCH               = 0,
    FIT_HORIZONTAL        = 1,
    FIT_VERTICAL          = 2,
    FIT_BY_SMALLEST_DIM   = 3,
    FIT_BY_LARGEST_DIM    = 4,
    NO_SCALE              = 5
};

struct ImageDisplayFilter
{
    DisplayType mDisplayType;
    float* mPixels;
    int mChannels;
    bool  mMask;
    bool  mInvertMask;
    float mMix;
    int mRenderWidth;   // using this because state->mImageWidth is the padded width
    int mRenderHeight;  // using this because state->mImageHeight is the padded height
};

export const uniform ImageDisplayFilter * uniform
ImageDisplayFilter_get(const uniform DisplayFilter * uniform displayFilter)
{
    return DISPLAYFILTER_GET_ISPC_CPTR(ImageDisplayFilter, displayFilter);
}

static void
filter(const uniform DisplayFilter * uniform me,
       const uniform InputBuffer * const uniform * const uniform inputBuffers,
       const varying DisplayFilterState * const uniform state,
       varying Color * uniform result)
{
    const uniform ImageDisplayFilter * uniform self = ImageDisplayFilter_get(me);
    const uniform InputBuffer const * uniform inBuffer = inputBuffers[0];
    int x = state->mOutputPixelX, y = state->mOutputPixelY;
    varying Color src = InputBuffer_getPixel(inBuffer, x, y);

    float mix = DISPLAYFILTER_mixAndMask(self->mMix,
                                         self->mMask ? inputBuffers[1] : nullptr,
                                         state->mOutputPixelX,
                                         state->mOutputPixelY,
                                         self->mInvertMask);
    if (isZero(mix)) {
        *result = src;
        return;
    }

    // if the file couldn't be read, just display the original image
    if (self->mPixels == nullptr) {
        *result = src;
    } else {
        // img is flipped in buffer
        y = self->mRenderHeight - y - 1;
        varying int blockStart = x * self->mChannels + y * self->mRenderWidth * self->mChannels;
        varying Color imgCol = Color_ctor(self->mPixels[blockStart], 
                                          self->mPixels[blockStart + 1], 
                                          self->mPixels[blockStart + 2]); 
        *result = imgCol;
    }

    if (!isOne(mix)) {
        *result = lerp(src, *result, mix);
    }
}

DEFINE_DISPLAY_FILTER(ImageDisplayFilter, filter)

//---------------------------------------------------------------------------

