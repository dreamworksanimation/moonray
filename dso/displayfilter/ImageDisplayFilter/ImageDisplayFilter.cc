// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0


#include "attributes.cc"
#include "ImageDisplayFilter_ispc_stubs.h"

#include <moonray/rendering/displayfilter/DisplayFilter.h>
#include <moonray/rendering/displayfilter/InputBuffer.h>

#include <scene_rdl2/common/math/MathUtil.h>
#include <scene_rdl2/scene/rdl2/rdl2.h>

// Make sure atomic<float> template specialization is defined
// before including OIIO headers
#include <scene_rdl2/render/util/AtomicFloat.h>
#include <OpenImageIO/imagebuf.h>
#include <OpenImageIO/imagebufalgo.h>
#include <OpenImageIO/imagecache.h>
#include <OpenImageIO/imageio.h>

using namespace moonray;
using namespace scene_rdl2::math;

#define HORIZONTAL_SCALE 0
#define VERTICAL_SCALE 1

RDL2_DSO_CLASS_BEGIN(ImageDisplayFilter, DisplayFilter)

public:
    ImageDisplayFilter(const SceneClass& sceneClass, const std::string& name);
    ~ImageDisplayFilter();

    virtual void update() override;

private:
    virtual void getInputData(const displayfilter::InitializeData& initData,
                              displayfilter::InputData& inputData) const override;
    bool loadImageFile(int w, int h);
    void resizeImage(OIIO::ImageBuf &src, int w, int h);
    void stretchAndSave(OIIO::ImageBuf &src, int w, int h);
    void fitAndSave(OIIO::ImageBuf &src, int w, int h, int imgWidth, int imgHeight, int fitMode);
    ispc::DisplayType getDisplayType(int w, int h, int imgWidth, int imgHeight);

    ispc::ImageDisplayFilter mIspc;
    std::vector<float> mPixelBuf;           // buffer to temporarily hold pixel data

    OIIO::ImageCache* mImageCache;

RDL2_DSO_CLASS_END(ImageDisplayFilter)

//---------------------------------------------------------------------------

ImageDisplayFilter::ImageDisplayFilter(
        const SceneClass& sceneClass, const std::string& name) :
    Parent(sceneClass, name), mPixelBuf{}
{
    mFilterFuncv = (DisplayFilterFuncv) ispc::ImageDisplayFilter_getFilterFunc();

    mIspc.mDisplayType = static_cast<ispc::DisplayType>(0); 
    mIspc.mPixels = nullptr;
    mIspc.mChannels = 0;
    mIspc.mMask = false;
    mIspc.mInvertMask = false;
    mIspc.mMix = 0.f;
    mIspc.mRenderWidth = 0;
    mIspc.mRenderHeight = 0;

    // We need to create a separate, non-shared image cache or by default OIIO will
    // use the shared one created by MoonRay's texture sampler (which doesn't allow
    // non-tiled or non-mipped images.)
    mImageCache = OIIO::ImageCache::create(false);
}

ImageDisplayFilter::~ImageDisplayFilter()
{
    OIIO::ImageCache::destroy(mImageCache);
}

/** Used for cases where we want to stretch the image to fill a plane of width x height
 * @param src - reference to image buffer that contains the image data to be processed
 * @param width, @param height - render dimensions
 */
void ImageDisplayFilter::stretchAndSave(OIIO::ImageBuf &src, int width, int height) 
{
    // resize image to given dimensions
    OIIO::ROI roi (0, width, 0, height, 0, 1, 0, src.nchannels());
    OIIO::ImageBuf resized = OIIO::ImageBufAlgo::resize(src, "", 0, roi);

    // save pointer to float buffer
    resized.get_pixels(roi, OIIO::TypeDesc::FLOAT, &mPixelBuf[0]);
    mIspc.mPixels = mPixelBuf.data();
}

/** Uniformly scale the image so that it fits the desired dimension, and crop so that it's
 *  the render height/width
 *  @param src - buffer that contains the image data to be processed
 *  @param width, @param height - render dimensions
 *  @param imgWidth, @param imgHeight - loaded image dimensions
 */
void ImageDisplayFilter::fitAndSave(OIIO::ImageBuf &src, int width, int height,
                                    int imgWidth, int imgHeight, int fitMode) 
{
    // find the horizontally/vertically scaled-up (or down) dimensions
    float scale = fitMode == HORIZONTAL_SCALE ? (float) width / imgWidth 
                                              : (float) height / imgHeight;
    int xDim = scale * imgWidth;
    int yDim = scale * imgHeight;

    // offsets to center the cropped image
    int xOffset = (xDim - width) / 2;
    int yOffset = (yDim - height) / 2; 

    // uniformly scale the image so that it fits the desired horizontal/vertical dimension
    OIIO::ROI roi_resize(0, xDim, 0, yDim, 0, 1, 0, src.nchannels());
    OIIO::ImageBuf resized = OIIO::ImageBufAlgo::fit(src, "", 0, false, roi_resize);

    // only use the sub-image of render widthxheight dimensions (centered)
    OIIO::ROI roi_crop (xOffset, width + xOffset, yOffset, height + yOffset, 0, 1, 0, src.nchannels());

    // save result to pixel buffer
    resized.get_pixels(roi_crop, OIIO::TypeDesc::FLOAT, &mPixelBuf[0]);
    mIspc.mPixels = mPixelBuf.data();
}

// Simplify the display type to four possibilites: STRETCH, FIT_HORIZONTAL, FIT_VERTICAL, NO_SCALE
ispc::DisplayType ImageDisplayFilter::getDisplayType(int width, int height, 
                                                     int imgWidth, int imgHeight) 
{
    ispc::DisplayType displayType = static_cast<ispc::DisplayType>(get(attrDisplayType));

    // if same aspect ratio, just use stretch method
    if ((imgWidth / (float) width) == (imgHeight / (float) height)) {
        return ispc::STRETCH;
    } 

    switch (displayType) {
        case ispc::FIT_BY_SMALLEST_DIM:
            if (imgWidth < imgHeight) return ispc::FIT_HORIZONTAL;
            if (imgWidth > imgHeight) return ispc::FIT_VERTICAL;
            // if img width/height are the same, just fit to render dimensions
            else return width < height ? ispc::FIT_HORIZONTAL : ispc::FIT_VERTICAL;

        case ispc::FIT_BY_LARGEST_DIM:
            if (imgHeight < imgWidth) return ispc::FIT_HORIZONTAL;
            if (imgHeight > imgWidth) return ispc::FIT_VERTICAL;
            // if img width/height are the same, just fit to render dimensions
            else return width < height ? ispc::FIT_HORIZONTAL : ispc::FIT_VERTICAL;

        default:
            return displayType;
    }
}

// Resize the input image and save to ispc float buffer
void ImageDisplayFilter::resizeImage(OIIO::ImageBuf &src, int width, int height) 
{
    int imgWidth = src.xend(), imgHeight = src.yend();
    mIspc.mDisplayType = getDisplayType(width, height, imgWidth, imgHeight); 

    mPixelBuf.clear();
    mPixelBuf.reserve(width*height*src.nchannels());

    switch (mIspc.mDisplayType) {
        case ispc::NO_SCALE:
            // just display top left widthxheight pixels
            src.get_pixels(OIIO::ROI(0, width, 0, height), OIIO::TypeDesc::FLOAT, &mPixelBuf[0]);
            mIspc.mPixels = mPixelBuf.data();
            break;
        case ispc::STRETCH:
            stretchAndSave(src, width, height);
            break;
        case ispc::FIT_HORIZONTAL:
            fitAndSave(src, width, height, imgWidth, imgHeight, HORIZONTAL_SCALE);
            break;
        case ispc::FIT_VERTICAL:
            fitAndSave(src, width, height, imgWidth, imgHeight, VERTICAL_SCALE);
            break;
        default:
            mIspc.mPixels = nullptr;
            break;
    }
}

// Process the image file and save it to a pixel buffer
bool ImageDisplayFilter::loadImageFile(int renderWidth, int renderHeight) 
{
    // read in file
    std::string filename = get(attrImagePath);
    OIIO::ImageBuf src(filename, 0, 0, mImageCache);

    if (!src.read(0, 0, true, OIIO::TypeDesc::FLOAT)){
        return false;
    }

    mIspc.mChannels = src.nchannels();
    resizeImage(src, renderWidth, renderHeight);

    return true;
}

void ImageDisplayFilter::update() 
{
    if (get(attrInput) == nullptr) {
        fatal("Missing \"input\" attribute");
        return;
    }
    if (get(attrImagePath) == "") {
        fatal("Missing \"image_path\" attribute");
        return;
    }
    // get width/height of the render output
    const SceneContext *ctx = getSceneClass().getSceneContext();
    MNRY_ASSERT_REQUIRE(ctx);
    mIspc.mRenderWidth = ctx->getSceneVariables().getRezedWidth();
    mIspc.mRenderHeight = ctx->getSceneVariables().getRezedHeight();

    if (!loadImageFile(mIspc.mRenderWidth, mIspc.mRenderHeight)){
        fatal("Failed to read file");
        return;
    }

    mIspc.mMask = get(attrMask) == nullptr ? false : true;
    mIspc.mInvertMask = get(attrInvertMask);
    mIspc.mMix = saturate(get(attrMix));
}

void
ImageDisplayFilter::getInputData(const displayfilter::InitializeData& initData,
                                 displayfilter::InputData& inputData) const
{
    inputData.mInputs.push_back(get(attrInput));
    inputData.mWindowWidths.push_back(1);

    // mask
    if (get(attrMask) != nullptr) {
        inputData.mInputs.push_back(get(attrMask));
        inputData.mWindowWidths.push_back(1);
    }
}

//---------------------------------------------------------------------------

