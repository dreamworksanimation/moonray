// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

#include "Image.h"

#include <scene_rdl2/common/except/exceptions.h>

#include <OpenImageIO/imageio.h>
#include <OpenImageIO/imagebuf.h>
#include <OpenImageIO/imagebufalgo.h>

#include <string>


using namespace scene_rdl2::math;


//---------------------------------------------------------------------------

Image::Image(const std::string &filename)
{
    auto in = OIIO::ImageInput::open(filename);
    if (!in) {
        throw scene_rdl2::except::IoError("Cannot find or open image file: \"" + filename +
                  "\" (" + in->geterror() + ")");
    }
 
    auto imageSpec = in->spec();

    if (imageSpec.nchannels != 3) {
        throw scene_rdl2::except::FormatError("Cannot handle images that are not 3-channels (RGB)");
    }

    mWidth = imageSpec.width;
    mHeight = imageSpec.height;
    mChannels = imageSpec.nchannels;
    const int floatCount = mWidth * mHeight * mChannels;
    mPixels = new float[floatCount];
    float *tmpBuffer = new float[floatCount];

    if (!in->read_image(OIIO::TypeDesc::FLOAT, tmpBuffer)) {
        delete [] tmpBuffer;
        throw scene_rdl2::except::IoError("Cannot read image file: \"" + filename +
                  "\" (" + in->geterror() + ")");
    }

    // Flip the buffer: our frame buffers are upside down compared to what
    // OIIO is giving us.
    imageSpec.format = OIIO::TypeDesc::FLOAT;
    OIIO::ImageBuf source("source", imageSpec, tmpBuffer);
    OIIO::ImageBuf flipped("flipped", imageSpec, mPixels);
    if (!OIIO::ImageBufAlgo::flip(flipped, source)) {
        delete [] tmpBuffer;
        throw scene_rdl2::except::IoError("Cannot flip image");
    }
    delete [] tmpBuffer;
}


void
Image::saveAs(const std::string &filename) const
{
    // Create an OIIO output buffer.
    auto out(OIIO::ImageOutput::create(filename.c_str()));
    if (!out) {
        std::stringstream errMsg;
        errMsg << "Failed to open " << filename << " for writing.";
        throw scene_rdl2::except::IoError(errMsg.str());
    }

    // Define the image spec.
    OIIO::ImageSpec spec(mWidth, mHeight, mChannels, OIIO::TypeDesc::FLOAT);

    // Flip the buffer: our frame buffers are upside down compared to what
    // OIIO is expecting.
    OIIO::ImageBuf srcBuffer(filename, spec, mPixels);
    OIIO::ImageBuf flippedBuffer(filename, spec);
    OIIO::ImageBufAlgo::flip(flippedBuffer, srcBuffer);

    // Open the output file, write the image buffer into it, and close it.
    if (!out->open(filename.c_str(), spec)) {
        throw scene_rdl2::except::IoError("Cannot open image file: \"" + filename +
                  "\" (" + out->geterror() + ")");
    }
    if (!flippedBuffer.write(out.get())) {
        throw scene_rdl2::except::IoError("Cannot write image to file: \"" + filename +
                  "\" (" + out->geterror() + ")");
    }
}


Color
Image::sampleColor(int xi, float y) const
{
    MNRY_ASSERT(y > 0.5f);

    int yi = int(scene_rdl2::math::floor(y - 0.5f));
    float weight = 1.0f - (y - (yi + 0.5f));
    MNRY_ASSERT(weight >= 0.0f  &&  weight <= 1.0f);
    int index1 = getPixelIndex(xi, yi);
    int index2 = getPixelIndex(xi, yi + 1);

    Color c = weight * getPixel(index1) + (1.0f - weight) * getPixel(index2);

    return c;
}


void
Image::drawPixel(int xi, float y, const Color &color)
{
    MNRY_ASSERT(y > 0.5f);

    int yi = int(scene_rdl2::math::floor(y - 0.5f));
    float weight = 1.0f - (y - (yi + 0.5f));
    MNRY_ASSERT(weight >= 0.0f  &&  weight <= 1.0f);
    int index1 = getPixelIndex(xi, yi);
    int index2 = getPixelIndex(xi, yi + 1);

    Color c;
    c = weight * color + (1.0f - weight) * getPixel(index1);
    setPixel(index1, c);
    c = (1.0f - weight) * color + weight * getPixel(index2);
    setPixel(index2, c);
}


void
Image::sobel(Image &result) const
{
    static const int filterSize = 3;
    static const float filterX[filterSize][filterSize] = {
            {-1.0f, 0.0f, 1.0f},
            {-2.0f, 0.0f, 2.0f},
            {-1.0f, 0.0f, 1.0f},
    };
    static const float filterY[filterSize][filterSize] = {
            { 1.0f,  2.0f,  1.0f},
            { 0.0f,  0.0f,  0.0f},
            {-1.0f, -2.0f, -1.0f},
    };

    // Prep result image
    result.clear();
    result.mWidth = mWidth;
    result.mHeight = mHeight;
    result.mChannels = mChannels;
    const int floatCount = result.mWidth * result.mHeight * result.mChannels;
    result.mPixels = new float[floatCount];
    memset(result.mPixels, 0, floatCount * sizeof(float));

    // For each pixel in the image except edge pixels
    const int xMax = mWidth - 1;
    const int yMax = mHeight - 1;
    Color gMax(zero);
    for (int y = 1; y < yMax; y++) {
        for (int x = 1; x < xMax; x++) {

            // Apply sobel filter in X and Y
            Color gx(zero);
            Color gy(zero);
            for (int j = 0; j < filterSize; j++) {
                int indexSrc = getPixelIndex(x-1, y-1 + j);
                for (int i = 0; i < filterSize; i++) {
                    Color c = getPixel(indexSrc);
                    c = Color(luminance(c));
                    gx += c * filterX[j][i];
                    gy += c * filterY[j][i];
                    indexSrc += mChannels;
                }
            }

            // Compute magnitude
            Color g = sqrt(gx * gx + gy * gy);
            gMax = max(gMax, g);

            // Store result
            const int indexResult = result.getPixelIndex(x, y);
            result.setPixel(indexResult, g);
        }
    }

    // Normalize result image
    const int size = result.mWidth * result.mHeight * result.mChannels;
    for (int index = 0; index < size; index += result.mChannels) {
        const Color c = result.getPixel(index) / gMax;
        result.setPixel(index, c);
    }
}


void
Image::subtract(const Image &other)
{
    if (mWidth != other.mWidth  ||  mHeight != other.mHeight  ||
            mChannels != other.mChannels) {
        throw scene_rdl2::except::FormatError("Cannot subtract images that have different sizes / number of channels");
    }

    const int floatCount = mWidth * mHeight * mChannels;

    for (int i = 0; i < floatCount; i++) {
        mPixels[i] -= other.mPixels[i];
    }
}


//---------------------------------------------------------------------------

