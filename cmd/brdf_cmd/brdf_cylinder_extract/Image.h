// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <scene_rdl2/common/math/Color.h>

#include <string>
#include <cstring>


//---------------------------------------------------------------------------

finline float flerp(float a, float b, float t) { return (1.0f - t) * a + t * b; }

//---------------------------------------------------------------------------

class Image {
public:
    Image() :
        mWidth(0),
        mHeight(0),
        mChannels(0),
        mPixels(nullptr) {}
    Image(int width, int height, int channels) :
        mWidth(width),
        mHeight(height),
        mChannels(channels) {
        const int floatCount = mWidth * mHeight * mChannels;
        mPixels = new float[floatCount];
        memset(mPixels, 0, floatCount * sizeof(float));
    }
    Image(const std::string &filename);
    ~Image() {
        clear();
    }

    Image(const Image &other) :
        mWidth(other.mWidth),
        mHeight(other.mHeight),
        mChannels(other.mChannels) {
        const int floatCount = mWidth * mHeight * mChannels;
        mPixels = new float[floatCount];
        memcpy(mPixels, other.mPixels, floatCount * sizeof(float));
    }
    Image& operator=(const Image &other) {
        if (this != &other) {
            clear();
            mWidth = other.mWidth;
            mHeight = other.mHeight;
            mChannels = other.mChannels;
            const int floatCount = mWidth * mHeight * mChannels;
            mPixels = new float[floatCount];
            memcpy(mPixels, other.mPixels, floatCount * sizeof(float));
        }
        return *this;
    }

    void saveAs(const std::string &filename) const;

    // Accessors
    int getWidth() const    {  return mWidth;  }
    int getHeight() const   {  return mHeight;  }

    // Get / Set pixels
    finline int getPixelIndex(int x, int y) const  {  return (y * mWidth + x) * mChannels;  }
    finline scene_rdl2::math::Color getPixel(int index) const  {
        MNRY_ASSERT(mChannels == 3  ||  mChannels == 1);
        return (mChannels == 3  ?
                scene_rdl2::math::Color(mPixels[index], mPixels[index+1], mPixels[index+2]) :
                scene_rdl2::math::Color(mPixels[index], mPixels[index], mPixels[index]));
    }
    finline void setPixel(int index, const scene_rdl2::math::Color &color) {
        MNRY_ASSERT(mChannels == 3  ||  mChannels == 1);
        mPixels[index++] = color.r;
        if (mChannels == 3) {
            mPixels[index++] = color.g;
            mPixels[index++] = color.b;
        }
    }

    // Sample color
    scene_rdl2::math::Color sampleColor(int xi, float y) const;

    // Draw a pixel
    void drawPixel(int xi, float y, const scene_rdl2::math::Color &color);


    // Apply a sobel operator to this image and store result in result image
    // (any previous content in result is cleared). The sobel filter is applied
    // to the luminance of this image and therefore the result is monochrome
    void sobel(Image &result) const;

    // Subtract other image from this image. We assume image properties match
    void subtract(const Image &other);

private:

    void clear() {
        mWidth = 0;
        mHeight = 0;
        delete [] mPixels;
        mPixels = nullptr;
    }


    int mWidth;
    int mHeight;
    int mChannels;
    float *mPixels;
};


//---------------------------------------------------------------------------

