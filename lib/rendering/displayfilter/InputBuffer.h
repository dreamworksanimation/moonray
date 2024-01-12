// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

//
//

#pragma once

#include "InputBuffer.hh"

#include <scene_rdl2/common/fb_util/VariablePixelBuffer.h>

namespace scene_rdl2 {

namespace math {
    class Color;
}
}

namespace moonray {

namespace displayfilter {

// InputBuffer is a wrapper around scene_rdl2::fb_util::VariablePixelBuffer. It is used as the
// input pixel buffers for DisplayFilter::filter functions.
class InputBuffer
{
public:
    typedef scene_rdl2::fb_util::VariablePixelBuffer VariablePixelBuffer;

    InputBuffer() : mPixelBuffer(nullptr) {}
    ~InputBuffer();
    void init(scene_rdl2::fb_util::VariablePixelBuffer::Format format, unsigned int width, unsigned int height);
    void init(const std::shared_ptr<scene_rdl2::fb_util::VariablePixelBuffer>& buffer);
    void setData(unsigned int offset, unsigned int length, const uint8_t * data);
    // The start pixel is the coordinate of the first pixel (lower left corner) of
    // the input buffer relative to the full image frame buffer.
    // If set to (0,0), then start pixel of InputBuffer is the same as the
    // start pixel of the full image frame.
    void setStartPixel(unsigned int x, unsigned int y);

    // Pixel getters. The (x,y) coordinate is relative to the full image frame.

    // Return Color pixel regardless of underlying pixel buffer format
    const scene_rdl2::math::Color getPixel(unsigned int x, unsigned int y) const;

    // All the following pixel getters require the input buffer format to match the
    // pixel format
    const scene_rdl2::fb_util::ByteColor& getRgb888Pixel(unsigned int x, unsigned int y) const;
    const scene_rdl2::fb_util::ByteColor4& getRgba8888Pixel(unsigned int x, unsigned int y) const;
    float getFloatPixel(unsigned int x, unsigned int y) const;
    const scene_rdl2::math::Vec2f& getFloat2Pixel(unsigned int x, unsigned int y) const;
    const scene_rdl2::math::Vec3f& getFloat3Pixel(unsigned int x, unsigned int y) const;
    const scene_rdl2::math::Vec4f& getFloat4Pixel(unsigned int x, unsigned int y) const;

    unsigned int getWidth() const;
    unsigned int getHeight() const;
    scene_rdl2::fb_util::VariablePixelBuffer::Format getFormat() const;

    const VariablePixelBuffer* getPixelBuffer() const { return mPixelBuffer; }

    static uint32_t hudValidation(bool verbose)
    {
        DISPLAYFILTER_INPUTBUFFER_VALIDATION;
    }

private:
    void reset();

    // shared_ptr for easy cleanup. InputBuffers can share their underlying
    // PixelBuffers.
    std::shared_ptr<scene_rdl2::fb_util::VariablePixelBuffer> mPixelBufferPtr;
    DISPLAYFILTER_INPUTBUFFER_MEMBERS;
};

MNRY_STATIC_ASSERT(sizeof(InputBuffer) % CACHE_LINE_SIZE == 0);

} // namespace displayfilter
} // namespace moonray

