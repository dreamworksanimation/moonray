// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

//
//

#include "InputBuffer.h"
#include <scene_rdl2/common/math/Color.h>

#include <cstring>

namespace moonray {
namespace displayfilter {

InputBuffer::~InputBuffer()
{
    reset();
}

void
InputBuffer::init(scene_rdl2::fb_util::VariablePixelBuffer::Format format, unsigned int width, unsigned int height)
{
    reset();

    mPixelBufferPtr = std::make_shared<scene_rdl2::fb_util::VariablePixelBuffer>();
    mPixelBuffer = mPixelBufferPtr.get();
    mPixelBufferPtr->init(format, width, height);
}

void
InputBuffer::init(const std::shared_ptr<scene_rdl2::fb_util::VariablePixelBuffer>& buffer)
{
    reset();
    mPixelBufferPtr = buffer;
    mPixelBuffer = mPixelBufferPtr.get();
}

void
InputBuffer::reset()
{
    mPixelBufferPtr.reset();
    mPixelBuffer = nullptr;
}

void
InputBuffer::setData(unsigned int offset, unsigned int length, const uint8_t * data)
{
    unsigned sizeOfPixel = mPixelBuffer->getSizeOfPixel();
    MNRY_ASSERT(offset + length < mPixelBuffer->getArea() * sizeOfPixel);
    uint8_t * bufferData = mPixelBuffer->getData();
    memcpy(bufferData + offset * sizeOfPixel, data, length * sizeOfPixel);
}

void
InputBuffer::setStartPixel(unsigned int x, unsigned int y)
{
    mStartX = x;
    mStartY = y;
}

const scene_rdl2::math::Color
InputBuffer::getPixel(unsigned int x, unsigned int y) const
{
    scene_rdl2::fb_util::VariablePixelBuffer::Format format = getFormat();
    switch (format) {
    case scene_rdl2::fb_util::VariablePixelBuffer::RGB888:
    {
        const scene_rdl2::fb_util::ByteColor& val = getRgb888Pixel(x, y);
        return scene_rdl2::math::Color(val.r, val.g, val.b);
    }
    case scene_rdl2::fb_util::VariablePixelBuffer::RGBA8888:
    {
        const scene_rdl2::fb_util::ByteColor4& val = getRgba8888Pixel(x, y);
        return scene_rdl2::math::Color(val.r, val.g, val.b);
    }
    case scene_rdl2::fb_util::VariablePixelBuffer::FLOAT:
    {
        const float val = getFloatPixel(x, y);
        return scene_rdl2::math::Color(val, val, val);
    }
    case scene_rdl2::fb_util::VariablePixelBuffer::FLOAT2:
    {
        const scene_rdl2::math::Vec2f& val = getFloat2Pixel(x, y);
        return scene_rdl2::math::Color(val.x, val.y, 0.f);
    }
    case scene_rdl2::fb_util::VariablePixelBuffer::FLOAT3:
    {
        const scene_rdl2::math::Vec3f& val = getFloat3Pixel(x, y);
        return scene_rdl2::math::Color(val.x, val.y, val.z);
    }
    case scene_rdl2::fb_util::VariablePixelBuffer::FLOAT4:
    {
        const scene_rdl2::math::Vec4f& val = getFloat4Pixel(x, y);
        return scene_rdl2::math::Color(val.x, val.y, val.z);
    }
    default:
        MNRY_ASSERT(0);
        return scene_rdl2::math::sBlack;
    }
}

const scene_rdl2::fb_util::ByteColor&
InputBuffer::getRgb888Pixel(unsigned int x, unsigned int y) const
{
    unsigned int relativeX = x - mStartX;
    unsigned int relativeY = y - mStartY;

    return mPixelBuffer->getRgb888Buffer().getPixel(relativeX, relativeY);
}

const scene_rdl2::fb_util::ByteColor4&
InputBuffer::getRgba8888Pixel(unsigned int x, unsigned int y) const
{
    unsigned int relativeX = x - mStartX;
    unsigned int relativeY = y - mStartY;

    return mPixelBuffer->getRgba8888Buffer().getPixel(relativeX, relativeY);
}

float
InputBuffer::getFloatPixel(unsigned int x, unsigned int y) const
{
    unsigned int relativeX = x - mStartX;
    unsigned int relativeY = y - mStartY;

    return mPixelBuffer->getFloatBuffer().getPixel(relativeX, relativeY);
}

const scene_rdl2::math::Vec2f&
InputBuffer::getFloat2Pixel(unsigned int x, unsigned int y) const
{
    unsigned int relativeX = x - mStartX;
    unsigned int relativeY = y - mStartY;

    return mPixelBuffer->getFloat2Buffer().getPixel(relativeX, relativeY);
}

const scene_rdl2::math::Vec3f&
InputBuffer::getFloat3Pixel(unsigned int x, unsigned int y) const
{
    unsigned int relativeX = x - mStartX;
    unsigned int relativeY = y - mStartY;

    return mPixelBuffer->getFloat3Buffer().getPixel(relativeX, relativeY);
}

const scene_rdl2::math::Vec4f&
InputBuffer::getFloat4Pixel(unsigned int x, unsigned int y) const
{
    unsigned int relativeX = x - mStartX;
    unsigned int relativeY = y - mStartY;

    return mPixelBuffer->getFloat4Buffer().getPixel(relativeX, relativeY);
}

unsigned int
InputBuffer::getWidth() const
{
    return mPixelBuffer->getWidth();
}

unsigned int
InputBuffer::getHeight() const
{
    return mPixelBuffer->getHeight();
}

scene_rdl2::fb_util::VariablePixelBuffer::Format
InputBuffer::getFormat() const
{
    return mPixelBuffer->getFormat();
}

} // namespace displayfilter
} // namespace moonray


