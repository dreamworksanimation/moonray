// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

//
//

#include "InputBuffer.isph"

export uniform uint32_t
InputBuffer_hudValidation(uniform bool verbose)
{
    DISPLAYFILTER_INPUTBUFFER_VALIDATION;
}

void
InputBuffer_reset(uniform InputBuffer * uniform buffer)
{
    if (buffer->mPixelBuffer) {
        VariablePixelBuffer_cleanUp(buffer->mPixelBuffer);
    }
}

void
InputBuffer_init(uniform InputBuffer * uniform buffer,
                              uniform Format format,
                              uniform unsigned int width,
                              uniform unsigned int height)
{
    InputBuffer_reset(buffer);
    VariablePixelBuffer_init(buffer->mPixelBuffer, format, width, height);
}

void
InputBuffer_setData(uniform InputBuffer * uniform buffer,
                                 uniform unsigned int offset,
                                 uniform unsigned int len,
                                 const uniform uint8_t * uniform data)
{
    uniform unsigned int sizeOfPixel = VariablePixelBuffer_getSizeOfPixel(buffer->mPixelBuffer);
    MNRY_ASSERT(offset + len < VariablePixelBuffer_getArea(buffer->mPixelBuffer) * sizeOfPixel);
    uniform uint8_t * uniform bufferData = VariablePixelBuffer_getData(buffer->mPixelBuffer);
    memcpy((void * uniform)(bufferData + offset * sizeOfPixel), (void * uniform)data, len * sizeOfPixel);
}

void
InputBuffer_setStartPixel(uniform InputBuffer * uniform buffer,
                                       uniform unsigned int x,
                                       uniform unsigned int y)
{
    buffer->mStartX = x;
    buffer->mStartY = y;
}

varying Color
InputBuffer_getPixel(const uniform InputBuffer * uniform buffer,
                                  varying unsigned int x,
                                  varying unsigned int y)
{
    uniform Format format = InputBuffer_getFormat(buffer);
    switch(format) {
    case RGB888:
    {
        const ByteColor val = InputBuffer_getRgb888Pixel(buffer, x, y);
        return Color_ctor(val.r, val.g, val.b);
    }
    case RGBA8888:
    {
        const ByteColor4 val = InputBuffer_getRgba8888Pixel(buffer, x, y);
        return Color_ctor(val.r, val.g, val.b);
    }
    case FLOAT:
    {
        const float val = InputBuffer_getFloatPixel(buffer, x, y);
        return Color_ctor(val, val, val);
    }
    case FLOAT2:
    {
        const Vec2f val = InputBuffer_getFloat2Pixel(buffer, x, y);
        return Color_ctor(val.x, val.y, 0.f);
    }
    case FLOAT3:
    {
        const Vec3f val = InputBuffer_getFloat3Pixel(buffer, x, y);
        return Color_ctor(val.x, val.y, val.z);
    }
    case FLOAT4:
    {
        const Vec4f val = InputBuffer_getFloat4Pixel(buffer, x, y);
        return Color_ctor(val.x, val.y, val.z);
    }
    default:
        MNRY_ASSERT(0);
        return sBlack;
    }
}

varying ByteColor
InputBuffer_getRgb888Pixel(const uniform InputBuffer * uniform buffer,
                                        varying unsigned int x,
                                        varying unsigned int y)
{
    unsigned int relativeX = x - buffer->mStartX;
    unsigned int relativeY = y - buffer->mStartY;

    return VariablePixelBuffer_getRgb888Pixel(buffer->mPixelBuffer, relativeX, relativeY);
}

varying ByteColor4
InputBuffer_getRgba8888Pixel(const uniform InputBuffer * uniform buffer,
                                          varying unsigned int x,
                                          varying unsigned int y)
{
    unsigned int relativeX = x - buffer->mStartX;
    unsigned int relativeY = y - buffer->mStartY;

    return VariablePixelBuffer_getRgba8888Pixel(buffer->mPixelBuffer, relativeX, relativeY);
}

varying float
InputBuffer_getFloatPixel(const uniform InputBuffer * uniform buffer,
                                        varying unsigned int x,
                                        varying unsigned int y)
{
    unsigned int relativeX = x - buffer->mStartX;
    unsigned int relativeY = y - buffer->mStartY;

    return VariablePixelBuffer_getFloatPixel(buffer->mPixelBuffer, relativeX, relativeY);
}

varying Vec2f
InputBuffer_getFloat2Pixel(const uniform InputBuffer * uniform buffer,
                                        varying unsigned int x,
                                        varying unsigned int y)
{
    unsigned int relativeX = x - buffer->mStartX;
    unsigned int relativeY = y - buffer->mStartY;

    return VariablePixelBuffer_getFloat2Pixel(buffer->mPixelBuffer, relativeX, relativeY);
}

varying Vec3f
InputBuffer_getFloat3Pixel(const uniform InputBuffer * uniform buffer,
                                        varying unsigned int x,
                                        varying unsigned int y)
{
    unsigned int relativeX = x - buffer->mStartX;
    unsigned int relativeY = y - buffer->mStartY;
    return VariablePixelBuffer_getFloat3Pixel(buffer->mPixelBuffer, relativeX, relativeY);
}

varying Vec4f
InputBuffer_getFloat4Pixel(const uniform InputBuffer * uniform buffer,
                                        varying unsigned int x,
                                        varying unsigned int y)
{
    unsigned int relativeX = x - buffer->mStartX;
    unsigned int relativeY = y - buffer->mStartY;

    return VariablePixelBuffer_getFloat4Pixel(buffer->mPixelBuffer, relativeX, relativeY);
}

uniform unsigned int
InputBuffer_getWidth(const uniform InputBuffer * uniform buffer)
{
    return VariablePixelBuffer_getWidth(buffer->mPixelBuffer);
}

uniform unsigned int
InputBuffer_getHeight(const uniform InputBuffer * uniform buffer)
{
    return VariablePixelBuffer_getHeight(buffer->mPixelBuffer);
}

uniform Format
InputBuffer_getFormat(const uniform InputBuffer * uniform buffer)
{
    return VariablePixelBuffer_getFormat(buffer->mPixelBuffer);
}


