// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

//
//
#pragma once

#include <scene_rdl2/common/fb_util/VariablePixelBuffer.h>
#include <scene_rdl2/common/platform/Platform.h> // finline

namespace moonray {
namespace engine_tool {

enum class ImgEncodingType : int {
    ENCODING_UNKNOWN = 0,

    ENCODING_RGBA8,
    ENCODING_RGB888,
    ENCODING_LINEAR_FLOAT,
    ENCODING_FLOAT,
    ENCODING_FLOAT2,
    ENCODING_FLOAT3,
};

finline ImgEncodingType
fbFormatToImgEncodingType(const scene_rdl2::fb_util::VariablePixelBuffer::Format &format)
{
    ImgEncodingType enco = ImgEncodingType::ENCODING_UNKNOWN;
    switch (format) {
    case scene_rdl2::fb_util::VariablePixelBuffer::RGB888   : enco = ImgEncodingType::ENCODING_RGB888;       break;
    case scene_rdl2::fb_util::VariablePixelBuffer::RGBA8888 : enco = ImgEncodingType::ENCODING_RGBA8;        break;
    case scene_rdl2::fb_util::VariablePixelBuffer::FLOAT    : enco = ImgEncodingType::ENCODING_FLOAT;        break;
    case scene_rdl2::fb_util::VariablePixelBuffer::FLOAT2   : enco = ImgEncodingType::ENCODING_FLOAT2;       break;
    case scene_rdl2::fb_util::VariablePixelBuffer::FLOAT3   : enco = ImgEncodingType::ENCODING_FLOAT3;       break;
    case scene_rdl2::fb_util::VariablePixelBuffer::FLOAT4   : enco = ImgEncodingType::ENCODING_LINEAR_FLOAT; break;
    default : break;
    }
    return enco;
}

finline scene_rdl2::fb_util::VariablePixelBuffer::Format
imgEncodingTypeToFbFormat(ImgEncodingType imgEncoding)
{
    scene_rdl2::fb_util::VariablePixelBuffer::Format fmt = scene_rdl2::fb_util::VariablePixelBuffer::UNINITIALIZED;
    switch (imgEncoding) {
    case ImgEncodingType::ENCODING_RGBA8        : fmt = scene_rdl2::fb_util::VariablePixelBuffer::RGBA8888; break;
    case ImgEncodingType::ENCODING_RGB888       : fmt = scene_rdl2::fb_util::VariablePixelBuffer::RGB888;   break;
    case ImgEncodingType::ENCODING_FLOAT        : fmt = scene_rdl2::fb_util::VariablePixelBuffer::FLOAT;    break;
    case ImgEncodingType::ENCODING_FLOAT2       : fmt = scene_rdl2::fb_util::VariablePixelBuffer::FLOAT2;   break;
    case ImgEncodingType::ENCODING_FLOAT3       : fmt = scene_rdl2::fb_util::VariablePixelBuffer::FLOAT3;   break;
    case ImgEncodingType::ENCODING_LINEAR_FLOAT : fmt = scene_rdl2::fb_util::VariablePixelBuffer::FLOAT4;   break;
    default: break;
    }
    return fmt;
}

} // namespace engine_tool
} // namespace moonray

