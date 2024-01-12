// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

/// @file ConvertImageEncoding.h
#pragma once

#include <moonray/engine/messages/base_frame/BaseFrame.h>
#include <scene_rdl2/common/fb_util/VariablePixelBuffer.h>

namespace moonray {
namespace mcrt_comp {

finline fb_util::VariablePixelBuffer::Format
convertImageEncoding(network::BaseFrame::ImageEncoding encoding)
{
    switch (encoding)
    {
    case network::BaseFrame::ENCODING_RGBA8:
        return fb_util::VariablePixelBuffer::RGBA8888;

    case network::BaseFrame::ENCODING_RGB888:
        return fb_util::VariablePixelBuffer::RGB888;

    case network::BaseFrame::ENCODING_FLOAT:
        return fb_util::VariablePixelBuffer::FLOAT;

    case network::BaseFrame::ENCODING_FLOAT2:
        return fb_util::VariablePixelBuffer::FLOAT2;

    case network::BaseFrame::ENCODING_FLOAT3:
        return fb_util::VariablePixelBuffer::FLOAT3;

    case network::BaseFrame::ENCODING_LINEAR_FLOAT:
        return fb_util::VariablePixelBuffer::FLOAT4;

    default:
        MNRY_ASSERT(0);
    }
    return fb_util::VariablePixelBuffer::UNINITIALIZED;
}

finline network::BaseFrame::ImageEncoding
renderOutputImageEncoding(unsigned int nChans)
{
    switch (nChans)
    {
    case 1:  return network::BaseFrame::ENCODING_FLOAT;
    case 2:  return network::BaseFrame::ENCODING_FLOAT2;
    case 3:  return network::BaseFrame::ENCODING_FLOAT3;
    default: MNRY_ASSERT(0 && "unhandled nChans");
    }

    return network::BaseFrame::ENCODING_UNKNOWN;
}


} // namespace mcrt_comp
} // namespace moonray


