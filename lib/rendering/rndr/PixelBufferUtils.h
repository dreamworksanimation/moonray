// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0


#pragma once
#ifndef PIXELBUFFERUTILS_H
#define PIXELBUFFERUTILS_H

#include <scene_rdl2/common/fb_util/VariablePixelBuffer.h>
#include <scene_rdl2/common/math/Viewport.h>
#include <vector>

namespace scene_rdl2 {

namespace rdl2 {
    class SceneObject;
}
}

namespace moonray {

namespace rndr {

/**
 * Add 2 LDR buffers together. The pixel values in srcBuffer are added to those
 * in destBuffer with clamping to max. 
 *
 * @param   destBuffer  The desintation buffer to write into which is also read 
 *                      from.
 * @param   srcBuffer   The buffer to add in.
 * @param   options     parallel supported
 */
void addBuffers(scene_rdl2::fb_util::Rgb888Buffer& destBuffer, const scene_rdl2::fb_util::Rgb888Buffer& srcBuffer,
                scene_rdl2::fb_util::PixelBufferUtilOptions options);

/**
 * Uses OpenImageIO to read in an image file to a given pixel buffer.
 *
 * @param   buf         The buffer to read into.
 * @param   filename    The name of the input file on the filesystem.
 */
template<typename BufferType>
void readPixelBuffer(BufferType& buf, const std::string& filename,
        unsigned overrideWidth = 0, unsigned overrideHeight = 0);

/**
 * Uses OpenImageIO to write out the given pixel buffer to a file. The
 * appropriate output plugin will be chosen by the extension of the filename.
 *
 * @param   buf         The buffer to write out.
 * @param   filename    The name of the output file on the filesystem.
 * @param   metadata    The exr header metadata for the file.
 */
template<typename BufferType>
void writePixelBuffer(const BufferType& buf, const std::string& filename, const scene_rdl2::rdl2::SceneObject *metadata,
                      const scene_rdl2::math::HalfOpenViewport& aperture, const scene_rdl2::math::HalfOpenViewport& region);

} // namespace rndr
} // namespace moonray

#endif // PIXELBUFFERUTILS_H
