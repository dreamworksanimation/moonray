// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0


#include "PixelBufferUtils.h"
#include "ExrUtils.h"
#include "Util.h"

#include <scene_rdl2/common/except/exceptions.h>
#include <scene_rdl2/common/fb_util/PixelBufferUtilsGamma8bit.h>
#include <scene_rdl2/render/util/Strings.h>

#include <OpenImageIO/imageio.h>
#include <OpenImageIO/imagebuf.h>
#include <OpenImageIO/imagebufalgo.h>

#include <functional>
#include <memory>
#include <type_traits>

namespace moonray {
namespace rndr {

using namespace scene_rdl2::math;

void
addBuffers(scene_rdl2::fb_util::Rgb888Buffer& destBuffer, const scene_rdl2::fb_util::Rgb888Buffer& srcBuffer, scene_rdl2::fb_util::PixelBufferUtilOptions options)
{
    const bool parallel = options & scene_rdl2::fb_util::PIXEL_BUFFER_UTIL_OPTIONS_PARALLEL;

    MNRY_ASSERT(destBuffer.getWidth() == srcBuffer.getWidth());
    MNRY_ASSERT(destBuffer.getHeight() == srcBuffer.getHeight());

    unsigned area = destBuffer.getWidth() * destBuffer.getHeight();

    uint8_t *dst = &destBuffer.getData()->r;
    const uint8_t *src = &srcBuffer.getData()->r;

    // TODO: This is super slow, optimize as needed.
    simpleLoop(parallel, 0u, area * 3, [&](unsigned i) {
        unsigned v = unsigned(dst[i]) + unsigned(src[i]);
        if (v > 255) v = 255;
        dst[i] = uint8_t(v);
    });
}

namespace detail {

template <typename Buffer>
struct BufferTraits
{
    using BufferType = Buffer;
    using CastType = void;
    static const char* colorspace;
    static const int numChannels;
    static const OIIO::TypeDesc oiioType;
    CastType* castTo(const BufferType&);
};

template<>
struct BufferTraits<scene_rdl2::fb_util::RenderBuffer>
{
    using BufferType = scene_rdl2::fb_util::RenderBuffer;
    using CastType = BufferType::PixelType;
    static const char* colorspace;
    static const int numChannels;
    static const OIIO::TypeDesc oiioType;
    CastType* castTo(const BufferType& buffer)
    {
        return const_cast<CastType*>(buffer.getData());
    }
};

const char* BufferTraits<scene_rdl2::fb_util::RenderBuffer>::colorspace = "Linear";
const OIIO::TypeDesc BufferTraits<scene_rdl2::fb_util::RenderBuffer>::oiioType = OIIO::TypeDesc::FLOAT;
const int BufferTraits<scene_rdl2::fb_util::RenderBuffer>::numChannels = 4;

template<>
struct BufferTraits<scene_rdl2::fb_util::Rgb888Buffer>
{
    using BufferType = scene_rdl2::fb_util::Rgb888Buffer;
    using CastType = uint8_t;
    static const char* colorspace;
    static const int numChannels;
    static const OIIO::TypeDesc oiioType;
    CastType* castTo(const BufferType& buffer)
    {
        return reinterpret_cast<CastType*>(const_cast<scene_rdl2::fb_util::Rgb888Buffer::PixelType*>(buffer.getData()));
    }
};

const char* BufferTraits<scene_rdl2::fb_util::Rgb888Buffer>::colorspace = "sRGB";
const OIIO::TypeDesc BufferTraits<scene_rdl2::fb_util::Rgb888Buffer>::oiioType = OIIO::TypeDesc::UINT8;
const int BufferTraits<scene_rdl2::fb_util::Rgb888Buffer>::numChannels = 3;

template<>
struct BufferTraits<scene_rdl2::fb_util::FloatBuffer>
{
    using BufferType = scene_rdl2::fb_util::FloatBuffer;
    using CastType = float;
    static const char* colorspace;
    static const int numChannels;
    static const OIIO::TypeDesc oiioType;
    CastType* castTo(const BufferType& buffer)
    {
        return reinterpret_cast<CastType*>(const_cast<scene_rdl2::fb_util::FloatBuffer::PixelType*>(buffer.getData()));
    }
};

const char* BufferTraits<scene_rdl2::fb_util::FloatBuffer>::colorspace = "Linear";
const OIIO::TypeDesc BufferTraits<scene_rdl2::fb_util::FloatBuffer>::oiioType = OIIO::TypeDesc::FLOAT;
const int BufferTraits<scene_rdl2::fb_util::FloatBuffer>::numChannels = 1;

}

template<typename Buffer>
void
readPixelBuffer(Buffer& buf, const std::string& filename,
        unsigned overrideWidth, unsigned overrideHeight)
{
    std::unique_ptr<OIIO::ImageInput> in(OIIO::ImageInput::create(filename));
    if (in.get() == NULL) {
        throw scene_rdl2::except::IoError("Cannot find image file: \"" + filename +
                  "\" (file not found)");
    }

    OIIO::ImageSpec imageSpec;
    if (in->open(filename, (imageSpec)) == false) {
        throw scene_rdl2::except::IoError("Cannot open image file: \"" + filename +
                  "\" (" + in->geterror() + ")");
    }

    // read in source image
    void *tmpBufferSrc = malloc(imageSpec.width * imageSpec.height *
                         imageSpec.nchannels * imageSpec.format.size());

    MNRY_ASSERT_REQUIRE(tmpBufferSrc);

    if (!in->read_image(imageSpec.format, tmpBufferSrc)) {
        free(tmpBufferSrc);
        throw scene_rdl2::except::IoError("Cannot read image file: \"" + filename +
                  "\" (" + in->geterror() + ")");
    }
    OIIO::ImageBuf source("source", imageSpec, tmpBufferSrc);

    detail::BufferTraits<Buffer> traits;

    // convert to pixel buffer's number of channels
    if (imageSpec.nchannels != traits.numChannels) {
        OIIO::ImageBuf channelConvert;
        bool success;

        if ((imageSpec.nchannels == 3 || imageSpec.nchannels == 4) &&
             traits.numChannels == 1) {
            float lumaweights[3] = { .2126f, .7152f, .0722f };
            OIIO::ROI roi = source.roi();
            roi.chbegin = 0; roi.chend = 3;
            success = OIIO::ImageBufAlgo::channel_sum(channelConvert, source, lumaweights, roi);

        } else if (imageSpec.nchannels == 1 && traits.numChannels == 3) {
            int channelorder[] = { 0, 0, 0 };
            float channelvalues[] = { 0.0f, 0.0f, 0.0f };
            success = OIIO::ImageBufAlgo::channels(channelConvert, source, traits.numChannels,
                                        channelorder, channelvalues);

        } else if (imageSpec.nchannels == 1 && traits.numChannels == 4) {
            int channelorder[] = { 0, 0, 0, -1 };
            float channelvalues[] = { 0.0f, 0.0f, 0.0f, 1.0f };
            success = OIIO::ImageBufAlgo::channels(channelConvert, source, traits.numChannels,
                                        channelorder, channelvalues);

        } else if (imageSpec.nchannels == 3 && traits.numChannels == 4) {
            int channelorder[] = { 0, 1, 2, -1 };
            float channelvalues[] = { 0.0f, 0.0f, 0.0f, 1.0f };
            std::string channelnames[] = { "", "", "", "A" };
            success = OIIO::ImageBufAlgo::channels(channelConvert, source, traits.numChannels,
                                        channelorder, channelvalues, channelnames);

        } else if (imageSpec.nchannels == 4 && traits.numChannels == 3) {
            success = OIIO::ImageBufAlgo::channels(channelConvert, source, traits.numChannels, NULL);

        } else {
            success = false;
        }

        if (!success) {
            free(tmpBufferSrc);
            throw scene_rdl2::except::IoError("\"" + filename + "\": cannot convert " +
                    std::to_string(imageSpec.nchannels) + " channels to " +
                    std::to_string(traits.numChannels) + " channels.");
        }

        source.clear();
        source.copy(channelConvert);
    }

    unsigned finalImageWidth = imageSpec.width;
    unsigned finalImageHeight = imageSpec.height;

    // Resize the image to the buffer size
    if ((overrideWidth != 0 && overrideHeight != 0) &&
        (imageSpec.width != overrideWidth || imageSpec.height != overrideHeight)) {
        OIIO::ImageSpec imageSpecResized(overrideWidth, overrideHeight, traits.numChannels, traits.oiioType);
        OIIO::ImageBuf resized("resized", imageSpecResized);
        if (!OIIO::ImageBufAlgo::resize(resized, source)) {
            free(tmpBufferSrc);
            throw scene_rdl2::except::IoError("Cannot resize image \"" + filename + "\"");
        }
        source.clear();
        source.copy(resized);

        finalImageWidth = overrideWidth;
        finalImageHeight = overrideHeight;
    }

    // initialize pixel buffer data
    if (buf.getArea() == 0) {
        buf.init(finalImageWidth, finalImageHeight);
    }
    auto *data = traits.castTo(buf);

    // Flip the buffer: our frame buffers are upside down compared to what
    // OIIO is giving us.
    OIIO::ImageSpec imageSpecFlipped(finalImageWidth, finalImageHeight, traits.numChannels, traits.oiioType);
    OIIO::ImageBuf flipped("flipped", imageSpecFlipped, data);
    if (!OIIO::ImageBufAlgo::flip(flipped, source)) {
        free(tmpBufferSrc);
        throw scene_rdl2::except::IoError("Cannot flip image \"" + filename + "\"");
    }

    free(tmpBufferSrc);
}

// Specify the different buffer versions for readPixelBuffer
template
void
readPixelBuffer<scene_rdl2::fb_util::RenderBuffer>(scene_rdl2::fb_util::RenderBuffer& buf, const std::string& filename,
        unsigned overrideWidth, unsigned overrideHeight);

template
void
readPixelBuffer<scene_rdl2::fb_util::Rgb888Buffer>(scene_rdl2::fb_util::Rgb888Buffer& buf, const std::string& filename,
        unsigned overrideWidth, unsigned overrideHeight);

template
void
readPixelBuffer<scene_rdl2::fb_util::FloatBuffer>(scene_rdl2::fb_util::FloatBuffer& buf, const std::string& filename,
        unsigned overrideWidth, unsigned overrideHeight);

template<typename Buffer>
void
writePixelBuffer(const Buffer& buf, const std::string& filename, const scene_rdl2::rdl2::SceneObject *metadata,
                 const scene_rdl2::math::HalfOpenViewport& aperture, const scene_rdl2::math::HalfOpenViewport& region)
{
    detail::BufferTraits<Buffer> traits;

    // Create an OIIO output buffer.
    std::unique_ptr<OIIO::ImageOutput> out(
            OIIO::ImageOutput::create(filename.c_str()));
    if (!out) {
        throw scene_rdl2::except::IoError(scene_rdl2::util::buildString(
                "Failed create ImageOutput for '", filename, "'."));
    }

    // Define the image spec.
    OIIO::ImageSpec srcSpec(buf.getWidth(), buf.getHeight(), traits.numChannels, traits.oiioType);
    srcSpec.attribute("oiio:ColorSpace", traits.colorspace);

    // add metadata to image spec
    if (metadata) {
        writeExrHeader(srcSpec, metadata->asA<scene_rdl2::rdl2::Metadata>());
    }

    // Flip the buffer (our frame buffers are upside down compared to what
    // OIIO is expecting). This must be done in a coordinate system relative
    // to the region window, where top left corner is 0,0 and bottom right is
    // buf.getWidth, buf.getHeight.
    OIIO::ImageBuf srcBuffer(filename, srcSpec, traits.castTo(buf));
    OIIO::ImageBuf flippedBuffer(filename, srcSpec);
    OIIO::ImageBufAlgo::flip(flippedBuffer, srcBuffer);

    // Create image buffer with aperture and region window metadata.
    OIIO::ImageSpec destSpec(srcSpec);
    destSpec.x = region.min().x;
    // flip y coordinate relative to display window
    destSpec.y = aperture.max().y - region.max().y;
    destSpec.full_x = aperture.min().x;
    destSpec.full_y = aperture.min().y;
    destSpec.full_width = aperture.width();
    destSpec.full_height = aperture.height();
    OIIO::ImageBuf destBuffer(filename, destSpec);

    // Copy flipped image into image buffer with proper metadata.
    size_t dataSize = traits.oiioType == OIIO::TypeDesc::UINT8 ? 1 : 4;
    void* buffer = malloc(dataSize * traits.numChannels * buf.getWidth() * buf.getHeight());
    flippedBuffer.get_pixels(OIIO::get_roi(srcSpec), traits.oiioType, buffer);
    destBuffer.set_pixels(OIIO::get_roi(destSpec), traits.oiioType, buffer);
    free(buffer);

    // Open the output buffer, write the flipped buffer to it, and close it.
    if (!out->open(filename.c_str(), destSpec)) {
        throw scene_rdl2::except::IoError(scene_rdl2::util::buildString("Failed to open '",
                filename, "' for writing."));
    }
    destBuffer.write(out.get());
    out->close();
}

// Specify the different buffer versions for writePixelBuffer
template
void
writePixelBuffer<scene_rdl2::fb_util::RenderBuffer>(const scene_rdl2::fb_util::RenderBuffer& buf,
                                        const std::string& filename,
                                        const scene_rdl2::rdl2::SceneObject *metadata,
                                        const scene_rdl2::math::HalfOpenViewport& aperture,
                                        const scene_rdl2::math::HalfOpenViewport& region);

template
void
writePixelBuffer<scene_rdl2::fb_util::Rgb888Buffer>(const scene_rdl2::fb_util::Rgb888Buffer& buf,
                                        const std::string& filename,
                                        const scene_rdl2::rdl2::SceneObject *metadata,
                                        const scene_rdl2::math::HalfOpenViewport& aperture,
                                        const scene_rdl2::math::HalfOpenViewport& region);

template
void
writePixelBuffer<scene_rdl2::fb_util::FloatBuffer>(const scene_rdl2::fb_util::FloatBuffer& buf,
                                       const std::string& filename,
                                       const scene_rdl2::rdl2::SceneObject *metadata,
                                       const scene_rdl2::math::HalfOpenViewport& aperture,
                                       const scene_rdl2::math::HalfOpenViewport& region);

} // namespace rndr
} // namespace moonray

