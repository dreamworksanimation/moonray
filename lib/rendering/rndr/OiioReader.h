// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

//
//
#pragma once

#include "OiioUtils.h"
#include "Util.h"

//#define SINGLE_THREAD

#ifndef SINGLE_THREAD
#include <tbb/parallel_for.h>
#endif // end !SINGLE_THREAD

namespace moonray {
namespace rndr {

//
// Moonray generated image file read and access wrapper
//
class OiioReader
{
public:
    // Constructor did open file (but not read data itself). Destructor automatically close file
    explicit OiioReader(const std::string &filename) : mFilename(filename), mIn(OiioUtils::openFile(filename)) {
        if (mIn) mSpec = mIn->spec();    // copy initial spec into current
    }

    explicit operator bool() const noexcept { return (bool)mIn; }

    const std::string &filename() const { return mFilename; }

    bool readData(const int subImageId = 0); // read entire image from file and keep inside

    int getWidth() const { return mSpec.width; }
    int getHeight() const { return mSpec.height; }
    int getTiledAlignedWidth() const { return (mSpec.width + 7) & ~7; }
    int getTiledAlignedHeight() const { return (mSpec.height + 7) & ~7; }

    inline bool getMetadata(const std::string &name, int &out) const;
    inline bool getMetadata(const std::string &name, std::string &out) const;
    inline bool getMetadata(const std::string &name, float out[3]) const;
    inline int getChannelTotal() const { return mSpec.nchannels; }
    inline int getPixChanOffset(const std::string &channelName) const;

    //
    // Crawl all pixels by pixel access function with tiled memory layout consideration.
    // Pixel operation is defined by pixFunc argument.
    // mPixels is non-tiled memory layout but this idea might be worth if pixFunc access tiled layout memory
    // This template is only used for debug purpose so far.
    //
    template <typename F>
    void
    crawlAllPixels(F pixFunc) {
        const OIIO::ImageSpec &spec = mSpec;
        int totalTileY = (spec.height + 7) / 8;
        int totalTileX = (spec.width + 7) / 8;

#ifdef SINGLE_THREAD
        constexpr bool parallel = false;
#else        
        constexpr bool parallel = true;
#endif
        simpleLoop2<parallel>(0, totalTileY, [&](unsigned tileY) {
                int startY = tileY * 8;
                int endY = (startY + 8 < spec.height)? startY + 8: spec.height;

                for (int tileX = 0; tileX < totalTileX; ++tileX) {
                    int startX = tileX * 8;
                    int endX = (startX + 8 < spec.width)? startX + 8: spec.width;

                    for (int pixY = startY; pixY < endY; ++pixY) {
                        for (int pixX = startX; pixX < endX; ++pixX) {
                            int pixCount = (spec.height - pixY - 1) * spec.width + pixX; // flip Y to access pix
                            int pixOffsetId = pixCount * spec.nchannels;
                            pixFunc(pixX, pixY, spec.width, &mPixels[pixOffsetId]);
                        } // pixX
                    } // pixY
                } // tileX
            });
    }

    //
    // Crawl all pixels by scanline segment access function with tiled memory layout consideration.
    // Scanline operation is defined by scanlineFunc argument.
    // mPixels is non-tiled memory layout but this idea is very worth if scanlineFunc access tiled layout memory
    // This template is used by RenderOutputDriver read API.
    //
    template <typename F>
    void
    crawlAllTiledScanline(F scanlineFunc) {
        const OIIO::ImageSpec &spec = mSpec;
        int totalTileY = (spec.height + 7) / 8;
        int totalTileX = (spec.width + 7) / 8;

#ifdef SINGLE_THREAD
        constexpr bool parallel = false;
#else        
        constexpr bool parallel = true;
#endif
        simpleLoop2<parallel>(0, totalTileY, [&](unsigned tileY) {
                int startY = tileY * 8;
                int endY = (startY + 8 < spec.height)? startY + 8: spec.height;

                for (int tileX = 0; tileX < totalTileX; ++tileX) {
                    int startX = tileX * 8;
                    int endX = (startX + 8 < spec.width)? startX + 8: spec.width;

                    for (int pixY = startY; pixY < endY; ++pixY) {
                        int startPixCount = (spec.height - pixY - 1) * spec.width + startX; // flip Y to access pix
                        int startPixOffsetId = startPixCount * spec.nchannels;
                        scanlineFunc(startX, endX, pixY, spec.nchannels, &mPixels[startPixOffsetId]);
                    } // pixY
                } // tileX
            });
    }

    std::string showSpec(const std::string &hd) const;

private:
    std::string mFilename;      // current open filename
    OiioUtils::ImageInputUqPtr mIn;
    OIIO::ImageSpec mSpec;      // current subimage ImageSpec
    std::vector<float> mPixels; // current read_image() result.
};

inline bool
OiioReader::getMetadata(const std::string &name, int &out) const
{
    return OiioUtils::getMetadata(mSpec, name, out);
}

inline bool    
OiioReader::getMetadata(const std::string &name, std::string &out) const
{
    return OiioUtils::getMetadata(mSpec, name, out);
}

inline bool
OiioReader::getMetadata(const std::string &name, float out[3]) const
{
    return OiioUtils::getMetadata(mSpec, name, out);
}

inline int
OiioReader::getPixChanOffset(const std::string &channelName) const
{
    return OiioUtils::getPixChanOffset(mSpec, channelName);
}

} // namespace rndr
} // namespace moonray

