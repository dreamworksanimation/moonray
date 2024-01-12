// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

//
//
#pragma once

#include <scene_rdl2/common/fb_util/FbTypes.h>

#include <string>

namespace moonray {
namespace rndr {

class DebugSamplesRecArray
{
public:
    enum class Mode : int {
        SAVE = 0,
        LOAD,
        DEAD
    };

    DebugSamplesRecArray(const unsigned xSize, const unsigned ySize); // construct as save mode
    DebugSamplesRecArray(const std::string &filename); // construct as load mode

    explicit operator bool() const noexcept { return mMode != Mode::DEAD; }
    Mode mode() const { return mMode; }

    void pixSamples(const unsigned x, const unsigned y, const unsigned sampleId,
                    float &r, float &g, float &b, float &a);

    bool save(const std::string &filename) const; // explicitly serialize and save the data into file

    std::string show(const std::string &hd) const;

private:
    using PixSamples = std::vector<scene_rdl2::fb_util::RenderColor>;
    using ScanlineSamples = std::vector<PixSamples>;
    using ImageSamples = std::vector<ScanlineSamples>;

    //------------------------------

    Mode mMode;

    unsigned mSizeX;
    unsigned mSizeY;
    ImageSamples mImg;

    //------------------------------

    bool load(const std::string &filename);

    void pixSamplesSave(const unsigned x, const unsigned y, const unsigned sampleId,
                        float &r, float &g, float &b, float &a);
    void pixSamplesLoad(const unsigned x, const unsigned y, const unsigned sampleId,
                        float &r, float &g, float &b, float &a) const;

    template <typename F> void crawlAllPixels(F pixelFunc) const;

    size_t totalMemorySize() const;
    unsigned minPixelSamples() const;
    unsigned maxPixelSamples() const;

    std::string memStr(const size_t byte) const;
};

template <typename F>    
void
DebugSamplesRecArray::crawlAllPixels(F pixelFunc) const
{
    for (unsigned y = 0; y < mSizeY; ++y) {
        for (unsigned x = 0; x < mSizeX; ++x) {
            pixelFunc(x, y);
        }
    }
}

} // namespace rndr
} // namespace moonray

