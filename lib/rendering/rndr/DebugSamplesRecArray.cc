// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

//
//
#include "DebugSamplesRecArray.h"

#include <scene_rdl2/scene/rdl2/ValueContainerDeq.h>
#include <scene_rdl2/scene/rdl2/ValueContainerEnq.h>

#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>

namespace moonray {
namespace rndr {

DebugSamplesRecArray::DebugSamplesRecArray(const unsigned xSize, const unsigned ySize) :
    mMode(Mode::SAVE), mSizeX(xSize), mSizeY(ySize)
//
// construct as save mode
//
{
    mImg.resize(mSizeY);
    for (unsigned y = 0; y < mSizeY; ++y) {
        mImg[y].resize(mSizeX);
    }
}

DebugSamplesRecArray::DebugSamplesRecArray(const std::string &filename)
//
// construct as load mode
//
{
    if (!load(filename)) {
        mMode = Mode::DEAD;
    } else {
        mMode = Mode::LOAD;
    }
}

void
DebugSamplesRecArray::pixSamples(const unsigned x,
                                 const unsigned y,
                                 const unsigned sampleId,
                                 float &r,
                                 float &g,
                                 float &b,
                                 float &a)
{
    if (mMode == Mode::SAVE) pixSamplesSave(x, y, sampleId, r, g, b, a);
    else                     pixSamplesLoad(x, y, sampleId, r, g, b, a);
}

bool
DebugSamplesRecArray::save(const std::string &filename) const
{
    if (mMode != Mode::SAVE) return true; // just in case.

    std::ofstream out(filename.c_str(), std::ios::trunc | std::ios::binary);
    if (!out) return false;

    std::string buff;
    scene_rdl2::rdl2::ValueContainerEnq vContainerEnq(&buff);
    vContainerEnq.enqUInt(mSizeX);
    vContainerEnq.enqUInt(mSizeY);
    crawlAllPixels([&](const size_t x, const size_t y) {
            vContainerEnq.enqUInt((unsigned int)mImg[y][x].size());
            for (unsigned i = 0; i < mImg[y][x].size(); ++i) {
                vContainerEnq.enqVec4f(mImg[y][x][i]);
            }
        });

    size_t size = vContainerEnq.finalize();
    out.write((const char *)&size, sizeof(size));
    out.write(buff.data(), size); // binary data write

    return true;
}

std::string
DebugSamplesRecArray::show(const std::string &hd) const
{
    std::ostringstream ostr;
    ostr << hd << "DebugSamplesRecArray {\n";
    ostr << hd << "  mSizeX:" << mSizeX << " mSizeY:" << mSizeY << '\n';
    ostr << hd << "  totalMemory:" << totalMemorySize() << " byte = " << memStr(totalMemorySize()) << '\n';
    ostr << hd << "  minPixelSamples:" << minPixelSamples() << '\n';
    ostr << hd << "  maxPixelSamples:" << maxPixelSamples() << '\n';
    ostr << hd << "}";
    return ostr.str();
}

//------------------------------------------------------------------------------

bool
DebugSamplesRecArray::load(const std::string &filename)
//
// So far only called from constructor. This means mImg is empty.
//
{
    std::ifstream in(filename.c_str(), std::ios::binary);
    if (!in) return false;

    std::string buff;
    size_t size;
    in.read((char *)&size, sizeof(size));
    buff.resize(size);
    in.read(&buff[0], size);

    scene_rdl2::rdl2::ValueContainerDeq vContainerDeq(buff.data(), size);
    vContainerDeq.deqUInt(mSizeX);
    vContainerDeq.deqUInt(mSizeY);
    mImg.resize(mSizeY);
    for (unsigned y = 0; y < mSizeY; ++y) {
        mImg[y].resize(mSizeX);
        for (unsigned x = 0; x < mSizeX; ++x) {
            unsigned samples;
            vContainerDeq.deqUInt(samples);
            mImg[y][x].resize(samples);
            for (unsigned i = 0; i < samples; ++i) {
                scene_rdl2::math::Vec4f c;
                vContainerDeq.deqVec4f(c);
                // c *= 0.25f; // test
                mImg[y][x][i] = c;
            }
        }
    }

    return true;
}

void    
DebugSamplesRecArray::pixSamplesSave(const unsigned x,
                                     const unsigned y,
                                     const unsigned sampleId,
                                     float &r,
                                     float &g,
                                     float &b,
                                     float &a)
{
    if (x >= mSizeX || y >= mSizeY) return;

    PixSamples &pix = mImg[y][x];
    if (pix.size() != sampleId) {
        std::cerr << ">> DebugSamplesRecArray.cc ====== RUNTIME-OVERRUN-ERROR ======  pixSamplesSave() failed."
                  << " x:" << x << " y:" << y << " sampleId:" << sampleId << std::endl;
        return;                 // duplicate sampling id
    }

    pix.emplace_back(r, g, b, a);
}

void    
DebugSamplesRecArray::pixSamplesLoad(const unsigned x,
                                     const unsigned y,
                                     const unsigned sampleId,
                                     float &r,
                                     float &g,
                                     float &b,
                                     float &a) const
{
    if (x >= mSizeX || y >= mSizeY) return;

    const PixSamples &pix = mImg[y][x];
    if (sampleId >= pix.size()) {
        std::cerr << ">> DebugSamplesRecArray.cc ====== RUNTIME-OVERRUN-ERROR ====== pixSamplesLoad() failed."
                  << " x:" << x << " y:" << y << " sampleId:" << sampleId << " >= pix.size():" << pix.size()
                  << std::endl;
        return;
    }

    const scene_rdl2::fb_util::RenderColor &sample = pix[sampleId];
    r = sample[0];
    g = sample[1];
    b = sample[2];
    a = sample[3];
}

size_t
DebugSamplesRecArray::totalMemorySize() const
{
    size_t total = 0;
    crawlAllPixels([&](const unsigned x, const unsigned y) {
            total += mImg[y][x].size() * sizeof(scene_rdl2::fb_util::RenderColor);
        });
    return total;
}

unsigned
DebugSamplesRecArray::minPixelSamples() const
{
    unsigned min = std::numeric_limits<unsigned>::max();
    crawlAllPixels([&](const unsigned x, const unsigned y) {
            min = scene_rdl2::math::min(min, static_cast<unsigned>(mImg[y][x].size()));
        });
    return min;
}

unsigned
DebugSamplesRecArray::maxPixelSamples() const
{
    unsigned max = 0;
    crawlAllPixels([&](const unsigned x, const unsigned y) {
            max = scene_rdl2::math::max(max, static_cast<unsigned>(mImg[y][x].size()));
        });
    return max;
}

std::string
DebugSamplesRecArray::memStr(const size_t byte) const
{
    std::ostringstream ostr;
    if (byte < 1024) {
        ostr << byte << " byte";
    } else if (byte < (size_t)1024 * (size_t)1024) {
        float kByte = (float)byte / 1024.f;
        ostr << std::setw(7) << std::fixed << std::setprecision(2) << kByte << " Kbyte";
    } else if (byte < (size_t)1024 * (size_t)1024 * (size_t)1024) {
        float mByte = (float)byte / 1024.f / 1024.f;
        ostr << std::setw(7) << std::fixed << std::setprecision(2) << mByte << " Mbyte";
    } else if (byte < (size_t)1024 * (size_t)1024 * (size_t)1024 * (size_t)1024) {
        float gByte = (float)byte / 1024.f / 1024.f / 1024.f;
        ostr << std::setw(7) << std::fixed << std::setprecision(2) << gByte << " Gbyte";
    } else {
        float tByte = (float)byte / 1024.f / 1024.f / 1024.f / 1024.f;
        ostr << std::setw(7) << std::fixed << std::setprecision(2) << tByte << " Tbyte";
    }
    return ostr.str();
}

} // namespace rndr
} // namespace moonray

