// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

//
//
#include "PixSampleRuntimeVerify.h"
#include "SampleIdBuff.h"

#include <scene_rdl2/render/util/StrUtil.h>

#include <algorithm> // max()
#include <iomanip>
#include <sstream>

namespace moonray {
namespace rndr {

#ifdef RUNTIME_VERIFY_PIX_SAMPLE_COUNT

PixSampleCountRuntimeVerify::PixSampleCountRuntimeVerify(unsigned width, unsigned height) :
    mWidth(width),
    mHeight(height),
    mPix(mWidth * mHeight)
{
    for (size_t i = 0; i < mWidth * mHeight; ++i) {
        std::atomic_init(std::addressof(mPix[i]), static_cast<unsigned>(0));
    }
}

std::string
PixSampleCountRuntimeVerify::show() const
{
    int w0 = scene_rdl2::str_util::getNumberOfDigits(mHeight - 1);
    int w1 = scene_rdl2::str_util::getNumberOfDigits(getMax());

    std::ostringstream ostr;
    ostr << "PixSampleCount (w:" << mWidth << " h:" << mHeight << ") \":\" indicates boundary of 4 tiles {\n";
    for (int y = mHeight - 1; y >= 0; --y) {
        ostr << "  " << std::setw(w0) << y << ":";
        for (unsigned x = 0; x < mWidth; ++x) {
            ostr << std::setw(w1) << mPix[pixOffset(x, y)];
            if ((x + 1) % 32 == 0) {
                ostr << ((x < mWidth - 1) ? ":" : "");
            } else {
                ostr << ((w1 == 1) ? "" : " ");
            }
        }
        ostr << '\n';
    }
    ostr << "}";
    return ostr.str();
}

//------------------------------------------------------------------------------------------

static std::unique_ptr<PixSampleCountRuntimeVerify> sPixSampleCountRuntimeVerify;

// static function
void
PixSampleCountRuntimeVerify::init(unsigned width, unsigned height)
{
    sPixSampleCountRuntimeVerify.reset(new PixSampleCountRuntimeVerify(width, height));
}

// static function
PixSampleCountRuntimeVerify *
PixSampleCountRuntimeVerify::get()
{
    return sPixSampleCountRuntimeVerify.get();
}

#endif // end RUNTIME_VERIFY_PIX_SAMPLE_COUNT

//==========================================================================================

#ifdef RUNTIME_VERIFY_PIX_SAMPLE_SPAN

std::string
PixSmpSpan::show(int sampleValueFieldWidth) const
{
    std::ostringstream ostr;
    ostr << std::setw(sampleValueFieldWidth) << mStartSampleId << " - "
         << std::setw(sampleValueFieldWidth) << mEndSampleId;
    return ostr.str();
}

bool operator < (const PixSmpSpan &A, const PixSmpSpan &B)
{
    return A.getStartSampleId() < B.getStartSampleId();
}

bool
PixSmp::verify(const unsigned startSample, const unsigned lastSample) const
{
    if (mSamples.empty()) {
        return (startSample == lastSample);
    }

    PixSmpSpanArray vec = mSamples;
    std::sort(vec.begin(), vec.end());

    bool flag = true;
    if (vec[0].getStartSampleId() != startSample) flag = false;

    for (size_t i = 0; i < vec.size() - 1; ++i) {
        if (vec[i].getEndSampleId() != vec[i+1].getStartSampleId()) {
            flag = false;
            break;
        }
    }
    if (vec.back().getEndSampleId() != lastSample) flag = false;
    return flag;
}

std::string
PixSmp::show() const
{
    return showSamples(mSamples);
}

// static function
std::string
PixSmp::showSamples(const PixSmpSpanArray &samples)
{
    int w0 = scene_rdl2::str_util::getNumberOfDigits(samples.size());
    int w1 = scene_rdl2::str_util::getNumberOfDigits(getBiggestEndSampleId(samples));

    auto showSub = [&](unsigned i) -> std::string {
        std::ostringstream ostr;
        ostr << std::setw(w0) << i << ": " + samples[i].show(w1);
        return ostr.str();
    };

    std::ostringstream ostr;
    ostr << "samples (total:" << samples.size() << ") {\n";
    for (unsigned i = 0; i < samples.size(); ++i) {
        ostr << scene_rdl2::str_util::addIndent(showSub(i)) << '\n';
    }
    ostr << "}";
    return ostr.str();
}

// static function
unsigned
PixSmp::getBiggestEndSampleId(const PixSmpSpanArray &samples)
{
    unsigned maxId = 0;
    for (const auto &itr: samples) { maxId = std::max(maxId, itr.getEndSampleId()); }
    return maxId;
}

void
PixSampleSpanRuntimeVerify::reset(unsigned width, unsigned height)
{
    mWidth = width;
    mHeight = height;
    mPixels.resize(mWidth * mHeight);
    mPixels.shrink_to_fit();
    for (auto &itr: mPixels) { itr.reset(); }
}

bool
PixSampleSpanRuntimeVerify::verify(const SampleIdBuff *startSampleIdBuff,
                                   const unsigned startSampleId,
                                   const FloatBuffer &lastWeightBuff) const
{
    bool flag =
        crawlAllPix([&](unsigned sx, unsigned sy) {
                unsigned startSample = getStartSample(sx, sy, startSampleIdBuff, startSampleId);
                unsigned lastSample = getLastSample(sx, sy, lastWeightBuff);
                if (!mPixels[pixOffset(sx, sy)].verify(startSample, lastSample)) {
                    std::cerr << "pix(" << sx << ',' << sy << ") " << showPixInfo(sx, sy) << '\n'
                              << " startSample:" << startSample
                              << " lastWeight:" << lastSample << '\n';
                    return false;
                }
                return true;
            });
    std::cerr << "RUNTIME-VERIFY PixSampleSpanRuntimeVerify::verify():" << (flag ? "OK" : "NG") << '\n';

    return flag;
}

std::string
PixSampleSpanRuntimeVerify::show(unsigned x, unsigned y) const
{
    std::ostringstream ostr;
    ostr << "pix(" << x << ',' << y << ") " << showPixInfo(x, y);
    return ostr.str();
}

std::string
PixSampleSpanRuntimeVerify::show(unsigned x, unsigned y,
                                 const SampleIdBuff *startSampleIdBuff,
                                 const unsigned startSampleId,
                                 const FloatBuffer &lastWeightBuff) const
{
    std::ostringstream ostr;
    ostr << show(x, y)
         << " start:" << getStartSample(x, y, startSampleIdBuff, startSampleId)
         << " last:" << getLastSample(x, y, lastWeightBuff);
    return ostr.str();
}

// static function
unsigned
PixSampleSpanRuntimeVerify::getStartSample(unsigned sx, unsigned sy,
                                           const SampleIdBuff *startSampleIdBuff, const unsigned startSampleId)
{
    if (!startSampleIdBuff) return startSampleId;
    return startSampleIdBuff->getSampleId(sx, sy);
}

unsigned
PixSampleSpanRuntimeVerify::getLastSample(unsigned sx, unsigned sy, const FloatBuffer &lastWeightBuff) const
{
    scene_rdl2::fb_util::Tiler tiler(mWidth, mHeight);
    
    unsigned px, py;
    tiler.linearToTiledCoords(sx, sy, &px, &py);
    return static_cast<unsigned>(lastWeightBuff.getPixel(px, py));
}

//------------------------------------------------------------------------------------------

static std::unique_ptr<PixSampleSpanRuntimeVerify> sPixSampleSpanRuntimeVerify;
    
// static function
void
PixSampleSpanRuntimeVerify::init(unsigned width, unsigned height)
{
    sPixSampleSpanRuntimeVerify.reset(new PixSampleSpanRuntimeVerify);
    sPixSampleSpanRuntimeVerify->reset(width, height);
}

// static function
PixSampleSpanRuntimeVerify *
PixSampleSpanRuntimeVerify::get()
{
    return sPixSampleSpanRuntimeVerify.get();
}

#endif // end RUNTIME_VERIFY_PIX_SAMPLE_SPAN

} // namespace rndr
} // namespace moonray

