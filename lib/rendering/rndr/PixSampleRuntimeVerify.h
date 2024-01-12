// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

//
//
#pragma once

#include "RunTimeVerify.h"

#include <scene_rdl2/common/fb_util/FbTypes.h>

#include <atomic>
#include <mutex>
#include <string>
#include <vector>

namespace moonray {
namespace rndr {

#ifdef RUNTIME_VERIFY_PIX_SAMPLE_COUNT

class PixSampleCountRuntimeVerify
//
// Just accumulate all pixels processed sample and show them for debugging purposes.
// Sometimes it is pretty useful to dump final processed sample totals not using weight value under
// a multi-threaded environment.
// The class has an impact on the execution performance and this class should not be used for
// the release version of moonray. This should be only used for debugging purposes.
//
{
public:
    static void init(unsigned width, unsigned height);
    static PixSampleCountRuntimeVerify *get();

    PixSampleCountRuntimeVerify(unsigned width, unsigned height);
    ~PixSampleCountRuntimeVerify() {};

    void add(unsigned x, unsigned y, unsigned sample) { mPix[pixOffset(x, y)] += sample; }

    std::string show() const;

private:    
    unsigned pixOffset(unsigned x, unsigned y) const { return y * mWidth + x; }
    unsigned getMax() const { return *std::max_element(mPix.begin(), mPix.end()); }

    unsigned mWidth, mHeight;
    std::vector<std::atomic<unsigned>> mPix;
};

#endif // end RUNTIME_VERIFY_PIX_SAMPLE_COUNT

//------------------------------------------------------------------------------------------

#ifdef RUNTIME_VERIFY_PIX_SAMPLE_SPAN

class SampleIdBuff;

class PixSmpSpan
//
// pixel sample span information
//
{
public:
    PixSmpSpan(unsigned startSampleId, unsigned endSampleId) :
        mStartSampleId(startSampleId),
        mEndSampleId(endSampleId)
    {}

    unsigned getStartSampleId() const { return mStartSampleId; }
    unsigned getEndSampleId() const { return mEndSampleId; }

    std::string show(int sampleValueFieldWidth) const;

private:
    unsigned mStartSampleId;
    unsigned mEndSampleId;
};

bool operator < (const PixSmpSpan &A, const PixSmpSpan &B);

class PixSmp
//
// pixel sample information. This includes all processed sample spans
//
{
public:
    PixSmp() {};
    PixSmp(const PixSmp &src) : mSamples(src.mSamples) { }

    void reset() { std::lock_guard<std::mutex> lock(mMutex); mSamples.clear(); }

    void add(unsigned startSampleId, unsigned endSampleId)
    {
        std::lock_guard<std::mutex> lock(mMutex);
        mSamples.emplace_back(startSampleId, endSampleId);
    }

    unsigned getBiggestEndSampleId() const { return getBiggestEndSampleId(mSamples); }

    bool verify(const unsigned startSample, const unsigned lastSample) const;

    std::string show() const;

private:
    using PixSmpSpanArray = std::vector<PixSmpSpan>;

    static std::string showSamples(const PixSmpSpanArray &samples);
    static unsigned getBiggestEndSampleId(const PixSmpSpanArray &sampales);

    mutable std::mutex mMutex;
    PixSmpSpanArray mSamples;
};

class PixSampleSpanRuntimeVerify
//
// This class records all sampling spans regarding all pixels by multiple threads and verifies them
// after finishing render.
// The class has an impact on the execution performance and this class should not be used for
// the release version of moonray. This should be only used for debugging purposes.
//
{
public:
    using FloatBuffer = scene_rdl2::fb_util::FloatBuffer;

    PixSampleSpanRuntimeVerify() :
        mWidth(0),
        mHeight(0)
    {};

    static void init(unsigned width, unsigned height);
    static PixSampleSpanRuntimeVerify *get();

    void reset(unsigned width, unsigned height);

    void setInitialSample(const FloatBuffer &weightBuff);

    void add(unsigned sx, unsigned sy, unsigned startSample, unsigned endSample)
    {
        mPixels[pixOffset(sx, sy)].add(startSample, endSample);
    }

    unsigned getBiggestEndSampleId(unsigned sx, unsigned sy) const {
        return mPixels[pixOffset(sx, sy)].getBiggestEndSampleId();
    }

    bool verify(const SampleIdBuff *startSampleIdBuff,
                const unsigned startSampleId,
                const FloatBuffer &lastWeightBuff) const;

    std::string show(unsigned x, unsigned y) const;
    std::string show(unsigned x,
                     unsigned y,
                     const SampleIdBuff *startSampleIdBuff,
                     const unsigned startSampleId,
                     const FloatBuffer &lastWeightBuff) const;

private:
    unsigned pixOffset(unsigned sx, unsigned sy) const { return sy * mWidth + sx; }
    std::string showPixInfo(unsigned sx, unsigned sy) const { return mPixels[pixOffset(sx, sy)].show(); }

    static unsigned getStartSample(unsigned sx, unsigned sy,
                                   const SampleIdBuff *startSampleIdBuff, const unsigned startSampleId);
    unsigned getLastSample(unsigned sx, unsigned sy, const FloatBuffer &lastWeightBuff) const;

    template <typename F>
    bool
    crawlAllPix(F pixFunc) const
    {
        for (unsigned y = 0; y < mHeight; ++y) {
            for (unsigned x = 0; x < mWidth; ++x) {
                if (!pixFunc(x, y)) return false;
            }
        }
        return true;
    }

    //------------------------------

    unsigned mWidth;
    unsigned mHeight;
    std::vector<PixSmp> mPixels;
};

#endif // end RUNTIME_VERIFY_PIX_SAMPLE_SPAN

} // namespace rndr
} // namespace moonray

