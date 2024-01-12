// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

//
//
#pragma once

#include "scene_rdl2/common/fb_util/Tiler.h"

#include <moonray/common/mcrt_util/Atomic.h>

#include <functional>
#include <iostream>

namespace moonray {
namespace rndr {

class SampleIdBuff
//
// This class is used to keep sampleId for all pixels individually and increase them in an atomic way.
// Also provides API for verification of final condition.
//
{
public:
    using FloatBuffer = scene_rdl2::fb_util::FloatBuffer;
    using Tiler = scene_rdl2::fb_util::Tiler;

    SampleIdBuff() :
        mWidth(0),
        mHeight(0),
        mPixDepth(0)
    {}

    bool isValid() const { return (mWidth != 0 && mHeight != 0); }

    void init(const FloatBuffer &srcWeightBuff, unsigned maxSampleId);

    unsigned getSampleId(int sx, int sy) const { return mGetFunc(sx, sy); }
    void setSampleId(int sx, int sy, unsigned sampleId) { mSetFunc(sx, sy, sampleId); }
    unsigned atomicReserveSampleId(int sx, int sy, unsigned sampleDelta) {
        return mAtomicReserveFunc(sx, sy, sampleDelta);
    }

    bool verify(const FloatBuffer &weightBuff) const; // Run verification as compared with final weight buff

    std::string show() const;

private:
    using GetFunc = std::function<unsigned(int sx, int sy)>;
    using SetFunc = std::function<void(int sx, int sy, unsigned v)>;
    using AtomicReserveFunc = std::function<unsigned(int sx, int sy, unsigned sampleDelta)>;
    using GetSrcWeightFunc = std::function<unsigned(int sx, int sy)>;

    void initSub(unsigned width, unsigned height, unsigned maxSampleId);
    void setupSrcWeight(unsigned maxSampleId, const GetSrcWeightFunc &getSrcWeightFunc);
    unsigned getSampleId(const FloatBuffer &buff, unsigned sx, unsigned sy) const;
    
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

    template <typename T>
    uintptr_t pixAddr(int sx, int sy) const
    {
        return (reinterpret_cast<uintptr_t>(mArray.get()) +
                static_cast<uintptr_t>(mTiler.linearCoordsToTiledOffset(sx, sy) * sizeof(T)));
    }

    template <typename T>
    unsigned getPix(int sx, int sy) const
    {
        return static_cast<unsigned>(*reinterpret_cast<T *>(pixAddr<T>(sx, sy)));
    }

    template <typename T>
    void setPix(int sx, int sy, unsigned v)
    {
        *reinterpret_cast<T *>(pixAddr<T>(sx, sy)) = static_cast<T>(v);
    }

    template <typename T>
    unsigned atomicReservePix(int sx, int sy, unsigned sampleDelta)
    {
        T castDelta = static_cast<T>(sampleDelta);
        T *addr = reinterpret_cast<T*>(pixAddr<T>(sx, sy));
        T expected = util::atomicLoad(addr, std::memory_order_relaxed);
        while (true) {
            T desired = expected + castDelta;
            if (util::atomicCompareAndSwapWeak(addr, expected, desired, std::memory_order_relaxed)) break;
        }
        return static_cast<unsigned>(expected);
    }

    //------------------------------

    unsigned mWidth;
    unsigned mHeight;
    Tiler mTiler;
    unsigned mPixDepth;         // byte. 1, 2 or 4

    // Store all sample id data by the continuous memory buffer .
    // Pixel value depth is dynamically adjusted 1, 2, or 4 bytes based on the max sampling id.
    // And use predefined access functions depending on the pixel value depth.
    std::unique_ptr<char[]> mArray;
    GetFunc mGetFunc;
    SetFunc mSetFunc;
    AtomicReserveFunc mAtomicReserveFunc;
};

} // namespace rndr
} // namespace moonray

