// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0


#pragma once

#include <scene_rdl2/common/math/MathUtil.h>
#include <scene_rdl2/common/platform/Platform.h>
#include <scene_rdl2/render/util/BitUtils.h>

#include <tbb/mutex.h>

namespace moonray {
    constexpr int getMutexCount(int log2MutexCount)
    {
        return 1 << log2MutexCount;
    }

#if 1
    template <int log2MutexCount>
    inline unsigned getMutexIndex(unsigned px, unsigned py) noexcept
    {
        constexpr int mutexCount = getMutexCount(log2MutexCount);

        // This function allows us to look up which mutex to use from a
        // power-of-two sized array. It will allow us to access the mutexes in a
        // way that neighboring cells in a 2D grid don't access the same mutex.
        static_assert(scene_rdl2::util::isPowerOfTwo(mutexCount), "Must be a power of two");

        // The "& ~1" isn't strictly necessary, but it generates more efficient code on every value I've tested. It
        // makes the number even.
        // The "| 1" makes the number odd, so that it's co-prime with the power of two mutexCount.
        const unsigned idx = py * (mutexCount/(log2MutexCount & ~1) | 1) + px;

        // mutexCount should be a power of two. The compiler should change
        // this mod operation into a mask.
        return idx % mutexCount;
    }

    template <>
    inline unsigned getMutexIndex<0>(unsigned px, unsigned py) noexcept
    {
        return 0;
    }

    template <>
    inline unsigned getMutexIndex<1>(unsigned px, unsigned py) noexcept
    {
        constexpr int mutexCount = 2;
        const unsigned idx = py + px;

        // mutexCount should be a power of two. The compiler should change
        // this mod operation into a mask.
        return idx % mutexCount;
    }
#endif

    template <int sLog2MutexCount, typename MutexType = tbb::mutex>
    class MutexPool2D
    {
    public:
        MutexType& getMutex(unsigned px, unsigned py)
        {
            return mMutexList[getMutexIndex<sLog2MutexCount>(px, py)].mMutex;
        }

        void exclusiveLockAll()
        {
            for (auto& m : mMutexList) {
                m.mMutex.lock();
            }
        }

        void unlockAll()
        {
            for (auto& m : mMutexList) {
                m.mMutex.unlock();
            }
        }

    private:
        struct alignas(CACHE_LINE_SIZE) MutexPad
        {
            // Padded to avoid false sharing.
            MutexType mMutex;
        };
        std::array<MutexPad, getMutexCount(sLog2MutexCount)> mMutexList;
    };

} // namespace moonray

