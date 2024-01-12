// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

/// @file Clock.h
#pragma once

#include <time.h>

namespace moonray {
namespace mcrt_common {

/// @brief an RAII interface to unix clock_gettime()
/// Usage:
/// \code
///    int64_t nsec = 0;
///    {
///        Clock(&nsec);
///        ...
///    }
///    double secs = Clock::seconds(nsec);
/// \endcode

class Clock
{
public:
    // Clock counts number of nanoseconds, convert to seconds.
    __forceinline static double seconds(int64_t nsecs)
    {
        return static_cast<double>(nsecs) / NSPERSEC;
    }

    // Seconds convert to clock counts number of nanoseconds
    __forceinline static int64_t nanoseconds(double sec)
    {
        return static_cast<int64_t>(sec * NSPERSEC);
    }

    __forceinline Clock(int64_t *stat,
                        bool startNow = true,
                        clockid_t clkId = CLOCK_THREAD_CPUTIME_ID):
        mStat(stat),
        mClkId(clkId),
        mStopped(true)
    {
        if (startNow) {
            start();
        }
    }

    __forceinline ~Clock()
    {
        stop();
    }

    __forceinline void start()
    {
        if (mStat && mStopped) {
            clock_gettime(mClkId, &mStart);
            mStopped = false;
        }
    }

    __forceinline void stop()
    {
        if (mStat && !mStopped) {
            struct timespec end;
            clock_gettime(mClkId, &end);
            const int64_t sdiff = end.tv_sec - mStart.tv_sec;
            const int64_t nsdiff = end.tv_nsec - mStart.tv_nsec;
            *mStat += sdiff * NSPERSEC + nsdiff;
            mStopped = true;
        }
    }

private:
    static const int64_t NSPERSEC = 1000000000;
    int64_t *mStat; // inactive if null
    clockid_t mClkId;
    struct timespec mStart;
    bool mStopped;
};

#define MCRT_COMMON_CLOCK_OPEN(STAT) { mcrt_common::Clock clock((STAT));

#define MCRT_COMMON_CLOCK_CLOSE() }
        
} // namespace mcrt_common
} // namespace moonray

