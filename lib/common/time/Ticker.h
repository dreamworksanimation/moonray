// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

//

#pragma once


#include <moonray/common/time/Timer.h>
#include <moonray/common/mcrt_util/Average.h>
#include <scene_rdl2/common/platform/Platform.h>

#include <x86intrin.h>

namespace moonray {
namespace time {


//---------------------------------------------------------------------------

// Returns a performance counter in ticks
__forceinline uint64 getTicks()  {
    return __rdtsc();
}


/*
 * This class computes and keeps track of the number of Ticks / Second
 */
class TicksPerSecond {
public:
    TicksPerSecond():
        mStartSeconds(0.0),
        mStartTicks(0),
        mTicksPerSecond(0.0),
        mTicksPerSecondComputed(false)
    {
    }

    void init()
    {
        getTimeSample(&mStartSeconds, &mStartTicks);
        mTicksPerSecondComputed = false;
    }

    double getCached() const
    {
        if (!mTicksPerSecondComputed) {
            mTicksPerSecond = getUncached();
            mTicksPerSecondComputed = true;
        }
        return mTicksPerSecond;
    }

    // This version returns the latest most accurate count at the cost of 
    // recomputing it. Use TicksPerSecond::getCached() to retrieve the fast
    // cached version.
    double getUncached() const
    {
        // You must call init before calling this function.
        MNRY_ASSERT(mStartTicks);

        double endSeconds;
        uint64 endTicks;
        getTimeSample(&endSeconds, &endTicks);

        MNRY_ASSERT(endSeconds > mStartSeconds);

        return (endTicks - mStartTicks) / (endSeconds - mStartSeconds);
    }

private:
    // TODO: For most accurate readings, don't let compiler or CPU reorder our
    // sampling points.
    void getTimeSample(double *seconds, uint64 *ticks) const
    {
        *seconds = time::getTime();
        *ticks = getTicks();
    }

    double mStartSeconds;
    uint64 mStartTicks;
    mutable double mTicksPerSecond;
    mutable bool mTicksPerSecondComputed;
};


/*
 * cpu clock ticks timer that is scope based.
 * timing starts in the constructor and stops in the destructor.
 * It updates a given stat object via its operator+=() method.
 */
template <typename T>
class RAIITicker
{
public:
    __forceinline RAIITicker(T &stat) {
        mStat = &stat;
        mStart = getTicks();
    }

    // timer stops and updates upon destruction
    __forceinline ~RAIITicker() {
        (*mStat) += (getTicks() - mStart);
    }

private:
    T*  mStat;
    int64 mStart;
};


/**
 * This is similar to the RAIITicker but also takes care to decrement
 * the parentStat's exclusive tick count.
 * 
 * You would use this ticker when you have parent-task/child-task relationships
 * in your timings and would like the parent to not include the time spent
 * in the child.
 */
template <typename T>
class RAIIInclusiveExclusiveTicker 
{
public:
    __forceinline RAIIInclusiveExclusiveTicker(
            util::InclusiveExclusiveAverage<T> &stat,
            util::InclusiveExclusiveAverage<T> &parentStat)
    {
        mStat = &stat;
        mParentStat = &parentStat;
        mStart = getTicks();
    }

    __forceinline ~RAIIInclusiveExclusiveTicker() {
        int64 elapsed = (getTicks() - mStart);
        (*mStat) += elapsed;
        mParentStat->decrementExclusive(elapsed);
    }

private:
    util::InclusiveExclusiveAverage<T>* mStat;
    util::InclusiveExclusiveAverage<T>* mParentStat;
    int64 mStart;
};

typedef RAIITicker<util::AverageInt64> RAIITickerAverageInt64;


/*
 * cpu clock ticks manual timer that has a classic start / stop / lap interface
 * the constructor does NOT start the timer
 * the destructor does NOT stop the timer
 * starting or stopping a timer that is already started / stopped (respectively)
 * will trip a MNRY_ASSERT
 */
template <typename T>
class Ticker
{
public:
    __forceinline Ticker(T &stat) {
        MNRY_DURING_ASSERTS(mStarted = false);
        mStat = &stat;
        mStart = getTicks();
    }
    __forceinline ~Ticker() {}

    __forceinline void start() {
        MNRY_ASSERT(mStarted == false);
        MNRY_DURING_ASSERTS(mStarted = true);
        mStart = getTicks();
    }

    __forceinline void stop() {
        MNRY_ASSERT(mStarted == true);
        MNRY_DURING_ASSERTS(mStarted = false);
        (*mStat) += (getTicks() - mStart);
    }

    __forceinline void lap() {
        MNRY_ASSERT(mStarted == true);
        int64 tick = getTicks();
        (*mStat) += (tick - mStart);
        mStart = tick;
    }

private:
    T*  mStat;
    int64 mStart;
    MNRY_DURING_ASSERTS(bool mStarted);
};

typedef Ticker<int64> TickerInt64;


//---------------------------------------------------------------------------

} // namespace time
} // namespace moonray


