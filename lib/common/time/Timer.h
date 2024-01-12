// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

//

#pragma once


#include <moonray/common/mcrt_util/Average.h>
#include <scene_rdl2/common/platform/Platform.h>

namespace moonray {
namespace time {


//---------------------------------------------------------------------------

// Returns current time in seconds
__forceinline double getTime()
{
    return scene_rdl2::util::getSeconds();
}


/*
 * Timer that is scope based.
 * timing starts in the constructor and stops in the destructor.
 * It updates a given stat object via its operator+=() method.
 */
template <typename T>
class RAIITimer
{
public:
    __forceinline RAIITimer(T &stat) {
        mStat = &stat;
        mStart = getTime();
    }

    // timer stops and updates upon destruction
    __forceinline ~RAIITimer() {
        (*mStat) += (getTime() - mStart);
    }

private:
    T*  mStat;
    double mStart;
};

typedef RAIITimer<moonray::util::AverageDouble> RAIITimerAverageDouble;


/*
 * Manual timer that has a classic start / stop / lap interface
 * the constructor does NOT start the timer
 * the destructor does NOT stop the timer
 * starting or stopping a timer that is already started / stopped (respectively)
 * will trip a MNRY_ASSERT
 */
template <typename T>
class Timer
{
public:
    typedef T type;

    __forceinline Timer(T &stat) {
        MNRY_DURING_ASSERTS(mStarted = false);
        mStat = &stat;
        mStart = getTime();
    }
    __forceinline ~Timer() {}

    __forceinline void start() {
        MNRY_ASSERT(mStarted == false);
        MNRY_DURING_ASSERTS(mStarted = true);
        mStart = getTime();
    }

    __forceinline void stop() {
        MNRY_ASSERT(mStarted == true);
        MNRY_DURING_ASSERTS(mStarted = false);
        (*mStat) += (getTime() - mStart);
    }

    __forceinline void lap() {
        MNRY_ASSERT(mStarted == true);
        double tick = getTime();
        (*mStat) += (tick - mStart);
        mStart = tick;
    }

private:
    T*  mStat;
    double mStart;
    MNRY_DURING_ASSERTS(bool mStarted);
};


typedef Timer<double> TimerDouble;
typedef Timer<moonray::util::AverageDouble> TimerAverageDouble;


//---------------------------------------------------------------------------

} // namespace time
} // namespace moonray


