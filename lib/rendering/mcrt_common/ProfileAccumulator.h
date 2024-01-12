// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

//
//
// Functionality for profiling the render portion of the frame. There are some
// assumptions which make it unsuitable for the update portion. These are:
//
//  1. All threads are running flat out.
//  2. All threads are equally likely to to work on any particular job.
//
// This allows us to divide "thread wall clock time" by the thread count
// to accurately estimate actual wall clock time.
//
#pragma once
#include <scene_rdl2/render/util/Memory.h>
#include <scene_rdl2/render/util/MiscUtils.h>
#include <x86intrin.h>

#include <atomic>
#include <cstring>

// UNIQUE_IDENTIFIER returns an unique identifer for each line of a source file.
#define UNIQUE_INNER(prefix, x)     prefix##x
#define UNIQUE_OUTER(prefix, x)     UNIQUE_INNER(prefix, x)
#define UNIQUE_IDENTIFIER           UNIQUE_OUTER(unique_id_, __LINE__)

// Comment out this line to turn off scoped accumulators.
#define PROFILE_ACCUMULATORS_ENABLED

namespace moonray {
namespace mcrt_common {

__forceinline uint64_t
getProfileAccumulatorTicks()
{
#if 1
    return __rdtsc();
#else
    unsigned aux;
    return __rdtscp(&aux);
#endif
}

enum AccumulatorFlags
{
    ACCFLAG_NONE            = 0x0000,

    ACCFLAG_DISPLAYABLE     = 0x0001,

    // Sorts this entry at the end after all non-bracketed but before
    // any bracketed entries. At most one accumulator should be initialized
    // with this flag.
    ACCFLAG_TOTAL           = 0x0002,
};

struct Accumulator;
struct ThreadLocalAccumulator;

extern bool gAccumulatorsActive;
MNRY_DURING_ASSERTS(extern alignas(CACHE_LINE_SIZE) std::atomic_int gNumAccumulatorsActive);

struct AccumulatorResult
{
    AccumulatorResult() :
        mName(nullptr),
        mFlags(ACCFLAG_NONE),
        mTotalTime(0.0),
        mTimePerThread(0.0),
        mPercentageOfTotal(0.0) {}

    const char *mName;
    AccumulatorFlags mFlags;

    // The total seconds spent over all threads.
    double mTotalTime;

    // The average time spent for each thread.
    double mTimePerThread;

    // Assuming we were running in parallel the whole time, this is the percentage
    // of the total time recorded by this accumulator.
    double mPercentageOfTotal;
};

//-----------------------------------------------------------------------------

#ifdef PROFILE_ACCUMULATORS_ENABLED

// Public interface

// Entry point to activate an accumulator. Assumes the existence of a
// getAccumulatorThreadIndex function which will return the current thread index
// given the first arg to this macro.
#define ACCUMULATOR_PROFILE(thread, acc)                    moonray::mcrt_common::ScopedAccumulator UNIQUE_IDENTIFIER(getAccumulatorThreadIndex(thread), getAccumulator(acc))

// Alternate begin/end convenience interface for invoking ProfileAccumulator.
// Automatic scoping to current code block is still done if the ACCUMULATOR_STOP call is omitted.
#define ACCUMULATOR_GET_AND_START(thread, acc, handle)      moonray::mcrt_common::ScopedAccumulator handle(getAccumulatorThreadIndex(thread), getAccumulator(acc))
#define ACCUMULATOR_GET_PAUSED(thread, acc, handle)         moonray::mcrt_common::ScopedAccumulator handle(getAccumulatorThreadIndex(thread), getAccumulator(acc), true)
#define ACCUMULATOR_PAUSE(handle)                           (handle).pause()
#define ACCUMULATOR_UNPAUSE(handle)                         (handle).unpause()
#define ACCUMULATOR_STOP(handle)                            (handle).stop()

#else   // #ifdef PROFILE_ACCUMULATORS_ENABLED

//
// Empty stubs if this code is compiled out.
//

#define ACCUMULATOR_PROFILE(thread, acc)
#define ACCUMULATOR_GET_AND_START(thread, acc, handle)
#define ACCUMULATOR_GET_PAUSED(thread, acc, handle)
#define ACCUMULATOR_PAUSE(handle)
#define ACCUMULATOR_UNPAUSE(handle)
#define ACCUMULATOR_STOP(handle)

#endif // #ifdef PROFILE_ACCUMULATORS_ENABLED


//
// We provide 2 separate parameters here since when initializing the global
// TLS pool, we create 2 extra TLS objects also initTLSPhase2
// (see ThreadLocalState.cc). One for use by the main thread, which is important
// during the scene building phase of a frame. The other is a special GUI thread
// TLS to allow picking, and scene intersections.
//
// If code which uses scoped accumulators is invoked from either of the 2 above
// thread, then we'd write past the end of our ThreadLocalAccumulator
// array if we only allocated an array sized for numThreads. Therefore to
// avoid this issue, we allow the number of TLS objects to be allocated separately.
// These extra TLS objects (over the number of render threads) are ignored when
// tallying up the profiling stats.
//
// The root accumulator is returned.
//
Accumulator *initAccumulators(unsigned numThreads, unsigned numTLS);

// Cleans up everything. It is ok to call initAccumulators after calling this.
void cleanUpAccumulators();

//
// If desc starts with the '[' character, it's sorted after non '[' descs when
// displaying stats.
//
Accumulator *allocAccumulator(const char *desc, AccumulatorFlags flags);

// How many accumulators are allocated in total (including the root accumulator).
unsigned getNumAccumulators();

// How many threads are accumulators recording ticks for in parallel.
unsigned getNumAccumulatorThreads();

void setAccumulatorActiveState(bool active);
bool getAccumulatorActiveState();

// Initialize all the counters to 0.
// It's only valid to call this when no accumulators are currently active.
void resetAllAccumulators();

//
// It's only valid to call snapshot<Type>Accumulator when no accumulators are
// currently active. The return value contains the number of AccumulatorResult
// elements filled in.
//
// snapshotRawAccumulators will always returns the AccumulatorResult array in
// the order which Accumulators were originally created.
//
// snapshotSortedAccumulators will cull any time readings which are less than
// threshold and return an array sorted by time, highest to lowest.
//
unsigned snapshotRawAccumulators(std::vector<AccumulatorResult> *dstResults,
                                 double rcpTickFrequency);

unsigned snapshotSortedAccumulators(std::vector<AccumulatorResult> *dstResults,
                                    double rcpTickFrequency, double threshold = 0.00001f);

//-----------------------------------------------------------------------------

// Private implementation:

// Per thread data for a particular accumulator, doesn't support nesting.
struct CACHE_ALIGN ThreadLocalAccumulator
{
    ThreadLocalAccumulator()
    {
        reset();
    }

    void reset()
    {
        // This function clears the entire cache line!!
        MNRY_STATIC_ASSERT(alignof(ThreadLocalAccumulator) == CACHE_LINE_SIZE);
        MNRY_STATIC_ASSERT(sizeof(*this) == CACHE_LINE_SIZE);
        MNRY_ASSERT(gNumAccumulatorsActive == 0);
        memset(this, 0, CACHE_LINE_SIZE);
    }

    __forceinline void start()
    {
        // If this assert triggers, this accumulator is probably nested.
        MNRY_ASSERT(!mTimerActive);

        MNRY_DURING_ASSERTS(++gNumAccumulatorsActive);
        mLastStartTime = getProfileAccumulatorTicks();
        ++mTotalCallCount;
        mTimerActive = true;
    }

    __forceinline void stop()
    {
        // If this assert triggers, this accumulator is probably nested.
        MNRY_ASSERT(mTimerActive);

        mTimerActive = false;
        uint64_t endTime = getProfileAccumulatorTicks();
        MNRY_ASSERT(endTime >= mLastStartTime);
        mTotalTime += endTime - mLastStartTime;
        MNRY_ASSERT(--gNumAccumulatorsActive >= 0);
    }

    __forceinline bool canStart() const
    {
        return !mTimerActive;
    }

    __forceinline bool canStop() const
    {
        return mTimerActive;
    }

    uint64_t    mLastStartTime;
    uint64_t    mTotalTime;
    unsigned    mTotalCallCount;
    bool        mTimerActive;
};

// Ensure that the derived class also fits in a cache line!
MNRY_STATIC_ASSERT(sizeof(ThreadLocalAccumulator) <= CACHE_LINE_SIZE);

//
// Container for all the ThreadLocalAccumulators associated with a particular "tag".
//
struct Accumulator
{
    Accumulator(const char *desc, unsigned index, unsigned numTLS, AccumulatorFlags flags);
    ~Accumulator();

    void reset();
    uint64_t getAccumulatedTicks() const;

    std::string             mName;
    unsigned                mIndex; // A zero based index determined by the creation order.
    ThreadLocalAccumulator *mThreadLocal;
    AccumulatorFlags        mFlags;
};

//
// Helper class which adds RAII functionality.
//
class ScopedAccumulator
{
public:
    __forceinline ScopedAccumulator(unsigned threadIdx, Accumulator *acc) :
        mTLAccumulator(nullptr)
    {
        if (gAccumulatorsActive) {
            mTLAccumulator = &MNRY_VERIFY(acc)->mThreadLocal[threadIdx];
            mTLAccumulator->start();
        }
    }

    // The startPaused parameter is not used directly. It's here so we can
    // differentiate between the different constructors.
    __forceinline ScopedAccumulator(unsigned threadIdx, Accumulator *acc, bool startPaused) :
        mTLAccumulator(nullptr)
    {
        if (gAccumulatorsActive) {
            mTLAccumulator = &MNRY_VERIFY(acc)->mThreadLocal[threadIdx];
        }
    }

    __forceinline ~ScopedAccumulator()
    {
        // The ThreadLocalAccumulator may be in a stopped state already. We only
        // need to stop it here if it's not.
        if (mTLAccumulator && mTLAccumulator->canStop()) {
            mTLAccumulator->stop();
        }
    }

    // Use the ACCUMULATOR_START macro to trigger this.
    __forceinline void start()
    {
        MNRY_ASSERT(mTLAccumulator && mTLAccumulator->canStart());
        mTLAccumulator->start();
    }

    // Use the ACCUMULATOR_STOP macro to trigger this.
    __forceinline void stop()
    {
        MNRY_ASSERT(mTLAccumulator && mTLAccumulator->canStop());
        mTLAccumulator->stop();
        mTLAccumulator = nullptr;
    }

    // Use the ACCUMULATOR_PAUSE macro to trigger this.
    __forceinline void pause()
    {
        MNRY_ASSERT(mTLAccumulator && mTLAccumulator->canStop());
        mTLAccumulator->stop();
    }

    // Use the ACCUMULATOR_UNPAUSE macro to trigger this.
    __forceinline void unpause()
    {
        MNRY_ASSERT(mTLAccumulator && mTLAccumulator->canStart());
        mTLAccumulator->start();
    }

private:
    ThreadLocalAccumulator *mTLAccumulator;
};

} //namespace mcrt_common

// Override as necessary for higher level libraries.
inline unsigned
getAccumulatorThreadIndex(unsigned threadIdx)
{
    return threadIdx;
}

// Override as necessary for higher level libraries.
inline mcrt_common::Accumulator *
getAccumulator(mcrt_common::Accumulator *acc)
{
    return acc;
}

} //namespace moonray


