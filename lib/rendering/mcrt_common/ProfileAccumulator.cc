// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

//
//
#include "ProfileAccumulator.h"
#include <moonray/common/mcrt_macros/moonray_static_check.h>
#include <scene_rdl2/render/logging/logging.h>

#include <algorithm>
#include <string>
#include <vector>

namespace moonray {
namespace mcrt_common {

bool gAccumulatorsActive = false;
MNRY_DURING_ASSERTS(alignas(CACHE_LINE_SIZE) std::atomic_int gNumAccumulatorsActive);

namespace
{

struct Private
{
    Private() :
        mRoot(nullptr),
        mNumThreads(0),
        mNumTLS(0) {}

    Accumulator *mRoot;

    unsigned mNumThreads;
    unsigned mNumTLS;

    std::vector<Accumulator *> mAccumulators;
};

Private gPrivate;

}   // End of anon namespace.

//-----------------------------------------------------------------------------

Accumulator *
initAccumulators(unsigned numThreads, unsigned numTLS)
{
    MNRY_ASSERT(gPrivate.mRoot == nullptr && numThreads > 0 && numTLS >= numThreads);

    MOONRAY_THREADSAFE_STATIC_WRITE(gPrivate.mNumThreads = numThreads);
    MOONRAY_THREADSAFE_STATIC_WRITE(gPrivate.mNumTLS = numTLS);

    // This is freed when we are freeing the contents of the gPrivate.mAccumulators array.
    MOONRAY_THREADSAFE_STATIC_WRITE(gPrivate.mRoot = scene_rdl2::util::alignedMallocCtorArgs<Accumulator>
                                  (CACHE_LINE_SIZE, "Root", 0, numTLS, ACCFLAG_NONE));

    gPrivate.mAccumulators.push_back(gPrivate.mRoot);

    return gPrivate.mRoot;
}

void
cleanUpAccumulators()
{
    if (!gPrivate.mRoot) {
        return;
    }

    for (size_t i = 0; i < gPrivate.mAccumulators.size(); ++i) {
        scene_rdl2::util::alignedFreeDtor(gPrivate.mAccumulators[i]);
    }

    gPrivate.mAccumulators.clear();

    MOONRAY_THREADSAFE_STATIC_WRITE(gPrivate.mNumThreads = 0);
    MOONRAY_THREADSAFE_STATIC_WRITE(gPrivate.mNumTLS = 0);
    MOONRAY_THREADSAFE_STATIC_WRITE(gPrivate.mRoot = nullptr);
}

Accumulator *
allocAccumulator(const char *desc, AccumulatorFlags flags)
{
    MNRY_ASSERT(gPrivate.mNumTLS);

    unsigned index = (unsigned)gPrivate.mAccumulators.size();

    Accumulator *acc = scene_rdl2::util::alignedMallocCtorArgs<Accumulator>
                            (CACHE_LINE_SIZE, desc, index, gPrivate.mNumTLS, flags);

    gPrivate.mAccumulators.push_back(acc);

    return acc;
}

unsigned
getNumAccumulators()
{
    return (unsigned)gPrivate.mAccumulators.size();
}

unsigned
getNumAccumulatorThreads()
{
    return gPrivate.mNumThreads;
}

void
setAccumulatorActiveState(bool active)
{
    MOONRAY_THREADSAFE_STATIC_WRITE(gAccumulatorsActive = active);
}

bool
getAccumulatorActiveState()
{
    return gAccumulatorsActive;
}

void
resetAllAccumulators()
{
    MNRY_ASSERT(gNumAccumulatorsActive == 0);

    for (unsigned i = 0; i < gPrivate.mAccumulators.size(); ++i) {
        gPrivate.mAccumulators[i]->reset();
    }

    MNRY_ASSERT(gNumAccumulatorsActive == 0);
}

unsigned
snapshotRawAccumulators(std::vector<AccumulatorResult> *dstResults, double rcpTickFrequency)
{
    MNRY_ASSERT(gNumAccumulatorsActive == 0);
    MNRY_ASSERT(gPrivate.mRoot && gPrivate.mNumThreads);
    MNRY_ASSERT(dstResults);
    MNRY_ASSERT(rcpTickFrequency > 0.0);

    uint64_t rootTicks = gPrivate.mRoot->getAccumulatedTicks();
    if (rootTicks == 0) {
        return 0;
    }

    const size_t numAccumulators = gPrivate.mAccumulators.size();

    // "results" is an alias for dstResults.
    std::vector<AccumulatorResult> &results = *dstResults;
    results.resize(numAccumulators);

    // Root is assumed to be only timed on the main thread outside of any
    // rendering threads. Because of this, we don't need to divide by the number
    // of threads to get the actual wall clock time.
    const double wallClockTime = double(rootTicks) * rcpTickFrequency;

    // Root result is a special case since it's only run on one thread.
    // The root accumulator is always the first in the array.
    MNRY_ASSERT(gPrivate.mAccumulators[0] == gPrivate.mRoot);

    AccumulatorResult &rootResult = results[0];
    rootResult.mName = gPrivate.mRoot->mName.c_str();
    rootResult.mFlags = gPrivate.mRoot->mFlags;
    rootResult.mTotalTime = wallClockTime * double(gPrivate.mNumThreads);
    rootResult.mTimePerThread = wallClockTime;
    rootResult.mPercentageOfTotal = 100.0;

    // Gather remaining results.
    const double rcpNumThreads = 1.0 / double(gPrivate.mNumThreads);
    const double rcpWallClockTime = 1.0 / wallClockTime;
    unsigned numResults = 1;

    for (unsigned iacc = 1; iacc < gPrivate.mAccumulators.size(); ++iacc) {

        Accumulator *acc = MNRY_VERIFY(gPrivate.mAccumulators[iacc]);
        AccumulatorResult &result = results[numResults];

        uint64_t ticks = acc->getAccumulatedTicks();

        result.mName = acc->mName.c_str();
        result.mFlags = acc->mFlags;
        result.mTotalTime = double(ticks) * rcpTickFrequency;
        result.mTimePerThread = result.mTotalTime * rcpNumThreads;
        result.mPercentageOfTotal = result.mTimePerThread * rcpWallClockTime * 100.0;

        ++numResults;
    }

    // The root accumulator should always be at the start of the list.
    MNRY_ASSERT(results[0].mName == gPrivate.mRoot->mName.c_str());

    MNRY_ASSERT(gNumAccumulatorsActive == 0);

    return (unsigned)numAccumulators;
}

unsigned
snapshotSortedAccumulators(std::vector<AccumulatorResult> *dstResults,
                           double rcpTickFrequency,
                           double threshold)
{
    if (!snapshotRawAccumulators(dstResults, rcpTickFrequency)) {
        return 0;
    }

    // "results" is an alias for dstResults.
    std::vector<AccumulatorResult> &results = *dstResults;

    // Cull results which fall below the threshold.
    auto newEnd = std::remove_if(results.begin(), results.end(), [&](const AccumulatorResult &acc) {
        return acc.mTotalTime < threshold || ((acc.mFlags & ACCFLAG_DISPLAYABLE) == 0);
    });

    results.erase(newEnd, results.end());

    // Greater than.
    std::stable_sort(results.begin(), results.end(), [](const AccumulatorResult &a, const AccumulatorResult &b) -> bool
    {
        MNRY_ASSERT(a.mName && b.mName);

        bool bracketA = (a.mName[0] == '[');
        bool bracketB = (b.mName[0] == '[');

        if (bracketA) {
            if (bracketB) {
                return a.mTimePerThread > b.mTimePerThread;
            } else {
                return false;
            }
        } else {
            if (bracketB) {
                return true;
            } else {
                // If an entry is marked with the TOTAL flag, then sort it
                // at the end of the exclusive times, but before any bracketed
                // times.
                if (a.mFlags & ACCFLAG_TOTAL) return false;
                if (b.mFlags & ACCFLAG_TOTAL) return true;
                return a.mTimePerThread > b.mTimePerThread;
            }
        }
    });

    return (unsigned)results.size();
}

//-----------------------------------------------------------------------------

Accumulator::Accumulator(const char *desc, unsigned index, unsigned numTLS, AccumulatorFlags flags) :
    mName(desc ? desc : "Unnamed"),
    mIndex(index),
    mThreadLocal(nullptr),
    mFlags(flags)
{
    mThreadLocal = scene_rdl2::util::alignedMallocArrayCtor<ThreadLocalAccumulator>(numTLS, CACHE_LINE_SIZE);
}

Accumulator::~Accumulator()
{
    scene_rdl2::util::alignedFreeArrayDtor(mThreadLocal, gPrivate.mNumTLS);
}

void Accumulator::reset()
{
    for (unsigned i = 0; i < gPrivate.mNumTLS; ++i) {
        mThreadLocal[i].reset();
    }
}

uint64_t Accumulator::getAccumulatedTicks() const
{
    uint64_t localTicks = 0;

    for (unsigned i = 0; i < gPrivate.mNumThreads; ++i) {

        const auto &tla = mThreadLocal[i];

        if (tla.mTotalTime > 0) {
            localTicks += tla.mTotalTime;
        }
    }

    return localTicks;
}

//-----------------------------------------------------------------------------

} //namespace mcrt_common
} //namespace moonray

