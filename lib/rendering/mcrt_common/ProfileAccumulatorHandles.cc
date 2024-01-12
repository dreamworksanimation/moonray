// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

//
//

#include "ProfileAccumulatorHandles.h"

namespace moonray {
namespace mcrt_common {

namespace {

#ifdef PROFILE_ACCUMULATORS_ENABLED

inline unsigned
getAccIndex(const Accumulator *acc)
{
    return MNRY_VERIFY(acc)->mIndex;
}

inline unsigned
getAccIndex(ExclAccType acc)
{
    MNRY_ASSERT(acc < NUM_EXCLUSIVE_ACC);
    return gAccumulatorHandles.mExclusiveAccumulators[acc]->mIndex;
}

inline unsigned
getAccIndex(OverlappedAccType acc)
{
    MNRY_ASSERT(acc < NUM_OVERLAPPED_ACC);
    return gAccumulatorHandles.mOverlappingAccumulators[acc]->mIndex;
}

inline unsigned
getAccIndex(InternalAccType acc)
{
    MNRY_ASSERT(acc < NUM_INTERNAL_ACC);
    return gAccumulatorHandles.mInternalAccumulators[acc]->mIndex;
}

#define ACC(acc)    (times[getAccIndex(acc)])

// Times passed in are assumed to all be per-thread (aka wallclock) times.
void
conditionTimes(std::vector<double> &times)
{
    //
    // This logic is common for both bundled and non-bundled execution.
    //

    ACC(INT_ACCUM_RENDER_DRIVER_SERIAL) = ACC(ACCUM_ROOT) - ACC(ACCUM_RENDER_DRIVER_PARALLEL);

    double totalExclTime = 0.0;
    for (unsigned i = 0; i < NUM_EXCLUSIVE_ACC; ++i) {
        double time = ACC(ExclAccType(i));
        MNRY_ASSERT_REQUIRE(time >= 0.0);
        totalExclTime += time;
    }

    ACC(INT_ACCUM_MISSING_TIME) = std::max(ACC(ACCUM_RENDER_DRIVER_PARALLEL) - totalExclTime, 0.0);
    ACC(INT_ACCUM_TOTALS) = ACC(ACCUM_ROOT);

    // Add up totals for various categories.

    ACC(INT_ACCUM_TOTAL_EMBREE) = ACC(EXCL_ACCUM_EMBREE_INTERSECTION) +
                                  ACC(EXCL_ACCUM_EMBREE_OCCLUSION) +
                                  ACC(EXCL_ACCUM_EMBREE_PRESENCE) +
                                  ACC(EXCL_ACCUM_EMBREE_VOLUME);

    ACC(INT_ACCUM_TOTAL_INTEGRATION) = ACC(EXCL_ACCUM_INTEGRATION) +
                                       ACC(EXCL_ACCUM_SSS_INTEGRATION) +
                                       ACC(EXCL_ACCUM_VOL_INTEGRATION) +
                                       ACC(EXCL_ACCUM_POST_INTEGRATION);

    ACC(INT_ACCUM_TOTAL_RAY_INTERSECTION) = ACC(INT_ACCUM_TOTAL_EMBREE) +
                                            ACC(EXCL_ACCUM_RAY_HANDLER) +
                                            ACC(EXCL_ACCUM_OCCL_QUERY_HANDLER) +
                                            ACC(EXCL_ACCUM_PRESENCE_QUERY_HANDLER) +
                                            ACC(EXCL_ACCUM_RAY_SORT_KEY_GEN);

    ACC(INT_ACCUM_TOTAL_SHADING) = ACC(EXCL_ACCUM_SHADE_HANDLER) +
                                   ACC(EXCL_ACCUM_DEFER_SHADE_ENTRIES) +
                                   ACC(EXCL_ACCUM_SHADING);

    ACC(INT_ACCUM_TOTAL_SCALAR_TIME) = ACC(INT_ACCUM_TOTALS) - (ACC(INT_ACCUM_TOTAL_EMBREE) + ACC(ACCUM_TOTAL_ISPC));
}

#endif  // #ifdef PROFILE_ACCUMULATORS_ENABLED

}  // End of anon namespace.

AccumulatorHandles gAccumulatorHandles;

AccumulatorHandles::AccumulatorHandles()
{
    // False positive from cppcheck, this struct doesn't contain a std::string!
    // cppcheck-suppress memsetClass // (error) Using 'memset' on struct that contains a 'std::string'
    memset(this, 0, sizeof(*this));
}

void
AccumulatorHandles::init(unsigned numThreads, unsigned numTLS)
{
    MNRY_ASSERT(mRoot == nullptr);

    // Top level accumulator.
    mRoot = initAccumulators(numThreads, numTLS);

    //
    // Accumulators to aid measuring exclusive blocks of time during the MCRT phase.
    //
    mExclusiveAccumulators[EXCL_ACCUM_AOVS]                     = allocAccumulator("AOVs", ACCFLAG_DISPLAYABLE);
    mExclusiveAccumulators[EXCL_ACCUM_ADD_HEAT_MAP_HANDLER]     = allocAccumulator("Add heat map handler", ACCFLAG_DISPLAYABLE);
    mExclusiveAccumulators[EXCL_ACCUM_ADD_SAMPLE_HANDLER]       = allocAccumulator("Add sample handler", ACCFLAG_DISPLAYABLE);
    mExclusiveAccumulators[EXCL_ACCUM_COMPUTE_RAY_DERIVATIVES]  = allocAccumulator("Compute ray derivatives", ACCFLAG_DISPLAYABLE);
    mExclusiveAccumulators[EXCL_ACCUM_DEFER_SHADE_ENTRIES]      = allocAccumulator("Defer shade entries (WARNING SIGN!)", ACCFLAG_DISPLAYABLE);
    mExclusiveAccumulators[EXCL_ACCUM_EMBREE_INTERSECTION]      = allocAccumulator("Embree intersection rays", ACCFLAG_DISPLAYABLE);
    mExclusiveAccumulators[EXCL_ACCUM_EMBREE_OCCLUSION]         = allocAccumulator("Embree occlusion rays", ACCFLAG_DISPLAYABLE);
    mExclusiveAccumulators[EXCL_ACCUM_GPU_INTERSECTION]         = allocAccumulator("GPU intersection rays", ACCFLAG_DISPLAYABLE);
    mExclusiveAccumulators[EXCL_ACCUM_GPU_OCCLUSION]            = allocAccumulator("GPU occlusion rays", ACCFLAG_DISPLAYABLE);
    mExclusiveAccumulators[EXCL_ACCUM_EMBREE_PRESENCE]          = allocAccumulator("Embree presence rays", ACCFLAG_DISPLAYABLE);
    mExclusiveAccumulators[EXCL_ACCUM_EMBREE_VOLUME]            = allocAccumulator("Embree volume rays", ACCFLAG_DISPLAYABLE);
    mExclusiveAccumulators[EXCL_ACCUM_INIT_INTERSECTION]        = allocAccumulator("Init Intersection", ACCFLAG_DISPLAYABLE);
    mExclusiveAccumulators[EXCL_ACCUM_INTEGRATION]              = allocAccumulator("Integration", ACCFLAG_DISPLAYABLE);
    mExclusiveAccumulators[EXCL_ACCUM_OCCL_QUERY_HANDLER]       = allocAccumulator("Occl query handler (excl. embree)", ACCFLAG_DISPLAYABLE);
    mExclusiveAccumulators[EXCL_ACCUM_OIIO]                     = allocAccumulator("Texturing (OIIO)", ACCFLAG_DISPLAYABLE);
    mExclusiveAccumulators[EXCL_ACCUM_POST_INTEGRATION]         = allocAccumulator("Post integration (SOA->AOS/queuing)", ACCFLAG_DISPLAYABLE);
    mExclusiveAccumulators[EXCL_ACCUM_PRESENCE_QUERY_HANDLER]   = allocAccumulator("Presence query handler (excl. embree)", ACCFLAG_DISPLAYABLE);
    mExclusiveAccumulators[EXCL_ACCUM_PRIMARY_RAY_GEN]          = allocAccumulator("Primary ray generation", ACCFLAG_DISPLAYABLE);
    mExclusiveAccumulators[EXCL_ACCUM_QUEUE_LOGIC]              = allocAccumulator("Queuing logic (incl. sorting)", ACCFLAG_DISPLAYABLE);
    mExclusiveAccumulators[EXCL_ACCUM_RAY_HANDLER]              = allocAccumulator("Ray handler (excl. embree)", ACCFLAG_DISPLAYABLE);
    mExclusiveAccumulators[EXCL_ACCUM_RAY_SORT_KEY_GEN]         = allocAccumulator("Ray sortkey gen", ACCFLAG_DISPLAYABLE);
    mExclusiveAccumulators[EXCL_ACCUM_RAYSTATE_ALLOCS]          = allocAccumulator("RayState allocs", ACCFLAG_DISPLAYABLE);
    mExclusiveAccumulators[EXCL_ACCUM_RENDER_DRIVER_OVERHEAD]   = allocAccumulator("Render driver overhead", ACCFLAG_DISPLAYABLE);
    mExclusiveAccumulators[EXCL_ACCUM_SHADE_HANDLER]            = allocAccumulator("Shade handler (excl. isect+shading)", ACCFLAG_DISPLAYABLE);
    mExclusiveAccumulators[EXCL_ACCUM_SHADING]                  = allocAccumulator("Shading (excl. OIIO)", ACCFLAG_DISPLAYABLE);
    mExclusiveAccumulators[EXCL_ACCUM_SSS_INTEGRATION]          = allocAccumulator("Subsurface integration", ACCFLAG_DISPLAYABLE);
    mExclusiveAccumulators[EXCL_ACCUM_TLS_ALLOCS]               = allocAccumulator("TLS allocs (excl. RayStates)", ACCFLAG_DISPLAYABLE);
    mExclusiveAccumulators[EXCL_ACCUM_UNRECORDED]               = allocAccumulator("Unrecorded", ACCFLAG_DISPLAYABLE);
    mExclusiveAccumulators[EXCL_ACCUM_VOL_INTEGRATION]          = allocAccumulator("Volume integration", ACCFLAG_DISPLAYABLE);
    mExclusiveAccumulators[EXCL_BUILD_ADAPTIVE_TREE]            = allocAccumulator("Adaptive tree rebuild", ACCFLAG_DISPLAYABLE);
    mExclusiveAccumulators[EXCL_QUERY_ADAPTIVE_TREE]            = allocAccumulator("Adaptive tree query", ACCFLAG_DISPLAYABLE);
    mExclusiveAccumulators[EXCL_EXCL_LOCK_ADAPTIVE_TREE]        = allocAccumulator("Adaptive tree exclusive lock", ACCFLAG_DISPLAYABLE);

    // General misc accumulators for programmer convenience.
    mExclusiveAccumulators[EXCL_ACCUM_MISC_A]                   = allocAccumulator("Misc A", ACCFLAG_DISPLAYABLE);
    mExclusiveAccumulators[EXCL_ACCUM_MISC_B]                   = allocAccumulator("Misc B", ACCFLAG_DISPLAYABLE);
    mExclusiveAccumulators[EXCL_ACCUM_MISC_C]                   = allocAccumulator("Misc C", ACCFLAG_DISPLAYABLE);
    mExclusiveAccumulators[EXCL_ACCUM_MISC_D]                   = allocAccumulator("Misc D", ACCFLAG_DISPLAYABLE);
    mExclusiveAccumulators[EXCL_ACCUM_MISC_E]                   = allocAccumulator("Misc E", ACCFLAG_DISPLAYABLE);
    mExclusiveAccumulators[EXCL_ACCUM_MISC_F]                   = allocAccumulator("Misc F", ACCFLAG_DISPLAYABLE);

    //
    // Accumulators to aid measuring the amount of time spent accomplishing a
    // particular task. These times overlap with the exclusive blocks above.
    //
    mOverlappingAccumulators[ACCUM_AOS_TO_SOA_EMB_ISECT_RAYS]   = allocAccumulator("[AOS->SOA embree isect rays]", ACCFLAG_DISPLAYABLE);
    mOverlappingAccumulators[ACCUM_AOS_TO_SOA_EMB_OCCL_RAYS]    = allocAccumulator("[AOS->SOA embree occl rays]", ACCFLAG_DISPLAYABLE);
    mOverlappingAccumulators[ACCUM_AOS_TO_SOA_INTERSECTIONS]    = allocAccumulator("[AOS->SOA intersections]", ACCFLAG_DISPLAYABLE);
    mOverlappingAccumulators[ACCUM_AOS_TO_SOA_RAYSTATES]        = allocAccumulator("[AOS->SOA RayStates]", ACCFLAG_DISPLAYABLE);
    mOverlappingAccumulators[ACCUM_CL1_ALLOC_STALLS]            = allocAccumulator("[TLState CL1 alloc stalls]", ACCFLAG_DISPLAYABLE);
    mOverlappingAccumulators[ACCUM_CL2_ALLOC_STALLS]            = allocAccumulator("[TLState CL2 alloc stalls]", ACCFLAG_DISPLAYABLE);
    mOverlappingAccumulators[ACCUM_CL4_ALLOC_STALLS]            = allocAccumulator("[TLState CL4 alloc stalls]", ACCFLAG_DISPLAYABLE);
    mOverlappingAccumulators[ACCUM_DRAIN_QUEUES]                = allocAccumulator("[Drain queues]", ACCFLAG_DISPLAYABLE);
    mOverlappingAccumulators[ACCUM_NON_RENDER_DRIVER]           = allocAccumulator("___ Non render driver ___", ACCFLAG_NONE);
    mOverlappingAccumulators[ACCUM_RAYSTATE_STALLS]             = allocAccumulator("[RayState alloc stalls]", ACCFLAG_DISPLAYABLE);
    mOverlappingAccumulators[ACCUM_RENDER_DRIVER_PARALLEL]      = allocAccumulator("___ Render driver parallel time ___", ACCFLAG_NONE);
    mOverlappingAccumulators[ACCUM_SOA_TO_AOS_EMB_ISECT_RAYS]   = allocAccumulator("[SOA->AOS embree isect rays]", ACCFLAG_DISPLAYABLE);
    mOverlappingAccumulators[ACCUM_SOA_TO_AOS_OCCL_RAYS]        = allocAccumulator("[SOA->AOS bundled occl rays]", ACCFLAG_DISPLAYABLE);
    mOverlappingAccumulators[ACCUM_SOA_TO_AOS_RADIANCES]        = allocAccumulator("[SOA->AOS radiances]", ACCFLAG_DISPLAYABLE);
    mOverlappingAccumulators[ACCUM_SOA_TO_AOS_RAYSTATES]        = allocAccumulator("[SOA->AOS RayStates]", ACCFLAG_DISPLAYABLE);
    mOverlappingAccumulators[ACCUM_SORT_RAY_BUCKETS]            = allocAccumulator("[Sort rays]", ACCFLAG_DISPLAYABLE);
    mOverlappingAccumulators[ACCUM_TOTAL_ISPC]                  = allocAccumulator("[Total ISPC time (excl. embree)]", ACCFLAG_DISPLAYABLE);

    // General misc accumulators for programmer convenience.
    mOverlappingAccumulators[ACCUM_MISC_A]                      = allocAccumulator("[Misc A]", ACCFLAG_DISPLAYABLE);
    mOverlappingAccumulators[ACCUM_MISC_B]                      = allocAccumulator("[Misc B]", ACCFLAG_DISPLAYABLE);
    mOverlappingAccumulators[ACCUM_MISC_C]                      = allocAccumulator("[Misc C]", ACCFLAG_DISPLAYABLE);
    mOverlappingAccumulators[ACCUM_MISC_D]                      = allocAccumulator("[Misc D]", ACCFLAG_DISPLAYABLE);
    mOverlappingAccumulators[ACCUM_MISC_E]                      = allocAccumulator("[Misc E]", ACCFLAG_DISPLAYABLE);
    mOverlappingAccumulators[ACCUM_MISC_F]                      = allocAccumulator("[Misc F]", ACCFLAG_DISPLAYABLE);

    //
    // Internal accumulators using for intermediate calculations or to aid displaying results.
    //
    mInternalAccumulators[INT_ACCUM_MISSING_TIME]               = allocAccumulator("MISSING TIME", ACCFLAG_DISPLAYABLE);
    mInternalAccumulators[INT_ACCUM_RENDER_DRIVER]              = allocAccumulator("__ Render driver __", ACCFLAG_DISPLAYABLE);
    mInternalAccumulators[INT_ACCUM_RENDER_DRIVER_SERIAL]       = allocAccumulator("Render driver serial time", ACCFLAG_DISPLAYABLE);
    mInternalAccumulators[INT_ACCUM_TOTAL_EMBREE]               = allocAccumulator("[Total embree time]", ACCFLAG_DISPLAYABLE);
    mInternalAccumulators[INT_ACCUM_TOTAL_INTEGRATION]          = allocAccumulator("[Total integration]", ACCFLAG_DISPLAYABLE);
    mInternalAccumulators[INT_ACCUM_TOTAL_RAY_INTERSECTION]     = allocAccumulator("[Total ray intersection]", AccumulatorFlags(ACCFLAG_DISPLAYABLE));
    mInternalAccumulators[INT_ACCUM_TOTAL_SCALAR_TIME]          = allocAccumulator("[Total scalar time (excl. embree)]", AccumulatorFlags(ACCFLAG_DISPLAYABLE));
    mInternalAccumulators[INT_ACCUM_TOTAL_SHADING]              = allocAccumulator("[Total shading time]", AccumulatorFlags(ACCFLAG_DISPLAYABLE));
    mInternalAccumulators[INT_ACCUM_TOTALS]                     = allocAccumulator("Totals", AccumulatorFlags(ACCFLAG_TOTAL | ACCFLAG_DISPLAYABLE));

    // Check we didn't miss any.
    for (unsigned i = 0; i < NUM_EXCLUSIVE_ACC; ++i) {
        MNRY_ASSERT(mExclusiveAccumulators[i]);
    }

    for (unsigned i = 0; i < NUM_OVERLAPPED_ACC; ++i) {
        MNRY_ASSERT(mOverlappingAccumulators[i]);
    }

    for (unsigned i = 0; i < NUM_INTERNAL_ACC; ++i) {
        MNRY_ASSERT(mInternalAccumulators[i]);
    }

    // Reset accumulators.
    resetAllAccumulators();
}

void
AccumulatorHandles::cleanUp()
{
    cleanUpAccumulators();

    // False positive from cppcheck, this struct doesn't contain a std::string!
    // cppcheck-suppress memsetClass // (error) Using 'memset' on struct that contains a 'std::string'
    memset(this, 0, sizeof(*this));
}

unsigned
snapshotAccumulators(std::vector<AccumulatorResult> *dstResults,
                     double rcpTickFrequency,
                     double threshold)
{
#ifdef PROFILE_ACCUMULATORS_ENABLED

    MNRY_ASSERT(dstResults);

    if (!snapshotRawAccumulators(dstResults, rcpTickFrequency)) {
        return 0;
    }

    // Copy time data into each to manipulate array.
    std::vector<AccumulatorResult> &results = *dstResults;
    std::vector<double> times;
    times.resize(results.size());
    for (size_t i = 0; i < results.size(); ++i) {
        times[i] = results[i].mTimePerThread;
    }

    // Process various times recorded into more digestible times.
    conditionTimes(times);

    // Now convert conditioned times back into AccumulatorResult format.
    const double threadMultiplier = double(getNumAccumulatorThreads());
    const double pctMultiplier = 100.0 / ACC(gAccumulatorHandles.mRoot);

    for (size_t i = 0; i < results.size(); ++i) {
        AccumulatorResult &result = results[i];
        result.mTotalTime = times[i] * threadMultiplier;
        result.mTimePerThread = times[i];
        result.mPercentageOfTotal = times[i] * pctMultiplier;
    }


    // Cull results which fall below the threshold or aren't displayable.
    auto newEnd = std::remove_if(results.begin(), results.end(), [&](const AccumulatorResult &acc) {
        return acc.mTotalTime < threshold || ((acc.mFlags & ACCFLAG_DISPLAYABLE) == 0);
    });

    results.erase(newEnd, results.end());

    // Greater than.
    std::stable_sort(results.begin(), results.end(), [](const AccumulatorResult &a, const AccumulatorResult &b) -> bool {
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

#else

    return 0;

#endif
}

//-----------------------------------------------------------------------------

ExclusiveAccumulators::ExclusiveAccumulators()
{
    memset(this, 0, sizeof(ExclusiveAccumulators));
}

void
ExclusiveAccumulators::cacheThreadLocalAccumulators(unsigned threadIdx)
{
    memset(this, 0, sizeof(ExclusiveAccumulators));

    for (unsigned i = 0; i < NUM_EXCLUSIVE_ACC; ++i) {
        mAccumulators[i] = MNRY_VERIFY(&gAccumulatorHandles.mExclusiveAccumulators[i]->mThreadLocal[threadIdx]);
        MNRY_ASSERT(mAccumulators[i]->canStart());
    }
}

//-----------------------------------------------------------------------------

extern "C"
{

intptr_t
CPP_startOverlappedAccumulator(mcrt_common::BaseTLState *tls, OverlappedAccType type)
{
    unsigned threadIdx = getAccumulatorThreadIndex(tls);
    ThreadLocalAccumulator *tlAcc = &MNRY_VERIFY(getAccumulator(type))->mThreadLocal[threadIdx];
    tlAcc->start();
    return intptr_t(tlAcc);
}

void
CPP_pauseOverlappedAccumulator(intptr_t acc)
{
    ThreadLocalAccumulator *tlAcc = MNRY_VERIFY((ThreadLocalAccumulator *)acc);
    tlAcc->stop();
}

void
CPP_unpauseOverlappedAccumulator(intptr_t acc)
{
    ThreadLocalAccumulator *tlAcc = MNRY_VERIFY((ThreadLocalAccumulator *)acc);
    tlAcc->start();
}

void
CPP_stopOverlappedAccumulator(intptr_t acc)
{
    ThreadLocalAccumulator *tlAcc = MNRY_VERIFY((ThreadLocalAccumulator *)acc);
    tlAcc->stop();
}

}

} //namespace mcrt_common
} //namespace moonray


