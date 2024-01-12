// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

//
//
// Accumulator handles for profiling the render portion of the frame.
// Not suitable for the update portion of the frame.
//
#pragma once
#include "ProfileAccumulator.h"
#include "ExecutionMode.h"
#include "Util.h"

#ifdef PROFILE_ACCUMULATORS_ENABLED
    // Assumes the existence of a function called GetExclusiveAccumulators defined
    // in scoped, which returns the ExclusiveAccumulator struct for this class.
    // Type is one the of ExclAccType enumerations.
    #define EXCL_ACCUMULATOR_PROFILE(tls, type)                 moonray::mcrt_common::ScopedExclAccumulator UNIQUE_IDENTIFIER(getExclusiveAccumulators(tls), (type))
    #define EXCL_ACCUMULATOR_GET_AND_START(tls, type, handle)   moonray::mcrt_common::ScopedExclAccumulator handle(getExclusiveAccumulators(tls), (type))
    #define EXCL_ACCUMULATOR_STOP(handle)                       handle.pop()
    #define EXCL_ACCUMULATOR_IS_RUNNING(tls, type)              (getExclusiveAccumulators(tls)->isRunning(type))
#else
    #define EXCL_ACCUMULATOR_PROFILE(tls, type)
    #define EXCL_ACCUMULATOR_GET_AND_START(tls, type, handle)
    #define EXCL_ACCUMULATOR_STOP(handle)
    #define EXCL_ACCUMULATOR_IS_RUNNING(tls, type)
#endif

#define TLS_OFFSET_TO_EXCL_ACCUMULATORS                         56u

namespace moonray {

// We are deliberately defining the accumulator related enums inside of the moonray namespace.
#include "ProfileAccumulatorHandles.hh"

//-----------------------------------------------------------------------------

//
// Top level accumulator.
//

// Total time spent in the rendering portion of the frame.
#define ACCUM_ROOT      moonray::mcrt_common::gAccumulatorHandles.mRoot

namespace mcrt_common {

//
// Global owner of the basic known set of Accumulators in the system.
//
struct AccumulatorHandles
{
    AccumulatorHandles();

    void init(unsigned numThreads, unsigned numTLS);
    void cleanUp();

    // Top level accumulator.
    Accumulator *mRoot;

    //
    // Accumulators to aid measuring exclusive blocks of time during the MCRT phase.
    //
    Accumulator *mExclusiveAccumulators[NUM_EXCLUSIVE_ACC];

    //
    // Accumulators to aid measuring the amount of time spent accomplishing a
    // particular task. These times overlap with the exclusive blocks above.
    //
    Accumulator *mOverlappingAccumulators[NUM_OVERLAPPED_ACC];

    //
    // Internal accumulators using for intermediate calculations or to aid displaying results.
    //
    Accumulator *mInternalAccumulators[NUM_INTERNAL_ACC];
};

extern AccumulatorHandles gAccumulatorHandles;

// This function has the job a querying the raw accumulators and transforming
// them into exclusive times.
unsigned snapshotAccumulators(std::vector<mcrt_common::AccumulatorResult> *dstResults,
                              double rcpTickFrequency,
                              double threshold);

//-----------------------------------------------------------------------------

//
// Thread local exclusive accumulator functionality.
//

#define MAX_EXCL_ACCUM_STACK_SIZE   1024

// Thread local.
class CACHE_ALIGN ExclusiveAccumulators
{
public:
                    ExclusiveAccumulators();

    // Call this once per frame at the start of the MCRT phase.
    void            cacheThreadLocalAccumulators(unsigned threadIdx);

    inline bool     isRunning(ExclAccType type) const;
    inline bool     isValid() const;

    // Returns true if the new type was pushed, or false if it was already on
    // the top of the stack.
    inline bool     push(ExclAccType type);

    // You should only call pop on entries for which push returned true.
    inline void     pop();

    inline unsigned getStackSize() const   { return mStackSize; }

private:
    inline void     startAccumulator(ThreadLocalAccumulator *acc);
    inline void     stopAccumulator(ThreadLocalAccumulator *acc);

    inline bool     isRunning(ThreadLocalAccumulator *acc) const;

    ThreadLocalAccumulator *mAccumulators[NUM_EXCLUSIVE_ACC];
    CACHE_ALIGN uint32_t    mPad;
    uint32_t                mStackSize;
    ThreadLocalAccumulator *mStack[MAX_EXCL_ACCUM_STACK_SIZE];
};

inline bool
ExclusiveAccumulators::isRunning(ExclAccType type) const
{
    MNRY_ASSERT(type < NUM_EXCLUSIVE_ACC);
    return isRunning(mAccumulators[type]);
}

inline bool
ExclusiveAccumulators::isValid() const
{
    return true;
}

inline bool
ExclusiveAccumulators::push(ExclAccType type)
{
    MNRY_ASSERT(type < NUM_EXCLUSIVE_ACC);
    MNRY_ASSERT_REQUIRE(mStackSize < MAX_EXCL_ACCUM_STACK_SIZE);

    if (mStackSize) {
        // Check if the desired accumulator is already running and if so,
        // treat this as a no-op.
        ThreadLocalAccumulator *stackTop = mStack[mStackSize - 1];
        if (stackTop == mAccumulators[type])
            return false;

        stopAccumulator(stackTop);
    }

    ThreadLocalAccumulator *acc = mAccumulators[type];
    startAccumulator(acc);

    mStack[mStackSize++] = acc;

    return true;
}

inline void
ExclusiveAccumulators::pop()
{
    MNRY_ASSERT(mStackSize);

    stopAccumulator(mStack[mStackSize - 1]);

    --mStackSize;

    if (mStackSize) {
        startAccumulator(mStack[mStackSize - 1]);
    }
}

inline void
ExclusiveAccumulators::startAccumulator(ThreadLocalAccumulator *acc)
{
    MNRY_ASSERT(acc);

#ifdef PROFILE_ACCUMULATORS_ENABLED
    MNRY_ASSERT(!isRunning(acc));
    MNRY_VERIFY(acc)->start();
#else
    uint64_t &active = reinterpret_cast<uint64_t &>(acc);
    ++active;
#endif
}

inline void
ExclusiveAccumulators::stopAccumulator(ThreadLocalAccumulator *acc)
{
    MNRY_ASSERT(acc);

#ifdef PROFILE_ACCUMULATORS_ENABLED
    MNRY_ASSERT(isRunning(acc));
    MNRY_VERIFY(acc)->stop();
#else
    uint64_t &active = reinterpret_cast<uint64_t &>(acc);
    --active;
#endif
}

inline bool
ExclusiveAccumulators::isRunning(ThreadLocalAccumulator *acc) const
{
#ifdef PROFILE_ACCUMULATORS_ENABLED
    // Accumulator will be null in the case of the GuiTLS. TODO setup "impostor" dummy objects
    return acc->mTimerActive != 0;
#else
    uint64_t active = reinterpret_cast<const uint64_t &>(acc);
    return active != 0;
#endif
}

//-----------------------------------------------------------------------------

class ScopedExclAccumulator
{
public:
    ScopedExclAccumulator(ExclusiveAccumulators *exclAcc, ExclAccType type) :
        mExclAcc(nullptr)
    {
        if (exclAcc && exclAcc->push(type)) {
            mExclAcc = exclAcc;

#ifdef DEBUG
            mDebugStackSize = exclAcc->getStackSize();
#endif
        }
    }

    ~ScopedExclAccumulator()
    {
        pop();
    }

    void pop()
    {
        if (mExclAcc) {
#ifdef DEBUG
            MNRY_ASSERT(mDebugStackSize == mExclAcc->getStackSize());
#endif
            mExclAcc->pop();
            mExclAcc = nullptr;
        }
    }

private:
    ExclusiveAccumulators *   mExclAcc;

#ifdef DEBUG
    unsigned                mDebugStackSize;    // For debugging only.
#endif
};

//-----------------------------------------------------------------------------

} //namespace mcrt_common

inline mcrt_common::Accumulator *
getAccumulator(OverlappedAccType acc)
{
    MNRY_ASSERT(acc < NUM_OVERLAPPED_ACC);
    return mcrt_common::gAccumulatorHandles.mOverlappingAccumulators[acc];
}

inline mcrt_common::ExclusiveAccumulators *
getExclusiveAccumulators(mcrt_common::ExclusiveAccumulators *exclAcc)
{
    return exclAcc;
}

// We use a hardcoded offset here to access the exclusive accumulators to avoid
// a dependency on rendering/pbr/core/PbrTLState.h.
namespace pbr { class TLState; }
inline mcrt_common::ExclusiveAccumulators *
getExclusiveAccumulators(pbr::TLState *tls)
{
    MNRY_ASSERT(tls);
#pragma warning push
#pragma warning disable 1684
    return *(mcrt_common::ExclusiveAccumulators **)(uint64_t(tls) + TLS_OFFSET_TO_EXCL_ACCUMULATORS);
#pragma warning pop
}

// Functions exposed to ISPC:
extern "C"
{

intptr_t CPP_startOverlappedAccumulator(mcrt_common::BaseTLState *tls, OverlappedAccType type);

// These all take the return value from CPP_startOverlappedAccumulator as their input parameter.
void CPP_pauseOverlappedAccumulator(intptr_t tlAcc);
void CPP_unpauseOverlappedAccumulator(intptr_t tlAcc);
void CPP_stopOverlappedAccumulator(intptr_t tlAcc);

}


} //namespace moonray


