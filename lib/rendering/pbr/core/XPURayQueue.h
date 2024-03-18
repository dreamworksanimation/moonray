// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <moonray/rendering/pbr/Types.h>
#include <moonray/rendering/mcrt_common/Bundle.h>
#include <moonray/rendering/pbr/core/PbrTLState.h>
#include <moonray/rendering/rt/gpu/GPURay.h>

// warning #1684: conversion from pointer to
// same-sized integral type (potential portability problem)
// needed for reinterpret_cast<intptr_t>(light)
#pragma warning push
#pragma warning disable 1684

namespace moonray {
namespace pbr {

class XPURayQueue
{
public:
    // The CPU handler that is called when the GPU is busy
    typedef void (*CPUHandler)(mcrt_common::ThreadLocalState *tls,
                               unsigned numEntries,
                               RayState **entryData,
                               void *userData);

    // The GPU handler that calls the GPU
    typedef void (*GPUHandler)(mcrt_common::ThreadLocalState *tls,
                               unsigned numEntries,
                               RayState **entryData,
                               const rt::GPURay *gpuRays,
                               tbb::spin_mutex& mutex);

    XPURayQueue(unsigned numCPUThreads,
                unsigned cpuThreadQueueSize,
                CPUHandler cpuThreadQueueHandler,
                GPUHandler gpuQueueHandler,
                void *handlerData) :
        mNumCPUThreads(numCPUThreads),
        mCPUThreadQueueSize(cpuThreadQueueSize),
        mCPUThreadQueueHandler(cpuThreadQueueHandler),
        mGPUQueueHandler(gpuQueueHandler),
        mHandlerData(handlerData)
    {
        MNRY_ASSERT(numCPUThreads);
        MNRY_ASSERT(cpuThreadQueueSize);
        MNRY_ASSERT(mCPUThreadQueueHandler);
        MNRY_ASSERT(mGPUQueueHandler);

        // Create a queue for each CPU thread
        mCPUThreadQueueEntries.resize(mNumCPUThreads);
        mCPUThreadQueueNumQueued.resize(mNumCPUThreads);
        for (size_t i = 0; i < numCPUThreads; i++) {
            mCPUThreadQueueEntries[i] = scene_rdl2::util::alignedMallocArray<RayState*>(mCPUThreadQueueSize, CACHE_LINE_SIZE);
            mCPUThreadQueueNumQueued[i] = 0;
        }

        mThreadsWaitingForGPU = 0;
    }

    virtual ~XPURayQueue()
    {
        for (size_t i = 0; i < mNumCPUThreads; i++) {
            MNRY_ASSERT(mCPUThreadQueueNumQueued[i] == 0);
            scene_rdl2::util::alignedFree(mCPUThreadQueueEntries[i]);
        }
    }

    unsigned getMemoryUsed() const
    {
        return (sizeof(RayState*) * mCPUThreadQueueSize * mNumCPUThreads) + sizeof(*this);
    }

    bool isValid() const
    {
        MNRY_ASSERT(mGPUQueueHandler);
        return true;
    }

    // The CPU threads call this function to queue up rays (entries.)
    void addEntries(mcrt_common::ThreadLocalState *tls,
                    unsigned numEntries,
                    RayState **entries)
    {
        MNRY_ASSERT(tls);
        MNRY_ASSERT(numEntries);
        MNRY_ASSERT(entries);

        EXCL_ACCUMULATOR_PROFILE(tls, EXCL_ACCUM_QUEUE_LOGIC);

        while (numEntries >= mCPUThreadQueueSize) {
            // The number of rays that have been submitted exceeds or equals the
            // CPU thread queue size.  Process them in batches of mCPUThreadQueueSize.
            processRays(tls, mCPUThreadQueueSize, entries);
            entries += mCPUThreadQueueSize;
            numEntries -= mCPUThreadQueueSize;
        }

        int threadIdx = tls->mThreadIdx;

        MNRY_ASSERT(mCPUThreadQueueNumQueued[threadIdx] < mCPUThreadQueueSize);
        unsigned totalEntries = mCPUThreadQueueNumQueued[threadIdx] + numEntries;

        // Is there enough room in the CPU thread queue for the new entries?
        if (totalEntries <= mCPUThreadQueueSize) {
            // Copy entries into CPU thread's queue as there is room for them in that queue.
            memcpy(mCPUThreadQueueEntries[threadIdx] + mCPUThreadQueueNumQueued[threadIdx],
                   entries,
                   numEntries * sizeof(RayState*));
            mCPUThreadQueueNumQueued[threadIdx] = totalEntries;
            return;
        }

        // Else there isn't enough room in the CPU thread's queue for all of the new
        // entries.  We need to flush the queue to free up space.
        processRays(tls, mCPUThreadQueueNumQueued[threadIdx], mCPUThreadQueueEntries[threadIdx]);
        mCPUThreadQueueNumQueued[threadIdx] = 0;

        // Now that the queue is empty, we can add the new entries.
        memcpy(mCPUThreadQueueEntries[threadIdx],
               entries,
               numEntries * sizeof(RayState*));
        mCPUThreadQueueNumQueued[threadIdx] = numEntries;

        MNRY_ASSERT(mCPUThreadQueueNumQueued[threadIdx] < mCPUThreadQueueSize);
    }

    // Explicit flush of the CPU queues per thread.
    unsigned flush(mcrt_common::ThreadLocalState *tls, scene_rdl2::alloc::Arena *arena)
    {
        EXCL_ACCUMULATOR_PROFILE(tls, EXCL_ACCUM_QUEUE_LOGIC);

        int threadIdx = tls->mThreadIdx;

        int numFlushed = mCPUThreadQueueNumQueued[threadIdx];
        if (mCPUThreadQueueNumQueued[threadIdx]) {
            processRays(tls, mCPUThreadQueueNumQueued[threadIdx], mCPUThreadQueueEntries[threadIdx]);
            mCPUThreadQueueNumQueued[threadIdx] = 0;
        }

        return numFlushed;
    }

protected:

    void processRays(mcrt_common::ThreadLocalState *tls,
                     unsigned numRays,
                     RayState **rays)
    {
        EXCL_ACCUMULATOR_PROFILE(tls, EXCL_ACCUM_QUEUE_LOGIC);

        MNRY_ASSERT(numRays);

        if (mThreadsWaitingForGPU.load() < 3 && numRays > 1024) {
            // There are an acceptable number of threads waiting to access the GPU, so we
            // just wait our turn.  But, before we try to acquire the GPU lock, we get the
            // buffer of rays ready for the GPU.
            // The value 2 was determined empirically.  Higher values do not provide benefit.

            // Another thread can sneak in here before we increment, but it doesn't matter
            // because 2 is just a heuristic anyway and it doesn't matter if there's a small
            // chance we end up with 3.
            mThreadsWaitingForGPU++;

            pbr::TLState *pbrTls = tls->mPbrTls.get();
            scene_rdl2::alloc::Arena *arena = &tls->mArena;
            rt::GPURay* gpuRays = arena->allocArray<rt::GPURay>(numRays, CACHE_LINE_SIZE);

            // Now that the thread knows it's running on the GPU, it still needs to wait its
            // turn for the GPU.  Before waiting, it needs to prepare the GPURays.

            // TODO: copy RayStates to gpuRays

            // Acquire the GPU.
            mGPUDeviceMutex.lock();

            // This thread is no longer waiting to access the GPU, because it has the GPU.
            mThreadsWaitingForGPU--;

            // The handler unlocks the GPU device mutex internally once the GPU
            // is finished but there is still some CPU code left for this thread to run.
            (*mGPUQueueHandler)(tls,
                                numRays,
                                rays,
                                gpuRays,
                                mGPUDeviceMutex);

        } else {
            // There's too many threads already waiting for the GPU, and we would need to wait
            // too long.  Process these rays on the CPU instead.
            (*mCPUThreadQueueHandler)(tls, numRays, rays, mHandlerData);            
        }
    }

    unsigned                     mNumCPUThreads;
    unsigned                     mCPUThreadQueueSize;
    std::vector<RayState**>      mCPUThreadQueueEntries;
    std::vector<unsigned>        mCPUThreadQueueNumQueued;
    CPUHandler                   mCPUThreadQueueHandler;
    std::atomic<int>             mThreadsWaitingForGPU;
    GPUHandler                   mGPUQueueHandler;
    tbb::spin_mutex              mGPUDeviceMutex;
    void *                       mHandlerData;
};

} // namespace pbr
} // namespace moonray
