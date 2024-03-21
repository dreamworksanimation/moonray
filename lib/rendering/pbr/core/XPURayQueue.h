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

    ~XPURayQueue()
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

    void addEntries(mcrt_common::ThreadLocalState *tls,
                    unsigned numEntries, RayState **entries, scene_rdl2::alloc::Arena *arena)
    {
        EXCL_ACCUMULATOR_PROFILE(tls, EXCL_ACCUM_QUEUE_LOGIC);

        int threadIdx = tls->mThreadIdx;

        MNRY_ASSERT(numEntries);
        MNRY_ASSERT(mCPUThreadQueueNumQueued[threadIdx] < mCPUThreadQueueSize);

        uint32_t totalEntries = mCPUThreadQueueNumQueued[threadIdx] + numEntries;

        if (totalEntries < mCPUThreadQueueSize) {
            // Copy data into queue.
            memcpy(mCPUThreadQueueEntries[threadIdx] + mCPUThreadQueueNumQueued[threadIdx],
                   entries, numEntries * sizeof(RayState*));

            mCPUThreadQueueNumQueued[threadIdx] = totalEntries;

            return;
        }

        flushInternal(tls, numEntries, entries, arena);

        MNRY_ASSERT(mCPUThreadQueueNumQueued[threadIdx] < mCPUThreadQueueSize);
    }

    // Explicit flush of what's currently in the queue.
    unsigned flush(mcrt_common::ThreadLocalState *tls, scene_rdl2::alloc::Arena *arena)
    {
        int threadIdx = tls->mThreadIdx;

        MNRY_ASSERT(mCPUThreadQueueNumQueued[threadIdx] < mCPUThreadQueueSize);

        if (mCPUThreadQueueNumQueued[threadIdx] == 0) {
            return 0;
        }

        return flushInternal(tls, 0, nullptr, arena);
    }

protected:
    unsigned flushInternal(mcrt_common::ThreadLocalState *tls,
                           uint32_t numNewEntries, RayState **newEntries,
                           scene_rdl2::alloc::Arena *arena)
    {
        EXCL_ACCUMULATOR_PROFILE(tls, EXCL_ACCUM_QUEUE_LOGIC);

        SCOPED_MEM(arena);

        int threadIdx = tls->mThreadIdx;
        uint32_t totalEntries = mCPUThreadQueueNumQueued[threadIdx] + numNewEntries;

        MNRY_ASSERT(totalEntries);

        // We always want to copy entries since there may be cycles which allow
        // other code to add to this queue further down in the callstack. This is
        // something this queue supports and encourages.
        RayState **entries = arena->allocArray<RayState*>(totalEntries);

        // Copy initial entries.
        memcpy(entries, mCPUThreadQueueEntries[threadIdx], sizeof(RayState*) * mCPUThreadQueueNumQueued[threadIdx]);

        // Copy additional entries.
        if (numNewEntries) {
            memcpy(entries + mCPUThreadQueueNumQueued[threadIdx], newEntries, sizeof(RayState*) * numNewEntries);
        }

        unsigned entriesToFlush = totalEntries;

        // Only flush a multiple of the number of lanes we have unless we're doing an
        // explicit flush (totalEntries < mQueueSize) or queue is exactly full.
        mCPUThreadQueueNumQueued[threadIdx] = 0;
        if (totalEntries > mCPUThreadQueueSize) {

            unsigned potentiallyQueued = totalEntries & (VLEN - 1);

            // Copy the left overs back into the primary queue. It's safe to do this
            // now since we always copy the entries before calling the handler.
            if (potentiallyQueued && potentiallyQueued < mCPUThreadQueueSize) {
                mCPUThreadQueueNumQueued[threadIdx] = potentiallyQueued;
                entriesToFlush -= potentiallyQueued;
                memcpy(mCPUThreadQueueEntries[threadIdx], entries + entriesToFlush, sizeof(RayState*) * potentiallyQueued);
            }
        }

        MNRY_ASSERT(mCPUThreadQueueNumQueued[threadIdx] < VLEN);
        MNRY_ASSERT(mCPUThreadQueueNumQueued[threadIdx] < mCPUThreadQueueSize);
        MNRY_ASSERT(mCPUThreadQueueNumQueued[threadIdx] + entriesToFlush == totalEntries);

        // Call handler. The entries are only valid for the duration of this call.
        // Other threads may also call this handler simultaneously with different entries.

/*
        // TODO: this is just the code from the occlusion ray processing copied in here
        // as a rough template for setting up the GPU.  For now we just call the CPU handler.

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
            ++tls->mHandlerStackDepth;
            (*mGPUQueueHandler)(tls,
                                numRays,
                                rays,
                                gpuRays,
                                mGPUDeviceMutex);
            MNRY_ASSERT(tls->mHandlerStackDepth > 0);
            --tls->mHandlerStackDepth;

        } else {
            // There's too many threads already waiting for the GPU, and we would need to wait
            // too long.  Process these rays on the CPU instead.
            ++tls->mHandlerStackDepth;
            (*mCPUThreadQueueHandler)(tls, numRays, rays, mHandlerData);       
            MNRY_ASSERT(tls->mHandlerStackDepth > 0);
            --tls->mHandlerStackDepth;     
        }
*/

        ++tls->mHandlerStackDepth;
        (*mCPUThreadQueueHandler)(tls, entriesToFlush, entries, mHandlerData);
        MNRY_ASSERT(tls->mHandlerStackDepth > 0);
        --tls->mHandlerStackDepth;

        return unsigned(entriesToFlush);
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
