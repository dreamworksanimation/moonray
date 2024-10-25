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
                               unsigned numRays,
                               RayState **entryData,
                               void *userData);

    // The GPU handler that calls the GPU
    typedef void (*GPUHandler)(mcrt_common::ThreadLocalState *tls,
                               unsigned numRays,
                               RayState **entryData,
                               const rt::GPURay *gpuRays,
                               std::atomic<int>& threadsUsingGPU);

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

        mNumThreadsUsingGPU = 0;
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

    void reset()
    {
        for (size_t i = 0; i < mNumCPUThreads; i++) {
            mCPUThreadQueueNumQueued[i] = 0;
        }
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

        unsigned numRays = totalEntries;

        // Only flush a multiple of the number of lanes we have unless we're doing an
        // explicit flush (totalEntries < mQueueSize) or queue is exactly full.
        mCPUThreadQueueNumQueued[threadIdx] = 0;
        if (totalEntries > mCPUThreadQueueSize) {

            unsigned potentiallyQueued = totalEntries & (VLEN - 1);

            // Copy the left overs back into the primary queue. It's safe to do this
            // now since we always copy the entries before calling the handler.
            if (potentiallyQueued && potentiallyQueued < mCPUThreadQueueSize) {
                mCPUThreadQueueNumQueued[threadIdx] = potentiallyQueued;
                numRays -= potentiallyQueued;
                memcpy(mCPUThreadQueueEntries[threadIdx], entries + numRays, sizeof(RayState*) * potentiallyQueued);
            }
        }

        MNRY_ASSERT(mCPUThreadQueueNumQueued[threadIdx] < VLEN);
        MNRY_ASSERT(mCPUThreadQueueNumQueued[threadIdx] < mCPUThreadQueueSize);
        MNRY_ASSERT(mCPUThreadQueueNumQueued[threadIdx] + numRays == totalEntries);

        // Call handler. The entries are only valid for the duration of this call.
        // Other threads may also call this handler simultaneously with different entries.

#ifndef __APPLE__ // Doesn't support regular rays yet

        // Epirically-determined maximum number of threads that can be waiting on the GPU.
        // Might want to make this configurable.
        constexpr int maxThreads = 5;

        if ((mNumThreadsUsingGPU.load() < maxThreads) && numRays > 1024) {
            // There are an acceptable number of threads using the GPU, so we
            // can go ahead.

            pbr::TLState *pbrTls = tls->mPbrTls.get();
            scene_rdl2::alloc::Arena *arena = &tls->mArena;

            const FrameState &fs = *pbrTls->mFs;
            rt::GPUAccelerator *accel = const_cast<rt::GPUAccelerator*>(fs.mGPUAccel);

            rt::GPURay* gpuRays = arena->allocArray<rt::GPURay>(numRays, CACHE_LINE_SIZE);

            for (size_t i = 0; i < numRays; ++i) {
                RayState* rs = entries[i];
                // Optix doesn't access these values from the cpu ray directly, so we copy the
                // values here into a GPU-accessible buffer
//#ifndef __APPLE__
                gpuRays[i].mOriginX = rs->mRay.org.x;
                gpuRays[i].mOriginY = rs->mRay.org.y;
                gpuRays[i].mOriginZ = rs->mRay.org.z;
                gpuRays[i].mDirX = rs->mRay.dir.x;
                gpuRays[i].mDirY = rs->mRay.dir.y;
                gpuRays[i].mDirZ = rs->mRay.dir.z;
                gpuRays[i].mMinT = rs->mRay.tnear;
                gpuRays[i].mMaxT = rs->mRay.tfar;
                gpuRays[i].mTime = rs->mRay.time;
//#endif
                gpuRays[i].mMask = rs->mRay.mask;
                gpuRays[i].mShadowReceiverId = 0; // unused for regular rays
                gpuRays[i].mLightId = 0; // unused for regular rays
            }

            ++tls->mHandlerStackDepth;
            (*mGPUQueueHandler)(tls,
                                numRays,
                                entries,
                                gpuRays,
                                mNumThreadsUsingGPU);
            MNRY_ASSERT(tls->mHandlerStackDepth > 0);
            --tls->mHandlerStackDepth;

        } else
#endif // __APPLE__
        {
            // There's too many threads using the GPU, and we would need to wait
            // too long.  Process these rays on the CPU instead.
            ++tls->mHandlerStackDepth;
            (*mCPUThreadQueueHandler)(tls, numRays, entries, mHandlerData);
            MNRY_ASSERT(tls->mHandlerStackDepth > 0);
            --tls->mHandlerStackDepth;
        }

        return unsigned(numRays);
    }

    unsigned                     mNumCPUThreads;
    unsigned                     mCPUThreadQueueSize;
    std::vector<RayState**>      mCPUThreadQueueEntries;
    std::vector<unsigned>        mCPUThreadQueueNumQueued;
    CPUHandler                   mCPUThreadQueueHandler;
    std::atomic<int>             mNumThreadsUsingGPU;
    GPUHandler                   mGPUQueueHandler;
    void *                       mHandlerData;
};

} // namespace pbr
} // namespace moonray
