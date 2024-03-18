// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

//
//

#pragma once
#include "ThreadLocalState.h"
#include <scene_rdl2/render/util/Arena.h>
#include <scene_rdl2/render/util/SortUtil.h>
#include <tbb/spin_mutex.h>
#include <algorithm>

namespace moonray {
namespace mcrt_common {

class ThreadLocalState;

//
//  Properties and assumptions:
//
//  -   ALLOWS CYCLES!
//
//  -   Payload size: There is a copy on insertion so small payloads work better
//                    for this. Additionally there is a second copy on flush
//                    which allow cycles in the graph. We heavily depend on this
//                    behavior for the RayQueue as the mechanism to avoid stalling.
//                    If the payload is large and you don't need to support
//                    cycles then consider using LocalLargeEntryQueue instead.
//  -   Thread safe:  No. Everything is assumed to be thread local. Entries cannot
//                    be added by threads other than ourselves.
//
template<typename T>
class LocalQueue
{
public:
    typedef T EntryType;
    typedef void (*Handler)(mcrt_common::ThreadLocalState *tls,
                            unsigned numEntries, EntryType *entryData,
                            void *userData);

    LocalQueue() :
        mQueueSize {0},
        mMaxEntries(0),
        mEntries(nullptr),
        mHandler(nullptr),
        mHandlerData(nullptr),
        mNumQueued(0)
    {
    }

    // entryMemory is the memory to be associated with each queue entry
    void init(unsigned maxEntries, void *entryMemory)
    {
        MNRY_ASSERT(maxEntries);

        mMaxEntries = mQueueSize = uint32_t(maxEntries);
        mNumQueued = 0;
        mEntries = (EntryType *)entryMemory;
    }

    unsigned getMemoryUsed() const
    {
        return (sizeof(T) * mMaxEntries) + sizeof(*this);
    }

    // The queue size default to mMaxEntries but users can set smaller sizes if desired.
    // It's legal to call this function even if entries already exist in the queue
    // as long as the new queue size can accommodate all the existing elements.
    // As long as this is the case, this function can also be called from other threads.
    void setQueueSize(unsigned size)
    {
        MNRY_ASSERT(size <= mMaxEntries);
        MNRY_ASSERT(size > mNumQueued);
        mQueueSize = size;
    }

    void setHandler(Handler handler, void *handlerData)
    {
        mHandler = handler;
        mHandlerData = handlerData;
    }

    void reset()
    {
        mNumQueued = 0;
    }

    bool isEmpty() const
    {
        return mNumQueued == 0;
    }

    unsigned getMaxEntries() const
    {
        return unsigned(mMaxEntries);
    }

    unsigned getQueueSize() const
    {
        return unsigned(mQueueSize);
    }

    bool isValid() const
    {
        MNRY_ASSERT(mMaxEntries > 0);
        MNRY_ASSERT(mMaxEntries >= mQueueSize);
        MNRY_ASSERT(mNumQueued < mQueueSize);
        MNRY_ASSERT(mHandler);
        return true;
    }

    void addEntries(mcrt_common::ThreadLocalState *tls,
                    unsigned numEntries, T *entries, scene_rdl2::alloc::Arena *arena)
    {
        EXCL_ACCUMULATOR_PROFILE(tls, EXCL_ACCUM_QUEUE_LOGIC);

        MNRY_ASSERT(numEntries);
        MNRY_ASSERT(mNumQueued < mQueueSize);

        uint32_t totalEntries = mNumQueued + numEntries;

        if (totalEntries < mQueueSize) {
            // Copy data into queue.
            memcpy(mEntries + mNumQueued, entries, numEntries * sizeof(EntryType));

            mNumQueued = totalEntries;

            return;
        }

        flushInternal(tls, numEntries, entries, arena);

        MNRY_ASSERT(mNumQueued < mQueueSize);
    }

    // Explicit flush of what's currently in the queue.
    unsigned flush(mcrt_common::ThreadLocalState *tls, scene_rdl2::alloc::Arena *arena)
    {
        MNRY_ASSERT(mNumQueued < mQueueSize);

        if (isEmpty()) {
            return 0;
        }

        return flushInternal(tls, 0, nullptr, arena);
    }

protected:
    unsigned flushInternal(mcrt_common::ThreadLocalState *tls,
                           uint32_t numNewEntries, T *newEntries,
                           scene_rdl2::alloc::Arena *arena)
    {
        EXCL_ACCUMULATOR_PROFILE(tls, EXCL_ACCUM_QUEUE_LOGIC);

        SCOPED_MEM(arena);

        uint32_t totalEntries = mNumQueued + numNewEntries;

        MNRY_ASSERT(totalEntries);

        // We always want to copy entries since there may be cycles which allows
        // other code to add to this queue further down in the callstack. This is
        // something this queue supports and encourages.
        EntryType *entries = arena->allocArray<EntryType>(totalEntries);

        // Copy initial entries.
        memcpy(entries, mEntries, sizeof(EntryType) * mNumQueued);

        // Copy additional entries.
        if (numNewEntries) {
            memcpy(entries + mNumQueued, newEntries, sizeof(EntryType) * numNewEntries);
        }

        unsigned entriesToFlush = totalEntries;

        // Only flush a multiple of the number of lanes we have unless we're doing an
        // explicit flush (totalEntries < mQueueSize) or queue is exactly full.
        mNumQueued = 0;
        if (totalEntries > mQueueSize) {

            unsigned potentiallyQueued = totalEntries & (VLEN - 1);

            // Copy the left overs back into the primary queue. It's safe to do this
            // now since we always copy the entries before calling the handler.
            if (potentiallyQueued && potentiallyQueued < mQueueSize) {
                mNumQueued = potentiallyQueued;
                entriesToFlush -= potentiallyQueued;
                memcpy(mEntries, entries + entriesToFlush, sizeof(EntryType) * potentiallyQueued);
            }
        }

        MNRY_ASSERT(mNumQueued < VLEN);
        MNRY_ASSERT(mNumQueued < mQueueSize);
        MNRY_ASSERT(mNumQueued + entriesToFlush == totalEntries);

        // Call handler. The entries are only valid for the duration of this call.
        // Other threads may also call this handler simultaneously with different entries.
        ++tls->mHandlerStackDepth;
        (*mHandler)(tls, entriesToFlush, entries, mHandlerData);
        MNRY_ASSERT(tls->mHandlerStackDepth > 0);
        --tls->mHandlerStackDepth;

        return unsigned(entriesToFlush);
    }

    tbb::atomic<uint32_t>   mQueueSize;   // must not exceed mMaxEntries
    uint32_t                mMaxEntries;
    EntryType *             mEntries;     // all data offsets are relative to this address
    Handler                 mHandler;
    void *                  mHandlerData;
    uint32_t                mNumQueued;
};

//-------------------------------------------------------------------------

//
//  Properties and assumptions:
//
//  -   DOESN'T ALLOW CYCLES! (so we save on copying data)
//
//  -   Payload size: This is like LocalQueue except optimized to avoid copying where
//                    possible which make it more suitable for larger entries.
//                    DOESN'T SUPPORT CYCLES!
//  -   Sorting:      Optional with an inline sortkey, sortkey not required if sorting is off.
//  -   Thread safe:  No. Everything is assumed to be thread local. Entries cannot be added
//                    by threads other than ourselves.
//
template<typename T, bool SORTED = false, unsigned SORT_KEY_OFFSET = 0>
class LocalLargeEntryQueue
{
public:
    typedef T EntryType;
    typedef void (*Handler)(mcrt_common::ThreadLocalState *tls, unsigned numEntries,
                            EntryType **entryData, void *userData);

    LocalLargeEntryQueue() :
        mQueueSize {0},
        mMaxEntries(0),
        mEntries(nullptr),
        mHandler(nullptr),
        mHandlerData(nullptr),
        mNumQueued(0)
    {
    }

    // entryMemory is the memory to be associated with each queue entry
    void init(unsigned maxEntries, void *entryMemory)
    {
        MNRY_ASSERT(maxEntries);

        mMaxEntries = mQueueSize = uint32_t(maxEntries);
        mNumQueued = 0;
        mEntries = (EntryType *)entryMemory;
    }

    unsigned getMemoryUsed() const
    {
        return (sizeof(T) * mMaxEntries) + sizeof(*this);
    }

    // The queue size default to mMaxEntries but users can set smaller sizes if desired.
    // It's legal to call this function even if entries already exist in the queue
    // as long as the new queue size can accomodate all the existing elements.
    // As long as this is the case, this function can also be called from other threads.
    void setQueueSize(unsigned size)
    {
        MNRY_ASSERT(size <= mMaxEntries);
        MNRY_ASSERT(size > mNumQueued);
        mQueueSize = size;
    }

    void setHandler(Handler handler, void *handlerData)
    {
        mHandler = handler;
        mHandlerData = handlerData;
    }

    void reset()
    {
        mNumQueued = 0;
    }

    bool isEmpty() const
    {
        return mNumQueued == 0;
    }

    unsigned getMaxEntries() const
    {
        return unsigned(mMaxEntries);
    }

    unsigned getQueueSize() const
    {
        return unsigned(mQueueSize);
    }

    bool isValid() const
    {
        MNRY_ASSERT(mMaxEntries > 0);
        MNRY_ASSERT(mMaxEntries >= mQueueSize);
        MNRY_ASSERT(mNumQueued < mQueueSize);
        MNRY_ASSERT(mHandler);
        return true;
    }

    void addEntries(mcrt_common::ThreadLocalState *tls,
                    unsigned numEntries, T *entries,
                    scene_rdl2::alloc::Arena *arena)
    {
        EXCL_ACCUMULATOR_PROFILE(tls, EXCL_ACCUM_QUEUE_LOGIC);

        MNRY_ASSERT(numEntries);
        MNRY_ASSERT(mNumQueued < mQueueSize);

        uint32_t totalEntries = mNumQueued + numEntries;

        if (totalEntries < mQueueSize) {
            // Copy data into queue.
            memcpy(mEntries + mNumQueued, entries, numEntries * sizeof(EntryType));

            mNumQueued = totalEntries;

            return;
        }

        flushInternal(tls, numEntries, entries, arena);

        MNRY_ASSERT(mNumQueued < mQueueSize);
    }

    // Explicit flush of what's currently in the queue.
    unsigned flush(mcrt_common::ThreadLocalState *tls, scene_rdl2::alloc::Arena *arena)
    {
        MNRY_ASSERT(mNumQueued < mQueueSize);

        if (isEmpty()) {
            return 0;
        }

        return flushInternal(tls, 0, nullptr, arena);
    }

protected:
    unsigned flushInternal(mcrt_common::ThreadLocalState *tls,
                           uint32_t numNewEntries, T *newEntries,
                           scene_rdl2::alloc::Arena *arena)
    {
        // This if statement is evaluated at compile time.
        if (SORTED) {
            return flushSortedInternal(tls, numNewEntries, newEntries, arena);
        } else {
            return flushUnsortedInternal(tls, numNewEntries, newEntries, arena);
        }
    }

    unsigned flushUnsortedInternal(mcrt_common::ThreadLocalState *tls,
                                   uint32_t numNewEntries, T *newEntries,
                                   scene_rdl2::alloc::Arena *arena)
    {
        EXCL_ACCUMULATOR_PROFILE(tls, EXCL_ACCUM_QUEUE_LOGIC);

        SCOPED_MEM(arena);

        uint32_t totalEntries = mNumQueued + numNewEntries;

        MNRY_ASSERT(totalEntries);

        // Create a temp list of pointers to pass to handler.
        EntryType **entries = arena->allocArray<EntryType *>(totalEntries);

        // Copy old pointers into entries array.
        for (uint32_t i = 0; i < mNumQueued; ++i) {
            entries[i] = &mEntries[i];
        }

        // Copy additional entries.
        for (uint32_t i = 0; i < numNewEntries; ++i) {
            entries[mNumQueued + i] = &newEntries[i];
        }

        unsigned entriesToFlush = totalEntries;

        // Only flush a multiple of the number of lanes we have unless we're doing an
        // explicit flush (totalEntries < mQueueSize) or queue is exactly full.
        mNumQueued = 0;
        if (totalEntries > mQueueSize) {

            unsigned potentiallyQueued = totalEntries & (VLEN - 1);
            if (potentiallyQueued && potentiallyQueued < mQueueSize) {
                mNumQueued = potentiallyQueued;
                entriesToFlush -= potentiallyQueued;
            }
        }

        // Call handler. The entries are only valid for the duration of this call.
        // Other threads may also call this handler simultaneously with different entries.
        ++tls->mHandlerStackDepth;
        (*mHandler)(tls, entriesToFlush, entries, mHandlerData);
        MNRY_ASSERT(tls->mHandlerStackDepth > 0);
        --tls->mHandlerStackDepth;

        // Copy the left overs back into the primary queue.
        for (uint32_t i = 0; i < mNumQueued; ++i) {
            mEntries[i] = *entries[entriesToFlush + i];
        }

        // If you hit any of these asserts, make sure this queue type doesn't appear
        // in any cycles in the data flow graph.
        MNRY_ASSERT(mNumQueued < VLEN);
        MNRY_ASSERT(mNumQueued < mQueueSize);
        MNRY_ASSERT(mNumQueued + entriesToFlush == totalEntries);

        return unsigned(entriesToFlush);
    }

    unsigned flushSortedInternal(mcrt_common::ThreadLocalState *tls,
                                 uint32_t numNewEntries, T *newEntries,
                                 scene_rdl2::alloc::Arena *arena)
    {
        EXCL_ACCUMULATOR_PROFILE(tls, EXCL_ACCUM_QUEUE_LOGIC);

        SCOPED_MEM(arena);

        uint32_t totalEntries = mNumQueued + numNewEntries;

        MNRY_ASSERT(totalEntries);

        struct SortedEntry
        {
            EntryType * mEntry;
            uint32_t    mSortKey;
        };

        // Create a temp list of pointers to pass to handler.
        SortedEntry *sortedEntries = arena->allocArray<SortedEntry>(totalEntries);

        // Copy old pointers into sortedEntries array.
        for (uint32_t i = 0; i < mNumQueued; ++i) {
            sortedEntries[i].mEntry = &mEntries[i];
            sortedEntries[i].mSortKey = ((uint32_t *)&mEntries[i])[SORT_KEY_OFFSET >> 2];
        }

        // Copy additional sortedEntries.
        for (uint32_t i = 0; i < numNewEntries; ++i) {
            sortedEntries[mNumQueued + i].mEntry = &newEntries[i];
            sortedEntries[mNumQueued + i].mSortKey = ((uint32_t *)&newEntries[i])[SORT_KEY_OFFSET >> 2];
        }

        unsigned entriesToFlush = totalEntries;

        // Only flush a multiple of the number of lanes we have unless we're doing an
        // explicit flush (totalEntries < mQueueSize) or queue is exactly full.
        mNumQueued = 0;
        if (totalEntries > mQueueSize) {

            unsigned potentiallyQueued = totalEntries & (VLEN - 1);

            // Copy the left overs back into the primary queue. It's safe to do this
            // now since we always copy the entries before calling the handler.
            if (potentiallyQueued && potentiallyQueued < mQueueSize) {
                mNumQueued = potentiallyQueued;
                entriesToFlush -= potentiallyQueued;
            }
        }

        scene_rdl2::util::inPlaceSort32<SortedEntry, sizeof(EntryType *), 100>(entriesToFlush,
                                                             sortedEntries, arena);

        // Now the entry list is sorted, remove the sort keys in place from it
        // before we pass the list to the handler.
        EntryType **entries = (EntryType **)sortedEntries;
        for (uint32_t i = 1; i < entriesToFlush; ++i) {
            entries[i] = sortedEntries[i].mEntry;
        }

        // Call handler. The entries are only valid for the duration of this call.
        // Other threads may also call this handler simultaneously with different entries.
        ++tls->mHandlerStackDepth;
        (*mHandler)(tls, entriesToFlush, entries, mHandlerData);
        MNRY_ASSERT(tls->mHandlerStackDepth > 0);
        --tls->mHandlerStackDepth;

        // Copy the left overs back into the primary queue.
        for (uint32_t i = 0; i < mNumQueued; ++i) {
            mEntries[i] = *sortedEntries[entriesToFlush + i].mEntry;
        }

        // If you hit any of these asserts, make sure this queue type doesn't appear
        // in any cycles in the data flow graph.
        MNRY_ASSERT(mNumQueued < VLEN);
        MNRY_ASSERT(mNumQueued < mQueueSize);
        MNRY_ASSERT(mNumQueued + entriesToFlush == totalEntries);

        return unsigned(entriesToFlush);
    }

    tbb::atomic<uint32_t>   mQueueSize;   // must not exceed mMaxEntries
    uint32_t                mMaxEntries;
    EntryType *             mEntries;     // all data offsets are relative to this address
    Handler                 mHandler;
    void *                  mHandlerData;
    uint32_t                mNumQueued;
};

//-------------------------------------------------------------------------

//
//  Properties and assumptions:
//
//  -   ALLOWS CYCLES!
//
//  -   Payload size: Preferably small since sorting moves entries in memory.
//  -   Sorting:      Optional with an inline sortkey, sortkey not required if sorting is off.
//  -   Thread safe:  Yes, safe to add entries from multiple threads and multiple threads
//                    can flush different bundles of entries simultaneously.
//
template<typename T, bool SORTED = false, unsigned SORT_KEY_OFFSET = 0>
class CACHE_ALIGN SharedQueue
{
public:
    typedef T EntryType;
    typedef void (*Handler)(mcrt_common::ThreadLocalState *tls, unsigned numEntries,
                            EntryType *entryData, void *userData);

    SharedQueue() :
        mQueueSize {0},
        mMaxEntries(0),
        mEntries(nullptr),
        mHandler(nullptr),
        mHandlerData(nullptr),
        mNumQueued(0)
    {
    }

    // entryMemory is the memory to be associated with each queue entry.
    void init(unsigned maxEntries, void *entryMemory)
    {
        MNRY_ASSERT(maxEntries);

        mMaxEntries = mQueueSize = uint32_t(maxEntries);
        mNumQueued = 0;
        mEntries = (EntryType *)entryMemory;
    }

    unsigned getMemoryUsed() const
    {
        return (sizeof(T) * mMaxEntries) + sizeof(*this);
    }

    // The queue size default to mMaxEntries but users can set smaller sizes if desired.
    // It's legal to call this function even if entries already exist in the queue
    // as long as the new queue size can accomodate all the existing elements.
    // As long as this is the case, this function can also be called from other threads.
    void setQueueSize(unsigned size)
    {
        MNRY_ASSERT(size <= mMaxEntries);
        MNRY_ASSERT(size > mNumQueued);
        mQueueSize = size;
    }

    void setHandler(Handler handler, void *handlerData)
    {
        mHandler = handler;
        mHandlerData = handlerData;
    }

    void reset()
    {
        mNumQueued = 0;
    }

    bool isEmpty() const
    {
        return mNumQueued == 0;
    }

    unsigned getMaxEntries() const
    {
        return unsigned(mMaxEntries);
    }

    unsigned getQueueSize() const
    {
        return unsigned(mQueueSize);
    }

    bool isValid() const
    {
        MNRY_ASSERT(mMaxEntries > 0);
        MNRY_ASSERT(mMaxEntries >= mQueueSize);
        MNRY_ASSERT(mNumQueued < mQueueSize);
        MNRY_ASSERT(mHandler);
        return true;
    }

    // Optimized for batch entries, single entries are considered the exception
    // rather than the rule.
    void addEntries(mcrt_common::ThreadLocalState *tls, unsigned numEntries,
                    const T *entries, scene_rdl2::alloc::Arena *arena)
    {
        EXCL_ACCUMULATOR_PROFILE(tls, EXCL_ACCUM_QUEUE_LOGIC);

        MNRY_ASSERT(numEntries);
        MNRY_ASSERT(mNumQueued < mQueueSize);

        // We are locking here, but the cost gets amortized over numEntries,
        // and also should typically be contended very seldomly.
        mMutex.lock();

        uint32_t freeIdx = mNumQueued;
        uint32_t totalEntries = freeIdx + numEntries;

        if (totalEntries < mQueueSize) {
            // Copy data into free idx.
            memcpy(mEntries + freeIdx, entries, numEntries * sizeof(EntryType));

            mNumQueued = totalEntries;

            mMutex.unlock();

            return;
        }

        // Flush as soon as the queue is full.
        flushInternal(tls, numEntries, entries, arena);

        MNRY_ASSERT(mNumQueued < mQueueSize);
    }

    // Explicit flush of what's currently in the queue.
    unsigned flush(mcrt_common::ThreadLocalState *tls, scene_rdl2::alloc::Arena *arena)
    {
        EXCL_ACCUMULATOR_PROFILE(tls, EXCL_ACCUM_QUEUE_LOGIC);

        MNRY_ASSERT(mNumQueued < mQueueSize);

        // Speculative early out check to avoid taking an unnecessary lock.
        if (isEmpty()) {
            return 0;
        }

        mMutex.lock();

        // Reliable isEmpty check.
        if (isEmpty()) {
            mMutex.unlock();
            return 0;
        }

        // This call unlocks mMutex internally.
        return flushInternal(tls, 0, nullptr, arena);
    }

    // Like flush but only empties up to the specified number of entries.
    unsigned drain(mcrt_common::ThreadLocalState *tls,
                   unsigned maxEntriesToFlush,
                   scene_rdl2::alloc::Arena *arena)
    {
        EXCL_ACCUMULATOR_PROFILE(tls, EXCL_ACCUM_QUEUE_LOGIC);

        MNRY_ASSERT(mNumQueued < mQueueSize);

        // Speculative early out check to avoid taking an unnecessary lock.
        if (isEmpty()) {
            return 0;
        }

        mMutex.lock();

        // Reliable isEmpty check.
        if (isEmpty()) {
            mMutex.unlock();
            return 0;
        }

        SCOPED_MEM(arena);

        unsigned entriesToFlush = std::min(maxEntriesToFlush, mNumQueued);
        MNRY_ASSERT(entriesToFlush);

        // We always want to copy entries in the shared queue case so that other
        // threads don't have to wait on the handler to complete before they can
        // start adding entries again.
        EntryType *entries = arena->allocArray<EntryType>(entriesToFlush);

        // Copy entries to flush into new arena memory.
        memcpy(entries, mEntries, sizeof(EntryType) * entriesToFlush);

        // Shift remaining entries to start of mEntries buffer.
        unsigned remainingEntries = mNumQueued - entriesToFlush;
        if (remainingEntries) {
            memmove(mEntries, mEntries + entriesToFlush, sizeof(EntryType) * remainingEntries);
        }

        mNumQueued = remainingEntries;

        MNRY_ASSERT(mNumQueued < mQueueSize);

        mMutex.unlock();

        if (SORTED) {
            std::sort(entries, entries + entriesToFlush, [](const T &a, const T &b) -> bool
            {
                return EXTRACT_KEY32(a, SORT_KEY_OFFSET) < EXTRACT_KEY32(b, SORT_KEY_OFFSET);
            });
        }

        // Call handler. The entries are only valid for the duration of this call.
        // Other threads may also call this handler simultaneously with different entries.
        ++tls->mHandlerStackDepth;
        (*mHandler)(tls, entriesToFlush, entries, mHandlerData);
        MNRY_ASSERT(tls->mHandlerStackDepth > 0);
        --tls->mHandlerStackDepth;

        return entriesToFlush;
    }

protected:
    // mMutex is assumed to be locked when we enter this function.
    unsigned flushInternal(mcrt_common::ThreadLocalState *tls,
                           uint32_t numNewEntries, const T *newEntries,
                           scene_rdl2::alloc::Arena *arena)
    {
        EXCL_ACCUMULATOR_PROFILE(tls, EXCL_ACCUM_QUEUE_LOGIC);

        SCOPED_MEM(arena);

        uint32_t totalEntries = mNumQueued + numNewEntries;

        MNRY_ASSERT(totalEntries);

        // We always want to copy entries in the shared queue case so that other
        // threads don't have to wait on the handler to complete before they can
        // start adding entries again.
        EntryType *entries = arena->allocArray<EntryType>(totalEntries);

        // Copy initial entries.
        memcpy(entries, mEntries, sizeof(EntryType) * mNumQueued);

        // Copy additional entries.
        if (numNewEntries) {
            memcpy(entries + mNumQueued, newEntries, sizeof(EntryType) * numNewEntries);
        }

        unsigned entriesToFlush = totalEntries;

        // Only flush a multiple of the number of lanes we have unless we're doing
        // an explicit flush (totalEntries < mQueueSize) or queue is exactly full.
        mNumQueued = 0;
        if (totalEntries > mQueueSize) {

            unsigned potentiallyQueued = totalEntries & (VLEN - 1);

            // Copy the left overs back into the primary queue. It's safe to do this
            // now since we always copy the entries before calling the handler.
            if (potentiallyQueued && potentiallyQueued < mQueueSize) {
                mNumQueued = potentiallyQueued;
                entriesToFlush -= potentiallyQueued;
                memcpy(mEntries, entries + entriesToFlush, sizeof(EntryType) * potentiallyQueued);
            }
        }

        MNRY_ASSERT(mNumQueued < VLEN);
        MNRY_ASSERT(mNumQueued < mQueueSize);
        MNRY_ASSERT(mNumQueued + entriesToFlush == totalEntries);

        mMutex.unlock();

        if (SORTED) {
            scene_rdl2::util::inPlaceSort32<EntryType, SORT_KEY_OFFSET, 200>(entriesToFlush, entries, arena);
        }

        // Call handler. The entries are only valid for the duration of this call.
        // Other threads may also call this handler simultaneously with different entries.
        ++tls->mHandlerStackDepth;
        (*mHandler)(tls, (unsigned)entriesToFlush, entries, mHandlerData);
        MNRY_ASSERT(tls->mHandlerStackDepth > 0);
        --tls->mHandlerStackDepth;

        return unsigned(entriesToFlush);
    }

    tbb::atomic<uint32_t>       mQueueSize;   // must not exceed mMaxEntries
    uint32_t                    mMaxEntries;
    EntryType *                 mEntries;     // all data offsets are relative to this address
    Handler                     mHandler;
    void *                      mHandlerData;
    uint32_t                    mNumQueued;

    // Move onto separate cache line.
    CACHE_ALIGN tbb::spin_mutex mMutex;
};

//-------------------------------------------------------------------------

} // namespace mcrt_common
} // namespace moonray


