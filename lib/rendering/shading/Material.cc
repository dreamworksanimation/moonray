// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

#include "Material.h"

#include <moonray/rendering/mcrt_common/ThreadLocalState.h>

namespace moonray {
namespace shading {

tbb::mutex Material::sMaterialListMutex;
MaterialPtrList Material::sAllMaterials;
MaterialPtrList Material::sQueuelessMaterials;

tbb::mutex Material::sShadeQueueMutex;
ShadeQueueList Material::sShadeQueues;

tbb::atomic<size_t> Material::sFlushCycleIdx;

tbb::atomic<uint32_t> Material::sDeferredEntryCalls;
tbb::atomic<uint32_t> Material::sTotalDeferredEntries;

Material::Material(const scene_rdl2::rdl2::SceneObject & owner) :
    RootShader(owner),
    mShadeQueue(nullptr),
    mShadeEntries(nullptr),
    mMaterialLabelId(-1),   // no material label
    mLpeMaterialLabelId(-1) // no lpe material label
{
    tbb::mutex::scoped_lock lock(sMaterialListMutex);
    sAllMaterials.push_back(this);
    sQueuelessMaterials.push_back(this);
    mMaterialId = 0;
}

Material::~Material()
{
    // Not thread safe, all objects should be created/destroyed on a single thread.
    if (mShadeQueue) {

        {
            tbb::mutex::scoped_lock lock(sShadeQueueMutex);

            // Check the shade queue size also since it may have already been destroyed
            // during global program destruction time.
            if (sShadeQueues.size()) {

                // Remove ourselves from global list of shade queues.
                for (auto it = sShadeQueues.begin(); it != sShadeQueues.end(); ++it) {
                    if (*it == mShadeQueue) {
                        sShadeQueues.erase(it);
                        break;
                    }
                }
            }
        }
        scene_rdl2::util::alignedFreeArray(mShadeEntries);
        scene_rdl2::util::alignedFreeDtor(mShadeQueue);
    }

    {
        tbb::mutex::scoped_lock lock(sMaterialListMutex);

        // Remove ourselves from global list of Materials.
        for (auto it = sAllMaterials.begin(); it != sAllMaterials.end(); ++it) {
            if (*it == this) {
                sAllMaterials.erase(it);
                break;
            }
        }
    }
}

void
Material::deferEntriesForLaterProcessing(mcrt_common::ThreadLocalState *tls,
                                         unsigned numEntries,
                                         SortedRayState *entries)
{
    if (!numEntries) {
        return;
    }

    EXCL_ACCUMULATOR_PROFILE(tls, EXCL_ACCUM_DEFER_SHADE_ENTRIES);

    // Keep track of calls, if this gets too high then we'll need to investigate.
    ++sDeferredEntryCalls;
    sTotalDeferredEntries += numEntries;

    unsigned deferredCalls = sDeferredEntryCalls;

    const unsigned maxCallsToLog = 5;

    if (deferredCalls < maxCallsToLog) {
        mcrt_common::ExclusiveAccumulators *acc = MNRY_VERIFY(getExclusiveAccumulators(tls));
        unsigned accStack = acc->getStackSize();
        scene_rdl2::logging::Logger::warn("Call to Material::deferEntriesForLaterProcessing encountered (accumulator stack = ", accStack, ", handler stack = ", tls->mHandlerStackDepth, ".");
    } else if (deferredCalls == maxCallsToLog) {
        scene_rdl2::logging::Logger::warn("Multiple calls to Material::deferEntriesForLaterProcessing encountered, no more will be reported this frame.");
    }

    {
        tbb::mutex::scoped_lock lock(mDeferredEntryMutex);
        mDeferredEntries.insert(mDeferredEntries.end(), entries, entries + numEntries);
    }
}

void
Material::retrieveDeferredEntries(mcrt_common::ThreadLocalState *tls,
                                  scene_rdl2::alloc::Arena *arena,
                                  unsigned &numEntries,
                                  SortedRayState *&entries)
{
    if (mDeferredEntries.empty()) {
        return;
    }

    EXCL_ACCUMULATOR_PROFILE(tls, EXCL_ACCUM_DEFER_SHADE_ENTRIES);

    mDeferredEntryMutex.lock();

    unsigned numDeferredEntries = (unsigned)mDeferredEntries.size();
    unsigned totalEntries = numEntries + numDeferredEntries;

    SortedRayState *allEntries = arena->allocArray<SortedRayState>(totalEntries);

    memcpy(allEntries, &mDeferredEntries[0], numDeferredEntries * sizeof(SortedRayState));
    mDeferredEntries.clear();

    // Don't hold the lock any longer than we strictly have to.
    mDeferredEntryMutex.unlock();

    memcpy(allEntries + numDeferredEntries, entries, numEntries * sizeof(SortedRayState));

    numEntries = totalEntries;
    entries = allEntries;

    // The shadeBundleHandler expects sorted entries so redo that sort since
    // we're concatenating various SortRayState arrays from different threads.
    std::sort(entries, entries + numEntries,
              [](const SortedRayState &a, const SortedRayState &b) -> bool
    {
        return EXTRACT_KEY32(a, offsetof(SortedRayState, mSortKey)) <
               EXTRACT_KEY32(b, offsetof(SortedRayState, mSortKey));
    });

    MNRY_ASSERT( (scene_rdl2::util::isSorted32<SortedRayState, offsetof(SortedRayState, mSortKey)>(numEntries, entries)) );
}

// Thread safe, but should only be called on a single thread after scene update
// time but before rendering starts.
void
Material::allocShadeQueues(unsigned shadeQueueSize, ShadeQueue::Handler handler)
{
    tbb::mutex::scoped_lock lockMaterialMutex(sMaterialListMutex);
    tbb::mutex::scoped_lock lockShadeQueueMutex(sShadeQueueMutex);

    for (auto it = sQueuelessMaterials.begin(); it != sQueuelessMaterials.end(); ++it) {
        (*it)->allocShadeQueue(shadeQueueSize, handler);
    }

    sQueuelessMaterials.clear();
}

void
Material::initMaterialIds()
{
    uint32_t id = 1;
    for (auto it = sQueuelessMaterials.begin(); it != sQueuelessMaterials.end(); ++it) {
        (*it)->setMaterialId(id++);
    }

    sQueuelessMaterials.clear();
}

bool
Material::areAllShadeQueuesEmpty()
{
    for (auto it = sShadeQueues.begin(); it != sShadeQueues.end(); ++it) {
        if (!(*it)->isEmpty()) {
            return false;
        }
    }

    // We also need to check for the case where a Material's queue may be empty
    // but it could still have some entries stored for deferred processing.
    for (auto it = sAllMaterials.begin(); it != sAllMaterials.end(); ++it) {
        if (!(*it)->mDeferredEntries.empty()) {
            return false;
        }
    }

    return true;
}

unsigned
Material::flushNonEmptyShadeQueue(mcrt_common::ThreadLocalState *tls)
{
    // Always force sFlushCycleIdx to increment by at least 1.
    size_t startIdx = sFlushCycleIdx++;

    scene_rdl2::alloc::Arena *arena = &tls->mArena;
    SCOPED_MEM(arena);

    for (size_t i = 0; i < sShadeQueues.size(); ++i) {

        size_t cycleIdx = startIdx + i;
        ShadeQueue *queue = sShadeQueues[cycleIdx % sShadeQueues.size()];
#if 1
        unsigned flushed = queue->drain(tls, VLEN, arena);
#else
        unsigned flushed = queue->flush(tls, arena);
#endif

        if (flushed) {

            // There is a harmless race here (so we can avoid any heavier synchronization).
            // Overall it's fine and won't cause any problems.
            if (cycleIdx > sFlushCycleIdx) {
                sFlushCycleIdx = cycleIdx;
            }
            return flushed;
        }
    }

    // We also need to check for the case where a Material's queue may be empty
    // but it could still have some entries stored for deferred processing.
    // In this case we move the deferred entries into the actual shader
    // queues and return signal that there is still shade queues which contain
    // entries.
    unsigned numDeferredEntries = 0;
    for (auto it = sAllMaterials.begin(); it != sAllMaterials.end(); ++it) {
        if (!(*it)->mDeferredEntries.empty()) {
            Material *material = *it;
            unsigned numEntries = 0;
            SortedRayState *entries = nullptr;
            material->retrieveDeferredEntries(tls, arena, numEntries, entries);

            // This check is here since another thread may have stolen the
            // deferred entries in the meantime.
            if (numEntries && entries) {
                numDeferredEntries += numEntries;
                ShadeQueue *queue = material->mShadeQueue;
                MNRY_VERIFY(queue)->addEntries(tls, numEntries, entries, arena);
            }
        }
    }

    // This is expected to be zero 99.99% of the time.
    return numDeferredEntries;
}

void
Material::allocShadeQueue(unsigned shadeQueueSize, ShadeQueue::Handler handler)
{
    // This should only ever get called once at most per material.
    MNRY_ASSERT(!mShadeQueue);

    mShadeQueue = scene_rdl2::util::alignedMallocCtor<ShadeQueue>(CACHE_LINE_SIZE);
    mShadeEntries = scene_rdl2::util::alignedMallocArray<ShadeQueue::EntryType>(shadeQueueSize, CACHE_LINE_SIZE);
    mShadeQueue->init(shadeQueueSize, mShadeEntries);
    mShadeQueue->setHandler(handler, this);

    sShadeQueues.push_back(mShadeQueue);

    // TODO: We don't support deleting primitives currently. We'll need to add
    // some extra logic to maintain a valid bundled id when that gets implemented.
    mMaterialId = uint32_t(sShadeQueues.size());
}

void
Material::printDeferredEntryWarnings()
{
    unsigned numCalls = sDeferredEntryCalls;
    unsigned numEntries = sTotalDeferredEntries;

    if (numCalls) {
        scene_rdl2::logging::Logger::warn(numCalls, " call(s) to Material::deferEntriesForLaterProcessing containing ", numEntries, " entries.");
    }
}

void
Material::resetDeferredEntryState()
{
    for (auto it = sAllMaterials.begin(); it != sAllMaterials.end(); ++it) {
        Material *material = *it;
        tbb::mutex::scoped_lock lock(material->mDeferredEntryMutex);
        material->mDeferredEntries.clear();
    }

    sDeferredEntryCalls = 0;
    sTotalDeferredEntries = 0;
}

} // namespace shading
} // namespace moonray


