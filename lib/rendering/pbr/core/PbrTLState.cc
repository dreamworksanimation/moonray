// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

//
//
#include "DebugRay.h"
#include "PbrTLState.h"
#include "RayState.h"
#include "XPUOcclusionRayQueue.h"
#include "XPURayQueue.h"
#include <moonray/rendering/pbr/handlers/RayHandlers.h>
#include <moonray/rendering/shading/Types.h>
#include <moonray/common/mcrt_macros/moonray_static_check.h>
#include <tbb/mutex.h>

// These aren't free so only turn it on if you are doing memory profiling.
// This will print out the peak number of pool items used for a particular run.
//#define DEBUG_RECORD_PEAK_RAYSTATE_USAGE
//#define DEBUG_RECORD_PEAK_CL1_USAGE

extern "C" bool isRenderCanceled();

namespace ispc {
extern "C" uint32_t PbrTLState_hudValidation(bool);
}

namespace scene_rdl2 {
using namespace alloc;
using namespace math;
using namespace util;
}

namespace moonray {
namespace pbr {


//-----------------------------------------------------------------------------

// Private:
namespace
{

// constants used for allocList and freeList
const int ALLOC_LIST_MAX_ITEM_SIZE = CACHE_LINE_SIZE;
const int ALLOC_LIST_MAX_NUM_ITEMS = CACHE_LINE_SIZE / sizeof(uint32_t);
// ALLOC_LIST_MAX_NUM_ITEMS is used for stack array allocations, keep small.
// going as high as 256 is probably ok, but this assert is meant to
// remind us to verify this.
MNRY_STATIC_ASSERT(ALLOC_LIST_MAX_NUM_ITEMS == 16);
// allocList returns and freeList accepts a 4 byte handle instead
// of a raw pointer.  this saves precious space in space-constrained
// structures that must store them.  per thread, we maintain a relatively
// small number of items (currently 64k), this means we have a little
// bit of extra space available in the index header to store additional
// information about the list.
const unsigned int ALLOC_LIST_INFO_BIT_SHIFT = 28;
const unsigned int ALLOC_LIST_INFO_BITS      = (0xFu << ALLOC_LIST_INFO_BIT_SHIFT);
MNRY_STATIC_ASSERT(ALLOC_LIST_MAX_NUM_ITEMS <=
                  ((ALLOC_LIST_INFO_BITS >> ALLOC_LIST_INFO_BIT_SHIFT) + 1));

// Per frame counter, gets reset each frame.
CACHE_ALIGN tbb::atomic<unsigned> gFailedRayStateAllocs;
CACHE_ALIGN tbb::atomic<unsigned> gFailedCL1Allocs;

// For memory profiling, see DEBUG_RECORD_PEAK_RAYSTATE_USAGE.
unsigned MAYBE_UNUSED gPeakRayStateUsage = 0;
unsigned MAYBE_UNUSED gPeakCL1Usage      = 0;

typedef mcrt_common::TLSInitParams::MemBlockType MemBlockType;
typedef scene_rdl2::MemBlockManager<MemBlockType> MemBlockManagerType;

struct PoolInfo
{
    PoolInfo():
        mActualPoolSize(0),
        mMemBlockManager(nullptr),
        mBlockMemory(nullptr),
        mEntryMemory(nullptr)
    {
    }

    unsigned          mActualPoolSize;
    MemBlockManagerType *mMemBlockManager;
    MemBlockType     *mBlockMemory;
    uint8_t          *mEntryMemory;
};

struct Private
{
    Private() :
        mRefCount(0),
        mInitParams(nullptr)
    {
    }

    unsigned mRefCount;
    const mcrt_common::TLSInitParams *mInitParams;

    PoolInfo mRayState;
    PoolInfo mCL1;
};

Private gPrivate;
tbb::mutex gInitMutex;

void
initPool(const unsigned poolSize, const unsigned numTBBThreads,
         const unsigned entrySize, const char * const poolName,
         PoolInfo &p)
{
    // Using poolSize * numTBBThreads isn't adequate for XPU mode because we
    // run out of space with low numbers of threads for things like BundledOcclRayData.
    // poolSize * 8 seems to be adequate, so for safety poolSize * 16 is used as the
    // minimum number of totalEntries.
    const unsigned totalEntries    = poolSize * std::max(numTBBThreads, 16u);

    const unsigned entryStride     = entrySize;
    const unsigned entriesPerBlock = MemBlockType::getNumEntries();

    unsigned numBlocks = totalEntries / entriesPerBlock;
    if (numBlocks * entriesPerBlock < totalEntries) {
        ++numBlocks;
    }

    MNRY_ASSERT(numBlocks * entriesPerBlock >= totalEntries);

    numBlocks = std::max(numBlocks, numTBBThreads);

    // Update the stored pool size so that the assertions in <typeName>ToIndex
    // and indexTo<typeName> remain valid
    p.mActualPoolSize = numBlocks * entriesPerBlock;

    const size_t entryMemorySize = MemBlockManagerType::queryEntryMemoryRequired(numBlocks, entryStride);

    // Uncomment to see how much memory is being allocated for each pool.
    //Logger::info("Attempting to allocate ", entryMemorySize, " bytes for ", poolName, " pool.\n");

    p.mEntryMemory = scene_rdl2::alignedMallocArray<uint8_t>(entryMemorySize, CACHE_LINE_SIZE);
    p.mBlockMemory = scene_rdl2::alignedMallocArrayCtor<MemBlockType>(numBlocks, CACHE_LINE_SIZE);
    p.mMemBlockManager = MNRY_VERIFY(scene_rdl2::alignedMallocCtor<MemBlockManagerType>(CACHE_LINE_SIZE));
    p.mMemBlockManager->init(numBlocks, p.mBlockMemory, p.mEntryMemory, entryStride);
}


void
initPrivate(const mcrt_common::TLSInitParams &initParams)
{
    MNRY_ASSERT(gPrivate.mRefCount == 0);
    MNRY_ASSERT(!gPrivate.mInitParams);
    MNRY_ASSERT(!gPrivate.mRayState.mMemBlockManager);
    MNRY_ASSERT(!gPrivate.mCL1.mMemBlockManager);

    MOONRAY_START_THREADSAFE_STATIC_WRITE

    //
    // Allocate pooled memory:
    //
    if (initParams.mPerThreadRayStatePoolSize) {
        initPool(initParams.mPerThreadRayStatePoolSize, initParams.mDesiredNumTBBThreads,
                 sizeof(RayState), "RayState", gPrivate.mRayState);
    }
    if (initParams.mPerThreadCL1PoolSize) {
        initPool(initParams.mPerThreadCL1PoolSize, initParams.mDesiredNumTBBThreads,
                 sizeof(TLState::CacheLine1), "CL1", gPrivate.mCL1);
    }

    // initParams is owned by the top level ThreadLocalState object so we know
    // it persists whilst this TLState persists.
    gPrivate.mInitParams = &initParams;

    MOONRAY_FINISH_THREADSAFE_STATIC_WRITE
}

void
cleanUpPrivate()
{
    MNRY_ASSERT(gPrivate.mRefCount == 0);
    MNRY_ASSERT(gPrivate.mInitParams);

    scene_rdl2::alignedFreeDtor(gPrivate.mRayState.mMemBlockManager);
    scene_rdl2::alignedFreeArray(gPrivate.mRayState.mBlockMemory);
    scene_rdl2::alignedFreeArray(gPrivate.mRayState.mEntryMemory);

    scene_rdl2::alignedFreeDtor(gPrivate.mCL1.mMemBlockManager);
    scene_rdl2::alignedFreeArray(gPrivate.mCL1.mBlockMemory);
    scene_rdl2::alignedFreeArray(gPrivate.mCL1.mEntryMemory);

    // Reset internal TLS related data.
    gPrivate.~Private();
    new (&gPrivate) Private;

    MNRY_ASSERT(gPrivate.mRayState.mMemBlockManager == nullptr);
    MNRY_ASSERT(gPrivate.mCL1.mMemBlockManager == nullptr);
}

template <typename T>
void
inline setQueueSize(T *queue, float t)
{
    MNRY_ASSERT(queue);

    unsigned maxEntries = queue->getMaxEntries();

    // Only mess with queue sizes if we have any queue elements allocated in the
    // first place.
    if (maxEntries > 0) {

        unsigned size = std::max(unsigned(float(maxEntries) * t), 1u);

        // Round up to VLEN.
        size = (size + VLEN) & ~VLEN;

        queue->setQueueSize(std::min(size, maxEntries));
    }
}

}   // End of anon namespace.

//-----------------------------------------------------------------------------

TLState::TLState(mcrt_common::ThreadLocalState *tls,
                 const mcrt_common::TLSInitParams &initParams,
                 bool okToAllocBundledResources) :
    BaseTLState(tls->mThreadIdx, tls->mArena, tls->mPixelArena),
    mTopLevelTls(tls),
    mRadianceQueue(nullptr),
    mAovQueue(nullptr),
    mHeatMapQueue(nullptr),
    mXPUOcclusionRayQueue(nullptr),
    mXPURayQueue(nullptr),
    mFs(nullptr),
    mCancellationState(DISABLED),
    mCurrentPassIdx(0),
    mRayRecorder(nullptr),
    mRayEntries(nullptr),
    mOcclusionEntries(nullptr),
    mPresenceShadowsEntries(nullptr),
    mRadianceEntries(nullptr),
    mAovEntries(nullptr),
    mHeatMapEntries(nullptr)
{
    mExclusiveAccumulators = scene_rdl2::alignedMallocCtor<ExclusiveAccumulators>(CACHE_LINE_SIZE);

    if (okToAllocBundledResources) {

        if (gPrivate.mRayState.mMemBlockManager) {
            mRayStatePool.init(gPrivate.mRayState.mMemBlockManager);
        }

        if (gPrivate.mCL1.mMemBlockManager) {
            mCL1Pool.init(gPrivate.mCL1.mMemBlockManager);
        }

        // Allocate primary ray queue.
        if (initParams.mRayQueueSize) {
            unsigned queueSize = initParams.mRayQueueSize;
            mRayEntries = scene_rdl2::alignedMallocArray<RayQueue::EntryType>
                                     (queueSize, CACHE_LINE_SIZE);
            mRayQueue.init(queueSize, mRayEntries);
            uint32_t rayHandlerFlags = 0;
            mRayQueue.setHandler(rayBundleHandler, (void *)((uint64_t)rayHandlerFlags));
        }

        // Allocate occlusion queue.
        if (initParams.mOcclusionQueueSize) {
            unsigned queueSize = initParams.mOcclusionQueueSize;
            mOcclusionEntries = scene_rdl2::alignedMallocArray<OcclusionQueue::EntryType>
                                    (queueSize, CACHE_LINE_SIZE);
            mOcclusionQueue.init(queueSize, mOcclusionEntries);
            uint32_t rayHandlerFlags = 0;
            mOcclusionQueue.setHandler(occlusionQueryBundleHandler, (void *)((uint64_t)rayHandlerFlags));
        }

        // Allocate presence shadows queue.
        if (initParams.mPresenceShadowsQueueSize) {
            unsigned queueSize = initParams.mPresenceShadowsQueueSize;
            mPresenceShadowsEntries = scene_rdl2::alignedMallocArray<PresenceShadowsQueue::EntryType>
                                    (queueSize, CACHE_LINE_SIZE);
            mPresenceShadowsQueue.init(queueSize, mPresenceShadowsEntries);
            // don't need to set handler flags because presence shadows code only does scalar
            //  ray tracing
            mPresenceShadowsQueue.setHandler(presenceShadowsQueryBundleHandler, nullptr);
        }

        // Allocate radiance queue.
        if (initParams.mRadianceQueueSize) {
            mRadianceQueue = scene_rdl2::alignedMallocCtor<RadianceQueue>(CACHE_LINE_SIZE);
            unsigned queueSize = initParams.mRadianceQueueSize;
            mRadianceEntries = scene_rdl2::alignedMallocArray<RadianceQueue::EntryType>
                                   (queueSize, CACHE_LINE_SIZE);
            mRadianceQueue->init(queueSize, mRadianceEntries);
            // Radiance queue handler is setup by the RenderDriver.
        }

        // Allocate aov queue
        if (initParams.mAovQueueSize) {
            mAovQueue = scene_rdl2::alignedMallocCtor<AovQueue>(CACHE_LINE_SIZE);
            unsigned queueSize = initParams.mAovQueueSize;
            mAovEntries = scene_rdl2::alignedMallocArray<AovQueue::EntryType>
                                   (queueSize, CACHE_LINE_SIZE);
            mAovQueue->init(queueSize, mAovEntries);
            // Aov queue handler is setup by the RenderDriver.
        }

        // Allocate heat map queue
        if (initParams.mHeatMapQueueSize) {
            mHeatMapQueue = scene_rdl2::alignedMallocCtor<HeatMapQueue>(CACHE_LINE_SIZE);
            unsigned queueSize = initParams.mHeatMapQueueSize;
            mHeatMapEntries = scene_rdl2::alignedMallocArray<HeatMapQueue::EntryType>
                                   (queueSize, CACHE_LINE_SIZE);
            mHeatMapQueue->init(queueSize, mHeatMapEntries);
            // HeatMap queue handler is setup by the RenderDriver.
        }
    }

    std::fill(mPrimaryRaysSubmitted, mPrimaryRaysSubmitted + MAX_RENDER_PASSES, 0);

    mRayRecorder = new DebugRayRecorder(tls->mThreadIdx);
    mRayVertexStack.reserve(32);
}

TLState::~TLState()
{
    MNRY_ASSERT(gPrivate.mRefCount);

    delete mRayRecorder;

    scene_rdl2::alignedFreeDtor(mRadianceQueue);
    scene_rdl2::alignedFreeDtor(mAovQueue);
    scene_rdl2::alignedFreeDtor(mHeatMapQueue);
    scene_rdl2::alignedFreeArray(mRadianceEntries);
    scene_rdl2::alignedFreeArray(mAovEntries);
    scene_rdl2::alignedFreeArray(mOcclusionEntries);
    scene_rdl2::alignedFreeArray(mPresenceShadowsEntries);
    scene_rdl2::alignedFreeArray(mRayEntries);
    scene_rdl2::alignedFreeArray(mHeatMapEntries);

    mRayStatePool.cleanUp();
    mCL1Pool.cleanUp();

    {
        // Protect against races the during gPrivate clean up.
        tbb::mutex::scoped_lock lock(gInitMutex);

        MOONRAY_THREADSAFE_STATIC_WRITE(--gPrivate.mRefCount);
        if (gPrivate.mRefCount == 0) {
            cleanUpPrivate();
        }
    }

    scene_rdl2::alignedFreeDtor(mExclusiveAccumulators);
}

void
TLState::reset()
{
    mRayQueue.reset();
    mOcclusionQueue.reset();
    mPresenceShadowsQueue.reset();

    if (mRadianceQueue) {
        mRadianceQueue->reset();
    }
    if (mAovQueue) {
        mAovQueue->reset();
    }
    if (mHeatMapQueue) {
        mHeatMapQueue->reset();
    }

    setAllQueueSizes(1.f);

    std::fill(mPrimaryRaysSubmitted, mPrimaryRaysSubmitted + MAX_RENDER_PASSES, 0);

    mFs = nullptr;

    mTilesRenderedTo.clearAll();

    mCancellationState = DISABLED;
    mCurrentPassIdx = 0;

    // From class BaseTLState.
    mIspcAccumulator = nullptr;

    MOONRAY_THREADSAFE_STATIC_WRITE(gFailedRayStateAllocs = 0);
    MOONRAY_THREADSAFE_STATIC_WRITE(gFailedCL1Allocs = 0);
}

template<typename ResType, typename PoolType>
inline void
TLState::poolAlloc(const char * const typeName,
                   PoolType &pool,
                   unsigned numEntries,
                   ResType **entries,
                   OverlappedAccType accumStall,
                   tbb::atomic<unsigned> &numFailedAllocs)
{
    // 99.9999% case, allocation should succeed.
    bool success = pool.allocList(numEntries, entries);

    if (!success) {

        // 0.0001% case:

        // Watchdog loop, report if we can't allocate a new entry
        // within a set amount of time.
        ACCUMULATOR_PROFILE(mThreadIdx, accumStall);

        const double allocTimeout = 1.f;
        double time = scene_rdl2::util::getSeconds();

        do {
            ++numFailedAllocs;
            scene_rdl2::logging::Logger::warn("Couldn't allocate a ", typeName, " on thread ", mThreadIdx,
                " (numEntries: ", numEntries, " sizeof(Entry): ", sizeof(ResType),
                "), falling back to early draining of queues (", numFailedAllocs, ").\n");

            // Try and free up some entries with an explicit flush.
            flushLocalQueues();

            if (pool.allocList(numEntries, entries)) {
                success = true;
                break;
            }

            // Sleep for a while as a last resort.
            mcrt_common::threadSleep();

        } while (scene_rdl2::util::getSeconds() - time < allocTimeout);

        if (!success) {
            scene_rdl2::logging::Logger::fatal("Rendering thread timed out whilst trying to allocate a ",
                          typeName, "! Try increasing pool size.");
        }
    }
}

RayState **
TLState::allocRayStates(unsigned numRayStates)
{
    EXCL_ACCUMULATOR_PROFILE(this, EXCL_ACCUM_RAYSTATE_ALLOCS);

    MNRY_ASSERT(numRayStates);

    RayState **rayStates = mArena->allocArray<RayState *>(numRayStates);

    poolAlloc<RayState, RayStatePool>("RayState", mRayStatePool, numRayStates,
                                      rayStates, ACCUM_RAYSTATE_STALLS,
                                      gFailedRayStateAllocs);

#ifdef DEBUG_RECORD_PEAK_RAYSTATE_USAGE
    // Slowdown here - turn off unless doing memory profiles.
    unsigned numRayStatesAllocated = 0;
    forEachTLS([&](pbr::TLState *tls) {
        numRayStatesAllocated += tls->mRayStatePool.getNumEntriesAllocated();
    });

    if (numRayStatesAllocated > gPeakRayStateUsage) {
        MOONRAY_THREADSAFE_STATIC_WRITE(gPeakRayStateUsage = numRayStatesAllocated);
    }
#endif

    return rayStates;
}

uint32_t
TLState::acquireDeepData(uint32_t deepDataHandle)
{
    if (deepDataHandle != pbr::nullHandle) {
        pbr::DeepData *deepData = static_cast<pbr::DeepData*>(getListItem(deepDataHandle, 0));
        ++(deepData->mRefCount);
    }
    return deepDataHandle;
}

void
TLState::releaseDeepData(uint32_t deepDataHandle)
{
    if (deepDataHandle != pbr::nullHandle) {
        pbr::DeepData *deepData = static_cast<pbr::DeepData*>(getListItem(deepDataHandle, 0));
        MNRY_ASSERT(deepData->mRefCount > 0);
        if (--(deepData->mRefCount) == 0) {
            freeList(deepDataHandle);
        }
    }
}

uint32_t
TLState::acquireCryptomatteData(uint32_t cryptomatteDataHandle)
{
    if (cryptomatteDataHandle != pbr::nullHandle) {
        pbr::CryptomatteData *cryptomatteData =
            static_cast<pbr::CryptomatteData*>(getListItem(cryptomatteDataHandle, 0));
        ++(cryptomatteData->mRefCount);
    }
    return cryptomatteDataHandle;
}

void
TLState::releaseCryptomatteData(uint32_t cryptomatteDataHandle)
{
    if (cryptomatteDataHandle != pbr::nullHandle) {
        pbr::CryptomatteData *cryptomatteData =
            static_cast<pbr::CryptomatteData*>(getListItem(cryptomatteDataHandle, 0));
        MNRY_ASSERT(cryptomatteData->mRefCount > 0);
        if (--(cryptomatteData->mRefCount) == 0) {
            freeList(cryptomatteDataHandle);
        }
    }
}

void
TLState::freeRayStates(unsigned numRayStates, RayState **rayStates)
{
    EXCL_ACCUMULATOR_PROFILE(this, EXCL_ACCUM_RAYSTATE_ALLOCS);

    for (size_t i = 0; i < numRayStates; i++) {
        RayState *rs = rayStates[i];
        if (rs->mDeepDataHandle != nullHandle) {
            releaseDeepData(rs->mDeepDataHandle);
        }
        if (rs->mCryptomatteDataHandle != nullHandle) {
            releaseCryptomatteData(rs->mCryptomatteDataHandle);
        }
    }

    mRayStatePool.freeList(numRayStates, rayStates);
}

bool
TLState::verifyNoOutstandingAllocs()
{
    MNRY_ASSERT(mRayStatePool.verifyNoOutstandingAllocs());
    MNRY_ASSERT(mCL1Pool.verifyNoOutstandingAllocs());

    return true;
}

void
TLState::addRayQueueEntries(unsigned numEntries, RayState **entries)
{
    if (!numEntries) {
        return;
    }
    if (mXPURayQueue) {
        mXPURayQueue->addEntries(mTopLevelTls, numEntries, entries);
    } else {
        mRayQueue.addEntries(mTopLevelTls, numEntries, entries, mArena);
    }
}

void
TLState::addRadianceQueueEntries(unsigned numEntries, BundledRadiance *entries)
{
    if (!numEntries) {
        return;
    }
    mRadianceQueue->addEntries(mTopLevelTls, numEntries, entries, mArena);
}

void
TLState::addAovQueueEntries(unsigned numEntries, BundledAov *entries)
{
    if (!numEntries) {
        return;
    }
    mAovQueue->addEntries(mTopLevelTls, numEntries, entries, mArena);
}

void
TLState::addHeatMapQueueEntries(unsigned numEntries, BundledHeatMapSample *entries)
{
    if (!numEntries) {
        return;
    }
    mHeatMapQueue->addEntries(mTopLevelTls, numEntries, entries, mArena);
}

void
TLState::addOcclusionQueueEntries(unsigned numEntries, BundledOcclRay *entries)
{
    if (!numEntries) {
        return;
    }
    if (mXPUOcclusionRayQueue) {
        mXPUOcclusionRayQueue->addEntries(mTopLevelTls, numEntries, entries);
    } else {
        mOcclusionQueue.addEntries(mTopLevelTls, numEntries, entries, mArena);
    }
}

void
TLState::addPresenceShadowsQueueEntries(unsigned numEntries, BundledOcclRay *entries)
{
    if (!numEntries) {
        return;
    }
    mPresenceShadowsQueue.addEntries(mTopLevelTls, numEntries, entries, mArena);
}

void
TLState::setXPUOcclusionRayQueue(XPUOcclusionRayQueue* queue)
{
    mXPUOcclusionRayQueue = queue;
}

void
TLState::setXPURayQueue(XPURayQueue* queue)
{
    mXPURayQueue = queue;
}

void
TLState::flushRadianceQueue()
{
     mRadianceQueue->flush(mTopLevelTls, mArena);
}

unsigned
TLState::flushLocalQueues()
{
    if (MNRY_VERIFY(mFs)->mExecutionMode == mcrt_common::ExecutionMode::SCALAR) {
        return 0;
    }

    unsigned processed = 0;

    processed += mRayQueue.flush(mTopLevelTls, mArena);
    if (isCanceled()) {
        return 0;
    }

    processed += mOcclusionQueue.flush(mTopLevelTls, mArena);
    if (isCanceled()) {
        return 0;
    }

    processed += mPresenceShadowsQueue.flush(mTopLevelTls, mArena);
    if (isCanceled()) {
        return 0;
    }

    processed += mRadianceQueue->flush(mTopLevelTls, mArena);
    processed += mAovQueue->flush(mTopLevelTls, mArena);
    processed += mHeatMapQueue->flush(mTopLevelTls, mArena);

    if (isCanceled()) {
        return 0;
    }

    return processed;
}

bool
TLState::areAllLocalQueuesEmpty()
{
    if (!mRayQueue.isEmpty()) {
        return false;
    }

    if (!mOcclusionQueue.isEmpty()) {
        return false;
    }

    if (!mPresenceShadowsQueue.isEmpty()) {
        return false;
    }

    if (!mRadianceQueue->isEmpty()) {
        return false;
    }
    if (!mAovQueue->isEmpty()) {
        return false;
    }
    if (!mHeatMapQueue->isEmpty()) {
        return false;
    }

    return true;
}

void
TLState::setAllQueueSizes(float t)
{
    MNRY_ASSERT(t >= 0.f && t <= 1.f);

    setQueueSize(&mRayQueue, t);
    setQueueSize(&mOcclusionQueue, t);
    setQueueSize(&mPresenceShadowsQueue, t);

    if (mRadianceQueue) {
        setQueueSize(mRadianceQueue, t);
    }
    if (mAovQueue) {
        setQueueSize(mAovQueue, t);
    }
    if (mHeatMapQueue) {
        setQueueSize(mHeatMapQueue, t);
    }
}

void
TLState::enableCancellation(bool waitUntilReadyForDisplay)
{
    MNRY_ASSERT(mFs);

    // To ignore the sample per pixel constraint, mCancellationState can be
    // initialized to WAITING_FOR_CANCEL. We do this for batch mode.
    mCancellationState = waitUntilReadyForDisplay ? WAITING_FOR_SAMPLE_PER_PIXEL
                                                  : WAITING_FOR_CANCEL;
    mCurrentPassIdx = 0;
}

void
TLState::disableCancellation()
{
    MNRY_ASSERT(mFs);
    mCancellationState = DISABLED;
    mCurrentPassIdx = 0;
}

bool
TLState::isCanceled()
{
    if (mCancellationState <= CANCELED) {
        return mCancellationState != DISABLED;
    }

    MNRY_ASSERT(mFs);

    switch (mCancellationState) {

    case WAITING_FOR_SAMPLE_PER_PIXEL:
        if (mCurrentPassIdx > 0) {
            mCancellationState = WAITING_FOR_CANCEL;
            if (isRenderCanceled()) {
                mCancellationState = CANCELED;
                return true;
            }
        }
        break;

    case WAITING_FOR_CANCEL:
        if (isRenderCanceled()) {
            mCancellationState = CANCELED;
            return true;
        }
        break;

    default:
        MNRY_ASSERT(0);
    }

    return false;
}

void
TLState::cacheThreadLocalAccumulators()
{
    MNRY_VERIFY(mExclusiveAccumulators)->cacheThreadLocalAccumulators(mThreadIdx);
    mIspcAccumulator = &getAccumulator(ACCUM_TOTAL_ISPC)->mThreadLocal[mThreadIdx];
}

bool
TLState::isIntegratorAccumulatorRunning() const
{
    return mExclusiveAccumulators->isRunning(EXCL_ACCUM_INTEGRATION) ||
           mExclusiveAccumulators->isRunning(EXCL_ACCUM_SSS_INTEGRATION) ||
           mExclusiveAccumulators->isRunning(EXCL_ACCUM_VOL_INTEGRATION);
}

void
TLState::setRadianceQueueHandler(RadianceQueue::Handler handler, void *handlerData)
{
    mRadianceQueue->setHandler(handler, handlerData);
}

void
TLState::setAovQueueHandler(AovQueue::Handler handler, void *handlerData)
{
    mAovQueue->setHandler(handler, handlerData);
}

void
TLState::setHeatMapQueueHandler(HeatMapQueue::Handler handler, void *handlerData)
{
    mHeatMapQueue->setHandler(handler, handlerData);
}

std::shared_ptr<TLState>
TLState::allocTls(mcrt_common::ThreadLocalState *tls,
                  const mcrt_common::TLSInitParams &initParams,
                  bool okToAllocBundledResources)
{
    {
        // Protect against races the very first time we initialize gPrivate.
        tbb::mutex::scoped_lock lock(gInitMutex);

        if (gPrivate.mRefCount == 0) {
            initPrivate(initParams);
        }

        MOONRAY_THREADSAFE_STATIC_WRITE(++gPrivate.mRefCount);
    }

    return std::make_shared<pbr::TLState>(tls, initParams, okToAllocBundledResources);
}

void
resetPools()
{
    if (gPrivate.mRefCount == 0) {
        return;
    }

    gPrivate.mRayState.mMemBlockManager->fullReset();
    gPrivate.mCL1.mMemBlockManager->fullReset();

    forEachTLS([&](TLState *tls) {
        tls->mRayStatePool.fastReset();   // The call to fastReset is deliberate.
        tls->mCL1Pool.fastReset();
    });

#ifdef DEBUG_RECORD_PEAK_RAYSTATE_USAGE
    static unsigned prevPeakRayStateUsage = 0;
    if (gPeakRayStateUsage > prevPeakRayStateUsage) {
        scene_rdl2::logging::Logger::info("\nPeak RayState usage = ", gPeakRayStateUsage, ", (", gPrivate.mRayState.mActualPoolSize, " allocated)\n");
        MOONRAY_THREADSAFE_STATIC_WRITE(prevPeakRayStateUsage = gPeakRayStateUsage);
    }
#endif

#ifdef DEBUG_RECORD_PEAK_CL1_USAGE
    static unsigned prevPeakCL1Usage = 0;
    if (gPeakCL1Usage > prevPeakCL1Usage) {
        scene_rdl2::logging::Logger::info("\nPeak CL1 pool usage = ", gPeakCL1Usage, ", (", gPrivate.mCL1.mActualPoolSize, " allocated)\n");
        MOONRAY_THREADSAFE_STATIC_WRITE(prevPeakCL1Usage = gPeakCL1Usage);
    }
#endif
}

unsigned
getRayStatePoolSize()
{
    return gPrivate.mRayState.mActualPoolSize;
}

shading::RayStateIndex
rayStateToIndex(const RayState *rs)
{
    MNRY_ASSERT(gPrivate.mRayState.mEntryMemory);

    auto base = (const RayState *)gPrivate.mRayState.mEntryMemory;
    MNRY_ASSERT(rs >= base && rs < base + gPrivate.mRayState.mActualPoolSize);

    return shading::RayStateIndex(rs - base);
}

RayState *
indexToRayState(shading::RayStateIndex index)
{
    MNRY_ASSERT(gPrivate.mRayState.mEntryMemory);
    MNRY_ASSERT(index < gPrivate.mRayState.mActualPoolSize);

    RayState *base = (RayState *)gPrivate.mRayState.mEntryMemory;

    return base + index;
}

static inline void *
handleToPtr(unsigned handle)
{
    void *result = nullptr;

    CacheLine1 * const baseCL1 = (CacheLine1 *) gPrivate.mCL1.mEntryMemory;

    // mask off the info bits
    handle = handle & ~ALLOC_LIST_INFO_BITS;

    MNRY_ASSERT(handle < gPrivate.mCL1.mActualPoolSize);
    result = baseCL1 + handle;

    return result;
}

static inline uint32_t
ptrToHandle(void *bytes, unsigned numItems)
{
    MNRY_ASSERT(bytes);
    unsigned result = nullHandle;

    CacheLine1 * const baseCL1 = (CacheLine1 *) gPrivate.mCL1.mEntryMemory;

    if ((const CacheLine1 *) bytes >= baseCL1 &&
        (const CacheLine1 *) bytes < baseCL1 + gPrivate.mCL1.mActualPoolSize) {
        // from the cache line 1 pool
        result = (const CacheLine1 *) bytes - baseCL1;
    } else {
        MNRY_ASSERT(0 && "invalid pointer passed to ptrToHandle");
    }

    // encode number of items in the info bits
    MNRY_ASSERT(!(result & ALLOC_LIST_INFO_BITS));
    result = result | ((numItems - 1) << ALLOC_LIST_INFO_BIT_SHIFT);

    MNRY_ASSERT(handleToPtr(result) == bytes);

    return result;
}

uint32_t
TLState::allocList(unsigned itemSize, unsigned numItems)
{
    EXCL_ACCUMULATOR_PROFILE(this, EXCL_ACCUM_TLS_ALLOCS);

    // we need a 4 byte index for every item in the list
    MNRY_ASSERT(numItems <= ALLOC_LIST_MAX_NUM_ITEMS);
    if (numItems > ALLOC_LIST_MAX_NUM_ITEMS) {
        return nullHandle; // fail, should never happen
    }

    // itemSize must be within our max allocation
    MNRY_ASSERT(itemSize <= ALLOC_LIST_MAX_ITEM_SIZE);
    if (itemSize > ALLOC_LIST_MAX_ITEM_SIZE) {
        return nullHandle; // fail, should never happen
    }

    uint32_t *header = nullptr;

    if (numItems == 1) {
        // special case a list of 1 item.  there is no need to allocate
        // a header and an item.
        CacheLine1 *results[1];
        poolAlloc<CacheLine1, CL1Pool>("CacheLine1 (1 item)", mCL1Pool, numItems,
                                       results, ACCUM_CL1_ALLOC_STALLS, gFailedCL1Allocs);
        header = reinterpret_cast<uint32_t *>(results[0]);
    } else {
        // allocate numItems + 1 (for the header)
        CacheLine1 *results[ALLOC_LIST_MAX_NUM_ITEMS + 1];
        poolAlloc<CacheLine1, CL1Pool>("CacheLine1", mCL1Pool, numItems + 1,
                                       results, ACCUM_CL1_ALLOC_STALLS, gFailedCL1Allocs);

        // initialize the header
        header = reinterpret_cast<uint32_t *>(results[0]);
        for (unsigned i = 0; i < numItems; ++i) {
            header[i] = ptrToHandle(results[i + 1], /* numItems = */ 1);
        }

        // set a nullIndex to indicate end of list, if we used less
        // than a full cache line
        const int HEADER_ENTRIES_PER_LINE = CACHE_LINE_SIZE / sizeof(uint32_t);
        if (numItems % HEADER_ENTRIES_PER_LINE) {
            header[numItems] = nullHandle;
        }
    }

#ifdef DEBUG_RECORD_PEAK_CL1_USAGE
    // Slowdown here - turn off unless doing memory profiles.
    unsigned numAllocated = 0;
    forEachTLS([&](pbr::TLState *tls) {
            numAllocated += tls->mCL1Pool.getNumEntriesAllocated();
        });

    if (numAllocated > gPeakCL1Usage) {
        MOONRAY_THREADSAFE_STATIC_WRITE(gPeakCL1Usage = numAllocated);
    }
#endif

    // return pointer to list header, as an index
    return ptrToHandle(header, numItems);
}

void
TLState::freeList(uint32_t listPtr)
{
    EXCL_ACCUMULATOR_PROFILE(this, EXCL_ACCUM_TLS_ALLOCS);

    MNRY_ASSERT(listPtr != nullHandle);

    CacheLine1 *results[ALLOC_LIST_MAX_NUM_ITEMS + 1];
    results[0] = static_cast<CacheLine1 *>(handleToPtr(listPtr));

    MNRY_ASSERT((results[0] >= (CacheLine1 *) gPrivate.mCL1.mEntryMemory) &&
               (results[0] < ((CacheLine1 *) gPrivate.mCL1.mEntryMemory) + gPrivate.mCL1.mActualPoolSize));

    const unsigned numItems = getNumListItems(listPtr);
    unsigned numResultsToFree = numItems;

    if (numItems > 1) {
        // we are an actual list, fill out list pointers
        uint32_t *header = reinterpret_cast<uint32_t *>(results[0]);

        MNRY_ASSERT(numItems <= ALLOC_LIST_MAX_NUM_ITEMS);

        for (unsigned i = 0; i < numItems; ++i) {
            results[i + 1] = static_cast<CacheLine1 *>(handleToPtr(header[i]));
            MNRY_ASSERT((results[i + 1] >= (CacheLine1 *) gPrivate.mCL1.mEntryMemory) &&
                       (results[i + 1] < ((CacheLine1 *) gPrivate.mCL1.mEntryMemory) +
                        gPrivate.mCL1.mActualPoolSize));
        }

        numResultsToFree = numItems + 1;
    }

    mCL1Pool.freeList(numResultsToFree, results);
}

unsigned
TLState::getNumListItems(uint32_t listPtr)
{
    // list length is stored in the info bits
    return ((listPtr & ALLOC_LIST_INFO_BITS) >> ALLOC_LIST_INFO_BIT_SHIFT) + 1;
}

void *
TLState::getListItem(uint32_t listPtr, unsigned item)
{
    const unsigned numListItems = getNumListItems(listPtr);
    MNRY_ASSERT(item < numListItems);

    // special case of a single item list has no header
    if (numListItems == 1) {
        return handleToPtr(listPtr);
    }

    uint32_t *header = static_cast<uint32_t *>(handleToPtr(listPtr));
    return handleToPtr(header[item]);
}


// hooks for ispc - must match declarations in TLState.ispc
extern "C" uint32_t
CPP_PbrTLState_allocList(TLState *pbrTls, unsigned itemSize, unsigned numItems)
{
    return pbrTls->allocList(itemSize, numItems);
}

extern "C" void
CPP_PbrTLState_freeList(TLState *pbrTls, uint32_t listPtr)
{
    return pbrTls->freeList(listPtr);
}

extern "C" unsigned
CPP_PbrTLState_getNumListItems(TLState *pbrTls, uint32_t listPtr)
{
    return pbrTls->getNumListItems(listPtr);
}

extern "C" void *
CPP_PbrTLState_getListItem(TLState *pbrTls, uint32_t listPtr, unsigned item)
{
    return pbrTls->getListItem(listPtr, item);
}

extern "C" void
CPP_PbrTLState_acquireDeepData(TLState *pbrTls, uint32_t deepPtr)
{
    pbrTls->acquireDeepData(deepPtr);
}

extern "C" void
CPP_PbrTLState_releaseDeepData(TLState *pbrTls, uint32_t deepPtr)
{
    pbrTls->releaseDeepData(deepPtr);
}

extern "C" void
CPP_PbrTLState_acquireCryptomatteData(TLState *pbrTls, uint32_t cryptomattePtr)
{
    pbrTls->acquireCryptomatteData(cryptomattePtr);
}

extern "C" void
CPP_PbrTLState_releaseCryptomatteData(TLState *pbrTls, uint32_t cryptomattePtr)
{
    pbrTls->releaseCryptomatteData(cryptomattePtr);
}

void heatMapBundledUpdate(TLState *pbrTls,
                          int64_t ticks,
                          const pbr::RayState * const *rayStates,
                          unsigned numEntries)
{
    MNRY_ASSERT(numEntries > 0);
    // This function is an estimate!  It assumes that the amount
    // of time spent on any one of the ray-states is equal to any of the
    // others.  So it works best when numEntries is minimal.
    const int64_t ticksPerEntry = ticks / numEntries;

    if (!ticksPerEntry) return;

    SCOPED_MEM(pbrTls->mArena);

    BundledHeatMapSample *b = pbrTls->mArena->allocArray<BundledHeatMapSample>(numEntries, CACHE_LINE_SIZE);
    unsigned bIndx;
    BundledHeatMapSample::initArray(b, bIndx, rayStates[0]->mSubpixel.mPixel);

    // This works best when there is some pixel coherence
    // in rayStates.  If random w.r.t pixels, then we'll
    // submit an entry for every ray state.
    for (unsigned i = 0; i < numEntries; ++i) {
        BundledHeatMapSample::addTicksToArray(b, bIndx, ticksPerEntry,
                                              rayStates[i]->mSubpixel.mPixel);
    }

    pbrTls->addHeatMapQueueEntries(bIndx + 1, b);
}

HUD_VALIDATOR( PbrTLState );

//-----------------------------------------------------------------------------

} // namespace pbr
} // namespace moonray

