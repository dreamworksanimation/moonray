// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

//
//
#include "ThreadLocalState.h"
#include "ProfileAccumulatorHandles.h"
#include <moonray/common/mcrt_macros/moonray_static_check.h>
#include <moonray/common/time/Ticker.h>
#include <scene_rdl2/render/logging/logging.h>
#include <scene_rdl2/render/util/Memory.h>
#include <tbb/enumerable_thread_specific.h>
#include <tbb/task_scheduler_init.h>

// There are on average 3 entries added to the profiler stack for each single
// entry on the handler stack. This heuristic is used to compute the
// MAX_HANDLER_STACK_SIZE.
#define MAX_HANDLER_STACK_SIZE      (MAX_EXCL_ACCUM_STACK_SIZE / 3)

// See comments in getFrameUpdateTLS().
#define MAX_OVERFLOW_POOL_SIZE      4

// Mimic depth first code flow in bundled mode by forcing all queue sizes to 1.
// The will ensure that as soon as an entry is added, the queue will get flushed,
// and therefore any queue in the system can never hold more than 1 entry.
//#define FORCE_SINGLE_ENTRY_QUEUES

namespace ispc {
extern "C" uint32_t BaseTLState_hudValidation(bool);
}

using namespace scene_rdl2::alloc;
using namespace scene_rdl2::math;
using namespace scene_rdl2::util;

namespace moonray {

namespace mcrt_common {

// Private:
namespace
{

// Counter to hand out unique indices to TLSProxy objects.
tbb::atomic<unsigned> gNextFrameUpdateTLSIndex;

// These are lightweight objects which we put into a tbb::enumerable_thread_specific
// container so that we can map OS thread ids to consistent top level ThreadLocalState
// objects. We have one type for frame updates, and a second type for rendering.
struct FrameUpdateTLSProxy
{
    FrameUpdateTLSProxy() :
        mTLSIndex(gNextFrameUpdateTLSIndex.fetch_and_increment())
    {
    }

    // Index into Private::mTLSList which resolves to a concrete
    // ThreadLocalState object.
    unsigned mTLSIndex;
};

//
// Data shared between all threads.
//
struct Private
{
    Private() :
        mTLSInitialized(false),
        mRenderPhaseOfFrame(false),
        mTLSList(nullptr),
        mGuiTLS(nullptr)
    {
        memset(mOverflowPool, 0, sizeof(mOverflowPool));
    }

    void cleanUpOverflowPool()
    {
        // mOverflowPool is an array of pointers which are all initialized to nullptr.
        // Iterate and free up objects until we hit the first nullptr, then our work is done.
        for (unsigned i = 0; i < MAX_OVERFLOW_POOL_SIZE; ++i) {
            if (mOverflowPool[i] == nullptr) {
                break;
            }
            alignedFreeDtor<ThreadLocalState>(mOverflowPool[i]);
        }

        memset(mOverflowPool, 0, sizeof(mOverflowPool));
    }

    TLSInitParams mInitParams;

    bool mTLSInitialized;
    bool mRenderPhaseOfFrame;

    // Full featured render thread TLS objects.
    ThreadLocalState *mTLSList;

    // Create one extra "Unsafe TLS" that functions outside of our TBB thread
    // pool may be used for special operations such as picking.
    ThreadLocalState *mGuiTLS;

    // Used to compute an accurate tick frequency for accumulator timing purposes.
    time::TicksPerSecond mTicksPerSecond;

    /// Mapping of update phase thread local ids to concrete ThreadLocalState objects.
    tbb::enumerable_thread_specific<FrameUpdateTLSProxy> mFrameUpdateTLSMapping;

    // See comments in getUpdatePhaseTLS().
    ThreadLocalState *  mOverflowPool[MAX_OVERFLOW_POOL_SIZE];
};

Private gPrivate;

// Grab the very latest OIIO texture TLS in case it has somehow changed
// since we last grabbed it.
inline void
updateToLatestOIIOTLS(ThreadLocalState *tlsToUpdate)
{
    MNRY_ASSERT(tlsToUpdate);

    if (gPrivate.mInitParams.initTLSTextureSupport && tlsToUpdate->mShadingTls) {
        gPrivate.mInitParams.initTLSTextureSupport(tlsToUpdate->mShadingTls.get());
    }
}

}   // End of anon namespace.

//-----------------------------------------------------------------------------

TLSInitParams::TLSInitParams()
{
    // cppcheck-suppress memsetClassFloat // floating point memset to 0 is fine
    memset(this, 0, sizeof(TLSInitParams));
}

// TODO. Client code should be fine tuning the sizes of queues in bundled mode.
void TLSInitParams::setVectorizedDefaults(bool realtimeRender)
{
    mPerThreadRayStatePoolSize = 1024 * 64;

    // TLState allocList configuration
    // 64k items per thread is often way more than we need in CPU vector mode,
    // but XPU has larger queues and buffers more data.
    // For some production scenes, 512k items are needed.
    // Set to 1024k items to be safe.
    // Use the DEBUG_RECORD_PEAK_CL1_USAGE in TLState.cc to analyze.
    // number of items must fit in 24 bits (see TLSTate.cc for details)
    const int CL1_POOL_SIZE = 1024 * 1024;
    const int MAX_CL1_POOL_SIZE = 0x0FFFFFFF;
    MNRY_STATIC_ASSERT(CL1_POOL_SIZE < MAX_CL1_POOL_SIZE);
    mPerThreadCL1PoolSize = CL1_POOL_SIZE;

#ifdef FORCE_SINGLE_ENTRY_QUEUES
    mRayQueueSize = 1;
    mOcclusionQueueSize = 1;
    mPresenceShadowsQueueSize = 1;
    mShadeQueueSize = 1;
    mRadianceQueueSize = 1;
    mAovQueueSize = 1;
    mHeatMapQueueSize = 1;
#else

    if (!realtimeRender) {
        // BATCH mode parameters 
        mRayQueueSize = 1024;
        mOcclusionQueueSize = 1024;
        mPresenceShadowsQueueSize = 1024;
        mShadeQueueSize = 128;

    } else {
        // realtime render mode parameters
        mRayQueueSize = 128;
        mOcclusionQueueSize = 64;
        mPresenceShadowsQueueSize = 64;
        mShadeQueueSize = 64;
    }

    mRadianceQueueSize = 512;
    mAovQueueSize = 256;
    mHeatMapQueueSize = 256;
#endif
}

HUD_VALIDATOR( BaseTLState );

ThreadLocalState::ThreadLocalState(uint32_t threadIdx, bool okToAllocBundledResources) :
    mPad(0),
    mHandlerStackDepth(0),
    mThreadIdx(threadIdx)
{
    mArena.init(MNRY_VERIFY(gPrivate.mInitParams.mArenaBlockPool));
    mPixelArena.init(MNRY_VERIFY(gPrivate.mInitParams.mArenaBlockPool));

    //
    // Create always:
    //

    if (gPrivate.mInitParams.initShadingTls) {
        mShadingTls = gPrivate.mInitParams.initShadingTls(this, gPrivate.mInitParams,
                                                          okToAllocBundledResources);
    }

    if (gPrivate.mInitParams.initPbrTls) {
        mPbrTls = gPrivate.mInitParams.initPbrTls(this, gPrivate.mInitParams,
                                                  okToAllocBundledResources);
    }

    //
    // Only create if we're not the GUI TLS.
    //
    if (okToAllocBundledResources) {
        if (gPrivate.mInitParams.initGeomTls) {
            mGeomTls = gPrivate.mInitParams.initGeomTls(this, gPrivate.mInitParams,
                                                        okToAllocBundledResources);
        }
    }
}

ThreadLocalState::~ThreadLocalState()
{
    mGeomTls = nullptr;
    mPbrTls = nullptr;
    mShadingTls = nullptr;

    mArena.cleanUp();
    mPixelArena.cleanUp();
}

void
ThreadLocalState::reset()
{
    mHandlerStackDepth = 0;

    if (mGeomTls != nullptr) {
        reinterpret_cast<BaseTLState *>(mGeomTls.get())->reset();
    }

    if (mPbrTls != nullptr) {
        reinterpret_cast<BaseTLState *>(mPbrTls.get())->reset();
    }

    if (mShadingTls != nullptr) {
        reinterpret_cast<BaseTLState *>(mShadingTls.get())->reset();
    }

    mArena.clear();
    mPixelArena.clear();
}

bool
ThreadLocalState::checkForHandlerStackOverflowRisk()
{
    ExclusiveAccumulators *acc = MNRY_VERIFY(getExclusiveAccumulators(this));
    unsigned accStack = acc->getStackSize();

    // A little padding is added here for safety since each additional handler
    // put on the C stack may result in multiple profiler accumulators being
    // added to the accumulator stack, which would also be in risk of asserting
    // when overflowed.
    const unsigned accStackThreshold = (MAX_EXCL_ACCUM_STACK_SIZE - 20);

    // Check for underflow.
    MNRY_STATIC_ASSERT(accStackThreshold < 1000000);

    // Check for both accumulator and handler stack overflows.
    return accStack >= accStackThreshold ||
           mHandlerStackDepth >= MAX_HANDLER_STACK_SIZE;
}

//-----------------------------------------------------------------------------

void
initTLS(const TLSInitParams &initParams)
{
    if (gPrivate.mTLSInitialized) {
        return;
    }

    MNRY_ASSERT_REQUIRE(!gPrivate.mRenderPhaseOfFrame);

    MOONRAY_START_THREADSAFE_STATIC_WRITE;

    gPrivate.mInitParams = initParams;

    if (gPrivate.mInitParams.mDesiredNumTBBThreads == 0) {
        gPrivate.mInitParams.mDesiredNumTBBThreads = tbb::task_scheduler_init::default_num_threads();
    }

    MNRY_ASSERT_REQUIRE(gPrivate.mInitParams.mDesiredNumTBBThreads);

    gNextFrameUpdateTLSIndex = 0;

    // Initialize tick counter.
    gPrivate.mTicksPerSecond.init();

    //
    // Preallocate all render TLS objects.
    //

    unsigned numThreads = getNumTBBThreads();

    gPrivate.mTLSList = reinterpret_cast<ThreadLocalState *>
                            (scene_rdl2::util::alignedMalloc(sizeof(ThreadLocalState) * numThreads,
                            CACHE_LINE_SIZE));

    for (unsigned i = 0; i < numThreads; ++i) {
        new (&gPrivate.mTLSList[i]) ThreadLocalState(i, true);
    }

    // Create special GUI TLS here.
    gPrivate.mGuiTLS = reinterpret_cast<ThreadLocalState *>
                        (scene_rdl2::util::alignedMalloc(sizeof(ThreadLocalState),
                        alignof(ThreadLocalState)));
    new (gPrivate.mGuiTLS) ThreadLocalState(numThreads, false);

    // Verify that overflow pool is empty.
    MNRY_STATIC_ASSERT(MAX_OVERFLOW_POOL_SIZE > 0);
    MNRY_ASSERT_REQUIRE(gPrivate.mOverflowPool[0] == nullptr);

    // Setup profiling accumulators.
    gAccumulatorHandles.init(getNumTBBThreads(), getNumTLSAllocated());

    // TLS is now initialized.
    gPrivate.mTLSInitialized = true;

    MOONRAY_FINISH_THREADSAFE_STATIC_WRITE;
}

void
cleanUpTLS()
{
    if (!gPrivate.mTLSInitialized) {
        return;
    }

    MOONRAY_START_THREADSAFE_STATIC_WRITE;

    gAccumulatorHandles.cleanUp();

    gPrivate.mFrameUpdateTLSMapping.clear();

    MNRY_ASSERT_REQUIRE(gPrivate.mGuiTLS);
    gPrivate.mGuiTLS->~ThreadLocalState();
    scene_rdl2::util::alignedFree(gPrivate.mGuiTLS);

    MNRY_ASSERT_REQUIRE(gPrivate.mTLSList);
    unsigned numThreads = getNumTBBThreads();
    for (unsigned i = 0; i < numThreads; ++i) {
        gPrivate.mTLSList[i].~ThreadLocalState();
    }
    scene_rdl2::util::alignedFree(gPrivate.mTLSList);

    // Clean up any extraneous update phase TLS objects.
    gPrivate.cleanUpOverflowPool();

    // Reset internal TLS related data.
    gPrivate.~Private();
    new (&gPrivate) Private;

    MNRY_ASSERT_REQUIRE(gPrivate.mTLSInitialized == false);

    MOONRAY_FINISH_THREADSAFE_STATIC_WRITE;
}

const TLSInitParams &
getTLSInitParams()
{
    return gPrivate.mInitParams;
}

unsigned
getNumTBBThreads()
{
    return gPrivate.mInitParams.mDesiredNumTBBThreads;
}

ThreadLocalState *
getTLSList()
{
    return gPrivate.mTLSList;
}

unsigned
getNumTLSAllocated()
{
    // Add 1 to account for the GUI thread. We ignore any threads in the
    // overflow pool however.
    return getNumTBBThreads() + 1;
}

unsigned
getMaxNumTLS()
{
    
    return getNumTLSAllocated() + MAX_OVERFLOW_POOL_SIZE;
}

ThreadLocalState *
getGuiTLS()
{
    ThreadLocalState *tls = gPrivate.mGuiTLS;

    // Update the OIIO perthread tls.
    updateToLatestOIIOTLS(tls);

    return tls;
}

void
startUpdatePhaseOfFrame()
{
    MNRY_ASSERT_REQUIRE(gPrivate.mTLSInitialized);

    MOONRAY_THREADSAFE_STATIC_WRITE(gNextFrameUpdateTLSIndex = 0);

    gPrivate.cleanUpOverflowPool();
    gPrivate.mFrameUpdateTLSMapping.clear();

    MOONRAY_THREADSAFE_STATIC_WRITE(gPrivate.mRenderPhaseOfFrame = false);
}

void
startRenderPhaseOfFrame()
{
    MNRY_ASSERT_REQUIRE(gPrivate.mTLSInitialized);

    MOONRAY_THREADSAFE_STATIC_WRITE(gNextFrameUpdateTLSIndex = 0);

    gPrivate.cleanUpOverflowPool();
    gPrivate.mFrameUpdateTLSMapping.clear();

    MOONRAY_THREADSAFE_STATIC_WRITE(gPrivate.mRenderPhaseOfFrame = true);
}

ThreadLocalState *getFrameUpdateTLS()
{
    MNRY_ASSERT_REQUIRE(gPrivate.mTLSInitialized);
    MNRY_ASSERT_REQUIRE(!gPrivate.mRenderPhaseOfFrame);

    FrameUpdateTLSProxy *proxy = &gPrivate.mFrameUpdateTLSMapping.local();
    MNRY_ASSERT_REQUIRE(proxy);

    const unsigned globalTLSIdx = proxy->mTLSIndex;

    // 99.99% case.
    if (globalTLSIdx < getNumTBBThreads()) {

        ThreadLocalState *tls = gPrivate.mTLSList + globalTLSIdx;
        updateToLatestOIIOTLS(tls);
        return tls;
    }

    //
    // There can be some strange conditions where additional TBB threads are spawned
    // and added to the TBB thread pool on rare occasion.
    //
    // See https://software.intel.com/en-us/forums/intel-threading-building-blocks/topic/782789
    // for details. To counteract this, we allocate new TLS objects as needed on the fly so that
    // The function always returns a valid TLS object regardless. These go into an overflow pool
    // which is cleared when the update phase completes.
    //
    // The caveat to this scheme is that we don't know up front the complete range of thread
    // indices which will be returned, and they may not be tightly packed. This is relevant
    // if you need to access pre-allocated arrays using the returned thread index. Be careful!!
    //

    const unsigned overflowIdx = globalTLSIdx - getNumTBBThreads();
    MNRY_ASSERT_REQUIRE(overflowIdx < MAX_OVERFLOW_POOL_SIZE);

    if (gPrivate.mOverflowPool[overflowIdx]) {

        MNRY_ASSERT_REQUIRE(gPrivate.mOverflowPool[overflowIdx]->mThreadIdx == globalTLSIdx);
        updateToLatestOIIOTLS(gPrivate.mOverflowPool[overflowIdx]);
        return gPrivate.mOverflowPool[overflowIdx];
    }

    //
    // At this point we have no choice but to create a new TLS object and add it
    // to the overflow pool.
    //

    ThreadLocalState *overflowTLS = alignedMallocCtorArgs<ThreadLocalState>(CACHE_LINE_SIZE, globalTLSIdx, false);

    updateToLatestOIIOTLS(overflowTLS);

    MOONRAY_THREADSAFE_STATIC_WRITE(gPrivate.mOverflowPool[overflowIdx] = overflowTLS);

    return overflowTLS;
}

double computeTicksPerSecond()
{
    return gPrivate.mTicksPerSecond.getUncached();
}

} // namespace mcrt_common
} // namespace moonray

