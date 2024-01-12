// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "ProfileAccumulatorHandles.h"
#include "ThreadLocalState.hh"
#include "Types.h"
#include <scene_rdl2/render/util/Arena.h>
#include <scene_rdl2/render/util/MemPool.h>
#include <memory>

using scene_rdl2::alloc::Arena;

namespace moonray {

namespace geom    {
namespace internal {
class TLState;
}
}

namespace pbr     { class TLState; }
namespace shading { class TLState; }

namespace mcrt_common {

class ThreadLocalState;

//-----------------------------------------------------------------------------

// Should contain just simple POD types.
struct TLSInitParams
{
    TLSInitParams();

    // @@@ By default we assume non-bundled execution for now. This is a temp
    // call to initialize bundled related variable to some reasonable defaults
    // suitable for bunded execution.
    void setVectorizedDefaults(bool realtimeRender);

    // Define the basic memory pool building block for RayState pool.
    typedef scene_rdl2::alloc::MemBlock<uint64_t, uint64_t> MemBlockType;

    // Set this to true if we are initializing in the context of performing unit
    // tests, in which case, we can take a bunch of short cuts.
    bool            mUnitTests;

    // Authoritative, pass in 0 to let the system decide.
    unsigned        mDesiredNumTBBThreads;

    scene_rdl2::alloc::ArenaBlockPool *mArenaBlockPool;

    // This is the total number of RayState objects allocated per thread.
    // In actual fact, we allocate mPerThreadRayStatePoolSize * numThreads
    // RayStates in a big pool which is shared between all threads. This means a
    // single thread is not limited to mPerThreadRayStatePoolSize objects for any
    // given single point in time.
    //
    // To figure out the optimal value for, grep for
    // DEBUG_RECORD_PEAK_RAYSTATE_USAGE in the source code.
    unsigned        mPerThreadRayStatePoolSize;

    // This controls the number of items in the TLState's memory
    // pools.  As with the ray state pool, we multiply this number
    // by numThreads to determine the total allocation size.
    unsigned        mPerThreadCL1PoolSize;

    // The number of entries in *each* thread local ray queue, set to
    // zero if not in bundled mode.
    unsigned        mRayQueueSize;

    // The number of entries in *each* thread local occlusion queue, set to zero
    // if not in bundled mode.
    unsigned        mOcclusionQueueSize;

    // The number of entries in *each* thread local presence shadows queue, set to zero
    // if not in bundled mode.
    unsigned        mPresenceShadowsQueueSize;

    // The number of entries in *each* material shade queue. Not thread local.
    unsigned        mShadeQueueSize;

    // The number of entries in *each* thread local radiance queue, set to zero
    // if not in bundled mode.
    unsigned        mRadianceQueueSize;

    // The number of entries in *each* thread local aov queue, set to zero
    // if not in bundled mode.
    unsigned        mAovQueueSize;

    // The number of entries in *each* thread local heat map queue, set to zero
    // if not in bundled mode.
    unsigned        mHeatMapQueueSize;

    // Callbacks for initializing TLState objects. Applications don't have to
    // fill these in manually.
    std::shared_ptr<geom::internal::TLState> (*initGeomTls) (ThreadLocalState *tls,
            const TLSInitParams &initParams, bool okToAllocBundledResources);
    std::shared_ptr<pbr::TLState> (*initPbrTls) (ThreadLocalState *tls,
            const TLSInitParams &initParams, bool okToAllocBundledResources);
    std::shared_ptr<shading::TLState> (*initShadingTls) (ThreadLocalState *tls,
            const TLSInitParams &initParams, bool okToAllocBundledResources);

    // Callback for grabbing the real OIIO Perthread object for a specific thread.
    // This is a callback so we can avoid creating a dependency on the shading
    // library.
    void (*initTLSTextureSupport)(shading::TLState *);
};

//-----------------------------------------------------------------------------

// Class to allow access to parts of the primary ThreadLocalState objects which
// should be accessable to each TLState object.
struct BaseTLState
{
    typedef mcrt_common::ThreadLocalAccumulator ThreadLocalAccumulator;

    BaseTLState(const uint32_t threadIdx, scene_rdl2::alloc::Arena &arena, scene_rdl2::alloc::Arena &pixelArena) :
        mThreadIdx(threadIdx),
        mArena(&arena),
        mPixelArena(&pixelArena),
        mIspcAccumulator(nullptr),
        mExclusiveAccumulatorsPtr(nullptr) {}

    virtual ~BaseTLState() {}

    // For profiling.
    ExclusiveAccumulators *getInternalExclusiveAccumulatorsPtr()  { return mExclusiveAccumulatorsPtr; }

    void startIspcAccumulator()             { if (mIspcAccumulator) mIspcAccumulator->start(); }
    void stopIspcAccumulator()              { if (mIspcAccumulator) mIspcAccumulator->stop(); }
    bool isIspcAccumulatorRunning() const   { return mIspcAccumulator ? mIspcAccumulator->canStop() : false; }

    // Call between frames to reset this TLState.
    virtual void reset() = 0;

    /// HUD validation.
    static uint32_t hudValidation(bool verbose) { BASE_TL_STATE_VALIDATION; }

    // This typedef is needed to since BASE_TL_STATE_MEMBERS can't refer
    // to namespaced members directly if they are required to be visible on the
    // ISPC side.
    typedef scene_rdl2::alloc::Arena Arena;

    BASE_TL_STATE_MEMBERS;  // look in ThreadLocalState.hh for definition
};

//-----------------------------------------------------------------------------

///
/// All RaaS related thread local data should go in here. This object is passed
/// around as an argument to many functions throughout the system. Any data put
/// in here doesn't need to be locked when written to or read from, and
/// operations don't need to be atomic (e.g. no need for tbb::atomics).
///
class CACHE_ALIGN ThreadLocalState
{
public:
                        ThreadLocalState(uint32_t threadIdx, bool allocBundledResources);

                        ~ThreadLocalState();

    // Call between frames to reset this TLState.
    void                reset();

    bool                checkForHandlerStackOverflowRisk();

    // Always keep this as the very first element in this struct so we can
    // easily convert from the pointer to the thread index.
    uint32_t            mPad;               // Pad so that mThreadIdx is at the same
                                            // offset in ThreadLocalState and BaseTLState.
    uint32_t            mHandlerStackDepth; // Used to prevent stack overflows
                                            // in vectorized mode.
    const uint32_t      mThreadIdx;         // Zero based incrementing counter

    // Primary thread local memory arena.
    scene_rdl2::alloc::Arena        mArena;

    // Arena for pixel data.  Persists for the rendering of an entire pixel.
    // This is only true for batch mode and is not useful for progressive and
    // realtime modes as they render pixels with multiple passes.
    scene_rdl2::alloc::Arena        mPixelArena;

    // Memory deallocation reponsibilities are given to shared_ptr. It is a
    // good fit since internally it does some magic so that it doesn't need to
    // be able to see the concrete destructor of an object in order to be able
    // to delete it.
    std::shared_ptr<geom::internal::TLState> mGeomTls;
    std::shared_ptr<pbr::TLState>            mPbrTls;
    std::shared_ptr<shading::TLState>        mShadingTls;

    DISALLOW_COPY_OR_ASSIGNMENT(ThreadLocalState);
};

//-----------------------------------------------------------------------------

#pragma warning push
#pragma warning disable 1875    // offsetof applied to non-POD (Plain Old Data)
                                // types is nonstandard
// mcrt_common::getThreadIdx depends on these conditions being true.
MNRY_STATIC_ASSERT(offsetof(ThreadLocalState, mThreadIdx) == 8);
MNRY_STATIC_ASSERT(sizeof(ThreadLocalState::mThreadIdx) == 4);
MNRY_STATIC_ASSERT(offsetof(BaseTLState, mThreadIdx) == 8);
MNRY_STATIC_ASSERT(sizeof(BaseTLState::mThreadIdx) == 4);
#pragma warning pop

/// Init calls must be executed before any calls to getFrameUpdateTLS().
/// Currently these are called automatically by the RenderDriver when it's
/// initialized.

// TLS initialization.
void initTLS(const TLSInitParams &initParams);

/// Clears the container of all instances of ThreadLocalState. Called by the
/// RenderDriver.
void cleanUpTLS();

/// This function may be called anytime after initTLS is executed to
/// retrieve the global TLS initialization params.
const TLSInitParams &getTLSInitParams();

// Authoritative, this returns the number of TBB threads we've allocated for
// rendering.
unsigned getNumTBBThreads();

/// Returns the list of render thread ThreadLocalState objects. This is useful
/// when we need to iterate over them. The GUI TLS is separate and can be retrieved
/// using the getGuiTLS() call. This call returns an array which is
/// getNumTBBThreads() elements in size, one per render thread.
ThreadLocalState *getTLSList();

// This will be one larger than the number of TBB threads allocated since we
// allocate an extra TLS for the GUI thread to use. The overflow pool isn't
// counted in the returned total.
unsigned getNumTLSAllocated();

// This is the number of tbb threads, plus one for the gui thread, plus
// our overflow pool size.  All thread indices will be one less than this
// value.  It can be used to allocate arrays that are indexed based on thread index.
// The downside is that this is usually an over allocation because the
// overflow pool is often unused.
unsigned getMaxNumTLS();

/// This function exists for cases where we need to perform some raytracing operation
/// on a GUI thread outside of our tbb threadpool. One example we have currently
/// is the orbit camera. Other potential use cases are mouse picking.
/// The returned TLS can be used whilst rendering without side effects.
/// The GUI TLS *is* counted as one of the TLS objects returned
/// by getNumTLSAllocated(). Its thread index will equal to the value returned
/// by getNumTBBThreads().
ThreadLocalState *getGuiTLS();

///
/// This can be called during the update phase of the frame to get a top level
/// ThreadLocalState object. It may *not* be called during the rendering phase
/// of a frame. Internally we reuse the same pool of TLS objects between both
/// phases, but the mapping of TLS objects to concrete OS threads may differ.
///
ThreadLocalState *getFrameUpdateTLS();

/// Internally we mark the start of render prep and mcrt phases of a frame.
/// This is to prevent calls to getFrameUpdateTLS() during the rendering portion
/// of the frame and vice versa.
void startUpdatePhaseOfFrame();
void startRenderPhaseOfFrame();

// Returns the CPU ticks per second as measured by __rdtsc.
// Actually recomputes the value internally so the longer the app runs for,
// the more accurate this number will be.
double computeTicksPerSecond();

//
// Convenience function for iterating over existing top level render TLS
// instances. This doesn't include the GUI TLS.
//
// Example of syntax:
//
//    forEachTLS([&](ThreadLocalState *tls)
//    {
//        tls->doWork();
//    });
//
template <typename Body>
finline void
forEachTLS(const Body &body)
{
    unsigned numTLS = getNumTBBThreads();
    ThreadLocalState *tlsList = getTLSList();
    for (unsigned i = 0; i < numTLS; ++i) {
        body(tlsList + i);
    }
}

} // namespace mcrt_common

inline mcrt_common::ExclusiveAccumulators *
getExclusiveAccumulators(mcrt_common::ThreadLocalState *tls)
{
    MNRY_ASSERT(tls);
    return getExclusiveAccumulators(tls->mPbrTls.get());
}

} // namespace moonray

