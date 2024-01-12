// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

//
//
#pragma once

#include "PbrTLState.hh"

#include "Statistics.h"
#include <moonray/rendering/mcrt_common/Bundle.h>
#include <moonray/rendering/mcrt_common/ProfileAccumulator.h>
#include <moonray/rendering/mcrt_common/ThreadLocalState.h>
#include <moonray/rendering/pbr/Types.h>
#include <moonray/rendering/shading/Types.h>
#include <moonray/rendering/texturing/sampler/TextureSampler.h>
#include <scene_rdl2/render/logging/logging.h>

#define CHECK_CANCELLATION(tls, action)             { if ((tls)->isCanceled()) { action; }; }
#define CHECK_CANCELLATION_IN_LOOP(tls, i, action)  { if ((i & 15) == 0 && (tls)->isCanceled()) { action; }; }

namespace moonray {
namespace pbr {

class DebugRayRecorder;
class DebugRayVertex;
class XPUOcclusionRayQueue;
class XPURayQueue;

// Expose for HUD validation.
class TLState;
typedef pbr::TLState PbrTLState;

//-----------------------------------------------------------------------------

///
/// All pbr related thread local data should go in here. One of these is
/// allocated per thread and held onto by the higher level ThreadLocalState
/// object defined in rendering/mcrt_common/ThreadLocalState.h.
///
class CACHE_ALIGN TLState : public mcrt_common::BaseTLState
{
public:
    typedef scene_rdl2::alloc::MemPool<mcrt_common::TLSInitParams::MemBlockType, RayState> RayStatePool;

    typedef uint8_t CacheLine1[1 * CACHE_LINE_SIZE];
    typedef scene_rdl2::alloc::MemPool<mcrt_common::TLSInitParams::MemBlockType, CacheLine1> CL1Pool;

    typedef mcrt_common::LocalQueue<RayState*>                 RayQueue;
    typedef mcrt_common::LocalLargeEntryQueue<BundledOcclRay>  OcclusionQueue;
    typedef mcrt_common::LocalLargeEntryQueue<BundledOcclRay>  PresenceShadowsQueue;

    typedef mcrt_common::ExclusiveAccumulators    ExclusiveAccumulators;

#pragma warning push
#pragma warning disable 1875
    // These lines generate warning #1875: offsetof applied to non-POD
    // (Plain Old Data) types is nonstandard
    typedef mcrt_common::LocalLargeEntryQueue<BundledRadiance,
        true, offsetof(BundledRadiance, mPixel)>  RadianceQueue;
    typedef mcrt_common::LocalLargeEntryQueue<BundledAov,
        true, offsetof(BundledAov, mPixel)> AovQueue;
    typedef mcrt_common::LocalLargeEntryQueue<BundledHeatMapSample,
        true, offsetof(BundledHeatMapSample, mPixel)> HeatMapQueue;
#pragma warning pop

    TLState(mcrt_common::ThreadLocalState *tls,
            const mcrt_common::TLSInitParams &initParams,
            bool okToAllocBundledResources);
    virtual ~TLState();

    // This resets everything except for the memory pools.
    // To reset those, call resetPools() separately.
    virtual void        reset() override;

    // RayState management.
    RayState **         allocRayStates(unsigned numRayStates);
    void                freeRayStates(unsigned numRayStates, RayState **rayStates);

    uint32_t acquireDeepData(uint32_t deepDataHandle);
    void releaseDeepData(uint32_t deepDataHandle);

    uint32_t acquireCryptomatteData(uint32_t cryptomatteDataHandle);
    void releaseCryptomatteData(uint32_t cryptomatteDataHandle);

    // Allocate a list of items.
    // Lists can be allocated on one thread, and efficiently freed on another.
    // Each item in the list is CACHE_LINE_SIZE aligned.
    // Max numItems is 16
    // Max itemSize is 64 bytes
    uint32_t           allocList(unsigned itemSize, unsigned numItems);
    void               freeList(uint32_t listPtr);
    unsigned           getNumListItems(uint32_t listPtr);
    void *             getListItem(uint32_t listPtr, unsigned item);

    // Verify that the RayState pool and memory pools
    // have no outstanding allocations.
    bool                verifyNoOutstandingAllocs();

    void                addRayQueueEntries(unsigned numEntries, RayState **entries);
    void                addRadianceQueueEntries(unsigned numEntries, BundledRadiance *entries);
    void                addAovQueueEntries(unsigned numEntries, BundledAov *entries);
    void                addOcclusionQueueEntries(unsigned numEntries, BundledOcclRay *entries);
    void                addPresenceShadowsQueueEntries(unsigned numEntries, BundledOcclRay *entries);
    void                addHeatMapQueueEntries(unsigned numEntries, BundledHeatMapSample *entries);

    // Sets the pointers to the XPU ray queues, which are owned by the RenderDriver
    // and are only not null if we are in XPU mode and RenderContext::renderPrep() has successfully
    // initialized the GPUAccelerator.
    void                setXPUOcclusionRayQueue(XPUOcclusionRayQueue* queue);
    void                setXPURayQueue(XPURayQueue* queue);

    //
    // Queue helpers:
    //
    void                flushRadianceQueue();
    unsigned            flushLocalQueues();
    bool                areAllLocalQueuesEmpty();

    // t is a value in [0, 1] which is a hint for what proportion of the max
    // queue entries to size each queue. It's useful to fine control the
    // balancing throughput vs. latency wrt to samples being displayed.
    void                setAllQueueSizes(float t);

    //
    // Cancellation functionality:
    //
    void                enableCancellation(bool waitUntilReadyForDisplay);
    void                disableCancellation();
    bool                isCanceled();

    //
    // Profiling:
    //
    void                cacheThreadLocalAccumulators();
    ExclusiveAccumulators *getInternalExclusiveAccumulators() { return mExclusiveAccumulators; }

    bool                isIntegratorAccumulatorRunning() const;

    //
    // For initialization. Should only be called from RenderDriver.
    //
    void                setRadianceQueueHandler(RadianceQueue::Handler handler,
                                                void *handlerData);
    void                setAovQueueHandler(AovQueue::Handler handler,
                                           void *handlerData);
    void                setHeatMapQueueHandler(HeatMapQueue::Handler handler,
                                               void *handlerData);

    // Used as a callback which is registered with TLSInitParams.
    static std::shared_ptr<TLState> allocTls(mcrt_common::ThreadLocalState *tls,
                                             const mcrt_common::TLSInitParams &initParams,
                                             bool okToAllocBundledResources);

    // HUD validation.
    static uint32_t hudValidation(bool verbose) { PBR_TL_STATE_VALIDATION; }

    //
    // Data:
    //
    typedef mcrt_common::ThreadLocalState ThreadLocalState;
    typedef mcrt_common::ThreadLocalAccumulator ThreadLocalAccumulator;

    // Cancellation functionality:
    enum CancellationState
    {
        DISABLED = 0,
        CANCELED = 1,
        WAITING_FOR_SAMPLE_PER_PIXEL = 2,
        WAITING_FOR_CANCEL = 3,
    };

    PBR_TL_STATE_MEMBERS;

private:
    template <typename QueueType>
    finline void addFilmQueueEntries(unsigned numEntries,
                                     typename QueueType::EntryType *entries,
                                     QueueType *queue);

    template <typename ResType, typename PoolType>
    finline void poolAlloc(const char * const typeName,
                           PoolType &pool,
                           unsigned numEntries,
                           ResType **entries,
                           OverlappedAccType accumStall,
                           tbb::atomic<unsigned> &numFailedAlloc);

    DISALLOW_COPY_OR_ASSIGNMENT(TLState);
};

//-----------------------------------------------------------------------------

// If this fails, it means you need update TLS_OFFSET_TO_EXCL_ACCUMULATORS to be the correct offset.
#pragma warning(push)
#pragma warning(disable:1684) // conversion from pointer to same-sized integral type (potential portability problem)
MNRY_STATIC_ASSERT((offsetof(TLState, mExclusiveAccumulators)) == TLS_OFFSET_TO_EXCL_ACCUMULATORS);
#pragma warning(pop)

// Shorten the TLS queue type names for convenience.
typedef TLState::RayStatePool      RayStatePool;
typedef TLState::RayQueue          RayQueue;
typedef TLState::OcclusionQueue    OcclusionQueue;
typedef TLState::PresenceShadowsQueue PresenceShadowsQueue;
typedef TLState::RadianceQueue     RadianceQueue;

typedef TLState::CacheLine1        CacheLine1;
typedef TLState::CL1Pool           CL1Pool;

//
// Convenience function for iterating over all existing pbr TLS instances.
//
// Example of syntax:
//
//    pbr::forEachTLS([&](pbr::TLState *tls)
//    {
//        tls->doWork();
//    });
//
template <typename Body>
finline void forEachTLS(Body const &body)
{
    unsigned numTLS = mcrt_common::getNumTBBThreads();
    mcrt_common::ThreadLocalState *tlsList = mcrt_common::getTLSList();
    for (unsigned i = 0; i < numTLS; ++i) {
        auto pbrTls = tlsList[i].mPbrTls.get();
        if (pbrTls) {
            body(pbrTls);
        }
    }
}

// Not thread safe. This takes care of reseting the
// ray state and memory pool allocators
void resetPools();

// Helpers for converting between a 4 byte and an 8 byte reference to a RayState.
shading::RayStateIndex rayStateToIndex(const RayState *rs);
RayState *indexToRayState(shading::RayStateIndex index);

// Value returned when allocList fails
const uint32_t nullHandle = PBR_TL_STATE_NULL_HANDLE;

// Helper function for accumulating, estimating, and submitting
// bundled pixel info from a total tick count and set of ray states.
void heatMapBundledUpdate(TLState *pbrTls,
                          int64_t ticks,
                          const pbr::RayState * const *rayStates,
                          unsigned numEntries);

//-----------------------------------------------------------------------------

} // namespace pbr
} // namespace moonray

