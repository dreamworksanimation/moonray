// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <moonray/rendering/bvh/shading/RootShader.h>
#include <moonray/rendering/mcrt_common/AffinityManager.h>
#include <moonray/rendering/mcrt_common/Bundle.h>
#include <moonray/rendering/shading/Types.h>
#include <scene_rdl2/scene/rdl2/Map.h>
#include <scene_rdl2/scene/rdl2/Material.h>

#include <memory>

// ShadeQueue access log information is dumped at the end of rendering when this directive is used.
// This should be commented out for the release version.
//#define SHADEQUEUE_ACCESS_LOG

// This directive appies more precise checks but more cost.
//#define STRICT_CHECK_FOR_GETINFOPTR

namespace moonray {
namespace shading {

class Material;
typedef std::vector<Material *> MaterialPtrList;

typedef mcrt_common::SharedQueue<SortedRayState, true, offsetof(SortedRayState, mSortKey)> ShadeQueue;
typedef std::vector<ShadeQueue *> ShadeQueueList;

class ShadeQueueListInfo;
class ShadeQueueListInfoManager;

class ShadeQueueInfo
//
// This class keeps single ShadeQueue related memory which is allocated from a particular
// NUMA-node when memory affinity is enabled.
// Otherwise, memory is allocated regular malloc way when memory affinity is disabled.
//
{
public:
    ShadeQueueInfo(const unsigned numaNodeId,
                   const unsigned shadeQueueSize,
                   ShadeQueue::Handler handler,
                   const Material* material,
                   ShadeQueueListInfo* shadeQueueListInfoPtr);
    ~ShadeQueueInfo();

    ShadeQueue* getShadeQueue() const
    {
#ifdef SHADEQUEUE_ACCESS_LOG
        accessLog();
#endif // end SHADEQUEUE_ACCESS_LOG
        return mShadeQueue;
    }

    void freeMemory();
    
private:    

#ifdef SHADEQUEUE_ACCESS_LOG
    void accessLog() const;
#endif // end SHADEQUEUE_ACCESS_LOG

    //------------------------------

    const unsigned mNumaNodeId {~static_cast<unsigned>(0)};
    const Material* mMaterial {nullptr};
    ShadeQueueListInfo* mShadeQueueListInfo {nullptr};
    ShadeQueue* mShadeQueue {nullptr};
    ShadeQueue::EntryType* mShadeEntries {nullptr};

    unsigned mShadeEntriesAllocatedSize {0};
};

class ShadeQueueInfoManager
//
// This class keeps a multibank version of ShadeQueue when Memory-Affinity is enabled.
// Otherwise, keep standard ShadeQueue for Non-Memory-Affinity situations.
//
{
public:
    ShadeQueueInfoManager(const unsigned shadeQueueSize,
                          ShadeQueue::Handler handler,
                          Material* material,
                          ShadeQueueListInfoManager* shadeQueueListInfoManager);

    unsigned getMaterialId() const { return mMaterialId; }
    size_t getNumaNodeTblSize() const { return mNumaNodeShadeQueueInfoTbl.size(); }

    ShadeQueue* getShadeQueue(const unsigned numaNodeId) const
    {
        return getShadeQueueInfoPtr(numaNodeId)->getShadeQueue();
    }

private:
    ShadeQueueInfo* getShadeQueueInfoPtr(const unsigned numaNodeId) const
    {
        if (numaNodeId == ~0) return mShadeQueueInfo.get();
#ifdef STRICT_CHECK_FOR_GETQINFOPTR
        if (numaNodeId >= getNumaNodeTblSize()) {
            std::cerr << "RUNTIME_ERROR: ShadeQueueInfoManager getShadeQueueInfoPtr() failed."
                      << " numaNodeId:" << numaNodeId
                      << " >= getNumaNodeTblSize():" << getNumaNodeTblSize() << '\n';
            return nullptr;
        }
        if (!mNumaNodeShadeQueueInfoTbl[numaNodeId]) {
            std::cerr << "RUNTIME_ERROR: ShadeQueueInfoManager getShadeQueueInfoPtr() failed."
                      << " numaNodeId:" << numaNodeId
                      << " mNumaNodeShadeQueueInfoTbl[numaNodeId] is nullptr\n";
            return nullptr;
        }
#endif // end STRICT_CHECK_FOR_GETQINFOPTR
        return mNumaNodeShadeQueueInfoTbl[numaNodeId].get();
    }

    unsigned mMaterialId {0};
    std::shared_ptr<ShadeQueueInfo> mShadeQueueInfo; // for non NUMA-node based memory management

    // size == total NUMA node of this host, some of them might be null
    std::vector<std::shared_ptr<ShadeQueueInfo>> mNumaNodeShadeQueueInfoTbl;
};

//------------------------------------------------------------------------------------------

class ShadeQueueListInfo
//
// This class keeps all the Material's ShadeQueue for one of the NUMA-node as a list when memory
// affinity is enabled.
// Keep all the Material's ShadeQueue regarding the entire MoonRay process when memory affinity
// is disabled.
//
{
public:
    ShadeQueueListInfo(const unsigned numaNodeId)
        : mNumaNodeId {numaNodeId}
    {}

    unsigned pushBackShadeQueue(ShadeQueue* shadeQueue)
    {
        mShadeQueues.push_back(shadeQueue);
        return mShadeQueues.size();
    }

    void removeShadeQueue(ShadeQueue* shadeQueue); // MTsafe

    bool areAllShadeQueueEmpty() const;
    unsigned flushNonEmptyShadeQueue(mcrt_common::ThreadLocalState* tls);

    unsigned getShadeQueuesSize() const { return mShadeQueues.size(); }
    ShadeQueueList& getShadeQueues() { return mShadeQueues; }

    std::string show() const;

private:
    const unsigned mNumaNodeId {~static_cast<unsigned>(0)};
    std::mutex mShadeQueueMutex;
    ShadeQueueList mShadeQueues;

    // This is used by the flushNonEmptyShadeQueue function to iterate through all queues
    // in a cyclic fashion as opposed to starting the iteration at the beginning of the
    // queue list each time.
    std::atomic<size_t> mFlushCycleIdx {0};
};

class ShadeQueueListInfoManager
//
// This class keeps a multibank version of ShadeQueue list when Memory-Affinity is enabled.
// Otherwise, keep the standard ShadeQueue list for Non-Memory-Affinity situations.
//
{
public:
    ShadeQueueListInfoManager();

    unsigned getAllShadeQueuesCount();

    ShadeQueueListInfo* getShadeQueueListInfoPtr(const unsigned numaNodeId) const
    {
        if (numaNodeId == ~0) return mShadeQueueListInfo.get();
#ifdef STRICT_CHECK_FOR_GETQINFOPTR
        if (numaNodeId >= getNumaNodeTblSize()) {
            std::cerr << "RUNTIME_ERROR: ShadeQueueListInfoManager getShadeQueueListInfoPtr() failed."
                      << " numaNodeId:" << numaNodeId
                      << " >= getNumaNodeTblSize():" << getnumaNodeTblSisze() << '\n';
            return nullptr;
        }
        if (!mNumaNodeShadeQueueListInfoTbl[numaNodeId]) {
            std::cerr << "RUNTIME_ERROR : ShadeQueueListInfoManager getShadeQueueListInfoPtr() failed."
                      << " numaNodeId:" << numaNodeId
                      << " mNumaNodeShadeQueueListInfoTbl[numaNodeId] is nullptr\n";
            return nullptr;
        }
#endif // end STRICT_CHECK_FOR_GETINFOPTR
        return mNumaNodeShadeQueueListInfoTbl[numaNodeId].get();
    }

    bool getMemAffinityEnable() const { return mMemAffinityEnable; }
    size_t getNumaNodeTblSize() const { return mNumaNodeShadeQueueListInfoTbl.size(); }

    unsigned pushBackShadeQueue(const unsigned numaNodeId, ShadeQueue* shadeQueue)
    {
        return getShadeQueueListInfoPtr(numaNodeId)->pushBackShadeQueue(shadeQueue);
    }

    bool areAllShadeQueueEmpty(const unsigned numaNodeId) const
    {
        return getShadeQueueListInfoPtr(numaNodeId)->areAllShadeQueueEmpty();
    }
    unsigned flushNonEmptyShadeQueue(const unsigned numaNodeId, mcrt_common::ThreadLocalState* tls)
    {
        return getShadeQueueListInfoPtr(numaNodeId)->flushNonEmptyShadeQueue(tls);
    }
    
    ShadeQueueList& getShadeQueues(const unsigned numaNodeId)
    {
        return getShadeQueueListInfoPtr(numaNodeId)->getShadeQueues();
    }

    std::string show() const;

private:
    bool mMemAffinityEnable {false};
    std::shared_ptr<ShadeQueueListInfo> mShadeQueueListInfo;
    std::vector<std::shared_ptr<ShadeQueueListInfo>> mNumaNodeShadeQueueListInfoTbl;
};

//------------------------------------------------------------------------------------------

class DeferredEntriesInfo
//
// This class keeps single DeferredEntries data which is allocated from a particular
// NUMA-node when memory affinity is enabled.
// Otherwise, DeferredEntries data is allocated regular malloc way when memory affinity is
// disabled.
//
{
public:
    void insert(const SortedRayState* const entries, const unsigned numEntries)
    {
        std::lock_guard<std::mutex> lock(mDeferredEntryMutex);
        mDeferredEntries.insert(mDeferredEntries.end(), entries, entries + numEntries);
    }

    void retrieveDeferredEntries(mcrt_common::ThreadLocalState* tls,
                                 scene_rdl2::alloc::Arena* arena,
                                 unsigned& numEntries,
                                 SortedRayState*& entries);
    bool isEmpty() const { return mDeferredEntries.empty(); }

    void clear(); // MTsafe

private:
    // Deferred entries. In general we shouldn't be locking or allocating heap
    // memory during the mcrt phase. Since this is an exceptional case and
    // should only happen in extreme cases, we're using a mutex and vector now.
    // We will also get warned when executing this codepath at render time via
    // the logger so if it becomes a common case, we need to revisit and remove
    // these locks and heap allocations.
    std::mutex mDeferredEntryMutex;
    std::vector<SortedRayState> mDeferredEntries;
};

class DeferredEntriesManager
//
// This class keeps a multibank version of DeferredEntries when Memory-Affinity is enabled.
// Otherwise, keep DeferredEntries which are allocated by regular malloc for
// Non-Memory-Affinity situations.
//
{
public:
    DeferredEntriesManager();

    size_t getNumaNodeTblSize() const { return mNumaNodeDeferredEntriesInfoTbl.size(); }

    void insert(const unsigned numaNodeId, const SortedRayState* const entries, const unsigned numEntries)
    {
        getDeferredEntriesInfoPtr(numaNodeId)->insert(entries, numEntries);
    }

    void retrieveDeferredEntries(const unsigned numaNodeId,
                                 mcrt_common::ThreadLocalState* tls,
                                 scene_rdl2::alloc::Arena* arena,
                                 unsigned& numEntries,
                                 SortedRayState*& entries)
    {
        getDeferredEntriesInfoPtr(numaNodeId)->retrieveDeferredEntries(tls, arena, numEntries, entries);
    }

    bool isEmpty(const unsigned numaNodeId)
    {
        return getDeferredEntriesInfoPtr(numaNodeId)->isEmpty();
    }

    void clear();

private:

    DeferredEntriesInfo* getDeferredEntriesInfoPtr(const unsigned numaNodeId) const
    {
        if (numaNodeId == ~0) return mDeferredEntriesInfo.get();
#ifdef STRICT_CHECK_FOR_GETINFOPTR
        if (numaNodeId >= getNumaNodeTblSize()) {
            std::cerr << "RUNTIME_ERROR: DeferredEntriesManager getDeferredEntriesInfoPtr() failed."
                      << " numaNodeId:" << numaNodeId
                      << " >= getNumaNodeTblSize():" << getNumaNodeTblSize() << '\n';
            return nullptr;
        }
        if (!mNumaNodeDeferredEntriesInfoTbl[numaNodeId]) {
            std::cerr << "RUNTIME_ERROR: DeferredEntriesManager getDeferredEntriesInfoPtr() failed."
                      << " numaNodeId:" << numaNodeId
                      << " mNumaNodeDeferredEntriesInfoTbl[numaNodeId] is nullptr\n";
            return nullptr;
        }
#endif // end STRICT_CHECK_FOR_GETINFOPTR
        return mNumaNodeDeferredEntriesInfoTbl[numaNodeId].get();
    }

    std::shared_ptr<DeferredEntriesInfo> mDeferredEntriesInfo;
    std::vector<std::shared_ptr<DeferredEntriesInfo>> mNumaNodeDeferredEntriesInfoTbl;
};

//------------------------------------------------------------------------------------------

//
// Convenience function for iterating over all shade queues.
// Not thread safe.
//
// Example of syntax:
//
//    forEachShadeQueue(tls, [&](ThreadLocalState *tls, ShadeQueue *queue)
//    {
//        queue->doSomething();
//    });
//

class Material : public RootShader
{
public:
    typedef std::vector<int> LobeLabelIds;
    typedef std::vector<int> LpeLobeLabelIds;

    struct ExtraAov
    {
        int mLabelId;
        const scene_rdl2::rdl2::Map *mMap;
    };

    explicit Material(const scene_rdl2::rdl2::SceneObject & owner);
    virtual ~Material();

    //
    // Primitive Attribute Aovs
    // 
    std::vector<char> &getAovFlags() { return mAovFlags; }
    const std::vector<char> &getAovFlags() const { return mAovFlags; }

    //
    // Material Aov: Lobe LabelIds
    //
    LobeLabelIds &getLobeLabelIds() { return mLobeLabelIds; }
    const LobeLabelIds &getLobeLabelIds() const { return mLobeLabelIds; }

    //
    // Light Aov: Lobe LabelIds
    //
    LpeLobeLabelIds &getLpeLobeLabelIds() { return mLpeLobeLabelIds; }
    const LpeLobeLabelIds &getLpeLobeLabelIds() const { return mLpeLobeLabelIds; }

    //
    // Material Aov: Material LabelId
    //
    void setMaterialLabelId(const int32_t labelId) { mMaterialLabelId = labelId; }
    int32_t getMaterialLabelId() const { return mMaterialLabelId; }

    //
    // Light Aov: Material LabelId
    //
    void setLpeMaterialLabelId(const int32_t labelId) { mLpeMaterialLabelId = labelId; }
    int32_t getLpeMaterialLabelId() const { return mLpeMaterialLabelId; }

    //
    // Extra Aovs
    //
    void setExtraAovs(const std::vector<ExtraAov> &ea) { mExtraAovs = ea; }
    void setPostScatterExtraAovs(const std::vector<ExtraAov> &ea) { mPostScatterExtraAovs = ea; }
    const std::vector<ExtraAov> &getExtraAovs() const { return mExtraAovs; }
    const std::vector<ExtraAov> &getPostScatterExtraAovs() const { return mPostScatterExtraAovs; }

    //
    // Shade queue APIs.
    //

    // Const hole here, we're returning a non-const ShadeQueue.
    ShadeQueue *getShadeQueue(const unsigned numaNodeId) const
    {
        return mShadeQueueInfoManager->getShadeQueue(numaNodeId);
    }

    //
    // The following 2 APIs were added to address MOONRAY-2431.
    // The shade queue is a shared resource, so threads can insert both primary
    // and secondary ray hits into any shade queue at anytime. In extreme cases,
    // where there are many bounces like in hair, this can potentially result
    // in a stack overflow.
    //

    // This function stores a set of rays which need to be shaded for later
    // processing.
    void deferEntriesForLaterProcessing(mcrt_common::ThreadLocalState *tls,
                                        const unsigned numEntries,
                                        SortedRayState *entries);

    // This function allocates from the passed in arena. It's up to the caller
    // to preserve the allocations until they are no longer needed.
    void retrieveDeferredEntries(mcrt_common::ThreadLocalState *tls,
                                 scene_rdl2::alloc::Arena *arena,
                                 unsigned &numEntries,
                                 SortedRayState *&entries);

    // This function will allocate shade queues for each material which doesn't already
    // own a shade queue. If a material already owns a shade queue, it is ignored by this call.
    // It also sets the material ids.
    // It should only be called when in bundled mode.
    static void allocShadeQueues(const unsigned shadeQueueSize, ShadeQueue::Handler handler);
    // This function sets the material ids for scalar mode.
    static void initMaterialIds();

    static ShadeQueueList &getAllShadeQueues(const unsigned numaNodeId);
    static unsigned getAllShadeQueuesCount();

    // Check that all queues are now empty.
    static bool areAllShadeQueuesEmptyAllNumaNode();
    static bool areAllShadeQueuesEmpty(const unsigned numaNodeId);

    // Returns the number of entries flushed, or 0 if there were no non-empty queues.
    static unsigned flushNonEmptyShadeQueue(mcrt_common::ThreadLocalState *tls);

    // Print out diagnostics for deferred entries.
    static void printDeferredEntryWarnings();

    // Reset all deferred entry state.
    static void resetDeferredEntryState();

protected:
    friend class ShadeQueueInfo;

    // This must be called when in bundled mode before starting to render.
    // Calling Material::allocShadeQueues will ensure that it gets called.
    void allocShadeQueue(unsigned shadeQueueSize, ShadeQueue::Handler handler);
    void allocDeferredEntriesManager();

    inline void setMaterialId(uint32_t id) { mMaterialId = id; }

    std::shared_ptr<ShadeQueueInfoManager> mShadeQueueInfoManager;

    static std::shared_ptr<ShadeQueueListInfoManager> sShadeQueueListInfoManager;

    // These are used to mask the aovSchema
    std::vector<char>       mAovFlags;
    // These map bsdf lobe label bits to material aov labelIds
    LobeLabelIds            mLobeLabelIds;
    // These map bsdf lobe label bits to light aov labelIds
    LpeLobeLabelIds         mLpeLobeLabelIds;
    // The labelId used in material aovs
    int32_t                 mMaterialLabelId;
    // the labelId used in light aovs
    int32_t                 mLpeMaterialLabelId;
    // Extra Aovs
    std::vector<ExtraAov>   mExtraAovs;
    std::vector<ExtraAov>   mPostScatterExtraAovs;

    std::shared_ptr<DeferredEntriesManager> mDeferredEntriesManager;
    
    static tbb::mutex       sMaterialListMutex;
    static MaterialPtrList  sAllMaterials;
    static MaterialPtrList  sQueuelessMaterials;

    // Shared between all Materials.
    static std::atomic<uint32_t> sDeferredEntryCalls;
    static std::atomic<uint32_t> sTotalDeferredEntries;
};

template <typename Body>
inline void
forEachShadeQueue(mcrt_common::ThreadLocalState *tls, const Body &body)
{
    auto crawlNumaNodeShadeQueues = [&](const unsigned numaNodeId) {
        ShadeQueueList &shadeQueues = Material::getAllShadeQueues(numaNodeId);
        for (auto it = shadeQueues.begin(); it != shadeQueues.end(); ++it) {
            body(tls, *it);
        }
    };

    if (!tls) {
        // This is a special case and should crawl all the NUMA-nod's shade queues sequentially
        std::shared_ptr<mcrt_common::MemoryAffinityManager> memAffMgr = mcrt_common::AffinityManager::get()->getMem();
#ifndef PLATFORM_APPLE
        if (!memAffMgr->getMemAffinityEnable()) {
#endif // end of Non PLATFORM_APPLE
            //
            // Memory Affinity disabled
            //
            crawlNumaNodeShadeQueues(~0);
#ifndef PLATFORM_APPLE
        } else {
            //
            // Memory Affinity enabled
            //
            const unsigned memNodeTblSize = memAffMgr->getMemNodeTblSize();
            for (unsigned memNodeId = 0; memNodeId < memNodeTblSize; ++memNodeId) {
                if (memAffMgr->isActiveMemNode(memNodeId)) {
                    unsigned numaNodeId = (memAffMgr->getMemNodeTbl())[memNodeId]->getNumaNodeId();
                    crawlNumaNodeShadeQueues(numaNodeId);
                }
            }
        }
#endif // end of Non PLATFORM_APPLE

    } else {
        crawlNumaNodeShadeQueues(tls->mArena.getNumaNodeId());
    }
}

} // namespace shading
} // namespace moonray


